#!/usr/bin/env python3
import argparse
import os
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


BASE_DIR = Path(__file__).resolve().parents[2]

TR_REPO = Path(os.environ.get("TR_REPO", str(BASE_DIR / "external/askcos_retro_mamba/template_relevance"))).expanduser().resolve()
MODEL_DIR = (
    TR_REPO
    / "checkpoints/grid_search_uspto_50k/ep30_bs512_hid1024x1024_drop0.3_lr0.001"
).resolve()

MODEL_PATH = MODEL_DIR / "model_latest.pt"
TEMPLATE_FILE = (TR_REPO / "data/uspto_50k/processed/templates.jsonl").resolve()

MODEL_PATH = Path(os.environ.get("TR_CHECKPOINT", str(MODEL_PATH))).expanduser().resolve()
TEMPLATE_FILE = Path(os.environ.get("TR_TEMPLATES", str(TEMPLATE_FILE))).expanduser().resolve()

OUTPUT_ROOT = BASE_DIR / "outputs"

MODEL_NAME = "template_relevance"

FP_RADIUS = 2
FP_SIZE = 32681
MAX_CUM_PROB = 0.999
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_csv(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "sample_index",
        "id",
        "target",
        "model",
        "rank",
        "score",
        "candidate",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_targets_from_csv(input_csv: Path):
    df = pd.read_csv(input_csv)

    required = {"sample_index", "id", "target"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"input_csv missing required columns: {missing}")

    if "status" in df.columns:
        df = df[df["status"].astype(str) == "OK"].copy()

    return df


def canonicalize_smiles_local(smiles):
    smiles = "".join(str(smiles).split())
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")

    cano = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    mol = Chem.MolFromSmiles(cano)
    if mol is None:
        return None

    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


class TemplateRelevancePredictor:
    def __init__(self):
        if not TR_REPO.exists():
            raise FileNotFoundError(f"TR_REPO not found: {TR_REPO}")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

        if not TEMPLATE_FILE.exists():
            raise FileNotFoundError(f"TEMPLATE_FILE not found: {TEMPLATE_FILE}")

        sys.path.insert(0, str(TR_REPO))

        import templ_rel_parser
        from utils import get_model, load_templates_as_list, mol_smi_to_count_fp
        from rdchiral.initialization import rdchiralReactants, rdchiralReaction
        from rdchiral.main import rdchiralRun

        self.templ_rel_parser = templ_rel_parser
        self.get_model = get_model
        self.load_templates_as_list = load_templates_as_list
        self.mol_smi_to_count_fp = mol_smi_to_count_fp
        self.rdchiralReactants = rdchiralReactants
        self.rdchiralReaction = rdchiralReaction
        self.rdchiralRun = rdchiralRun

        self.args = self._build_args()
        self.templates, self.template_attributes = self.load_templates_as_list(
            template_file=str(TEMPLATE_FILE)
        )

        print(f"[INFO] Loaded {len(self.templates)} templates from {TEMPLATE_FILE}")
        print(f"[INFO] Loading model from {MODEL_PATH}")
        print(f"[INFO] Device: {DEVICE}")

        self.model, _ = self.get_model(self.args, device=DEVICE)
        self.model.eval()

    def _build_args(self):
        parser = argparse.ArgumentParser("template_relevance_internal")
        self.templ_rel_parser.add_model_opts(parser)
        self.templ_rel_parser.add_preprocess_opts(parser)
        self.templ_rel_parser.add_train_opts(parser)
        self.templ_rel_parser.add_predict_opts(parser)

        args, _ = parser.parse_known_args([])

        args.load_from = str(MODEL_PATH)
        args.local_rank = -1

        if not hasattr(args, "radius"):
            args.radius = FP_RADIUS

        if not hasattr(args, "fp_size"):
            args.fp_size = FP_SIZE

        return args

    def predict(self, target_smiles, top_k=10):
        target = canonicalize_smiles_local(target_smiles)

        if target is None:
            raise ValueError(f"Invalid target SMILES: {target_smiles}")

        prod_fp = self.mol_smi_to_count_fp(
            target,
            self.args.radius,
            self.args.fp_size,
        )

        final_fp = torch.as_tensor(prod_fp.toarray()).float().to(DEVICE)

        with torch.no_grad():
            logits = self.model(final_fp)
            scores = F.softmax(logits, dim=1)
            scores = scores.squeeze(dim=0).cpu().numpy()

        indices = np.argsort(-scores)
        sorted_scores = scores[indices]

        if MAX_CUM_PROB:
            exceeds = np.nonzero(np.cumsum(sorted_scores) >= MAX_CUM_PROB)[0]
            if exceeds.size:
                max_index = exceeds[0] + 1
                indices = indices[:max_index]
                sorted_scores = sorted_scores[:max_index]

        result_by_candidate = {}
        product_reactants = self.rdchiralReactants(target)

        for template_rank, (template_idx, template_score) in enumerate(
            zip(indices, sorted_scores),
            start=1,
        ):
            template = self.templates[int(template_idx)]
            reaction_smarts = template["reaction_smarts"]

            reaction_smarts_one = "(" + reaction_smarts.replace(">>", ")>>(") + ")"

            try:
                rxn = self.rdchiralReaction(str(reaction_smarts_one))
                reactants = self.rdchiralRun(
                    rxn,
                    product_reactants,
                    return_mapped=False,
                )
            except Exception:
                continue

            if not reactants:
                continue

            for reactant in reactants:
                smiles_list = reactant.split(".")

                if template.get("intra_only") and len(smiles_list) > 1:
                    continue

                if template.get("dimer_only") and (
                    len(set(smiles_list)) != 1 or len(smiles_list) != 2
                ):
                    continue

                if target in smiles_list:
                    continue

                candidate = ".".join(sorted(smiles_list))
                candidate = canonicalize_smiles_local(candidate)

                if candidate is None:
                    continue

                if candidate not in result_by_candidate:
                    result_by_candidate[candidate] = {
                        "score": float(template_score),
                        "template_rank": template_rank,
                    }

                if len(result_by_candidate) >= top_k:
                    break

            if len(result_by_candidate) >= top_k:
                break

        ranked = sorted(
            result_by_candidate.items(),
            key=lambda x: x[1]["score"],
            reverse=True,
        )[:top_k]

        rows = []

        for rank, (candidate, info) in enumerate(ranked, start=1):
            rows.append(
                {
                    "model": MODEL_NAME,
                    "rank": rank,
                    "score": info["score"],
                    "candidate": candidate,
                }
            )

        return rows


def main():
    parser = argparse.ArgumentParser(
        description="Predict using local ASKCOS Template Relevance."
    )

    parser.add_argument("--target", required=False)
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--work_name", default="default")

    parser.add_argument("--batch_mode", action="store_true")
    parser.add_argument("--input_csv", required=False)
    parser.add_argument("--output_csv", required=False)

    args = parser.parse_args()

    predictor = TemplateRelevancePredictor()

    if args.batch_mode:
        if not args.input_csv or not args.output_csv:
            raise ValueError("--batch_mode requires --input_csv and --output_csv")

        targets_df = load_targets_from_csv(Path(args.input_csv))
        rows = []

        for _, item in targets_df.iterrows():
            sample_index = item["sample_index"]
            sample_id = item["id"]
            target = item["target"]

            print("=" * 100)
            print(f"[TEMPLATE_RELEVANCE] sample_index={sample_index} id={sample_id}")
            print(f"[TARGET] {target}")

            try:
                pred_rows = predictor.predict(target, top_k=args.top_k)

                for row in pred_rows:
                    rows.append(
                        {
                            "sample_index": sample_index,
                            "id": sample_id,
                            "target": target,
                            "model": MODEL_NAME,
                            "rank": row["rank"],
                            "score": row["score"],
                            "candidate": row["candidate"],
                        }
                    )

            except Exception as e:
                print(
                    f"[TEMPLATE_RELEVANCE_ERROR] "
                    f"sample_index={sample_index} id={sample_id}: {e}"
                )
                continue

        output_path = Path(args.output_csv)

    else:
        if not args.target:
            raise ValueError("Single-target mode requires --target")

        pred_rows = predictor.predict(args.target, top_k=args.top_k)
        target = canonicalize_smiles_local(args.target)

        rows = [
            {
                "sample_index": 0,
                "id": "single",
                "target": target,
                "model": MODEL_NAME,
                "rank": row["rank"],
                "score": row["score"],
                "candidate": row["candidate"],
            }
            for row in pred_rows
        ]

        output_path = (
            OUTPUT_ROOT
            / args.work_name
            / "model_predictions"
            / "template_relevance.csv"
        )

    write_csv(rows, output_path)

    print(f"[INFO] Wrote {len(rows)} candidates to {output_path}")


if __name__ == "__main__":
    main()
