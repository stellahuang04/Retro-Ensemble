#!/usr/bin/env python3
import argparse
import os
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem


BASE_DIR = Path(__file__).resolve().parents[2]

AT_REPO = Path(os.environ.get("AT_REPO", str(BASE_DIR / "external/baselines/augmented_transformer"))).expanduser().resolve()
MODEL_PATH = (
    AT_REPO / "checkpoints/uspto_50k/model_step_500000.pt"
).resolve()

MODEL_PATH = Path(os.environ.get("AT_CHECKPOINT", str(MODEL_PATH))).expanduser().resolve()

OUTPUT_ROOT = BASE_DIR / "outputs"

MODEL_NAME = "augmented_transformer"

N_BEST = 30
BEAM_SIZE = 30
GPU = -1


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


def validate_paths():
    if not AT_REPO.exists():
        raise FileNotFoundError(f"AT_REPO not found: {AT_REPO}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")


def load_targets_from_csv(input_csv: Path):
    df = pd.read_csv(input_csv)

    required = {"sample_index", "id", "target"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"input_csv missing required columns: {missing}")

    if "status" in df.columns:
        df = df[df["status"].astype(str) == "OK"].copy()

    return df


def canonicalize_precursor_set(canonicalize_smiles_func, smiles: str):
    parts = smiles.split(".")
    canonical_parts = []

    for part in parts:
        canonical = canonicalize_smiles_func(part)
        if not canonical:
            return None
        canonical_parts.append(canonical)

    return ".".join(sorted(canonical_parts))


def normalize_model_outputs(products, scores, n_targets: int, n_best: int):
    """
    ONMT ServerModel.run can return either:
    1. flat list: [target1_best1, target1_best2, ..., target2_best1, ...]
    2. nested list: [[target1_best1, ...], [target2_best1, ...]]
    This function normalizes both into list[(products_for_target, scores_for_target)].
    """
    if n_targets == 1:
        if products and isinstance(products[0], list):
            return [(products[0], scores[0])]
        return [(products, scores)]

    if products and isinstance(products[0], list):
        return list(zip(products, scores))

    if len(products) % n_targets != 0:
        raise ValueError(
            f"Cannot split flat AT outputs: len(products)={len(products)}, n_targets={n_targets}"
        )

    stride = len(products) // n_targets
    grouped = []

    for i in range(n_targets):
        start = i * stride
        end = (i + 1) * stride
        grouped.append((products[start:end], scores[start:end]))

    return grouped


class AugmentedTransformerBatchPredictor:
    def __init__(self):
        validate_paths()

        sys.path.insert(0, str(AT_REPO))

        from utils import canonicalize_smiles, smi_tokenizer
        from onmt.translate.translation_server import ServerModel as ONMTServerModel

        self.canonicalize_smiles = canonicalize_smiles
        self.smi_tokenizer = smi_tokenizer

        use_cpu = GPU < 0 or not torch.cuda.is_available()

        print(f"[INFO] AT_REPO: {AT_REPO}")
        print(f"[INFO] MODEL_PATH: {MODEL_PATH}")
        print(f"[INFO] Device: {'CPU' if use_cpu else f'cuda:{GPU}'}")

        self.model = ONMTServerModel(
            opt={
                "models": str(MODEL_PATH),
                "n_best": N_BEST,
                "beam_size": BEAM_SIZE,
                "gpu": -1 if use_cpu else GPU,
            },
            model_id=0,
            load=True,
        )

    def predict_one(self, target_smiles: str, top_k: int):
        target = self.canonicalize_smiles(target_smiles)

        if not target:
            raise ValueError(f"Invalid target SMILES: {target_smiles}")

        target_mol = Chem.MolFromSmiles(target)

        if target_mol is None:
            raise ValueError(f"RDKit cannot parse target SMILES: {target_smiles}")

        tokenized_target = self.smi_tokenizer(target)

        products, scores, _, _, _ = self.model.run(
            inputs=[{"src": tokenized_target}]
        )

        return self._rank_predictions(
            target=target,
            target_mol=target_mol,
            products=products,
            scores=scores,
            top_k=top_k,
        )

    def predict_batch(self, targets_df: pd.DataFrame, top_k: int):
        valid_meta = []
        inputs = []
        target_mols = []

        for _, row in targets_df.iterrows():
            target_raw = str(row["target"])
            target = self.canonicalize_smiles(target_raw)

            if not target:
                print(f"[AT_SKIP] Invalid target: {target_raw}")
                continue

            target_mol = Chem.MolFromSmiles(target)

            if target_mol is None:
                print(f"[AT_SKIP] RDKit cannot parse target: {target_raw}")
                continue

            inputs.append({"src": self.smi_tokenizer(target)})
            valid_meta.append(
                {
                    "sample_index": row["sample_index"],
                    "id": row["id"],
                    "target": target,
                }
            )
            target_mols.append(target_mol)

        if not inputs:
            return []

        print(f"[AT_BATCH] Running {len(inputs)} targets...")

        products, scores, _, _, _ = self.model.run(inputs=inputs)
        grouped_outputs = normalize_model_outputs(
            products=products,
            scores=scores,
            n_targets=len(inputs),
            n_best=N_BEST,
        )

        all_rows = []

        for meta, target_mol, (target_products, target_scores) in zip(
            valid_meta,
            target_mols,
            grouped_outputs,
        ):
            ranked_rows = self._rank_predictions(
                target=meta["target"],
                target_mol=target_mol,
                products=target_products,
                scores=target_scores,
                top_k=top_k,
            )

            for row in ranked_rows:
                all_rows.append(
                    {
                        "sample_index": meta["sample_index"],
                        "id": meta["id"],
                        "target": meta["target"],
                        "model": MODEL_NAME,
                        "rank": row["rank"],
                        "score": row["score"],
                        "candidate": row["candidate"],
                    }
                )

        return all_rows

    def _rank_predictions(self, target, target_mol, products, scores, top_k):
        valid = {}

        for product, score in zip(products, scores):
            candidate = "".join(str(product).split())
            candidate = canonicalize_precursor_set(self.canonicalize_smiles, candidate)

            if not candidate:
                continue

            candidate_mol = Chem.MolFromSmiles(candidate)

            if candidate_mol is None:
                continue

            if target_mol.GetNumAtoms() - candidate_mol.GetNumAtoms() > 4:
                continue

            model_score = float(np.exp(float(score)))

            if candidate not in valid:
                valid[candidate] = model_score
            else:
                valid[candidate] += model_score

        ranked = sorted(valid.items(), key=lambda x: x[1], reverse=True)[:top_k]

        rows = []

        for rank, (candidate, score) in enumerate(ranked, start=1):
            rows.append(
                {
                    "rank": rank,
                    "score": score,
                    "candidate": candidate,
                }
            )

        return rows


def main():
    parser = argparse.ArgumentParser(
        description="Predict using local ASKCOS Augmented Transformer."
    )

    parser.add_argument("--target", required=False)
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--work_name", default="default")

    parser.add_argument("--batch_mode", action="store_true")
    parser.add_argument("--input_csv", required=False)
    parser.add_argument("--output_csv", required=False)

    args = parser.parse_args()

    predictor = AugmentedTransformerBatchPredictor()

    if args.batch_mode:
        if not args.input_csv or not args.output_csv:
            raise ValueError("--batch_mode requires --input_csv and --output_csv")

        targets_df = load_targets_from_csv(Path(args.input_csv))
        rows = predictor.predict_batch(targets_df, top_k=args.top_k)
        output_path = Path(args.output_csv)

    else:
        if not args.target:
            raise ValueError("Single-target mode requires --target")

        target = predictor.canonicalize_smiles(args.target)
        rows_single = predictor.predict_one(args.target, top_k=args.top_k)

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
            for row in rows_single
        ]

        output_path = (
            OUTPUT_ROOT
            / args.work_name
            / "model_predictions"
            / "augmented_transformer.csv"
        )

    write_csv(rows, output_path)

    print(f"[INFO] Wrote {len(rows)} candidates to {output_path}")


if __name__ == "__main__":
    main()
