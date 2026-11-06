#!/usr/bin/env python3
import argparse
import os
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")


BASE_DIR = Path(__file__).resolve().parents[2]

RETROSIM_REPO = Path(os.environ.get("RETROSIM_REPO", str(BASE_DIR / "external/askcos_retro_mamba/retrosim"))).expanduser().resolve()

DATA_FILE = (
    RETROSIM_REPO
    / "retrosim/data/data_processed.csv"
).resolve()

DATA_FILE = Path(os.environ.get("RETROSIM_DATA", str(DATA_FILE))).expanduser().resolve()

OUTPUT_ROOT = BASE_DIR / "outputs"

MODEL_NAME = "retrosim"

FP_RADIUS = 2
USE_FEATURES = True
N_NEIGHBORS = 100


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


def canonicalize_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def remove_atom_map(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")

    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def get_fp(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None

    return AllChem.GetMorganFingerprint(
        mol,
        FP_RADIUS,
        useFeatures=USE_FEATURES,
    )


class RetroSimPredictor:
    def __init__(self):
        if not RETROSIM_REPO.exists():
            raise FileNotFoundError(f"RETROSIM_REPO not found: {RETROSIM_REPO}")

        if not DATA_FILE.exists():
            raise FileNotFoundError(f"DATA_FILE not found: {DATA_FILE}")

        sys.path.insert(0, str(RETROSIM_REPO))

        from retrosim.data.get_data import get_data_df, split_data_df
        from retrosim.utils.generate_retro_templates import process_an_example
        from rdchiral.main import rdchiralRun, rdchiralReaction, rdchiralReactants

        self.get_data_df = get_data_df
        self.split_data_df = split_data_df
        self.process_an_example = process_an_example
        self.rdchiralRun = rdchiralRun
        self.rdchiralReaction = rdchiralReaction
        self.rdchiralReactants = rdchiralReactants

        self.train_df, self.train_fps = self._load_training_corpus()

    def _load_training_corpus(self):
        print(f"[INFO] Loading RetroSim data from: {DATA_FILE}")

        data = self.get_data_df(str(DATA_FILE))
        self.split_data_df(data)

        if "dataset" not in data.columns:
            raise ValueError("split_data_df did not create a 'dataset' column.")

        train_df = data.loc[data["dataset"] == "train"].copy()

        if len(train_df) == 0:
            raise ValueError("No training data found after split_data_df().")

        required_cols = {"rxn_smiles", "prod_smiles"}
        missing = required_cols - set(train_df.columns)

        if missing:
            raise ValueError(
                f"Retrosim data must contain {required_cols}, missing: {missing}"
            )

        valid_rows = []
        fps = []

        for _, row in train_df.iterrows():
            prod = canonicalize_smiles(row["prod_smiles"])
            if prod is None:
                continue

            fp = get_fp(prod)
            if fp is None:
                continue

            row = row.copy()
            row["prod_smiles"] = prod
            valid_rows.append(row)
            fps.append(fp)

        if len(valid_rows) == 0:
            raise ValueError("No valid training fingerprints found.")

        train_df = pd.DataFrame(valid_rows).reset_index(drop=True)

        print(f"[INFO] Loaded {len(train_df)} training reactions.")

        return train_df, fps

    def predict(self, target_smiles, top_k=10):
        target = canonicalize_smiles(target_smiles)

        if target is None:
            raise ValueError(f"Invalid target SMILES: {target_smiles}")

        target_fp = get_fp(target)
        if target_fp is None:
            raise ValueError(f"Cannot compute fingerprint for target: {target}")

        target_reactants = self.rdchiralReactants(target)

        sims = DataStructs.BulkTanimotoSimilarity(target_fp, self.train_fps)
        neighbor_indices = np.argsort(sims)[::-1][:N_NEIGHBORS]

        candidate_scores = {}

        for idx in neighbor_indices:
            train_rxn = self.train_df.loc[idx, "rxn_smiles"]
            product_sim = float(sims[idx])

            try:
                retro_template = self.process_an_example(
                    train_rxn,
                    super_general=True,
                )
            except Exception:
                continue

            if not retro_template:
                continue

            template = "(" + retro_template.replace(">>", ")>>")

            train_reactants = train_rxn.split(">")[0]
            train_reactants = remove_atom_map(train_reactants)

            if train_reactants is None:
                continue

            train_reactants_fp = get_fp(train_reactants)

            if train_reactants_fp is None:
                continue

            try:
                rxn = self.rdchiralReaction(template)
                outcomes = self.rdchiralRun(
                    rxn,
                    target_reactants,
                    combine_enantiomers=False,
                )
            except Exception:
                continue

            for candidate in outcomes:
                candidate = canonicalize_smiles(candidate)

                if candidate is None:
                    continue

                candidate_fp = get_fp(candidate)

                if candidate_fp is None:
                    continue

                precursor_sim = DataStructs.TanimotoSimilarity(
                    candidate_fp,
                    train_reactants_fp,
                )

                score = float(product_sim * precursor_sim)

                if candidate not in candidate_scores:
                    candidate_scores[candidate] = score
                else:
                    candidate_scores[candidate] = max(
                        candidate_scores[candidate],
                        score,
                    )

        ranked = sorted(
            candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        rows = []

        for rank, (candidate, score) in enumerate(ranked, start=1):
            rows.append(
                {
                    "model": MODEL_NAME,
                    "rank": rank,
                    "score": score,
                    "candidate": candidate,
                }
            )

        return rows


def main():
    parser = argparse.ArgumentParser(
        description="Predict using official RetroSim split/data logic."
    )

    parser.add_argument("--target", required=False)
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--work_name", default="default")

    parser.add_argument("--batch_mode", action="store_true")
    parser.add_argument("--input_csv", required=False)
    parser.add_argument("--output_csv", required=False)

    args = parser.parse_args()

    predictor = RetroSimPredictor()

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
            print(f"[RETROSIM] sample_index={sample_index} id={sample_id}")
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
                print(f"[RETROSIM_ERROR] sample_index={sample_index} id={sample_id}: {e}")
                continue

        output_path = Path(args.output_csv)

    else:
        if not args.target:
            raise ValueError("Single-target mode requires --target")

        pred_rows = predictor.predict(args.target, top_k=args.top_k)
        target = canonicalize_smiles(args.target)

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
            / "retrosim.csv"
        )

    write_csv(rows, output_path)

    print(f"[INFO] Wrote {len(rows)} candidates to {output_path}")


if __name__ == "__main__":
    main()
