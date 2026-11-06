#!/usr/bin/env python3
import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


BASE_DIR = Path(__file__).resolve().parents[2]

G2S_REPO = Path(os.environ.get("G2S_REPO", str(BASE_DIR / "external/baselines/graph2smiles"))).expanduser().resolve()
MODEL_DIR = (G2S_REPO / "checkpoints/uspto_50k").resolve()
MODEL_PATH = MODEL_DIR / "model.500000_49.pt"
PROCESSED_DATA_PATH = (G2S_REPO / "data/uspto_50k/processed").resolve()

MODEL_PATH = Path(os.environ.get("G2S_CHECKPOINT", str(MODEL_PATH))).expanduser().resolve()
PROCESSED_DATA_PATH = Path(os.environ.get("G2S_DATA_DIR", str(PROCESSED_DATA_PATH))).expanduser().resolve()

OUTPUT_ROOT = BASE_DIR / "outputs"

MODEL_NAME = "graph2smiles"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEAM_SIZE = 30
N_BEST = 30
TEMPERATURE = 1.0
PREDICT_MIN_LEN = 1
PREDICT_MAX_LEN = 512


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


class Graph2SmilesPredictor:
    def __init__(self):
        if not G2S_REPO.exists():
            raise FileNotFoundError(f"G2S_REPO not found: {G2S_REPO}")

        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"MODEL_PATH not found: {MODEL_PATH}")

        if not PROCESSED_DATA_PATH.exists():
            raise FileNotFoundError(f"PROCESSED_DATA_PATH not found: {PROCESSED_DATA_PATH}")

        vocab_file = PROCESSED_DATA_PATH / "vocab.txt"
        if not vocab_file.exists():
            raise FileNotFoundError(f"vocab.txt not found: {vocab_file}")

        sys.path.insert(0, str(G2S_REPO))

        os.chdir(str(G2S_REPO))

        from models.graph2smiles import Graph2SMILES
        from train import get_model
        from utils import parsing
        from utils.data_utils import G2SDataset
        from utils.preprocess_utils import (
            get_graph_features_from_smi,
            get_token_ids,
            load_vocab,
            smi_tokenizer,
        )

        self.Graph2SMILES = Graph2SMILES
        self.get_model = get_model
        self.parsing = parsing
        self.G2SDataset = G2SDataset
        self.get_graph_features_from_smi = get_graph_features_from_smi
        self.get_token_ids = get_token_ids
        self.load_vocab = load_vocab
        self.smi_tokenizer = smi_tokenizer

        print(f"[INFO] Loading checkpoint from {MODEL_PATH}")
        print(f"[INFO] Device: {DEVICE}")

        state = torch.load(MODEL_PATH, map_location=DEVICE)

        self.pretrain_args = state["args"]
        self.pretrain_args.local_rank = -1
        self.pretrain_args.load_from = None

        if not hasattr(self.pretrain_args, "n_latent"):
            self.pretrain_args.n_latent = 1

        if not hasattr(self.pretrain_args, "shared_attention_layer"):
            self.pretrain_args.shared_attention_layer = 0

        self.args = self._build_predict_args()
        self.vocab = self.load_vocab(self.args)
        self.vocab_tokens = [
            k for k, v in sorted(self.vocab.items(), key=lambda tup: tup[1])
        ]

        self.model, _ = self.get_model(
            self.pretrain_args,
            self.Graph2SMILES,
            self.vocab,
            DEVICE,
        )

        if hasattr(self.model, "module"):
            self.model = self.model.module

        state_dict = {
            k.replace("module.", ""): v
            for k, v in state["state_dict"].items()
        }

        self.model.load_state_dict(state_dict)
        self.model.eval()

        print("[INFO] Model loaded successfully")

    def _build_predict_args(self):
        parser = argparse.ArgumentParser("graph2smiles_internal")
        self.parsing.add_common_args(parser)
        self.parsing.add_preprocess_args(parser)
        self.parsing.add_train_args(parser)
        self.parsing.add_predict_args(parser)

        args, _ = parser.parse_known_args([])

        args.processed_data_path = str(PROCESSED_DATA_PATH)
        args.model_path = str(MODEL_DIR)
        args.load_from = str(MODEL_PATH)
        args.local_rank = -1
        args.task = "retrosynthesis"
        args.model = "graph2smiles"

        args.compute_graph_distance = True
        args.batch_type = "tokens"
        args.predict_batch_size = 4096
        args.enable_amp = False
        args.mask_rel_chirality = getattr(self.pretrain_args, "mask_rel_chirality", 0)

        args.beam_size = BEAM_SIZE
        args.n_best = N_BEST
        args.temperature = TEMPERATURE
        args.predict_min_len = PREDICT_MIN_LEN
        args.predict_max_len = PREDICT_MAX_LEN

        args.max_src_len = getattr(self.pretrain_args, "max_src_len", 512)
        args.max_tgt_len = getattr(self.pretrain_args, "max_tgt_len", 512)
        args.rel_pos_buckets = getattr(self.pretrain_args, "rel_pos_buckets", 10)
        args.num_cores = 1
        args.verbose = False

        return args

    def _make_single_npz(self, target_smiles, tmp_dir):
        target = canonicalize_smiles_local(target_smiles)

        if target is None:
            raise ValueError(f"Invalid target SMILES: {target_smiles}")

        src_line = self.smi_tokenizer(target) + "\n"
        dummy_tgt = self.smi_tokenizer("CC") + "\n"

        src_tokens = src_line.strip().split()
        tgt_tokens = dummy_tgt.strip().split()

        src_token_ids, src_lens = self.get_token_ids(
            src_tokens,
            self.vocab,
            max_len=self.args.max_src_len,
        )

        tgt_token_ids, tgt_lens = self.get_token_ids(
            tgt_tokens,
            self.vocab,
            max_len=self.args.max_tgt_len,
        )

        src_token_ids = np.array([src_token_ids], dtype=np.int32)
        src_lens = np.array([src_lens], dtype=np.int32)
        tgt_token_ids = np.array([tgt_token_ids], dtype=np.int32)
        tgt_lens = np.array([tgt_lens], dtype=np.int32)

        graph_features = self.get_graph_features_from_smi((0, src_line))

        (
            a_scopes,
            a_scopes_lens,
            b_scopes,
            b_scopes_lens,
            a_features,
            a_features_lens,
            b_features,
            b_features_lens,
            a_graphs,
            b_graphs,
        ) = graph_features

        output_file = Path(tmp_dir) / "single.npz"

        np.savez(
            output_file,
            src_token_ids=src_token_ids,
            src_lens=src_lens,
            tgt_token_ids=tgt_token_ids,
            tgt_lens=tgt_lens,
            a_scopes=a_scopes,
            b_scopes=b_scopes,
            a_features=a_features,
            b_features=b_features,
            a_graphs=a_graphs,
            b_graphs=b_graphs,
            a_scopes_lens=np.array([a_scopes_lens], dtype=np.int32),
            b_scopes_lens=np.array([b_scopes_lens], dtype=np.int32),
            a_features_lens=np.array([a_features_lens], dtype=np.int32),
            b_features_lens=np.array([b_features_lens], dtype=np.int32),
        )

        return output_file

    def predict(self, target_smiles, top_k=10):
        with tempfile.TemporaryDirectory() as tmp_dir:
            single_npz = self._make_single_npz(target_smiles, tmp_dir)

            dataset = self.G2SDataset(self.args, file=str(single_npz))
            dataset.batch(
                batch_type=self.args.batch_type,
                batch_size=self.args.predict_batch_size,
            )

            batch = dataset[0]
            batch.to(DEVICE)

            with torch.no_grad():
                results = self.model.predict_step(
                    reaction_batch=batch,
                    batch_size=batch.size,
                    beam_size=self.args.beam_size,
                    n_best=max(self.args.n_best, top_k),
                    temperature=self.args.temperature,
                    min_length=self.args.predict_min_len,
                    max_length=self.args.predict_max_len,
                )

        result_by_candidate = {}

        predictions = results["predictions"][0]
        scores = results["scores"][0]

        for prediction, score in zip(predictions, scores):
            predicted_idx = prediction.detach().cpu().numpy()
            raw_score = float(score.detach().cpu().numpy())

            predicted_tokens = [
                self.vocab_tokens[idx]
                for idx in predicted_idx[:-1]
            ]

            smi = "".join(predicted_tokens)
            candidate = canonicalize_smiles_local(smi)

            if candidate is None:
                continue

            prob_like_score = float(np.exp(raw_score))

            if candidate not in result_by_candidate:
                result_by_candidate[candidate] = 0.0

            result_by_candidate[candidate] += prob_like_score

        ranked = sorted(
            result_by_candidate.items(),
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
        description="Predict using local ASKCOS Graph2SMILES."
    )

    parser.add_argument("--target", required=False)
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--work_name", default="default")

    parser.add_argument("--batch_mode", action="store_true")
    parser.add_argument("--input_csv", required=False)
    parser.add_argument("--output_csv", required=False)

    args = parser.parse_args()

    predictor = Graph2SmilesPredictor()

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
            print(f"[G2S] sample_index={sample_index} id={sample_id}")
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
                print(f"[G2S_ERROR] sample_index={sample_index} id={sample_id}: {e}")
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
            / "graph2smiles.csv"
        )

    write_csv(rows, output_path)

    print(f"[INFO] Wrote {len(rows)} candidates to {output_path}")


if __name__ == "__main__":
    main()
