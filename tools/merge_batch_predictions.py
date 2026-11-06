#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "sample_index",
    "id",
    "target",
    "model",
    "rank",
    "score",
    "raw_candidate",
    "canonical_candidate",
]


def load_prediction_csvs(input_dir: Path) -> pd.DataFrame:
    """
    Load all canonicalized prediction CSV files from input_dir.

    Expected file format:
        sample_index,id,target,model,rank,score,raw_candidate,canonical_candidate
    """
    input_dir = Path(input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    csv_paths = sorted(input_dir.glob("*.csv"))

    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    dfs = []

    for csv_path in csv_paths:
        print(f"[LOAD] {csv_path}")
        df = pd.read_csv(csv_path)

        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {missing}")

        df = df[REQUIRED_COLUMNS].copy()

        # Keep model name from file if column is empty for any reason.
        if df["model"].isna().any() or (df["model"].astype(str).str.strip() == "").any():
            df["model"] = csv_path.stem

        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True)
    return merged


def clean_merged_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean merged candidates before assigning candidate_id.

    Important behavior:
    - Drop empty canonical_candidate rows.
    - Convert sample_index and rank to int.
    - Sort deterministically.
    - Deduplicate only within the same sample_index + model + canonical_candidate.
    - Do NOT deduplicate across different models, because cross-model repetition is useful.
    """
    df = df.copy()

    df["sample_index"] = df["sample_index"].astype(int)
    df["id"] = df["id"].astype(str)
    df["target"] = df["target"].astype(str)
    df["model"] = df["model"].astype(str)
    df["rank"] = df["rank"].astype(int)

    df["canonical_candidate"] = df["canonical_candidate"].fillna("").astype(str).str.strip()
    df["raw_candidate"] = df["raw_candidate"].fillna("").astype(str)

    before = len(df)
    df = df[df["canonical_candidate"] != ""].copy()
    after_drop_empty = len(df)

    print(f"[CLEAN] Rows before drop empty canonical_candidate: {before}")
    print(f"[CLEAN] Rows after drop empty canonical_candidate: {after_drop_empty}")

    # Deterministic order:
    # sample_index -> model -> rank -> canonical_candidate
    df = df.sort_values(
        ["sample_index", "model", "rank", "canonical_candidate"],
        kind="mergesort",
    ).reset_index(drop=True)

    before_dedup = len(df)

    # Remove duplicate same canonical candidate from the same model for the same target.
    # Keep cross-model duplicates because they represent model agreement.
    df = df.drop_duplicates(
        subset=["sample_index", "model", "canonical_candidate"],
        keep="first",
    ).reset_index(drop=True)

    after_dedup = len(df)

    print(f"[CLEAN] Rows before within-model dedup: {before_dedup}")
    print(f"[CLEAN] Rows after within-model dedup: {after_dedup}")

    return df


def add_candidate_ids(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Add candidate_id to merged_candidates.csv.

    Rule:
    - candidate_id is shared by identical canonical_candidate within the same sample_index.
    - Rows are NOT aggregated.
    - Therefore, the same precursor set predicted by different models keeps multiple rows,
      but those rows share the same candidate_id.

    Example:
        sample_index=0, canonical_candidate=A.B from retrosim       -> C0001
        sample_index=0, canonical_candidate=A.B from graph2smiles   -> C0001
        sample_index=0, canonical_candidate=C.D from retrosim       -> C0002

    candidate_id is local to each sample_index:
        sample_index=0 has C0001, C0002, ...
        sample_index=1 also has C0001, C0002, ...
    """
    merged = merged.copy()

    if "sample_index" not in merged.columns:
        raise ValueError("Cannot add candidate_id because sample_index column is missing.")

    if "canonical_candidate" not in merged.columns:
        raise ValueError("Cannot add candidate_id because canonical_candidate column is missing.")

    merged["canonical_candidate"] = merged["canonical_candidate"].astype(str).str.strip()

    candidate_id_by_index = {}

    for sample_index, group in merged.groupby("sample_index", sort=False):
        precursor_to_id = {}
        next_id = 1

        for idx, row in group.iterrows():
            canonical_candidate = str(row["canonical_candidate"]).strip()

            if canonical_candidate not in precursor_to_id:
                precursor_to_id[canonical_candidate] = f"C{next_id:04d}"
                next_id += 1

            candidate_id_by_index[idx] = precursor_to_id[canonical_candidate]

    merged["candidate_id"] = merged.index.map(candidate_id_by_index)

    if merged["candidate_id"].isna().any():
        raise ValueError("Some rows failed to receive candidate_id.")

    # Put candidate_id before model for readability.
    cols = list(merged.columns)

    if "candidate_id" in cols and "model" in cols:
        cols.remove("candidate_id")
        model_pos = cols.index("model")
        cols.insert(model_pos, "candidate_id")
        merged = merged[cols]

    return merged


def validate_candidate_ids(merged: pd.DataFrame):
    """
    Validate candidate_id consistency.

    Must hold:
    1. Same sample_index + canonical_candidate has exactly one candidate_id.
    2. Same sample_index + candidate_id maps to exactly one canonical_candidate.
    """
    bad_precursor_to_id = (
        merged.groupby(["sample_index", "canonical_candidate"])["candidate_id"]
        .nunique()
        .reset_index(name="n_candidate_ids")
    )

    bad_precursor_to_id = bad_precursor_to_id[
        bad_precursor_to_id["n_candidate_ids"] > 1
    ]

    if len(bad_precursor_to_id) > 0:
        print("[VALIDATION_ERROR] Same precursor got multiple candidate_id values:")
        print(bad_precursor_to_id.head(20).to_string(index=False))
        raise ValueError("candidate_id validation failed: precursor -> multiple ids")

    bad_id_to_precursor = (
        merged.groupby(["sample_index", "candidate_id"])["canonical_candidate"]
        .nunique()
        .reset_index(name="n_precursors")
    )

    bad_id_to_precursor = bad_id_to_precursor[
        bad_id_to_precursor["n_precursors"] > 1
    ]

    if len(bad_id_to_precursor) > 0:
        print("[VALIDATION_ERROR] Same candidate_id maps to multiple precursors:")
        print(bad_id_to_precursor.head(20).to_string(index=False))
        raise ValueError("candidate_id validation failed: id -> multiple precursors")

    print("[VALIDATION] candidate_id consistency OK")


def merge_batch_predictions(input_dir: Path, output_csv: Path) -> pd.DataFrame:
    """
    Merge all canonicalized model prediction CSV files into one merged_candidates.csv.

    Output behavior:
    - Keeps one row per sample_index + model + canonical_candidate.
    - Keeps cross-model duplicate canonical_candidate rows.
    - Adds candidate_id shared by identical canonical_candidate within the same sample_index.
    """
    input_dir = Path(input_dir)
    output_csv = Path(output_csv)

    print(f"[MERGE] input_dir={input_dir}")
    print(f"[MERGE] output_csv={output_csv}")

    merged = load_prediction_csvs(input_dir)
    merged = clean_merged_candidates(merged)
    merged = add_candidate_ids(merged)
    validate_candidate_ids(merged)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False)

    print(f"[MERGE] Saved: {output_csv}")
    print(f"[MERGE] Rows: {len(merged)}")
    print(f"[MERGE] Unique sample_index: {merged['sample_index'].nunique()}")
    print(
        "[MERGE] Unique candidate_id per sample_index mean: "
        f"{merged.groupby('sample_index')['candidate_id'].nunique().mean():.3f}"
    )

    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_csv", required=True)

    args = parser.parse_args()

    merge_batch_predictions(
        input_dir=Path(args.input_dir),
        output_csv=Path(args.output_csv),
    )


if __name__ == "__main__":
    main()
