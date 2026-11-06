#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from rdkit import Chem


REQUIRED_COLUMNS = [
    "sample_index",
    "id",
    "target",
    "model",
    "rank",
    "score",
    "candidate",
]


OUTPUT_COLUMNS = [
    "sample_index",
    "id",
    "target",
    "model",
    "rank",
    "score",
    "raw_candidate",
    "canonical_candidate",
    "error",
]


def canonicalize_smiles(smiles: str) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None

    mol = Chem.MolFromSmiles(smiles.strip())

    if mol is None:
        return None

    for atom in mol.GetAtoms():
        if atom.HasProp("molAtomMapNumber"):
            atom.ClearProp("molAtomMapNumber")

    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)


def canonicalize_precursors(precursor_string: str) -> str | None:
    if not isinstance(precursor_string, str) or not precursor_string.strip():
        return None

    parts = precursor_string.split(".")
    canonical_parts = []

    for part in parts:
        canonical = canonicalize_smiles(part)
        if canonical is None:
            return None
        canonical_parts.append(canonical)

    canonical_parts = sorted(canonical_parts)

    return ".".join(canonical_parts)


def read_csv(path: Path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def canonicalize_file(input_csv: Path, output_csv: Path):
    rows = read_csv(input_csv)
    new_rows = []

    for row in rows:
        for col in REQUIRED_COLUMNS:
            if col not in row:
                raise ValueError(f"{input_csv} missing required column: {col}")

        raw = row.get("candidate", "")

        try:
            canonical = canonicalize_precursors(raw)
            error = ""

            if canonical is None:
                canonical = ""
                error = "canonicalize_failed"

        except Exception as e:
            canonical = ""
            error = str(e)

        new_rows.append(
            {
                "sample_index": row["sample_index"],
                "id": row["id"],
                "target": row["target"],
                "model": row["model"],
                "rank": row["rank"],
                "score": row["score"],
                "raw_candidate": raw,
                "canonical_candidate": canonical,
                "error": error,
            }
        )

    write_csv(new_rows, output_csv)
    print(f"[CANONICALIZE] {input_csv.name} -> {output_csv}")


def canonicalize_batch(input_dir: Path, output_dir: Path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input prediction directory not found: {input_dir}")

    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No prediction CSV files found in: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    for csv_file in csv_files:
        canonicalize_file(
            input_csv=csv_file,
            output_csv=output_dir / csv_file.name,
        )

    print("[CANONICALIZE] All files finished.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    canonicalize_batch(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
