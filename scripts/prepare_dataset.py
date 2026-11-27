#!/usr/bin/env python3
"""Prepare a reaction CSV for Retro-Ensemble inference.

Preserve input row order, reaction IDs and original reaction SMILES.
Remove atom mapping only from the product and canonicalize it with RDKit.
"""
import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem


def canonical_product(rxn_smiles: str) -> str:
    parts = rxn_smiles.split(">")
    if len(parts) != 3 or not parts[2].strip():
        raise ValueError("Expected reactants>reagents>product with a nonempty product")
    mol = Chem.MolFromSmiles(parts[2].strip())
    if mol is None:
        raise ValueError("Invalid product SMILES")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", required=True, type=Path)
    parser.add_argument("--output_csv", required=True, type=Path)
    parser.add_argument("--expected_rows", type=int,
                        help="Require this many input and output reactions")
    args = parser.parse_args()

    if args.input_csv.resolve() == args.output_csv.resolve():
        parser.error("Input and output CSV paths must differ")
    if args.expected_rows is not None and args.expected_rows < 1:
        parser.error("--expected_rows must be positive")

    data = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    if "id" not in data.columns:
        raise ValueError("Input CSV must contain an id column")
    if "rxn_smiles" in data.columns:
        reaction_col = "rxn_smiles"
    elif "reactants>reagents>production" in data.columns:
        reaction_col = "reactants>reagents>production"
    else:
        raise ValueError("Input CSV must contain rxn_smiles or reactants>reagents>production")

    if data.empty:
        raise ValueError("Input CSV contains no reactions")
    if args.expected_rows is not None and len(data) != args.expected_rows:
        raise ValueError(f"Expected {args.expected_rows} reactions, found {len(data)}")

    rows = []
    for index, row in data.iterrows():
        sample_id = row["id"]
        rxn_smiles = row[reaction_col]
        try:
            if not sample_id.strip():
                raise ValueError("Empty reaction ID")
            target = canonical_product(rxn_smiles)
            if not target:
                raise ValueError("Empty canonical product")
        except Exception as exc:
            raise ValueError(
                f"Invalid reaction at row {index + 1}, id={sample_id!r}: {exc}"
            ) from exc
        rows.append({"id": sample_id, "rxn_smiles": rxn_smiles, "target": target})

    out = pd.DataFrame(rows, columns=["id", "rxn_smiles", "target"])
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"[INFO] Input rows: {len(data)}")
    print(f"[INFO] Output rows: {len(out)}")
    print(f"[INFO] Saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
