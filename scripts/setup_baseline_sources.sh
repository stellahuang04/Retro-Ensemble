#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/external/askcos_retro_mamba"
mkdir -p "$DEST"
# Refuse to overwrite or modify existing source trees.
if [ -e "$DEST/augmented_transformer" ]; then echo "Already exists: $DEST/augmented_transformer; use a fresh checkout or configure *_REPO in .env." >&2; exit 1; fi
if [ -e "$DEST/graph2smiles" ]; then echo "Already exists: $DEST/graph2smiles; use a fresh checkout or configure *_REPO in .env." >&2; exit 1; fi
if [ -e "$DEST/template_relevance" ]; then echo "Already exists: $DEST/template_relevance; use a fresh checkout or configure *_REPO in .env." >&2; exit 1; fi
if [ -e "$DEST/retrosim" ]; then echo "Already exists: $DEST/retrosim; use a fresh checkout or configure *_REPO in .env." >&2; exit 1; fi
git clone --no-checkout "https://gitlab.com/mlpds_mit/askcosv2/retro/augmented_transformer.git" "$DEST/augmented_transformer"
git -C "$DEST/augmented_transformer" checkout --detach "2d053bbaec51af5cceebc755f7041a90cb9191be"
git clone --no-checkout "https://gitlab.com/mlpds_mit/askcosv2/retro/graph2smiles.git" "$DEST/graph2smiles"
git -C "$DEST/graph2smiles" checkout --detach "bc6c34d533682c837f2c7c4583ffcd7aa9bed848"
git -C "$DEST/graph2smiles" apply --check "$ROOT/patches/graph2smiles.patch"
git -C "$DEST/graph2smiles" apply "$ROOT/patches/graph2smiles.patch"
git clone --no-checkout "https://gitlab.com/mlpds_mit/askcosv2/retro/template_relevance.git" "$DEST/template_relevance"
git -C "$DEST/template_relevance" checkout --detach "8268a97769c3af39fb07f214aa71c07bd24de785"
git clone --no-checkout "https://github.com/connorcoley/retrosim.git" "$DEST/retrosim"
git -C "$DEST/retrosim" checkout --detach "0a272f0b5de833c448f41491e81e4dc00b4d85b0"
git -C "$DEST/retrosim" apply --check "$ROOT/patches/retrosim.patch"
git -C "$DEST/retrosim" apply "$ROOT/patches/retrosim.patch"
echo "Baseline sources prepared. Install environments and supply model assets next."
