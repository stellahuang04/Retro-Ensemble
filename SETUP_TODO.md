# Author setup checklist

- [ ] Verify exact experiment job arguments: LLM model IDs/providers, candidate cutoff, agent Top-k and aggregator Top-k. The source defaults do not match the intended Llama + GPT → Gemma combination; use explicit CLI arguments.
- [ ] Supply the four baseline source trees or their exact repository URLs and commits, including local changes. A checkpoint alone is insufficient.
- [ ] Supply AT `model_step_500000.pt` and G2S `model.500000_49.pt` plus its matching `vocab.txt`.
- [ ] Supply TR `model_latest.pt` from the configured grid-search run and matching `templates.jsonl`.
- [ ] Supply RetroSim `data_processed.csv` and the exact preprocessing/split implementation.
- [ ] Export the actual selector and four baseline conda/mamba environments. Remove credentials/private package index URLs from exports before sharing.
- [ ] Verify assets against the job logs/results used in the paper; record file sizes, SHA-256 values and training-data provenance in a model release manifest.
- [ ] Choose a model hosting location and provide verified download URLs; no dummy download script is included.
- [ ] Set local resource/interpreter paths and API keys in `.env` (do not upload this file).
- [ ] Run `scripts/check_setup.py`, then a one-sample end-to-end job on the original server; compare candidate pool and prompts with a saved experimental run.
- [ ] Select code license and confirm third-party/data/model redistribution terms before making the repository public.

The package intentionally excludes original train/test CSVs, outputs, plausibility evaluations and downstream MoA analyses. Existing candidate pools can be used without distributing those files in Git.
