# Author setup checklist

- [ ] Validate a user-configured LLM/provider combination and one-target run. Users choose their own cutoff and ranking settings; paper experiment job scripts are not required.
- [x] Record baseline repository URLs, commits and tracked runtime compatibility changes. See baseline_sources.json and patches/.
- [ ] Verify clean upstream checkout and patch application; run real baseline inference.
- [ ] Supply AT `model_step_500000.pt` and G2S `model.500000_49.pt` plus its matching `vocab.txt`.
- [ ] Supply TR `model_latest.pt` from the configured grid-search run and matching `templates.jsonl`.
- [ ] Supply RetroSim `data_processed.csv` and the exact preprocessing/split implementation.
- [x] Collect selector and four baseline environment exports; confirm baseline Python mappings and remove machine-specific prefixes.
- [ ] Verify installation of the exported environments on a clean Linux system.
- [ ] Verify assets against the job logs/results used in the paper; record file sizes, SHA-256 values and training-data provenance in a model release manifest.
- [ ] Choose a model hosting location and provide verified download URLs; no dummy download script is included.
- [ ] Set local resource/interpreter paths and API keys in `.env` (do not upload this file).
- [ ] Run `scripts/check_setup.py`, then a one-sample end-to-end job on the original server; compare candidate pool and prompts with a saved experimental run.
- [ ] Select code license and confirm third-party/data/model redistribution terms before making the repository public.

The package intentionally excludes original train/test CSVs, outputs, plausibility evaluations and downstream MoA analyses. Existing candidate pools can be used without distributing those files in Git.
