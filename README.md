# RetroEnsemble

LLM-based selection of retrosynthesis candidates from Augmented Transformer (AT), Graph2SMILES (G2S), Template Relevance (TR), and RetroSim (RS).

The main pipeline generates baseline candidates, canonicalizes them, merges their model/rank/score information, queries proposer agents, and asks an aggregator to produce the final ranking. Single-LLM inference is also retained because it shares code with MoA. Plausibility evaluation, downstream MoA analyses, weighted voting/ensemble training, and RetroKNN inference are outside this release.

## Status

This is an initial portable source package derived from `stellahuang04/askcos-llm-selector`, branch `uspto50k-askcos-moa`. It is not yet a fully verified paper reproduction: external baseline source code, exact model assets, and baseline environment exports must be supplied. See [SETUP_TODO.md](SETUP_TODO.md).

## Installation

Use Python 3.10 or newer. Run commands from this repository root.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

`requirements.txt` covers the orchestration/LLM environment only. Its minimum versions are not claimed to reproduce the original experiment environment. `requirements-original.txt` preserves the incomplete dependency list from the source branch for provenance. Each baseline may require a separate environment, including its own PyTorch/OpenNMT/RDChiral dependencies. Do not install all baseline frameworks into the selector environment blindly.

Set the required API keys in `.env`. Supported providers are `cerebras` and `nchc`; model identifiers must match the selected provider's accessible API. For NCHC, set `NCHC_API_KEY_NAME` to the variable name containing the selected key (for example `NCHC_API1`). No credentials are included.

## Baseline code and resources

Default code directories are `external/askcos_retro_mamba/augmented_transformer`, `graph2smiles`, `template_relevance`, and `retrosim`. These directories are not bundled. Alternatively set the four `*_REPO` variables in `.env` to existing absolute paths. Set `*_PYTHON` to each baseline's interpreter. The pipeline loads `.env` and passes these variables to its subprocesses.

| Baseline | Default resource path relative to its code directory | Override |
| --- | --- | --- |
| AT | `checkpoints/uspto_50k/model_step_500000.pt` | `AT_CHECKPOINT` |
| G2S | `checkpoints/uspto_50k/model.500000_49.pt` | `G2S_CHECKPOINT` |
| G2S vocabulary | `data/uspto_50k/processed/vocab.txt` | `G2S_DATA_DIR` (directory) |
| TR | `checkpoints/grid_search_uspto_50k/ep30_bs512_hid1024x1024_drop0.3_lr0.001/model_latest.pt` | `TR_CHECKPOINT` |
| TR templates | `data/uspto_50k/processed/templates.jsonl` | `TR_TEMPLATES` |
| RetroSim corpus | `retrosim/data/data_processed.csv` | `RETROSIM_DATA` |

RetroSim uses its training corpus and template extraction code, not a neural-network checkpoint. The paths above are the source branch's configured paths; the author must verify the exact experiment assets before release. Downloads and checksums are intentionally not fabricated.

## Run MoA from an existing candidate pool

This skips all baseline inference. Use a `merged_candidates.csv` produced by the original pipeline or this repository. It must retain the original sample IDs, model labels, ranks, scores, and candidate IDs.

The following command explicitly selects Llama + GPT as agents and Gemma as aggregator. Confirm the exact API model IDs/providers and candidate cutoff against your experiment job scripts before claiming paper reproduction.

```bash
python batch_run_moa_testset.py \
  --existing_merged_csv /absolute/path/to/merged_candidates.csv \
  --batch_work_name moa_smoke \
  --candidate_cutoff 20 \
  --agent_output_top_k 20 \
  --agent_models llama-3.3-70b gpt-oss-120b \
  --agent_providers cerebras cerebras \
  --aggregator_model gemma-4-31b \
  --aggregator_provider nchc \
  --reasoning --limit 1
```

The example provider/model mapping is a configuration example, not a guarantee of current account availability. Change it as needed. API calls can incur charges. Use a new `--batch_work_name` for a fresh run; remove `--limit 1` for the full input. `--start` and `--limit` select sample-index ranges in an existing pool. Replace `--reasoning` with `--no-reasoning` for the corresponding prompts.

## Run the full pipeline

Configure baseline resources and interpreters first, then replace `--existing_merged_csv ...` in the command above with:

```bash
--test_csv examples/targets.csv --top_k 300
```

Input CSV requires `id` and `rxn_smiles`; `target` is optional. If `target` is absent, the product is extracted after the last `>` in `rxn_smiles`. The example uses an empty reactant side solely to supply a product; it is not an experimental ground truth. Dataset files from the old repository are not bundled.

`--top_k` is the baseline candidate limit; `--candidate_cutoff` limits each model's ranks passed to the LLM. AT and G2S retain the original beam size and n-best of 30, so requesting 300 does not generate 300 outputs from these models. Default cutoff remains 20; set the experimental value explicitly. Agents default to Top-20 and the aggregator returns Top-10 (`FINAL_OUTPUT_TOP_K` in `moa_prompt.py`).

## Outputs and other entry points

Results are written to `outputs/<batch_work_name>/batch_results.csv`; sample directories contain prompts, raw responses, parsed JSON and final reports. Invalid candidate IDs and duplicate selections retain their original rank slots. JSON repair changes syntax, not the intended candidate selections.

- `batch_run_moa_testset.py`: full MoA pipeline or existing-pool selection.
- `batch_run_testset.py`: Single-LLM pipeline and shared preparation helpers; retain this file even for MoA-only use.
- `rerun_moa_aggregator.py`: rerun only the aggregator from saved agent outputs (`--help` lists inputs).
- `prompt.py`, `moa_prompt.py`: shared agent and aggregator prompts.
- `tools/predict_scripts/`: four baseline adapters, not their underlying frameworks.

Run `python scripts/check_setup.py` to check assets/interpreters and `python scripts/check_setup.py --llm-only` when using an existing pool. Checks do not call an LLM or load model weights. Prompt-size checks lazily load the original Qwen tokenizer on the first actual inference; set `TOKENIZER_NAME` to a local tokenizer directory for offline use.

`--resume` retains the original behavior: any sample index already recorded in the result CSV is skipped, including errors. Inspect errors before using it. No automatic retry of previously recorded failures is promised.

## Provenance and licensing

See `SOURCE.json` for the source commit and the scope of changes. Prompt content and ranking logic are retained; deployment paths, dependency declarations, tokenizer loading and documentation were adjusted. The source branch has no license file. No new license is assigned here: the author must select the repository license and confirm redistribution terms for external frameworks, data, and weights before public release. No paper DOI or citation metadata is invented.
