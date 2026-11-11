# RetroEnsemble

LLM-based selection of retrosynthesis candidates from Augmented Transformer (AT), Graph2SMILES (G2S), Template Relevance (TR), and RetroSim (RS).

The main pipeline generates baseline candidates, canonicalizes them, merges their model/rank/score information, queries proposer agents, and asks an aggregator to produce the final ranking. Single-LLM inference is also retained because it shares code with MoA. Plausibility evaluation, downstream MoA analyses, weighted voting/ensemble training, and RetroKNN inference are outside this release.

## Status

This portable source package was derived from `stellahuang04/askcos-llm-selector`, branch `uspto50k-askcos-moa`. A clean-clone installation on the author's older CentOS Linux server has verified four baseline source checkouts, installed their environments and released assets, and completed a one-target baseline inference. The offline MoA mock run and a replay of one recorded real three-model response also completed, matching the saved final Top-10 precursor SMILES. These checks do **not** validate live LLM API inference, the entire USPTO-50K experiment, or exact paper reproducibility. See [SETUP_TODO.md](SETUP_TODO.md).

## Installation

Use **Python 3.10 or newer** for the selector environment. The server's default `python` may be older (for example, Python 3.8 cannot satisfy the `transformers>=4.51` requirement). Run commands from this repository root. A Conda environment is a straightforward option:

```bash
conda create -n retroensemble_selector python=3.10 -y
conda activate retroensemble_selector
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python scripts/check_setup.py --llm-only
```

Alternatively, if a suitable interpreter is already installed, create a virtual environment with `python3.10 -m venv .venv` and `source .venv/bin/activate` before installing requirements. Do not run a bare `python -m venv` without checking `python --version`.

**Older CentOS / glibc 2.17:** Unconstrained pip may download a newer pandas source archive and attempt compilation with an old GCC, which failed during the clean-install test. Instead, **in a freshly created Python 3.10 environment**, replace the ordinary requirements installation command above with:

```bash
python -m pip install --only-binary=:all: \
  -r requirements.txt -c constraints-centos7.txt
```

This uses `pandas==2.2.3`, for which a compatible prebuilt wheel was available on the tested server. The constraints file is a compatibility workaround, not the exact paper dependency lockfile; installation on other platforms is not guaranteed. Verify the environment with `python -m pip check` and `python scripts/check_setup.py --llm-only`.

`requirements.txt` covers the orchestration/LLM environment only. Its minimum versions are not claimed to reproduce the original experiment environment. `requirements-original.txt` preserves the incomplete dependency list from the source branch for provenance. Each baseline may require a separate environment, including its own PyTorch/OpenNMT/RDChiral dependencies. Do not install all baseline frameworks into the selector environment blindly.

Set the required API keys in `.env`. Supported providers are `cerebras` and `nchc`; model identifiers must match the selected provider's accessible API. For NCHC, set `NCHC_API_KEY_NAME` to the variable name containing the selected key (for example `NCHC_API1`). No credentials are included.

## Baseline code and resources

Default code directories are `external/baselines/augmented_transformer`, `graph2smiles`, `template_relevance`, and `retrosim`. These directories are not bundled. See [baseline installation](docs/INSTALL_BASELINES.md) for pinned source checkout, compatibility patches and environment creation. Alternatively set the four `*_REPO` variables in `.env` to existing absolute paths. Set `*_PYTHON` to each baseline's interpreter. The pipeline loads `.env` and passes these variables to its subprocesses.

| Baseline | Default resource path relative to its code directory | Override |
| --- | --- | --- |
| AT | `checkpoints/uspto_50k/model_step_500000.pt` | `AT_CHECKPOINT` |
| G2S | `checkpoints/uspto_50k/model.500000_49.pt` | `G2S_CHECKPOINT` |
| G2S vocabulary | `data/uspto_50k/processed/vocab.txt` | `G2S_DATA_DIR` (directory) |
| TR | `checkpoints/grid_search_uspto_50k/ep30_bs512_hid1024x1024_drop0.3_lr0.001/model_latest.pt` | `TR_CHECKPOINT` |
| TR templates | `data/uspto_50k/processed/templates.jsonl` | `TR_TEMPLATES` |
| RetroSim corpus | `retrosim/data/data_processed.csv` | `RETROSIM_DATA` |

RetroSim uses its training corpus and template extraction code, not a neural-network checkpoint. The paths above are the source branch's configured paths; the author must verify the exact experiment assets before release. Download and verification instructions are provided below.


## Download model assets

The [baseline-assets-v1 release](https://github.com/stellahuang04/RetroEnsemble/releases/tag/baseline-assets-v1) contains `retroensemble-baselines-v1.tar.gz` (530,503,210 bytes) and `SHA256SUMS.txt`. Download both from that page. While this repository is private, sign in with an account that has repository access. Place both files in `downloads/` under this repository.

GitHub reports this SHA-256 for the uploaded archive:

```text
e158429e62270fe163deec16cdfc3e02f302b82accc555c8cb053fa1c3a8e367
```

After preparing the baseline sources, run from the repository root:

```bash
(
  set -e
  cd downloads
  sha256sum -c SHA256SUMS.txt
  printf '%s  %s\n' 'e158429e62270fe163deec16cdfc3e02f302b82accc555c8cb053fa1c3a8e367' 'retroensemble-baselines-v1.tar.gz' | sha256sum -c -
)
```

Continue only if both checks succeed. Inspect the archive listing:

```bash
tar -tzf downloads/retroensemble-baselines-v1.tar.gz
```

It should contain only the six relative model-resource paths listed above, under the four baseline directories. Extract into a fresh baseline checkout:

```bash
tar -xzf downloads/retroensemble-baselines-v1.tar.gz \
  --keep-old-files --no-same-owner \
  -C external/baselines
```

Extraction refuses to overwrite existing files. If upstream already includes one of these data files, compare it with the release copy in a separate temporary directory before replacing it. Alternatively, extract the archive into a separate empty asset directory and configure the resource overrides in `.env` (including `G2S_DATA_DIR`).

The downloaded release archive passed SHA-256 checks during the clean installation on the author's server, and the four baseline models produced candidates on one target. This checks practical asset loading but does not establish that the released weights, templates or training data are identical to the final paper experiment assets.

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

Run `python scripts/check_setup.py` to check assets/interpreters and `python scripts/check_setup.py --llm-only` when using an existing pool. Checks do not call an LLM or load model weights. Configure **each** `AT_PYTHON`, `G2S_PYTHON`, `TR_PYTHON` and `RETROSIM_PYTHON` in `.env`: when omitted the baseline runner falls back to the selector Python, and a setup check reporting an executable as `OK` does not prove its framework dependencies work. Prompt-size checks lazily load the original Qwen tokenizer on the first actual inference; set `TOKENIZER_NAME` to a local tokenizer directory for offline use.

`--resume` retains the original behavior: any sample index already recorded in the result CSV is skipped, including errors. Inspect errors before using it. No automatic retry of previously recorded failures is promised.

## Validation scope

On the author's older CentOS Linux server, a new clone successfully generated baseline candidates, completed an offline MoA run using synthetic replies, and replayed a recorded Llama/GPT/Gemma sample through the MoA code. The replayed final Top-10 SMILES matched the saved result; two recorded responses required JSON syntax repair. Offline test scripts, recorded replies, original datasets and test outputs are **not** included in this repository. These results demonstrate the checked paths only; live LLM API operation, invalid-ID edge cases and full USPTO-50K reproducibility remain unverified.

## Provenance and licensing

See `SOURCE.json` for the source commit and the scope of changes. Prompt content and ranking logic are retained; deployment paths, dependency declarations, tokenizer loading and documentation were adjusted. The source branch has no license file. No new license is assigned here: the author must select the repository license and confirm redistribution terms for external frameworks, data, and weights before public release. No paper DOI or citation metadata is invented.
