# Retro-Ensemble

Retro-Ensemble integrates predictions from four single-step retrosynthesis models and reranks candidate precursor sets using large language models (LLMs). It supports single-LLM selection and Mixture-of-Agents (MoA) aggregation without training an additional aggregation model.

## Retrosynthesis models and dataset

| Model | Source repository |
| --- | --- |
| Augmented Transformer | [ASKCOS implementation](https://gitlab.com/mlpds_mit/askcosv2/retro/augmented_transformer) |
| Graph2SMILES | [ASKCOS implementation](https://gitlab.com/mlpds_mit/askcosv2/retro/graph2smiles) |
| Template Relevance | [ASKCOS implementation](https://gitlab.com/mlpds_mit/askcosv2/retro/template_relevance) |
| RetroSim | [Original implementation](https://github.com/connorcoley/retrosim) |

The exact source revisions and compatibility patches are recorded in [baseline_sources.json](baseline_sources.json).

**USPTO-50K dataset**

- Original data: [Schneider et al., J. Chem. Inf. Model. (2016)](https://doi.org/10.1021/acs.jcim.6b00564) — [download the original supporting data (ZIP)](https://pubs.acs.org/doi/suppl/10.1021/acs.jcim.6b00564/suppl_file/ci6b00564_si_002.zip).
- Processed benchmark: [Coley et al., ACS Cent. Sci. (2017)](https://doi.org/10.1021/acscentsci.7b00355) — [data_processed.csv](https://github.com/connorcoley/retrosim/blob/0a272f0b5de833c448f41491e81e4dc00b4d85b0/retrosim/data/data_processed.csv).

The study uses 40,008 training, 5,001 validation and 5,007 test reactions. Obtain the benchmark dataset separately and prepare the input CSV as described below. The inference code and baseline model resources are provided here.

## 1. Installation

Use Linux with Git, Conda and Python 3.10 for the selector environment. Each retrosynthesis model has a separate environment.

```bash
git clone https://github.com/stellahuang04/Retro-Ensemble.git
cd Retro-Ensemble
conda create -n retroensemble_selector python=3.10 -y
conda activate retroensemble_selector
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp .env.example .env
python -m pip check
python scripts/check_setup.py --llm-only
```

On older Linux distributions with glibc 2.17 (e.g., CentOS 7), replace the requirements installation command with:

```bash
python -m pip install --only-binary=:all: \
  -r requirements.txt -c constraints-centos7.txt
```

The selector uses remote LLM APIs and does not require local PyTorch, TensorFlow or Flax. Model-specific dependencies are installed separately. The first LLM run downloads the [Qwen tokenizer](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507) for prompt-size checks. Set `TOKENIZER_NAME` in `.env` to a local tokenizer directory if needed.

## 2. Set up the retrosynthesis models

Skip this section and the model-asset download if you already have a compatible `merged_candidates.csv` and only want to rerank it.

```bash
bash scripts/setup_baseline_sources.sh
```

Follow [docs/INSTALL_BASELINES.md](docs/INSTALL_BASELINES.md) to create the four environments and configure their Python interpreters in `.env`.

The default source directories are under `external/baselines/`. Existing installations may be configured using the `*_REPO` variables in `.env.example`.

## 3. Download model assets

Download `retroensemble-baselines-v1.tar.gz` and `SHA256SUMS.txt` from the [baseline-assets-v1 release](https://github.com/stellahuang04/Retro-Ensemble/releases/tag/baseline-assets-v1) into `downloads/` (create the directory with `mkdir -p downloads`).

Verify the archive before extracting it:

```bash
(
  set -e
  cd downloads
  sha256sum -c SHA256SUMS.txt
  printf '%s  %s\n' 'e158429e62270fe163deec16cdfc3e02f302b82accc555c8cb053fa1c3a8e367' 'retroensemble-baselines-v1.tar.gz' | sha256sum -c -
)
```

Continue only if both checks pass.

```bash
tar -tzf downloads/retroensemble-baselines-v1.tar.gz
tar -xzf downloads/retroensemble-baselines-v1.tar.gz \
  --skip-old-files --no-same-owner -C external/baselines
```

The archive contains six resources:

| Model | Resource path relative to the model's source directory | Environment override |
| --- | --- | --- |
| Augmented Transformer | `checkpoints/uspto_50k/model_step_500000.pt` | `AT_CHECKPOINT` |
| Graph2SMILES | `checkpoints/uspto_50k/model.500000_49.pt` | `G2S_CHECKPOINT` |
| Graph2SMILES | `data/uspto_50k/processed/vocab.txt` | `G2S_DATA_DIR` (directory) |
| Template Relevance | `checkpoints/grid_search_uspto_50k/ep30_bs512_hid1024x1024_drop0.3_lr0.001/model_latest.pt` | `TR_CHECKPOINT` |
| Template Relevance | `data/uspto_50k/processed/templates.jsonl` | `TR_TEMPLATES` |
| RetroSim | `retrosim/data/data_processed.csv` | `RETROSIM_DATA` |

RetroSim uses a reference reaction corpus rather than a neural-network checkpoint. Extraction skips existing files; compare any pre-existing resource against the release copy before using it. If they differ, extract the released resource into a separate directory and set its override in `.env`.

After configuring the four interpreters and resources:

```bash
python scripts/check_setup.py
```

This checks dependencies, executable paths and resource locations; it does not load model weights.

## 4. Configure LLM APIs

Supported providers are [NCHC GenAI](https://rap-docs.genai.nchc.org.tw/docs/api/taiwan-ai-rap-api/) and [Cerebras](https://inference-docs.cerebras.ai/).

For NCHC, set the following in the local `.env` file:

```dotenv
NCHC_API_KEY_NAME=NCHC_API1
NCHC_API1=YOUR_NCHC_API_KEY
LLM_MAX_COMPLETION_TOKENS=8192
```

For Cerebras, set `CEREBRAS_API=YOUR_CEREBRAS_API_KEY`. Keep `.env` local and never commit API keys. Use valid credentials and model identifiers supported by your provider; availability, quota and usage charges depend on your account.

## 5. Run inference

### Input format

The input CSV requires `id` and `rxn_smiles`. The optional `target` column contains the product SMILES; otherwise, the product is taken from the reaction SMILES after the last `>`. See [examples/targets.csv](examples/targets.csv) for a single-target input. For a full benchmark run, use a CSV prepared from the appropriate test split with these columns.

### Single-LLM selection

```bash
python batch_run_testset.py \
  --test_csv examples/targets.csv \
  --batch_work_name single_llm \
  --top_k 30 \
  --candidate_cutoff 20 \
  --llm_output_top_k 20 \
  --llm_provider nchc \
  --llm_model gemma-4-31B-it \
  --reasoning
```

### Mixture-of-Agents (MoA)

```bash
python batch_run_moa_testset.py \
  --test_csv examples/targets.csv \
  --batch_work_name moa \
  --top_k 30 \
  --candidate_cutoff 20 \
  --agent_output_top_k 20 \
  --agent_models Llama-3.3-70B-Instruct gpt-oss-120b \
  --agent_providers nchc nchc \
  --aggregator_model gemma-4-31B-it \
  --aggregator_provider nchc \
  --agent_workers 1 \
  --reasoning
```

Each of the two agents produces a ranked list of up to 20 precursor sets. The aggregator receives both responses and the original candidate pool and returns up to 10 precursor sets.

Model identifiers are provider-specific. Change them if needed. Use `--no-reasoning` to request rankings without explicit rationales. `--top_k` is the baseline prediction limit, and `--candidate_cutoff` controls the number of predictions per model supplied to the LLMs. The examples use a baseline limit of 30, not a specification of all paper experiment settings.

### Reranking an existing candidate pool

Reuse the `merged_candidates.csv` generated by the pipeline. For single-LLM selection, copy the existing pool into a new work directory and skip candidate generation:

```bash
mkdir -p outputs/single_rerank
cp outputs/single_llm/merged_candidates.csv outputs/single_rerank/merged_candidates.csv

python batch_run_testset.py \
  --test_csv examples/targets.csv \
  --batch_work_name single_rerank \
  --candidate_cutoff 20 --llm_output_top_k 20 \
  --llm_provider nchc --llm_model gemma-4-31B-it \
  --reasoning --skip_models --skip_canonicalize --skip_merge
```

With all three skip flags, the existing merged pool determines the input samples; the required `--test_csv` argument is not read.

For MoA, use the MoA command above with `--existing_merged_csv /path/to/merged_candidates.csv` in place of `--test_csv examples/targets.csv --top_k 30` and choose a new `--batch_work_name`. This bypasses the four retrosynthesis models.

To generate the candidate pool without LLM inference, add `--skip_moa` to the full MoA command and use a separate work name.

## 6. Outputs

Results are saved in `outputs/<batch_work_name>/`:

- `model_predictions/`: baseline predictions.
- `canonicalized_predictions/`: standardized precursor sets.
- `merged_candidates.csv`: combined candidates with model, rank, score and candidate ID.
- `batch_results.csv`: ranked precursor SMILES and per-sample status.
- `samples/`: prompts, raw responses, parsed JSON and ranking reports.

Rank positions are preserved, including invalid candidate IDs and duplicate selections. Use a new work directory for each run. With `--resume`, previously recorded samples (including errors) are skipped; inspect the status column before resuming. Use `--start` and `--limit` for a subset of the input.

The repository provides inference code and baseline setup resources, not the full evaluation and analysis pipeline. Refer to [SOURCE.json](SOURCE.json) for code provenance. Third-party code, data and model resources retain their respective terms. No repository-wide license has been assigned.
