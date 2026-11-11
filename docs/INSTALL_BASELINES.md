# Install the baseline models

These instructions prepare the four baseline implementations used by the adapters. They are for running the architecture on your own inputs; paper job settings are not required.

## Source code

From the repository root, run:

```bash
bash scripts/setup_baseline_sources.sh
```

This clones the repositories and checks out the exact commits in `baseline_sources.json`, then applies the included compatibility patches. It refuses existing source directories. If you already have working baseline installations, configure `AT_REPO`, `G2S_REPO`, `TR_REPO`, and `RETROSIM_REPO` in `.env` instead.

The G2S patch loads checkpoints onto the CPU initially and reports loading failures. The RetroSim patch replaces obsolete dictionary and pandas APIs. Benchmark scripts and training jobs are not part of this setup.

## Environments

The YAML files are exports from the author's original Linux environments, with machine-specific `prefix` removed. They are reference snapshots, **not** minimal cross-platform dependency lists or exact paper environment locks. A clean installation of the four baseline environments was completed on the author's older CentOS server, but GPU compatibility and installation on other systems remain unverified. Use a separate selector environment installed as described in the main README.

With Conda, create each baseline environment:

```bash
for name in transformer graph2smiles template_relevance retrosim; do
    conda env create -n "retroensemble_${name}" \
      -f "environments/${name}.environment.yml" || break
done
```

The equivalent Micromamba command is `micromamba create -n retroensemble_transformer -f environments/transformer.environment.yml` (repeat for the other three YAML files). Do **not** recreate an existing environment: first check `conda env list` and verify its dependencies. Some Conda and pip packages overlap; resolve installation failures before proceeding.

Use distinct new environment names to avoid changing existing environments. All four baseline interpreter mappings were confirmed from the author's batch runner:

| Adapter | Exported environment | .env variable |
| --- | --- | --- |
| AT | transformer | AT_PYTHON |
| G2S | graph2smiles | G2S_PYTHON |
| TR | template_relevance | TR_PYTHON |
| RetroSim | retrosim | RETROSIM_PYTHON |

Run `conda env list` (or `micromamba env list`) to locate the created environments. Set **all four** `*_PYTHON` variables above in your local `.env` to each environment's absolute `bin/python` path. For example, `conda run -n retroensemble_transformer python -c 'import sys; print(sys.executable)'` prints the AT interpreter path. If these variables are omitted, baseline subprocesses default to the selector's interpreter; `scripts/check_setup.py` alone does not catch missing baseline-specific dependencies. The optional `askcos_llm.environment.yml` is a historical export, not required when the selector was installed from `requirements.txt`; its Cerebras SDK version differs from the lightweight requirements, so do not layer both installations blindly. Real API compatibility remains unverified.

## Model assets and validation

Supply the six resource files at the paths in the main README, or set their overrides in `.env`. They are distributed in the baseline-assets-v1 Release. Follow the download, checksum and extraction instructions in the main README. Do not regenerate the vocabulary or templates independently of their checkpoints.

```bash
micromamba run -n retroensemble_askcos_llm python scripts/check_setup.py
```

Run the README's one-target example with `--skip_moa` to test actual baseline predictions without contacting an LLM:

```bash
python batch_run_moa_testset.py \
  --test_csv examples/targets.csv \
  --batch_work_name baseline_one_target \
  --top_k 30 --limit 1 --skip_moa
```

Verify that all four prediction CSVs are nonempty and `outputs/baseline_one_target/merged_candidates.csv` exists. Source checkout/patches and one-target baseline inference passed in the author's clean-clone test. `scripts/check_setup.py` checks file existence and interpreter paths only, not model loading, API access, or accuracy.
