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

The supplied YAML files are exports from the author's Linux environments, with the machine-specific `prefix` removed. They are reference snapshots, not minimal cross-platform dependency lists. GPU driver compatibility and a fresh installation have not yet been verified. Some conda and pip packages overlap; resolve installation failures before proceeding.

```bash
for name in askcos_llm transformer graph2smiles template_relevance retrosim; do
    micromamba create -n "retroensemble_${name}" -f "environments/${name}.environment.yml"
done
```

Use distinct new environment names to avoid changing existing environments. All four baseline interpreter mappings were confirmed from the author's batch runner:

| Adapter | Exported environment | .env variable |
| --- | --- | --- |
| AT | transformer | AT_PYTHON |
| G2S | graph2smiles | G2S_PYTHON |
| TR | template_relevance | TR_PYTHON |
| RetroSim | retrosim | RETROSIM_PYTHON |

Run `micromamba env list` to locate the created environments. Set each variable above to its absolute `bin/python` path. Use `retroensemble_askcos_llm` for the main pipeline. Its export contains Cerebras SDK 1.67.0; the original lightweight `requirements.txt` pins 1.59.0. These are alternative environment setups: do not install requirements.txt over the exported environment without intentionally changing versions. Real API compatibility remains to be tested.

## Model assets and validation

Supply the six resource files at the paths in the main README, or set their overrides in `.env`. They are distributed in the baseline-assets-v1 Release. Follow the download, checksum and extraction instructions in the main README. Do not regenerate the vocabulary or templates independently of their checkpoints.

```bash
micromamba run -n retroensemble_askcos_llm python scripts/check_setup.py
```

Then run the README's one-target example. A setup check confirms files and dependencies, not successful model inference. Source clone/patch application against upstream and full GPU/API inference have not yet been executed in this environment.
