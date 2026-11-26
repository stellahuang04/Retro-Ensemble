# Baseline installation

## Source code

From the RetroEnsemble repository root:

```bash
bash scripts/setup_baseline_sources.sh
```

The script checks out the pinned revisions listed in [baseline_sources.json](../baseline_sources.json) and applies the supplied compatibility patches. It does not overwrite existing source directories.

| Model | Source repository |
| --- | --- |
| Augmented Transformer | [ASKCOS](https://gitlab.com/mlpds_mit/askcosv2/retro/augmented_transformer) |
| Graph2SMILES | [ASKCOS](https://gitlab.com/mlpds_mit/askcosv2/retro/graph2smiles) |
| Template Relevance | [ASKCOS](https://gitlab.com/mlpds_mit/askcosv2/retro/template_relevance) |
| RetroSim | [Original implementation](https://github.com/connorcoley/retrosim) |

The Graph2SMILES patch handles checkpoint loading. The RetroSim patch updates obsolete dictionary and pandas APIs.

## Python environments

Use separate Conda environments for the four models. The YAML files are Linux environment exports; platform-specific packages and GPU dependencies may require adjustment. Use the selector environment from the [main README](../README.md).

```bash
for name in transformer graph2smiles template_relevance retrosim; do
  conda env create -n "retroensemble_${name}" \
    -f "environments/${name}.environment.yml" || break
done
```

Alternatively, use Micromamba with the same environment files. Do not recreate an environment that already exists without checking it first.

## Configure interpreters

Locate the environment paths using `conda env list`. Set the following variables in your local `.env` file to their corresponding absolute Python executable paths.

| Model | Conda environment | Environment variable |
| --- | --- | --- |
| Augmented Transformer | `retroensemble_transformer` | `AT_PYTHON` |
| Graph2SMILES | `retroensemble_graph2smiles` | `G2S_PYTHON` |
| Template Relevance | `retroensemble_template_relevance` | `TR_PYTHON` |
| RetroSim | `retroensemble_retrosim` | `RETROSIM_PYTHON` |

For example:

```bash
conda run -n retroensemble_transformer python -c 'import sys; print(sys.executable)'
```

Existing baseline installations may also be configured using `AT_REPO`, `G2S_REPO`, `TR_REPO` and `RETROSIM_REPO`. See [the environment template](../.env.example).

## Model assets

Follow the download and extraction instructions in the [main README](../README.md). The vocabulary and reaction templates must match the released checkpoints. After configuring all resources, activate the selector environment and run:

```bash
conda activate retroensemble_selector
python scripts/check_setup.py
```

Proceed to the inference commands in the main README. The selector and baseline environments must remain separate. The optional historical selector environment export (`askcos_llm.environment.yml`) is an alternative to installing `requirements.txt`, not an additional installation step.
