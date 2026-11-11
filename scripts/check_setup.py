#!/usr/bin/env python3
"""Check configuration without loading weights, downloading tokenizers or calling APIs."""
import argparse
import importlib.util
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--llm-only', action='store_true')
    args = parser.parse_args()
    if importlib.util.find_spec('dotenv'):
        from dotenv import load_dotenv
        load_dotenv(os.environ.get('ASKCOS_ENV_FILE', str(ROOT / '.env')))
    failures = []

    def check(label, valid):
        print(f"{'OK' if valid else 'MISSING'}: {label}")
        if not valid:
            failures.append(label)

    check('selector Python >= 3.10', sys.version_info >= (3, 10))
    if sys.version_info < (3, 10):
        print('Install Python 3.10+ (e.g. a separate Conda environment) before selector requirements.')

    for module in ('pandas', 'rdkit', 'transformers', 'json_repair', 'dotenv', 'requests', 'cerebras'):
        check(f'selector dependency {module}', importlib.util.find_spec(module) is not None)
    if not args.llm_only:
        for prefix, folder in [('AT', 'augmented_transformer'), ('G2S', 'graph2smiles'), ('TR', 'template_relevance'), ('RETROSIM', 'retrosim')]:
            repo = Path(os.environ.get(prefix + '_REPO', str(ROOT / 'external/baselines' / folder))).expanduser()
            check(prefix + ' source directory', repo.is_dir())
            python_var = prefix + '_PYTHON'
            exe = os.environ.get(python_var, sys.executable)
            if not os.environ.get(python_var):
                print(f'NOTE: {python_var} not set; falling back to selector Python: {sys.executable}')
            check(prefix + ' Python executable', shutil.which(exe) is not None)
            resources = {
                'AT': [('AT_CHECKPOINT', repo / 'checkpoints/uspto_50k/model_step_500000.pt')],
                'G2S': [('G2S_CHECKPOINT', repo / 'checkpoints/uspto_50k/model.500000_49.pt')],
                'TR': [('TR_CHECKPOINT', repo / 'checkpoints/grid_search_uspto_50k/ep30_bs512_hid1024x1024_drop0.3_lr0.001/model_latest.pt'), ('TR_TEMPLATES', repo / 'data/uspto_50k/processed/templates.jsonl')],
                'RETROSIM': [('RETROSIM_DATA', repo / 'retrosim/data/data_processed.csv')],
            }[prefix]
            for key, default in resources:
                check(key, Path(os.environ.get(key, str(default))).expanduser().is_file())
            if prefix == 'G2S':
                data = Path(os.environ.get('G2S_DATA_DIR', str(repo / 'data/uspto_50k/processed'))).expanduser()
                check('G2S vocabulary', (data / 'vocab.txt').is_file())
    print('API credentials, model availability, framework compatibility and checkpoint identity are not validated.')
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
