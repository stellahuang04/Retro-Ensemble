#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

MODELS = [
    {
        "name": "augmented_transformer",
        "python": os.environ.get("AT_PYTHON", sys.executable),
        "script": "predict_scripts/augmented_transformer.py",
    },
    {
        "name": "graph2smiles",
        "python": os.environ.get("G2S_PYTHON", sys.executable),
        "script": "predict_scripts/graph2smiles.py",
    },
    {
        "name": "retrosim",
        "python": os.environ.get("RETROSIM_PYTHON", sys.executable),
        "script": "predict_scripts/retrosim.py",
    },
    {
        "name": "template_relevance",
        "python": os.environ.get("TR_PYTHON", sys.executable),
        "script": "predict_scripts/template_relevance.py",
    },
]


def run_command(cmd):
    print("=" * 100)
    print("[CMD]", " ".join(cmd))

    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.stdout:
        print("[STDOUT]")
        print(result.stdout)

    if result.stderr:
        print("[STDERR]")
        print(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {' '.join(cmd)}"
        )

    return result


def run_all_models_batch(
    target_csv: Path,
    output_dir: Path,
    top_k: int,
    use_retroknn: bool = False,
):
    target_csv = Path(target_csv).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    models = list(MODELS)
    if use_retroknn:
        raise ValueError("RetroEnsemble includes AT, G2S, TR and RetroSim only; omit --use_retroknn")

    for model in models:
        script_path = BASE_DIR / model["script"]

        if not script_path.exists():
            raise FileNotFoundError(f"Required baseline script not found: {script_path}")

        output_csv = output_dir / f"{model['name']}.csv"

        cmd = [
            model["python"],
            str(script_path),
            "--input_csv",
            str(target_csv),
            "--output_csv",
            str(output_csv),
            "--top_k",
            str(top_k),
            "--batch_mode",
        ]

        print("=" * 100)
        print(f"[RUN_MODEL_BATCH] {model['name']}")
        run_command(cmd)

    print("=" * 100)
    print("[INFO] All model batch predictions finished.")
