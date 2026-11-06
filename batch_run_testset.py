#!/usr/bin/env python3
"""Single-LLM pipeline aligned with MoA's proposer/agent implementation.

The model call uses batch_run_moa_testset.run_one_agent, so the Single-LLM
and the MoA agent have exactly the same prompt construction, JSON syntax
repair, retry policy, and raw-output behavior. No aggregator is called.

IMPORTANT: apply the shared rank-preserving resolver update before running.
Otherwise the existing shared resolver still removes invalid IDs.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from tools.canonicalize_batch import canonicalize_batch
from tools.llm_batch_selector import (
    append_result,
    build_batch_fieldnames,
    build_candidate_id_to_smiles,
    build_candidates,
    init_output_csv,
    load_candidate_group_map,
    load_existing_result_indices,
    make_empty_result_row,
    resolve_top_candidate_ids_to_smiles,
    save_json,
    write_error_traceback,
)
from tools.llm_selector import DEFAULT_LLM_MODEL, validate_llm_provider
from tools.merge_batch_predictions import merge_batch_predictions
from tools.run_all_models_batch import run_all_models_batch

OUTPUT_ROOT = Path("outputs")
DEFAULT_CANDIDATE_CUTOFF = 20
DEFAULT_OUTPUT_TOP_K = 20


def get_product_from_rxn(rxn_smiles: str) -> str:
    if not isinstance(rxn_smiles, str) or ">" not in rxn_smiles:
        raise ValueError(f"Invalid rxn_smiles: {rxn_smiles!r}")
    product = rxn_smiles.split(">")[-1].strip()
    if not product:
        raise ValueError(f"Empty reaction product: {rxn_smiles!r}")
    return product


def prepare_targets(
    test_csv: Path,
    start: int,
    limit: int | None,
    output_csv: Path,
) -> Path:
    """Preserve absolute sample_index when running a test-set subset."""
    data = pd.read_csv(test_csv)
    required = {"id", "rxn_smiles"}
    if not required.issubset(data.columns):
        raise ValueError(f"test_csv missing {sorted(required - set(data.columns))}")
    if start < 0 or (limit is not None and limit < 1):
        raise ValueError("Require start >= 0 and limit >= 1 when supplied")
    end = len(data) if limit is None else min(start + limit, len(data))
    rows = []
    for index, row in data.iloc[start:end].iterrows():
        sample_id = str(row["id"])
        try:
            target_value = row.get("target")
            target = (
                str(target_value).strip()
                if pd.notna(target_value) and str(target_value).strip()
                else get_product_from_rxn(row["rxn_smiles"])
            )
            rows.append({
                "sample_index": int(index), "id": sample_id,
                "target": target, "rxn_smiles": row["rxn_smiles"],
                "status": "OK", "error_type": "", "error_message": "",
            })
        except Exception as exc:
            rows.append({
                "sample_index": int(index), "id": sample_id,
                "target": "", "rxn_smiles": row.get("rxn_smiles", ""),
                "status": "ERROR", "error_type": "[BAD_INPUT]",
                "error_message": str(exc),
            })
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"[TARGETS] {len(rows)} rows saved to {output_csv}", flush=True)
    return output_csv


def resolve_output_csv(batch_dir: Path, llm_output_csv: str | None) -> Path:
    if not llm_output_csv:
        return batch_dir / "batch_results.csv"
    path = Path(llm_output_csv)
    return path if path.is_absolute() else batch_dir / path


def run_single_for_one_sample(
    sample_index: int,
    sample_id: str,
    target: str,
    target_df: pd.DataFrame,
    batch_dir: Path,
    output_csv: Path,
    candidate_cutoff: int,
    output_top_k: int,
    provider: str,
    model: str,
    include_reasoning: bool,
    fieldnames: list[str],
) -> None:
    # Delayed import avoids a circular import: the MoA module imports
    # prepare_targets and resolve_output_csv from this file.
    from batch_run_moa_testset import run_one_agent

    row = make_empty_result_row(sample_index, sample_id, target, output_top_k)
    sample_dir = batch_dir / "samples" / f"{sample_index}_{sample_id}"
    sample_dir.mkdir(parents=True, exist_ok=True)
    try:
        candidates = build_candidates(target_df, candidate_cutoff)
        if not candidates:
            raise ValueError("No candidates available for this sample")
        id_to_smiles = build_candidate_id_to_smiles(candidates)
        save_json(sample_dir / "llm_candidates.json", candidates)

        # This is the SAME inference entry point used by the MoA proposer.
        agent = run_one_agent(
            agent_index=1,
            provider=provider,
            model=model,
            target=target,
            candidates=candidates,
            candidate_id_to_smiles=id_to_smiles,
            agent_output_top_k=output_top_k,
            include_reasoning=include_reasoning,
            sample_index=sample_index,
            sample_id=sample_id,
            sample_dir=sample_dir,
        )

        # Preserve the raw response and parsed metadata, including all IDs.
        raw_path = Path(agent["raw_path"])
        parsed_path = Path(agent["parsed_path"])
        shutil.copyfile(raw_path, sample_dir / "llm_raw_output.txt")
        decision = json.loads(parsed_path.read_text(encoding="utf-8"))
        raw_ids = decision.get("original_selected_candidate_ids")
        if raw_ids is None:
            raise ValueError("Agent output lacks original_selected_candidate_ids")

        # After the common resolver is updated, this maps invalid IDs into
        # unmistakable non-SMILES markers WITHOUT changing ranking positions.
        top_ids, top_smiles, flagged = resolve_top_candidate_ids_to_smiles(
            top_candidate_ids=raw_ids,
            candidate_id_to_smiles=id_to_smiles,
            output_top_k=output_top_k,
        )
        decision["original_selected_candidate_ids"] = raw_ids
        decision["resolved_selected_candidate_ids"] = top_ids
        decision["resolved_selected_smiles"] = top_smiles
        decision["flagged_candidate_selections"] = flagged
        decision["selection_policy"] = "preserve_raw_id_positions_v1"
        save_json(sample_dir / "llm_output.json", decision)

        reason = str(decision.get("reason", "")) if include_reasoning else ""
        report = sample_dir / "result.txt"
        lines = [
            f"Single-LLM Top-{output_top_k} (same inference as MoA Agent)",
            f"sample_index={sample_index}; id={sample_id}",
            f"provider={provider}; model={model}; reasoning={include_reasoning}",
            "", "Original model selection (one line per original rank):",
        ]
        for rank, (cid, value) in enumerate(zip(top_ids, top_smiles), 1):
            lines.append(f"{rank}. {cid!s} -> {value}")
        lines.extend(["", "Flagged choices:", json.dumps(flagged, ensure_ascii=False, indent=2)])
        if include_reasoning:
            lines.extend(["", "Reason:", reason])
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")

        row["status"] = "OK"
        row["result_path"] = str(report)
        row["reason"] = reason
        # MoA's content-only inference path does not return token usage.
        # Do not fabricate an estimate or overwrite with a false zero.
        row["total_tokens"] = ""
        for rank, value in enumerate(top_smiles, 1):
            row[f"top_{rank}"] = value
        print(f"[SINGLE_OK] sample={sample_index} flagged={len(flagged)}", flush=True)
    except Exception as exc:
        import traceback
        row["status"] = "ERROR"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)[:2000]
        row["result_path"] = str(write_error_traceback(batch_dir, sample_index, sample_id))
        print(f"[SINGLE_ERROR] sample={sample_index}: {exc}", flush=True)
    append_result(output_csv, row, fieldnames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--batch_work_name", required=True)
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--candidate_cutoff", type=int, default=DEFAULT_CANDIDATE_CUTOFF)
    parser.add_argument("--llm_output_top_k", type=int, default=DEFAULT_OUTPUT_TOP_K)
    parser.add_argument("--llm_provider", choices=["cerebras", "nchc"])
    parser.add_argument("--llm_model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_models", action="store_true")
    parser.add_argument("--skip_canonicalize", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--use_retroknn", action="store_true")
    parser.add_argument("--llm_output_csv")
    parser.add_argument("--skip_success_from_csv")
    args = parser.parse_args()

    if args.start < 0 or args.limit is not None and args.limit < 1:
        parser.error("start must be >= 0 and limit must be >= 1")
    if min(args.top_k, args.candidate_cutoff, args.llm_output_top_k) < 1:
        parser.error("all Top-K values must be positive")
    if not args.skip_llm and not args.llm_provider:
        parser.error("--llm_provider is required unless --skip_llm")
    if args.skip_success_from_csv:
        parser.error(
            "--skip_success_from_csv disabled: old batch CSVs can contain "
            "compacted invalid IDs. Start a new experiment with raw rankings."
        )

    batch_dir = OUTPUT_ROOT / args.batch_work_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    target_csv = batch_dir / "targets.csv"
    model_dir = batch_dir / "model_predictions"
    canonical_dir = batch_dir / "canonicalized_predictions"
    merged_csv = batch_dir / "merged_candidates.csv"
    batch_csv = resolve_output_csv(batch_dir, args.llm_output_csv)

    if not (args.skip_models and args.skip_canonicalize and args.skip_merge):
        prepare_targets(Path(args.test_csv), args.start, args.limit, target_csv)
    if not args.skip_models:
        run_all_models_batch(
            target_csv=target_csv,
            output_dir=model_dir,
            top_k=args.top_k,
            use_retroknn=args.use_retroknn,
        )
    if not args.skip_canonicalize:
        canonicalize_batch(input_dir=model_dir, output_dir=canonical_dir)
    if not args.skip_merge:
        merge_batch_predictions(input_dir=canonical_dir, output_csv=merged_csv)
    if args.skip_llm:
        print("[SKIP] Single-LLM selector", flush=True)
        return
    if not merged_csv.exists():
        raise FileNotFoundError(f"Merged candidates not found: {merged_csv}")

    fieldnames = build_batch_fieldnames(args.llm_output_top_k, include_total_tokens=True)
    init_output_csv(batch_csv, fieldnames)
    grouped = load_candidate_group_map(merged_csv)
    prior_indices = load_existing_result_indices(batch_csv) if args.resume else set()
    if not args.resume and len(pd.read_csv(batch_csv)):
        raise ValueError(f"Output CSV already contains rows: {batch_csv}. Use a NEW run name")
    provider = validate_llm_provider(args.llm_provider)
    selected_indices = [
        i for i in sorted(grouped)
        if i >= args.start and (args.limit is None or i < args.start + args.limit)
    ]
    for sample_index in selected_indices:
        if sample_index in prior_indices:
            print(f"[RESUME_SKIP] {sample_index}", flush=True)
            continue
        frame = grouped[sample_index]
        run_single_for_one_sample(
            sample_index=sample_index,
            sample_id=str(frame["id"].iloc[0]),
            target=str(frame["target"].iloc[0]),
            target_df=frame,
            batch_dir=batch_dir,
            output_csv=batch_csv,
            candidate_cutoff=args.candidate_cutoff,
            output_top_k=args.llm_output_top_k,
            provider=provider,
            model=str(args.llm_model).strip(),
            include_reasoning=args.reasoning,
            fieldnames=fieldnames,
        )
    print(f"[SINGLE_DONE] {batch_csv}", flush=True)


if __name__ == "__main__":
    main()
