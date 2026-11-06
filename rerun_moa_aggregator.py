#!/usr/bin/env python3
"""Re-run only the MoA aggregator using ORIGINAL unfiltered Agent IDs.

Fully standalone source file (no *_original.py). Prior results can be read,
but historical compacted no-reasoning Agent responses are reconstructed from
original_selected_candidate_ids in each saved Agent output. The prompt is
then rebuilt; reusing old saved prompts could silently reintroduce filtering.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from moa_prompt import FINAL_OUTPUT_TOP_K, build_aggregator_prompts
from tools.llm_batch_selector import (
    append_result, build_batch_fieldnames, build_candidate_id_to_smiles,
    build_filter_summary, call_llm_and_parse_json, classify_error,
    combine_reason_with_filter_summary, init_output_csv,
    is_invalid_candidate_id_marker, load_existing_result_indices,
    make_empty_result_row, normalize_selected_candidate_ids,
    parse_llm_json_with_repair, resolve_top_candidate_ids_to_smiles,
    save_json, strip_reasoning_fields, write_error_traceback,
)
from tools.llm_selector import validate_llm_provider

DEFAULT_AGGREGATOR_MODEL = "gpt-oss-120b"


def _sample_index_and_id(sample_dir: Path) -> tuple[int, str]:
    m = re.fullmatch(r"(\d+)_(.+)", sample_dir.name)
    if not m:
        raise ValueError(f"Unexpected sample directory: {sample_dir}")
    return int(m.group(1)), m.group(2)


def locate_original_source(inputs: list[Path]) -> dict[int, Path]:
    found = {}
    for root in inputs:
        for path in Path(root).rglob("aggregator_input.json"):
            if path.parent.name == "samples":
                continue
            index, _ = _sample_index_and_id(path.parent)
            if index in found:
                raise ValueError(f"Duplicate sample_index {index}: {found[index]} and {path}")
            found[index] = path
    return found


def _raw_agent_ids(decision: dict, top_k: int) -> list:
    keys = ("original_selected_candidate_ids", f"top_{top_k}_candidate_ids",
            "selected_candidate_ids", "top_candidate_ids", "top_20_candidate_ids",
            "top_10_candidate_ids")
    for key in keys:
        if key in decision:
            return normalize_selected_candidate_ids(decision[key])
    raise ValueError("Agent response lacks original selected candidate IDs")


def _original_agent_responses(source_dir: Path, payload: dict) -> list[str]:
    cfg_path = source_dir / "moa_config.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    top_k = int(cfg.get("agent_output_top_k", 20))
    include_reasoning = bool(payload.get("include_reasoning", cfg.get("include_reasoning", True)))
    entries = payload.get("agent_responses", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No saved agent responses in {source_dir}")
    rebuilt = []
    for pos, entry in enumerate(entries, 1):
        if isinstance(entry, str):
            entry = {"response": entry, "response_index": pos}
        if not isinstance(entry, dict):
            raise ValueError(f"Invalid Agent response metadata for {source_dir}")
        agent_index = int(entry.get("agent_index", pos))
        matches = list((source_dir / "agents").glob(f"agent_{agent_index:02d}_*_output.json"))
        if len(matches) != 1:
            raise ValueError(f"Cannot uniquely find saved Agent {agent_index} in {source_dir}")
        saved = json.loads(matches[0].read_text(encoding="utf-8"))
        raw_ids = saved.get("original_selected_candidate_ids")
        if raw_ids is None:
            raw_path = matches[0].with_name(matches[0].name.replace("_output.json", "_raw.txt"))
            if not raw_path.is_file():
                raise ValueError(f"Original Agent IDs missing: {matches[0]}")
            parsed, _ = parse_llm_json_with_repair(raw_path.read_text(encoding="utf-8"), label="agent reuse")
            raw_ids = _raw_agent_ids(parsed, top_k)
        raw_ids = normalize_selected_candidate_ids(raw_ids)
        # The stored response may be compacted; reconstruct it if its IDs do not
        # exactly match the saved model original selection.
        response_raw = str(entry.get("response", ""))
        parsed = None
        if response_raw:
            try:
                parsed, _ = parse_llm_json_with_repair(response_raw, label="old aggregator input")
            except (ValueError, TypeError):
                parsed = None
        if isinstance(parsed, dict):
            try:
                same = _raw_agent_ids(parsed, top_k) == raw_ids
            except ValueError:
                same = False
        else:
            same = False
        if same:
            result = parsed
        else:
            result = {f"top_{top_k}_candidate_ids": raw_ids}
            # Reasoning-mode mismatch should not silently discard genuine
            # explanations if the source is otherwise usable.
            if include_reasoning:
                raise ValueError(
                    f"Reasoning Agent {agent_index} cannot be safely reconstructed "
                    f"from compacted Aggregator input: {source_dir}")
        if not include_reasoning:
            result = strip_reasoning_fields(result)
        rebuilt.append(json.dumps(result, ensure_ascii=False))
    return rebuilt


def rerun_one(
    input_file: Path, output_root: Path, model: str, provider: str,
    prompt_mode: str = "rebuild",
) -> dict:
    input_file = Path(input_file)
    sample_dir = input_file.parent
    sample_index, sample_id = _sample_index_and_id(sample_dir)
    payload = json.loads(input_file.read_text(encoding="utf-8"))
    target = str(payload.get("target", ""))
    row = make_empty_result_row(sample_index, sample_id, target, FINAL_OUTPUT_TOP_K)
    if prompt_mode != "rebuild":
        raise ValueError("saved prompts may contain filtered Agent IDs; use --prompt_mode rebuild")
    try:
        candidates = payload["candidates"]
        id_map = build_candidate_id_to_smiles(candidates)
        agents = _original_agent_responses(sample_dir, payload)
        include_reasoning = bool(payload.get("include_reasoning", True))
        system, user = build_aggregator_prompts(
            target=target, candidates=candidates, agent_responses=agents,
            final_output_top_k=FINAL_OUTPUT_TOP_K,
            include_reasoning=include_reasoning,
        )
        out_dir = Path(output_root) / "samples" / f"{sample_index}_{sample_id}"
        agg_dir = out_dir / "aggregator"
        agg_dir.mkdir(parents=True, exist_ok=True)
        save_json(out_dir / "aggregator_input.json", {
            "target": target, "candidates": candidates,
            "agent_responses": [{"response_index": i, "response": response}
                                for i, response in enumerate(agents, 1)],
            "include_reasoning": include_reasoning,
            "source_aggregator_input_path": str(input_file),
        })
        system_path = agg_dir / "aggregator_system_prompt.txt"
        user_path = agg_dir / "aggregator_user_prompt.txt"
        system_path.write_text(system, encoding="utf-8")
        user_path.write_text(user, encoding="utf-8")
        raw_path = agg_dir / "aggregator_raw.txt"
        raw, decision, repaired, attempts, repaired_path = call_llm_and_parse_json(
            provider=provider, model=model, system_prompt=system, user_prompt=user,
            metadata={"sample_index": sample_index, "id": sample_id,
                      "moa_stage": "aggregator_only_rerun"},
            raw_path=raw_path, label=f"aggregator_only sample={sample_index}",
        )
        raw_ids = None
        for key in (f"top_{FINAL_OUTPUT_TOP_K}_candidate_ids", "selected_candidate_ids", "top_10_candidate_ids"):
            if key in decision:
                raw_ids = decision[key]
                break
        if raw_ids is None:
            raise ValueError("Aggregator response contains no Top-10 candidate IDs")
        ids, smiles, flags = resolve_top_candidate_ids_to_smiles(raw_ids, id_map, FINAL_OUTPUT_TOP_K)
        if not include_reasoning:
            decision = strip_reasoning_fields(decision)
        original_reason = str(decision.get("reason", "")) if include_reasoning else ""
        good = sum(not is_invalid_candidate_id_marker(x) for x in smiles)
        reason = (combine_reason_with_filter_summary(original_reason, build_filter_summary(flags, good))
                  if include_reasoning else "")
        decision.update({
            "rerun_mode": "aggregator_only", "llm_model": model, "llm_provider": provider,
            "include_reasoning": include_reasoning, "prompt_mode": "rebuild",
            "original_top_10_candidate_ids": raw_ids,
            "resolved_top_10_candidate_ids": ids,
            "resolved_top_10_smiles": smiles,
            "removed_candidate_selections": flags, "flagged_candidate_selections": flags,
            "valid_selection_count": good,
            "selection_policy": "preserve_raw_candidate_ID_positions_v1",
            "source_aggregator_input_path": str(input_file),
            "raw_output_path": str(raw_path), "json_repaired": repaired,
            "json_attempts": attempts, "json_repaired_path": str(repaired_path or ""),
        })
        if include_reasoning:
            decision["original_reason"] = original_reason
            decision["reason"] = reason
        save_json(agg_dir / "aggregator_output.json", decision)
        report = out_dir / "result.txt"
        report.write_text("\n".join(
            ["Aggregator-only rerun", f"sample_index={sample_index}, ID={sample_id}"]
            + [f"{i}. {cid} -> {smiles[i-1]}" for i, cid in enumerate(ids, 1)]
            + [f"Flagged: {flags}", f"Reason: {reason}"]
        ), encoding="utf-8")
        row.update(status="OK", reason=reason, result_path=str(report))
        for rank, value in enumerate(smiles, 1):
            row[f"top_{rank}"] = value
        print(f"[AGGREGATOR_RERUN_OK] {sample_index}", flush=True)
    except Exception as exc:
        row.update(status="ERROR", error_type=classify_error(exc),
                   error_message=str(exc)[:2000],
                   result_path=str(write_error_traceback(output_root, sample_index, sample_id)))
        print(f"[AGGREGATOR_RERUN_ERROR] {sample_index}: {exc}", flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source_work_dirs", nargs="+", required=True)
    parser.add_argument("--output_work_dir", required=True)
    parser.add_argument("--llm_output_csv", default="batch_results.csv")
    parser.add_argument("--aggregator_model", default=DEFAULT_AGGREGATOR_MODEL)
    parser.add_argument("--aggregator_provider", choices=["nchc", "cerebras"], required=True)
    parser.add_argument("--prompt_mode", choices=["rebuild"], default="rebuild")
    parser.add_argument("--skip_success_from_csv")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.skip_success_from_csv:
        parser.error("Cannot copy historical filtered CSV; rerun original Agent rankings instead")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite conflict")
    source = locate_original_source([Path(x) for x in args.source_work_dirs])
    selected = [i for i in sorted(source) if i >= args.start and
                (args.limit is None or i < args.start + args.limit)]
    output = Path(args.output_work_dir)
    output.mkdir(parents=True, exist_ok=True)
    batch_csv = Path(args.llm_output_csv)
    if not batch_csv.is_absolute():
        batch_csv = output / batch_csv
    if batch_csv.is_file() and args.overwrite:
        batch_csv.unlink()
    columns = build_batch_fieldnames(FINAL_OUTPUT_TOP_K)
    init_output_csv(batch_csv, columns)
    if len(pd.read_csv(batch_csv)) and not args.resume:
        raise ValueError(f"Nonempty previous output: {batch_csv}")
    existing = load_existing_result_indices(batch_csv) if args.resume else set()
    provider = validate_llm_provider(args.aggregator_provider)
    for index in selected:
        if index in existing:
            continue
        row = rerun_one(source[index], output, args.aggregator_model, provider, args.prompt_mode)
        append_result(batch_csv, row, columns)
    print(f"[AGGREGATOR_RERUN_DONE] {batch_csv}", flush=True)


if __name__ == "__main__":
    main()
