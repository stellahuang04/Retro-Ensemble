#!/usr/bin/env python3
"""USPTO-50K Mixture-of-Agents pipeline with ORIGINAL rank positions.

This file is standalone within the `uspto50k-askcos-moa` project: it does
not load or import a previous implementation named *_original.py.
Both fresh and reused agents forward their original candidate IDs to the
aggregator for reasoning and no-reasoning runs. Invalid IDs are NEVER
filtered or replaced with later-ranked candidates.
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from batch_run_testset import OUTPUT_ROOT, prepare_targets, resolve_output_csv
from moa_prompt import FINAL_OUTPUT_TOP_K, build_agent_prompts, build_aggregator_prompts
from tools.canonicalize_batch import canonicalize_batch
from tools.llm_batch_selector import (
    MAX_SAFE_PROMPT_TOKENS, MAX_SAFE_PROMPT_CHARACTERS,
    DEFAULT_JSON_MAX_ATTEMPTS, DEFAULT_JSON_RETRY_SLEEP_SECONDS,
    append_result, build_batch_fieldnames, build_candidate_id_to_smiles,
    build_candidates, build_filter_summary, call_llm_and_parse_json,
    classify_error, combine_reason_with_filter_summary, init_output_csv,
    is_invalid_candidate_id_marker, load_candidate_group_map,
    load_existing_result_indices, make_empty_result_row,
    normalize_selected_candidate_ids, parse_llm_json_with_repair,
    resolve_top_candidate_ids_to_smiles, save_json, strip_reasoning_fields,
    write_error_traceback,
)
from tools.llm_selector import validate_llm_provider
from tools.merge_batch_predictions import merge_batch_predictions
from tools.run_all_models_batch import run_all_models_batch
from tools.token_counter import count_chat_characters, count_chat_tokens

DEFAULT_CANDIDATE_CUTOFF = 20
DEFAULT_AGENT_OUTPUT_TOP_K = 20
DEFAULT_AGENT_MODELS = ["gpt-oss-120b", "gemma-4-31b"]
DEFAULT_AGGREGATOR_MODEL = "gpt-oss-120b"


def check_prompt_size(label: str, system_prompt: str, user_prompt: str) -> None:
    tokens = count_chat_tokens(system_prompt, user_prompt)
    chars = count_chat_characters(system_prompt, user_prompt)
    print(f"[{label}_PROMPT_SIZE] {tokens} tokens; {chars} chars", flush=True)
    if tokens > MAX_SAFE_PROMPT_TOKENS or chars > MAX_SAFE_PROMPT_CHARACTERS:
        raise ValueError(f"{label} prompt too large")


def safe_filename(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value))


def extract_agent_ids(decision: dict, agent_output_top_k: int) -> list:
    """Prefer the model's raw choice list; *never* prioritize compacted lists."""
    keys = (
        "original_selected_candidate_ids",
        f"top_{agent_output_top_k}_candidate_ids",
        "selected_candidate_ids", "top_candidate_ids",
        "top_20_candidate_ids", "top_10_candidate_ids",
    )
    for key in keys:
        if key in decision:
            return normalize_selected_candidate_ids(decision[key])
    raise ValueError(f"No original candidate-ID ranking present: {list(decision)}")


def run_one_agent(
    agent_index: int, provider: str, model: str, target: str,
    candidates: list[dict], candidate_id_to_smiles: dict[str, str],
    agent_output_top_k: int, include_reasoning: bool, sample_index: int,
    sample_id: str, sample_dir: Path,
) -> dict:
    """Also used by Single-LLM, so the selection processes are identical."""
    provider = validate_llm_provider(provider)
    system, user = build_agent_prompts(
        target=target, candidates=candidates,
        agent_output_top_k=agent_output_top_k,
        include_reasoning=include_reasoning,
    )
    check_prompt_size("PROPOSER", system, user)
    root = Path(sample_dir) / "agents"
    stem = f"agent_{agent_index:02d}_{safe_filename(provider)}_{safe_filename(model)}"
    root.mkdir(parents=True, exist_ok=True)
    sys_path, user_path = root / f"{stem}_system_prompt.txt", root / f"{stem}_user_prompt.txt"
    raw_path, parsed_path = root / f"{stem}_raw.txt", root / f"{stem}_output.json"
    sys_path.write_text(system, encoding="utf-8")
    user_path.write_text(user, encoding="utf-8")
    raw, decision, repaired, attempts, repair_path = call_llm_and_parse_json(
        provider=provider, model=model, system_prompt=system, user_prompt=user,
        metadata={"sample_index": sample_index, "id": sample_id,
                  "moa_stage": "proposer", "agent_index": agent_index,
                  "llm_provider": provider, "llm_model": model,
                  "agent_output_top_k": agent_output_top_k,
                  "include_reasoning": bool(include_reasoning)},
        raw_path=raw_path,
        label=f"proposer sample={sample_index} agent={agent_index}",
    )
    raw_ids = extract_agent_ids(decision, agent_output_top_k)
    ids, smiles, flags = resolve_top_candidate_ids_to_smiles(
        raw_ids, candidate_id_to_smiles, agent_output_top_k)
    if not include_reasoning:
        decision = strip_reasoning_fields(decision)
    decision.update({
        "agent_index": agent_index, "llm_provider": provider, "llm_model": model,
        "include_reasoning": bool(include_reasoning),
        "requested_output_top_k": agent_output_top_k,
        "original_selected_candidate_ids": raw_ids,
        "resolved_selected_candidate_ids": ids,
        "resolved_selected_smiles": smiles,
        "removed_candidate_selections": flags,  # historical key; nothing removed
        "flagged_candidate_selections": flags,
        "valid_selection_count": sum(not is_invalid_candidate_id_marker(x) for x in smiles),
        "selection_policy": "preserve_raw_candidate_ID_positions_v1",
        "system_prompt_path": str(sys_path), "user_prompt_path": str(user_path),
        "raw_output_path": str(raw_path), "json_repaired": bool(repaired),
        "json_attempts": int(attempts), "json_repaired_path": str(repair_path or ""),
        "reused_precomputed_agent": False,
    })
    save_json(parsed_path, decision)
    # Forward the *original selected IDs*, not resolved_selected_candidate_ids.
    # JSON serialization changes syntax formatting only, not any selection.
    forwarded = dict(decision)
    for key in ("resolved_selected_candidate_ids", "resolved_selected_smiles",
                "removed_candidate_selections", "flagged_candidate_selections",
                "valid_selection_count", "selection_policy"):
        forwarded.pop(key, None)
    if not include_reasoning:
        forwarded = strip_reasoning_fields(forwarded)
    # Crucial: do not include post-processing metadata beyond the raw model JSON.
    # Make the serialized original decision from the immediate LLM call again.
    parsed_raw, _ = parse_llm_json_with_repair(raw, label="fresh agent response")
    original_payload = strip_reasoning_fields(parsed_raw) if not include_reasoning else parsed_raw
    forwarded_json = json.dumps(original_payload, ensure_ascii=False)
    return {
        "agent_index": agent_index, "provider": provider, "model": model,
        "raw_response": raw, "aggregator_response": forwarded_json,
        "system_prompt_path": str(sys_path), "user_prompt_path": str(user_path),
        "raw_path": str(raw_path), "parsed_path": str(parsed_path),
        "resolved_ids": ids, "reused": False, "json_repaired": repaired,
        "json_attempts": attempts,
    }


def resolve_reuse_source_sample_dir(reuse_agent_work_dir: Path, sample_index: int, sample_id: str) -> Path:
    root = Path(reuse_agent_work_dir)
    pattern = f"{sample_index}_{sample_id}"
    direct = root / "samples" / pattern
    if direct.is_dir():
        return direct
    matches = sorted(x for x in root.glob(f"*/samples/{pattern}") if x.is_dir())
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one reused sample, found {len(matches)}: {root}, {pattern}")
    return matches[0]


def find_matching_precomputed_agent_output(source_sample_dir: Path, provider: str, model: str) -> Path:
    candidates = []
    for path in (source_sample_dir / "agents").glob("*_output.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if (str(data.get("llm_provider", "")).lower() == provider.lower()
                    and str(data.get("llm_model", "")) == model):
                candidates.append(path)
        except (ValueError, OSError):
            continue
    if len(candidates) != 1:
        raise ValueError(f"Expected one original reused agent for {provider}/{model}, found {len(candidates)}")
    return candidates[0]


def load_reused_agent(
    agent_index: int, provider: str, model: str, reuse_agent_work_dir: Path,
    candidates: list[dict], candidate_id_to_smiles: dict[str, str],
    candidate_cutoff: int, agent_output_top_k: int, include_reasoning: bool,
    sample_index: int, sample_id: str, sample_dir: Path,
) -> dict:
    src_sample = resolve_reuse_source_sample_dir(reuse_agent_work_dir, sample_index, sample_id)
    candidate_path = src_sample / "moa_candidates.json"
    if not candidate_path.is_file():
        raise FileNotFoundError(f"Cannot validate reused candidate pool: {candidate_path}")
    old_candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    if old_candidates != candidates:
        raise ValueError(f"Reused candidate pool/order differs: {src_sample}")
    output_path = find_matching_precomputed_agent_output(src_sample, provider, model)
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    if "include_reasoning" in saved and bool(saved["include_reasoning"]) != bool(include_reasoning):
        raise ValueError(f"Reused agent reasoning mode differs from current run: {output_path}")
    if ("requested_output_top_k" in saved
            and int(saved["requested_output_top_k"]) != int(agent_output_top_k)):
        raise ValueError(f"Reused agent output Top-K differs: {output_path}")
    raw_ids = saved.get("original_selected_candidate_ids")
    src_raw = output_path.with_name(output_path.name.replace("_output.json", "_raw.txt"))
    raw_path_field = saved.get("raw_output_path")
    if raw_path_field and Path(raw_path_field).is_file():
        src_raw = Path(raw_path_field)
    parsed_raw = None
    raw_text = None
    if src_raw.is_file():
        raw_text = src_raw.read_text(encoding="utf-8")
        parsed_raw, _ = parse_llm_json_with_repair(raw_text, label="reused agent")
        if not isinstance(parsed_raw, dict):
            raise ValueError(f"Invalid reused agent raw JSON: {src_raw}")
    if raw_ids is None:
        if parsed_raw is None:
            raise ValueError(f"No proof of original reused IDs: {output_path}")
        raw_ids = extract_agent_ids(parsed_raw, agent_output_top_k)
    raw_ids = normalize_selected_candidate_ids(raw_ids)
    if parsed_raw is not None:
        from_raw = extract_agent_ids(parsed_raw, agent_output_top_k)
        if from_raw != raw_ids:
            # Historical no-reasoning reused raw payload may be compacted.
            # Always prefer original_selected_candidate_ids over filtered IDs.
            if include_reasoning:
                raise ValueError(f"Reasoning raw ID mismatch: {output_path}")
            parsed_raw = None
    # A source saved JSON may include filtered results from a legacy run.
    ids, smiles, flags = resolve_top_candidate_ids_to_smiles(raw_ids, candidate_id_to_smiles, agent_output_top_k)
    stem = f"agent_{agent_index:02d}_{safe_filename(provider)}_{safe_filename(model)}_reused"
    out_dir = Path(sample_dir) / "agents"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_raw = out_dir / f"{stem}_raw.txt"
    out_json = out_dir / f"{stem}_output.json"
    # Rebuild from the saved original choice list, preserving reasoning when safe.
    forwarded = dict(parsed_raw) if parsed_raw is not None else {f"top_{agent_output_top_k}_candidate_ids": raw_ids}
    if not include_reasoning:
        forwarded = strip_reasoning_fields(forwarded)
    forwarded_text = json.dumps(forwarded, ensure_ascii=False)
    out_raw.write_text(forwarded_text, encoding="utf-8")
    saved.update({
        "agent_index": agent_index, "original_selected_candidate_ids": raw_ids,
        "resolved_selected_candidate_ids": ids, "resolved_selected_smiles": smiles,
        "removed_candidate_selections": flags,
        "flagged_candidate_selections": flags,
        "valid_selection_count": sum(not is_invalid_candidate_id_marker(x) for x in smiles),
        "selection_policy": "preserve_raw_candidate_ID_positions_v1",
        "reused_precomputed_agent": True,
        "reused_from_output_path": str(output_path), "raw_output_path": str(out_raw),
    })
    save_json(out_json, saved)
    return {
        "agent_index": agent_index, "provider": provider, "model": model,
        "raw_response": forwarded_text, "aggregator_response": forwarded_text,
        "parsed_path": str(out_json), "raw_path": str(out_raw),
        "resolved_ids": ids, "reused": True,
        "json_repaired": False, "json_attempts": 0,
    }


def run_moa_for_one_sample(
    sample_index: int, sample_id: str, target: str, target_df: pd.DataFrame,
    work_dir: Path, output_csv: Path, candidate_cutoff: int,
    agent_output_top_k: int, agent_providers: list[str], agent_models: list[str],
    aggregator_provider: str, aggregator_model: str, agent_workers: int,
    include_reasoning: bool, reuse_agent_index: int | None = None,
    reuse_agent_work_dir: Path | None = None,
) -> dict:
    row = make_empty_result_row(sample_index, sample_id, target, FINAL_OUTPUT_TOP_K)
    sample_dir = Path(work_dir) / "samples" / f"{sample_index}_{sample_id}"
    try:
        candidates = build_candidates(target_df, candidate_cutoff)
        if not candidates:
            raise ValueError("No candidates")
        id_map = build_candidate_id_to_smiles(candidates)
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_json(sample_dir / "moa_candidates.json", candidates)
        save_json(sample_dir / "moa_config.json", {
            "candidate_cutoff": candidate_cutoff,
            "agent_output_top_k": agent_output_top_k,
            "agent_models": agent_models, "agent_providers": agent_providers,
            "aggregator_model": aggregator_model,
            "aggregator_provider": aggregator_provider,
            "final_output_top_k": FINAL_OUTPUT_TOP_K,
            "include_reasoning": bool(include_reasoning),
            "reuse_agent_index": reuse_agent_index,
            "reuse_agent_work_dir": str(reuse_agent_work_dir or ""),
        })
        specs = list(enumerate(zip(agent_providers, agent_models), start=1))
        results = []
        with ThreadPoolExecutor(max_workers=max(1, int(agent_workers))) as pool:
            futures = {}
            for index, (provider, model) in specs:
                if reuse_agent_index == index:
                    if reuse_agent_work_dir is None:
                        raise ValueError("Missing reuse_agent_work_dir")
                    results.append(load_reused_agent(
                        index, provider, model, reuse_agent_work_dir, candidates, id_map,
                        candidate_cutoff, agent_output_top_k, include_reasoning,
                        sample_index, sample_id, sample_dir,
                    ))
                else:
                    future = pool.submit(
                        run_one_agent, index, provider, model, target, candidates,
                        id_map, agent_output_top_k, include_reasoning,
                        sample_index, sample_id, sample_dir,
                    )
                    futures[future] = index
            for future in as_completed(futures):
                results.append(future.result())
        results.sort(key=lambda result: result["agent_index"])
        original_responses = [result["aggregator_response"] for result in results]
        save_json(sample_dir / "aggregator_input.json", {
            "target": target, "candidates": candidates,
            "agent_responses": [
                {"response_index": pos, "response": data,
                 "agent_index": results[pos-1]["agent_index"],
                 "provider": results[pos-1]["provider"],
                 "model": results[pos-1]["model"],
                 "reused": bool(results[pos-1]["reused"])}
                for pos, data in enumerate(original_responses, 1)
            ], "include_reasoning": bool(include_reasoning),
        })
        system, user = build_aggregator_prompts(
            target=target, candidates=candidates,
            agent_responses=original_responses,
            final_output_top_k=FINAL_OUTPUT_TOP_K,
            include_reasoning=include_reasoning,
        )
        check_prompt_size("AGGREGATOR", system, user)
        agg_dir = sample_dir / "aggregator"
        agg_dir.mkdir(parents=True, exist_ok=True)
        sys_path = agg_dir / "aggregator_system_prompt.txt"
        user_path = agg_dir / "aggregator_user_prompt.txt"
        raw_path = agg_dir / "aggregator_raw.txt"
        parsed_path = agg_dir / "aggregator_output.json"
        sys_path.write_text(system, encoding="utf-8")
        user_path.write_text(user, encoding="utf-8")
        raw, decision, repaired, attempts, repaired_path = call_llm_and_parse_json(
            provider=aggregator_provider, model=aggregator_model,
            system_prompt=system, user_prompt=user,
            metadata={"sample_index": sample_index, "id": sample_id,
                      "moa_stage": "aggregator", "include_reasoning": bool(include_reasoning)},
            raw_path=raw_path, label=f"aggregator sample={sample_index}",
        )
        raw_ids = extract_agent_ids(decision, FINAL_OUTPUT_TOP_K)
        ids, smiles, flags = resolve_top_candidate_ids_to_smiles(raw_ids, id_map, FINAL_OUTPUT_TOP_K)
        if not include_reasoning:
            decision = strip_reasoning_fields(decision)
        original_reason = str(decision.get("reason", "")) if include_reasoning else ""
        good = sum(not is_invalid_candidate_id_marker(x) for x in smiles)
        summary = build_filter_summary(flags, good)
        final_reason = combine_reason_with_filter_summary(original_reason, summary) if include_reasoning else ""
        decision.update({
            "llm_provider": aggregator_provider, "llm_model": aggregator_model,
            "include_reasoning": include_reasoning,
            "agent_output_top_k": agent_output_top_k,
            "agent_providers": agent_providers, "agent_models": agent_models,
            "requested_output_top_k": FINAL_OUTPUT_TOP_K,
            "original_top_10_candidate_ids": raw_ids,
            "resolved_top_10_candidate_ids": ids,
            "resolved_top_10_smiles": smiles,
            "removed_candidate_selections": flags,
            "flagged_candidate_selections": flags,
            "valid_selection_count": good,
            "selection_policy": "preserve_raw_candidate_ID_positions_v1",
            "system_prompt_path": str(sys_path), "user_prompt_path": str(user_path),
            "raw_output_path": str(raw_path), "json_repaired": bool(repaired),
            "json_attempts": attempts, "json_repaired_path": str(repaired_path or ""),
            "agent_output_paths": [x["parsed_path"] for x in results],
        })
        if include_reasoning:
            decision["original_reason"] = original_reason
            decision["reason"] = final_reason
        save_json(parsed_path, decision)
        report = sample_dir / "result.txt"
        lines = [f"MoA Top-{FINAL_OUTPUT_TOP_K}", f"Target: {target}", "Original aggregator ranking:"]
        lines += [f"{rank}. {cid} -> {smiles[rank-1]}" for rank, cid in enumerate(ids, 1)]
        lines.append(f"Flagged selections: {json.dumps(flags, ensure_ascii=False)}")
        if include_reasoning:
            lines.append(f"Reason: {final_reason}")
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        row.update(status="OK", reason=final_reason,
                   result_path=str(report))
        for rank, smiles_string in enumerate(smiles, 1):
            row[f"top_{rank}"] = smiles_string
        print(f"[MOA_OK] sample={sample_index} raw_slots={len(smiles)} flagged={len(flags)}", flush=True)
    except Exception as error:
        row.update(status="ERROR", error_type=classify_error(error),
                   error_message=str(error)[:2000],
                   result_path=str(write_error_traceback(work_dir, sample_index, sample_id)))
        print(f"[MOA_ERROR] sample={sample_index} {error}", flush=True)
    append_result(output_csv, row, build_batch_fieldnames(FINAL_OUTPUT_TOP_K))
    return row


def run_moa_batch(
    merged_csv: Path, output_csv: Path, work_dir: Path,
    candidate_cutoff: int = DEFAULT_CANDIDATE_CUTOFF,
    agent_output_top_k: int = DEFAULT_AGENT_OUTPUT_TOP_K,
    agent_providers: list[str] | None = None,
    agent_models: list[str] | None = None,
    aggregator_provider: str | None = None,
    aggregator_model: str = DEFAULT_AGGREGATOR_MODEL,
    agent_workers: int = 2, include_reasoning: bool = True,
    resume: bool = False, skip_success_from_csv: Path | None = None,
    start: int = 0, limit: int | None = None,
    reuse_agent_index: int | None = None,
    reuse_agent_work_dir: Path | None = None,
) -> None:
    if skip_success_from_csv:
        raise ValueError("Historical CSV reuse disabled: old results might have compacted invalid IDs")
    if len(agent_providers or []) != len(agent_models or []):
        raise ValueError("One provider per agent model is required")
    if not aggregator_provider or not agent_providers:
        raise ValueError("Agent and Aggregator providers must be specified")
    columns = build_batch_fieldnames(FINAL_OUTPUT_TOP_K)
    init_output_csv(output_csv, columns)
    groups = load_candidate_group_map(merged_csv)
    seen = load_existing_result_indices(output_csv) if resume else set()
    if not resume and len(pd.read_csv(output_csv)):
        raise ValueError("Output CSV already contains data; use a new batch_work_name")
    selected = [i for i in sorted(groups) if i >= start and (limit is None or i < start + limit)]
    for sample_index in selected:
        if sample_index in seen:
            continue
        group = groups[sample_index]
        run_moa_for_one_sample(
            sample_index, str(group["id"].iloc[0]), str(group["target"].iloc[0]),
            group, work_dir, output_csv, candidate_cutoff, agent_output_top_k,
            agent_providers, agent_models, aggregator_provider, aggregator_model,
            agent_workers, include_reasoning, reuse_agent_index, reuse_agent_work_dir,
        )
    print(f"[MOA_BATCH_DONE] {output_csv}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_csv")
    parser.add_argument("--existing_merged_csv")
    parser.add_argument("--batch_work_name", required=True)
    parser.add_argument("--top_k", type=int, default=300)
    parser.add_argument("--candidate_cutoff", type=int, default=DEFAULT_CANDIDATE_CUTOFF)
    parser.add_argument("--agent_output_top_k", type=int, default=DEFAULT_AGENT_OUTPUT_TOP_K)
    parser.add_argument("--agent_models", nargs="+", default=DEFAULT_AGENT_MODELS)
    parser.add_argument("--agent_providers", nargs="+", choices=["cerebras", "nchc"])
    parser.add_argument("--aggregator_model", default=DEFAULT_AGGREGATOR_MODEL)
    parser.add_argument("--aggregator_provider", choices=["cerebras", "nchc"])
    parser.add_argument("--agent_workers", type=int, default=2)
    parser.add_argument("--reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse_agent_index", type=int)
    parser.add_argument("--reuse_agent_work_dir")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip_models", action="store_true")
    parser.add_argument("--skip_canonicalize", action="store_true")
    parser.add_argument("--skip_merge", action="store_true")
    parser.add_argument("--skip_moa", action="store_true")
    parser.add_argument("--use_retroknn", action="store_true")
    parser.add_argument("--llm_output_csv")
    parser.add_argument("--skip_success_from_csv")
    args = parser.parse_args()
    if not args.existing_merged_csv and not args.test_csv:
        parser.error("--test_csv is required without --existing_merged_csv")
    if not args.skip_moa and (not args.agent_providers or not args.aggregator_provider):
        parser.error("--agent_providers and --aggregator_provider are required")
    if (args.reuse_agent_index is None) != (args.reuse_agent_work_dir is None):
        parser.error("Specify both --reuse_agent_index and --reuse_agent_work_dir")
    batch_dir = OUTPUT_ROOT / args.batch_work_name
    batch_dir.mkdir(parents=True, exist_ok=True)
    if args.existing_merged_csv:
        merged_csv = Path(args.existing_merged_csv)
    else:
        merged_csv = batch_dir / "merged_candidates.csv"
        targets = batch_dir / "targets.csv"
        model_dir = batch_dir / "model_predictions"
        canon_dir = batch_dir / "canonicalized_predictions"
        prepare_targets(Path(args.test_csv), args.start, args.limit, targets)
        if not args.skip_models:
            run_all_models_batch(target_csv=targets, output_dir=model_dir,
                                 top_k=args.top_k, use_retroknn=args.use_retroknn)
        if not args.skip_canonicalize:
            canonicalize_batch(input_dir=model_dir, output_dir=canon_dir)
        if not args.skip_merge:
            merge_batch_predictions(input_dir=canon_dir, output_csv=merged_csv)
    if args.skip_moa:
        print("[SKIP] MoA inference", flush=True)
        return
    if not merged_csv.is_file():
        raise FileNotFoundError(merged_csv)
    output_csv = resolve_output_csv(batch_dir, args.llm_output_csv)
    run_moa_batch(
        merged_csv=merged_csv, output_csv=output_csv, work_dir=batch_dir,
        candidate_cutoff=args.candidate_cutoff,
        agent_output_top_k=args.agent_output_top_k,
        agent_providers=args.agent_providers, agent_models=args.agent_models,
        aggregator_provider=args.aggregator_provider, aggregator_model=args.aggregator_model,
        agent_workers=args.agent_workers, include_reasoning=args.reasoning,
        resume=args.resume, skip_success_from_csv=args.skip_success_from_csv,
        start=args.start, limit=args.limit,
        reuse_agent_index=args.reuse_agent_index,
        reuse_agent_work_dir=(Path(args.reuse_agent_work_dir) if args.reuse_agent_work_dir else None),
    )


if __name__ == "__main__":
    main()
