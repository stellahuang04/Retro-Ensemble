#!/usr/bin/env python3
"""Single-LLM selection helpers shared with MoA; preserves raw candidate-ID ranks.

Self-contained replacement for tools/llm_batch_selector.py on
`uspto50k-askcos-moa`. Does NOT import *_original.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from json_repair import repair_json

from tools.cerebras_client import TokenPerMinuteLimitError
from tools.llm_selector import (
    DEFAULT_LLM_MODEL,
    estimate_total_prompt_characters,
    estimate_total_prompt_tokens,
    get_llm_client,
    validate_llm_provider,
)
from prompt import get_system_prompt, build_user_prompt

MAX_SAFE_PROMPT_TOKENS = 100000
MAX_SAFE_PROMPT_CHARACTERS = 1000000
DEFAULT_CANDIDATE_CUTOFF = 20
DEFAULT_OUTPUT_TOP_K = 20
DEFAULT_JSON_MAX_ATTEMPTS = 3
DEFAULT_JSON_RETRY_SLEEP_SECONDS = 1.0
INVALID_CANDIDATE_ID_PREFIX = "__INVALID_CANDIDATE_ID__:"
REASONING_KEYS = {"reason", "summary", "analysis", "explanation", "decision_trace"}


def validate_output_top_k(output_top_k: int) -> int:
    result = int(output_top_k)
    if result < 1:
        raise ValueError("output_top_k must be >= 1")
    return result


def build_batch_fieldnames(output_top_k: int, include_total_tokens: bool = False) -> list[str]:
    fields = ["sample_index", "id", "target", "status", "error_type", "error_message"]
    fields += [f"top_{i}" for i in range(1, validate_output_top_k(output_top_k) + 1)]
    if include_total_tokens:
        fields.append("total_tokens")
    fields += ["reason", "result_path"]
    return fields


def is_invalid_candidate_id_marker(value: Any) -> bool:
    return isinstance(value, str) and value.startswith(INVALID_CANDIDATE_ID_PREFIX)


def safe_json_load(text: str, *, verbose: bool = True) -> Any:
    """Fix JSON *syntax* only. Never rewrite candidate IDs or model rankings."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("LLM output is not a nonempty text string")
    raw = text.strip()
    for payload in (raw, re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I).strip()):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            pass
        first, last = payload.find("{"), payload.rfind("}")
        if first >= 0 and last > first:
            try:
                return json.loads(payload[first:last + 1])
            except json.JSONDecodeError:
                pass
    repaired = repair_json(raw)
    result = json.loads(repaired)
    if verbose:
        print("[JSON_SYNTAX_REPAIRED]", flush=True)
    return result


def parse_llm_json_with_repair(text: str, *, label: str = "LLM", verbose: bool = True):
    """Return parsed object and whether JSON syntax was repaired."""
    try:
        return json.loads(text), False
    except (json.JSONDecodeError, TypeError):
        return safe_json_load(text, verbose=verbose), True


def call_llm_and_parse_json(
    *, provider: str, model: str, system_prompt: str, user_prompt: str,
    metadata: dict, raw_path: Path, label: str,
    max_attempts: int = DEFAULT_JSON_MAX_ATTEMPTS,
    retry_sleep_seconds: float = DEFAULT_JSON_RETRY_SLEEP_SECONDS,
) -> tuple[str, dict, bool, int, Path | None]:
    """The common inference function used by Single-LLM and both MoA agents.

    Re-generates only after unrepairable JSON. An unknown candidate ID is still
    a valid *JSON* output and MUST NOT cause another stochastic generation.
    """
    raw_path = Path(raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    provider = validate_llm_provider(provider)
    client = get_llm_client(provider, model)
    last_error = None
    for attempt in range(1, int(max_attempts) + 1):
        settings = dict(metadata or {})
        settings.update(json_attempt=attempt, json_max_attempts=max_attempts)
        raw = client.complete_content_only(
            prompt=user_prompt,
            system_prompt=system_prompt,
            response_format=None,
            metadata=settings,
        )
        raw_path.with_name(f"{raw_path.stem}_attempt_{attempt:02d}{raw_path.suffix}").write_text(
            raw, encoding="utf-8"
        )
        raw_path.write_text(raw, encoding="utf-8")
        try:
            parsed, repaired = parse_llm_json_with_repair(raw, label=label)
            if not isinstance(parsed, dict):
                raise ValueError("LLM JSON must be an object")
            repaired_path = None
            if repaired:
                repaired_path = raw_path.with_name(f"{raw_path.stem}_repaired.json")
                save_json(repaired_path, parsed)
            return raw, parsed, repaired, attempt, repaired_path
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
            print(f"[JSON_RETRY] {label} attempt={attempt}/{max_attempts}: {exc}", flush=True)
            if attempt < int(max_attempts):
                time.sleep(float(retry_sleep_seconds))
    raise ValueError(f"{label}: JSON failed after {max_attempts} attempts: {last_error}")


def strip_reasoning_fields(decision: dict) -> dict:
    return {key: value for key, value in decision.items() if key not in REASONING_KEYS}


def classify_error(error: Exception) -> str:
    msg = str(error).lower()
    if isinstance(error, TokenPerMinuteLimitError) or "tokens_per_minute" in msg:
        return "[LLM_TOKEN_PER_MINUTE_LIMIT]"
    if "429" in msg or "rate_limit" in msg or "rate limit" in msg:
        return "[LLM_RATE_LIMIT]"
    if "json" in msg:
        return "[PARSE_ERROR]"
    return f"[{type(error).__name__.upper()}]"


def init_output_csv(path: Path, fieldnames: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        header = list(pd.read_csv(path, nrows=0).columns)
        if header != fieldnames:
            raise ValueError(f"CSV header differs from requested output fields: {path}")
    else:
        with path.open("w", newline="", encoding="utf-8") as fp:
            csv.DictWriter(fp, fieldnames=fieldnames).writeheader()


def append_result(path: Path, row: dict, fieldnames: list[str]) -> None:
    with Path(path).open("a", newline="", encoding="utf-8") as fp:
        csv.DictWriter(fp, fieldnames=fieldnames).writerow({k: row.get(k, "") for k in fieldnames})


def normalize_result_row(row: pd.Series | dict, fieldnames: list[str]) -> dict:
    result = {}
    for field in fieldnames:
        value = row.get(field, "")
        result[field] = "" if pd.isna(value) else value
    return result


def load_existing_result_indices(path: Path) -> set[int]:
    path = Path(path)
    if not path.is_file():
        return set()
    df = pd.read_csv(path)
    return set(pd.to_numeric(df.get("sample_index", pd.Series(dtype=float)), errors="coerce").dropna().astype(int))


def load_previous_rows_in_order(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "sample_index" not in frame:
        raise ValueError("Previous CSV missing sample_index")
    frame["sample_index"] = pd.to_numeric(frame["sample_index"], errors="raise").astype(int)
    return (frame.sort_values("sample_index").drop_duplicates("sample_index", keep="last")
            .sort_values("sample_index").reset_index(drop=True))


def load_candidate_group_map(merged_csv: Path) -> dict[int, pd.DataFrame]:
    frame = pd.read_csv(merged_csv, low_memory=False)
    required = {"sample_index", "id", "target", "candidate_id", "model", "rank", "score", "canonical_candidate"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in merged_candidates.csv: {sorted(missing)}")
    frame["sample_index"] = pd.to_numeric(frame["sample_index"], errors="raise").astype(int)
    frame["candidate_id"] = frame["candidate_id"].astype(str)
    frame["id"] = frame["id"].astype(str)
    return {int(i): group.copy() for i, group in frame.groupby("sample_index", sort=True)}


def make_empty_result_row(sample_index, sample_id, target, output_top_k: int) -> dict:
    row = {"sample_index": sample_index, "id": sample_id, "target": target,
           "status": "", "error_type": "", "error_message": "", "total_tokens": "",
           "reason": "", "result_path": ""}
    row.update({f"top_{k}": "" for k in range(1, validate_output_top_k(output_top_k) + 1)})
    return row


def build_candidates(target_df: pd.DataFrame, candidate_cutoff: int) -> list[dict]:
    """Exactly the original MoA/Single candidate sorting and per-model cutoff."""
    cutoff = int(candidate_cutoff)
    if cutoff < 1:
        raise ValueError("candidate_cutoff must be >= 1")
    frame = target_df.copy()
    frame["model"] = frame["model"].astype(str).str.strip()
    frame["_numeric_rank"] = pd.to_numeric(frame["rank"], errors="coerce")
    frame["_original_order"] = range(len(frame))
    bad = frame["_numeric_rank"].isna() | (frame["_numeric_rank"] < 1)
    if bad.any():
        raise ValueError(f"Invalid source rank in {int(bad.sum())} candidates")
    frame = frame.sort_values(["model", "_numeric_rank", "_original_order"], kind="stable")
    frame = frame.groupby("model", sort=False, group_keys=False).head(cutoff).copy()
    result = []
    for _, row in frame.iterrows():
        candidate_id = str(row["candidate_id"]).strip()
        model = str(row["model"]).strip()
        smiles = str(row["canonical_candidate"]).strip()
        if not candidate_id or candidate_id.lower() == "nan" or not smiles or smiles.lower() == "nan":
            raise ValueError("Candidate has empty ID or canonical SMILES")
        score = row.get("score", "")
        result.append({
            "candidate_id": candidate_id,
            "model": model,
            "rank": int(row["_numeric_rank"]),
            "score": "" if pd.isna(score) else str(score),
            "canonical_precursors": smiles,
        })
    return result


def build_candidate_id_to_smiles(candidates: list[dict]) -> dict[str, str]:
    result = {}
    for item in candidates:
        candidate_id = str(item["candidate_id"]).strip()
        smiles = str(item["canonical_precursors"]).strip()
        if not candidate_id or not smiles:
            raise ValueError("Missing candidate ID or SMILES")
        if candidate_id in result and result[candidate_id] != smiles:
            raise ValueError(f"Inconsistent candidate mapping for {candidate_id}")
        result[candidate_id] = smiles
    return result


def normalize_selected_candidate_ids(value) -> list:
    return [] if value is None else (value if isinstance(value, list) else [value])


def resolve_top_candidate_ids_to_smiles(
    top_candidate_ids, candidate_id_to_smiles: dict[str, str], output_top_k: int,
) -> tuple[list[str], list[str], list[dict]]:
    """Never compact, refit, or re-rank. Invalid candidate IDs are marked wrong."""
    original = normalize_selected_candidate_ids(top_candidate_ids)
    preserved_ids, precursor_smiles, flags = [], [], []
    seen_ids, seen_smiles = set(), set()
    for position, value in enumerate(original[:validate_output_top_k(output_top_k)], start=1):
        original_id = "" if value is None else str(value)
        candidate_id = original_id.strip()
        preserved_ids.append(original_id)
        if not candidate_id or candidate_id.lower() in {"nan", "none", "null"}:
            reason = "empty_candidate_id"
        elif candidate_id not in candidate_id_to_smiles:
            reason = "candidate_id_not_in_input_pool"
        else:
            reason = None
        if reason:
            flags.append({"original_position": position, "candidate_id": original_id, "reason": reason})
            precursor_smiles.append(INVALID_CANDIDATE_ID_PREFIX + (original_id or "<EMPTY>"))
            continue
        smiles = candidate_id_to_smiles[candidate_id]
        if candidate_id in seen_ids:
            flags.append({"original_position": position, "candidate_id": original_id,
                          "reason": "duplicate_candidate_id"})
        elif smiles in seen_smiles:
            flags.append({"original_position": position, "candidate_id": original_id,
                          "reason": "duplicate_canonical_precursors"})
        seen_ids.add(candidate_id)
        seen_smiles.add(smiles)
        precursor_smiles.append(smiles)
    return preserved_ids, precursor_smiles, flags


def build_filter_summary(removed_selections: list[dict], valid_count: int) -> str:
    if not removed_selections:
        return ""
    counts = {}
    for item in removed_selections:
        counts[item["reason"]] = counts.get(item["reason"], 0) + 1
    return (f"Flagged {len(removed_selections)} selections ({counts}); all original rank "
            f"positions retained; {valid_count} IDs appeared in the candidate pool.")


def combine_reason_with_filter_summary(original_reason: str, filter_summary: str) -> str:
    reason = str(original_reason or "").strip()
    return reason + ("\n\n[Candidate ID validation; positions preserved] " + filter_summary
                     if filter_summary else "")


def save_json(path: Path, data) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def save_result_to_txt(
    output_path: Path, target: str, selected_smiles: list, decision_trace: list,
    reason: str, selected_candidate_ids: list | None = None,
    removed_selections: list[dict] | None = None, llm_provider: str | None = None,
    llm_model: str | None = None, candidate_cutoff: int | None = None,
    output_top_k: int = DEFAULT_OUTPUT_TOP_K, include_reasoning: bool = True,
) -> None:
    lines = [f"=== Retrosynthesis Top-{output_top_k} ===", f"Target: {target}",
             f"Provider/model: {llm_provider}/{llm_model}", "Original-rank selections:"]
    for rank, (candidate_id, smiles) in enumerate(zip(selected_candidate_ids or [], selected_smiles), 1):
        lines.append(f"{rank}. {candidate_id} -> {smiles}")
    lines.append("Flagged selections: " + json.dumps(removed_selections or [], ensure_ascii=False))
    if include_reasoning:
        lines.extend(["Decision trace: " + json.dumps(decision_trace, ensure_ascii=False), "Reason: " + reason])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_error_traceback(work_dir: Path, sample_index: int, sample_id: str) -> Path:
    path = Path(work_dir) / "errors" / f"{sample_index}_{sample_id}.traceback.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(traceback.format_exc(), encoding="utf-8")
    return path


def get_selected_id_key(decision: dict, output_top_k: int) -> str:
    for key in (f"top_{output_top_k}_candidate_ids", "original_selected_candidate_ids",
                "selected_candidate_ids", "top_candidate_ids", "top_20_candidate_ids",
                "top_10_candidate_ids"):
        if key in decision:
            return key
    return f"top_{output_top_k}_candidate_ids"


def run_llm_for_one_sample(
    sample_index: int, sample_id: str, target: str, target_df: pd.DataFrame,
    work_dir: Path, output_csv: Path, candidate_cutoff: int, output_top_k: int,
    llm_provider: str, llm_model: str, include_reasoning: bool,
    fieldnames: list[str], skip_success_from_csv: Path | None = None,
) -> dict:
    result = make_empty_result_row(sample_index, sample_id, target, output_top_k)
    sample_dir = Path(work_dir) / "samples" / f"{sample_index}_{sample_id}"
    try:
        candidates = build_candidates(target_df, candidate_cutoff)
        if not candidates:
            raise ValueError("No candidates available")
        id_map = build_candidate_id_to_smiles(candidates)
        system_prompt = get_system_prompt(candidates, output_top_k=output_top_k)
        user_prompt = build_user_prompt(target=target, candidates=candidates,
                                        output_top_k=output_top_k, include_reasoning=include_reasoning)
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_json(sample_dir / "llm_candidates.json", candidates)
        (sample_dir / "llm_system_prompt.txt").write_text(system_prompt, encoding="utf-8")
        (sample_dir / "llm_user_prompt.txt").write_text(user_prompt, encoding="utf-8")
        raw, decision, repaired, attempts, repaired_path = call_llm_and_parse_json(
            provider=llm_provider, model=llm_model,
            system_prompt=system_prompt, user_prompt=user_prompt,
            metadata={"sample_index": sample_index, "id": sample_id, "include_reasoning": include_reasoning},
            raw_path=sample_dir / "llm_raw_output.txt", label=f"single sample={sample_index}",
        )
        key = get_selected_id_key(decision, output_top_k)
        raw_ids = decision.get(key, [])
        ids, smiles, flags = resolve_top_candidate_ids_to_smiles(raw_ids, id_map, output_top_k)
        if not include_reasoning:
            decision = strip_reasoning_fields(decision)
        reason = str(decision.get("reason", "")) if include_reasoning else ""
        good = sum(not is_invalid_candidate_id_marker(sm) for sm in smiles)
        summary = build_filter_summary(flags, good)
        final_reason = combine_reason_with_filter_summary(reason, summary) if include_reasoning else ""
        decision.update({
            "llm_provider": llm_provider, "llm_model": llm_model,
            "include_reasoning": include_reasoning,
            "original_selected_candidate_ids": raw_ids,
            "resolved_selected_candidate_ids": ids,
            "resolved_selected_smiles": smiles,
            "removed_candidate_selections": flags,  # legacy field name; nothing removed
            "flagged_candidate_selections": flags,
            "valid_selection_count": good,
            "selection_policy": "preserve_raw_candidate_ID_positions_v1",
            "json_repaired": repaired, "json_attempts": attempts,
            "json_repaired_path": str(repaired_path or ""),
        })
        if include_reasoning:
            decision["original_reason"] = reason
            decision["reason"] = final_reason
        save_json(sample_dir / "llm_output.json", decision)
        report = sample_dir / "result.txt"
        save_result_to_txt(report, target, smiles, decision.get("decision_trace", []),
                           final_reason, ids, flags, llm_provider, llm_model,
                           candidate_cutoff, output_top_k, include_reasoning)
        result.update(status="OK", reason=final_reason, result_path=str(report))
        for rank, item in enumerate(smiles, 1):
            result[f"top_{rank}"] = item
        print(f"[SINGLE_OK] sample={sample_index}; raw_slots={len(smiles)}; invalid={len(flags)}", flush=True)
    except Exception as error:
        result.update(status="ERROR", error_type=classify_error(error),
                      error_message=str(error)[:2000],
                      result_path=str(write_error_traceback(work_dir, sample_index, sample_id)))
        print(f"[SINGLE_ERROR] sample={sample_index}: {error}", flush=True)
    append_result(output_csv, result, fieldnames)
    return result


def run_llm_batch_selector(
    merged_csv: Path, output_csv: Path, work_dir: Path,
    candidate_cutoff: int = DEFAULT_CANDIDATE_CUTOFF,
    output_top_k: int = DEFAULT_OUTPUT_TOP_K,
    llm_provider: str | None = None, llm_model: str = DEFAULT_LLM_MODEL,
    include_reasoning: bool = True, resume: bool = False,
    skip_success_from_csv: Path | None = None, start: int = 0,
    limit: int | None = None, **kwargs,
) -> None:
    if skip_success_from_csv:
        raise ValueError("Cannot copy historical CSV: old outputs may have compacted invalid IDs")
    if not llm_provider:
        raise ValueError("Explicit --llm_provider is required")
    cols = build_batch_fieldnames(output_top_k, include_total_tokens=True)
    init_output_csv(output_csv, cols)
    groups = load_candidate_group_map(merged_csv)
    seen = load_existing_result_indices(output_csv) if resume else set()
    if not resume and len(pd.read_csv(output_csv)):
        raise ValueError(f"Output CSV already contains data: {output_csv}")
    selected = [i for i in sorted(groups) if i >= start and (limit is None or i < start + limit)]
    for sample_index in selected:
        if sample_index in seen:
            continue
        frame = groups[sample_index]
        run_llm_for_one_sample(sample_index, str(frame["id"].iloc[0]),
                               str(frame["target"].iloc[0]), frame,
                               work_dir, output_csv, candidate_cutoff, output_top_k,
                               llm_provider, llm_model, include_reasoning, cols)
    print(f"[SINGLE_BATCH_DONE] {output_csv}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--candidate_cutoff", type=int, default=DEFAULT_CANDIDATE_CUTOFF)
    parser.add_argument("--output_top_k", type=int, default=DEFAULT_OUTPUT_TOP_K)
    parser.add_argument("--llm_provider", choices=["cerebras", "nchc"], required=True)
    parser.add_argument("--llm_model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--reasoning", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run_llm_batch_selector(Path(args.merged_csv), Path(args.output_csv), Path(args.work_dir),
                           candidate_cutoff=args.candidate_cutoff, output_top_k=args.output_top_k,
                           llm_provider=args.llm_provider, llm_model=args.llm_model,
                           include_reasoning=args.reasoning, resume=args.resume,
                           start=args.start, limit=args.limit)


if __name__ == "__main__":
    main()
