from __future__ import annotations

import json


def validate_output_top_k(output_top_k: int) -> int:
    output_top_k = int(output_top_k)
    if output_top_k < 1:
        raise ValueError("output_top_k must be at least 1.")
    return output_top_k


def build_base_system_prompt(output_top_k: int) -> str:
    output_top_k = validate_output_top_k(output_top_k)
    return f"""
You are a retrosynthesis candidate selection assistant.

Your task is to select and rank the top {output_top_k} precursor sets
from the provided model candidates.

Core rules:
- Choose ONLY from the provided candidate_id values.
- Do NOT invent, modify, repair, or rewrite candidate IDs.
- Every output candidate ID must exactly match one candidate_id value from the input.
- Candidates with identical canonical_precursors share the same candidate_id and represent the same precursor set.
- Repeated candidate_id values across different models may indicate model agreement.
- Use rank and score only within the same model.
- Do NOT compare scores across different models because each model uses a different scoring scale.
- Rank the selected candidate IDs from best to worst.
- Return JSON only.
""".strip()


def build_retroknn_system_prompt(output_top_k: int) -> str:
    output_top_k = validate_output_top_k(output_top_k)
    return f"""
You are a retrosynthesis candidate selection assistant.

Your task is to select and rank the top {output_top_k} precursor sets
from the provided model candidates.

Core rules:
- Choose ONLY from the provided candidate_id values.
- Do NOT invent, modify, repair, or rewrite candidate IDs.
- Every output candidate ID must exactly match one candidate_id value from the input.
- Candidates with identical canonical_precursors share the same candidate_id and represent the same precursor set.
- Repeated candidate_id values across different models may indicate model agreement.
- Use rank and score only within the same model.
- Do NOT compare scores across different models because each model uses a different scoring scale.
- In this project, retroknn has shown higher top-k accuracy than the original four models, so treat retroknn as an important reference signal.
- However, do NOT blindly trust retroknn. Use chemical judgment, model agreement, within-model rank, and precursor plausibility.
- Rank the selected candidate IDs from best to worst.
- Return JSON only.
""".strip()


def has_retroknn_candidate(candidates: list) -> bool:
    for candidate in candidates:
        model = str(candidate.get("model", "")).strip().lower()
        if model == "retroknn":
            return True
    return False


def get_system_prompt(candidates: list, output_top_k: int = 10) -> str:
    if has_retroknn_candidate(candidates):
        return build_retroknn_system_prompt(output_top_k)
    return build_base_system_prompt(output_top_k)


def build_candidate_field_description(include_retroknn: bool) -> str:
    if include_retroknn:
        model_examples = (
            "augmented_transformer, graph2smiles, retrosim, "
            "template_relevance, or retroknn"
        )
    else:
        model_examples = (
            "augmented_transformer, graph2smiles, retrosim, "
            "or template_relevance"
        )

    return f"""
Each candidate is a JSON object with the following fields:
- candidate_id: identifier of the precursor set. Return this field when selecting candidates.
- model: model source, such as {model_examples}.
- rank: rank within that model. Lower rank is better within the same model.
- score: model-specific score. Use only within the same model.
- canonical_precursors: standardized predicted reactant SMILES.
Candidates with the same canonical_precursors always have the same candidate_id.
""".strip()


def build_retroknn_guidance() -> str:
    return """
RetroKNN guidance:
- In this project, retroknn has shown higher top-k accuracy than the original four models.
- You may give retroknn candidates stronger consideration, especially when the retroknn candidate is high-ranked or agrees with another model.
- However, retroknn is not always correct. Do NOT blindly select retroknn candidates only because they come from retroknn.
- Choose freely based on the candidate list, chemical plausibility, ranks within each model, and model agreement.
""".strip()


def build_selection_constraint(output_top_k: int) -> str:
    output_top_k = validate_output_top_k(output_top_k)
    output_key = f"top_{output_top_k}_candidate_ids"
    return f"""
***SELECTION CONSTRAINT***
You are NOT generating retrosynthesis answers from scratch.
You are ONLY selecting from the model-provided candidate list.

***EVERY selected ID in {output_key} MUST be copied EXACTLY from a candidate_id value in the Candidates list.***
***Do NOT output SMILES in {output_key}.***
***Do NOT invent, rewrite, or alter any candidate ID.***
***Before returning the final JSON, verify that every selected {output_key} entry appears exactly in the input Candidates list.***
***If a selected ID does not appear in the input Candidates list, it is invalid for this task and must be replaced.***
***Do not select the same candidate_id more than once.***
***Return the selected candidate IDs in descending order of preference.***
""".strip()


def build_checklist(include_retroknn: bool, output_top_k: int) -> str:
    output_top_k = validate_output_top_k(output_top_k)
    output_key = f"top_{output_top_k}_candidate_ids"
    checklist = [
        f"1. Are all {output_key} copied exactly from the provided candidate_id values?",
        (
            f"2. Are there exactly {output_top_k} selected candidate IDs, "
            "unless fewer unique candidates are provided?"
        ),
        "3. Are there no invented, modified, repeated, or rewritten candidate IDs?",
        "4. Are the selected candidate IDs ordered from best to worst?",
    ]
    if include_retroknn:
        checklist.append(
            "5. Did you consider retroknn as a strong reference signal without blindly trusting it?"
        )
    return "\n".join(checklist)


def build_output_format(
    include_retroknn: bool,
    output_top_k: int,
    include_reasoning: bool = True,
) -> str:
    output_top_k = validate_output_top_k(output_top_k)
    output_key = f"top_{output_top_k}_candidate_ids"

    if not include_reasoning:
        example = {
            output_key: [
                "candidate_id_1",
                "candidate_id_2",
            ]
        }
        return (
            "Return JSON only in exactly this schema. Do not include reason, "
            "summary, analysis, explanation, or decision_trace fields:\n"
            + json.dumps(example, indent=2, ensure_ascii=False)
        )

    if include_retroknn:
        trace_instruction = (
            "Brief observable reason for choosing candidate_id_1. "
            "Mention model support, rank, retroknn support if relevant, "
            "and chemical plausibility."
        )
        final_reason = (
            "A short final summary of the ranking. Mention how retroknn "
            "support, model agreement, and chemical plausibility were considered."
        )
    else:
        trace_instruction = (
            "Brief observable reason for choosing candidate_id_1. "
            "Mention model support, rank, model agreement if relevant, "
            "and chemical plausibility."
        )
        final_reason = (
            "A short final summary of the ranking. Mention how model "
            "agreement and chemical plausibility were considered."
        )

    example = {
        output_key: [
            "candidate_id_1",
            "candidate_id_2",
        ],
        "decision_trace": [
            trace_instruction,
            "Brief observable reason for choosing candidate_id_2.",
        ],
        "reason": final_reason,
    }
    return "Return JSON only in this format:\n" + json.dumps(
        example,
        indent=2,
        ensure_ascii=False,
    )


def build_user_prompt(
    target: str,
    candidates: list,
    output_top_k: int = 10,
    include_reasoning: bool = True,
) -> str:
    output_top_k = validate_output_top_k(output_top_k)
    include_retroknn = has_retroknn_candidate(candidates)

    sections = [
        f"Target product:\n{target}",
        build_candidate_field_description(include_retroknn),
    ]

    if include_retroknn:
        sections.append(build_retroknn_guidance())

    sections.extend(
        [
            build_selection_constraint(output_top_k),
            "Candidates:\n"
            + json.dumps(candidates, indent=2, ensure_ascii=False),
            f"Select and rank the top {output_top_k} best precursor sets.",
            (
                "Before finalizing, perform this checklist internally. "
                "Do not expose the checklist unless the requested output schema includes reasoning:\n"
                + build_checklist(
                    include_retroknn=include_retroknn,
                    output_top_k=output_top_k,
                )
            ),
            build_output_format(
                include_retroknn=include_retroknn,
                output_top_k=output_top_k,
                include_reasoning=include_reasoning,
            ),
        ]
    )

    return "\n\n".join(sections)


# Backward-compatible defaults.
BASE_SYSTEM_PROMPT = build_base_system_prompt(10)
RETROKNN_SYSTEM_PROMPT = build_retroknn_system_prompt(10)
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT
