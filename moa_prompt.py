from __future__ import annotations

import json

from prompt import (
    build_candidate_field_description,
    build_checklist,
    build_output_format,
    build_retroknn_guidance,
    build_selection_constraint,
    build_user_prompt,
    get_system_prompt,
    has_retroknn_candidate,
)


FINAL_OUTPUT_TOP_K = 10


def build_agent_prompts(
    target: str,
    candidates: list[dict],
    agent_output_top_k: int,
    include_reasoning: bool = True,
) -> tuple[str, str]:
    """
    Build the MoA agent prompt with exactly the same prompt logic
    as the Single-LLM selector.

    The intended difference is only output_top_k:
      - Single LLM normally outputs Top-10.
      - MoA agents normally output Top-20.
    """
    if agent_output_top_k < 1:
        raise ValueError("agent_output_top_k must be at least 1.")

    system_prompt = get_system_prompt(
        candidates=candidates,
        output_top_k=agent_output_top_k,
    )

    user_prompt = build_user_prompt(
        target=target,
        candidates=candidates,
        output_top_k=agent_output_top_k,
        include_reasoning=include_reasoning,
    )

    return system_prompt, user_prompt


def build_aggregator_system_prompt(
    final_output_top_k: int = FINAL_OUTPUT_TOP_K,
) -> str:
    """
    Keep the original aggregator-role framing, while leaving the general
    candidate-selection rules/constraints to the same prompt helpers used
    by Single LLM and agents.
    """
    if final_output_top_k < 1:
        raise ValueError("final_output_top_k must be at least 1.")

    return f"""
You are the final aggregator in a Mixture-of-Agents retrosynthesis
candidate selection system.

You receive:
1. the original target product,
2. the complete original candidate list,
3. anonymous responses generated independently by multiple LLM agents.

Your task is to critically evaluate the agent responses and synthesize
one new final ranking of the top {final_output_top_k} precursor sets.

Important aggregation rules:
- Do not merely choose one agent response.
- Do not blindly copy one complete agent ranking.
- Agent responses are auxiliary information and may contain mistakes.
- The original Candidates list is the source of truth.
- You may select any candidate_id from the original Candidates list,
  even if no agent selected it.

Return JSON only.
""".strip()


def format_anonymous_agent_responses(
    agent_responses: list[str],
) -> str:
    if not agent_responses:
        raise ValueError("At least one agent response is required.")

    return "\n\n".join(
        f"Response {index}:\n{response}"
        for index, response in enumerate(
            agent_responses,
            start=1,
        )
    )


def build_aggregator_prompts(
    target: str,
    candidates: list[dict],
    agent_responses: list[str],
    final_output_top_k: int = FINAL_OUTPUT_TOP_K,
    include_reasoning: bool = True,
) -> tuple[str, str]:
    """
    Build the final aggregator prompt.

    Design:
    - Aggregator role/system prompt: preserve the old MoA role framing.
    - General candidate-selection instructions: identical helpers to Single/agents.
    - Candidate description: identical to Single/agents.
    - RetroKNN guidance: identical to Single/agents.
    - Selection constraint: identical to Single/agents.
    - Checklist: identical to Single/agents.
    - Output/reasoning schema: identical to Single/agents.
    - Final output size: Top-10 by default.
    - MoA-only additions: anonymous agent responses and aggregation instruction.
    """
    if final_output_top_k < 1:
        raise ValueError("final_output_top_k must be at least 1.")
    if not agent_responses:
        raise ValueError("At least one agent response is required.")

    include_retroknn = has_retroknn_candidate(candidates)

    system_prompt = build_aggregator_system_prompt(
        final_output_top_k=final_output_top_k,
    )

    sections = [
        f"Target product:\n{target}",
        build_candidate_field_description(
            include_retroknn=include_retroknn,
        ),
    ]

    if include_retroknn:
        sections.append(
            build_retroknn_guidance()
        )

    sections.extend(
        [
            build_selection_constraint(
                output_top_k=final_output_top_k,
            ),

            "Candidates:\n"
            + json.dumps(
                candidates,
                indent=2,
                ensure_ascii=False,
            ),

            "Responses from agents:\n\n"
            + format_anonymous_agent_responses(
                agent_responses
            ),

            (
                "Critically evaluate all responses and synthesize a new final "
                f"ranking of the top {final_output_top_k} best precursor sets. "
                "Do not merely choose one response. Use the agent responses as "
                "auxiliary information together with the original Candidates list."
            ),

            (
                "Before finalizing, perform this checklist internally. "
                "Do not expose the checklist unless the requested output schema "
                "includes reasoning:\n"
                + build_checklist(
                    include_retroknn=include_retroknn,
                    output_top_k=final_output_top_k,
                )
            ),

            build_output_format(
                include_retroknn=include_retroknn,
                output_top_k=final_output_top_k,
                include_reasoning=include_reasoning,
            ),
        ]
    )

    user_prompt = "\n\n".join(sections)

    return system_prompt, user_prompt
