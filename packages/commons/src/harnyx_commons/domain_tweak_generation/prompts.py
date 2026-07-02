"""Prompt builders for the domain-tweak ADK harness."""

from __future__ import annotations

import json
from datetime import datetime

from harnyx_commons.domain_tweak_generation.types import DomainTweakAdkPhase
from harnyx_commons.miner_task_generation import (
    DomainTweakFormReview,
    DomainTweakPairInput,
    DomainTweakQuestionCandidate,
)

SOFT_TIMEOUT_FEEDBACK = (
    "The previous attempt spent too much time searching before producing a final JSON answer.",
    "Time is almost gone. Stop broadening search now.",
    "Use the strongest canonical or structured evidence already found, plus only essential targeted checks.",
    "Write the best supported answer now. If evidence is incomplete, say exactly what is missing and do not fabricate.",
    "Return one valid JSON object with the required reference_answer shape.",
)


def phase_instruction(phase: DomainTweakAdkPhase) -> str:
    base = (
        "You are a grounded research agent. Search before deciding. "
        "Use public evidence as the source of truth. Return raw JSON only. "
        "Do not reveal hidden chain-of-thought."
    )
    match phase:
        case "question_generation":
            return (
                f"{base} Generate at most one new question that preserves the source question's "
                "operation structure while changing the topic domain."
            )
        case "form_review":
            return f"{base} Review whether the generated question preserves the original operation form."
        case "reference_answer":
            return (
                f"{base} Write an answer-first deep-research reference answer with citations and "
                "an explicit false-premise assessment."
            )


def question_generation_prompt(pair_input: DomainTweakPairInput) -> str:
    return (
        f"Current timestamp: {pair_input.timestamp.isoformat()}\n"
        f"Pair id: {pair_input.pair_id}\n\n"
        "Source form question. Preserve this question's operation structure:\n"
        f"{pair_input.deepsearchqa_form_target}\n\n"
        "Domain seed question. Use this only as the new topic/domain seed:\n"
        f"{pair_input.deepresearch9k_domain_target}\n\n"
        "Procedure:\n"
        "- Search broadly before generating.\n"
        "- Extract the original question's operation form: filters, joins, aggregation, comparison, "
        "exclusion, time basis, and answer cardinality.\n"
        "- Build a new question in the domain target that preserves that operation form.\n"
        "- Prefer aggregation or multi-source reconciliation over direct lookup.\n"
        "- Reject shallow single-page lookups and old-memory questions.\n"
        "- If the form cannot be preserved with grounded evidence, return no_generate.\n\n"
        "Valid output shapes:\n"
        '{"question": "...", "short_answer": "...", "solution_plan": "- ...\\n- ..."}\n'
        'or {"no_generate": true, "reason": "...", "retry_recommended": false}\n\n'
        "Rules:\n"
        "- solution_plan must be Markdown unordered bullet-list plan text.\n"
        "- Return exactly one JSON object and no markdown fences."
    )


def question_generation_repair_prompt(
    pair_input: DomainTweakPairInput,
    candidate: DomainTweakQuestionCandidate,
    review: DomainTweakFormReview,
) -> str:
    candidate_payload = candidate.model_dump(mode="json")
    review_payload = review.model_dump(mode="json")
    return (
        f"Current timestamp: {pair_input.timestamp.isoformat()}\n"
        f"Pair id: {pair_input.pair_id}\n\n"
        "Source form question. Preserve this question's operation structure:\n"
        f"{pair_input.deepsearchqa_form_target}\n\n"
        "Domain seed question. Use this only as the new topic/domain seed:\n"
        f"{pair_input.deepresearch9k_domain_target}\n\n"
        "Previous generated candidate:\n"
        f"{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}\n\n"
        "Independent form-review feedback:\n"
        f"{json.dumps(review_payload, ensure_ascii=False, indent=2)}\n\n"
        "Repair task:\n"
        "- Generate a replacement question for the same source/domain pair.\n"
        "- Preserve the source form's operation structure more strictly than the previous candidate.\n"
        "- Address reviewer_feedback directly.\n"
        "- Keep the replacement grounded in the domain seed.\n"
        "- If the reviewer feedback cannot be repaired with grounded evidence, return no_generate.\n\n"
        "Valid output shapes:\n"
        '{"question": "...", "short_answer": "...", "solution_plan": "- ...\\n- ..."}\n'
        'or {"no_generate": true, "reason": "...", "retry_recommended": false}\n\n'
        "Rules:\n"
        "- solution_plan must be Markdown unordered bullet-list plan text.\n"
        "- Return exactly one JSON object and no markdown fences."
    )


def form_review_prompt(pair_input: DomainTweakPairInput, candidate: DomainTweakQuestionCandidate) -> str:
    candidate_payload = candidate.model_dump(mode="json")
    return (
        f"Current timestamp: {pair_input.timestamp.isoformat()}\n"
        f"Pair id: {pair_input.pair_id}\n\n"
        "Source form question:\n"
        f"{pair_input.deepsearchqa_form_target}\n\n"
        "Generated candidate:\n"
        f"{json.dumps(candidate_payload, ensure_ascii=False, indent=2)}\n\n"
        "Review task:\n"
        "- Compare operation structure, not topic words.\n"
        "- Check filters, joins, aggregation/comparison, answer cardinality, time basis, and exclusions.\n"
        "- Check whether the generated question introduces a false or unsupported premise.\n"
        "- Decide whether the candidate can proceed to reference-answer generation.\n\n"
        "Return JSON with shape:\n"
        '{"form_match": true|false, "false_premise_status": "none|possible|confirmed", '
        '"reviewer_feedback": "...", "retry_recommended": true|false}\n'
        "Return exactly one JSON object and no markdown fences."
    )


def reference_answer_prompt(candidate: DomainTweakQuestionCandidate, *, timestamp: datetime) -> str:
    return (
        f"Current timestamp: {timestamp.isoformat()}\n\n"
        "You are a dedicated deep-research reference-answer writer.\n\n"
        "Question:\n"
        f"{candidate.question}\n\n"
        "Objective:\n"
        "- Produce a trustworthy reference answer for the exact question.\n"
        "- Use public evidence, exact scope, claim binding, and relevant citations.\n"
        "- Prefer bounded research: find high-yield canonical evidence first, and stop broadening search once the "
        "answer, scope, and load-bearing calculations are adequately supported.\n"
        "- Do not include hidden chain-of-thought.\n\n"
        "Answer rules:\n"
        "- First, check whether the question contains a false, partially supported, or unresolved premise.\n"
        "- If the premise is false, partially supported, or unresolved, reference_answer.text must say that near "
        "the beginning and then provide corrected facts or the strongest evidence-backed partial answer.\n"
        "- If the premise is supported, cite premise support when it is non-obvious or source-dependent.\n"
        "- reference_answer.text must start with the direct answer.\n"
        "- For set or list questions, put the complete final set near the top.\n"
        "- For exhaustive set or list questions, include an included-entity proof table or compact proof list.\n"
        "- For each included entity, bind every query predicate to evidence: candidate-pool membership, "
        "inclusion criteria, exclusion criteria not triggered, date/time basis, and values used in filters "
        "or calculations.\n"
        "- Then explain the key evidence, candidate pool, filters, and calculations needed "
        "to prove the answer is complete.\n"
        "- Every non-obvious or search-dependent claim in reference_answer.text must be supported by a "
        "citation object.\n"
        "- A citation note is scorer-visible evidence. Write each note as a compact factual grounding snippet "
        "that says exactly which visible claim it supports.\n"
        "- If one citation does not support all major subclaims, add targeted citations for the unsupported "
        "subclaims.\n"
        "- Cover every requested entity, date, filter, comparison side, calculation operand, exclusion, and "
        "output ordering.\n"
        "- For list-all questions, include completeness evidence. Do not claim broad all-others exclusions or "
        "universal negatives unless an exhaustive source directly supports the relevant predicate.\n"
        "- Discuss excluded or borderline entities only when they are necessary to prove completeness; support "
        "those exclusions with targeted citation notes or do not make the exclusion claim.\n"
        "- For comparisons or calculations, cite each operand and the conclusion.\n"
        "- Prefer official, primary, canonical, database, or specialized sources; avoid relying only on Wikipedia.\n"
        "- Use a small set of strong citations. A single canonical source is acceptable when it is exhaustive for "
        "the requested scope; otherwise use enough independent source support to make the answer trustworthy.\n"
        "- Do not use irrelevant citations. Fewer strong citations are better than many weak citations.\n"
        "- If sources conflict, state the conflict and why the final answer uses the chosen source.\n"
        "- If evidence is incomplete after diligent search, give the best supported answer and state the exact "
        "missing evidence. Do not fabricate.\n\n"
        "Search procedure:\n"
        "1. Decompose the question into premise truth, identity, candidate pool, filters, comparisons, "
        "calculation operands, final ordering, and exclusions.\n"
        "2. Search specifically for premise support or contradiction before solving.\n"
        "3. Search first for official, canonical, structured, or database-like sources that cover the candidate "
        "pool and multiple requested operands at once.\n"
        "4. Search separately only for filters, comparisons, or calculation operands that are not already covered "
        "by the high-yield sources.\n"
        "5. Search for excluded or borderline cases only when the final answer would otherwise be ambiguous or "
        "likely incomplete.\n"
        "6. Cross-check the final answer when no single canonical source fully covers the requested scope.\n"
        "7. If the answer depends on a complete cast, list, ranking, or table, cite that source as candidate-pool "
        "evidence and separately cite entity-level predicates when the pool source does not cover them.\n"
        "8. Write reference_answer.text only after all load-bearing claims have source support.\n\n"
        "Compact exhaustive-list example:\n"
        "- Good: `Answer: A and B. Proof: A -- pool source establishes A is in scope; source X supports "
        "criterion 1; source Y supports criterion 2; source Z supports the relevant date. B -- same "
        "predicate-by-predicate support. Completeness: canonical table covers the pool; borderline C is excluded "
        "because source W directly shows the failed predicate.`\n"
        "- Bad: `Answer: A and B; all others fail` without evidence for every included entity's predicates and "
        "without targeted support for excluded or borderline entities.\n\n"
        "Return JSON with shape:\n"
        '{"question": "...", "premise_assessment": "...", '
        '"reference_answer": {"text": "...", "citations": [{"url": "...", "title": "...", "note": "..."}]}}\n'
        "Return exactly one JSON object and no markdown fences."
    )


def feedback_prompt(feedback: tuple[str, ...]) -> str:
    return (
        "Your previous output failed deterministic validation. Revise it using this feedback:\n"
        + "\n".join(f"- {item}" for item in feedback)
        + "\nReturn one corrected JSON object only."
    )


def soft_timeout_feedback_prompt(feedback: tuple[str, ...] = SOFT_TIMEOUT_FEEDBACK) -> str:
    return (
        "Your previous reference-answer attempt ran too long before finalizing.\n"
        + "\n".join(f"- {item}" for item in feedback)
        + "\nDo not restart broad research. Return one corrected JSON object only."
    )


__all__ = [
    "SOFT_TIMEOUT_FEEDBACK",
    "feedback_prompt",
    "form_review_prompt",
    "phase_instruction",
    "question_generation_prompt",
    "question_generation_repair_prompt",
    "reference_answer_prompt",
    "soft_timeout_feedback_prompt",
]
