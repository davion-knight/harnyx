"""Deterministic validators for domain-tweak ADK phase outputs."""

from __future__ import annotations

import json
from collections.abc import Callable

from pydantic import ValidationError

from harnyx_commons.domain_tweak_generation.types import (
    DomainTweakAdkTerminalStatus,
    DomainTweakNoGenerateDecision,
    DomainTweakValidationOutcome,
)
from harnyx_commons.miner_task_generation import (
    DomainTweakFormReview,
    DomainTweakQuestionCandidate,
    DomainTweakReferenceAnswerCandidate,
)


def validate_question_generation_output(text: str) -> DomainTweakValidationOutcome:
    payload, error = _extract_json_payload(text)
    if error is not None:
        return error
    if payload.get("no_generate") is True:
        try:
            parsed = DomainTweakNoGenerateDecision.model_validate(payload)
        except ValidationError as exc:
            return _validation_error(exc)
        return DomainTweakValidationOutcome(
            ok=True,
            terminal_status="no_generate",
            parsed_output=None,
            feedback=(f"no_generate: {parsed.reason}",),
        )
    return _validate_model(
        payload,
        model=DomainTweakQuestionCandidate,
        terminal_status="validated",
        extra_checks=_question_checks,
    )


def validate_form_review_output(text: str) -> DomainTweakValidationOutcome:
    payload, error = _extract_json_payload(text)
    if error is not None:
        return error
    try:
        review = DomainTweakFormReview.model_validate(payload)
    except ValidationError as exc:
        return _validation_error(exc)
    if not review.form_match:
        return DomainTweakValidationOutcome(
            ok=True,
            terminal_status="form_rejected",
            parsed_output=review,
            feedback=(review.reviewer_feedback,),
        )
    return DomainTweakValidationOutcome(ok=True, terminal_status="validated", parsed_output=review)


def validate_reference_answer_output(text: str) -> DomainTweakValidationOutcome:
    payload, error = _extract_json_payload(text)
    if error is not None:
        return error
    return _validate_model(
        payload,
        model=DomainTweakReferenceAnswerCandidate,
        terminal_status="validated",
        extra_checks=_reference_answer_checks,
    )


def _validate_model(
    payload: dict[str, object],
    *,
    model: type[DomainTweakQuestionCandidate] | type[DomainTweakReferenceAnswerCandidate],
    terminal_status: DomainTweakAdkTerminalStatus,
    extra_checks: Callable[[object], tuple[str, ...]],
) -> DomainTweakValidationOutcome:
    try:
        parsed = model.model_validate(payload)
    except ValidationError as exc:
        return _validation_error(exc)
    feedback = extra_checks(parsed)
    if feedback:
        return DomainTweakValidationOutcome(
            ok=False,
            terminal_status="validation_failed",
            parsed_output=None,
            feedback=feedback,
        )
    return DomainTweakValidationOutcome(ok=True, terminal_status=terminal_status, parsed_output=parsed)


def _question_checks(parsed: object) -> tuple[str, ...]:
    candidate = parsed if isinstance(parsed, DomainTweakQuestionCandidate) else None
    if candidate is None:
        return ("question generation returned the wrong payload type",)
    feedback: list[str] = []
    if not any(line.strip().startswith(("-", "*")) for line in candidate.solution_plan.splitlines()):
        feedback.append("solution_plan must include Markdown unordered bullet-list plan text.")
    if candidate.question.strip().lower() == candidate.short_answer.strip().lower():
        feedback.append("short_answer must be a benchmark-style answer, not a copy of the question.")
    return tuple(feedback)


def _reference_answer_checks(parsed: object) -> tuple[str, ...]:
    candidate = parsed if isinstance(parsed, DomainTweakReferenceAnswerCandidate) else None
    if candidate is None:
        return ("reference answer generation returned the wrong payload type",)
    feedback: list[str] = []
    reference = candidate.reference_answer
    if not reference.text.strip():
        feedback.append("reference_answer.text must be non-empty.")
    if reference.text.lstrip().startswith(("-", "*", "1.")):
        feedback.append("reference_answer.text must be report-style prose, not solution bullets.")
    if not candidate.premise_assessment.strip():
        feedback.append("premise_assessment must explicitly address false-premise status.")
    if not reference.citations:
        feedback.append("reference_answer.citations must include at least one supporting citation.")
    else:
        for citation in reference.citations:
            if not citation.url.strip():
                feedback.append("every citation must include a non-empty url.")
            if not (citation.title or "").strip():
                feedback.append("every citation must include a non-empty title.")
            if not (citation.note or "").strip():
                feedback.append("every citation must include a claim-bearing note.")
    return tuple(dict.fromkeys(feedback))


def _extract_json_payload(text: str) -> tuple[dict[str, object], DomainTweakValidationOutcome | None]:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {}, DomainTweakValidationOutcome(
            ok=False,
            terminal_status="validation_failed",
            feedback=("Return exactly one JSON object.",),
            error_type="JSONDecodeError",
            error="no JSON object found",
        )
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        return {}, DomainTweakValidationOutcome(
            ok=False,
            terminal_status="validation_failed",
            feedback=("Return valid JSON.",),
            error_type=type(exc).__name__,
            error=str(exc),
        )
    if not isinstance(payload, dict):
        return {}, DomainTweakValidationOutcome(
            ok=False,
            terminal_status="validation_failed",
            feedback=("Return a JSON object, not an array or scalar.",),
            error_type="TypeError",
            error=f"expected object, got {type(payload).__name__}",
        )
    return payload, None


def _validation_error(exc: ValidationError) -> DomainTweakValidationOutcome:
    return DomainTweakValidationOutcome(
        ok=False,
        terminal_status="validation_failed",
        feedback=tuple(str(error) for error in exc.errors()[:5]),
        error_type=type(exc).__name__,
        error=str(exc),
    )


__all__ = [
    "validate_form_review_output",
    "validate_question_generation_output",
    "validate_reference_answer_output",
]
