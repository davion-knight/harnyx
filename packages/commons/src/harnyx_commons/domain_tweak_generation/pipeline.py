"""Pipeline facade for one domain-tweak pair."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from uuid import UUID, uuid4

from pydantic import BaseModel

from harnyx_commons.domain.miner_task import AnswerCitation, MinerTask, Query
from harnyx_commons.domain.shared_config import COMMONS_STRICT_CONFIG
from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.domain.tool_usage_accounting import merge_tool_usage_summaries
from harnyx_commons.domain_tweak_generation.adk_runner import DomainTweakAdkRunner
from harnyx_commons.domain_tweak_generation.prompts import (
    form_review_prompt,
    question_generation_prompt,
    question_generation_repair_prompt,
    reference_answer_prompt,
)
from harnyx_commons.domain_tweak_generation.types import (
    DomainTweakAdkPhaseResult,
    DomainTweakAdkRunConfig,
    DomainTweakFailedFinalization,
    DomainTweakFinalizedTask,
    DomainTweakReviewedQuestion,
    DomainTweakValidationOutcome,
)
from harnyx_commons.domain_tweak_generation.validation import (
    validate_form_review_output,
    validate_question_generation_output,
    validate_reference_answer_output,
)
from harnyx_commons.miner_task_generation import (
    DomainTweakFormReview,
    DomainTweakPairInput,
    DomainTweakQuestionCandidate,
    DomainTweakReferenceAnswerCandidate,
)


class DomainTweakPairRunResult(BaseModel):
    model_config = COMMONS_STRICT_CONFIG

    question_generation: DomainTweakAdkPhaseResult
    form_review: DomainTweakAdkPhaseResult | None = None
    tool_usage: ToolUsageSummary


class DomainTweakGenerationPipeline:
    """Runs pair-level domain-tweak phases without selecting batch policy."""

    def __init__(
        self,
        *,
        config: DomainTweakAdkRunConfig,
        runner: DomainTweakAdkRunner | None = None,
        form_review_retries: int | None = None,
    ) -> None:
        if form_review_retries is not None and form_review_retries < 0:
            raise ValueError("form_review_retries must be non-negative")
        self._config = config
        self._runner = runner or DomainTweakAdkRunner()
        self._form_review_retries = config.max_retries if form_review_retries is None else form_review_retries

    async def generate_question(
        self,
        pair_input: DomainTweakPairInput,
        *,
        prompt: str | None = None,
    ) -> DomainTweakAdkPhaseResult:
        return await self._runner.run_phase(
            phase="question_generation",
            prompt=prompt or question_generation_prompt(pair_input),
            config=self._config,
            validate=validate_question_generation_output,
        )

    async def review_form(
        self,
        pair_input: DomainTweakPairInput,
        question_candidate: DomainTweakQuestionCandidate,
    ) -> DomainTweakAdkPhaseResult:
        return await self._runner.run_phase(
            phase="form_review",
            prompt=form_review_prompt(pair_input, question_candidate),
            config=self._config,
            validate=validate_form_review_output,
        )

    async def generate_reference_answer(
        self,
        question_candidate: DomainTweakQuestionCandidate,
        *,
        timestamp: datetime,
    ) -> DomainTweakAdkPhaseResult:
        return await self._runner.run_phase(
            phase="reference_answer",
            prompt=reference_answer_prompt(question_candidate, timestamp=timestamp),
            config=self._config,
            validate=lambda text: _validate_and_postprocess_reference_answer(text, question_candidate),
        )

    async def generate_reviewed_question(self, pair_input: DomainTweakPairInput) -> DomainTweakPairRunResult:
        question_result = await self.generate_question(pair_input)
        total_usage = question_result.tool_usage
        repair_attempts = 0
        while True:
            if not isinstance(question_result.parsed_output, DomainTweakQuestionCandidate):
                return DomainTweakPairRunResult(question_generation=question_result, tool_usage=total_usage)

            form_result = await self.review_form(pair_input, question_result.parsed_output)
            total_usage = merge_tool_usage_summaries(total_usage, form_result.tool_usage)
            if not isinstance(form_result.parsed_output, DomainTweakFormReview):
                return DomainTweakPairRunResult(
                    question_generation=question_result,
                    form_review=form_result,
                    tool_usage=total_usage,
                )
            if form_result.terminal_status == "validated":
                return DomainTweakPairRunResult(
                    question_generation=question_result,
                    form_review=form_result,
                    tool_usage=total_usage,
                )
            if not _should_retry_question_from_form_review(
                form_result.parsed_output,
                repair_attempts=repair_attempts,
                max_repair_attempts=self._form_review_retries,
            ):
                return DomainTweakPairRunResult(
                    question_generation=question_result,
                    form_review=form_result,
                    tool_usage=total_usage,
                )

            repair_attempts += 1
            question_result = await self.generate_question(
                pair_input,
                prompt=question_generation_repair_prompt(
                    pair_input,
                    question_result.parsed_output,
                    form_result.parsed_output,
                ),
            )
            total_usage = merge_tool_usage_summaries(total_usage, question_result.tool_usage)

    async def finalize_task(
        self,
        reviewed_question: DomainTweakReviewedQuestion,
        *,
        task_id_factory: Callable[[], UUID] = uuid4,
    ) -> DomainTweakFinalizedTask | DomainTweakFailedFinalization:
        reference_result = await self.generate_reference_answer(
            reviewed_question.question_candidate,
            timestamp=reviewed_question.pair_input.timestamp,
        )
        if not isinstance(reference_result.parsed_output, DomainTweakReferenceAnswerCandidate):
            return DomainTweakFailedFinalization(
                reviewed_question=reviewed_question,
                reference_answer_results=(reference_result,),
                tool_usage=reference_result.tool_usage,
            )

        task = MinerTask(
            task_id=task_id_factory(),
            query=Query(text=reviewed_question.question_candidate.question),
            reference_answer=reference_result.parsed_output.reference_answer,
        )
        return DomainTweakFinalizedTask(
            reviewed_question=reviewed_question,
            reference_answer_result=reference_result,
            task=task,
            tool_usage=reference_result.tool_usage,
        )


def _validate_and_postprocess_reference_answer(
    text: str,
    question_candidate: DomainTweakQuestionCandidate,
) -> DomainTweakValidationOutcome:
    outcome = validate_reference_answer_output(text)
    if not outcome.ok or not isinstance(outcome.parsed_output, DomainTweakReferenceAnswerCandidate):
        return outcome
    expected_question = _canonical_question(question_candidate.question)
    actual_question = _canonical_question(outcome.parsed_output.question)
    if actual_question != expected_question:
        return DomainTweakValidationOutcome(
            ok=False,
            terminal_status="validation_failed",
            parsed_output=None,
            feedback=("reference answer question must exactly match the requested question.",),
        )
    formatted_candidate = _format_reference_answer_citations(outcome.parsed_output)
    feedback = _reference_answer_postprocess_feedback(formatted_candidate)
    if feedback:
        return DomainTweakValidationOutcome(
            ok=False,
            terminal_status="validation_failed",
            parsed_output=None,
            feedback=feedback,
        )
    return outcome.model_copy(update={"parsed_output": formatted_candidate})


def _format_reference_answer_citations(
    candidate: DomainTweakReferenceAnswerCandidate,
) -> DomainTweakReferenceAnswerCandidate:
    citations = candidate.reference_answer.citations
    if citations is None:
        return candidate
    formatted = tuple(
        AnswerCitation(
            url=citation.url.strip(),
            title=(citation.title or "").strip(),
            note=(citation.note or "").strip(),
        )
        for citation in citations
    )
    return candidate.model_copy(
        update={
            "reference_answer": candidate.reference_answer.model_copy(update={"citations": formatted}),
        }
    )


def _reference_answer_postprocess_feedback(
    candidate: DomainTweakReferenceAnswerCandidate,
) -> tuple[str, ...]:
    feedback: list[str] = []
    citations = candidate.reference_answer.citations
    if not citations:
        feedback.append("reference_answer.citations must include at least one formatted citation.")
    else:
        for citation in citations:
            if not citation.url:
                feedback.append("every formatted citation must include a non-empty url.")
            if not citation.title:
                feedback.append("every formatted citation must include a non-empty title.")
            if not citation.note:
                feedback.append("every formatted citation must include a claim-bearing note.")
    return tuple(dict.fromkeys(feedback))


def _canonical_question(question: str) -> str:
    return " ".join(question.split()).casefold()


def _should_retry_question_from_form_review(
    review: DomainTweakFormReview,
    *,
    repair_attempts: int,
    max_repair_attempts: int,
) -> bool:
    return bool(review.retry_recommended and repair_attempts < max_repair_attempts)


__all__ = [
    "DomainTweakGenerationPipeline",
    "DomainTweakPairRunResult",
]
