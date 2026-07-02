"""Google ADK runner boundary for domain-tweak generation phases."""

from __future__ import annotations

import asyncio
import inspect
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.domain.tool_usage_accounting import merge_tool_usage_summaries
from harnyx_commons.domain_tweak_generation.adk_events import (
    final_text_from_event,
    summarize_adk_event,
    tool_usage_from_adk_events,
)
from harnyx_commons.domain_tweak_generation.prompts import (
    SOFT_TIMEOUT_FEEDBACK,
    feedback_prompt,
    phase_instruction,
    soft_timeout_feedback_prompt,
)
from harnyx_commons.domain_tweak_generation.types import (
    DomainTweakAdkAttempt,
    DomainTweakAdkEventSummary,
    DomainTweakAdkPhase,
    DomainTweakAdkPhaseResult,
    DomainTweakAdkPromptKind,
    DomainTweakAdkRunConfig,
    DomainTweakAdkTerminalStatus,
    DomainTweakValidationOutcome,
)

ValidationFunction = Callable[[str], DomainTweakValidationOutcome]


@dataclass(frozen=True, slots=True)
class DomainTweakAdkTurn:
    final_text: str
    events: tuple[DomainTweakAdkEventSummary, ...]


class DomainTweakAdkTurnExecutor(Protocol):
    async def __call__(
        self,
        *,
        phase: DomainTweakAdkPhase,
        prompt: str,
        attempt_index: int,
        config: DomainTweakAdkRunConfig,
        agent_instruction: str,
    ) -> DomainTweakAdkTurn:
        """Run one ADK turn. Tests use this to avoid live Google APIs."""


class DomainTweakAdkRunner:
    """Runs one ADK phase with retry feedback inside one session."""

    def __init__(self, *, turn_executor: DomainTweakAdkTurnExecutor | None = None) -> None:
        self._turn_executor = turn_executor

    async def run_phase(
        self,
        *,
        phase: DomainTweakAdkPhase,
        prompt: str,
        config: DomainTweakAdkRunConfig,
        validate: ValidationFunction,
    ) -> DomainTweakAdkPhaseResult:
        started = time.perf_counter()
        deadline = started + config.phase_timeout_seconds
        attempts: list[DomainTweakAdkAttempt] = []
        total_usage = ToolUsageSummary.zero()
        agent_instruction = phase_instruction(phase)
        live_context = None
        if self._turn_executor is None:
            live_context = await _LiveAdkContext.create(
                phase=phase,
                config=config,
                agent_instruction=agent_instruction,
            )
        try:
            attempt_index = 0
            validation_failure_count = 0
            while True:
                prompt_kind = _prompt_kind_for_attempt(attempt_index, attempts)
                now = time.perf_counter()
                turn_prompt = _prompt_for_attempt(
                    prompt,
                    prompt_kind,
                    attempts,
                    elapsed_seconds=now - started,
                )
                remaining_timeout = deadline - now
                if remaining_timeout <= 0:
                    raise TimeoutError("ADK phase timeout exceeded before retry")
                turn_elapsed_seconds = now - started
                turn_soft_timeout_elapsed_seconds = _next_turn_soft_timeout_elapsed_seconds(
                    config=config,
                    attempts=attempts,
                    elapsed_seconds=turn_elapsed_seconds,
                )
                turn_timeout = _turn_timeout_seconds(
                    soft_timeout_elapsed_seconds=turn_soft_timeout_elapsed_seconds,
                    elapsed_seconds=turn_elapsed_seconds,
                    remaining_timeout=remaining_timeout,
                )
                turn_events: list[DomainTweakAdkEventSummary] = []
                try:
                    turn = await asyncio.wait_for(
                        self._run_turn(
                            phase=phase,
                            prompt=turn_prompt,
                            attempt_index=attempt_index,
                            config=config,
                            agent_instruction=agent_instruction,
                            live_context=live_context,
                            event_summaries=turn_events,
                        ),
                        timeout=turn_timeout,
                    )
                except TimeoutError as exc:
                    timeout_elapsed_seconds = time.perf_counter() - started
                    if _should_soft_timeout_retry(
                        soft_timeout_elapsed_seconds=turn_soft_timeout_elapsed_seconds,
                        elapsed_seconds=timeout_elapsed_seconds,
                        deadline=deadline,
                        started=started,
                    ):
                        total_usage = self._record_failed_attempt(
                            attempts=attempts,
                            total_usage=total_usage,
                            attempt_index=attempt_index,
                            prompt_kind=prompt_kind,
                            event_summaries=turn_events,
                            config=config,
                            validation_feedback=SOFT_TIMEOUT_FEEDBACK,
                        )
                        attempt_index += 1
                        continue
                    return self._failed_phase_result(
                        phase=phase,
                        attempts=attempts,
                        total_usage=total_usage,
                        attempt_index=attempt_index,
                        prompt_kind=prompt_kind,
                        event_summaries=turn_events,
                        config=config,
                        started=started,
                        exc=exc,
                        terminal_status="timeout",
                    )
                except Exception as exc:
                    if live_context is not None and not turn_events:
                        raise _FatalAdkRequestError(exc) from exc
                    return self._failed_phase_result(
                        phase=phase,
                        attempts=attempts,
                        total_usage=total_usage,
                        attempt_index=attempt_index,
                        prompt_kind=prompt_kind,
                        event_summaries=turn_events,
                        config=config,
                        started=started,
                        exc=exc,
                        terminal_status="invocation_error",
                    )
                validation = validate(turn.final_text)
                attempt_usage = tool_usage_from_adk_events(
                    turn.events,
                    provider=config.provider,
                    model=config.model,
                )
                attempts.append(
                    DomainTweakAdkAttempt(
                        attempt_index=attempt_index,
                        prompt_kind=prompt_kind,
                        final_text_preview=turn.final_text[:500],
                        final_text_length=len(turn.final_text),
                        validation_ok=validation.ok,
                        validation_feedback=validation.feedback,
                        event_summaries=turn.events,
                        tool_usage=attempt_usage,
                    )
                )
                total_usage = merge_tool_usage_summaries(total_usage, attempt_usage)
                if validation.ok:
                    return DomainTweakAdkPhaseResult(
                        phase=phase,
                        terminal_status=validation.terminal_status,
                        parsed_output=validation.parsed_output,
                        attempts=tuple(attempts),
                        tool_usage=total_usage,
                        elapsed_ms=_elapsed_ms(started),
                    )
                validation_failure_count += 1
                if validation_failure_count > config.max_retries:
                    return DomainTweakAdkPhaseResult(
                        phase=phase,
                        terminal_status="validation_failed",
                        parsed_output=None,
                        attempts=tuple(attempts),
                        tool_usage=total_usage,
                        elapsed_ms=_elapsed_ms(started),
                    )
                attempt_index += 1
        except _FatalAdkRequestError as exc:
            raise exc.original from exc
        except TimeoutError as exc:
            return DomainTweakAdkPhaseResult(
                phase=phase,
                terminal_status="timeout",
                attempts=tuple(attempts),
                tool_usage=total_usage,
                elapsed_ms=_elapsed_ms(started),
                error_type=type(exc).__name__,
                error=str(exc),
            )
        except Exception as exc:
            return DomainTweakAdkPhaseResult(
                phase=phase,
                terminal_status="invocation_error",
                attempts=tuple(attempts),
                tool_usage=total_usage,
                elapsed_ms=_elapsed_ms(started),
                error_type=type(exc).__name__,
                error=str(exc),
            )
        finally:
            if live_context is not None:
                await live_context.close()

    async def _run_turn(
        self,
        *,
        phase: DomainTweakAdkPhase,
        prompt: str,
        attempt_index: int,
        config: DomainTweakAdkRunConfig,
        agent_instruction: str,
        live_context: _LiveAdkContext | None,
        event_summaries: list[DomainTweakAdkEventSummary],
    ) -> DomainTweakAdkTurn:
        if self._turn_executor is not None:
            return await self._turn_executor(
                phase=phase,
                prompt=prompt,
                attempt_index=attempt_index,
                config=config,
                agent_instruction=agent_instruction,
            )
        if live_context is None:
            raise RuntimeError("live ADK context was not initialized")
        return await live_context.run_turn(prompt, event_summaries=event_summaries)

    def _failed_phase_result(
        self,
        *,
        phase: DomainTweakAdkPhase,
        attempts: list[DomainTweakAdkAttempt],
        total_usage: ToolUsageSummary,
        attempt_index: int,
        prompt_kind: DomainTweakAdkPromptKind,
        event_summaries: list[DomainTweakAdkEventSummary],
        config: DomainTweakAdkRunConfig,
        started: float,
        exc: BaseException,
        terminal_status: DomainTweakAdkTerminalStatus,
    ) -> DomainTweakAdkPhaseResult:
        total_usage = self._record_failed_attempt(
            attempts=attempts,
            total_usage=total_usage,
            attempt_index=attempt_index,
            prompt_kind=prompt_kind,
            event_summaries=event_summaries,
            config=config,
            validation_feedback=(str(exc),) if str(exc) else (),
        )
        return DomainTweakAdkPhaseResult(
            phase=phase,
            terminal_status=terminal_status,
            attempts=tuple(attempts),
            tool_usage=total_usage,
            elapsed_ms=_elapsed_ms(started),
            error_type=type(exc).__name__,
            error=str(exc),
        )

    def _record_failed_attempt(
        self,
        *,
        attempts: list[DomainTweakAdkAttempt],
        total_usage: ToolUsageSummary,
        attempt_index: int,
        prompt_kind: DomainTweakAdkPromptKind,
        event_summaries: list[DomainTweakAdkEventSummary],
        config: DomainTweakAdkRunConfig,
        validation_feedback: tuple[str, ...],
    ) -> ToolUsageSummary:
        attempt_usage = (
            tool_usage_from_adk_events(
                event_summaries,
                provider=config.provider,
                model=config.model,
            )
            if event_summaries
            else ToolUsageSummary.zero()
        )
        attempts.append(
            DomainTweakAdkAttempt(
                attempt_index=attempt_index,
                prompt_kind=prompt_kind,
                validation_ok=False,
                validation_feedback=validation_feedback,
                event_summaries=tuple(event_summaries),
                tool_usage=attempt_usage,
            )
        )
        return merge_tool_usage_summaries(total_usage, attempt_usage)


class _FatalAdkRequestError(Exception):
    def __init__(self, original: Exception) -> None:
        super().__init__(str(original))
        self.original = original


def _prompt_kind_for_attempt(
    attempt_index: int,
    attempts: list[DomainTweakAdkAttempt],
) -> DomainTweakAdkPromptKind:
    if attempt_index == 0:
        return "initial"
    if attempts and attempts[-1].validation_feedback == SOFT_TIMEOUT_FEEDBACK:
        return "soft_timeout_feedback"
    return "feedback"


def _prompt_for_attempt(
    initial_prompt: str,
    prompt_kind: DomainTweakAdkPromptKind,
    attempts: list[DomainTweakAdkAttempt],
    *,
    elapsed_seconds: float | None = None,
) -> str:
    match prompt_kind:
        case "initial":
            return initial_prompt
        case "soft_timeout_feedback":
            return soft_timeout_feedback_prompt(
                attempts[-1].validation_feedback,
                elapsed_seconds=elapsed_seconds,
            )
        case "feedback":
            return feedback_prompt(attempts[-1].validation_feedback)


def _turn_timeout_seconds(
    *,
    soft_timeout_elapsed_seconds: float | None,
    elapsed_seconds: float,
    remaining_timeout: float,
) -> float:
    if soft_timeout_elapsed_seconds is None:
        return remaining_timeout
    return min(max(soft_timeout_elapsed_seconds - elapsed_seconds, 0.0), remaining_timeout)


def _should_soft_timeout_retry(
    *,
    soft_timeout_elapsed_seconds: float | None,
    elapsed_seconds: float,
    deadline: float,
    started: float,
) -> bool:
    if soft_timeout_elapsed_seconds is None:
        return False
    if elapsed_seconds < soft_timeout_elapsed_seconds:
        return False
    return started + elapsed_seconds < deadline


def _next_turn_soft_timeout_elapsed_seconds(
    *,
    config: DomainTweakAdkRunConfig,
    attempts: list[DomainTweakAdkAttempt],
    elapsed_seconds: float,
) -> float | None:
    next_soft_timeout_seconds = _next_soft_timeout_elapsed_seconds(
        config=config,
        attempts=attempts,
    )
    if next_soft_timeout_seconds is None:
        return None
    return _next_future_soft_timeout_elapsed_seconds(
        config=config,
        next_soft_timeout_seconds=next_soft_timeout_seconds,
        elapsed_seconds=elapsed_seconds,
    )


def _soft_timeout_feedback_count(attempts: list[DomainTweakAdkAttempt]) -> int:
    return sum(1 for attempt in attempts if attempt.validation_feedback == SOFT_TIMEOUT_FEEDBACK)


def _next_soft_timeout_elapsed_seconds(
    *,
    config: DomainTweakAdkRunConfig,
    attempts: list[DomainTweakAdkAttempt],
) -> float | None:
    if config.soft_timeout_seconds is None:
        return None
    soft_timeout_count = _soft_timeout_feedback_count(attempts)
    if soft_timeout_count > 0 and config.soft_timeout_interval_seconds is None:
        return None
    if soft_timeout_count == 0:
        return config.soft_timeout_seconds
    if config.soft_timeout_interval_seconds is None:
        return None
    return config.soft_timeout_seconds + soft_timeout_count * config.soft_timeout_interval_seconds


def _next_future_soft_timeout_elapsed_seconds(
    *,
    config: DomainTweakAdkRunConfig,
    next_soft_timeout_seconds: float,
    elapsed_seconds: float,
) -> float | None:
    if elapsed_seconds < next_soft_timeout_seconds:
        return next_soft_timeout_seconds
    if config.soft_timeout_interval_seconds is None:
        return None
    missed_intervals = int(
        (elapsed_seconds - next_soft_timeout_seconds) // config.soft_timeout_interval_seconds
    ) + 1
    return next_soft_timeout_seconds + missed_intervals * config.soft_timeout_interval_seconds


class _LiveAdkContext:
    def __init__(self, *, runner: Any, user_id: str, session_id: str) -> None:
        self._runner = runner
        self._user_id = user_id
        self._session_id = session_id

    @classmethod
    async def create(
        cls,
        *,
        phase: DomainTweakAdkPhase,
        config: DomainTweakAdkRunConfig,
        agent_instruction: str,
    ) -> _LiveAdkContext:
        from google.adk.agents import Agent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        agent = Agent(
            name=_agent_name(phase),
            model=config.model,
            instruction=agent_instruction,
            tools=_adk_tools_for_phase(phase),
        )
        session_service = InMemorySessionService()
        session_id = f"{phase}-{int(time.time() * 1000)}"
        await session_service.create_session(
            app_name=config.app_name,
            user_id=config.user_id,
            session_id=session_id,
        )
        runner = Runner(app_name=config.app_name, agent=agent, session_service=session_service)
        return cls(runner=runner, user_id=config.user_id, session_id=session_id)

    async def run_turn(
        self,
        prompt: str,
        *,
        event_summaries: list[DomainTweakAdkEventSummary],
    ) -> DomainTweakAdkTurn:
        from google.genai import types

        content = types.Content(role="user", parts=[types.Part(text=prompt)])
        final_text = ""
        events: list[DomainTweakAdkEventSummary] = []
        async for event in self._runner.run_async(
            user_id=self._user_id,
            session_id=self._session_id,
            new_message=content,
        ):
            summary = summarize_adk_event(event)
            events.append(summary)
            event_summaries.append(summary)
            if summary.is_final_response:
                final_text = final_text_from_event(event)
        return DomainTweakAdkTurn(final_text=final_text, events=tuple(events))

    async def close(self) -> None:
        close = getattr(self._runner, "close", None)
        if not callable(close):
            return
        result = close()
        if inspect.isawaitable(result):
            await result


def _adk_tools_for_phase(phase: DomainTweakAdkPhase) -> list[Any]:
    from google.adk.tools import FunctionTool
    from google.adk.tools.google_search_tool import GoogleSearchTool

    tools: list[Any] = [GoogleSearchTool(bypass_multi_tools_limit=True)]
    if phase == "reference_answer":
        tools.append(FunctionTool(citation_formatter))
    return tools


def citation_formatter(url: str, title: str, note: str) -> dict[str, str]:
    """Return a normalized reference-answer citation object."""
    return {"url": url.strip(), "title": title.strip(), "note": note.strip()}


def _agent_name(phase: DomainTweakAdkPhase) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", f"domain_tweak_{phase}")


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 2)


__all__ = [
    "DomainTweakAdkRunner",
    "DomainTweakAdkTurn",
    "DomainTweakAdkTurnExecutor",
]
