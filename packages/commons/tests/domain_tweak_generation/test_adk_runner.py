from __future__ import annotations

import asyncio

import pytest
from google.adk.agents import Agent
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.google_search_agent_tool import GoogleSearchAgentTool
from google.adk.tools.google_search_tool import GoogleSearchTool
from pydantic import ValidationError

import harnyx_commons.domain_tweak_generation.adk_runner as adk_runner_mod
from harnyx_commons.domain_tweak_generation import (
    DomainTweakAdkEventSummary,
    DomainTweakAdkRunConfig,
    DomainTweakAdkRunner,
    DomainTweakValidationOutcome,
)
from harnyx_commons.domain_tweak_generation.adk_runner import _adk_tools_for_phase
from harnyx_commons.llm.schema import LlmUsage

pytestmark = pytest.mark.anyio("asyncio")


class _FakeTurnExecutor:
    async def __call__(self, **kwargs: object) -> adk_runner_mod.DomainTweakAdkTurn:
        _ = kwargs
        return adk_runner_mod.DomainTweakAdkTurn(final_text="{}", events=())


class _RecordingTurnExecutor:
    def __init__(self, *, responses: tuple[str | BaseException, ...] | None = None) -> None:
        self._responses = list(responses or ("{}",))
        self.prompts: list[str] = []
        self.attempt_indexes: list[int] = []

    async def __call__(self, **kwargs: object) -> adk_runner_mod.DomainTweakAdkTurn:
        self.prompts.append(str(kwargs["prompt"]))
        self.attempt_indexes.append(int(kwargs["attempt_index"]))
        response = self._responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return adk_runner_mod.DomainTweakAdkTurn(
            final_text=response,
            events=(
                DomainTweakAdkEventSummary(
                    is_final_response=True,
                    usage=LlmUsage(prompt_tokens=7, completion_tokens=2, total_tokens=9),
                ),
            ),
        )


class _PartialEventTimeoutContext:
    closed = False

    async def run_turn(
        self,
        prompt: str,
        *,
        event_summaries: list[DomainTweakAdkEventSummary],
    ) -> adk_runner_mod.DomainTweakAdkTurn:
        _ = prompt
        event_summaries.append(_partial_event_summary())
        await asyncio.sleep(60)
        return adk_runner_mod.DomainTweakAdkTurn(final_text="", events=tuple(event_summaries))

    async def close(self) -> None:
        self.closed = True


class _NoEventRequestSetupErrorContext:
    closed = False

    async def run_turn(
        self,
        prompt: str,
        *,
        event_summaries: list[DomainTweakAdkEventSummary],
    ) -> adk_runner_mod.DomainTweakAdkTurn:
        _ = (prompt, event_summaries)
        raise RuntimeError("missing ADC before request stream")

    async def close(self) -> None:
        self.closed = True


class _PartialEventErrorContext:
    closed = False

    async def run_turn(
        self,
        prompt: str,
        *,
        event_summaries: list[DomainTweakAdkEventSummary],
    ) -> adk_runner_mod.DomainTweakAdkTurn:
        _ = prompt
        event_summaries.append(_partial_event_summary())
        raise RuntimeError("stream failed after search")

    async def close(self) -> None:
        self.closed = True


async def test_live_adk_setup_errors_propagate_before_phase_retry_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fail_create(**kwargs: object) -> object:
        _ = kwargs
        raise RuntimeError("missing Vertex credentials")

    monkeypatch.setattr(adk_runner_mod._LiveAdkContext, "create", fail_create)
    runner = DomainTweakAdkRunner()

    with pytest.raises(RuntimeError, match="missing Vertex credentials"):
        await runner.run_phase(
            phase="question_generation",
            prompt="Generate one question.",
            config=DomainTweakAdkRunConfig(model="gemini-3.1-pro-preview"),
            validate=lambda text: DomainTweakValidationOutcome(
                ok=True,
                terminal_status="validated",
                parsed_output=None,
            ),
        )


async def test_phase_timeout_budget_is_shared_across_validation_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values = iter((100.0, 100.0, 103.0, 104.0))
    wait_for_timeouts: list[float | None] = []
    validation_calls = 0

    async def fake_wait_for(coro: object, *, timeout: float | None = None) -> object:
        wait_for_timeouts.append(timeout)
        return await coro

    def validate(_: str) -> DomainTweakValidationOutcome:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            return DomainTweakValidationOutcome(
                ok=False,
                terminal_status="validation_failed",
                feedback=("retry with a corrected response",),
            )
        return DomainTweakValidationOutcome(ok=True, terminal_status="validated")

    monkeypatch.setattr(adk_runner_mod.time, "perf_counter", lambda: next(clock_values))
    monkeypatch.setattr(adk_runner_mod.asyncio, "wait_for", fake_wait_for)

    result = await DomainTweakAdkRunner(turn_executor=_FakeTurnExecutor()).run_phase(
        phase="reference_answer",
        prompt="Answer the question.",
        config=DomainTweakAdkRunConfig(
            model="gemini-3.1-pro-preview",
            max_retries=1,
            phase_timeout_seconds=5.0,
        ),
        validate=validate,
    )

    assert result.terminal_status == "validated"
    assert wait_for_timeouts == [5.0, 2.0]


async def test_soft_timeout_retries_with_time_pressure_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values = iter((100.0, 100.0, 102.0, 102.0, 103.0))
    wait_for_timeouts: list[float | None] = []
    executor = _RecordingTurnExecutor()

    async def fake_wait_for(coro: object, *, timeout: float | None = None) -> object:
        wait_for_timeouts.append(timeout)
        if len(wait_for_timeouts) == 1:
            close = getattr(coro, "close", None)
            if callable(close):
                close()
            raise TimeoutError("soft timeout elapsed")
        return await coro

    monkeypatch.setattr(adk_runner_mod.time, "perf_counter", lambda: next(clock_values))
    monkeypatch.setattr(adk_runner_mod.asyncio, "wait_for", fake_wait_for)

    result = await DomainTweakAdkRunner(turn_executor=executor).run_phase(
        phase="reference_answer",
        prompt="Answer the question.",
        config=DomainTweakAdkRunConfig(
            model="gemini-3.1-pro-preview",
            max_retries=1,
            phase_timeout_seconds=10.0,
            soft_timeout_seconds=2.0,
        ),
        validate=lambda text: DomainTweakValidationOutcome(
            ok=True,
            terminal_status="validated",
            parsed_output=None,
        ),
    )

    assert result.terminal_status == "validated"
    assert wait_for_timeouts[0] == pytest.approx(2.0, abs=0.01)
    assert result.attempts[0].prompt_kind == "initial"
    assert result.attempts[0].validation_ok is False
    assert result.attempts[0].validation_feedback == adk_runner_mod.SOFT_TIMEOUT_FEEDBACK
    assert result.attempts[1].prompt_kind == "soft_timeout_feedback"
    assert executor.attempt_indexes == [1]
    assert "Time is almost gone" in executor.prompts[0]
    assert "Do not restart broad research" in executor.prompts[0]


async def test_early_provider_timeout_does_not_trigger_soft_timeout_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values = iter((100.0, 100.0, 101.0, 102.0))
    wait_for_timeouts: list[float | None] = []
    executor = _RecordingTurnExecutor(responses=(TimeoutError("provider early timeout"),))

    async def fake_wait_for(coro: object, *, timeout: float | None = None) -> object:
        wait_for_timeouts.append(timeout)
        return await coro

    monkeypatch.setattr(adk_runner_mod.time, "perf_counter", lambda: next(clock_values))
    monkeypatch.setattr(adk_runner_mod.asyncio, "wait_for", fake_wait_for)

    result = await DomainTweakAdkRunner(turn_executor=executor).run_phase(
        phase="reference_answer",
        prompt="Answer the question.",
        config=DomainTweakAdkRunConfig(
            model="gemini-3.1-pro-preview",
            max_retries=1,
            phase_timeout_seconds=1800.0,
            soft_timeout_seconds=600.0,
            soft_timeout_interval_seconds=300.0,
        ),
        validate=lambda text: DomainTweakValidationOutcome(ok=True, terminal_status="validated"),
    )

    assert result.terminal_status == "timeout"
    assert wait_for_timeouts == [600.0]
    assert executor.attempt_indexes == [0]
    assert len(result.attempts) == 1
    assert result.attempts[0].prompt_kind == "initial"
    assert result.attempts[0].validation_feedback == ("provider early timeout",)


async def test_soft_timeout_feedback_repeats_until_hard_timeout_with_elapsed_wall_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values = iter((1000.0, 1000.0, 1600.0, 1600.0, 1900.0, 1900.0, 1901.0))
    wait_for_timeouts: list[float | None] = []
    executor = _RecordingTurnExecutor(
        responses=(
            TimeoutError("first pressure point"),
            TimeoutError("second pressure point"),
            "{}",
        )
    )

    async def fake_wait_for(coro: object, *, timeout: float | None = None) -> object:
        wait_for_timeouts.append(timeout)
        return await coro

    monkeypatch.setattr(adk_runner_mod.time, "perf_counter", lambda: next(clock_values))
    monkeypatch.setattr(adk_runner_mod.asyncio, "wait_for", fake_wait_for)

    result = await DomainTweakAdkRunner(turn_executor=executor).run_phase(
        phase="reference_answer",
        prompt="Answer the question.",
        config=DomainTweakAdkRunConfig(
            model="gemini-3.1-pro-preview",
            max_retries=1,
            phase_timeout_seconds=1800.0,
            soft_timeout_seconds=600.0,
            soft_timeout_interval_seconds=300.0,
        ),
        validate=lambda text: DomainTweakValidationOutcome(
            ok=True,
            terminal_status="validated",
            parsed_output=None,
        ),
    )

    assert result.terminal_status == "validated"
    assert wait_for_timeouts == [600.0, 300.0, 300.0]
    assert executor.attempt_indexes == [0, 1, 2]
    assert "Elapsed wall time: 10 minutes (600 seconds)." in executor.prompts[1]
    assert "Elapsed wall time: 15 minutes (900 seconds)." in executor.prompts[2]
    assert result.attempts[0].prompt_kind == "initial"
    assert result.attempts[1].prompt_kind == "soft_timeout_feedback"
    assert result.attempts[2].prompt_kind == "soft_timeout_feedback"


async def test_soft_timeout_feedback_does_not_consume_validation_retry_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values = iter((100.0, 100.0, 700.0, 700.0, 701.0, 702.0))
    wait_for_timeouts: list[float | None] = []
    executor = _RecordingTurnExecutor(
        responses=(
            TimeoutError("pressure point"),
            '{"ok": false}',
            '{"ok": true}',
        )
    )
    validation_calls = 0

    async def fake_wait_for(coro: object, *, timeout: float | None = None) -> object:
        wait_for_timeouts.append(timeout)
        return await coro

    def validate(_: str) -> DomainTweakValidationOutcome:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            return DomainTweakValidationOutcome(
                ok=False,
                terminal_status="validation_failed",
                feedback=("retry with deterministic validation feedback",),
            )
        return DomainTweakValidationOutcome(ok=True, terminal_status="validated")

    monkeypatch.setattr(adk_runner_mod.time, "perf_counter", lambda: next(clock_values))
    monkeypatch.setattr(adk_runner_mod.asyncio, "wait_for", fake_wait_for)

    result = await DomainTweakAdkRunner(turn_executor=executor).run_phase(
        phase="reference_answer",
        prompt="Answer the question.",
        config=DomainTweakAdkRunConfig(
            model="gemini-3.1-pro-preview",
            max_retries=1,
            phase_timeout_seconds=1800.0,
            soft_timeout_seconds=600.0,
            soft_timeout_interval_seconds=300.0,
        ),
        validate=validate,
    )

    assert result.terminal_status == "validated"
    assert wait_for_timeouts == [600.0, 300.0, 299.0]
    assert validation_calls == 2
    assert "Elapsed wall time: 10 minutes (600 seconds)." in executor.prompts[1]
    assert "deterministic validation feedback" in executor.prompts[2]
    assert result.attempts[0].prompt_kind == "initial"
    assert result.attempts[1].prompt_kind == "soft_timeout_feedback"
    assert result.attempts[2].prompt_kind == "feedback"


async def test_soft_timeout_schedule_uses_phase_wall_time_after_validation_feedback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values = iter((100.0, 100.0, 400.0, 401.0))
    wait_for_timeouts: list[float | None] = []
    executor = _RecordingTurnExecutor(responses=('{"ok": false}', '{"ok": true}'))
    validation_calls = 0

    async def fake_wait_for(coro: object, *, timeout: float | None = None) -> object:
        wait_for_timeouts.append(timeout)
        return await coro

    def validate(_: str) -> DomainTweakValidationOutcome:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            return DomainTweakValidationOutcome(
                ok=False,
                terminal_status="validation_failed",
                feedback=("retry with deterministic validation feedback",),
            )
        return DomainTweakValidationOutcome(ok=True, terminal_status="validated")

    monkeypatch.setattr(adk_runner_mod.time, "perf_counter", lambda: next(clock_values))
    monkeypatch.setattr(adk_runner_mod.asyncio, "wait_for", fake_wait_for)

    result = await DomainTweakAdkRunner(turn_executor=executor).run_phase(
        phase="reference_answer",
        prompt="Answer the question.",
        config=DomainTweakAdkRunConfig(
            model="gemini-3.1-pro-preview",
            max_retries=1,
            phase_timeout_seconds=1800.0,
            soft_timeout_seconds=600.0,
            soft_timeout_interval_seconds=300.0,
        ),
        validate=validate,
    )

    assert result.terminal_status == "validated"
    assert wait_for_timeouts == [600.0, 300.0]
    assert result.attempts[0].prompt_kind == "initial"
    assert result.attempts[1].prompt_kind == "feedback"


async def test_late_validation_feedback_uses_next_future_soft_timeout_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values = iter((100.0, 100.0, 710.0, 711.0))
    wait_for_timeouts: list[float | None] = []
    executor = _RecordingTurnExecutor(responses=('{"ok": false}', '{"ok": true}'))
    validation_calls = 0

    async def fake_wait_for(coro: object, *, timeout: float | None = None) -> object:
        wait_for_timeouts.append(timeout)
        return await coro

    def validate(_: str) -> DomainTweakValidationOutcome:
        nonlocal validation_calls
        validation_calls += 1
        if validation_calls == 1:
            return DomainTweakValidationOutcome(
                ok=False,
                terminal_status="validation_failed",
                feedback=("retry with deterministic validation feedback",),
            )
        return DomainTweakValidationOutcome(ok=True, terminal_status="validated")

    monkeypatch.setattr(adk_runner_mod.time, "perf_counter", lambda: next(clock_values))
    monkeypatch.setattr(adk_runner_mod.asyncio, "wait_for", fake_wait_for)

    result = await DomainTweakAdkRunner(turn_executor=executor).run_phase(
        phase="reference_answer",
        prompt="Answer the question.",
        config=DomainTweakAdkRunConfig(
            model="gemini-3.1-pro-preview",
            max_retries=1,
            phase_timeout_seconds=1800.0,
            soft_timeout_seconds=600.0,
            soft_timeout_interval_seconds=300.0,
        ),
        validate=validate,
    )

    assert result.terminal_status == "validated"
    assert wait_for_timeouts == [600.0, 290.0]
    assert "deterministic validation feedback" in executor.prompts[1]
    assert result.attempts[0].prompt_kind == "initial"
    assert result.attempts[1].prompt_kind == "feedback"


async def test_provider_timeout_before_next_future_boundary_stays_terminal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock_values = iter((100.0, 100.0, 710.0, 720.0, 721.0))
    wait_for_timeouts: list[float | None] = []
    executor = _RecordingTurnExecutor(
        responses=('{"ok": false}', TimeoutError("provider timeout before next pressure"))
    )
    validation_calls = 0

    async def fake_wait_for(coro: object, *, timeout: float | None = None) -> object:
        wait_for_timeouts.append(timeout)
        return await coro

    def validate(_: str) -> DomainTweakValidationOutcome:
        nonlocal validation_calls
        validation_calls += 1
        return DomainTweakValidationOutcome(
            ok=False,
            terminal_status="validation_failed",
            feedback=("retry with deterministic validation feedback",),
        )

    monkeypatch.setattr(adk_runner_mod.time, "perf_counter", lambda: next(clock_values))
    monkeypatch.setattr(adk_runner_mod.asyncio, "wait_for", fake_wait_for)

    result = await DomainTweakAdkRunner(turn_executor=executor).run_phase(
        phase="reference_answer",
        prompt="Answer the question.",
        config=DomainTweakAdkRunConfig(
            model="gemini-3.1-pro-preview",
            max_retries=1,
            phase_timeout_seconds=1800.0,
            soft_timeout_seconds=600.0,
            soft_timeout_interval_seconds=300.0,
        ),
        validate=validate,
    )

    assert result.terminal_status == "timeout"
    assert wait_for_timeouts == [600.0, 290.0]
    assert validation_calls == 1
    assert result.attempts[0].prompt_kind == "initial"
    assert result.attempts[1].prompt_kind == "feedback"
    assert result.attempts[1].validation_feedback == ("provider timeout before next pressure",)


def test_adk_run_config_rejects_soft_timeout_without_hard_timeout_budget() -> None:
    with pytest.raises(ValidationError, match="soft_timeout_seconds must be lower"):
        DomainTweakAdkRunConfig(
            model="gemini-3.1-pro-preview",
            phase_timeout_seconds=10.0,
            soft_timeout_seconds=10.0,
        )


def test_adk_run_config_rejects_interval_without_soft_timeout() -> None:
    with pytest.raises(ValidationError, match="soft_timeout_interval_seconds requires soft_timeout_seconds"):
        DomainTweakAdkRunConfig(
            model="gemini-3.1-pro-preview",
            phase_timeout_seconds=1800.0,
            soft_timeout_seconds=None,
            soft_timeout_interval_seconds=300.0,
        )


async def test_live_adk_request_setup_errors_propagate_before_pair_retry_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _NoEventRequestSetupErrorContext()

    async def create_context(**kwargs: object) -> _NoEventRequestSetupErrorContext:
        _ = kwargs
        return context

    monkeypatch.setattr(adk_runner_mod._LiveAdkContext, "create", create_context)

    with pytest.raises(RuntimeError, match="missing ADC before request stream"):
        await DomainTweakAdkRunner().run_phase(
            phase="question_generation",
            prompt="Generate one question.",
            config=DomainTweakAdkRunConfig(
                model="gemini-3.1-pro-preview",
                max_retries=0,
                phase_timeout_seconds=10.0,
            ),
            validate=lambda text: DomainTweakValidationOutcome(ok=True, terminal_status="validated"),
        )

    assert context.closed is True


async def test_timeout_preserves_partial_live_adk_event_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _PartialEventTimeoutContext()

    async def create_context(**kwargs: object) -> _PartialEventTimeoutContext:
        _ = kwargs
        return context

    monkeypatch.setattr(adk_runner_mod._LiveAdkContext, "create", create_context)

    result = await DomainTweakAdkRunner().run_phase(
        phase="question_generation",
        prompt="Generate one question.",
        config=DomainTweakAdkRunConfig(
            model="gemini-3.1-pro-preview",
            max_retries=0,
            phase_timeout_seconds=0.01,
        ),
        validate=lambda text: DomainTweakValidationOutcome(ok=True, terminal_status="validated"),
    )

    assert result.terminal_status == "timeout"
    assert len(result.attempts) == 1
    assert len(result.attempts[0].event_summaries) == 1
    assert result.attempts[0].tool_usage.llm.call_count == 1
    assert result.attempts[0].tool_usage.llm.prompt_tokens == 11
    assert result.attempts[0].tool_usage.search_tool.call_count == 1
    assert result.tool_usage.llm.call_count == 1
    assert result.tool_usage.search_tool.call_count == 1
    assert context.closed is True


async def test_invocation_error_preserves_partial_live_adk_event_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _PartialEventErrorContext()

    async def create_context(**kwargs: object) -> _PartialEventErrorContext:
        _ = kwargs
        return context

    monkeypatch.setattr(adk_runner_mod._LiveAdkContext, "create", create_context)

    result = await DomainTweakAdkRunner().run_phase(
        phase="question_generation",
        prompt="Generate one question.",
        config=DomainTweakAdkRunConfig(
            model="gemini-3.1-pro-preview",
            max_retries=0,
            phase_timeout_seconds=10.0,
        ),
        validate=lambda text: DomainTweakValidationOutcome(ok=True, terminal_status="validated"),
    )

    assert result.terminal_status == "invocation_error"
    assert result.error == "stream failed after search"
    assert len(result.attempts) == 1
    assert len(result.attempts[0].event_summaries) == 1
    assert result.attempts[0].tool_usage.llm.call_count == 1
    assert result.attempts[0].tool_usage.llm.prompt_tokens == 11
    assert result.attempts[0].tool_usage.search_tool.call_count == 1
    assert result.tool_usage.llm.call_count == 1
    assert result.tool_usage.search_tool.call_count == 1
    assert context.closed is True


async def test_live_adk_setup_does_not_apply_repo_owned_model_family_guard() -> None:
    context = await adk_runner_mod._LiveAdkContext.create(
        phase="question_generation",
        config=DomainTweakAdkRunConfig(model="gemini-2.5-pro"),
        agent_instruction="Generate one grounded question.",
    )

    await context.close()


def test_adk_run_config_rejects_removed_formatter_tool_mode() -> None:
    with pytest.raises(ValidationError):
        DomainTweakAdkRunConfig(model="gemini-3.1-pro-preview", tool_mode="search_with_formatter")


def test_adk_tools_for_question_generation_use_supported_google_search_workaround() -> None:
    [search_tool] = _adk_tools_for_phase("question_generation")

    assert isinstance(search_tool, GoogleSearchTool)
    assert search_tool.bypass_multi_tools_limit is True


def test_adk_tools_for_reference_answer_include_formatter_with_search_workaround() -> None:
    tools = _adk_tools_for_phase("reference_answer")

    assert any(isinstance(tool, GoogleSearchTool) and tool.bypass_multi_tools_limit for tool in tools)
    assert any(isinstance(tool, FunctionTool) and tool.name == "citation_formatter" for tool in tools)


def test_adk_tools_for_form_review_do_not_include_formatter() -> None:
    tools = _adk_tools_for_phase("form_review")

    assert any(isinstance(tool, GoogleSearchTool) and tool.bypass_multi_tools_limit for tool in tools)
    assert not any(isinstance(tool, FunctionTool) for tool in tools)


async def test_reference_answer_tools_resolve_through_adk_multi_tool_workaround() -> None:
    agent = Agent(
        name="domain_tweak_reference_answer",
        model="gemini-3.1-pro-preview",
        tools=_adk_tools_for_phase("reference_answer"),
    )

    canonical_tools = await agent.canonical_tools()

    assert any(isinstance(tool, GoogleSearchAgentTool) for tool in canonical_tools)
    assert any(isinstance(tool, FunctionTool) and tool.name == "citation_formatter" for tool in canonical_tools)


def _partial_event_summary() -> DomainTweakAdkEventSummary:
    return DomainTweakAdkEventSummary(
        function_call_names=("google_search_agent",),
        usage=LlmUsage(prompt_tokens=11, completion_tokens=3, total_tokens=14),
        web_search_query_count=1,
    )
