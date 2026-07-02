from __future__ import annotations

import pytest

from harnyx_commons.domain_tweak_generation.adk_events import (
    merge_event_usage,
    summarize_adk_event,
    tool_usage_from_adk_events,
)


def test_summarize_adk_event_extracts_final_text_tools_usage_and_grounding_queries() -> None:
    event = _Event(
        final=True,
        text="Final answer",
        calls=("google_search_agent", "source_lookup_tool"),
        responses=("google_search_agent",),
        usage={
            "prompt_token_count": 100,
            "cached_content_token_count": 20,
            "candidates_token_count": 40,
            "total_token_count": 140,
            "thoughts_token_count": 7,
        },
        payload={"candidate": {"grounding_metadata": {"web_search_queries": ["query one", "query two"]}}},
    )

    summary = summarize_adk_event(event)

    assert summary.is_final_response is True
    assert summary.content_text_preview == "Final answer"
    assert summary.function_call_names == ("google_search_agent", "source_lookup_tool")
    assert summary.function_response_names == ("google_search_agent",)
    assert summary.usage.prompt_tokens == 100
    assert summary.usage.prompt_cached_tokens == 20
    assert summary.usage.reasoning_tokens == 7
    assert summary.web_search_queries == ("query one", "query two")
    assert summary.web_search_query_count == 2


def test_summarize_adk_event_counts_queries_before_truncating_preview() -> None:
    queries = [f"query {index}" for index in range(12)]

    summary = summarize_adk_event(
        _Event(payload={"grounding_metadata": {"web_search_queries": queries}})
    )
    usage = merge_event_usage((summary,))

    assert summary.web_search_queries == tuple(queries[:10])
    assert summary.web_search_query_count == 12
    assert usage.web_search_calls == 12


def test_merge_event_usage_uses_max_of_grounding_queries_and_search_function_calls() -> None:
    first = summarize_adk_event(
        _Event(
            calls=("google_search_agent",),
            usage={"prompt_token_count": 10, "candidates_token_count": 4, "total_token_count": 14},
        )
    )
    second = summarize_adk_event(
        _Event(
            payload={"grounding_metadata": {"web_search_queries": ["a", "b", "c"]}},
            usage={"prompt_token_count": 5, "candidates_token_count": 2, "total_token_count": 7},
        )
    )

    usage = merge_event_usage((first, second))
    tool_usage = tool_usage_from_adk_events((first, second), provider="vertex", model="unknown-model")

    assert usage.prompt_tokens == 15
    assert usage.completion_tokens == 6
    assert usage.total_tokens == 21
    assert usage.web_search_calls == 3
    assert tool_usage.search_tool.call_count == 3
    assert tool_usage.search_tool_cost == pytest.approx(0.105)


class _Part:
    def __init__(self, text: str) -> None:
        self.text = text


class _Content:
    def __init__(self, text: str) -> None:
        self.parts = [_Part(text)]


class _ToolCall:
    def __init__(self, name: str) -> None:
        self.name = name


class _Event:
    def __init__(
        self,
        *,
        final: bool = False,
        text: str = "",
        calls: tuple[str, ...] = (),
        responses: tuple[str, ...] = (),
        usage: dict[str, int] | None = None,
        payload: dict[str, object] | None = None,
    ) -> None:
        self.content = _Content(text) if text else None
        self.usage_metadata = usage
        self._final = final
        self._calls = calls
        self._responses = responses
        self._payload = payload or {}

    def is_final_response(self) -> bool:
        return self._final

    def get_function_calls(self) -> list[_ToolCall]:
        return [_ToolCall(name) for name in self._calls]

    def get_function_responses(self) -> list[_ToolCall]:
        return [_ToolCall(name) for name in self._responses]

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "python"
        return self._payload
