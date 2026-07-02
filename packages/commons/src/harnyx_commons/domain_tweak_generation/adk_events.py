"""ADK event extraction and usage accounting for domain-tweak generation."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

from harnyx_commons.domain.tool_usage import ToolUsageSummary
from harnyx_commons.domain.tool_usage_accounting import tool_usage_from_llm_usage
from harnyx_commons.domain_tweak_generation.types import DomainTweakAdkEventSummary
from harnyx_commons.llm.schema import LlmUsage

_SEARCH_TOOL_NAME = "google_search_agent"
_MAX_QUERY_PREVIEW_COUNT = 10


def summarize_adk_event(event: object) -> DomainTweakAdkEventSummary:
    """Summarize one ADK event without retaining raw event payloads."""
    text = final_text_from_event(event) if _is_final_response(event) else ""
    raw_payload = _model_dump(event)
    web_search_queries = _collect_web_search_queries(raw_payload)
    return DomainTweakAdkEventSummary(
        is_final_response=_is_final_response(event),
        function_call_names=tuple(_event_function_names(event, "get_function_calls")),
        function_response_names=tuple(_event_function_names(event, "get_function_responses")),
        content_text_preview=text[:500] if text else None,
        content_text_length=len(text),
        usage=_usage_from_metadata(_event_usage_metadata(event, raw_payload)),
        web_search_queries=tuple(_bounded_strings(web_search_queries)),
        web_search_query_count=len(web_search_queries),
    )


def final_text_from_event(event: object) -> str:
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        return ""
    return "\n".join(str(text) for text in (_part_text(part) for part in parts) if text)


def merge_event_usage(events: Iterable[DomainTweakAdkEventSummary]) -> LlmUsage:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0
    prompt_cached_tokens = 0
    reasoning_tokens = 0
    search_call_count = 0
    query_count = 0
    for event in events:
        usage = event.usage
        prompt_tokens += int(usage.prompt_tokens or 0)
        completion_tokens += int(usage.completion_tokens or 0)
        total_tokens += int(usage.total_tokens or 0)
        prompt_cached_tokens += int(usage.prompt_cached_tokens or 0)
        reasoning_tokens += int(usage.reasoning_tokens or 0)
        search_call_count += sum(1 for name in event.function_call_names if name == _SEARCH_TOOL_NAME)
        query_count += max(event.web_search_query_count, len(event.web_search_queries))
    return LlmUsage(
        prompt_tokens=prompt_tokens or None,
        completion_tokens=completion_tokens or None,
        total_tokens=total_tokens or None,
        prompt_cached_tokens=prompt_cached_tokens or None,
        reasoning_tokens=reasoning_tokens or None,
        web_search_calls=max(search_call_count, query_count) or None,
    )


def tool_usage_from_adk_events(
    events: Iterable[DomainTweakAdkEventSummary],
    *,
    provider: str,
    model: str,
) -> ToolUsageSummary:
    return tool_usage_from_llm_usage(merge_event_usage(events), provider=provider, model=model)


def _is_final_response(event: object) -> bool:
    is_final = getattr(event, "is_final_response", None)
    if callable(is_final):
        return bool(is_final())
    return bool(is_final)


def _event_function_names(event: object, method_name: str) -> tuple[str, ...]:
    method = getattr(event, method_name, None)
    if not callable(method):
        return ()
    names: list[str] = []
    for item in method() or ():
        name = getattr(item, "name", None)
        if isinstance(name, str) and name:
            names.append(name)
    return tuple(names)


def _part_text(part: object) -> str | None:
    value = getattr(part, "text", None)
    return value if isinstance(value, str) and value else None


def _event_usage_metadata(event: object, payload: Mapping[str, Any] | None) -> object | None:
    usage = getattr(event, "usage_metadata", None)
    if usage is not None:
        return usage
    if payload is None:
        return None
    return payload.get("usage_metadata") or payload.get("usageMetadata")


def _usage_from_metadata(metadata: object | None) -> LlmUsage:
    return LlmUsage(
        prompt_tokens=_metadata_int(metadata, "prompt_token_count", "promptTokenCount"),
        completion_tokens=_metadata_int(metadata, "candidates_token_count", "candidatesTokenCount"),
        total_tokens=_metadata_int(metadata, "total_token_count", "totalTokenCount"),
        prompt_cached_tokens=_metadata_int(
            metadata,
            "cached_content_token_count",
            "cachedContentTokenCount",
        ),
        reasoning_tokens=_metadata_int(metadata, "thoughts_token_count", "thoughtsTokenCount"),
    )


def _metadata_int(metadata: object | None, snake_name: str, camel_name: str) -> int | None:
    if metadata is None:
        return None
    if isinstance(metadata, Mapping):
        metadata_mapping = cast(Mapping[str, object], metadata)
        value = metadata_mapping.get(snake_name)
        if value is None:
            value = metadata_mapping.get(camel_name)
    else:
        value = getattr(metadata, snake_name, None)
        if value is None:
            value = getattr(metadata, camel_name, None)
    return int(value) if isinstance(value, int) else None


def _model_dump(event: object) -> Mapping[str, Any] | None:
    dump = getattr(event, "model_dump", None)
    if callable(dump):
        payload = dump(mode="python")
        return payload if isinstance(payload, Mapping) else None
    return None


def _collect_web_search_queries(value: object) -> tuple[str, ...]:
    queries: list[str] = []

    def _walk(current: object) -> None:
        if isinstance(current, Mapping):
            for key, item in current.items():
                if key in {"web_search_queries", "webSearchQueries"}:
                    queries.extend(_strings(item))
                else:
                    _walk(item)
        elif isinstance(current, list | tuple):
            for item in current:
                _walk(item)

    _walk(value)
    return tuple(queries)


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())


def _bounded_strings(value: object) -> tuple[str, ...]:
    strings: list[str] = []
    for item in _strings(value):
        if len(strings) >= _MAX_QUERY_PREVIEW_COUNT:
            break
        strings.append(item)
    return tuple(strings)


__all__ = [
    "final_text_from_event",
    "merge_event_usage",
    "summarize_adk_event",
    "tool_usage_from_adk_events",
]
