"""Miner-facing tool type definitions shared across host-side code."""

from __future__ import annotations

from harnyx_miner_sdk.tools.types import (
    EMBEDDING_TOOLS,
    LLM_TOOLS,
    SEARCH_TOOLS,
    TOOL_NAMES,
    EmbeddingToolName,
    LlmToolName,
    SearchToolName,
    ToolInvocationTimeout,
    ToolName,
    is_embedding_tool,
    is_search_tool,
    parse_tool_name,
)


def is_citation_source(name: str) -> bool:
    return is_search_tool(name)


__all__ = [
    "ToolInvocationTimeout",
    "ToolName",
    "SearchToolName",
    "EmbeddingToolName",
    "LlmToolName",
    "TOOL_NAMES",
    "SEARCH_TOOLS",
    "EMBEDDING_TOOLS",
    "LLM_TOOLS",
    "parse_tool_name",
    "is_search_tool",
    "is_embedding_tool",
    "is_citation_source",
]
