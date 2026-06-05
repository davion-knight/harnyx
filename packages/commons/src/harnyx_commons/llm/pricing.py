"""Pricing helpers for validator tool budgeting.

LLM prices here are reference rates for budgeting miner tool calls. Model rates
follow the configured reference provider for each canonical tool model.
External benchmarking uses its own pricing
(`apps/platform/scripts/miner_task_benchmark.py`) and must not import this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

from harnyx_commons.llm.provider_types import CHUTES_PROVIDER, OPENROUTER_PROVIDER
from harnyx_commons.llm.schema import LlmUsage
from harnyx_commons.llm.tool_models import (
    MINER_SELECTED_LLM_PROVIDER_MODELS,
    MinerSelectedLlmProviderName,
    ToolModelName,
)
from harnyx_commons.tools.types import SearchToolName

# Per-referenceable-result rates for search tools, keyed by tool name.
SEARCH_PRICING_PER_REFERENCEABLE_RESULT: dict[SearchToolName, float] = {
    "search_web": 0.0001,
    "search_ai": 0.0004,
    "fetch_page": 0.0005,
}

PARALLEL_SEARCH_BASE_RESULTS = 10
PARALLEL_SEARCH_BASE_COST_USD = 0.005
PARALLEL_SEARCH_ADDITIONAL_RESULT_COST_USD = 0.001
PARALLEL_EXTRACT_URL_COST_USD = 0.001


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float
    reasoning_per_million: float

    @property
    def billable_reasoning_per_million(self) -> float:
        if self.reasoning_per_million != 0.0:
            return self.reasoning_per_million
        return self.output_per_million


# Reference rates keyed by canonical model id.
MODEL_PRICING: Mapping[ToolModelName, ModelPricing] = {
    "openai/gpt-oss-20b": ModelPricing(0.03, 0.14, 0.0),
    "openai/gpt-oss-120b": ModelPricing(0.039, 0.18, 0.0),
    "deepseek-ai/DeepSeek-V3.2-TEE": ModelPricing(0.28, 0.42, 0.0),
    "zai-org/GLM-5-TEE": ModelPricing(0.95, 2.55, 0.0),
    "Qwen/Qwen3.6-27B-TEE": ModelPricing(0.50, 2.00, 0.0),
    "google/gemma-4-31B-turbo-TEE": ModelPricing(0.13, 0.38, 0.0),
}

MINER_TOOL_LLM_PRICING: Mapping[MinerSelectedLlmProviderName, Mapping[str, ModelPricing]] = {
    CHUTES_PROVIDER: {
        model: MODEL_PRICING[cast(ToolModelName, model)]
        for model in MINER_SELECTED_LLM_PROVIDER_MODELS[CHUTES_PROVIDER]
    },
    OPENROUTER_PROVIDER: {
        "openai/gpt-oss-20b": MODEL_PRICING["openai/gpt-oss-20b"],
        "openai/gpt-oss-120b": MODEL_PRICING["openai/gpt-oss-120b"],
        "deepseek/deepseek-v3.2": MODEL_PRICING["deepseek-ai/DeepSeek-V3.2-TEE"],
        "z-ai/glm-5": MODEL_PRICING["zai-org/GLM-5-TEE"],
        "qwen/qwen3.6-27b": MODEL_PRICING["Qwen/Qwen3.6-27B-TEE"],
        "google/gemma-4-31b-it": MODEL_PRICING["google/gemma-4-31B-turbo-TEE"],
    },
}


def price_llm(model: ToolModelName, usage: LlmUsage) -> float:
    """Return USD cost for a single LLM call using reference pricing."""
    pricing = MODEL_PRICING[model]
    return _price_tokens(pricing, usage)


def price_miner_llm(provider: str, model: str, usage: LlmUsage) -> float:
    """Return USD cost for a miner-selected provider/model LLM call."""
    if provider not in MINER_TOOL_LLM_PRICING:
        raise KeyError(provider)
    pricing_by_model = MINER_TOOL_LLM_PRICING[cast(MinerSelectedLlmProviderName, provider)]
    pricing = pricing_by_model[model]
    return _price_tokens(pricing, usage)


def _price_tokens(pricing: ModelPricing, usage: LlmUsage) -> float:
    """Return USD cost for token usage under the supplied per-model rates."""

    prompt_tokens = float(usage.prompt_tokens or 0)
    completion_tokens = float(usage.completion_tokens or 0)
    reasoning_tokens = float(usage.reasoning_tokens or 0)

    cost_input = (prompt_tokens / 1_000_000) * pricing.input_per_million
    cost_output = (completion_tokens / 1_000_000) * pricing.output_per_million
    cost_reasoning = (reasoning_tokens / 1_000_000) * pricing.billable_reasoning_per_million
    return cost_input + cost_output + cost_reasoning


def price_search(tool_name: SearchToolName, *, referenceable_results: int) -> float:
    """Return USD cost for a search call based on referenceable result count."""
    if referenceable_results < 0:
        raise ValueError("referenceable_results must be non-negative")
    return float(referenceable_results) * SEARCH_PRICING_PER_REFERENCEABLE_RESULT[tool_name]


def price_parallel_search(*, requested_results: int | None) -> float:
    """Return provider-billed USD cost for one Parallel Search request."""
    count = PARALLEL_SEARCH_BASE_RESULTS if requested_results is None else requested_results
    if count < 0:
        raise ValueError("requested_results must be non-negative when supplied")
    extra_results = max(0, count - PARALLEL_SEARCH_BASE_RESULTS)
    return PARALLEL_SEARCH_BASE_COST_USD + (
        float(extra_results) * PARALLEL_SEARCH_ADDITIONAL_RESULT_COST_USD
    )


def price_parallel_extract(*, url_count: int) -> float:
    """Return provider-billed USD cost for one Parallel Extract request."""
    if url_count < 0:
        raise ValueError("url_count must be non-negative")
    return float(url_count) * PARALLEL_EXTRACT_URL_COST_USD


__all__ = [
    "PARALLEL_EXTRACT_URL_COST_USD",
    "PARALLEL_SEARCH_ADDITIONAL_RESULT_COST_USD",
    "PARALLEL_SEARCH_BASE_COST_USD",
    "PARALLEL_SEARCH_BASE_RESULTS",
    "price_llm",
    "price_miner_llm",
    "price_parallel_extract",
    "price_parallel_search",
    "price_search",
    "MODEL_PRICING",
    "MINER_TOOL_LLM_PRICING",
    "SEARCH_PRICING_PER_REFERENCEABLE_RESULT",
    "ModelPricing",
]
