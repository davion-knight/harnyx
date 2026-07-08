from __future__ import annotations

import os

import pytest

from harnyx_commons.config.llm import LlmSettings
from harnyx_commons.llm.provider_factory import build_miner_paid_llm_provider
from harnyx_commons.llm.schema import LlmMessage, LlmMessageContentPart, LlmRequest, LlmThinkingConfig

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


def _api_key() -> str:
    api_key = os.environ.get("AI_GATEWAY_API_KEY", "").strip()
    assert api_key, "AI_GATEWAY_API_KEY must be configured"
    return api_key


def _request() -> LlmRequest:
    return LlmRequest(
        provider="ai_gateway",
        model="openai/gpt-oss-20b",
        messages=(
            LlmMessage(
                role="user",
                content=(LlmMessageContentPart.input_text('Reply with only "ok".'),),
            ),
        ),
        temperature=0.0,
        max_output_tokens=64,
        thinking=LlmThinkingConfig(enabled=False),
        timeout_seconds=180.0,
        extra={"providerOptions": {"gateway": {"only": ["groq"]}}},
    )


async def test_miner_paid_ai_gateway_groq_selection_live() -> None:
    settings = LlmSettings()
    provider = build_miner_paid_llm_provider(
        provider="ai_gateway",
        api_key=_api_key(),
        llm_settings=settings,
    )
    try:
        response = await provider.invoke(_request())
    finally:
        await provider.aclose()

    assert response.raw_text
    assert response.metadata is not None
    assert response.metadata["actual_cost_provider"] == "ai_gateway"
    assert response.metadata["actual_cost_usd"] >= 0.0
    assert response.metadata["actual_cost_evidence"]["settlement_source"] == "provider_returned"
    raw_response = response.metadata["raw_response"]
    assert raw_response["providerMetadata"]["gateway"]["cost"]
