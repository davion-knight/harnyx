from __future__ import annotations

from harnyx_commons.config.llm import LlmSettings


def test_openrouter_api_key_value_strips_secret() -> None:
    settings = LlmSettings(OPENROUTER_API_KEY=" test-openrouter-key ")

    assert settings.openrouter_api_key_value == "test-openrouter-key"


def test_ai_gateway_api_key_value_strips_secret() -> None:
    settings = LlmSettings(AI_GATEWAY_API_KEY=" test-ai-gateway-key ")

    assert settings.ai_gateway_api_key_value == "test-ai-gateway-key"
