from __future__ import annotations

from harnyx_commons.config.llm import LlmSettings


def test_openrouter_api_key_value_strips_secret() -> None:
    settings = LlmSettings(OPENROUTER_API_KEY=" test-openrouter-key ")

    assert settings.openrouter_api_key_value == "test-openrouter-key"
