from __future__ import annotations

import pytest
from pydantic import ValidationError

from harnyx_miner_sdk.tools.llm_provider_extra import (
    OpenRouterExtra,
    OpenRouterProviderSelection,
    validate_provider_extra,
)


def test_openrouter_provider_extra_accepts_provider_only_selection() -> None:
    parsed = validate_provider_extra(
        provider="openrouter",
        provider_extra={"provider": {"only": ["cerebras"]}},
    )

    assert isinstance(parsed, OpenRouterExtra)
    assert parsed.to_request_extra() == {"provider": {"only": ["cerebras"]}}


def test_openrouter_provider_extra_normalizes_provider_names_without_changing_case() -> None:
    parsed = validate_provider_extra(
        provider="openrouter",
        provider_extra=OpenRouterExtra(provider=OpenRouterProviderSelection(only=(" Cerebras ",))),
    )

    assert parsed is not None
    assert parsed.to_request_extra() == {"provider": {"only": ["Cerebras"]}}


def test_chutes_rejects_provider_extra() -> None:
    with pytest.raises(ValueError, match="provider_extra is not supported for provider 'chutes'"):
        validate_provider_extra(
            provider="chutes",
            provider_extra={"provider": {"only": ["cerebras"]}},
        )


def test_provider_extra_rejects_common_reasoning_field() -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="openrouter",
            provider_extra={"reasoning": {"effort": "high"}},
        )


def test_provider_extra_rejects_common_thinking_field() -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="openrouter",
            provider_extra={"thinking": {"enabled": True}},
        )


def test_openrouter_provider_extra_rejects_unapproved_provider_preferences() -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="openrouter",
            provider_extra={"provider": {"only": ["cerebras"], "allow_fallbacks": False}},
        )


@pytest.mark.parametrize(
    "provider_only",
    ([], [""], ["  "], ["cerebras", 1], "cerebras"),
)
def test_openrouter_provider_extra_rejects_invalid_provider_only_values(provider_only: object) -> None:
    with pytest.raises(ValidationError):
        validate_provider_extra(
            provider="openrouter",
            provider_extra={"provider": {"only": provider_only}},
        )
