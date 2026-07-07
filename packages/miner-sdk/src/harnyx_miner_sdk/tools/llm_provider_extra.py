"""Provider-specific extras for miner ``llm_chat`` calls."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


def _normalize_non_empty_string_sequence(value: object, *, label: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{label} must be a JSON array")
    if not value:
        raise ValueError(f"{label} must contain at least one entry")

    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{label} entries must be strings")
        provider_name = item.strip()
        if not provider_name:
            raise ValueError(f"{label} entries must be non-empty")
        normalized.append(provider_name)
    return tuple(normalized)


class OpenRouterProviderSelection(BaseModel):
    """OpenRouter provider selection accepted by miner ``llm_chat``."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    only: tuple[str, ...] = Field(min_length=1)

    @field_validator("only", mode="before")
    @classmethod
    def _normalize_only(cls, value: object) -> tuple[str, ...]:
        return _normalize_non_empty_string_sequence(value, label="OpenRouter provider.only")

    def to_provider_payload(self) -> dict[str, Any]:
        return {"only": list(self.only)}


class OpenRouterExtra(BaseModel):
    """Provider-specific extras accepted only for ``provider=\"openrouter\"``."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    provider: OpenRouterProviderSelection

    def to_request_extra(self) -> dict[str, Any]:
        return {"provider": self.provider.to_provider_payload()}


_OPENROUTER_EXTRA_ADAPTER = TypeAdapter(OpenRouterExtra)


def validate_provider_extra(*, provider: str, provider_extra: object) -> OpenRouterExtra | None:
    if provider_extra is None:
        return None
    if provider == "openrouter":
        return _OPENROUTER_EXTRA_ADAPTER.validate_python(provider_extra)
    raise ValueError(f"provider_extra is not supported for provider {provider!r}")


__all__ = [
    "OpenRouterExtra",
    "OpenRouterProviderSelection",
    "validate_provider_extra",
]
