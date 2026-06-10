"""Internal provider billing evidence for runtime tool settlement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Generic, Literal, TypeVar

from harnyx_commons.json_types import JsonObject

ProviderBillingSource = Literal[
    "response_body",
    "response_headers",
    "response_results",
    "reference_fallback",
    "live_cache",
    "live_fetch",
    "hard_coded_fallback",
]

TSearchResponse = TypeVar("TSearchResponse", covariant=True)


@dataclass(frozen=True, slots=True)
class ProviderBillingMetadata:
    actual_cost_provider: str
    actual_cost_usd: float | None = None
    billable_units: int | None = None
    provider_request_id: str | None = None
    usage_count: int | None = None
    service: str | None = None
    currency: str | None = None
    source: ProviderBillingSource = "reference_fallback"


@dataclass(frozen=True, slots=True)
class BillingAwareSearchResponse(Generic[TSearchResponse]):
    response: TSearchResponse
    billing: ProviderBillingMetadata | None


def billing_evidence_payload(billing: ProviderBillingMetadata | None) -> JsonObject | None:
    if billing is None:
        return None
    return {
        key: value
        for key, value in asdict(billing).items()
        if value is not None and _is_json_value(value)
    }


def _is_json_value(value: object) -> bool:
    return value is None or isinstance(value, str | int | float | bool | list | dict)


__all__ = [
    "BillingAwareSearchResponse",
    "ProviderBillingMetadata",
    "ProviderBillingSource",
    "billing_evidence_payload",
]
