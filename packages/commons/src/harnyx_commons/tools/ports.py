"""Ports for external tool providers shared across services."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from harnyx_commons.tools.provider_billing import BillingAwareSearchResponse
from harnyx_commons.tools.search_models import (
    FetchPageRequest,
    FetchPageResponse,
    SearchAiSearchRequest,
    SearchAiSearchResponse,
    SearchWebSearchRequest,
    SearchWebSearchResponse,
    SearchXResult,
    SearchXSearchRequest,
    SearchXSearchResponse,
)


class WebSearchProviderPort(Protocol):
    """Shared provider seam for miner-facing web tools."""

    async def search_web(self, request: SearchWebSearchRequest) -> SearchWebSearchResponse: ...

    async def search_ai(self, request: SearchAiSearchRequest) -> SearchAiSearchResponse: ...

    async def fetch_page(self, request: FetchPageRequest) -> FetchPageResponse: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class BillingAwareWebSearchProviderPort(WebSearchProviderPort, Protocol):
    """Optional search provider seam that preserves internal billing evidence."""

    async def search_web_with_billing(
        self,
        request: SearchWebSearchRequest,
    ) -> BillingAwareSearchResponse[SearchWebSearchResponse]: ...

    async def search_ai_with_billing(
        self,
        request: SearchAiSearchRequest,
    ) -> BillingAwareSearchResponse[SearchAiSearchResponse]: ...

    async def fetch_page_with_billing(
        self,
        request: FetchPageRequest,
    ) -> BillingAwareSearchResponse[FetchPageResponse]: ...


class DeSearchPort(Protocol):
    """Internal DeSearch seam for X-specific helpers."""

    async def search_links_twitter(
        self,
        request: SearchXSearchRequest,
    ) -> SearchXSearchResponse: ...

    async def fetch_twitter_post(self, *, post_id: str) -> SearchXResult | None: ...

__all__ = ["BillingAwareWebSearchProviderPort", "DeSearchPort", "WebSearchProviderPort"]
