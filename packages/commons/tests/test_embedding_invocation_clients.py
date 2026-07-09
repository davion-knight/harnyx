from __future__ import annotations

import pytest

from harnyx_commons.llm.providers.chutes import ChutesEmbeddingUsage, ChutesTextEmbeddingResponse
from harnyx_commons.llm.providers.openrouter import OpenRouterEmbeddingResponse, OpenRouterEmbeddingUsage
from harnyx_commons.tools.embedding_models import (
    QWEN3_CHUTES_EMBEDDING_MODEL,
    QWEN3_OPENROUTER_EMBEDDING_MODEL,
    EmbedTextRequest,
    parse_miner_selected_embedding_provider_model,
)
from harnyx_commons.tools.invocation_clients import ChutesEmbeddingProvider, OpenRouterEmbeddingProvider

pytestmark = pytest.mark.anyio("asyncio")


def test_miner_selected_embedding_provider_model_sets_are_provider_namespaces() -> None:
    assert parse_miner_selected_embedding_provider_model(
        provider="chutes",
        model=QWEN3_CHUTES_EMBEDDING_MODEL,
    ).model == QWEN3_CHUTES_EMBEDDING_MODEL
    assert parse_miner_selected_embedding_provider_model(
        provider="openrouter",
        model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
    ).model == QWEN3_OPENROUTER_EMBEDDING_MODEL
    with pytest.raises(ValueError, match="not supported"):
        parse_miner_selected_embedding_provider_model(
            provider="chutes",
            model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
        )
    with pytest.raises(ValueError, match="not supported"):
        parse_miner_selected_embedding_provider_model(
            provider="openrouter",
            model=QWEN3_CHUTES_EMBEDDING_MODEL,
        )
async def test_chutes_embedding_provider_formats_query_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        async def embed_many(self, texts: tuple[str, ...]) -> ChutesTextEmbeddingResponse:
            captured["texts"] = texts
            return ChutesTextEmbeddingResponse(
                vectors=((0.1, 0.2, 0.3),),
                usage=ChutesEmbeddingUsage(prompt_tokens=8, total_tokens=8),
            )

    provider = ChutesEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    result = await provider.embed_text(
        EmbedTextRequest(
            provider="chutes",
            model=QWEN3_CHUTES_EMBEDDING_MODEL,
            texts=("find subnet incentives",),
            input_type="query",
            instruction="Given a web search query, retrieve relevant passages that answer the query",
        )
    )

    assert captured["texts"] == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query:find subnet incentives",
    )
    assert result.response.data[0].embedding == [0.1, 0.2, 0.3]
    assert result.actual_cost_provider == "chutes"
    assert result.actual_cost_evidence["usd_per_second"] == pytest.approx(0.0005)


async def test_chutes_embedding_provider_leaves_document_text_unformatted(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        async def embed_many(self, texts: tuple[str, ...]) -> ChutesTextEmbeddingResponse:
            captured["texts"] = texts
            return ChutesTextEmbeddingResponse(
                vectors=((0.4, 0.5, 0.6),),
                usage=ChutesEmbeddingUsage(prompt_tokens=6, total_tokens=6),
            )

    provider = ChutesEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    await provider.embed_text(
        EmbedTextRequest(
            provider="chutes",
            model=QWEN3_CHUTES_EMBEDDING_MODEL,
            texts=("The subnet rewards useful miner answers.",),
            input_type="document",
        )
    )

    assert captured["texts"] == ("The subnet rewards useful miner answers.",)


async def test_chutes_embedding_provider_allows_missing_usage_tokens_when_elapsed_second_priced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        async def embed_many(self, texts: tuple[str, ...]) -> ChutesTextEmbeddingResponse:
            _ = texts
            return ChutesTextEmbeddingResponse(vectors=((0.4, 0.5, 0.6),))

    provider = ChutesEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    result = await provider.embed_text(
        EmbedTextRequest(
            provider="chutes",
            model=QWEN3_CHUTES_EMBEDDING_MODEL,
            texts=("The subnet rewards useful miner answers.",),
            input_type="document",
        )
    )

    assert result.actual_cost_provider == "chutes"
    assert result.actual_cost_evidence["usd_per_second"] == pytest.approx(0.0005)
    assert "input_tokens" not in result.actual_cost_evidence


async def test_openrouter_embedding_provider_posts_native_model_and_settles_static_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        async def embed_many(self, texts: tuple[str, ...]) -> OpenRouterEmbeddingResponse:
            captured["texts"] = texts
            return OpenRouterEmbeddingResponse(
                vectors=((0.7, 0.8, 0.9),),
                usage=OpenRouterEmbeddingUsage(prompt_tokens=12, total_tokens=12),
            )

    provider = OpenRouterEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    result = await provider.embed_text(
        EmbedTextRequest(
            provider="openrouter",
            model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
            texts=("find subnet incentives",),
            input_type="query",
        )
    )

    assert captured["texts"] == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query:find subnet incentives",
    )
    assert result.actual_cost_provider == "openrouter"
    assert result.actual_cost_evidence["input_tokens"] == 12
    assert result.actual_cost_evidence["input_per_million"] == pytest.approx(0.01)


async def test_openrouter_embedding_provider_forwards_provider_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _FakeClient:
        async def embed_many(
            self,
            texts: tuple[str, ...],
            *,
            extra: dict[str, object] | None = None,
        ) -> OpenRouterEmbeddingResponse:
            captured["texts"] = texts
            captured["extra"] = extra
            return OpenRouterEmbeddingResponse(
                vectors=((0.7, 0.8, 0.9),),
                usage=OpenRouterEmbeddingUsage(prompt_tokens=12, total_tokens=12),
            )

    provider = OpenRouterEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    result = await provider.embed_text(
        EmbedTextRequest(
            provider="openrouter",
            model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
            texts=("find subnet incentives",),
            input_type="query",
            provider_extra={"provider": {"only": ["nebius"], "allow_fallbacks": False}},
        )
    )

    assert captured["texts"] == (
        "Instruct: Given a web search query, retrieve relevant passages that answer the query\n"
        "Query:find subnet incentives",
    )
    assert captured["extra"] == {"provider": {"only": ["nebius"], "allow_fallbacks": False}}
    assert result.actual_cost_provider == "openrouter"


async def test_openrouter_embedding_provider_settles_zero_token_cache_hit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        async def embed_many(self, texts: tuple[str, ...]) -> OpenRouterEmbeddingResponse:
            _ = texts
            return OpenRouterEmbeddingResponse(
                vectors=((0.7, 0.8, 0.9),),
                usage=OpenRouterEmbeddingUsage(prompt_tokens=0, total_tokens=0),
            )

    provider = OpenRouterEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    result = await provider.embed_text(
        EmbedTextRequest(
            provider="openrouter",
            model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
            texts=("find subnet incentives",),
            input_type="query",
        )
    )

    assert result.actual_cost_provider == "openrouter"
    assert result.actual_cost_usd == 0.0
    assert result.actual_cost_evidence["input_tokens"] == 0
    assert result.response.usage is not None
    assert result.response.usage.prompt_tokens == 0
    assert result.response.usage.total_tokens == 0


async def test_openrouter_embedding_provider_rejects_missing_usage_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeClient:
        async def embed_many(self, texts: tuple[str, ...]) -> OpenRouterEmbeddingResponse:
            _ = texts
            return OpenRouterEmbeddingResponse(vectors=((0.7, 0.8, 0.9),))

    provider = OpenRouterEmbeddingProvider(api_key="test-key", timeout_seconds=1.0)
    monkeypatch.setattr(provider, "_client_for", lambda **_: _FakeClient())

    with pytest.raises(RuntimeError, match="missing usage tokens"):
        await provider.embed_text(
            EmbedTextRequest(
                provider="openrouter",
                model=QWEN3_OPENROUTER_EMBEDDING_MODEL,
                texts=("find subnet incentives",),
                input_type="query",
            )
        )
