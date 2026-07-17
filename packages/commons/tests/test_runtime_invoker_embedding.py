from __future__ import annotations

import pytest
from pydantic import ValidationError

from harnyx_commons.infrastructure.state.receipt_log import InMemoryReceiptLog
from harnyx_commons.tools.embedding_models import EmbedTextRequest, EmbedTextResponse
from harnyx_commons.tools.executor import ToolInvocationOutput
from harnyx_commons.tools.ports import EmbeddingProviderResult
from harnyx_commons.tools.runtime_invoker import RuntimeToolInvoker

pytestmark = pytest.mark.anyio("asyncio")


class _CapturingEmbeddingProvider:
    def __init__(self) -> None:
        self.requests: list[EmbedTextRequest] = []

    async def embed_text(self, request: EmbedTextRequest) -> EmbeddingProviderResult:
        self.requests.append(request)
        return EmbeddingProviderResult(
            response=EmbedTextResponse.model_validate(
                {
                    "provider": request.provider,
                    "model": request.model,
                    "input_type": request.input_type,
                    "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
                    "dimensions": 3,
                    "usage": {"prompt_tokens": 8, "total_tokens": 8},
                }
            ),
            actual_cost_usd=0.00000008,
            actual_cost_provider=request.provider,
            actual_cost_evidence={"settlement_source": "test"},
        )

    async def aclose(self) -> None:
        return None


class _UnavailableCostEmbeddingProvider(_CapturingEmbeddingProvider):
    async def embed_text(self, request: EmbedTextRequest) -> EmbeddingProviderResult:
        result = await super().embed_text(request)
        return EmbeddingProviderResult(
            response=result.response,
            actual_cost_usd=None,
            actual_cost_provider="openrouter",
            actual_cost_evidence={
                "settlement_source": "unavailable",
                "upstream_model": "Qwen/Qwen3-Embedding-8B",
                "provider_request_id": "gen-emb-unavailable",
            },
        )


async def test_runtime_invoker_lowers_openrouter_embedding_provider_extra() -> None:
    embedding_provider = _CapturingEmbeddingProvider()
    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        embedding_provider=embedding_provider,
        embedding_provider_name="openrouter",
    )

    output = await invoker.invoke(
        "embed_text",
        args=(),
        kwargs={
            "provider": "openrouter",
            "model": "qwen/qwen3-embedding-8b",
            "texts": ["What is Harnyx?"],
            "input_type": "query",
            "provider_extra": {"provider": {"only": ["nebius"], "allow_fallbacks": False}},
        },
    )

    assert isinstance(output, ToolInvocationOutput)
    assert len(embedding_provider.requests) == 1
    request = embedding_provider.requests[0]
    assert request.provider_extra is not None
    assert request.provider_extra.to_request_extra() == {
        "provider": {"only": ["nebius"], "allow_fallbacks": False}
    }
    assert output.actual_cost_provider == "openrouter"


async def test_runtime_invoker_rejects_chutes_embedding_provider_extra() -> None:
    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        embedding_provider=_CapturingEmbeddingProvider(),
        embedding_provider_name="chutes",
    )

    with pytest.raises(ValidationError):
        await invoker.invoke(
            "embed_text",
            args=(),
            kwargs={
                "provider": "chutes",
                "model": "Qwen/Qwen3-Embedding-8B-TEE",
                "texts": ["What is Harnyx?"],
                "input_type": "query",
                "provider_extra": {"provider": {"only": ["nebius"]}},
            },
        )


async def test_runtime_invoker_preserves_openrouter_embedding_when_cost_is_unavailable() -> None:
    invoker = RuntimeToolInvoker(
        InMemoryReceiptLog(),
        embedding_provider=_UnavailableCostEmbeddingProvider(),
        embedding_provider_name="openrouter",
    )

    output = await invoker.invoke(
        "embed_text",
        args=(),
        kwargs={
            "provider": "openrouter",
            "model": "qwen/qwen3-embedding-8b",
            "texts": ["What is Harnyx?"],
            "input_type": "query",
        },
    )

    assert isinstance(output, ToolInvocationOutput)
    assert output.public_payload["data"] == [{"index": 0, "embedding": [0.1, 0.2, 0.3]}]
    assert output.actual_cost_usd is None
    assert output.actual_cost_provider == "openrouter"
    assert output.actual_cost_evidence == {
        "settlement_source": "unavailable",
        "upstream_model": "Qwen/Qwen3-Embedding-8B",
        "provider_request_id": "gen-emb-unavailable",
    }
