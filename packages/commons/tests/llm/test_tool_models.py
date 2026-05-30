from __future__ import annotations

import pytest

from harnyx_commons.llm.tool_models import (
    ALLOWED_TOOL_MODELS,
    MINER_SELECTED_LLM_PROVIDER_MODELS,
    resolve_miner_selected_llm_provider_model,
    resolve_tool_model,
    tool_model_thinking_capability,
)


def test_tool_model_thinking_capabilities_share_the_canonical_model_owner() -> None:
    deepseek = tool_model_thinking_capability("deepseek-ai/deepseek-v3.2-tee", provider_name="chutes")
    glm = tool_model_thinking_capability("zai-org/GLM-5-TEE", provider_name="vertex")
    qwen36 = tool_model_thinking_capability(
        "Qwen/Qwen3.6-27B-TEE",
        provider_name="custom-openai-compatible:qwen36-cloud-run",
    )
    gemma_chutes = tool_model_thinking_capability("google/gemma-4-31B-turbo-TEE", provider_name="chutes")
    gemma_custom = tool_model_thinking_capability(
        "google/gemma-4-31B-turbo-TEE",
        provider_name="custom-openai-compatible:gemma4-cloud-run-turbo",
    )

    assert resolve_tool_model("deepseek-ai/deepseek-v3.2-tee") == "deepseek-ai/DeepSeek-V3.2-TEE"
    assert resolve_tool_model("openai/gpt-oss-20b") == "openai/gpt-oss-20b"
    assert resolve_tool_model("openai/gpt-oss-120b") == "openai/gpt-oss-120b"
    assert resolve_tool_model("qwen/qwen3.6-27b-tee") == "Qwen/Qwen3.6-27B-TEE"
    assert resolve_tool_model("Qwen/Qwen3-Next-80B-A3B-Instruct") is None
    assert resolve_tool_model("deepseek-ai/deepseek-v3.1-tee") is None
    assert deepseek is not None
    assert deepseek.chat_template_kwargs(enabled=True) == {"thinking": True}
    assert glm is not None
    assert glm.chat_template_kwargs(enabled=False) == {"enable_thinking": False}
    assert qwen36 is not None
    assert qwen36.chat_template_kwargs(enabled=False) == {"enable_thinking": False}
    assert gemma_chutes is None
    assert gemma_custom is not None
    assert gemma_custom.chat_template_kwargs(enabled=True) == {"enable_thinking": True}
    assert tool_model_thinking_capability("openai/gpt-oss-20b", provider_name="openrouter") is None
    assert tool_model_thinking_capability("openai/gpt-oss-120b", provider_name="openrouter") is None


def test_miner_selected_chutes_supports_only_chutes_models() -> None:
    assert (
        resolve_miner_selected_llm_provider_model(
            provider="chutes",
            model="deepseek-ai/DeepSeek-V3.2-TEE",
        ).model
        == "deepseek-ai/DeepSeek-V3.2-TEE"
    )
    assert (
        resolve_miner_selected_llm_provider_model(
            provider=" chutes ",
            model="qwen/qwen3.6-27b-tee",
        ).model
        == "Qwen/Qwen3.6-27B-TEE"
    )


def test_miner_selected_chutes_rejects_openrouter_only_models() -> None:
    for model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
        with pytest.raises(ValueError, match="not supported for miner-selected provider 'chutes'"):
            resolve_miner_selected_llm_provider_model(provider="chutes", model=model)


@pytest.mark.parametrize(
    ("model", "expected_model"),
    (
        ("openai/gpt-oss-20b", "openai/gpt-oss-20b"),
        ("openai/gpt-oss-120b", "openai/gpt-oss-120b"),
        ("deepseek/deepseek-v3.2", "deepseek-ai/DeepSeek-V3.2-TEE"),
        ("z-ai/glm-5", "zai-org/GLM-5-TEE"),
        ("qwen/qwen3.6-27b", "Qwen/Qwen3.6-27B-TEE"),
        ("google/gemma-4-31b-it", "google/gemma-4-31B-turbo-TEE"),
    ),
)
def test_miner_selected_openrouter_supports_openrouter_models_and_native_aliases(
    model: str,
    expected_model: str,
) -> None:
    resolved = resolve_miner_selected_llm_provider_model(provider="openrouter", model=model)

    assert resolved.provider == "openrouter"
    assert resolved.model == expected_model


def test_miner_selected_openrouter_supports_every_chutes_model() -> None:
    assert set(MINER_SELECTED_LLM_PROVIDER_MODELS["chutes"]) <= set(
        MINER_SELECTED_LLM_PROVIDER_MODELS["openrouter"]
    )


def test_miner_selected_openrouter_accepts_canonical_chutes_model_ids() -> None:
    for model in MINER_SELECTED_LLM_PROVIDER_MODELS["chutes"]:
        assert resolve_miner_selected_llm_provider_model(provider="openrouter", model=model).model == model


def test_miner_selected_openrouter_supports_openrouter_only_gpt_models() -> None:
    assert (
        resolve_miner_selected_llm_provider_model(
            provider="openrouter",
            model="openai/gpt-oss-20b",
        ).provider
        == "openrouter"
    )


def test_provider_native_aliases_are_not_cross_provider_aliases() -> None:
    with pytest.raises(ValueError, match="model 'qwen/qwen3.6-27b' is not allowed for validator tools"):
        resolve_miner_selected_llm_provider_model(provider="chutes", model="qwen/qwen3.6-27b")


def test_unknown_miner_selected_llm_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="miner-selected llm provider 'vertex' is not supported"):
        resolve_miner_selected_llm_provider_model(provider="vertex", model="openai/gpt-oss-20b")


def test_miner_selected_provider_model_sets_are_subsets_of_allowed_tool_models() -> None:
    for models in MINER_SELECTED_LLM_PROVIDER_MODELS.values():
        assert models
        assert set(models) <= set(ALLOWED_TOOL_MODELS)
