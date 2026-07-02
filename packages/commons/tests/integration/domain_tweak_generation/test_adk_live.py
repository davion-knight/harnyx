from __future__ import annotations

import os

import pytest

from harnyx_commons.config.vertex import VertexSettings
from harnyx_commons.domain_tweak_generation import (
    DomainTweakAdkPhaseResult,
    DomainTweakAdkRunConfig,
    DomainTweakAdkRunner,
)
from harnyx_commons.domain_tweak_generation.validation import validate_question_generation_output
from harnyx_commons.llm.providers.vertex.credentials import cleanup_credentials_file, prepare_credentials

pytestmark = [pytest.mark.integration, pytest.mark.expensive, pytest.mark.anyio("asyncio")]


async def test_adk_live_imports_and_runs_vertex_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    credentials_path = _configure_adk_vertex_environment(monkeypatch)

    runner = DomainTweakAdkRunner()
    try:
        result = await runner.run_phase(
            phase="question_generation",
            prompt=(
                "Return JSON only: "
                '{"question": "Which example entity is named?", "short_answer": "Example", '
                '"solution_plan": "- Read the prompt\\n- Return the named entity"}'
            ),
            config=DomainTweakAdkRunConfig(
                model=os.environ.get("DOMAIN_TWEAK_ADK_LIVE_MODEL", "gemini-3.1-flash-lite"),
                max_retries=1,
                phase_timeout_seconds=120,
            ),
            validate=validate_question_generation_output,
        )
    finally:
        cleanup_credentials_file(credentials_path)

    assert result.terminal_status == "validated", _phase_result_debug(result)
    assert result.attempts


def _phase_result_debug(result: DomainTweakAdkPhaseResult) -> str:
    lines = [f"terminal_status={result.terminal_status}"]
    if result.error_type or result.error:
        lines.append(f"error={result.error_type}: {result.error}")
    for attempt in result.attempts:
        lines.append(
            "attempt "
            f"{attempt.attempt_index} "
            f"prompt_kind={attempt.prompt_kind} "
            f"validation_ok={attempt.validation_ok} "
            f"feedback={list(attempt.validation_feedback)} "
            f"preview={attempt.final_text_preview!r}"
        )
    return "\n".join(lines)


def _configure_adk_vertex_environment(monkeypatch: pytest.MonkeyPatch) -> str | None:
    vertex = VertexSettings()
    project = _required_env_value(vertex.gcp_project_id, "GCP_PROJECT_ID")
    location = _required_env_value(vertex.gcp_location, "GCP_LOCATION")
    credentials_b64 = _required_env_value(vertex.gcp_sa_credential_b64_value, "GCP_SERVICE_ACCOUNT_CREDENTIAL_BASE64")

    _credentials, credentials_path = prepare_credentials(None, credentials_b64)
    if credentials_path is not None:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", credentials_path)
    monkeypatch.setenv("GOOGLE_GENAI_USE_ENTERPRISE", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", project)
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", location)
    return credentials_path


def _required_env_value(value: str | None, name: str) -> str:
    if not value:
        raise AssertionError(f"{name} must be configured for the ADK live smoke")
    return value
