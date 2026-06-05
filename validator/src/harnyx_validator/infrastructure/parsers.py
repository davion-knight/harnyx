"""Shared parsing helpers for validator infrastructure."""

from __future__ import annotations

import json
from collections.abc import Mapping
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from harnyx_commons.domain.miner_task import MinerTask
from harnyx_validator.application.dto.evaluation import MinerTaskBatchSpec, ScriptArtifactSpec

_TRANSPORT_CONFIG = ConfigDict(extra="forbid", frozen=True, strict=True, str_strip_whitespace=True)
_DOMAIN_BATCH_ADAPTER = TypeAdapter(MinerTaskBatchSpec)
_PLATFORM_RESPONSE_ONLY_BATCH_KEYS = frozenset({"champion_artifact_id", "completed_at", "failed_at"})
_PLATFORM_RESPONSE_ONLY_ARTIFACT_KEYS = frozenset({"submitted_at"})


class _TransportScriptArtifact(BaseModel):
    model_config = _TRANSPORT_CONFIG

    uid: int = Field(ge=0)
    artifact_id: UUID
    content_hash: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    miner_hotkey_ss58: str = Field(min_length=1)
    task_retry_count: int = Field(ge=0, le=3)

    def to_domain(self) -> ScriptArtifactSpec:
        return ScriptArtifactSpec(
            uid=self.uid,
            artifact_id=self.artifact_id,
            content_hash=self.content_hash,
            size_bytes=self.size_bytes,
            miner_hotkey_ss58=self.miner_hotkey_ss58,
            task_retry_count=self.task_retry_count,
        )


class _TransportMinerTaskBatch(BaseModel):
    model_config = _TRANSPORT_CONFIG

    batch_id: UUID
    cutoff_at: str = Field(min_length=1)
    created_at: str = Field(min_length=1)
    tasks: tuple[MinerTask, ...] = Field(min_length=1)
    artifacts: tuple[_TransportScriptArtifact, ...] = Field(min_length=1)

    @field_validator("artifacts")
    @classmethod
    def _validate_artifact_ids(
        cls,
        value: tuple[_TransportScriptArtifact, ...],
    ) -> tuple[_TransportScriptArtifact, ...]:
        artifact_ids = tuple(artifact.artifact_id for artifact in value)
        if len(set(artifact_ids)) != len(artifact_ids):
            raise ValueError("batch artifacts must be unique by artifact_id")
        return value

    def to_domain(self) -> MinerTaskBatchSpec:
        return MinerTaskBatchSpec(
            batch_id=self.batch_id,
            cutoff_at=self.cutoff_at,
            created_at=self.created_at,
            tasks=self.tasks,
            artifacts=tuple(artifact.to_domain() for artifact in self.artifacts),
        )


_TRANSPORT_BATCH_ADAPTER = TypeAdapter(_TransportMinerTaskBatch)


def parse_batch(payload: Mapping[str, object]) -> MinerTaskBatchSpec:
    """Normalize raw batch payloads into MinerTaskBatchSpec."""
    normalized_payload = _normalize_batch_payload(payload)
    transport = _TRANSPORT_BATCH_ADAPTER.validate_json(json.dumps(normalized_payload), strict=True)
    return _DOMAIN_BATCH_ADAPTER.validate_python(transport.to_domain(), strict=True)


def _normalize_batch_payload(payload: Mapping[str, object]) -> dict[str, object]:
    normalized_payload = {
        key: value for key, value in payload.items() if key not in _PLATFORM_RESPONSE_ONLY_BATCH_KEYS
    }
    artifacts = normalized_payload.get("artifacts")
    if not isinstance(artifacts, list):
        return normalized_payload
    normalized_payload["artifacts"] = [
        {
            key: value
            for key, value in artifact.items()
            if key not in _PLATFORM_RESPONSE_ONLY_ARTIFACT_KEYS
        }
        if isinstance(artifact, Mapping)
        else artifact
        for artifact in artifacts
    ]
    return normalized_payload


__all__ = ["parse_batch"]
