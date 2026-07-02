"""Deterministic pair-input loading for domain-tweak generation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol
from uuid import UUID

from harnyx_commons.miner_task_benchmark import (
    BenchmarkDatasetItem,
    BenchmarkDatasetSnapshot,
    load_deepresearch9k_l1_snapshot,
    load_deepsearchqa_snapshot,
)
from harnyx_commons.miner_task_generation import DomainTweakPairInput


class DomainTweakPairInputSource(Protocol):
    async def load_pair_inputs(
        self,
        *,
        batch_id: UUID,
        timestamp: datetime,
        requested_count: int,
    ) -> tuple[DomainTweakPairInput, ...]: ...


class DomainTweakBenchmarkPairInputSource:
    """Builds pair inputs from packaged DeepSearchQA and DeepResearch-9K L1 snapshots."""

    def __init__(
        self,
        *,
        deepsearchqa_snapshot: BenchmarkDatasetSnapshot | None = None,
        deepresearch9k_l1_snapshot: BenchmarkDatasetSnapshot | None = None,
    ) -> None:
        self._deepsearchqa_snapshot = deepsearchqa_snapshot
        self._deepresearch9k_l1_snapshot = deepresearch9k_l1_snapshot

    async def load_pair_inputs(
        self,
        *,
        batch_id: UUID,
        timestamp: datetime,
        requested_count: int,
    ) -> tuple[DomainTweakPairInput, ...]:
        if requested_count <= 0:
            return ()

        deepsearchqa_snapshot = self._deepsearchqa_snapshot or load_deepsearchqa_snapshot()
        deepresearch9k_l1_snapshot = self._deepresearch9k_l1_snapshot or load_deepresearch9k_l1_snapshot()
        _ensure_enough_items(
            suite_name=deepsearchqa_snapshot.manifest.suite_name,
            items=deepsearchqa_snapshot.items,
            requested_count=requested_count,
        )
        _ensure_enough_items(
            suite_name=deepresearch9k_l1_snapshot.manifest.suite_name,
            items=deepresearch9k_l1_snapshot.items,
            requested_count=requested_count,
        )

        deepsearchqa_offset = batch_id.int % len(deepsearchqa_snapshot.items)
        deepresearch9k_l1_offset = (
            batch_id.int // len(deepsearchqa_snapshot.items)
        ) % len(deepresearch9k_l1_snapshot.items)
        return tuple(
            _pair_input(
                deepsearchqa_snapshot=deepsearchqa_snapshot,
                deepsearchqa_item=deepsearchqa_item,
                deepresearch9k_l1_snapshot=deepresearch9k_l1_snapshot,
                deepresearch9k_l1_item=deepresearch9k_l1_item,
                timestamp=timestamp,
            )
            for deepsearchqa_item, deepresearch9k_l1_item in zip(
                _rotated(deepsearchqa_snapshot.items, deepsearchqa_offset, requested_count),
                _rotated(deepresearch9k_l1_snapshot.items, deepresearch9k_l1_offset, requested_count),
                strict=True,
            )
        )


def _ensure_enough_items(
    *,
    suite_name: str,
    items: Sequence[BenchmarkDatasetItem],
    requested_count: int,
) -> None:
    if len(items) < requested_count:
        raise RuntimeError(
            "domain-tweak benchmark pair input source underfilled: "
            f"{suite_name} has {len(items)} items, requested {requested_count}"
        )


def _rotated(
    records: Sequence[BenchmarkDatasetItem],
    offset: int,
    count: int,
) -> tuple[BenchmarkDatasetItem, ...]:
    return tuple(records[(offset + index) % len(records)] for index in range(count))


def _pair_input(
    *,
    deepsearchqa_snapshot: BenchmarkDatasetSnapshot,
    deepsearchqa_item: BenchmarkDatasetItem,
    deepresearch9k_l1_snapshot: BenchmarkDatasetSnapshot,
    deepresearch9k_l1_item: BenchmarkDatasetItem,
    timestamp: datetime,
) -> DomainTweakPairInput:
    return DomainTweakPairInput(
        pair_id=(
            f"{deepsearchqa_snapshot.manifest.suite_slug}:"
            f"{deepsearchqa_snapshot.manifest.dataset_version}:"
            f"{deepsearchqa_item.item_index}__"
            f"{deepresearch9k_l1_snapshot.manifest.suite_slug}:"
            f"{deepresearch9k_l1_snapshot.manifest.dataset_version}:"
            f"{deepresearch9k_l1_item.item_index}"
        ),
        deepsearchqa_form_target=deepsearchqa_item.problem,
        deepresearch9k_domain_target=deepresearch9k_l1_item.problem,
        timestamp=timestamp,
    )


__all__ = ["DomainTweakBenchmarkPairInputSource", "DomainTweakPairInputSource"]
