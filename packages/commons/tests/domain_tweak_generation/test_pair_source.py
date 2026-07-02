from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

import pytest

from harnyx_commons.domain_tweak_generation import DomainTweakBenchmarkPairInputSource
from harnyx_commons.miner_task_benchmark import (
    BenchmarkAnswerType,
    BenchmarkDatasetItem,
    BenchmarkDatasetManifest,
    BenchmarkDatasetSnapshot,
)

pytestmark = pytest.mark.anyio("asyncio")

_BATCH_ID = UUID("00000000-0000-0000-0000-000000000002")
_BATCH_CREATED_AT = datetime(2026, 6, 26, 5, 30, tzinfo=UTC)


async def test_benchmark_pair_source_rotates_packaged_snapshot_items_by_batch_id() -> None:
    source = DomainTweakBenchmarkPairInputSource(
        deepsearchqa_snapshot=_snapshot("deepsearchqa", "DeepSearchQA", "2026-04-02", 4),
        deepresearch9k_l1_snapshot=_snapshot("deepresearch9k-l1", "DeepResearch-9K L1", "2026-05-14", 5),
    )

    first = await source.load_pair_inputs(
        batch_id=_BATCH_ID,
        timestamp=_BATCH_CREATED_AT,
        requested_count=3,
    )
    second = await source.load_pair_inputs(
        batch_id=_BATCH_ID,
        timestamp=_BATCH_CREATED_AT,
        requested_count=3,
    )

    assert [item.pair_id for item in first] == [
        "deepsearchqa:2026-04-02:2__deepresearch9k-l1:2026-05-14:0",
        "deepsearchqa:2026-04-02:3__deepresearch9k-l1:2026-05-14:1",
        "deepsearchqa:2026-04-02:0__deepresearch9k-l1:2026-05-14:2",
    ]
    assert [item.pair_id for item in second] == [item.pair_id for item in first]
    assert first[0].deepsearchqa_form_target == "DeepSearchQA problem 2?"
    assert first[0].deepresearch9k_domain_target == "DeepResearch-9K L1 problem 0?"
    assert first[0].timestamp == _BATCH_CREATED_AT


async def test_benchmark_pair_source_rejects_underfilled_snapshots() -> None:
    source = DomainTweakBenchmarkPairInputSource(
        deepsearchqa_snapshot=_snapshot("deepsearchqa", "DeepSearchQA", "2026-04-02", 1),
        deepresearch9k_l1_snapshot=_snapshot("deepresearch9k-l1", "DeepResearch-9K L1", "2026-05-14", 2),
    )

    with pytest.raises(RuntimeError, match="DeepSearchQA has 1 items, requested 2"):
        await source.load_pair_inputs(batch_id=_BATCH_ID, timestamp=_BATCH_CREATED_AT, requested_count=2)


def _snapshot(
    suite_slug: str,
    suite_name: str,
    dataset_version: str,
    item_count: int,
) -> BenchmarkDatasetSnapshot:
    return BenchmarkDatasetSnapshot(
        manifest=BenchmarkDatasetManifest(
            suite_slug=suite_slug,
            suite_name=suite_name,
            dataset_version=dataset_version,
            scoring_version="correctness-v1",
            source_url="https://example.com/source.csv",
            source_page_url="https://example.com",
            license="test",
            sha256="0" * 64,
            row_count=item_count,
            file_name="source.csv",
            fetched_at=dataset_version,
        ),
        items=tuple(_item(suite_name, index) for index in range(item_count)),
    )


def _item(suite_name: str, index: int) -> BenchmarkDatasetItem:
    return BenchmarkDatasetItem(
        item_index=index,
        problem=f"{suite_name} problem {index}?",
        problem_category="test",
        answer=f"answer {index}",
        answer_type=BenchmarkAnswerType.SINGLE_ANSWER,
    )
