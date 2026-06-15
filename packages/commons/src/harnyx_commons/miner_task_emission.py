"""Miner-task champion emission policies."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite

OWNER_UID = 0
DEFAULT_MINER_PARTICIPATION_EMISSION = 0.004


class ParticipantEmissionTotalWeightError(ValueError):
    """Raised when participant emission would exceed total weight."""


@dataclass(frozen=True, slots=True)
class ParticipantEmissionScore:
    participant_key: str
    score: float


def compose_champion_weights(champion_uid: int | None) -> dict[int, float]:
    if champion_uid is None:
        return {}
    return {champion_uid: 1.0}


def apply_miner_emission_cap(
    weights: dict[int, float],
    batch_score: float,
    *,
    max_miner_emission_fraction: float,
) -> dict[int, float]:
    if not isfinite(batch_score) or batch_score < 0.0 or batch_score > 1.0:
        raise ValueError("miner task batch score must be between 0.0 and 1.0")
    if (
        not isfinite(max_miner_emission_fraction)
        or max_miner_emission_fraction < 0.0
        or max_miner_emission_fraction > 1.0
    ):
        raise ValueError("max miner emission fraction must be between 0.0 and 1.0")

    base = {uid: weight for uid, weight in weights.items() if uid != OWNER_UID}
    if not base:
        raise ValueError("miner weights are empty")
    total = float(sum(base.values()))
    if total <= 0.0:
        raise ValueError("miner weights must have positive miner total")

    miner_fraction = batch_score * max_miner_emission_fraction
    scaled: dict[int, float] = {
        uid: float(weight) / total * miner_fraction for uid, weight in base.items()
    }
    scaled[OWNER_UID] = 1.0 - miner_fraction
    return scaled


def participant_emission_fraction(
    participant_count: int,
    *,
    miner_participation_emission: float,
) -> float:
    if participant_count < 0:
        raise ValueError("participant count must be non-negative")
    if (
        not isfinite(miner_participation_emission)
        or miner_participation_emission < 0.0
        or miner_participation_emission > 1.0
    ):
        raise ValueError("miner participation emission must be between 0.0 and 1.0")
    miner_fraction = participant_count * miner_participation_emission
    if miner_fraction > 1.0:
        raise ParticipantEmissionTotalWeightError("participant emission exceeds total weight")
    return miner_fraction


def compose_participant_emission_weights(
    registered_participant_uids: tuple[int, ...],
    *,
    miner_participation_emission: float = DEFAULT_MINER_PARTICIPATION_EMISSION,
) -> dict[int, float]:
    distinct_uids = tuple(dict.fromkeys(uid for uid in registered_participant_uids if uid != OWNER_UID))
    participant_emission_fraction(
        len(distinct_uids),
        miner_participation_emission=miner_participation_emission,
    )
    return {uid: miner_participation_emission for uid in distinct_uids}


def compose_tiered_participant_emission_allocations(
    participant_scores: Sequence[ParticipantEmissionScore],
    *,
    miner_participation_emission: float = DEFAULT_MINER_PARTICIPATION_EMISSION,
) -> dict[str, float]:
    _validate_miner_participation_emission(miner_participation_emission)

    distinct_scores: dict[str, float] = {}
    for participant_score in participant_scores:
        if not participant_score.participant_key:
            raise ValueError("participant key must be non-empty")
        score = participant_score.score
        if not isfinite(score) or score < 0.0 or score > 1.0:
            raise ValueError("participant score must be between 0.0 and 1.0")
        existing = distinct_scores.get(participant_score.participant_key)
        if existing is None or score > existing:
            distinct_scores[participant_score.participant_key] = score

    ordered = tuple(
        sorted(
            distinct_scores.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )
    if not ordered:
        return {}

    top_floor = _score_floor(ordered, fraction=0.30)
    middle_floor = _score_floor(ordered, fraction=1.00)
    allocations: dict[str, float] = {}
    for participant_key, score in ordered:
        if score <= 0.0:
            continue
        if score >= top_floor:
            multiplier = 2.0
        elif score >= middle_floor:
            multiplier = 1.0
        else:
            continue
        allocations[participant_key] = miner_participation_emission * multiplier

    _validate_participant_miner_fraction(sum(allocations.values()))
    return allocations


def compose_emission_weights(*components: dict[int, float]) -> dict[int, float]:
    weights: dict[int, float] = {}
    for component in components:
        for uid, weight in component.items():
            if uid == OWNER_UID:
                continue
            weights[uid] = weights.get(uid, 0.0) + weight

    miner_fraction = sum(weights.values())
    if miner_fraction > 1.0:
        raise ValueError("emission exceeds total weight")
    weights[OWNER_UID] = 1.0 - miner_fraction
    return weights


def owner_fallback_weights() -> dict[int, float]:
    return {OWNER_UID: 1.0}


def _score_floor(ordered_scores: tuple[tuple[str, float], ...], *, fraction: float) -> float:
    cutoff_count = ceil(len(ordered_scores) * fraction)
    index = max(0, cutoff_count - 1)
    return ordered_scores[index][1]


def _validate_miner_participation_emission(miner_participation_emission: float) -> None:
    if (
        not isfinite(miner_participation_emission)
        or miner_participation_emission < 0.0
        or miner_participation_emission > 1.0
    ):
        raise ValueError("miner participation emission must be between 0.0 and 1.0")


def _validate_participant_miner_fraction(miner_fraction: float) -> None:
    if miner_fraction > 1.0:
        raise ParticipantEmissionTotalWeightError("participant emission exceeds total weight")


__all__ = [
    "DEFAULT_MINER_PARTICIPATION_EMISSION",
    "OWNER_UID",
    "ParticipantEmissionScore",
    "ParticipantEmissionTotalWeightError",
    "apply_miner_emission_cap",
    "compose_champion_weights",
    "compose_emission_weights",
    "compose_participant_emission_weights",
    "compose_tiered_participant_emission_allocations",
    "owner_fallback_weights",
    "participant_emission_fraction",
]
