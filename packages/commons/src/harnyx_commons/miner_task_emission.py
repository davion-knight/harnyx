"""Miner-task champion emission policies."""

from __future__ import annotations

from math import isfinite

OWNER_UID = 0
PARTICIPANT_MINER_EMISSION_WEIGHT = 0.004


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


def compose_participant_emission_weights(registered_participant_uids: tuple[int, ...]) -> dict[int, float]:
    distinct_uids = tuple(dict.fromkeys(uid for uid in registered_participant_uids if uid != OWNER_UID))
    miner_fraction = len(distinct_uids) * PARTICIPANT_MINER_EMISSION_WEIGHT
    if miner_fraction > 1.0:
        raise ValueError("participant emission exceeds total weight")
    return {uid: PARTICIPANT_MINER_EMISSION_WEIGHT for uid in distinct_uids}


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


__all__ = [
    "OWNER_UID",
    "PARTICIPANT_MINER_EMISSION_WEIGHT",
    "apply_miner_emission_cap",
    "compose_champion_weights",
    "compose_emission_weights",
    "compose_participant_emission_weights",
    "owner_fallback_weights",
]
