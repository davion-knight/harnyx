"""Internal protocol for platform-owned assigned artifact work."""

from __future__ import annotations

from typing import Protocol
from uuid import UUID

from harnyx_validator.application.dto.evaluation import MinerTaskWorkAssignment


class AssignedArtifactWork(Protocol):
    """Worker-owned handle for one platform-owned artifact assignment group."""

    async def take_for_startup(self) -> MinerTaskWorkAssignment:
        """Read queued work while the artifact sandbox is still starting."""

    def take_nowait_for_startup(self) -> MinerTaskWorkAssignment:
        """Read queued startup work without claiming task dispatch."""

    def drain_for_setup_failure(self) -> tuple[MinerTaskWorkAssignment, ...]:
        """Drain not-yet-dispatched assignments for setup-failure result creation."""

    def mark_dispatch_ready(self) -> None:
        """Mark the artifact ready to dispatch assigned tasks."""

    def claim_initial_for_dispatch(self, assignment: MinerTaskWorkAssignment) -> bool:
        """Claim a startup-drained assignment before runner scheduling."""

    async def claim_for_dispatch(self) -> MinerTaskWorkAssignment:
        """Wait for and claim a queued assignment for task dispatch."""

    def claim_nowait_for_dispatch(self) -> MinerTaskWorkAssignment:
        """Claim a queued assignment for task dispatch without waiting."""

    def mark_started(self, assignment: MinerTaskWorkAssignment, validator_session_id: UUID) -> bool:
        """Mark an already-dispatched assignment as having an issued session."""


__all__ = ["AssignedArtifactWork"]
