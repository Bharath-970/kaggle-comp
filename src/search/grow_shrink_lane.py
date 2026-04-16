from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CandidateState:
    channels: int
    kernel_size: int
    depth: int


def propose_growth_steps(state: CandidateState) -> list[CandidateState]:
    return [
        CandidateState(channels=state.channels + 1, kernel_size=state.kernel_size, depth=state.depth),
        CandidateState(channels=state.channels, kernel_size=min(5, state.kernel_size + 2), depth=state.depth),
        CandidateState(channels=state.channels, kernel_size=state.kernel_size, depth=state.depth + 1),
    ]


def propose_shrink_steps(state: CandidateState) -> list[CandidateState]:
    candidates: list[CandidateState] = []
    if state.channels > 1:
        candidates.append(
            CandidateState(channels=state.channels - 1, kernel_size=state.kernel_size, depth=state.depth)
        )
    if state.kernel_size > 1:
        candidates.append(
            CandidateState(channels=state.channels, kernel_size=max(1, state.kernel_size - 2), depth=state.depth)
        )
    if state.depth > 1:
        candidates.append(
            CandidateState(channels=state.channels, kernel_size=state.kernel_size, depth=state.depth - 1)
        )
    return candidates
