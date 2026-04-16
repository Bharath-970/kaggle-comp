from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass
class LaneStats:
    lane_name: str
    attempts: int = 0
    reward_sum: float = 0.0

    @property
    def mean_reward(self) -> float:
        if self.attempts == 0:
            return 0.0
        return self.reward_sum / self.attempts


class SubmissionAllocator:
    """UCB-style allocator for exploration vs exploitation under 5/day limit."""

    def __init__(self, exploration_weight: float = 1.4) -> None:
        self.exploration_weight = exploration_weight
        self._stats: dict[str, LaneStats] = {}

    @property
    def stats(self) -> dict[str, LaneStats]:
        return dict(self._stats)

    def register_lane(self, lane_name: str) -> None:
        if lane_name not in self._stats:
            self._stats[lane_name] = LaneStats(lane_name=lane_name)

    def update(self, lane_name: str, reward: float) -> None:
        self.register_lane(lane_name)
        lane = self._stats[lane_name]
        lane.attempts += 1
        lane.reward_sum += reward

    def select_next_lane(self) -> str:
        if not self._stats:
            raise ValueError("No lanes registered")

        total_attempts = sum(max(1, lane.attempts) for lane in self._stats.values())

        best_lane = None
        best_value = float("-inf")
        for lane in self._stats.values():
            if lane.attempts == 0:
                return lane.lane_name
            bonus = self.exploration_weight * math.sqrt(math.log(total_attempts) / lane.attempts)
            ucb = lane.mean_reward + bonus
            if ucb > best_value:
                best_value = ucb
                best_lane = lane.lane_name

        if best_lane is None:  # pragma: no cover
            raise RuntimeError("Failed to select lane")
        return best_lane
