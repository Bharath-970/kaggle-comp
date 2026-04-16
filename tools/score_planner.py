#!/usr/bin/env python3
"""Score planning utility for NeuroGolf leaderboard strategy."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    avg_score: float
    tasks_needed: int
    feasible: bool


def task_score_from_cost(cost: float) -> float:
    """Competition per-task score from model cost."""
    if cost <= 0:
        raise ValueError("Cost must be positive.")
    return max(1.0, 25.0 - math.log(cost))


def cost_from_task_score(score: float) -> float:
    """Inverse of score function for score > 1."""
    if score <= 1.0:
        raise ValueError("Score must be greater than 1.0 to invert with ln().")
    return math.exp(25.0 - score)


def tasks_needed(target_score: float, avg_score: float) -> int:
    if avg_score <= 0:
        raise ValueError("Average score must be positive.")
    return math.ceil(target_score / avg_score)


def average_needed(target_score: float, solved_tasks: int) -> float:
    if solved_tasks <= 0:
        raise ValueError("Solved tasks must be positive.")
    return target_score / solved_tasks


def parse_float_list(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        values.append(float(part))
    if not values:
        raise ValueError("No numeric values were provided.")
    return values


def parse_portfolio(raw: str) -> list[tuple[int, float]]:
    """Parse entries like '220:16,80:10'."""
    if not raw:
        return []

    items: list[tuple[int, float]] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid portfolio token: {token}")
        left, right = token.split(":", 1)
        items.append((int(left.strip()), float(right.strip())))
    return items


def build_scenarios(target_score: float, avg_scores: list[float], max_tasks: int) -> list[Scenario]:
    scenarios = []
    for avg in avg_scores:
        need = tasks_needed(target_score, avg)
        scenarios.append(Scenario(avg_score=avg, tasks_needed=need, feasible=need <= max_tasks))
    return scenarios


def portfolio_projection(portfolio: list[tuple[int, float]]) -> tuple[int, float]:
    solved = 0
    total_score = 0.0
    for count, avg_score in portfolio:
        solved += count
        total_score += count * avg_score
    return solved, total_score


def print_report(
    target_score: float,
    max_tasks: int,
    current_score: float,
    current_solved: int,
    scenarios: list[Scenario],
    score_markers: list[float],
    portfolio: list[tuple[int, float]],
) -> None:
    print(f"Target score: {target_score:.2f}")
    print(f"Task budget: {max_tasks}")
    print(f"Current score: {current_score:.2f}")
    print(f"Current solved tasks: {current_solved}")
    print(f"Gap to target: {max(0.0, target_score - current_score):.2f}")

    print("\nTasks needed by average score")
    print("avg_score | tasks_needed | feasible")
    print("--------- | ------------ | --------")
    for row in scenarios:
        print(f"{row.avg_score:8.2f} | {row.tasks_needed:12d} | {str(row.feasible):8s}")

    print("\nCost required for per-task score")
    print("score | implied_cost")
    print("----- | -----------")
    for marker in score_markers:
        if marker <= 1.0:
            print(f"{marker:5.2f} | n/a")
            continue
        print(f"{marker:5.2f} | {cost_from_task_score(marker):.0f}")

    needed_avg_all_tasks = average_needed(target_score, max_tasks)
    print(f"\nAverage score needed if all {max_tasks} tasks are solved: {needed_avg_all_tasks:.3f}")

    if portfolio:
        solved, projected_score = portfolio_projection(portfolio)
        projected_avg = (projected_score / solved) if solved else 0.0
        print("\nPortfolio projection")
        print("count:avg_score entries:", ", ".join(f"{c}:{a}" for c, a in portfolio))
        print(f"Projected solved tasks: {solved}")
        print(f"Projected total score: {projected_score:.2f}")
        print(f"Projected average score: {projected_avg:.3f}")
        if solved > max_tasks:
            print("Portfolio status: invalid (solved tasks exceed max task count)")
        else:
            print(f"Portfolio gap to target: {max(0.0, target_score - projected_score):.2f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Leaderboard target planner for NeuroGolf")
    parser.add_argument("--leader", type=float, default=1780.0, help="Current first-place score")
    parser.add_argument("--buffer", type=float, default=2500.0, help="Desired safety buffer")
    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Override direct target score (if set, ignores leader+buffer)",
    )
    parser.add_argument("--max-tasks", type=int, default=400, help="Maximum number of tasks")
    parser.add_argument("--current-score", type=float, default=0.0, help="Your current score")
    parser.add_argument("--current-solved", type=int, default=0, help="Your current solved tasks")
    parser.add_argument(
        "--avg-grid",
        type=str,
        default="11,12,13,14,15,16,17,18",
        help="Comma-separated average per-task scores to evaluate",
    )
    parser.add_argument(
        "--score-markers",
        type=str,
        default="10.7,12,13,14,15,16",
        help="Comma-separated per-task score markers for cost inversion",
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        default="",
        help="Optional projection in count:avg format, e.g. '220:16,80:10'",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    target_score = args.target if args.target is not None else (args.leader + args.buffer)

    avg_scores = parse_float_list(args.avg_grid)
    score_markers = parse_float_list(args.score_markers)
    portfolio = parse_portfolio(args.portfolio)

    scenarios = build_scenarios(target_score, avg_scores, args.max_tasks)

    print_report(
        target_score=target_score,
        max_tasks=args.max_tasks,
        current_score=args.current_score,
        current_solved=args.current_solved,
        scenarios=scenarios,
        score_markers=score_markers,
        portfolio=portfolio,
    )


if __name__ == "__main__":
    main()
