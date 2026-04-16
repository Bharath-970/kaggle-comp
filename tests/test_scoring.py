from neurogolf.scoring import CostBreakdown, max_cost_for_score, score_from_cost, score_from_cost_breakdown


def test_score_from_cost_floor() -> None:
    assert score_from_cost(10**30) == 1.0


def test_inverse_cost_mapping_round_trip() -> None:
    target_score = 14.0
    implied_cost = max_cost_for_score(target_score)
    actual_score = score_from_cost(implied_cost)
    assert abs(actual_score - target_score) < 1e-9


def test_score_from_cost_breakdown() -> None:
    cost = CostBreakdown(parameters=100, memory_bytes=500, macs=1000)
    score = score_from_cost_breakdown(cost)
    assert score > 1.0
    assert score == score_from_cost(1600)
