from scripts.analyze_retriever_deep import _cluster_permutation_p, _holm, _mcnemar_exact
from scripts.retriever_latency_benchmark import percentile


def test_mcnemar_exact_counts_discordant_pairs() -> None:
    result = _mcnemar_exact([1, 1, 0, 1], [0, 1, 1, 0])
    assert result["a_only"] == 2
    assert result["b_only"] == 1
    assert result["discordant"] == 3
    assert 0.0 <= result["p_value"] <= 1.0


def test_holm_adjustment_is_monotonic_in_sorted_p_values() -> None:
    adjusted = _holm([0.01, 0.04, 0.03])
    assert adjusted == [0.03, 0.06, 0.06]


def test_latency_percentile_uses_nearest_rank() -> None:
    assert percentile([1, 2, 3, 4, 5], 0.90) == 5


def test_cluster_permutation_returns_one_for_identical_models() -> None:
    rows = {"a": {"recall_at_5": 1.0}, "b": {"recall_at_5": 0.0}}
    assert (
        _cluster_permutation_p(
            rows,
            rows,
            {"a": "x", "b": "y"},
            "recall_at_5",
            rounds=100,
        )
        == 1.0
    )
