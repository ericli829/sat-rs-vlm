from scripts.analyze_retriever_gate import select_threshold


def test_select_threshold_respects_target_positive_recall() -> None:
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    assert select_threshold(scores, 1.0) == 0.1
    assert select_threshold(scores, 0.8) == 0.2
