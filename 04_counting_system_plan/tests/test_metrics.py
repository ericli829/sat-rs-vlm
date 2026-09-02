from counting_system.eval.metrics import detection_prf, summarize_counts


def test_summarize_counts():
    metrics = summarize_counts([(3, 3), (4, 3), (None, 2), (1, 1)])
    assert metrics["num_samples"] == 4
    assert metrics["parsed"] == 3
    assert metrics["exact_match"] == 0.5
    assert metrics["exact_accuracy"] == 0.5
    assert metrics["mae"] == 1 / 3


def test_detection_prf_perfect():
    boxes = [(0, 0, 10, 10), (20, 20, 30, 30)]
    scores = detection_prf(boxes, boxes)
    assert scores["precision"] == 1.0
    assert scores["recall"] == 1.0
