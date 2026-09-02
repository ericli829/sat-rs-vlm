from counting_system.data.xlrs_lite import is_counting_row, parse_answer_value, parse_region_name


def test_is_counting_row():
    assert is_counting_row({"l2-category": "Counting__Overall_counting", "question": "How many ships?"})
    assert is_counting_row({"category": "Counting/Regional counting", "question": "How many boats in the circled area?"})
    assert is_counting_row({"l2-category": "Counting__Counting_with_complex_reasoning", "question": "How many groups of containers?"})
    assert is_counting_row({"l2-category": "Counting__Counting_with_changing_detection", "question": "How many vehicles appeared?"})
    assert is_counting_row({"question": "How many airplanes are there?"})
    assert not is_counting_row({"question": "What color is the roof?", "category": "Object_color"})
    assert not is_counting_row(
        {
            "category": "Complex reasoning/Anomaly Detection and Interpretation",
            "question": "How many vehicles are there at the transportation hub in the picture, and what about the parking lots in the commercial district?",
        }
    )


def test_parse_region_and_answer():
    assert parse_region_name("How many cars in the top left of the picture?") == "TOP_LEFT"
    letter, value = parse_answer_value("B", ["2", "3", "4", "5"])
    assert letter == "B"
    assert value == 3


def test_regional_counting_is_not_entire(tmp_path):
    from counting_system.data.xlrs_lite import _row_to_sample

    (tmp_path / "a.jpg").write_bytes(b"x")
    sample = _row_to_sample(
        {
            "question": "How many houses are there inside the red circle?",
            "category": "Counting/Regional counting",
            "image_path": "a.jpg",
            "answer": "A",
            "options": ["3", "4"],
            "entire": True,
        },
        tmp_path,
    )
    assert sample is not None
    assert sample.entire is False
    assert sample.answer_value == 3
