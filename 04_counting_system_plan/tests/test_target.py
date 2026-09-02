from counting_system.target import build_target, extract_target_from_question, iter_prompt_variants


def test_extract_how_many_ship():
    spec = extract_target_from_question("How many ships are next to the harbor?")
    assert spec.name in {"ship", "boat"}
    assert spec.tiny is True


def test_extract_airplanes():
    spec = extract_target_from_question("How many airplanes are there in the image?")
    assert spec.name == "airplane"


def test_prompt_variants_include_synonyms():
    variants = list(iter_prompt_variants(build_target("ship")))
    assert "ship" in variants
    assert any("vessel" in v for v in variants)


def test_building_is_not_tiny():
    spec = build_target("building")
    assert spec.tiny is False
