from sat_rs_vlm.configuration.merge import deep_merge, set_dotted_value


def test_deep_merge_recurses_and_replaces_lists() -> None:
    merged = deep_merge(
        {"training": {"steps": 10, "tags": ["base"]}, "seed": 1},
        {"training": {"steps": 2, "tags": ["smoke"]}},
    )

    assert merged == {"training": {"steps": 2, "tags": ["smoke"]}, "seed": 1}


def test_dotted_override_has_highest_priority() -> None:
    config = {"training": {"max_steps": 20}}
    set_dotted_value(config, "training.max_steps", 1)
    assert config["training"]["max_steps"] == 1
