from taskgraph_lab.tools.build_planner_hard_curriculum import curriculum_factor


def test_curriculum_preserves_easy_and_prioritizes_hard_topologies() -> None:
    assert curriculum_factor("SIMPLE_COUNT", 1) == 1
    assert curriculum_factor("RELATIONAL_COUNT", 4) == 2
    assert curriculum_factor("RELATIONAL_COUNT", 8) == 3
    assert curriculum_factor("RELATIONAL_COUNT", 10) == 4
    assert curriculum_factor("OBJECT_RELATION", 7) == 3
    assert curriculum_factor("ROUTE_PLANNING", 4) == 3
    assert curriculum_factor("ROUTE_PLANNING", 10) == 5
    assert curriculum_factor("COMPLEX_REASONING", 3) == 4

