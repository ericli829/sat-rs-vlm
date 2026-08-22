from sat_rs_vlm.models.rs_merger_expert import BASE_EXPERT, COUNTING_EXPERT, route_for_task


def test_routing_uses_only_canonical_task_type():
    assert route_for_task("counting") == COUNTING_EXPERT
    assert route_for_task(" COUNTING ") == COUNTING_EXPERT
    assert route_for_task("detection") == BASE_EXPERT
    assert route_for_task("captioning") == BASE_EXPERT
    assert route_for_task("How many vehicles are visible?") == BASE_EXPERT
