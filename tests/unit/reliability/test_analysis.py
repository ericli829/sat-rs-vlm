from sat_rs_vlm.evaluation.reliability.analysis import summarize_conditions


def test_summarize_conditions_groups_layers_planes_projections_and_bits() -> None:
    rows = [
        {
            "target": "attention",
            "layers": [3],
            "bit_plane": "exponent",
            "changed_rate": 0.5,
            "records": [{"target_name": "model.layers.3.self_attn.q_proj.weight", "bit_index": 12}],
        },
        {
            "target": "attention",
            "layers": [3],
            "bit_plane": "exponent",
            "changed_rate": 1.0,
            "records": [{"target_name": "model.layers.3.self_attn.q_proj.weight", "bit_index": 12}],
        },
    ]
    result = summarize_conditions(rows)
    assert result["groups"]["attention|exponent|3"]["mean"] == 0.75
    assert result["bit_indices"]["12"]["count"] == 2.0
    assert result["projections"]["q_proj"]["mean"] == 0.75
