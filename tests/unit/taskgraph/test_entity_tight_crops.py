"""Tight-crop rendering for EntitySet visuals (entity_tight_crops mode)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from sat_rs_vlm.taskgraph.input_composer import InputComposer
from sat_rs_vlm.taskgraph.runtime_types import Entity, EntitySet, ImageRef, Region

WIDTH, HEIGHT = 320, 240


def _make_image(tmp_path: Path) -> ImageRef:
    path = tmp_path / "scene.png"
    Image.new("RGB", (WIDTH, HEIGHT), (120, 140, 160)).save(path)
    return ImageRef(str(path), width=WIDTH, height=HEIGHT)


def _entities(image: ImageRef) -> EntitySet:
    return EntitySet(
        (
            Entity(Region(image, (10, 10, 40, 40), {}), label="building", score=0.8),
            Entity(Region(image, (200, 150, 230, 190), {}), label="building", score=0.5),
        ),
        {"provider": "fake_detector", "resolution_status": "MULTIPLE_VALID"},
    )


def test_tight_crops_render_one_enlarged_crop_per_candidate(tmp_path: Path) -> None:
    image = _make_image(tmp_path)
    composer = InputComposer(
        tmp_path / "tight",
        entity_tight_crops=True,
        entity_tight_min_side=512,
        entity_tight_max_visuals=4,
        candidate_halo_ratio=0.2,
    )
    visuals, meta = composer._entity_set_visuals_with_metadata(_entities(image))

    assert meta["strategy"] == "tight_per_candidate"
    assert len(visuals) == 2
    assert meta["selected_candidate_indices"] == [0, 1]
    for visual in visuals:
        # visuals are materialized artifact paths; verify on-disk dimensions
        with Image.open(visual) as rendered:
            assert min(rendered.size) >= 512
            assert max(rendered.size) <= 1536
    mapping = meta["visual_entity_map"]
    assert [item["visual"] for item in mapping] == [1, 2]
    assert [item["index"] for item in mapping] == [0, 1]
    composer.close()


def test_tight_crops_rank_and_cap_visual_count(tmp_path: Path) -> None:
    image = _make_image(tmp_path)
    entities = EntitySet(
        tuple(
            Entity(Region(image, (10 + i * 30, 10, 40 + i * 30, 40), {}), label="x", score=1.0 - i * 0.1)
            for i in range(6)
        ),
        {},
    )
    composer = InputComposer(
        tmp_path / "tight2",
        entity_tight_crops=True,
        entity_tight_max_visuals=3,
    )
    visuals, meta = composer._entity_set_visuals_with_metadata(entities)
    assert len(visuals) == 3
    # highest score first
    assert meta["selected_candidate_indices"] == [0, 1, 2]
    assert meta["omitted_candidate_indices"] == [3, 4, 5]
    composer.close()


def test_tight_crops_structured_context_maps_visuals(tmp_path: Path) -> None:
    image = _make_image(tmp_path)
    composer = InputComposer(
        tmp_path / "tight3",
        entity_tight_crops=True,
        entity_tight_max_visuals=2,
    )
    model_input = composer.compose(
        [_entities(image)],
        question="What color are the buildings?",
        options=["(A) red", "(B) blue"],
    )
    text = model_input.structured_context
    assert "[VISUAL_MAP]" in text
    assert "Visual 1: candidate" in text
    assert "Visual 2: candidate" in text
    assert len(model_input.visual_inputs) == 2
    composer.close()
