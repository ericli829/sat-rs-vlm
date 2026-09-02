from counting_system.geometry import local_to_global
from counting_system.runtime import ImageRef
from counting_system.tiling import ownership_core, plan_tiles, scope_from_input
from counting_system.target import build_target


def test_native_tiles_cover_scope():
    image = ImageRef(path="x", width=2048, height=1536)
    _, scope = scope_from_input(image)
    tiles = plan_tiles(
        image,
        scope,
        build_target("ship"),
        {
            "scale": {
                "default_source_scale": 1024,
                "global": {"enabled": False},
                "native": {"enabled": True, "tile_size": 1024, "overlap": 256},
                "fine": {"enabled": False},
            }
        },
        entire=True,
    )
    assert tiles
    assert all(t.scale_id == "native" for t in tiles)
    # 每个 scope 像素中心都应落在某个 core
    misses = 0
    for y in (1, 512, 1024, 1535):
        for x in (1, 512, 1024, 1536, 2047):
            if not any(t.contains_center((x, y)) for t in tiles):
                misses += 1
    assert misses == 0


def test_ownership_core_shrinks_interior():
    scope = (0.0, 0.0, 1000.0, 1000.0)
    crop = (200.0, 200.0, 712.0, 712.0)
    core = ownership_core(crop, scope, overlap=256)
    assert core[0] > crop[0]
    assert core[1] > crop[1]
    assert core[2] < crop[2]
    assert core[3] < crop[3]


def test_border_tile_core_touches_scope():
    scope = (0.0, 0.0, 1000.0, 1000.0)
    crop = (0.0, 0.0, 512.0, 512.0)
    core = ownership_core(crop, scope, overlap=256)
    assert core[0] == 0.0
    assert core[1] == 0.0


def test_local_to_global_with_resize():
    crop = (100.0, 200.0, 356.0, 456.0)  # 256x256
    local = (10.0, 20.0, 40.0, 50.0)
    mapped = local_to_global(local, crop, local_size=(128.0, 128.0))
    # scale = 256/128 = 2
    assert mapped == (100 + 20, 200 + 40, 100 + 80, 200 + 100)
