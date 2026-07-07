from sat_rs_vlm.models.reliability.bitflip_simulator import flip_bit_at, flip_random_bit


def test_flip_random_bit_bytes() -> None:
    original = b"\x00"
    flipped = flip_random_bit(original, seed=1)
    assert isinstance(flipped, bytes)
    assert flipped != original


def test_flip_bit_at_int() -> None:
    assert flip_bit_at(0, 3) == 8
