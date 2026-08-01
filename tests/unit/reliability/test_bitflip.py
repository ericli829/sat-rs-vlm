import pytest

from sat_rs_vlm.models.reliability.bitflip import (
    flip_random_tensor_bits,
    flip_random_value_bits,
    flip_tensor_bit,
    flip_value_bit,
)


def test_flip_specific_bytes_bit_records_address_without_mutating_input() -> None:
    original = bytearray(b"\x00\x00")
    changed, record = flip_value_bit(original, bit_index=9, target_name="payload")

    assert original == bytearray(b"\x00\x00")
    assert changed == bytearray(b"\x00\x02")
    assert record.target_name == "payload"
    assert record.byte_index == 1
    assert record.bit_index == 1


def test_random_multi_bit_is_reproducible() -> None:
    first, first_records = flip_random_value_bits(b"\x00\x00", num_bits=4, seed=7)
    second, second_records = flip_random_value_bits(b"\x00\x00", num_bits=4, seed=7)

    assert first == second
    assert first_records == second_records
    assert len(first_records) == 4


def test_invalid_value_bit_index_fails() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        flip_value_bit(b"\x00", bit_index=8)


def test_tensor_flip_returns_copy_and_supports_sign_bit() -> None:
    torch = pytest.importorskip("torch")
    original = torch.tensor([1.0, 2.0], dtype=torch.float32)

    changed, record = flip_tensor_bit(original, flat_index=0, bit_index=31)

    assert original.tolist() == [1.0, 2.0]
    assert not torch.equal(original, changed)
    assert record.flat_index == 0
    assert record.bit_index == 31


def test_random_tensor_multi_bit_is_reproducible() -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.arange(8, dtype=torch.bfloat16)

    first, first_records = flip_random_tensor_bits(tensor, num_bits=5, seed=13)
    second, second_records = flip_random_tensor_bits(tensor, num_bits=5, seed=13)

    assert torch.equal(first, second)
    assert first_records == second_records


@pytest.mark.parametrize(("flat_index", "bit_index"), [(2, 0), (0, 8)])
def test_tensor_invalid_indices_fail(flat_index: int, bit_index: int) -> None:
    torch = pytest.importorskip("torch")
    tensor = torch.ones(1, dtype=torch.int8)

    with pytest.raises(ValueError):
        flip_tensor_bit(tensor, flat_index=flat_index, bit_index=bit_index)
