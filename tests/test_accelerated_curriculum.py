from __future__ import annotations

from silexcode.accelerated import build_packed_sequence_and_mask, generate_packed_chunk
from silexcode.dataset import generate_record
from silexcode.train import build_sequence_and_mask


def test_packed_chunk_preserves_single_record_mask() -> None:
    record = generate_record(1, 123)
    expected_input, expected_labels, expected_mask = build_sequence_and_mask(record, 1)
    chunk = build_packed_sequence_and_mask([record], 1, include_padding_loss=True)

    assert chunk.token_ids[:-1] == expected_input
    assert chunk.labels == expected_labels
    assert chunk.loss_mask == expected_mask
    assert chunk.record_indices == [123]
    assert chunk.target_tokens == sum(expected_mask)


def test_packed_chunk_can_skip_padding_loss() -> None:
    record = generate_record(1, 123)
    strict = build_packed_sequence_and_mask([record], 1, include_padding_loss=True)
    accelerated = build_packed_sequence_and_mask([record], 1, include_padding_loss=False)

    assert accelerated.token_ids == strict.token_ids
    assert accelerated.labels == strict.labels
    assert sum(accelerated.loss_mask) < sum(strict.loss_mask)
    assert accelerated.loss_mask[-1] == 0


def test_packed_chunk_adds_multiple_records_without_overflow() -> None:
    records = [generate_record(1, 10_000 + i) for i in range(32)]
    chunk = build_packed_sequence_and_mask(records, 1)

    assert len(chunk.token_ids) == 512
    assert len(chunk.labels) == 511
    assert len(chunk.loss_mask) == 511
    assert 1 <= len(chunk.record_indices) <= 32
    assert chunk.target_tokens == sum(chunk.loss_mask)
    assert all(0 <= token <= 257 for token in chunk.token_ids)


def test_generate_packed_chunk_is_deterministic() -> None:
    a = generate_packed_chunk(2, 222, max_records=4)
    b = generate_packed_chunk(2, 222, max_records=4)

    assert a == b
    assert a.target_tokens > 0
