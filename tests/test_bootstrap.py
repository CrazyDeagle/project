from __future__ import annotations

from silexcode.bootstrap import (
    build_bootstrap_chunk,
    generate_bootstrap_chunk,
    generate_bootstrap_record,
)
from silexcode.dataset import ast_whitelist_ok


def test_bootstrap_records_are_deterministic_ascii_and_valid_code() -> None:
    for level in range(5):
        a = generate_bootstrap_record(level, 123)
        b = generate_bootstrap_record(level, 123)
        assert a == b
        assert len(a["token_ids"]) <= 512
        code = a["C"][len("<C>\n") : -len("</C>\n")]
        assert ast_whitelist_ok(code)
        assert all(0 <= token <= 257 for token in a["token_ids"])


def test_bootstrap_chunk_masks_only_targets_by_default() -> None:
    record = generate_bootstrap_record(1, 456)
    chunk = build_bootstrap_chunk([record])
    strict = build_bootstrap_chunk([record], include_padding_loss=True)

    assert len(chunk.token_ids) == 512
    assert len(chunk.labels) == 511
    assert len(chunk.loss_mask) == 511
    assert chunk.target_tokens == sum(chunk.loss_mask)
    assert chunk.target_tokens < strict.target_tokens
    assert chunk.levels == [1]


def test_generate_bootstrap_chunk_packs_multiple_records() -> None:
    chunk = generate_bootstrap_chunk(0, 1000, max_records=16, candidate_multiplier=2)
    assert len(chunk.indices) > 1
    assert set(chunk.levels) == {0}
    assert chunk.target_tokens > 0
