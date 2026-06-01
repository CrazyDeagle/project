from silexcode.budget import (
    assert_tdd_invariants,
    logical_ternary_weight_count,
    memory_budget,
    physical_ternary_pack_bytes,
    plastic_parameter_count,
)
from silexcode.constants import SILEX_T18_6B_R64_CONFIG, s5
from silexcode.tokenizer import ByteLevelTokenizer


def test_tdd_constants() -> None:
    cfg = SILEX_T18_6B_R64_CONFIG
    assert cfg.vocab_size == 258
    assert cfg.layers == 64
    assert cfg.d_model == 4096
    assert cfg.d_ff == 16384
    assert cfg.d_z == 8192
    assert cfg.plastic_rank == 64
    assert s5(4096) == 832
    assert s5(8192) == 1664
    assert s5(16384) == 3296
    assert plastic_parameter_count(cfg) == 67_108_864
    assert logical_ternary_weight_count(cfg) == 18_355_331_072
    assert physical_ternary_pack_bytes(cfg) == 3_720_038_016
    budget = memory_budget(cfg)
    assert budget.total_mb == 7256.25598526001
    assert budget.free_mb == 935.7440147399902
    assert_tdd_invariants(cfg)


def test_byte_level_tokenizer_round_trip_bytes() -> None:
    tok = ByteLevelTokenizer()
    raw = bytes([0, 65, 10, 255])
    assert tok.encode_prompt_bytes(raw) == [256, 0, 65, 10, 255]
    assert tok.encode_train_bytes(raw) == [256, 0, 65, 10, 255, 257]
    assert tok.decode_generated_tokens_to_bytes([256, 0, 65, 10, 255, 257, 66]) == raw


def test_decode_invalid_utf8_replaces() -> None:
    tok = ByteLevelTokenizer()
    assert tok.decode_generated_tokens([0xF0, 0x28, 257]) == "\ufffd("
