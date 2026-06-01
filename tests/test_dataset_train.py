import pytest
import torch

from silexcode.constants import SILEX_T18_6B_R64_CONFIG
from silexcode.dataset import (
    RNG,
    ast_whitelist_ok,
    encode_ascii_record,
    generate_record,
    splitmix64_next,
    verify_candidate_code,
)
from silexcode.model import SilexCodeT18_6B_R64, sample_token_top_k_top_p
from silexcode.train import build_sequence_and_mask, compute_depth_losses, stage_ready


def test_splitmix64_and_rng_are_deterministic() -> None:
    assert splitmix64_next(0)[1] == splitmix64_next(0)[1]
    a = RNG(123)
    b = RNG(123)
    assert [a.u64() for _ in range(4)] == [b.u64() for _ in range(4)]


def test_encode_ascii_record_rejects_non_ascii() -> None:
    assert encode_ascii_record("A\n") == [256, 65, 10, 257]
    with pytest.raises(ValueError, match="NON_ASCII_BYTE_FORBIDDEN"):
        encode_ascii_record("á")


def test_generate_record_and_candidate_verifier() -> None:
    for stage in (1, 2, 3):
        record = generate_record(stage, 12345)
        assert record == generate_record(stage, 12345)
        assert len(record["token_ids"]) <= 512
        code = record["C"][len("<C>\n") : -len("</C>\n")]
        ok, signature, canonical_ast = verify_candidate_code(code, record, 16)
        assert ok
        assert signature is not None
        assert canonical_ast is not None


def test_ast_whitelist_blocks_dangerous_code() -> None:
    assert not ast_whitelist_ok("import os\n")
    assert not ast_whitelist_ok("def f(x):\n    return open('x')\n")
    assert not ast_whitelist_ok("def f(x):\n    return x.__class__\n")


def test_build_sequence_and_mask_shapes() -> None:
    for stage in (1, 2, 3):
        record = generate_record(stage, 777)
        input_ids, labels, mask = build_sequence_and_mask(record, stage)
        assert len(input_ids) == 511
        assert len(labels) == 511
        assert len(mask) == 511
        assert sum(mask) > 0


def test_depth_losses_and_thresholds_are_strict() -> None:
    logits = torch.zeros(5, 511, 258)
    labels = torch.zeros(511, dtype=torch.long)
    mask = torch.ones(511)
    depth = compute_depth_losses(logits, labels, mask)
    assert torch.isfinite(depth["nll"])
    assert stage_ready(
        3,
        {
            "nll4": 0.180,
            "mono": 0.0020,
            "latent_gain": 0.040,
            "compile_pass": 0.990,
            "unit_pass": 0.920,
        },
    )
    assert not stage_ready(
        3,
        {
            "nll4": 0.1801,
            "mono": 0.0020,
            "latent_gain": 0.040,
            "compile_pass": 0.990,
            "unit_pass": 0.920,
        },
    )


def test_top_k_top_p_sampler_is_seed_deterministic() -> None:
    logits = torch.zeros(258)
    logits[10] = 4.0
    logits[11] = 3.0
    logits[12] = 2.0
    g1 = torch.Generator(device="cpu").manual_seed(123)
    g2 = torch.Generator(device="cpu").manual_seed(123)
    seq1 = [
        sample_token_top_k_top_p(logits, temperature=0.7, top_p=0.9, top_k=3, generator=g1)
        for _ in range(8)
    ]
    seq2 = [
        sample_token_top_k_top_p(logits, temperature=0.7, top_p=0.9, top_k=3, generator=g2)
        for _ in range(8)
    ]
    assert seq1 == seq2
    assert set(seq1) <= {10, 11, 12}
    assert sample_token_top_k_top_p(logits, temperature=0.0, top_p=1.0, top_k=0, generator=g1) == 10


def test_generate_bytes_uses_forward_native_and_stop_bytes() -> None:
    class Dummy:
        generate_bytes = SilexCodeT18_6B_R64.generate_bytes

        def __init__(self):
            self.config = SILEX_T18_6B_R64_CONFIG
            self.gamma_out = torch.empty((), device="cpu")
            self.tokens = [
                ord("a"),
                ord("b"),
                ord("<"),
                ord("/"),
                ord("C"),
                ord(">"),
                ord("\n"),
                65,
            ]
            self.calls = 0

        def initial_state(self):
            return torch.zeros(1)

        def forward_native(self, token_ids, *, state=None, k=None, return_all_depths=False):
            logits = torch.full((int(token_ids.numel()), 258), -1000.0)
            logits[-1, self.tokens[min(self.calls, len(self.tokens) - 1)]] = 1000.0
            self.calls += 1
            return logits, state

    out = Dummy().generate_bytes(
        prompt_ids=[256],
        max_new_bytes=32,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=999,
        stop_bytes=b"</C>\n",
    )
    assert out == b"ab</C>\n"
