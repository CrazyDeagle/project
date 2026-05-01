from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def align_up(x: int, a: int) -> int:
    return ((x + a - 1) // a) * a


def s5(d_in: int) -> int:
    return align_up(ceil_div(d_in, 5), 32)


@dataclass(frozen=True)
class SilexConfig:
    batch: int = 1
    s_max: int = 8192
    u_train: int = 512
    vocab_size: int = 258
    layers: int = 64
    d_model: int = 4096
    d_ff: int = 16384
    d_z: int = 8192
    recurrent_slots: int = 8
    plastic_rank: int = 64
    q_kfac: int = 64
    k_train: int = 4
    k_infer: int = 16
    k_max: int = 32
    eps_norm: float = 2.0**-12
    eps_opt: float = 1.0e-12
    rho: float = 1.0 / sqrt(128.0)
    rho_z: float = 1.0 / 8.0


SILEX_T18_6B_R64_CONFIG = SilexConfig()

VALID_D_IN = {4096, 8192, 16384}
VALID_D_OUT = {258, 4096, 8192, 16384}

TOKEN_BOS = 256
TOKEN_EOS = 257
BYTE_MIN = 0
BYTE_MAX = 255
