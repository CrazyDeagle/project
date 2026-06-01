# Architecture Overview

This document gives a high-level map of the SilexCode codebase. It is meant for
new contributors who want to know *where to look*, not as a full mathematical
treatment of the model.

## Package Layout

```
silexcode/
├── constants.py     # SilexConfig dataclass and the SILEX_T18_6B_R64 preset
├── tokenizer.py     # Byte-level tokenizer (258-symbol alphabet incl. BOS/EOS)
├── budget.py        # TDD invariants and memory budget assertions
├── dataset.py       # Synthetic record generation and candidate verification
├── losses.py        # silex_latent_loss — multi-depth latent loss
├── model.py         # SilexCodeT18_6B_R64, TLinear, TernaryEmbedding
├── kfac.py          # BlockKFACOptimizer (Kronecker-factored second-order)
├── training.py      # train_chunk: one update over a 512-token chunk
├── train.py         # Curriculum scheduler and stage_ready policy
├── accelerated.py   # Multi-record packing variant of the curriculum loop
├── bootstrap.py     # Plastic-only warm-start on short Python snippets
├── checkpoint.py    # Native packed-weight checkpoint I/O (.silex format)
└── cuda/            # C++ / CUDA extension (compiled into silexcode._C)
    ├── bindings.cpp
    └── tlinear_kernels.cu
```

The top-level training entry points live at the repository root:

- [`run_curriculum.py`](../run_curriculum.py) — full curriculum.
- [`run_bootstrap.py`](../run_bootstrap.py) — bootstrap warm start.
- [`run_accelerated_curriculum.py`](../run_accelerated_curriculum.py) —
  accelerated, record-packed curriculum.

## Runtime Layers

```
                          ┌──────────────────────┐
   user CLI / runner      │ run_*.py             │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
   training orchestration │ silexcode.train      │
                          │ silexcode.accelerated│
                          │ silexcode.bootstrap  │
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
   one-step update        │ silexcode.training   │
                          │   train_chunk(...)   │
                          └──────────┬───────────┘
                                     │
            ┌────────────────────────┼──────────────────────────┐
            │                        │                          │
   ┌────────▼───────┐      ┌─────────▼────────┐       ┌─────────▼────────┐
   │ silexcode.model│      │ silexcode.losses │       │ silexcode.kfac   │
   │  SilexCodeT...│      │ silex_latent_loss│       │ BlockKFAC...    │
   │  TLinear / Emb│      │                  │       │                  │
   └────────┬───────┘      └──────────────────┘       └──────────────────┘
            │
   ┌────────▼───────┐
   │ silexcode._C   │  (compiled CUDA extension — TLinear kernels)
   └────────────────┘
```

## Key Concepts

### TDD Invariants

`silexcode.budget` defines the architectural invariants the training loop is
required to preserve (sequence length, attention shape, parameter counts,
memory budget). They are asserted at startup so a broken configuration fails
loudly instead of producing silently-wrong updates.

### Packed Ternary Weights

`TLinear` stores its weights as `uint8` packed ternaries plus a `bf16` scale
vector (`wpack`, `alpha`). The CUDA kernel under `silexcode/cuda/` unpacks on
the fly. Two runtime modes are supported:

- `deterministic_backbone=True` uses the FWHT fast path, which exactly matches
  the deterministic TDD initialisation.
- `deterministic_backbone=False` uses the packed kernel directly, which
  honours arbitrary checkpoint weights.

### Checkpoints

`silexcode.checkpoint` defines the `.silex` packed-weight format (magic
`SILEXCODE_T18_6B_R64`) and a lightweight *plastic* checkpoint
(`SILEXCODE_T18_6B_R64_PLASTIC`) that stores only the adapters touched by
bootstrap. The plastic checkpoint can be resumed into a full curriculum run.

### K-FAC Updates

`BlockKFACOptimizer` keeps per-block input and gradient covariances with EMA
decay, inverts them with damping, and applies a trust-region clip on each
parameter update. The optimizer can restrict itself to a subset of layers via
`active_layers`, which is what the bootstrap and output-only stages use.

## Where to Add Things

| You want to…                                | Edit                                |
| ------------------------------------------- | ----------------------------------- |
| Tune the curriculum schedule                | `silexcode/train.py`                |
| Add a new packing strategy                  | `silexcode/accelerated.py`          |
| Change the loss                             | `silexcode/losses.py`               |
| Add a CUDA kernel                           | `silexcode/cuda/`                   |
| Add a checkpoint field                      | `silexcode/checkpoint.py` and bump `SILEX_VERSION` |
| Add a new architectural invariant           | `silexcode/budget.py`               |
| Change the alphabet                         | `silexcode/tokenizer.py` and `silexcode/constants.py` |

If your change crosses two or more of these layers, please write an ADR under
[`docs/adr/`](adr/) before opening the PR.
