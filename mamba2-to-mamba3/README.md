# Mamba2 → Mamba3 Weight Adapter

Structural weight migration from a trained **Mamba2-2.7B** checkpoint to the
**Mamba3** architecture. No fine-tuning required; the adapted checkpoint is
immediately runnable and can be fine-tuned from the much better starting point.

**Final CE ratio: 1.0016×** — Mamba3 CE 13.0750 vs Mamba2 baseline 13.0531
on random tokens (A10G, bfloat16, seed=42).

**Reference checkpoint:** [RtaForge/Mamba3-2.7B](https://huggingface.co/RtaForge/Mamba3-2.7B) — alpha weights produced with this adapter.

---

## Background

Mamba2 and Mamba3 share topology but differ in their core SSM math:

| Change | Mamba2 | Mamba3 |
|--------|--------|--------|
| Conv → SiLU pipeline | explicit `conv1d + SiLU` | removed |
| Decay A | fixed `A_log` parameter | data-dependent, absorbed into `dt_bias` |
| Integration | pure Euler | trapezoidal (Trap gate) |
| Positional bias | none on B/C | RoPE on B/C vectors |
| SSM normalisation | none | `B_norm`, `C_norm` LayerNorms |

Naively transplanting Mamba2 weights breaks the model — the equations change.
The goal of this adapter is **behavioural approximation**: map the weights close
enough that short SFT can close the remaining gap (empirically ~0.16% CE overhead).

---

## Nine-Point Transformation

1. **Conv1d kernel folding** — absorb the final causal-conv kernel weight into
   the `x`, `B`, `C` projection rows, eliminating the spatial filter.

2. **SiLU linearization** — approximate `SiLU(z) ≈ 0.4408·z`, bake the scalar
   into the projection weights so the output magnitude is preserved.

3. **BCNorm magnitude restoration** — set `B_norm.weight = B_RMS = 0.2120` and
   `C_norm.weight = C_RMS = 0.2961` (measured empirically), restoring expected
   B/C signal amplitude after steps 1–2 change it.

4. **A_log → dt_bias (3-part)** — shift `dt_bias` to preserve the A·dt decay
   product; scale `x_proj` rows by `1/S`; scale `D` by `S` to keep the skip
   path consistent (`S = exp(A_log) / |A_M3|`).

5. **Trap gate zero-init** — `sigmoid(0) = 0.5` gives a pure trapezoidal start,
   the neutral default for the new lambda gate.

6. **RoPE zero-init** — zero rotation angles → no positional bias at transfer
   point; the model learns them during SFT.

7. **MIMO → SISO** — `is_mimo=False` selects the Triton SISO kernel, which runs
   on any CUDA GPU (not just those with the B6 MIMO Triton kernel patch).

8. **B\_bias / C\_bias** — left at the Mamba3 default `1.0`; overriding them
   with Mamba2 values was found to be unnecessary.

9. **is\_safe\_A** — `A_M3 = −ln(2) ≈ −0.6931` (always negative), ensuring
   the decay stays in the stable half-plane regardless of data-dependent shift.

---

## Files

| File | Purpose |
|------|---------|
| `mamba3_adapter.py` | The nine-point mapping; CLI entry point |
| `empirical_fit.py`  | Measurement rig for `SILU_ALPHA`, `B_RMS`, `C_RMS` |
| `check_ce.py`       | CE harness to verify baseline and adapted checkpoint |

---

## Quick Start

```bash
pip install torch transformers mamba-ssm[causal-conv1d]

# 1. Adapt a Mamba2 checkpoint to Mamba3
python mamba3_adapter.py \
    --source /path/to/mamba2-2.7b \
    --output /path/to/mamba3-2.7b-subsumed

# 2. Verify CE ratio
python check_ce.py --model /path/to/mamba2-2.7b          --arch mamba2
python check_ce.py --model /path/to/mamba3-2.7b-subsumed --arch mamba3

# 3. (Optional) Re-measure constants if using a different Mamba2 variant
python empirical_fit.py --model /path/to/mamba2-2.7b --tokens 1000
# → paste SILU_ALPHA, B_RMS, C_RMS into mamba3_adapter.py
```

---

## Requirements

- Python 3.10+
- PyTorch ≥ 2.1
- `transformers` ≥ 4.43 (for `Mamba2ForCausalLM`)
- `mamba-ssm` ≥ 2.3 with Mamba3 support

---

## Methodology

The measurement rig patches out `causal_conv1d_fn` at import time so
PyTorch hooks can intercept the pre-SiLU activations inside each Mamba2 mixer.
We then reproduce the Mamba2 forward pass manually (`F.linear + F.conv1d`) to
extract the actual activation distributions, and fit the SiLU linearisation
using `numpy.polyfit` on 1M samples drawn across all 64 layers × 1000 tokens.

This approach — empirically measuring activation statistics before absorbing
them into the weight map — is what allows the CE ratio to stay below 1.002×
without any gradient-based tuning.

---

## Citation

If you use this adapter in your work:

```bibtex
@misc{heist-mamba3-2026,
  title  = {Mamba2 to Mamba3 Structural Weight Migration},
  author = {Guha Krishnamurthy},
  year   = {2026},
  url    = {https://github.com/Rta-Forge/heists-galore}
}
```

---

## License

Apache 2.0.
