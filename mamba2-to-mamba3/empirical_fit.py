"""
Empirical measurement rig for Mamba2 → Mamba3 constants.

Runs a forward pass through a Mamba2 model, intercepts pre-SiLU activations
via hooks (patching out the fused causal_conv1d_fn so the module is visible),
and computes:

  alpha  — best linear approximation to SiLU(z) ≈ alpha·z
  B_RMS  — empirical RMS of post-SiLU B activations
  C_RMS  — empirical RMS of post-SiLU C activations

These constants are baked into mamba3_adapter.py. Re-run this only if you
are adapting a different Mamba2 variant (different d_model or training regime).

Usage:
    python empirical_fit.py --model /path/to/mamba2-2.7b [--tokens 1000]
"""

import argparse
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer


def run(model_path: str, n_tokens: int = 1000) -> None:
    import mamba_ssm.modules.mamba2

    # Patch out the fused kernel so nn.Conv1d hooks are reachable
    mamba_ssm.modules.mamba2.causal_conv1d_fn = None

    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    print("Loading model ...")
    model = MambaLMHeadModel.from_pretrained(model_path).to(device)
    model.eval()

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")

    # Use a fixed, representative text sample
    text = (
        "The selective state space model processes sequences by maintaining a "
        "compressed hidden state that evolves over time. Unlike transformers "
        "which attend to all past tokens, SSMs compress history into a fixed-size "
        "vector. This makes them O(L) in both time and memory during inference. "
    ) * 20
    ids = tokenizer(text, return_tensors="pt").input_ids[:, :n_tokens].to(device)
    print(f"Running {ids.shape[1]}-token forward pass ...")

    conv_inputs = []

    def _hook(module, inp, out):
        conv_inputs.append(inp[0].detach().cpu())

    for layer in model.backbone.layers:
        layer.mixer.register_forward_hook(_hook)

    with torch.no_grad():
        model(ids)

    print("Extracting activations ...")
    all_z, all_B, all_C = [], [], []

    for layer_idx, layer in enumerate(model.backbone.layers):
        u = conv_inputs[layer_idx]                           # (1, T, d_model)
        mixer = layer.mixer

        zxbcdt = F.linear(u, mixer.in_proj.weight.cpu(),
                          mixer.in_proj.bias.cpu() if mixer.in_proj.bias is not None else None)

        d_ssm    = mixer.d_ssm
        d_state  = mixer.d_state
        ngroups  = mixer.ngroups
        d_mlp    = (zxbcdt.shape[-1] - 2 * d_ssm - 2 * ngroups * d_state - mixer.nheads) // 2

        _, _, _, xBC, _ = torch.split(
            zxbcdt,
            [d_mlp, d_mlp, d_ssm, d_ssm + 2 * ngroups * d_state, mixer.nheads],
            dim=-1,
        )

        xBC_t = xBC.transpose(1, 2)                          # (1, channels, T)
        out = F.conv1d(
            xBC_t,
            mixer.conv1d.weight.cpu(),
            mixer.conv1d.bias.cpu() if mixer.conv1d.bias is not None else None,
            padding=mixer.conv1d.padding[0],
            groups=mixer.conv1d.groups,
        )
        out = out[:, :, :ids.shape[1]].transpose(1, 2)      # (1, T, channels)

        x_pre, B_pre, C_pre = torch.split(
            out, [d_ssm, ngroups * d_state, ngroups * d_state], dim=-1
        )

        all_z.append(out.reshape(-1).numpy())
        all_B.append(F.silu(B_pre).reshape(-1).numpy())
        all_C.append(F.silu(C_pre).reshape(-1).numpy())

    z = np.concatenate(all_z)
    if len(z) > 1_000_000:
        rng = np.random.default_rng(42)
        z = rng.choice(z, 1_000_000, replace=False)

    print("Fitting SiLU linear approximation ...")
    silu_z = z * (1.0 / (1.0 + np.exp(-z)))
    alpha, beta = np.polyfit(z, silu_z, 1)

    B_all = np.concatenate(all_B)
    C_all = np.concatenate(all_C)
    B_rms = float(np.sqrt(np.mean(B_all ** 2)))
    C_rms = float(np.sqrt(np.mean(C_all ** 2)))

    print()
    print("=" * 50)
    print(f"  SILU_ALPHA = {alpha:.16f}")
    print(f"  SILU_BETA  = {beta:.16f}  (ignored — zero-centered x)")
    print(f"  B_RMS      = {B_rms:.16f}")
    print(f"  C_RMS      = {C_rms:.16f}")
    print("=" * 50)
    print()
    print("Paste these into mamba3_adapter.py constants at the top of the file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure Mamba2 activation constants")
    parser.add_argument("--model",  required=True, help="Path to Mamba2 checkpoint directory")
    parser.add_argument("--tokens", type=int, default=1000, help="Number of tokens for measurement (default 1000)")
    args = parser.parse_args()
    run(args.model, n_tokens=args.tokens)
