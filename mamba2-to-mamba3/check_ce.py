"""
Cross-entropy measurement harness for Mamba2 / Mamba3 checkpoints.

Runs 5 batches of random tokens with a fixed seed and reports mean CE.
Use this to verify the CE ratio before and after adaptation.

Usage:
    # Baseline (Mamba2)
    python check_ce.py --model /path/to/mamba2-2.7b --arch mamba2

    # After adaptation (Mamba3)
    python check_ce.py --model /path/to/mamba3-2.7b-subsumed --arch mamba3

Expected result for 2.7B: Mamba3/Mamba2 CE ratio ≈ 1.0016×
"""

import argparse
import sys

import torch
import torch.nn as nn


def _run_ce(model: nn.Module, vocab_size: int, device: str, seed: int = 42) -> float:
    ce_fn = nn.CrossEntropyLoss()
    scores = []
    torch.manual_seed(seed)
    model.eval()
    with torch.no_grad():
        for _ in range(5):
            ids = torch.randint(0, vocab_size, (1, 32)).to(device)
            out = model(ids)
            logits = (
                out["logits"] if isinstance(out, dict)
                else out.logits if hasattr(out, "logits")
                else out[0]
            )
            shift = logits[:, :-1, :].reshape(-1, vocab_size)
            tgt   = ids[:, 1:].reshape(-1)
            scores.append(ce_fn(shift, tgt).item())
    return sum(scores) / len(scores)


def _load_mamba2(checkpoint: str, device: str) -> nn.Module:
    from transformers import Mamba2ForCausalLM, Mamba2Config
    import torch

    config = Mamba2Config(
        vocab_size=50288,
        hidden_size=2560,
        num_hidden_layers=64,
        state_size=128,
        conv_kernel=4,
        head_dim=64,
        num_heads=80,
        expand=2,
    )
    model = Mamba2ForCausalLM(config)
    sd = torch.load(checkpoint if checkpoint.endswith(".bin") else f"{checkpoint}/pytorch_model.bin",
                    map_location="cpu")
    # Handle HF key prefix variant
    sd = {k.replace("backbone.embedding.", "backbone.embeddings."): v for k, v in sd.items()}
    missing, _ = model.load_state_dict(sd, strict=False)
    if len(missing) > 5:
        print(f"WARNING: {len(missing)} missing keys — checkpoint may not match config")
    return model.to(device)


def _load_mamba3(checkpoint: str, device: str) -> nn.Module:
    from mamba_ssm.models.mamba3 import Mamba3CausalLM, Mamba3Config
    import torch

    config = Mamba3Config(
        d_model=2560,
        n_layer=64,
        vocab_size=50288,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=1,
        rope_fraction=0.5,
        is_safe_A=True,
        is_mimo=False,
    )
    model = Mamba3CausalLM(config)
    sd = torch.load(checkpoint if checkpoint.endswith(".bin") else f"{checkpoint}/pytorch_model.bin",
                    map_location="cpu")
    model.load_state_dict(sd, strict=False)
    return model.to(device)


def run(checkpoint: str, arch: str, device: str) -> float:
    print(f"Loading {arch} from {checkpoint} ...")
    if arch == "mamba2":
        model = _load_mamba2(checkpoint, device)
        vocab_size = 50288
    elif arch == "mamba3":
        model = _load_mamba3(checkpoint, device)
        vocab_size = 50288
    else:
        raise ValueError(f"Unknown arch: {arch}")

    ce = _run_ce(model, vocab_size, device)
    print(f"\n  {arch.upper()} CE: {ce:.4f}")
    return ce


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CE measurement harness")
    parser.add_argument("--model",  required=True, help="Path to checkpoint directory or .bin file")
    parser.add_argument("--arch",   required=True, choices=["mamba2", "mamba3"], help="Architecture type")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(args.model, args.arch, args.device)
