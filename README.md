# heists-galore

Open-source structural weight migrations between public model architectures.

Each subdirectory is a standalone adapter: no internal dependencies, CLI-driven paths, reproducible measurement rigs.

---

## Heists

| Heist | Source → Target | CE ratio | Docs |
|-------|-----------------|----------|------|
| [mamba2-to-mamba3](mamba2-to-mamba3/) | Mamba2-2.7B → Mamba3 | **1.0016×** | [README](mamba2-to-mamba3/README.md) |

---

## Reference checkpoint

The first heist produced this alpha checkpoint on HuggingFace:

**[RtaForge/Mamba3-2.7B](https://huggingface.co/RtaForge/Mamba3-2.7B)**

Structural transmutation from `state-spaces/mamba2-2.7b` — no fine-tuning applied yet. Use the adapter in `mamba2-to-mamba3/` to reproduce or adapt your own Mamba2 weights.

---

## License

Apache 2.0 (per subdirectory).
