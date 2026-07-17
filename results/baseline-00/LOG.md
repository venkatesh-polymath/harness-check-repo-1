# LOG — baseline-00 (probe)

**Date:** 2026-07-17  
**Goal:** TNT (Transformer-in-Transformer) CIFAR-10 probe — run 4 sanity checks to establish the baseline.

---

## Prior round context

The RESULTS.json and run.log in this directory were previously from a different experiment
(StaR-MoE FAILED — no public code). The EXPERIMENT.md now describes the TNT study.
This LOG covers the TNT probe only; all prior files are overwritten.

---

## Step 1 — Environment

- GPU: NVIDIA A10 (23 GB VRAM, CUDA 13.0, driver 580.95.05)
- Python: 3.12.10
- Packages installed: torch (2.7.0, pre-installed) + torchvision 0.28.0 + timm 1.0.28 +
  einops 0.8.2 + datasets 5.0.0 (HuggingFace)

---

## Step 2 — Architecture clone

```bash
git clone --depth=1 https://github.com/huawei-noah/CV-Backbones.git
git -C CV-Backbones rev-parse HEAD  →  f90e129b645c3b1684fe07cd361cd557d0ad71f7
```

TNT code lives in `CV-Backbones/tnt_pytorch/tnt.py`.  
We use it as an architecture reference; `src/tnt_cifar_probe.py` is a self-contained
reimplementation for 32×32 CIFAR.

**timm API note:** timm 1.0.28 removed `load_pretrained`, `register_model`, and the
`timm.models.helpers` / `timm.models.registry` paths used by the upstream TNT code.
The probe script imports only `DropPath`, `to_2tuple`, `trunc_normal_` (still present),
so no porting friction.

---

## Step 3 — Geometry & FLOP pre-analysis (hypothesis-chain gates 5 & 6)

| Parameter           | Value  | Notes                          |
|---------------------|--------|--------------------------------|
| outer_patch_size    | 8×8 px | 32 / 8 = 4 patches per side   |
| sub_patch_size      | 4×4 px | inner_stride=4                 |
| n_outer_patches     | 16     | 4×4 grid                       |
| n_inner (words/pch) | 4      | ceil(8/4)^2 = 4                |
| d_inner             | 24     | inner embedding dim            |
| d_outer             | 192    | outer embedding dim (probe)    |
| depth               | 6      | TNT blocks                     |

Inner-block attention FLOPs per layer: 2 × n_inner² × d_inner × n_outer = 2×16×24×16 = 12,288  
Outer-block attention FLOPs per layer: 2 × (n_outer+1)² × d_outer = 2×289×192 = 110,976  
Inner fraction of all attention FLOPs (6 layers): 73,728 / 739,584 ≈ **10.0 %**

**Finding (pre-registered):** n_inner = 4 < d_inner = 24 → the linear-attention
kernel's d² feature-map overhead dominates; linear attention has NO FLOP advantage
at this geometry. The original "12–18% total FLOP reduction" claim is arithmetically
refuted BEFORE any training run. Study reframed as expressivity-only (per EXPERIMENT.md).

---

## Step 4 — Data

Toronto's CIFAR-10 server served at ~860 bytes/s (170 MB → estimated 33 min).
HuggingFace CDN (uoft-cs/cifar10) served at ~2.8 MB/s.  
**Decision:** use `datasets.load_dataset("uoft-cs/cifar10")` and wrap in a
PyTorch `Dataset` subclass (`HFCifar10Dataset`). This is equivalent to
`torchvision.datasets.CIFAR10` but downloads in seconds.

---

## Step 5 — Sanity checks (4 gates from EXPERIMENT.md)

Script: `src/tnt_cifar_probe.py`

### (a) Initial cross-entropy ≈ ln(10) = 2.303

Random-init model, eval mode, forward pass on one batch.  
Result: **init_loss = 2.3568**, |diff from ln10| = 0.054 < 0.15 → **PASS**

### (b) Test accuracy > 10% after 500 steps

500 steps of AdamW (lr=1e-3, wd=0.05) on full train loader, then evaluate test set.  
Result: **test_acc = 37.53%** > 10% → **PASS**  
(~16 s on A10 GPU)

### (c) Overfit 32-image fixed batch to < 0.01 CE within 300 steps

**Optimizer tuning needed** (all attempts recorded for reproducibility):

| Attempt | Optimizer / LR | Grad clip | min@300 steps | Notes |
|---------|----------------|-----------|---------------|-------|
| 1 | AdamW lr=5e-3 | none | 1.315 | too slow |
| 2 | AdamW lr=5e-2 | none | 1.201 | still too slow |
| 3 | AdamW lr=0.5  | none | 2.190 | explosion then collapse |
| 4 | AdamW lr=1e-2 | max_norm=1 | 0.407 | reaches <0.01 at step 1131 |
| 5 | AdamW lr=5e-2 | max_norm=1 | 0.779 | reaches <0.01 at step 841 |
| 6 | **SGD Nesterov lr=0.5 + cosine sched + clip** | max_norm=1 | **0.0087** | **PASS @ step 65** ✓ |

**Why SGD+Nesterov wins:** The TNT loss surface has a long, curved valley. AdamW's
per-parameter scaling damps the large-step benefit of high LR. Nesterov momentum
exploits the smooth valley curvature with a look-ahead update; cosine schedule
decays LR from 0.5 to 1e-4 over 300 steps to prevent oscillation.

Result: **min_loss@300 = 0.00870** < 0.01, reached at step 65 → **PASS**

### (d) Reproducibility: |loss_run1[i] − loss_run2[i]| < 1e-6 for i∈{0..4}

Two independent forward-only runs with seed=42 (model+loader both seeded).  
Result: **max diff = 0.0** → **PASS**  
Losses (both runs identical): [2.376692, 2.342964, 2.336877, 2.320317, 2.309260]

### All sanity checks: ✓ PASS

---

## Step 6 — Final status

| Check | Value | Pass |
|-------|-------|------|
| (a) init CE | 2.3568 vs 2.3026 target | ✓ |
| (b) test acc @500 steps | 37.53% | ✓ |
| (c) overfit @300 steps | 0.0087 | ✓ |
| (d) reproducibility | diff = 0.0 | ✓ |

**status = SUCCESS**

---

## Decisions & rationale

- **outer_dim=192, depth=6:** Small enough for a fast probe (~15 s for 500 steps
  on A10), large enough to be a proper TNT architecture with inner+outer blocks.
- **patch_size=8, inner_stride=4:** Gives n_inner=4 per outer patch as required.
- **SGD+Nesterov for overfit check:** AdamW with any LR ≤ 0.5 required 840–1131 steps
  to reach <0.01 on this architecture; SGD+Nesterov+cosine hit it at step 65. This is
  consistent with known AdamW slow-convergence in small-data/overfit regimes.
- **Data via HuggingFace:** Functionally identical to torchvision CIFAR-10 (same split
  sizes: 50,000 train / 10,000 test, same 32×32 images), but downloads 3000× faster.
- **No weights committed:** All checkpoints are in-memory only; script never saves them.
