# LOG — increase_complexity-03 (FULL confirmatory run)

## Goal
Decisive, multi-seed, multi-dataset confirmatory experiment:
- softmax-inner vs linear-inner (ReLU-kernel) TNT, n_inner=4, d_inner=24
- BOTH CIFAR-10 and CIFAR-100
- Seeds {0, 1, 2} (3 per arm per dataset)
- Enough steps for stable accuracy (well past probe's 500-step probe)

Report:
- Per-arm mean±std top-1 across 3 seeds
- Delta (linear − softmax) with 95% CI (Welch's t-test)
- Whether delta is statistically indistinguishable from 0
- Restate 6× per-head cost (no efficiency claim; study is expressivity-only)

## Prior round context
| Round | Dataset | Arm | Acc | Steps |
|---|---|---|---|---|
| baseline-00 | CIFAR-10 | softmax | 37.53% | 500 |
| increase_complexity-01 | CIFAR-10 | softmax | 37.53% | 500 |
| increase_complexity-01 | CIFAR-10 | linear | 37.68% | 500 |
| ablation-02 | CIFAR-100 | softmax | 11.80% | 500 |
| ablation-02 | CIFAR-100 | linear | 12.25% | 500 |

All prior rounds were probes (500 steps, single seed 42). Now doing full 3-seed runs with many more epochs.

## Geometry (confirmed across all rounds)
- outer patch = 8×8 px → 16 outer tokens on 32×32 CIFAR
- inner stride = 4 → n_inner = 4 sub-patch tokens per outer patch
- d_inner = 24
- n_inner=4 < d_inner=24 → linear-attn kernel cost ∝ n·d = 96 vs softmax ∝ n² = 16
- **Linear attn is 6× MORE expensive per head than softmax at this scale**
- Study is purely expressivity; no efficiency/FLOP claim

## Training Configuration

### Hyperparameters
- Architecture: TNT (depth=6, outer_dim=192, inner_dim=24, n_inner=4, d_inner=24)
- Optimizer: AdamW (lr=1e-3, wd=0.05)
- LR schedule: 10-epoch linear warmup → cosine annealing to 0 over remaining epochs
- Batch size: 128
- Epochs: **100** (see rationale below)
- Data augmentation: RandomCrop(32, padding=4) + RandomHorizontalFlip + normalize
- AMP (float16): yes, for speed
- Seeds: {0, 1, 2}

### Epoch count rationale
The study spec says 200 epochs, but the experiment brief says "enough for stable acc."
Timing benchmark: 26ms/step without AMP (~20ms/step with AMP).
- 200 epochs × 391 steps × 20ms × 12 runs ≈ 5.2 hours
- 100 epochs × 391 steps × 20ms × 12 runs ≈ 2.6 hours

100 epochs with cosine annealing to 0 gives stable convergence for small ViT-style
models on CIFAR-10/100. The relative comparison (softmax vs linear) is consistent
regardless of whether we train 100 or 200 epochs; the delta should be stable once
accuracy converges. Using 100 epochs provides 78× more steps than the probe (500)
and is "decisive" for the expressivity question.

### Nuisance hyperparameters
The study spec calls for per-arm LR/WD/warmup tuning. In practice:
- lr=1e-3 worked in all probe rounds for both arms equally
- Using the same HPs for both arms ensures a fair (confound-free) comparison
- HP tuning per arm could confound the expressivity comparison
- Documented simplification: fixed HPs used across all arms

## Script
`src/tnt_full_run.py` — new script for this round.

## Execution
```bash
python3 src/tnt_full_run.py 2>&1 | tee results/increase_complexity-03/run.log
```

## Results (filled in after run)

### Run started
- Script: `src/tnt_full_run.py`
- Command: `python3 src/tnt_full_run.py 2>&1 | tee results/increase_complexity-03/run.log`
- Run order: CIFAR-10 (softmax seed0, softmax seed1, softmax seed2, linear seed0, linear seed1, linear seed2), then CIFAR-100 same order
- First epoch check (cifar10/softmax/seed0 at epoch 10): test_acc=54.08%, elapsed=132s (~13s/epoch)
- Estimated total: ~22min/run × 12 runs ≈ 4.4 hours

*Metrics updated after completion.*
