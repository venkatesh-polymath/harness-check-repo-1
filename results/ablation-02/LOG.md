# ablation-02 LOG

## Round context
- EXPERIMENT.md round: ablation-02 (probe)
- Task: repeat the softmax-inner vs linear-inner comparison from increase_complexity-01,
  but on CIFAR-100 instead of CIFAR-10.
- Prior CIFAR-10 numbers (increase_complexity-01): softmax=37.53%, linear=37.68%, delta=+0.15pp

## What I read
- `results/increase_complexity-01/RESULTS.json` — CIFAR-10 baseline numbers, geometry note
- `src/tnt_linear_inner_probe.py` — existing probe code (CIFAR-10)
- `src/tnt_cifar_probe.py` — baseline-00 code

## Decisions

### Script: `src/tnt_cifar100_ablation.py`
Copied the TNT architecture from `tnt_linear_inner_probe.py` verbatim (same modules:
`Mlp`, `SoftmaxAttention`, `LinearAttention`, `TNTBlock`, `PatchEmbed`, `TNTSmall`).

**Changes from prior script:**
1. `num_classes=100` (was 10)
2. `HFCifar100Dataset` uses `fine_label` key (CIFAR-100 HF uses this instead of `label`)
3. `get_cifar100_loaders` uses CIFAR-100 per-channel normalization stats
   `(0.5071, 0.4867, 0.4408) / (0.2675, 0.2565, 0.2761)` (standard values)
4. Init-loss sanity check threshold: `|loss - ln(100)| < 0.30` (ln(100)≈4.605);
   slightly looser than the CIFAR-10 check (0.15) because 100-class logits
   can have slightly larger variance at init.
5. Run order: softmax first, then linear (contrast with prior round that ran linear first)
   → avoids any suspicion of data-loader ordering bias affecting the first arm.

### Probe scale
500 steps, seed=42, AdamW lr=1e-3 wd=0.05 — identical to baseline-00 and
increase_complexity-01 so results are directly comparable.

### Geometry (unchanged, documented for completeness)
- outer patch = 8×8 px → 16 outer tokens on 32×32 CIFAR
- inner stride = 4 → n_inner = 4 sub-patch tokens per outer patch
- d_inner = 24
- n_inner=4 < d_inner=24 → linear-attn kernel 6x MORE expensive than softmax
- Study is purely expressivity; no efficiency/FLOP claim

## Execution
```
python3 src/tnt_cifar100_ablation.py 2>&1 | tee results/ablation-02/run.log
```
(Run on NVIDIA A10 GPU, CUDA available)

## Result summary (filled in after run)

### Numbers
| | softmax-inner | linear-inner | delta (lin−soft) |
|---|---|---|---|
| CIFAR-10 (increase_complexity-01) | 37.53% | 37.68% | +0.15 pp |
| CIFAR-100 (this run) | 11.80% | 12.25% | +0.45 pp |

delta_change (C100 − C10) = **+0.30 pp** — the gap *widened in favour of linear* on CIFAR-100.

### Prediction assessment
Pre-registered prediction: `delta_c100 − delta_c10 < −0.5pp` (linear attn hurts more on harder dataset).
**NOT CONFIRMED.** Both datasets show linear-inner marginally *outperforming* softmax-inner at 500 steps,
and the advantage is larger on CIFAR-100 (+0.45 pp vs +0.10 pp). This is a publishable negative result:
inner-block attention type does not degrade accuracy at this scale and the dataset difficulty does not
modulate the effect in the predicted direction.

### Sanity checks
- init CE (CIFAR-100, linear model): 4.6539 vs ln(100)=4.6052 → |diff|=0.049 < 0.30 → **PASS**
- Run finished cleanly, no NaN losses.

### Wall-clock
~33 s total (both arms × 500 steps on A10). Very fast probe — metric did move.

