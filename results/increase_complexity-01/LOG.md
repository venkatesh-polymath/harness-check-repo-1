# LOG — increase_complexity-01

## Goal
Replace softmax attention with **ReLU-kernel linear attention** ONLY in TNT's inner block
(n_inner=4, d_inner=24). Outer block stays softmax (identical to baseline-00).
Probe: 500 AdamW steps, same seed=42 setup as baseline-00.
Report test accuracy and Δ vs committed baseline (37.53%).

## Prior round context
- baseline-00: all 4 sanity checks pass; test_acc=37.53% at 500 steps.
- Geometry pre-analysis (from baseline): n_inner=4 < d_inner=24.
  - Linear attention cost ∝ n·d = 96 vs softmax ∝ n² = 16 → linear is **6× MORE expensive**
    per head at this scale. Efficiency framing completely dropped; study is expressivity-only.
  - Inner-block attention is ~10% of total attention FLOPs; max total-model cut from removing
    all inner-block attention < 10%.

## Decisions

### Linear attention implementation
Chose **Performer-style ReLU kernel**: φ(x) = ReLU(x) + ε (ε=1e-6).
- φ(Q) @ (φ(K)ᵀ @ V) with denominator D = φ(Q) @ sum(φ(K), dim=seq)
- Numerically stable: ε prevents all-zero rows, denominator clamped to ε.
- This is the simplest provably-correct linear-attention factorisation with a
  non-negative kernel (needed for normalisation stability).

### Why not cosine kernel?
Cosine kernel would require scaling Q/K to unit sphere; equivalent complexity at n=4.
ReLU is simpler, established in the literature (Performer), and sufficient for a probe.
If results are inconclusive, cosine can be added as a follow-on (Hypothesis chain step 9).

### Run design
1. Init-loss check on the linear-attention model (sanity).
2. Linear-inner training run (500 steps, seed=42, AdamW lr=1e-3 wd=0.05).
3. Softmax-inner control run (same process, same loader, seed=42) to measure
   within-process Δ free of any batch-ordering confound.
4. Compare both against committed baseline (37.53%).

## Execution
- Script: src/tnt_linear_inner_probe.py
- Log: results/increase_complexity-01/run.log
- Started: 2026-07-17

## Results

| Arm | Acc@500steps | Δ vs baseline |
|-----|-------------|---------------|
| baseline-00 (softmax-inner, committed) | 37.53% | — |
| linear-inner (this run, seed=42) | 37.68% | +0.15pp |
| softmax-inner control (this run, seed=42) | 37.53% | 0.00pp |

- **Δ vs committed baseline**: +0.15pp (linear-inner is marginally better — within noise)
- **Within-process delta** (linear vs softmax control, same loader): +0.15pp
- **Init CE (linear model)**: 2.3565 (PASS — within 0.15 of ln(10)=2.3026)
- Wall-clock: ~17s for 500 steps on NVIDIA A10 GPU (both arms equally fast at n=4)

## Interpretation

At n_inner=4, d_inner=24:
- Linear attention kernel cost ∝ n·d = 96, softmax ∝ n² = 16 → linear is **6× more expensive**
- Yet accuracy is statistically identical (+0.15pp is pure noise at probe scale)
- Result: at this tiny n_inner=4, the sub-patch sequence is so short that the expressivity
  difference between softmax and ReLU-kernel linear attention is negligible
- The softmax-inner control exactly reproduces the committed baseline (37.53%), confirming
  the same-process run is a valid control

## Next steps (per hypothesis chain)
- Hypothesis 8 (3-seed comparison): need 3 seeds per arm for statistical power
- Hypothesis 11 (CIFAR-100 gap): need CIFAR-100 runs to test if fine-grained classes
  show larger degradation from linearising the inner block
- For now, probe shows: **no degradation** from linear inner attention at n_inner=4

