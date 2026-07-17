# LOG — smoke-00 (GPU smoke test)

**Date:** 2026-07-17
**Round type:** PROBE / smoke test (does it run, does the metric move at all?)

## Goal
Per `EXPERIMENT.md`: verify the GPU is usable from PyTorch. Not a research
experiment — no training, no dataset downloads. Under 2 minutes.

## What I did & why
1. **`nvidia-smi`** to confirm a GPU is present and see driver/CUDA version.
   - Observed: 1x **NVIDIA A10** (23 GB), Driver 580.95.05, CUDA 13.0, idle
     (0% util, 0 MiB used, 33°C). No other processes running.
2. Checked PyTorch: `torch 2.13.0+cu130`.
   - `torch.cuda.is_available()` → **True**
   - `torch.cuda.get_device_name(0)` → **"NVIDIA A10"**
3. **Tiny GPU matmul**: `2048x2048 @ 2048x2048` fp32 on `cuda`.
   - Did a warmup matmul + `torch.cuda.synchronize()` before timing so the
     measured time reflects real kernel execution, not lazy dispatch / cold
     start. Timed with `time.perf_counter()` bracketed by `synchronize()`.
   - **matmul_ms ≈ 1.34 ms** (single timed iteration).
   - Sanity: result shape `(2048, 2048)`, finite sum ≈ -195525.

## Result
GPU is available and functional. Metric "moves" (matmul executes on device in
~1.3 ms). Status: **SUCCESS**.

## Files
- `src/smoke_test.py` — the experiment code (committed).
- `results/smoke-00/run.log` — raw stdout/stderr.
- `results/smoke-00/RESULTS.json` — final metrics.

## Notes / decisions
- Kept fp32 default (no TF32/AMP tweaks) — this is a plumbing check, not a perf
  benchmark, so a single warmed timing is sufficient.
- No weights/checkpoints produced; nothing to git-ignore beyond existing rules.
