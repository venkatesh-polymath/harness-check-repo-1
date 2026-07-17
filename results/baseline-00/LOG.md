# LOG — baseline-00 (probe)

**Date:** 2026-07-17  
**Goal:** Confirm StaR-MoE (arXiv 2605.17571) / SARA has runnable public code; if yes, run a tiny smoke test.

---

## Step 1 — Environment check

- GPU: NVIDIA A10 (23 GB VRAM, CUDA 13.0)
- Python: 3.12.10
- pip: 25.1.1
- Status: hardware ready.

---

## Step 2 — Code availability search (blocking gate)

EXPERIMENT.md is explicit: "FIRST: confirm StaR-MoE (arXiv 2605.17571) / SARA has runnable PUBLIC code + checkpoints — search GitHub, clone it. If NO usable public code exists, STOP and report status=FAILED."

**Searches conducted (all on 2026-07-17):**

1. Web search: "StaR-MoE arXiv 2605.17571 GitHub code repository"
   - Found: arXiv abstract page (https://arxiv.org/abs/2605.17571). No code link.

2. Web search: "SARA continual learning MoE GitHub repository code"
   - Found: Related MoE CL repos (MoE-Adapters4CL, MoE_PromptCL) but NOT StaR-MoE.

3. Web search: "Da-Wei Zhou Zirui Guo StaR-MoE github.com code release"
   - No GitHub repository found for these authors + StaR-MoE.

4. Fetched arXiv abstract page (https://arxiv.org/abs/2605.17571):
   - Tools like CatalyzeX listed but no active code link confirmed.

5. Fetched arXiv HTML full paper (https://arxiv.org/html/2605.17571):
   - No GitHub link, code URL, or code-availability statement in the paper.

6. Fetched arXiv PDF (https://arxiv.org/pdf/2605.17571):
   - Paper references other methods' "official implementations" for baselines.
   - No code release URL for StaR-MoE itself.

7. GitHub API search: q="StaR-MoE"
   - 93 results, none relevant (all unrelated repos).

8. GitHub API search: q="stable routing mixture experts class-incremental"
   - 0 results.

9. GitHub API search: q="SARA continual learning router"
   - 0 results.

10. Fetched https://github.com/LAMDA-CL (authors' lab GitHub org):
    - Repos present: PyCIL, LAMDA-PILOT, C3Box, Prism, ICCV25-ENGINE, ICCV2025-TUNA, PROOF, CIL_Survey, RevisitingCIL.
    - **No StaR-MoE or SARA repo.**

11. Web search: "LAMDA-CL OR LAMDA-NJ StaR-MoE continual learning 2026"
    - LAMDA-CL confirmed to have 14 repos; SAME (Stabilized MoE) at ICML 2026 is different.
    - StaR-MoE not among them.

12. Web search for author GitHub handles (ZiruiGuo, gzr2017, Zirui-Guo):
    - No matching accounts with StaR-MoE code found.

---

## Step 3 — Conclusion

The paper arXiv 2605.17571 ("Stable Routing for Mixture-of-Experts in Class-Incremental Learning") was submitted **May 2026**. As of July 2026, **no public code, checkpoint, or repository exists**. The authors' lab (LAMDA-CL / Nanjing University) has an active GitHub org but StaR-MoE is not published there.

Per EXPERIMENT.md: "If NO usable public code exists, STOP and report status=FAILED with notes='StaR-MoE public code unavailable' (do NOT reimplement from scratch)."

**Decision: STOP. Report FAILED.**

---

## What was NOT done (and why)

- No reimplementation from scratch — explicitly forbidden by EXPERIMENT.md.
- No sanity gates (Steps 1-4 from hypothesis chain) — prerequisite (code availability) not met.
- No training runs — no code to run.

---

## Next steps (for future rounds, if code is released)

1. Monitor LAMDA-CL GitHub org for a StaR-MoE repo release.
2. When released: clone, pin commit hash, verify CIFAR-100 accuracy ±0.5 pp (blocking gate).
3. Run tiny smoke: 1 small split, few steps, extract per-class router-input features.
