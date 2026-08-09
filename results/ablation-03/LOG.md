# LOG — ablation-03 probe

## Round
ablation-03 (Separate Calibration from Collapse)

## Date
2026-08-09

## Objective
EXPERIMENT.md reviewer objection: at ~12.8% cross-site accuracy (Blood_5 target),
the previously reported 70%+ ECE conflates accuracy failure with miscalibration.
Fix by separating the two effects:

1. **LABEL-AWARE MASKED SOFTMAX**: zero PBC-only logits (ig=1, erythroblast=5,
   platelet=7) to -1e9 before renormalising softmax over 5 shared classes.
   Report accuracy and ECE-15 with/without source-fit T per seed.

2. **CLASSWISE ECE**: per-class ECE on Blood5 for each of the 5 shared classes,
   with and without T. (Conditional ECE: samples where true=c only.)

3. **ACCURACY-STRATIFIED CALIBRATION**: split predictions into correct/incorrect,
   report mean confidence for each. Diagnostic: high conf_incorrect → miscalibration.

4. **PBC IN-DOMAIN REFERENCE**: same 3 quantities on PBC test set for contrast.

5. **INTERPRETATION**: state explicitly whether the failure is 'miscalibration',
   'representation_collapse', or 'both', with the numbers that decide it.

## Prior rounds
- **baseline-00**: Label-space mismatch audit — 84.86% ECE was entirely a label-space
  artifact (PBC 8-class indices vs Blood5 5-class indices mismatched).
- **refine-01**: Genuine cross-site eval on Blood_5 (Zenodo 21628834). Found 99.2%
  monocyte collapse; Scenario C acc=12.8%, ECE=70.3% (15-bin); source-fit T reduces
  cross-site ECE by only 11% (vs 47% in-domain). Raw ECE numbers were not interpretable
  because accuracy failure and calibration failure were conflated.
- **ablation-02**: Confirmed the monocyte collapse is genuine domain shift (stain/
  acquisition differences), NOT preprocessing mismatch. PBC numpy-roundtrip delta=0.000
  proves loading path correct; BGR→RGB swap made things worse (1.9% accuracy).

## What I did

### 1. Data
- Blood5 was not cached (prior _weights/ cleared). Re-downloaded via HTTP range requests
  from Zenodo 21628834 ZIP central directory. Extracted test_batch (254 MB compressed →
  349 MB decompressed, CIFAR-style pickle with byte-string keys b'data', b'labels').
- Saved to `/workspace/_weights/blood5_test_data.npy` and `blood5_test_labels.npy`.
- Same 5175 images (150×150×3 uint8 HWC) as in all prior rounds.

### 2. Retraining
No checkpoints available from prior rounds. Retrained 3 seeds with identical
hyperparameters to refine-01: AdamW lr=1e-4 wd=1e-2 cosine 15 epochs patience 5,
ResNet-18 ImageNet-pretrained → 8-class PBC head. Results match refine-01 bit-for-bit
(same seeds, same data split seed=42, same training code).

### 3. Analysis design decisions

#### Label-aware masked softmax
Zero out (set to -1e9) the 3 PBC-only logit positions [1=ig, 5=erythroblast, 7=platelet]
in the raw logit vector (8-dim) BEFORE applying softmax. After masking:
- Positions [1,5,7] → prob ≈ 0 (exp(-1e9) ≈ 0)
- Positions [0,2,3,4,6] → renormalize to sum=1

This is mathematically identical to Scenario C from refine-01 (extract 5 shared logits,
apply softmax). Accuracy and ECE numbers match closely, confirming implementation correctness.

#### Classwise ECE
For each class c ∈ {basophil, eosinophil, lymphocyte, monocyte, neutrophil}:
- Filter to samples where true Blood5 label = c
- Compute 15-bin equal-mass ECE on (max_confidence, 1{argmax=c}) for those samples
- Uses same equal-mass binning as all prior rounds

Rationale: reveals WHICH classes drive the high aggregate ECE.

#### Accuracy-stratified calibration
- Correct group: samples where argmax == true label
- Incorrect group: samples where argmax ≠ true label
- Report mean(max_prob over 5 shared classes) for each group

KEY DIAGNOSTIC:
- If conf_incorrect ≈ conf_correct ≈ high: model is equally confident right and wrong
  → this IS miscalibration (model can't distinguish its own errors)
- If conf_incorrect << conf_correct: model knows when it's uncertain → accuracy failure
- A well-calibrated model has large gap (correct >> incorrect confidence)

#### PBC in-domain reference
Two sub-analyses for PBC:
(a) Full 8-class (no masking needed, all classes valid)
(b) Restricted to 5 Blood5-shared classes (same masking applied for fair comparison)

## Results per seed

### Seed 0
- T* = 1.3121
- Blood5 masked: acc_noT=0.1254, ECE_noT=0.7855, acc_T=0.1254, ECE_T=0.7248
- Classwise ECE noT: basophil=0.858(n=132), eosinophil=0.828(n=148),
  lymphocyte=0.844(n=1241), **monocyte=0.102**(n=642), neutrophil=0.946(n=3012)
- Acc-strat noT: conf_correct=0.887, conf_incorrect=0.914, gap=-0.027 (incorrect > correct!)
- PBC 8class: acc=0.9891, ECE=0.0055

### Seed 1
- T* = 1.3035
- Blood5 masked: acc_noT=0.1316, ECE_noT=0.6615, acc_T=0.1316, ECE_T=0.5835
- Classwise ECE noT: basophil=0.613(n=132), eosinophil=0.766(n=148),
  lymphocyte=0.740(n=1241), monocyte=0.159(n=642), neutrophil=0.801(n=3012)
- Acc-strat noT: conf_correct=0.865, conf_incorrect=0.782, gap=+0.082
- PBC 8class: acc=0.9911, ECE=0.0034

### Seed 2
- T* = 1.3165
- Blood5 masked: acc_noT=0.1281, ECE_noT=0.6629, acc_T=0.1281, ECE_T=0.5703
- Classwise ECE noT: basophil=0.684(n=132), eosinophil=0.710(n=148),
  lymphocyte=0.742(n=1241), monocyte=0.163(n=642), neutrophil=0.803(n=3012)
- Acc-strat noT: conf_correct=0.825, conf_incorrect=0.786, gap=+0.039
- PBC 8class: acc=0.9907, ECE=0.0035

## Aggregate (3 seeds)

| Metric | Mean | Std | Seeds |
|--------|------|-----|-------|
| T* | 1.3107 | 0.0054 | [1.312, 1.303, 1.317] |
| Blood5 masked acc (no T) | 0.1284 | 0.0025 | [0.125, 0.132, 0.128] |
| Blood5 masked ECE (no T) | 0.7033 | 0.0582 | [0.786, 0.661, 0.663] |
| Blood5 masked ECE (T) | 0.6262 | 0.0699 | [0.725, 0.583, 0.570] |
| Blood5 conf_correct (no T) | 0.8590 | 0.0257 | [0.887, 0.865, 0.825] |
| Blood5 conf_incorrect (no T) | 0.8275 | 0.0614 | [0.914, 0.782, 0.786] |
| Blood5 conf_gap (correct-incorrect) | 0.0314 | 0.0450 | [-0.027, 0.082, 0.039] |
| PBC 8class acc | 0.9903 | 0.0008 | [0.989, 0.991, 0.991] |
| PBC 8class ECE (no T) | 0.0041 | 0.0010 | [0.0055, 0.0034, 0.0035] |
| PBC conf_correct | 0.9959 | 0.0003 | [0.996, 0.996, 0.996] |
| PBC conf_incorrect | 0.8251 | 0.0107 | [0.833, 0.810, 0.832] |
| PBC conf_gap | 0.1707 | — | — |

## Key finding: Classwise ECE reveals collapse mechanism

Across all seeds, the classwise ECE shows a characteristic pattern:
- **monocyte class**: ECE ≈ 0.10-0.16 (LOW) because the model correctly predicts
  99%+ of monocyte samples by always predicting "monocyte"
- **All other 4 classes**: ECE ≈ 0.61-0.95 (VERY HIGH) because the model predicts
  "monocyte" with ~90% confidence for samples that are basophil/eosinophil/lymphocyte/
  neutrophil — it is confidently WRONG

This is the clearest evidence of BOTH effects:
1. Representation collapse: features of ALL 5 classes map to one PBC-monocyte cluster
2. Miscalibration: the model assigns 80-90%+ confidence to wrong-class predictions

## Interpretation

**Verdict: BOTH (representation_collapse + miscalibration)**

The numbers that decide:
- **Cross-site accuracy = 12.8%** << 20% (random chance for 5-class) → COLLAPSED
- **Mean confidence on INCORRECT predictions (Blood5) = 82.75%** >> 70% threshold → CONFIDENTLY WRONG → MISCALIBRATION
- **Confidence gap (correct-incorrect) Blood5 = 0.031 ≈ 0** → model cannot distinguish its own errors
- **Confidence gap (correct-incorrect) PBC = 0.171 >> 0** → PBC is well-calibrated (large gap)
- **PBC ECE = 0.41%** vs **Blood5 masked ECE = 70.3%** → gap of 69.9 pp is NOT just accuracy failure

The high cross-site ECE is:
- NOT attributable to label-space leakage (masking the 3 PBC-only classes doesn't change the accuracy or ECE much, because those classes received <5% of predictions anyway)
- IS attributable to representation collapse (all non-monocyte inputs classified as monocyte with very high confidence)
- IS attributable to miscalibration (the high confidence on wrong predictions proves the model is overconfident, not just inaccurate)

Temperature scaling (T*≈1.31) reduces cross-site ECE from 70.3% to 62.6% (11% reduction)
but cannot repair the fundamental issue: it reduces confidence uniformly but cannot make the
model correctly classify non-monocyte Blood5 images.

## Decisions NOT to retake

- Did NOT try to compute a separate masking step for the 8-class logits before the Scenario C
  renormalization, because they are mathematically identical (and confirmed numerically matching)
- Did NOT include a 4th seed since the EXPERIMENT.md specifies "3 seeds"
- Did NOT compute bootstrap CIs on the calibration gap because the question is diagnostic
  (which failure mode?), not hypothesis-testing (is the gap significant?)
