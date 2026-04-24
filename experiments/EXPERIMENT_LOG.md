# AdaSR-Face: Experiment Log

**Objective:** Push face detection beyond current best (F1=0.802, AP=0.703) using
Adaptive Super-Resolution guided by detection confidence.

**Hypothesis:** 94% of missed faces are <32×32 px. Selectively applying SR to
low-confidence/small-face regions will recover these misses without the cost of
blind full-image SR.

---

## Timeline

### Phase 0 — Baseline Established (2026-04-24)
- **Current best:** RetinaFace (det_10g) + Tiled(640, 0.25) + MultiScale[0.75, 1.0, 1.5]
- P=0.921, R=0.711, F1=0.802, AP=0.703
- Failure analysis: 94.1% of 11,322 missed faces are <32×32 px
- 78.8% of misses are <16×16 px, 53.7% are <10×10 px
- Heavy blur recall: 0.565, Heavy occlusion: 0.405
- 26% of missed faces are in dark images (V<80)

### Phase 1 — SCRFD Model Swap (2026-04-24)
- **Goal:** Replace det_10g with SCRFD-10GF (or SCRFD-34GF) — same insightface API
- **Expected:** +10-15% AP on Hard subset
- **Status:** Building...

### Phase 2 — Super-Resolution Integration (2026-04-24)
- **Goal:** Add Real-ESRGAN as a preprocessing step
- Blind SR (full image) as upper bound
- Selective SR (confidence-guided) as novel contribution
- **Expected:** +5-10% recall on <32px faces
- **Status:** Building...

### Phase 3 — Adaptive Cascade Pipeline (2026-04-24)
- **Goal:** Stage 1 fast detect → identify weak regions → SR only those → re-detect
- This is the novel contribution for the paper
- **Status:** Building...

---

## Results Log

| # | Experiment | Precision | Recall | F1 | AP | Time(s) | Notes |
|---|-----------|-----------|--------|------|------|---------|-------|
| B0 | det_10g baseline | 0.957 | 0.511 | 0.666 | 0.507 | — | Original baseline |
| B1 | det_10g + Tiled + MS | 0.921 | 0.711 | 0.802 | 0.703 | — | Current best |
| E1 | E1_scrfd_baseline | 0.989 | 0.521 | 0.683 | 0.520 | 2 | none |
| E2 | E2_scrfd_tiled_ms | 0.933 | 0.764 | 0.840 | 0.760 | 25 | none |
| E3 | E3_scrfd_bicubic | 0.989 | 0.524 | 0.685 | 0.523 | 3 | bicubic |
| E4 | E4_scrfd_blind_sr | 0.991 | 0.489 | 0.655 | 0.489 | 334 | blind |
| E5 | E5_scrfd_adaptive_sr | 0.868 | 0.636 | 0.734 | 0.629 | 307 | adaptive |

---


### E1_scrfd_baseline — 2026-04-24 15:09:13
- **Description:** SCRFD (det_10g) at native resolution, no enhancements
- **Detector:** scrfd | SR: none | Tiled: False | MS: False
- **Results:** P=0.9885 R=0.5212 F1=0.6825 AP=0.5198
- **TP=172 FP=2 FN=158** | Time=2s
- **Recall by face size:**
  - 0-10px: 0.057 (6/106)
  - 10-16px: 0.600 (69/115)
  - 16-32px: 0.857 (60/70)
  - 32-64px: 0.882 (15/17)
  - 64-128px: 1.000 (21/21)
  - 128+px: 1.000 (1/1)


### E2_scrfd_tiled_ms — 2026-04-24 15:09:38
- **Description:** SCRFD + Tiled(640, 0.25) + MultiScale[0.75, 1.0, 1.5]
- **Detector:** scrfd | SR: none | Tiled: True | MS: True
- **Results:** P=0.9333 R=0.7636 F1=0.8400 AP=0.7599
- **TP=252 FP=18 FN=78** | Time=25s
- **Recall by face size:**
  - 0-10px: 0.396 (42/106)
  - 10-16px: 0.922 (106/115)
  - 16-32px: 0.943 (66/70)
  - 32-64px: 0.941 (16/17)
  - 64-128px: 1.000 (21/21)
  - 128+px: 1.000 (1/1)


### E3_scrfd_bicubic — 2026-04-24 15:10:46
- **Description:** SCRFD + Bicubic 2x upscale (control, no learned SR)
- **Detector:** scrfd | SR: bicubic | Tiled: False | MS: False
- **Results:** P=0.9886 R=0.5242 F1=0.6851 AP=0.5229
- **TP=173 FP=2 FN=157** | Time=3s
- **Recall by face size:**
  - 0-10px: 0.075 (8/106)
  - 10-16px: 0.591 (68/115)
  - 16-32px: 0.857 (60/70)
  - 32-64px: 0.882 (15/17)
  - 64-128px: 1.000 (21/21)
  - 128+px: 1.000 (1/1)


### E4_scrfd_blind_sr — 2026-04-24 15:35:10
- **Description:** SCRFD + Real-ESRGAN blind SR on full image (upper bound)
- **Detector:** scrfd | SR: blind | Tiled: False | MS: False
- **Results:** P=0.9915 R=0.4895 F1=0.6554 AP=0.4886
- **TP=116 FP=1 FN=121** | Time=334s
- **Recall by face size:**
  - 0-10px: 0.084 (7/83)
  - 10-16px: 0.626 (62/99)
  - 16-32px: 0.837 (41/49)
  - 32-64px: 1.000 (1/1)
  - 64-128px: 1.000 (5/5)
  - 128+px: 0.000 (0/0)


### E5_scrfd_adaptive_sr — 2026-04-24 15:54:32
- **Description:** SCRFD + AdaSR-Face (confidence-guided selective SR) — NOVEL
- **Detector:** scrfd | SR: adaptive | Tiled: False | MS: False
- **Results:** P=0.8675 R=0.6359 F1=0.7339 AP=0.6286
- **TP=131 FP=20 FN=75** | Time=307s
- **Recall by face size:**
  - 0-10px: 0.253 (19/75)
  - 10-16px: 0.793 (73/92)
  - 16-32px: 1.000 (39/39)
  - 32-64px: 0.000 (0/0)
  - 64-128px: 0.000 (0/0)
  - 128+px: 0.000 (0/0)
- **Cascade stats:** avg SR regions/image=42.0, avg new faces from SR=14.7

## Key Decisions & Observations

*(Updated incrementally as experiments run)*

