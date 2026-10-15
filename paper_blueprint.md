# Paper Blueprint (v5)

**Target:** *PLOS Computational Biology* (primary), *International Journal of Forecasting* or *Epidemics*. *Nature Communications* is a stretch if the prospective result is strong. About 7,000 words plus supplement.

**Workshop cut:** a 4-page KDD epiDAMIK / ML4H version on PLC + the CDC ensemble.

`[X]` placeholders are filled from `results/`; every number must trace to `run_all.sh`.

---

## Title options
1. *Forecasting from living data: the surveillance revision record as supervision for real-time influenza forecasting*
2. *Learning from data that keeps changing: revision-aware inputs, feedback and explanations for epidemic forecasts*
3. *The CDC influenza forecasts are under-covered in real time, and the data's own revision history fixes it*

## Abstract (template, about 250 words)

> **Background.** Epidemic surveillance numbers keep changing after publication. Forecasting systems treat these revisions as a nuisance: they clean inputs and evaluate against final values afterwards. They miss three consequences. Models cannot tell reliable from unreliable inputs. Online recalibration must learn from labels that are still being revised. Users cannot tell whether a forecast moved because reality changed or because the data were revised.
> **Methods.** We treat the revision record (every value as published each week, and every later correction) as supervision, and develop three instruments:
> (i) reliability-aware inputs: a plug-in for any forecaster, and a learned, monotone Kalman gain inside a strain-resolved renewal model;
> (ii) provisional-label conformal calibration (PLC), which recalibrates any forecaster online from revisable labels via soft errors and retroactive correction, and is provably valid against final labels;
> (iii) an exact decomposition of week-to-week forecast changes into data revisions, new data and dynamics.
> We evaluate them on two US influenza surveillance systems with opposite revision regimes: ILINet (16 seasons, mild revisions) and NHSN hospital admissions (the official CDC target; first reports `[13]%` too low on average). The evaluation includes the official CDC FluSight ensemble, retrospectively (2024-25, 2025-26) and in a pre-registered prospective test (2026-27).
> **Results.** The CDC ensemble's 90% intervals covered `[82]%` of final values in real time. Recalibration with provisional labels left a `[2.5]`-point shortfall, and waiting for final labels was unstable. PLC achieved `[90.0]%` (prospectively: `[X]%`). Reliability-aware inputs reduced WIS by `[X]%` for gradient-boosted quantile regression and by `[X]%` for the mechanistic forecaster versus classical assimilation. The decomposition was exact, more faithful than SHAP, and predicted forecast failures. Benefits scaled with revision severity.
> **Conclusions.** Surveillance revisions are a resource, not just noise: they tell forecasters what to trust, how to recalibrate, and why forecasts change.

## 1. Introduction (about 900 words)
1. Real-time forecasting supports public-health planning (FluSight). Its data are living data: P6 revision statistics, the NHSN 2024-25 regime shift.
2. Three unaddressed consequences (inputs, feedback, explanation), each with a number from our pilots.
3. Prior work treats revisions as an input-cleaning problem (nowcasting: NobBS, Delphi-RF, Back2Future) or as evaluation leakage (Bracher & Funk 2026). The calibration dilemma was recently posed in econometrics (Kuang 2026) with a heuristic.
4. Our reframing and the three instruments; the two surveillance systems; the CDC ensemble; the prospective pre-registered test.
5. Contributions (C1–C6, worded according to actual results).

## 2. Related work (about 800 words)
- Real-time data and revisions: nowcasting; real-time econometrics; Bracher & Funk 2026; Kuang 2026.
- Online conformal and calibration under shift and delay: ACI, conformal PID, delayed ACI (El Halabi & Brandt 2026), MultiQT, Shen et al. 2025 (epidemic quantile tracking against provisional truth), Rumack 2022, NCC.
- Online learning with evolving or proxy feedback: Mann 2019, Flaspohler 2021, Bar-On & Mansour 2025.
- Data assimilation for influenza (Shaman EAKF), learned gains (KalmanNet), hybrid models (EINNs, EARTH, Alleman 2026).
- Explaining forecast changes: macro news decomposition, weather observation impact, DeltaSHAP; epidemic XAI (levels only).
- Scope: no genomics (supplement S1, P2 evidence); no GNN.

## 3. Data (about 600 words + T1)
- ILINet vintages (Delphi), provider counts, WHO/NREVSS lab data.
- NHSN vintages (FluSight hub), with the release-timing rule.
- Hub forecasts (ensemble + 3 models).
- **T1: revision regimes** (severity, bias, by season and system).
- Prospective archive with hashes.

## 4. Methods (about 2,200 words)
- 4.1 Notation for living data: vintages, lags, availability rules, the "final" definition.
- 4.2 **I2 PLC**: calibrator, soft errors, retroactive correction, guarantee (Theorem 1 + proof sketch; full proof in the supplement), comparators.
- 4.3 **I1 reliability-aware inputs**: revision mean and variance model; R-GBQR; QKA inside DRIFT (brief architecture; full spec in the supplement).
- 4.4 **I3 exact explanations**: Shapley channels, trust attribution, evaluation design.
- 4.5 Evaluation design: splits, prospective protocol and pre-registration, metrics (MACE, local MACE, WIS, θ), statistics (block bootstraps, Holm, MDE), leakage audit.

## 5. Results (about 2,000 words)
- 5.1 **The CDC ensemble in real time** (T2, F2): raw coverage; naive vs wait vs PLC vs oracle; week-by-week coverage through NHSN 2024-25; hub models; **prospective 2026-27**.
- 5.2 Why naive fails: coverage gap vs revision bias by week (F3).
- 5.3 Severity scaling: ILINet vs NHSN (S-C3).
- 5.4 Reliability-aware inputs on ILINet (T3): R-GBQR vs GBQR; DRIFT/QKA vs constant-noise KF vs GBQR vs B7; learned trust map (F4).
- 5.5 Exact explanations (T5, F5): waterfall examples (holiday week; 2022-23); faithfulness; usefulness; observation impact.
- 5.6 Ablations, the 2 × 2 (answering Alleman 2026), identified parameters (supplement).

## 6. Discussion (about 900 words)
- **Operational implications:** hubs should recalibrate with PLC; report coverage against final values; publish revision records.
- What revisions tell us about surveillance quality (observation impact; when to distrust data).
- **Limitations:**
  - the "final" definition (lag 10);
  - only 2 retrospective NHSN seasons (hence the prospective test);
  - pilots touched the retrospective seasons (disclosed);
  - wILI is a proxy; US only.
- **Future work:** the theory of local adaptivity (Path B); other diseases and hubs (COVID, RSV); macroeconomic data.

## 7. Reproducibility
Code, as-of datasets, the prospective archive with SHA-256 log, the OSF pre-registration, and recalibrated forecasts in hub format.

---

## Reviewer objections, pre-empted

| Objection | Answer |
|---|---|
| "Kuang (2026) already posed this calibration problem." | Yes, in econometrics, with a hard switch heuristic and no guarantee. PLC adds soft revision-model errors, retroactive correction and validity against final labels for any revision model. We include Kuang's rule as a comparator (Kuang-hard). |
| "The validity bound is a corollary of delayed ACI." | The bound is not our main claim; the empirical and operational result is. The bound shows PLC is never worse in the long run, and the experiments show it adapts like an oracle while waiting does not. Local-adaptivity theory is left for future work. |
| "Your pilots used the test seasons." | Disclosed. The retrospective tests confirm pilot results with a frozen pipeline; the **pre-registered 2026-27 prospective test** is the independent evaluation. |
| "Just widen the intervals offline." | Rumack-recal and wait are comparators. Offline recalibration cannot track within-season regime shifts such as NHSN 2024-25. |
| "The gains on ILINet are small." | Expected (mild revisions); that is the severity-scaling result (S-C3), not a failure. |
| "Is the 'final' value final?" | Not entirely: lag 10 captures 81% of the eventual NHSN revision (P6). The results are unchanged against the latest vintage (PLC 0.895–0.901, naive 0.863–0.868), and PLC with L = 20 is reported as a sensitivity analysis. |
| "SHAP already explains forecasts." | We explain *changes*, exactly, separating revisions from new data; faithfulness and usefulness are tested (X1, X2). |
| "Why no genomics / GNN / foundation model?" | Supplement S1 (P2: 117-day delay); SpatialEpiBench / Thivierge; Chronos baseline included. |

## Figures

| ID | Content |
|---|---|
| F1 | The framework: the revision record (vintage triangle) feeding I1 inputs, I2 feedback and I3 explanations |
| **F2** | CDC ensemble, NHSN 2024-26 and 2026-27: rolling 8-week 90% coverage over time for raw / naive / wait / PLC / oracle, with the revision bias shaded |
| F3 | Coverage shortfall of naive vs the weekly mean revision bias (scatter by week × location) |
| F4 | Learned trust map (R-GBQR λ; QKA gain vs revision spread, holiday, lab n) |
| F5 | Explanation waterfall + faithfulness bars |
| F6 | Severity scaling: [MACE(naive) − MACE(PLC)] vs revision severity across systems, seasons and locations |
| S1 | Genomic feasibility (P2) |
| S2 | η sweeps (P5, P6) |
| S3 | ILINet forecasting tables and the 2 × 2 |
