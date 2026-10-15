You are a scientific-visualisation expert. Build ONE self-contained, polished visual report (an interactive HTML page or a slide-style deck, whichever you produce best) that explains a research method for influenza forecasting and shows how it performs. The audience is a university faculty panel (computer science / AI), and some members do not know epidemiology. Use clean, consistent colours; label every axis; one main message per figure, stated in the figure title. Use only the numbers given below: do not invent or round them differently.

## 1. The problem
- Task: forecast weekly influenza-like illness (ILI, % of doctor visits) for US regions 1, 2, 3, 4, 6 and 8 weeks ahead.
- Data: CDC ILINet only, on two datasets:
  - US-Regions: 10 HHS regions, weighted % ILI, 2002-2026;
  - US-States: 49 states, unweighted % ILI, 2010-2025.
- Four test tracks:
  - regions_t1: pre-COVID, test seasons 2014-15 to 2019-20;
  - regions_t2: post-COVID, test 2023-24 to 2025-26;
  - states_t1: pre-COVID, test 2016-17 to 2019-20;
  - states_t2: post-COVID, test 2023-24, 2024-25.
- Pandemic and COVID seasons are excluded as targets.
- Metric: RMSE on the original % scale, pooled over regions and weeks, in-season weeks (epiweek 40-20). Neural models use 5 seeds. Significance: Diebold-Mariano test.
- Baselines (8, all re-run on identical data): ARIMA, SARIMA, AR, LSTM, GRU, CNNRNN-Res, Cola-GNN, EpiGNN. Persistence ("next week = this week") is a reference line.
- Advisor's goal: decent accuracy plus an explainability advantage.

## 2. The method: NEE-GNN, a Neural Endemic-Epidemic forecaster with identified explanations
Draw an architecture diagram with two stages:
- **Stage 1, the identified core (statistical, fitted by bounded least squares on training data only).** For region i and horizon h:
  forecast = base_i + seasonal_i(week of target) + own coefficient x ILI_i(this week) + national coefficient x US-mean ILI(this week) + importation coefficient x sum over neighbours j of W_ij x ILI_j(this week)
  - W = border graph with power-law decay over the number of borders crossed.
  - The three spread coefficients are constrained >= 0 (as in the hhh4 endemic-epidemic model of Held 2005 / Meyer & Held 2014).
  - The core fixes HOW each forecast splits into parts, and a season-block bootstrap gives confidence intervals for that split.
- **Stage 2, neural modulation.** A small GRU reads each region's own last 20 weeks (shape-normalised, with a bounded level and the week of the season) and outputs a multiplier m_i(t) >= 0. It starts at exactly 1 and scales only the epidemic part (own + national + importation). It can make the forecast adapt week by week but can never change the split between own, national and importation.
- **Output:** every forecast is an EXACT sum of 5 named parts: base, seasonal, own momentum, national, importation from each neighbour (a who-contributes-to-whom matrix).
- **Why this design:** an ablation with fully free neural weights reached similar accuracy, but its explanation depended on the training start. For example, on states_t1 at 4 weeks the seasonal/own/spatial shares were 5/77/11% from one start and 34/23/34% from another, at similar RMSE. The explanation was not identifiable. Constraining the split to the statistical core fixes this.

## 3. Accuracy results: test RMSE, in-season (ours = mean of 5 seeds)
| track | h | ours | core alone | best baseline | persistence | ours vs best | DM p |
|---|---|---|---|---|---|---|---|
| regions_t1 | 1 | 0.575 | 0.551 | SARIMA 0.517 | 0.573 | +11.3% | 0.040 |
| regions_t1 | 2 | 0.929 | 0.855 | SARIMA 0.803 | 0.934 | +15.7% | 0.133 |
| regions_t1 | 3 | 1.172 | 1.069 | SARIMA 1.026 | 1.214 | +14.2% | 0.288 |
| regions_t1 | 4 | 1.300 | 1.227 | SARIMA 1.199 | 1.442 | +8.4% | 0.597 |
| regions_t1 | 6 | 1.461 | 1.423 | SARIMA 1.473 | 1.842 | -0.8% | 0.943 |
| regions_t1 | 8 | 1.500 | 1.500 | EpiGNN 1.587 | 2.182 | -5.5% | 0.224 |
| regions_t2 | 1 | 0.629 | 0.641 | Cola-GNN 0.583 | 0.689 | +7.8% | 0.080 |
| regions_t2 | 2 | 0.988 | 1.028 | EpiGNN 0.980 | 1.184 | +0.8% | 0.372 |
| regions_t2 | 3 | 1.249 | 1.239 | EpiGNN 1.201 | 1.524 | +4.0% | 0.153 |
| regions_t2 | 4 | 1.266 | 1.346 | EpiGNN 1.285 | 1.760 | -1.5% | 0.711 |
| regions_t2 | 6 | 1.498 | 1.502 | EpiGNN 1.453 | 2.148 | +3.0% | 0.181 |
| regions_t2 | 8 | 1.586 | 1.615 | EpiGNN 1.768 | 2.505 | -10.3% | 0.027 |
| states_t1 | 1 | 0.846 | 0.849 | SARIMA 0.824 | 0.820 | +2.6% | 0.358 |
| states_t1 | 2 | 1.229 | 1.229 | SARIMA 1.159 | 1.239 | +6.0% | 0.358 |
| states_t1 | 3 | 1.487 | 1.487 | SARIMA 1.408 | 1.565 | +5.6% | 0.459 |
| states_t1 | 4 | 1.658 | 1.658 | SARIMA 1.603 | 1.834 | +3.4% | 0.660 |
| states_t1 | 6 | 1.878 | 1.877 | SARIMA 1.871 | 2.284 | +0.4% | 0.962 |
| states_t1 | 8 | 1.980 | 1.979 | SARIMA 2.004 | 2.673 | -1.2% | 0.903 |
| states_t2 | 1 | 0.739 | 0.741 | Cola-GNN 0.685 | 0.762 | +7.9% | 0.021 |
| states_t2 | 2 | 1.129 | 1.136 | Cola-GNN 1.147 | 1.263 | -1.6% | 0.912 |
| states_t2 | 3 | 1.342 | 1.358 | EpiGNN 1.395 | 1.610 | -3.8% | 0.371 |
| states_t2 | 4 | 1.448 | 1.474 | EpiGNN 1.511 | 1.852 | -4.1% | 0.333 |
| states_t2 | 6 | 1.575 | 1.618 | EpiGNN 1.631 | 2.268 | -3.4% | 0.515 |
| states_t2 | 8 | 1.685 | 1.739 | EpiGNN 1.838 | 2.667 | -8.3% | 0.523 |

Seed ensembles: ours ranks 1st of 9 models in 10/24 cells and in the top 2 in 19/24. Significant: 2 losses (both 1 week ahead) and 1 win (regions_t2, 8 weeks, -10.3%).

**Consistency (gap to the best baseline in each of the 24 cells; key message: no single baseline is good in both regimes, ours is never far off):**
| model | worst gap | median gap | cells within 5% of best | cells beating best baseline |
|---|---|---|---|---|
| ours | 15.7% | 0.8% | 16/24 | 10/24 |
| core alone | 9.9% | 3.0% | 18/24 | 9/24 |
| SARIMA | 23.2% | 0.2% | 15/24 | - |
| EpiGNN | 38.0% | 3.5% | 12/24 | - |
| Cola-GNN | 29.2% | 12.3% | 4/24 | - |
| GRU | 28.1% | 16.0% | 1/24 | - |
| LSTM | 26.7% | 15.6% | 0/24 | - |
| ARIMA | 35.0% | 17.3% | 1/24 | - |
| AR | 39.4% | 23.0% | 0/24 | - |
| CNNRNN-Res | 53.1% | 26.8% | 0/24 | - |
SARIMA is best pre-COVID and weak post-COVID (seasonal timing changed); EpiGNN / Cola-GNN are best post-COVID and weak pre-COVID.

## 4. What the built-in explanation says (real data)
**Shares of the forecast (base / seasonal / own / national / importation, %), mean over seeds:**
| track | h=1 | h=2 | h=3 | h=4 | h=6 | h=8 |
|---|---|---|---|---|---|---|
| regions_t1 | 9/6/72/0/12 | 19/13/55/6/7 | 30/20/40/11/0 | 40/25/31/4/0 | 56/32/12/0/0 | 65/35/0/0/0 |
| regions_t2 | 7/5/79/0/9 | 13/10/66/11/0 | 20/15/54/12/0 | 26/19/45/10/0 | 42/27/31/0/0 | 51/31/18/0/0 |
| states_t1 | 11/7/66/0/15 | 19/14/54/12/2 | 24/19/41/16/0 | 29/23/32/16/0 | 41/29/22/8/0 | 50/33/18/0/0 |
| states_t2 | 7/6/75/5/8 | 12/11/64/14/0 | 16/15/54/15/0 | 19/18/45/17/0 | 28/24/34/13/0 | 39/29/31/1/0 |
Message: short-range forecasts are driven by the region's own momentum; long-range forecasts become climatology (base + seasonal); spread from neighbours matters only about one week ahead.

**Spread share with uncertainty** (cross-region share of the epidemic part, 1 week ahead, season-block bootstrap 90% interval):
| track | share [90% interval] |
|---|---|
| regions_t1 | 0.14 [0.08, 0.22] |
| regions_t2 | 0.10 [0.05, 0.13] |
| states_t1 | 0.18 [0.00, 0.28] |
| states_t2 | 0.14 [0.11, 0.17] |
At 4-8 weeks the intervals are about [0, 0.5], so the spread share is not identifiable there. Show this honestly.

**Explanation quality vs a black-box GNN explained afterwards.** The black box is GCN-GRU, a standard spatio-temporal GNN, explained by occlusion and by Integrated Gradients (IG).
- Stability across training seeds (Pearson of cross-region contributions / agreement on the top source region):

| track | h | ours | black box + occlusion | black box + IG |
|---|---|---|---|---|
| regions_t1 | 1 | 1.00 / 1.00 | 0.85 / 0.67 | 0.84 / 0.67 |
| regions_t1 | 4 | 0.89 / 1.00 | 0.97 / 0.79 | 0.97 / 0.80 |
| regions_t2 | 1 | 1.00 / 1.00 | 0.73 / 0.67 | 0.79 / 0.69 |
| regions_t2 | 4 | 0.93 / 1.00 | 0.90 / 0.68 | 0.91 / 0.72 |
| states_t1 | 1 | 1.00 / 1.00 | -0.31 / 0.96 | -0.32 / 0.96 |
| states_t1 | 4 | 1.00 / 1.00 | 0.80 / 0.78 | 0.83 / 0.79 |
| states_t2 | 1 | 1.00 / 1.00 | 0.84 / 0.74 | 0.89 / 0.79 |
| states_t2 | 4 | 1.00 / 0.99 | 0.68 / 0.60 | 0.67 / 0.60 |

- Completeness (do the attributions add up to the forecast? 0 = exact): ours exact (storage error < 0.004 % ILI); occlusion 0.027; IG 0.001.

## 5. Semi-synthetic benchmark: is the explanation CORRECT?
**Simulator:** a stochastic flu epidemic on the 49 states (Census populations) where the true share of each week's new infections coming from other states is known.
- c = fraction of transmission between states.
- True network: "gravity" (population x distance) or "longrange" (3 random air-travel-like partners, invisible to a border graph).
- Methods: ours (border graph), ours (gravity kernel), the free-weight ablation, and the black box + occlusion / IG.

**Estimated imported share vs truth:**
| scenario (true share) | ours, border | ours, gravity | free-weight ablation | black box + occlusion | black box + IG |
|---|---|---|---|---|---|
| no spread, c=0 (0.000) | 0.129 | 0.132 | 0.141 | 0.130 | 0.121 |
| gravity c=0.1 (0.108) | 0.172 | 0.208 | 0.243 | 0.301 | 0.274 |
| gravity c=0.3 (0.304) | 0.329 | 0.472 | 0.198 | 0.327 | 0.317 |
| longrange c=0.1 (0.115) | 0.154 | 0.176 | 0.145 | 0.185 | 0.175 |
| longrange c=0.3 (0.305) | 0.194 | 0.232 | 0.148 | 0.199 | 0.165 |

**Correlation of the estimated with the true imported share over state-weeks (does the explanation know WHEN and WHERE spread happens?):**
| scenario | ours, border | ours, gravity | black box + occlusion | black box + IG |
|---|---|---|---|---|
| gravity c=0.1 | 0.36 | 0.79 | -0.03 | 0.00 |
| gravity c=0.3 | 0.54 | 0.73 | 0.06 | 0.06 |
| longrange c=0.1 | 0.49 | 0.76 | 0.20 | 0.24 |
| longrange c=0.3 | 0.70 | 0.68 | 0.07 | 0.11 |

**Top-3 source overlap with the true top-3** (chance = 0.06):
- gravity scenarios: ours (gravity) 0.66-0.68, ours (border) 0.49-0.56, black box 0.55-0.57;
- longrange scenarios: all methods 0.05-0.09 (the graph is wrong, as expected).

Key messages: the built-in explanation tracks the true spread over time while black-box post-hoc explanations do not; the free-weight ablation gives different answers depending on its training start.

## 6. Honest limitations (show them)
1. With no spread at all, every method reports about 0.13 spread: a shared season looks like spread.
2. Long-range spread is underestimated by any border-graph method.
3. Accuracy is decent but not state of the art: worst gap 15.7%, with 2 significant losses at 1 week ahead.
4. The neural stage adds little over the statistical core on real data.
5. Baselines use their repository defaults (untuned); forecasts are point forecasts only.

## 7. What is new (closing slide)
1. A neural extension of the endemic-epidemic (hhh4) model in which a neural network modulates an identified statistical core. Every forecast splits exactly into base / seasonal / own / national / per-neighbour importation, with confidence intervals.
2. Evidence that explanations of flexible epidemic forecasters are not identifiable: equally accurate models give different explanations.
3. The first semi-synthetic benchmark with known spread routes for scoring explanations of epidemic forecasts, showing that post-hoc explanations of a black-box GNN do not track the true spread.
4. Decent accuracy that holds across the pre- and post-COVID regimes, where each baseline is good in only one.

## Figures to produce (at least)
1. Architecture diagram (two stages, the five output parts, what the neural part can and cannot change).
2. Data/tracks timeline (training / validation / test seasons, COVID gap).
3. Heatmap of "ours vs best baseline" (%) over 4 tracks x 6 horizons, with significant cells marked.
4. Line charts of RMSE vs horizon per track: ours, core alone, best baseline, persistence.
5. Consistency chart: worst gap and cells within 5% for all 10 models.
6. Stacked-area chart of explanation shares vs horizon per track.
7. Bar chart of spread share at 1 week with 90% intervals.
8. Stability comparison chart (ours vs occlusion vs IG).
9. Synthetic benchmark: estimated vs true imported share (grouped bars with the truth as a line) and the correlation table as a heatmap.
10. A limitations panel and a "what is new" summary panel.
