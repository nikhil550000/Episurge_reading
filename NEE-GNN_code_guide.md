# NEE-GNN: code guide for `code/nee_phase2.py`

> **What this is.** A walk through every cell and important function of [code/nee_phase2.py](../../code/nee_phase2.py): the notebook that produced all the results in the method guide. For each piece: what it does, what each line means, the array shapes, why it is written that way, and how to run, debug and change it.
>
> **Read first:** `NEE-GNN_complete_guide.md` (what the model does). This document explains **how the code does it**.
>
> **Assumed knowledge:** basic Python, NumPy arrays, and what a PyTorch model, optimizer and loss are. Less common idioms are explained where they first appear and in the glossary (section 8). Links jump to exact lines (Ctrl+click in VS Code).

**Two naming notes before you start**
1. The code uses two internal names that you will see in printouts:
   - **`v3` / variant `id`** = **our method** (identified core + neural multiplier);
   - **`v2` / variant `full`** = the **free-weight ablation** (the neural network sets every weight; used to show such explanations are not identified).
2. **Lines 1–18** are the header. The notebook has **19 cells**:
   - **cells 1–8** build data, graph, model and training (identical to `code/nee_gnn.py` cells 1–8);
   - **cells 9–19** are the experiments.

**Contents**
1. The big picture
2. Conventions used everywhere
3. The cells, one by one (1–19)
4. How to run it, and what each cell should print
5. Common errors and fixes
6. How to change things (recipes)
7. The data flow in one table
8. Glossary of Python / NumPy / PyTorch idioms
9. Code questions you may be asked

---

## 1. The big picture

### 1.1 What the notebook does
```
Cells 1-5   DATA LAYER       read CDC CSVs -> panels, flags, splits, scalers, windows, metrics; load baselines; DM test
Cells 6-8   MODEL LIBRARY    graph inputs; the models (ours = NEE3, ablation = NEE); training + saving one run
Cell 9      RUNS (ours)      our method, the core alone, the free-weight ablation (2 starts), all saved to disk
Cells 10-11 BLACK BOX        a standard spatio-temporal GNN (GCN-GRU) + two post-hoc explainers (occlusion, IG)
Cells 12-14 RESULTS          accuracy vs baselines; identifiability + bootstrap; explanation quality
Cell 15     FIGURES          decomposition plot, shares-vs-horizon plot
Cells 16-18 GROUND TRUTH     simulated epidemics with known spread; train everything on them; score recovery
Cell 19     BACKUP           zip everything for download
```

### 1.2 Why one `.py` file with `# %% Cell N` blocks
- Heavy runs happen on **Kaggle** (free GPU); the laptop is too weak.
- You copy each `# %% Cell N` block into one Kaggle cell and run them in order.
- Each cell starts with a header (`What it does / Needs`) naming its inputs.

### 1.3 Where things come from
| Input | Where | Used for |
|---|---|---|
| `ILINet_US_Regions.csv`, `ILINet_US_States.csv` | Kaggle dataset `influenza-datasets` | the data |
| `region-adj.txt`, `state-adj.txt`, `region785.txt`, `state360.txt` | downloaded from the Cola-GNN GitHub repo (Cell 2) | the border graph; data checks |
| `CenPop2010_Mean_ST.txt` | downloaded from the US Census Bureau (Cell 6) | population and centroids (gravity variant, simulator) |
| `baselines_backup.zip` | the Kaggle dataset you uploaded (made by `baselines_all.py`) | the 8 baselines' saved predictions |

---

## 2. Conventions used everywhere

### 2.1 Names you will see constantly
| Name | Type / shape | Meaning |
|---|---|---|
| `P` | dict: dataset → dict | **panels**: `P["regions"]`, `P["states"]`, and in Cell 16 also `P["syn_…"]` |
| `P[ds]["X_pct"]` | `[T, N]` | % ILI: T weeks × N nodes |
| `P[ds]["year"]`, `["week"]` | `[T]` | MMWR year and epiweek of each row |
| `P[ds]["season"]` | `[T]` | season = year of its epiweek 40 (2017 = 2017-18) |
| `P[ds]["wis"]` | `[T]` | **w**eek **i**n **s**eason, 0 at epiweek 40 |
| `P[ds]["valid"]` | `[T]` bool | allowed as a *target* (False in pandemic/COVID seasons) |
| `P[ds]["inseason"]` | `[T]` bool | epiweek 40–20, i.e. scored |
| `TRACKS`, `TR` | dict | the 4 experiments (`regions_t1`, `regions_t2`, `states_t1`, `states_t2`); `TR` adds per-node `min`/`max` scalers |
| `t` | int array | **origins**: row indices of the "now" week of each sample |
| `h` | int | horizon (1, 2, 3, 4, 6, 8) |
| `N` | int | nodes (10 or 49) |
| `WINDOW` | 20 | weeks of input per sample |
| `X`, `Y` | `[n, 20, N]`, `[n, N]` | scaled input windows and scaled targets |
| `GRAPH[ds]` | dict | `order` (hop order `[N,N]`), `order_rand`, `logpop` `[N]`, `logdist` `[N,N]` |
| `DEV` | device | `cuda` on Kaggle, `cpu` locally |
| `QUICK` | bool | smoke-test mode (`EPI_QUICK=1`) |
| `SEEDS`, `RUN_H` | lists | seeds and horizons actually run |
| `OUT`, `FIG` | paths | `results/nee_p2` and `results/nee_p2/figures` (set in Cell 9) |
| `MODEL` | `"NEE"` | file prefix |
| `MAIN` | `"id"` | our method's variant name |

### 2.2 Variant names (parsed by `flags`, Cell 7)
| Variant | Class | Meaning | Used in |
|---|---|---|---|
| `id` | `NEE3` | **our method** (border kernel) | Cells 9, 17 |
| `id-nomult` | `NEE3` | the statistical core alone (multiplier fixed at 1, no training) | Cells 9, 17 |
| `id-gravity` | `NEE3` | our method with the gravity kernel | Cell 17 |
| `full` | `NEE` | free-weight ablation, default start | Cells 9, 17 |
| `full-altinit` | `NEE` | free-weight ablation, a different (large) start | Cells 9, 17 |
| other suffixes: `-noimp`, `-nonat`, `-randgraph`, `-noshift`, `noseason` | | available in `flags` but **not run in this notebook** | — |

### 2.3 What a saved run looks like
Each run of our model or the ablation writes **one `.npz` file** (several NumPy arrays in one compressed file): `results/nee_p2/NEE-<variant>__<track>__h<h>__s<seed>.npz`.

| Key | Shape | Content |
|---|---|---|
| `pred` | `[n_test, N]` | forecast, % ILI (clipped at 0) |
| `true` | `[n_test, N]` | observed value at t + h, % ILI |
| `t` | `[n_test]` | origin indices (to recover dates, in-season flags) |
| `val_rmse` | scalar | validation RMSE (in-season) |
| `parts` | `[5, n_test, N]` | base / seas / own / nat / imp in % ILI; **they sum to the raw forecast** |
| `src` | `[n_test, N, N]` float16 | who-contributes-to-whom: `src[k, i, j]` = node j's contribution to node i's forecast |
| `core` | `[3]` | λ, ρ, φ (our method; NaN for the ablation) |
| `delta`, `d`, `tau` | | only meaningful for the free-weight ablation |
| `epochs`, `secs`, `clipped` | scalars | epochs trained, seconds, fraction of negative raw forecasts |

- The black-box files are `GCN__<track>__h<h>__s<seed>.npz`, with `pred`, `true`, `t`, plus the attribution matrices `occ` and `ig` and their completeness gaps.
- The baselines' files (inside `baselines_backup.zip`) use the same `pred` / `true` / `t` contract. That is why everything can be compared directly.

### 2.4 Resumability
- Every run function checks `if os.path.exists(path): return` first.
- If the Kaggle session stops, rerun Cells 1–8, then the long cell: finished runs are skipped.
- `MAX_HOURS` (default 11.5) stops long loops before Kaggle's 12-hour limit.

### 2.5 Self-checks
Cells 4, 6, 7 and 10 print `PASS` / `FAIL`. A FAIL means stop. Also, `run_one` **asserts** that the parts sum to the forecast and refuses to save otherwise.

### 2.6 Style you'll notice
- Short statements chained with `;` on one line: read them left to right.
- `s` instead of `self` in model classes.
- A trailing `_` in names (`tr_`, `r_`) only avoids clashing with another variable.

---

## 3. The cells, one by one

### Cell 1: imports, paths, experiment definition ([L20](../../code/nee_phase2.py#L20))
- `KAGGLE` is True when `/kaggle/working` exists.
- `DATA_DIR` is the Kaggle dataset folder, or the laptop folder as a fallback.
- `WORK` is where everything is written: `/kaggle/working`, or `$EPI_WORK` / `work` locally.
- `HORIZONS = [1,2,3,4,6,8]`, `WINDOW = 20`.
- `WEEKS53`: years with 53 epiweeks, needed to build a gap-free calendar.
- `EXCLUDED_SEASONS = {2008, 2009, 2020, 2021}`: never used as targets.
- `DATASETS`: per dataset, the CSV, which % ILI column, the **exact node list and order** (Cola-GNN's 49 states), and the start and end week.
- `TRACKS`: per experiment, which **seasons** are train / validation / test. A sample's role is decided by the season of its **target** week.

### Cell 2: reference files ([L63](../../code/nee_phase2.py#L63))
- `fetch(url, path)` ([L66](../../code/nee_phase2.py#L66)) downloads with 3 retries, skipping files that already exist.
- `ADJ_REF[ds]` is the border adjacency matrix (`[10,10]` or `[49,49]`, 1 on the diagonal).

### Cell 3: panels, flags, splits, helpers ([L83](../../code/nee_phase2.py#L83))
**`calendar`** ([L86](../../code/nee_phase2.py#L86)) lists every (year, week) from start to end. It is used to check that no week is missing.

**`load_panel`** ([L93](../../code/nee_phase2.py#L93)):
```python
df = pd.read_csv(..., skiprows=1, na_values=["X"], skipinitialspace=True)   # CDC file: line 1 is a title, "X" = missing
piv = df.pivot(index=["YEAR","WEEK"], columns="REGION", values=col)          # long table -> matrix: weeks x nodes
          .sort_index().loc[:, cfg["nodes"]]                                 # chronological; our fixed node ORDER
```
- `pivot` reshapes CDC's long format (one row per node per week) into a matrix.
- Selecting `cfg["nodes"]` drops unwanted nodes (Florida, DC, territories) and fixes the order to match the adjacency file.

**Flags** ([L109](../../code/nee_phase2.py#L109)):
- `season = year if week ≥ 40 else year − 1`;
- `wis` counts weeks since the last week 40;
- `valid` excludes the masked seasons and 2019-20 after week 10 of 2020;
- `inseason` = week ≥ 40 or ≤ 20.

**`origins(p, seasons, h)`** ([L122](../../code/nee_phase2.py#L122)), the definition of a sample:
```python
t = np.arange(W - 1, T - h)          # origins with 20 weeks of history and a target inside the data
tgt = t + h
return t[p["valid"][tgt] & np.isin(p["season"][tgt], list(seasons))]   # target allowed AND in the wanted seasons
```
- The input window may contain masked weeks (as history); the target may not.
- Nothing after `t` is ever an input, so there is no look-ahead.

**`scale` / `unscale`** ([L127](../../code/nee_phase2.py#L127)): min–max per node. `np.maximum(mx - mn, 1e-9)` avoids division by zero.

**`metrics`** ([L130](../../code/nee_phase2.py#L130)): RMSE, MAE and Pearson correlation, **pooled** over all nodes and weeks (`ravel` flattens).

**`TR`** ([L137](../../code/nee_phase2.py#L137)): per track, per-node `min` and `max` from **valid training target weeks only**, so the test data never affects scaling.

**`build_windows(track, h)`** ([L143](../../code/nee_phase2.py#L143)):
```python
Xs = scale(p["X_pct"], cfg["min"], cfg["max"]).astype(np.float32)   # [T, N]
t = origins(p, cfg[role], h, W)                                        # [n]
idx = t[:, None] + np.arange(-W + 1, 1)[None, :]                       # [n, 20] row indices t-19 .. t
out[role] = (Xs[idx], Xs[t + h], t)                                    # X [n,20,N], Y [n,N], t [n]
```
**Broadcasting + fancy indexing:** a column of origins plus a row of offsets gives an `[n, 20]` table of row numbers. Indexing `Xs` with it returns all windows at once, `[n, 20, N]`, with no Python loop.

- **`week_of_season_features`** ([L156](../../code/nee_phase2.py#L156)): sin and cos of 2π·wis/52.
- **`score`** ([L161](../../code/nee_phase2.py#L161)): metrics on in-season targets and on all targets.
- **`persistence`** ([L166](../../code/nee_phase2.py#L166)): the "next = now" reference.

### Cell 4: checks ([L177](../../code/nee_phase2.py#L177))
- Our counts equal Cola-GNN's files (≥ 780/785 rows for regions, because CDC revised two weeks; 360/360 for states).
- The adjacency matrices are square, symmetric, have self-loops, and have 16 and 103 edges.
- **Persistence must print 0.573 / 1.442 / 2.182** for regions_t1 at h = 1/4/8. That proves the data layer is identical to the baselines' notebook.

### Cell 5: baselines, DM test, compare ([L196](../../code/nee_phase2.py#L196))
**`load_baseline_runs`** ([L201](../../code/nee_phase2.py#L201)) unzips `baselines_backup.zip` and reads every `<model>__<track>__h<h>__s<seed>.npz` into a dict keyed by `(model, track, h, seed)`. Without the zip it returns `{}` and baseline columns show "-".

**`dm_test(e1, e2, h)`** ([L220](../../code/nee_phase2.py#L220)), the Diebold–Mariano test:
```python
d = (e1**2 - e2**2).mean(axis=1)                         # loss difference per origin (averaged over nodes)  [n]
gam(k) = autocovariance of d at lag k
var = (gam(0) + 2*sum(gam(k) for k in 1..h-1)) / n       # Newey-West: h-step errors overlap -> correlated
stat = mean(d)/sqrt(var) * sqrt((n+1-2h+h(h-1)/n)/n)     # Harvey-Leybourne-Newbold small-sample correction
p = 2 * t.sf(|stat|, n-1)                                # two-sided p-value (Student t)
```
- If the loss differences average to zero, the two forecasters are equally good.
- **Why the autocovariance terms:** consecutive h-step errors are correlated, and ignoring that would overstate significance.

**`compare(my_runs, base_runs, ...)`** ([L237](../../code/nee_phase2.py#L237)), per track and horizon, prints:
- your RMSE (mean ± sd over seeds) and persistence;
- the best baseline (lowest mean RMSE);
- the relative difference;
- the DM p-value of **your seed-averaged forecast vs the best baseline's seed-averaged forecast**.

### Cell 6: settings and graph inputs ([L272](../../code/nee_phase2.py#L272))
**Settings** ([L277–290](../../code/nee_phase2.py#L277)):

| Setting | Value |
|---|---|
| `DEV` | GPU if available |
| `SEEDS` | `[0,1,2,3,4]` |
| `MODEL` | `"NEE"` |
| `SAVE_SRC` | False here; Cell 9 sets it True |
| `RUN_H` | all horizons |
| `EPOCHS`, `PATIENCE` | 400, 40 |
| `MAX_HOURS` | 11.5 |
| `QUICK` | 2 epochs, 1 seed, horizons 1 and 4 |

**Census file and HHS map:** the 2010 population and population-weighted centroid per state. `HHS` lists the states of each region.

**`node_geo(ds)`** ([L305](../../code/nee_phase2.py#L305)) → population `[N]` and (lat, lon) `[N, 2]`. A region's population is the sum of its states; its centroid is their population-weighted average.

**`haversine(ll)`** ([L314](../../code/nee_phase2.py#L314)) → pairwise great-circle distances in km `[N, N]`, all pairs at once by broadcasting.

**`hop_order(A)`** ([L319](../../code/nee_phase2.py#L319)), breadth-first search with matrix products:
```python
O = inf, diagonal 0;  R = identity (reachable set);  B = A > 0
for k in 1..N:
    R2 = (R @ B) > 0        # reachable within k border crossings
    O[R2 & ~R] = k          # newly reached nodes get hop order k
    R = R | R2
```
Unreachable pairs stay `inf`, e.g. Alaska and Hawaii, which have no land neighbours.

**`random_graph(A)`** ([L326](../../code/nee_phase2.py#L326)): the same number of edges placed at random (a control; not used in this notebook's runs).

**`GRAPH[ds]`** ([L334](../../code/nee_phase2.py#L334)):
- `order`, `order_rand`;
- `logpop` = log(pop / mean pop);
- `logdist` = log(distance / 1000 km).

**Checks:** the HHS map covers 50 states + DC + PR exactly once, and all 49 nodes are in the Census file.

### Cell 7: the models ([L349](../../code/nee_phase2.py#L349))
**Constants:** `K_HARM = 4` (Fourier harmonics), `L_LAGS = 4` (ablation class only), `PARTS = ["base","seas","own","nat","imp"]`.

**`fourier(w)`** ([L355](../../code/nee_phase2.py#L355)): for week numbers `w`, returns 9 columns, `[1, sin(2πkw/52) (k=1..4), cos(2πkw/52) (k=1..4)]`. The core drops the leading 1 (`[:, 1:]`) because each node has its own intercept.

**`clim_coef(track)`** ([L359](../../code/nee_phase2.py#L359)): fits the 9 Fourier columns per node to the training weeks, giving each node's **climatology curve**. Two uses:
- the seasonal template of the ablation class;
- the "nothing unusual" reference for the black-box explainers.

**`flags(variant)`** ([L365](../../code/nee_phase2.py#L365)): parses the variant string (section 2.2) into `("id", kwargs)` → our class, or `("nee", kwargs)` → the ablation class.

#### The free-weight ablation class `NEE` ([L377](../../code/nee_phase2.py#L377))
The version where the neural network sets every weight. It is kept to show that such explanations are not identified.
- A GRU over each node's window outputs **5 numbers per node and week** ([forward L407](../../code/nee_phase2.py#L407)):
  - α (seasonal strength);
  - δ (seasonal phase shift, up to ±6 weeks, via `6·tanh`);
  - λ, ρ, φ.

  All except δ pass through `softplus`, so they are ≥ 0.
- `own` = a learned weighted average of the last 4 weeks (softmax lag weights `kap`, starting about 0.87 on the last week).
- `W()` ([L399](../../code/nee_phase2.py#L399)) builds the kernel from learned parameters:
  - `softmax(−d·log o)` = o^(−d), normalised over neighbours;
  - gravity: `τ·logpop − d·logdist`;
  - `masked_fill(−inf)` removes non-neighbours; `nan_to_num(0)` gives isolated nodes a zero row.
- `altinit=True` starts the seasonal and spatial heads large instead of small: the same model from a different starting point (variant `full-altinit`).

#### Our method, Stage I: kernel, design matrix, least squares
**`id_kernel(g, kern)`** ([L437](../../code/nee_phase2.py#L437)), the **fixed** kernel:
- border: `1/hop order` for reachable pairs, else 0;
- gravity: `pop_j^0.68 · dist^-0.95`;
- `none`: zeros.

Each row is divided by its sum. `np.maximum(…, 1e-12)` keeps isolated rows at 0.

**`core_design(X, wt, Wk, nat)`** ([L449](../../code/nee_phase2.py#L449)), the design matrix. Shapes for Regions:
```python
last = X[:, -1]                                   # [n, N]     this week's scaled values
F = fourier(wt)[:, 1:]                            # [n, 8]     Fourier of the TARGET week
I = broadcast_to(eye(N), (n, N, N))               # [n, N, N]  one-hot: row (sample k, node i) has 1 in column i -> intercepts
S = (eye(N)[None,:,:,None] * F[:,None,None,:])    # [n, N, N, 8]
      .reshape(n, N, -1)                          # [n, N, 8N] node i's 8 Fourier values only in node i's block
E = stack([last,                                  # own
           repeat(last.mean(1, keepdims=True), N, 1) * nat,   # national mean (zeroed if nat=False)
           last @ Wk.T], -1)                      # importation (W @ y)    -> [n, N, 3]
return concatenate([I, S, E], -1).reshape(n*N, -1)   # [n*N, 10 + 80 + 3] = [4170, 93] for regions_t1
```
- **The S trick:** an identity matrix with an extra axis, times F, puts each node's Fourier values only in that node's columns, so every node gets its own seasonal curve.
- **Row order** is sample-major: (sample 0, nodes 0..N−1), (sample 1, nodes 0..N−1), … The bootstrap in Cell 13 depends on this.

**`core_fit(X, wt, Y, Wk, nat, rows)`** ([L458](../../code/nee_phase2.py#L458)):
```python
A, y = core_design(X, wt, Wk, nat), Y.reshape(-1)    # targets flattened in the same row order
if rows is not None: A, y = A[rows], y[rows]         # bootstrap: keep only resampled rows
lb = full(A.shape[1], -inf); lb[-3:] = 0              # only lambda, rho, phi must be >= 0
c = lsq_linear(A, y, bounds=(lb, inf)).x              # SciPy bounded least squares
c[-2] *= nat; c[-1] *= bool(Wk.any())                 # an ablated term has an all-zero column -> set its coefficient to 0
return c                                              # [N intercepts | 8N seasonal | lambda rho phi]
```
`lsq_linear` solves **min ‖A·c − y‖²** within the bounds. The problem is **convex**: one optimum, no random start. **This one call is what makes the explanation identified.**

#### Our method, Stage II: `NEE3` ([L468](../../code/nee_phase2.py#L468))
```python
s.register_buffer("cf", coef); s.register_buffer("Wk", Wk)   # L475: buffers are stored and moved to GPU but NOT trained
s.gru = nn.GRU(4, H, 2, batch_first=True)                    # L476: 4 inputs/week, 64 hidden, 2 layers
s.head = nn.Linear(H, 1)                                     #        one number per node and week
s.head.weight.zero_(); s.head.bias.fill_(log(e - 1))         # L478: softplus(log(e-1)) = 1 -> multiplier starts at 1
```
`forward(x, sc, wt)` ([L480](../../code/nee_phase2.py#L480)), with `x` `[B,20,N]`, `sc` `[B,20,2]` (sin/cos of each input week) and `wt` `[B]` (target week):
```python
last = x[:, -1]                                               # [B, N]
xs = x.transpose(1, 2)                                        # [B, N, 20]  one sequence per node
z  = (xs - mean)/(std + 0.05)                                 # shape of the window, level removed
inp = cat([z, tanh(xs), sin, cos], -1).reshape(B*N, 20, 4)    # every node is one sequence
mult = softplus(head(gru(inp)[0][:, -1])).reshape(B, N)       # L485: GRU output at the last week -> m >= 0
F = [sin(2πkw/52), cos(2πkw/52)]  for the target week         # [B, 8]
l, q, p = cf[-3], cf[-2], cf[-1]                              # lambda, rho, phi (frozen)
parts = dict(base = cf[:N],                                   # L491
             seas = F @ cf[N:N+8N].reshape(N, 8).T,           # each node's seasonal curve
             own  = mult * l * last,
             nat  = mult * q * last.mean(1, keepdim=True),
             imp  = mult * p * (last @ Wk.T))
src = mult[...,None] * (p*Wk + q/N) * last[:, None, :] + diag(own)   # L494: [B, N, N] who-from-whom
return sum(parts.values()), parts, zeros                      # forecast = exact sum of the parts
```
- `cf` is ordered exactly like the design matrix columns, so with m = 1 `NEE3` reproduces Stage I exactly (Cell 7 checks this).
- `src`: off-diagonal (i, j) = node j's share of i's importation, plus its 1/N share of the national mean; the diagonal is i's own term. Each row sums to own + nat + imp.

**`make_model(variant, track, h)`** ([L497](../../code/nee_phase2.py#L497)):
- `full…` variants build the ablation class `NEE`;
- `id…` variants **run Stage I** on that track and horizon's training windows and wrap the result in `NEE3`.

**Self-checks** ([L504](../../code/nee_phase2.py#L504)):
1. **Stage I on synthetic data:** targets are generated exactly from the design matrix with λ, ρ, φ = 0.6, 0.1, 0.2. `core_fit` must recover them, and `NEE3` must reproduce the targets, i.e. the PyTorch model equals the design matrix.
2. **The ablation class:** for 5 configurations on the States graph, the parts sum to the forecast, kernel rows sum to 1 (0 for isolated nodes), and `src` rows sum to own + nat + imp.
3. It prints the nodes with no neighbours (Alaska, Hawaii).

### Cell 8: training and saving ([L526](../../code/nee_phase2.py#L526))
**`tensors(track, h)`** ([L530](../../code/nee_phase2.py#L530)): per split, PyTorch tensors on the device:
- `X` `[n,20,N]`;
- `sc` `[n,20,2]` (sin/cos of each window week);
- `wt` `[n]` (target week of season);
- `Y` `[n,N]`;
- plus the NumPy `t`.

**`fit(m, S)`** ([L539](../../code/nee_phase2.py#L539)), the one training routine for every neural model in the notebook:
```python
opt = Adam(m.parameters(), lr=1e-3, weight_decay=1e-5)   # for NEE3 only the GRU + head are parameters
for ep in range(EPOCHS):
    shuffle; for each mini-batch of 32:
        loss = mean((model(X_b)[0] - Y_b)**2)            # MSE on scaled targets (our metric is RMSE)
        backward; step
    validation MSE
    if better: remember a copy of the weights; reset counter
    else: counter += 1; stop if counter > PATIENCE (40)  # early stopping
restore the best weights; return (model, epochs run)
```

**`run_one(variant, track, h, seed)`** ([L564](../../code/nee_phase2.py#L564)), one full experiment:
1. If the file exists, return (resumable).
2. Fix the seeds (`torch.manual_seed`, `np.random.seed`).
3. `make_model` (includes Stage I) then `fit`. **Exception:** `id-nomult` has nothing to train, so it is just evaluated (`ep = 0`).
4. Predict validation and test. `orig()` converts back to % ILI; `val_rmse`; `pred` is clipped at 0.
5. **Parts to % ILI** ([L579](../../code/nee_phase2.py#L579)): each part × the node's range, and the node's minimum added to `base`. **Assert** ([L580](../../code/nee_phase2.py#L580)): the parts sum to the raw forecast within 1e-4.
6. Save the `.npz`. `src` is stored as float16 to halve the size (rounding below 0.004% ILI).

**Helpers:**
- `show` ([L589](../../code/nee_phase2.py#L589)) prints one line per run;
- `run_grid` ([L594](../../code/nee_phase2.py#L594)) loops `run_one` over variants × tracks × horizons × seeds and respects `MAX_HOURS`;
- `load_mine(model)` ([L604](../../code/nee_phase2.py#L604)) reads all files of a variant into `{(track, h, seed): (pred, true, t)}` (the format `compare()` wants), plus their validation RMSEs;
- `ens_table(model)` ([L611](../../code/nee_phase2.py#L611)) gives seed-ensemble RMSE (forecast averaged over seeds) for ours and every baseline, and our rank.

### Cell 9: settings and all runs on the real data ([L637](../../code/nee_phase2.py#L637))
**Settings** ([L645–654](../../code/nee_phase2.py#L645)):
- `OUT = results/nee_p2`, `FIG = results/nee_p2/figures`;
- `SAVE_SRC = True` (save who-from-whom), `MAIN = "id"`;
- `REAL` = the 4 tracks;
- `XAI_H = [1, 4]` (horizons for the explanation comparison), `XAI_SEEDS` = 3 seeds;
- `IG_STEPS = 32`, `N_BOOT = 40`, `SYN_SEEDS` = 3 training seeds for the simulations, `LONG_H = [4, 8]`.

**Runs:**

| Line | Call | Count | Purpose |
|---|---|---|---|
| [L655](../../code/nee_phase2.py#L655) | `run_grid(["id", "full"], REAL, RUN_H, SEEDS)` | 2 × 4 × 6 × 5 = 240 | our method and the free-weight ablation |
| [L656](../../code/nee_phase2.py#L656) | `run_grid(["id-nomult"], …, [0])` | 24 | the core alone (deterministic, 1 run) |
| [L657](../../code/nee_phase2.py#L657) | `run_grid(["full-altinit"], REAL, LONG_H, XAI_SEEDS)` | 4 × 2 × 3 = 24 | the ablation from a different start (identifiability) |

### Cell 10: black-box comparator and post-hoc explainers ([L660](../../code/nee_phase2.py#L660))
**`GCNGRU`** ([L666](../../code/nee_phase2.py#L666)), a standard spatio-temporal GNN:
```python
h = gru(per-node window + sin/cos)[:, -1]                 # [B, N, 64] each node's temporal summary
m = relu(mix(einsum("ij,bjh->bih", A, h)))                # graph convolution: average of neighbours' summaries
return x[:, -1] + out(cat([h, m]))                        # residual: last week + a learned correction
```
- `einsum("ij,bjh->bih", A, h)`: for every batch element and node i, the A-weighted sum (here: average) of its neighbours' feature vectors.
- It is trained with the **same `fit()`** as ours, so the comparison is fair.

**`clim_window(track, t)`** ([L683](../../code/nee_phase2.py#L683)): every input window filled with the node's climatology, the explainers' reference for "nothing unusual".

**`occlusion(m, x, sc, wt, x0)`** ([L688](../../code/nee_phase2.py#L688)): for each source node j, replace its window with the reference and record `y − y_occluded`. The result is `[n, target, source]`.

**`integrated_gradients(...)`** ([L695](../../code/nee_phase2.py#L695)):
```python
for a in midpoints (0.5, 1.5, ...)/steps:                 # points along the straight line x0 -> x
    xa = x0 + a*(x - x0) (with gradient tracking)
    for each target node i: gradient of y_i w.r.t. xa
        att[:, i] += (grad * (x - x0)).sum(over the 20 weeks) / steps
```
- IG = (input − reference) × average gradient along the path. Its attributions approximately sum to f(x) − f(x0) ("completeness").
- `cudnn.flags(enabled=False)`: cuDNN's GRU cannot compute gradients in eval mode, so it is switched off.

**`run_bb(track, h, seed)`** ([L708](../../code/nee_phase2.py#L708)):
- trains GCN-GRU;
- computes both attribution matrices and the **completeness gap**, |Σ attributions − (f(x) − f(x0))| / |f(x) − f(x0)|;
- saves `GCN__…npz` and prints RMSE and the gaps.

**Self-check** ([L728](../../code/nee_phase2.py#L728)): on a purely linear model, occlusion and IG must both equal the exact contribution `W_ij · x_j`.

### Cell 11: run the comparator ([L741](../../code/nee_phase2.py#L741))
4 tracks × h ∈ {1, 4} × 3 seeds = 24 black-box runs, each with occlusion and IG.

### Cell 12: accuracy vs every baseline ([L751](../../code/nee_phase2.py#L751))
For three variants:
- `id` (ours);
- `id-nomult` (core alone);
- `full` (free-weight ablation).

It calls `compare(load_mine(...), BASE, subset="inseason")`, then `ens_table` for ours. This produced the main accuracy table: 21/24 tied, 1 significant win, 2 significant losses.

### Cell 13: identifiability ([L762](../../code/nee_phase2.py#L762))
**Helpers:**
- `nee_runs(track, h, seeds, v)` ([L768](../../code/nee_phase2.py#L768)) loads saved runs;
- `ins_rows` returns the in-season mask;
- **`shares(zs, track, h)`** ([L774](../../code/nee_phase2.py#L774)): mean |part| over in-season test forecasts, normalised to sum to 1, averaged over seeds.

**I1** ([L780](../../code/nee_phase2.py#L780)): the shares (seasonal / own / national + importation) and RMSE of `full`, `full-altinit` and `id`. Similar RMSE with different shares means the explanation is not identified.

**`boot_core(track, h, B)`** ([L793](../../code/nee_phase2.py#L793)), the season-block bootstrap:
```python
season_row = np.repeat(p["season"][t + h], N)    # season of each design-matrix row (sample-major, N rows per sample)
for b in 0..B:                                   # b = 0: the real fit; 1..B: resamples
    rows = None if b == 0 else concat([where(season_row == s) for s in rng.choice(seasons, len(seasons))])
    l, q, pp = core_fit(..., rows=rows)[-3:]     # refit Stage I on resampled whole seasons
    cross = (national * (N-1)/N + importation) / (own + national + importation)   # on in-season TEST windows
```
- `choice(seasons, len(seasons))` draws whole seasons **with replacement**, because weeks inside a season are correlated.
- `(N−1)/N`: a node's own value is one of the N terms in the national mean, and that slice is not "from other regions".
- The 5th and 95th percentiles over the 40 resamples give the **90% interval**.

### Cell 14: explanation quality vs the black box ([L817](../../code/nee_phase2.py#L817))
- **`stability(mats)`** ([L824](../../code/nee_phase2.py#L824)), for every pair of seeds:
  - the Pearson correlation of all off-diagonal (cross-region) entries;
  - the fraction of forecasts whose biggest other-region contributor is the same (rows where both are non-zero).
- **A1:** ours (`src`) vs occlusion and IG, plus the black box's RMSE.
- **A2 completeness:** ours = the max |Σ src − (own + nat + imp)|, which is only float16 rounding; the explainers use their saved gaps.
- **A4:** our shares by horizon.

### Cell 15: figures ([L862](../../code/nee_phase2.py#L862))
- `fig1_decomposition_region4.png`: the 4-week-ahead forecast for Region 4 split into its parts, pre- and post-COVID.
- `fig2_shares_vs_horizon.png`: the share of each part vs horizon for each track.
- `matplotlib.use("Agg")` locally renders to files without a screen; on Kaggle figures also show inline.

### Cell 16: the semi-synthetic benchmark ([L893](../../code/nee_phase2.py#L893))
**`sim_kernel(kind, pop, dist)`** ([L902](../../code/nee_phase2.py#L902)), the **true** travel network K (rows sum to 1):
- gravity: `pop_j^0.7 / max(dist, 50 km)^2`;
- long-range: 3 random partners per state (symmetrised), weighted by `pop^0.5`.

**`simulate(c, kind, ...)`** ([L913](../../code/nee_phase2.py#L913)):
```python
phase = 18 + 5 * zscore(latitude)                         # each state's seasonal peak week
beta0 ~ uniform(1.6, 2.0) per season                      # season strength
S, inc = 0.6*pop, 1e-5*pop                                # susceptibles, this week's new infections
each season:  S = min(S + uniform(0.35, 0.55)*(pop - S), pop)        # immunity wanes between seasons
each week:
    beta = beta0 * (1 + 0.35*cos(2π(w - phase)/52))       # seasonal forcing
    loc  = (1-c) * inc/pop                                # local prevalence
    imp  = c * K @ (inc/pop)                              # prevalence imported along K
    new  = binomial(S, 1 - exp(-(beta*(loc + imp) + 1e-6)))   # random new infections
    share[i, j] = c*K_ij*prev_j / total_i ; share[i, i] = loc_i / total_i   # THE TRUTH, every week
    S -= new; inc = new
drop the first 5 of 25 seasons (burn-in)
ili = (base + 1.6 * 100*inc/pop) * exp(N(0, 0.08)) + N(0, 0.05)       # noisy surveillance-like % ILI
```
- **Why `1 − exp(−λ)`:** the chance of infection in one week under a constant hazard λ.

**Registration** ([L938](../../code/nee_phase2.py#L938)): each scenario is stored in `P`, `TR` and `GRAPH` exactly like real data, so **every model function works unchanged**.
- train = seasons 2000–2010, validation = 2011–2012, test = 2013–2019; in-season = weeks 0–32.
- `TRUTH[name]` holds the true share matrices.
- 5 scenarios ([SCEN L932](../../code/nee_phase2.py#L932)): `syn_c0` (no spread), `syn_g10`, `syn_g30` (gravity), `syn_l10`, `syn_l30` (long-range).
- One simulation per scenario (simulation seed 0).

### Cell 17: train everything on each scenario ([L953](../../code/nee_phase2.py#L953))
- `SYN_V = ["full", "full-altinit", "id", "id-gravity", "id-nomult"]` plus the black box, at h = 1, with 3 training seeds per scenario.
- **Why h = 1:** the simulator's truth (the source shares of next week's infections) is exactly defined one week ahead.

### Cell 18: can each explanation recover the truth? ([L964](../../code/nee_phase2.py#L964))
- **`imp_share(src)`** ([L969](../../code/nee_phase2.py#L969)) = 1 − |diagonal| / Σ|row|: the estimated share of a forecast coming from other nodes.
- **Active weeks:** in-season and ILI more than 0.5 above the node's median, so quiet weeks don't dominate.
- **`top3(est, tru, rows)`** ([L973](../../code/nee_phase2.py#L973)): the diagonal is set to −1 (excluded); `argsort(−·)[..., :3]` gives the 3 largest sources; the result is the fraction of the true top-3 found. Rows are those where the true imported share exceeds 0.02.
- **Per method:** test RMSE, mean estimated imported share, correlation with the true share across state-weeks, and top-3 overlap. Correlation and top-3 are undefined when c = 0 (no truth to correlate).

### Cell 19: backup ([L1000](../../code/nee_phase2.py#L1000))
Zips `results/nee_p2` (all runs and figures) into `nee_phase2_backup.zip`. Download it before the session ends.

---

## 4. How to run it, and what each cell should print

### 4.1 On Kaggle
1. New notebook. Inputs: **`influenza-datasets`** and the dataset with **`baselines_backup.zip`**. Settings: **Internet ON**, **GPU T4**.
2. One Kaggle cell per `# %% Cell N` block, **all copied from the same version of the file**.

| Cell | Time (T4) | Expected output |
|---|---|---|
| 1–3 | seconds | paths, panel shapes `{'regions': (1252, 10), 'states': (782, 49)}` |
| 4 | seconds | all PASS; persistence regions_t1 0.573 / 1.442 / 2.182 |
| 5 | seconds | `loaded 480 baseline runs; ['AR', 'ARIMA', 'CNNRNNRes', 'ColaGNN', 'EpiGNN', 'GRU', 'LSTM', 'SARIMA']` |
| 6 | ~10 s | population 312.5 M / 289.3 M, 2 × PASS, `device: cuda` |
| 7 | seconds | `PASS - v3 core: recovers known spread coefficients [0.6 0.1 0.2] …` + 5 × PASS; nodes with no neighbours: Alaska, Hawaii |
| 8 | instant | (function definitions) |
| 9 | ~40 min | one line per run: `NEE-id regions_t1 h=1 s=0 test RMSE 0.593 val 0.411 ep 125 13s` |
| 10 | seconds | `PASS - occlusion and Integrated Gradients recover a known linear attribution` |
| 11 | ~10–15 min | one line per black-box run, with completeness gaps |
| 12 | seconds | the accuracy tables (ours, core alone, ablation) and the seed-ensemble ranking |
| 13 | ~6 min | I1 table; I2 bootstrap table (`bootstrap time ~373 s`) |
| 14 | seconds | A1 stability, A2 completeness, A4 shares |
| 15 | seconds | `figures: ['fig1_decomposition_region4.png', 'fig2_shares_vs_horizon.png']` |
| 16 | ~1 min | one summary line per scenario (true imported share 0.000 / 0.109 / 0.303 / 0.114 / 0.306) |
| 17 | ~30 min | one line per run on the simulations |
| 18 | seconds | 5 result tables |
| 19 | seconds | `backup: …nee_phase2_backup.zip` |

The outputs of the completed run are summarised in [research/08_results/phase2_v3_results.md](../08_results/phase2_v3_results.md).

3. If the session stops, rerun Cells 1–8, then continue from the cell that was running. Finished runs are skipped.

### 4.2 Locally (smoke test only; the laptop must not run real experiments)
```bash
EPI_QUICK=1 EPI_WORK=<scratch folder> EPI_THREADS=2 python code/nee_phase2.py
```
- Quick mode: 2 epochs, 2 seeds, horizons 1 and 4, 2 tracks, 2 scenarios. It tests that the code runs, not the results.
- Run **one** process only.
- Without the baselines zip, baseline columns show "-". Set `EPI_BASELINES=<folder with neural/ and classical/>` to use extracted baseline files.

### 4.3 Environment variables
| Variable | Effect |
|---|---|
| `EPI_WORK` | output folder |
| `EPI_QUICK=1` | smoke-test mode |
| `EPI_THREADS` | PyTorch CPU threads (default 4) |
| `EPI_MAX_HOURS` | stop starting new runs after this many hours (default 11.5) |
| `EPI_BASELINES` | local folder with baseline `.npz` files |

---

## 5. Common errors and fixes
| Symptom | Cause | Fix |
|---|---|---|
| `TypeError: NEE() argument after ** must be a mapping, not tuple` | cells from **different file versions**: a new Cell 7 (`flags` returns a pair) with an old Cell 8 | recopy Cells 6, 7 and 8 from the current file and rerun them (this happened once in our run) |
| `no baselines_backup.zip attached - baseline comparison disabled` | the zip dataset isn't attached | attach it; rerun from Cell 5 |
| a `FAIL` in Cell 4 | the data differ from the benchmark | stop; check the input dataset |
| `could not download …` | Internet is off | enable Internet in the Kaggle settings |
| `time budget 11.5 h reached` | a long loop stopped itself | new session: Cells 1–8, then the same cell |
| cuDNN error during Integrated Gradients | GRU backward in eval mode | handled by `cudnn.flags(enabled=False)`; if you see it, your Cell 10 is outdated |
| `AssertionError: parts must sum to the forecast` | a code edit broke the exact decomposition | revert the edit |
| states_t1 runs identical across seeds | early stopping kept the multiplier at 1 (nothing to improve) | not a bug; it is a reported finding |
| the laptop freezes | a real run started locally | stop it; locally only `EPI_QUICK=1`, one process |

---

## 6. How to change things (recipes)
| Goal | Where | Change |
|---|---|---|
| run our method with the gravity kernel as the main model | Cell 9 | `MAIN = "id-gravity"` |
| more or fewer seeds | Cell 6 `SEEDS` | 5 is the final-table standard |
| more bootstrap resamples | Cell 9 `N_BOOT` | 100 gives smoother intervals (about 2.5× the Cell 13 time) |
| run graph ablations of our method | Cell 9 | add `run_grid(["id-noimp", "id-nonat", "id-randgraph"], REAL, RUN_H, XAI_SEEDS)`; they already exist in `flags` |
| a different spatial decay (e.g. 1/o²) | `id_kernel` (Cell 7) | `1.0 / O**2` instead of `1.0 / O` |
| explain more horizons with the black box | Cell 9 `XAI_H` | each extra horizon adds about 50% to Cells 11 and 14 |
| more independent simulations | Cell 16 | loop `simulate(..., seed=r)` over r and register `name_r{r}` |
| a new scenario | Cell 16 `SCEN` | add `"syn_x": (c, kind)`; Cells 17–18 pick it up |
| a different window or horizon | Cell 1 | breaks comparability: the baselines were run with window 20 and horizons 1–8 |

---

## 7. The data flow in one table
| Step | Function | Input → output |
|---|---|---|
| read | `load_panel` | CSV → `P[ds]["X_pct"]` `[T,N]` + flags |
| scale | `TR`, `scale` | `X_pct` → values about 0–1 per node (training min/max) |
| samples | `origins`, `build_windows` | → `X [n,20,N]`, `Y [n,N]`, `t [n]` |
| tensors | `tensors` | + `sc [n,20,2]`, `wt [n]` on the GPU |
| graph | `hop_order`, `id_kernel` | adjacency → hop order → `W [N,N]` |
| Stage I | `core_design`, `core_fit` | training windows → design matrix `[n·N, 93]` → coefficients `[93]` |
| Stage II | `NEE3`, `fit` | windows → multiplier m `[n,N]` → forecast = sum of 5 parts |
| save | `run_one` | → `.npz` (pred, true, t, parts, src, core) |
| evaluate | `compare`, `dm_test`, `ens_table` | `.npz` + baselines → RMSE tables, DM p-values |
| explain | `shares`, `boot_core`, `stability` | `.npz` → shares, 90% intervals, stability |
| ground truth | `simulate`, `imp_share`, `top3` | simulated data + truth → recovery scores |

---

## 8. Glossary of idioms
| Idiom | Meaning |
|---|---|
| **broadcasting** | dimensions of size 1 stretch automatically: `a[:, None] - a[None, :]` gives all pairwise differences |
| **fancy indexing** | indexing with an integer array: `Xs[idx]` with `idx` `[n, 20]` gives `[n, 20, N]` |
| `[:, None]`, `.unsqueeze(-1)` | add an axis of size 1 |
| `reshape(n*N, -1)` | merge axes; `-1` = whatever size fits |
| `x.transpose(1, 2)` | swap axes 1 and 2 |
| `einsum("ij,bjh->bih", A, h)` | multiply-and-sum written with index letters (here a graph convolution) |
| `@` | matrix multiplication |
| `np.savez_compressed` / `np.load` | save/load several arrays in one compressed file |
| `nn.Module`, `forward` | a PyTorch model and its computation |
| `register_buffer` | a tensor kept in the model but **not trained** |
| `nn.Parameter` | a trainable tensor |
| `m.train()` / `m.eval()` | training vs evaluation mode |
| `torch.no_grad()` | no gradient tracking (evaluation) |
| `state_dict()` / `load_state_dict()` | save and restore weights (best-epoch restore) |
| `torch.manual_seed(s)` | fixed random start, for reproducibility |
| `softplus(x)` | log(1 + eˣ), smooth and always positive |
| `softmax` | positive weights summing to 1 |
| `masked_fill(mask, -inf)` | entries that get weight 0 after softmax |
| `lsq_linear(A, y, bounds)` | SciPy bounded least squares |
| `np.linalg.lstsq` | ordinary least squares |
| `rng.choice(a, k)` | sampling with replacement (bootstrap) |

---

## 9. Code questions you may be asked
**Q1. Where does the code prevent data leakage?**
Four places:
- `origins` never uses input after the origin and only takes targets from the requested seasons;
- `TR` scalers use training weeks only;
- `core_fit` is called on training windows only (`make_model`);
- early stopping uses validation only.

The test split is touched only for the final prediction in `run_one`.

**Q2. How is the explanation guaranteed to be exact?**
- The forecast *is* `sum(parts.values())`.
- `run_one` asserts that the converted parts sum to the raw forecast.
- Cell 7 checks it on random inputs, including the per-source matrix rows.

**Q3. How is the split protected from training?**
- λ, ρ, φ are stored with `register_buffer`, so they are not parameters and the optimizer never changes them.
- The only trained part, the multiplier, scales the whole epidemic bracket.

**Q4. Are the results reproducible?**
Yes:
- seeds are fixed per run;
- the bootstrap and simulator use seeded generators;
- Stage I is deterministic.

GPU arithmetic may differ in the last digits, not in conclusions.

**Q5. Why save predictions instead of trained models?**
Every analysis (RMSE, DM tests, ensembles, shares, stability) needs only `pred`, `true`, `t`, `parts` and `src`. Saving them makes the analysis cells fast and resumable, and lets ours be compared with the baselines from another notebook.

**Q6. Why one training routine for everything?**
Our method, its ablation and the black box all use the same `fit()`. Any difference in results then comes from the model, not the training recipe.

**Q7. How big and how fast is the model?**
- **Stage I:** 93 coefficients for Regions (49 + 392 + 3 = 444 for States), one least-squares solve in under a second.
- **Stage II:** a GRU with 2 layers × 64 units plus a linear head, about 38,000 trainable parameters, seconds to a minute per run on a T4.
- **The whole notebook:** about 2–2.5 hours.

**Q8. Why float16 for `src`?**
The matrices are `[n, N, N]` per run (e.g. 179 × 49 × 49 for states). float16 halves the storage, and its rounding (< 0.004% ILI) is far below anything interpreted.

**Q9. Why does `id-nomult` show `ep 0`?**
It is the statistical core alone: there is nothing to train, so `run_one` evaluates it directly.

**Q10. Why are all five `regions_t1 h=8` runs of our method identical (RMSE 1.500)?**
- At h = 8 Stage I sets λ = ρ = φ = 0 (all non-negativity bounds active), so the epidemic part is zero.
- The multiplier then multiplies zero and has no effect; early stopping ends at the start, and every seed gives the core's forecast.
