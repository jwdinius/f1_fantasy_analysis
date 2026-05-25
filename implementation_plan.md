# F1 Fantasy Analysis Implementation Plan

This plan aims to align the codebase with the goals and features documented in [methodology_summary.txt](file:///Users/joe/Documents/work/f1_fantasy_analysis/methodology_summary.txt). It resolves the data processing gaps, implements a comprehensive feature engineering framework, updates the training models to correctly utilize FastF1 telemetry, fixes the 2025 out-of-sample evaluation script, and enhances the optimizer to support displaying the Top 5 Team configurations.

> Terminology in this plan follows [CONTEXT.md](./CONTEXT.md). Key terms: **Asset** (Driver or Constructor), **EV** (Expected Value — predicted fantasy points), **Round** (race weekend), **Team** (fantasy selection of 5 Drivers + 2 Constructors), **Constructor** (F1 team entity), **Team Lock** (cutoff after FP3), **Training Window** (N prior rounds used as rolling aggregates), **Practice Gap** (driver best lap vs. session fastest), **DotD** (Driver of the Day), **Budget Cap** ($100M), **Walk-Forward Validation**.

> Architecture decisions: see [ADR-0001](./docs/adr/0001-walk-forward-validation.md) (Walk-Forward Validation) and [ADR-0002](./docs/adr/0002-rolling-window-aggregates-over-per-round-lags.md) (Rolling Window Aggregates).

## User Review Required

> [!IMPORTANT]
> **Data Processing Integration:**
> Currently, the workspace has two conflicting data paths:
> 1. A unified `processed_fantasy.csv` containing FastF1 telemetry features (`lap_time_median_gap_fp1`, `lap_time_iqr_fp1`, etc.).
> 2. Separate `processed_fantasy_drivers.csv` and `processed_fantasy_constructors.csv` that contain SQL-only features and lack FastF1 telemetry.
>
> We will **merge the FastF1 telemetry and SQL data pipelines** into a single robust processing script that generates fully-featured Driver and Constructor CSVs.

---

## Agreed Feature Sets

### Driver Feature Vector

All current-round features must be available before Team Lock (derived from FP1–FP3 only). Training Window features use rolling aggregates over the last N rounds (default N=3; configurable for comparative analysis at N=5, 7, etc.).

**Current-round features (available before Team Lock):**

| Feature | Source | Description |
|---|---|---|
| `fp1_gap_sec`, `fp2_gap_sec`, `fp3_gap_sec` | f1db | Driver's best lap Practice Gap per session |
| `lap_time_median_gap_fp1/2/3` | FastF1 | Median Practice Gap per session |
| `lap_time_iqr_fp1/2/3` | FastF1 | IQR of lap times per session |
| `cost` | Cost data | Official Driver price |

**Training Window features (rolling over N prior rounds):**

| Feature | Type | Aggregation |
|---|---|---|
| `race_pos` | continuous | median, min, max — DNF/DSQ imputed as 25 |
| `quali_pos` | continuous | median, min, max |
| `pos_gained` | continuous | median, min, max |
| `fantasy_points` | continuous | median, min, max |
| `dotd` | binary | sum |
| `fastest_lap` | binary | sum |

### Constructor Feature Vector

**Derived features:**

| Feature | Source | Description |
|---|---|---|
| `drivers_predicted_points_sum` | Derived | Sum of actual driver points at train time; sum of Driver EVs at inference |
| `cost` | Cost data | Official Constructor price |

**Training Window features (rolling over N prior rounds):**

| Feature | Type | Aggregation |
|---|---|---|
| `fastest_pit_stop` | continuous | median, min, max |
| `both_q3` | binary | sum |
| `one_q3` | binary | sum |
| `both_q2` | binary | sum |

---

## Proposed Changes

### Data Processing Pipeline

We will update `src/data_processing.py` to produce the agreed feature vectors above.

#### [MODIFY] [data_processing.py](file:///Users/joe/Documents/work/f1_fantasy_analysis/src/data_processing.py)
- **FastF1 Telemetry Integration:** Restore `fetch_fastf1_stats` to add `lap_time_median_gap_fp1/2/3` and `lap_time_iqr_fp1/2/3` alongside the existing f1db Practice Gap columns. No preseason testing sessions.
- **Constructor Rebranding:** Maintain rebranding of historical Constructor IDs to 2025 canonical IDs inside `CONSTRUCTOR_REBRANDING`.
- **Rolling Window Aggregates:** Replace per-round lag columns (`*_n_1/2/3`) with rolling aggregates (median, min, max for continuous; sum for binary) over a configurable window width N (default 3). DNF/DSQ race positions imputed with 25 before aggregation.
- **Driver/Constructor Splitting:** Save two output files:
  - `data/processed_fantasy_drivers.csv`
  - `data/processed_fantasy_constructors.csv`
- **Bug Fix:** Resolve the type/naming mismatch bug during Driver merging. Ensure both datasets are merged using string-based canonical Driver IDs (e.g. `'max-verstappen'`).

---

### Modeling and EV Prediction

#### [MODIFY] [model.py](file:///Users/joe/Documents/work/f1_fantasy_analysis/src/model.py)
- Update `prepare_features` for Drivers to use the agreed feature set: f1db Practice Gap columns, FastF1 median gap and IQR columns, and rolling Training Window aggregates.
- Update `prepare_features` for Constructors to use: `drivers_predicted_points_sum`, `cost`, and rolling Training Window aggregates.
- Remove references to per-round lag columns (`*_n_1`, `*_n_2`, `*_n_3`).

#### [MODIFY] [predict_ev.py](file:///Users/joe/Documents/work/f1_fantasy_analysis/src/predict_ev.py)
- Maintain the two-stage EV model: train Driver EV predictor first, sum Driver EVs per Constructor to produce `drivers_predicted_points_sum`, then train Constructor EV predictor.
- Walk-Forward Validation must be preserved: neither stage may use any data from the target round or later rounds.

---

### Season Evaluation

#### [MODIFY] [evaluate_2025.py](file:///Users/joe/Documents/work/f1_fantasy_analysis/src/evaluate_2025.py)
- Remove the old `n_lags=3` Constructor argument.
- Refactor the evaluation loop to train two separate model instances (`asset_type='driver'` and `asset_type='constructor'`) on all rounds prior to the target round (Walk-Forward Validation).
- Generate and save the residuals plot `data/eval_2025_residuals.png`.

---

### Optimization

#### [MODIFY] [optimize_lp.py](file:///Users/joe/Documents/work/f1_fantasy_analysis/src/optimize_lp.py)
- Expose the `top_n` parameter in the command-line entry point.
- Update the output formatting to clearly display the Top 5 Teams, their total costs, and their predicted EVs.

---

## Verification Plan

1. **Data Processing:**
   ```bash
   PYTHONPATH=. /Users/joe/anaconda3/envs/f1_fantasy/bin/python src/data_processing.py
   ```
2. **EV Prediction (for a 2025 round, e.g., R1):**
   ```bash
   PYTHONPATH=. /Users/joe/anaconda3/envs/f1_fantasy/bin/python src/predict_ev.py 2025 1
   ```
3. **Team Optimization (with Top 5 configurations):**
   ```bash
   PYTHONPATH=. /Users/joe/anaconda3/envs/f1_fantasy/bin/python src/optimize_lp.py data/ev_reports/ev_report_2025_R1.csv 100.0
   ```
4. **2025 Out-of-Sample Evaluation & Residual Plots:**
   ```bash
   PYTHONPATH=. /Users/joe/anaconda3/envs/f1_fantasy/bin/python src/evaluate_2025.py
   ```
   Verify that `data/eval_2025_residuals.png` is generated successfully.
