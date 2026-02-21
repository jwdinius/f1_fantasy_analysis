# F1 Fantasy Analysis & Optimizer

This project provides a data-driven approach to F1 Fantasy team selection using Gradient Boosted Trees (XGBoost) and combinatorial optimization. It integrates historical race results with high-fidelity telemetry and timing data from the FastF1 API.

## Methodology

### 1. Data Sources
- **Historical Results:** Race, Qualifying, Sprint, and Pit Stop data from the local SQLite database.
- **High-Fidelity Timing (FastF1):** Lap-by-lap timing data for all sessions.
- **Fantasy Costs:** Official price data for drivers and constructors.

### 2. Predictive Modeling (XGBoost)
The core of the system is an **XGBoost Regressor** that predicts the expected points (EV) for every driver and constructor.
- **Features:**
    - **Historical Lags:** Points scored in the previous 3 rounds.
    - **Pace Gaps:** The median gap to the fastest lap in FP1, FP2, FP3, and Pre-season Testing.
    - **Consistency (Variance):** The standard deviation of lap times in practice and testing sessions to account for "peak vs. sustainable" pace.
    - **Asset Context:** Cost and asset type (driver vs. constructor).

### 3. Constraints & Assumptions
- **Strict Temporal Boundary:** The model is "pre-race." It is trained only on data from rounds prior to the target race. For the target round itself, it *only* looks at data available before the race start (FP1-3 and Testing).
- **Driver-Constructor Mapping:** Mappings are dynamically derived to ensure historical accuracy (e.g., handling mid-season driver swaps or rookie entries).
- **Rookie Handling:** Rookies are evaluated based on their Pre-season Testing performance and representative practice times.

### 4. Optimization (Knapsack Solver)
The system uses a combinatorial search to find the mathematically optimal configuration:
- **Team Composition:** Exactly 5 Drivers and 2 Constructors.
- **Budget Constraint:** Total cost must be within the user-defined limit (default $100M).
- **3-Asset Rule:** No more than 3 assets from a single constructor (e.g., you cannot have McLaren + Norris + Piastri + a 3rd McLaren-affiliated asset).

---

## Instructions for Use

### Setup
Ensure you are using the `f1_fantasy` conda environment:
```bash
conda activate f1_fantasy
```

### 1. Generate Latest Data
Before predicting, ensure your local processed dataset includes the latest practice and race data:
```bash
PYTHONPATH=. python src/data_processing.py
```

### 2. Generate Expected Values (EV)
To see the predicted points for all drivers and constructors for a specific round:
```bash
PYTHONPATH=. python src/predict_ev.py <year> <round>
```
*Example: `PYTHONPATH=. python src/predict_ev.py 2025 7`*

This will:
1. Train the model on all data prior to 2025 R7.
2. Output a sorted list of Drivers and Constructors to the console.
3. Save a full report to `data/ev_report_2025_R7.csv`.

### 3. Optimize Team Configuration
To find the top-scoring team based on the generated EV report and your budget:
```bash
PYTHONPATH=. python src/optimize_lp.py data/ev_report_<year>_R<round>.csv <budget>
```
*Example: `PYTHONPATH=. python src/optimize_lp.py data/ev_report_2025_R7.csv 102.5`*

### 4. Evaluation & Bias Check
To verify the model's accuracy across the entire 2025 season and generate residual plots:
```bash
PYTHONPATH=. python src/evaluate_2025.py
```
This produces `data/eval_2025_residuals.png`, which allows you to visualize if the model is over or under-predicting specific drivers.
