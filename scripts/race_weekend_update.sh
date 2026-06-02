#!/bin/bash
# Usage: ./scripts/race_weekend_update.sh [--top-n N] [--budget B] [--year Y --round R]
#
# Run after Friday FP3. Updates the f1db submodule, rebuilds the database,
# reprocesses all data, generates driver/constructor EV predictions for the
# current round (auto-detected), and prints the optimal team.
#
# PREREQUISITE: Before running this script, update the four Numbers spreadsheets
# in data/numbers/ (Drivers-Cost, Drivers-Points, Teams-Cost, Teams-Points)
# with the current round's costs and any new Game Score data, then save them.
# This script converts them to CSV as its first step.
#
# Year and round are auto-detected from the processed data: the first round
# with FP feature data but no race result yet is the target. Pass --year and
# --round to override if needed.
#
# Requires the f1_fantasy conda environment (environment.yml).

set -euo pipefail

CONDA_ENV=f1_fantasy
TOP_N=3
BUDGET=100.0
YEAR=""
ROUND=""

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --top-n)  TOP_N="$2";   shift 2 ;;
        --budget) BUDGET="$2";  shift 2 ;;
        --year)   YEAR="$2";    shift 2 ;;
        --round)  ROUND="$2";   shift 2 ;;
        *) echo "Unknown flag: $1"; exit 1 ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Step 1 — convert Numbers spreadsheets → actual_fantasy_*.csv
# ---------------------------------------------------------------------------
echo ""
echo "=== [1/5] Converting Numbers spreadsheets ==="
conda run --no-capture-output -n "$CONDA_ENV" \
    python -m src.convert_numbers

# ---------------------------------------------------------------------------
# Step 2 — update f1db submodule
# ---------------------------------------------------------------------------
echo ""
echo "=== [2/5] Updating f1db submodule ==="
git -C f1db fetch upstream && \
    git -C f1db checkout add-docker && \
    git -C f1db rebase upstream/main && \
    git -C f1db push origin add-docker -f

# ---------------------------------------------------------------------------
# Step 3 — rebuild f1db Docker image and copy artifacts
# ---------------------------------------------------------------------------
echo ""
echo "=== [3/5] Building f1db Docker image ==="
./scripts/build_f1db

# ---------------------------------------------------------------------------
# Step 4 — rebuild processed feature CSVs
# ---------------------------------------------------------------------------
echo ""
echo "=== [4/5] Rebuilding processed data ==="
conda run --no-capture-output -n "$CONDA_ENV" \
    python -m src.data_processing

# ---------------------------------------------------------------------------
# Step 5 — generate EV predictions + optimize team
# ---------------------------------------------------------------------------
echo ""
echo "=== [5/5] Generating EV predictions and optimising team ==="
mkdir -p "$PROJECT_ROOT/data/ev_reports"

PREDICT_ARGS=""
if [[ -n "$YEAR" && -n "$ROUND" ]]; then
    PREDICT_ARGS="--year $YEAR --round $ROUND"
fi

# predict_ev prints the target round and saves the EV report; capture the path
conda run --no-capture-output -n "$CONDA_ENV" \
    python -m src.predict_ev $PREDICT_ARGS

# Find the most recently written EV report and run the optimizer on it
EV_REPORT=$(ls -t "$PROJECT_ROOT/data/ev_reports"/ev_report_*.csv 2>/dev/null | head -1)
if [[ -z "$EV_REPORT" ]]; then
    echo "ERROR: No EV report found after prediction step."
    exit 1
fi

echo ""
echo "=== Optimising team (top ${TOP_N}, budget \$${BUDGET}M) ==="
conda run --no-capture-output -n "$CONDA_ENV" \
    python src/optimize_lp.py "$EV_REPORT" "$BUDGET" --top-n "$TOP_N"

echo ""
echo "Done. EV report: $EV_REPORT"
