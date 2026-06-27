#!/bin/bash
# Usage: ./scripts/race_weekend_update.sh [--top-n N] [--budget B] [--year Y --round R] [--inference]
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
# --inference: fast path that only refreshes costs/points from the Numbers
# spreadsheets and predicts using the saved model checkpoint in models/.
# Skips the f1db submodule update, Docker rebuild, and data_processing.
# Requires models/{driver,constructor}_predictor.joblib (run
# `python -m src.train_model` first if missing).
#
# Requires the f1_fantasy conda environment (environment.yml).

set -euo pipefail

CONDA_ENV=f1_fantasy
TOP_N=""
BUDGET=100.0
YEAR=""
ROUND=""
INFERENCE=0

usage() {
    cat <<'EOF'
Usage: ./scripts/race_weekend_update.sh [OPTIONS]

Run after Friday FP3 to update data, generate EV predictions for the current
round, and print the optimal fantasy team.

Options:
  --top-n N        Flat top-N ranking instead of the default hedge slate.
                   The default output is a 2-3 row slate (Headline + hedges
                   against tail risk on the top-2 EV drivers in Headline).
  --budget B       Budget cap in $M (default: 100.0)
  --year Y         Override auto-detected year (must be used with --round)
  --round R        Override auto-detected round (must be used with --year)
  --inference      Fast path: skip f1db update, Docker rebuild, and data
                   processing. Only refreshes costs/points from the Numbers
                   spreadsheets and predicts from the saved model checkpoint.
                   Requires models/{driver,constructor}_predictor.joblib.
  -h, --help       Show this help message and exit

Prerequisite: update the four Numbers spreadsheets in data/numbers/
(Drivers-Cost, Drivers-Points, Teams-Cost, Teams-Points) with the current
round's costs and any new Game Score data before running.

Requires the f1_fantasy conda environment (environment.yml).
EOF
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --top-n)     TOP_N="$2";   shift 2 ;;
        --budget)    BUDGET="$2";  shift 2 ;;
        --year)      YEAR="$2";    shift 2 ;;
        --round)     ROUND="$2";   shift 2 ;;
        --inference) INFERENCE=1;  shift   ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "Unknown flag: $1" >&2; usage >&2; exit 1 ;;
    esac
done

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ "$INFERENCE" -eq 1 ]]; then
    STEPS=2
else
    STEPS=5
fi

# ---------------------------------------------------------------------------
# Step 1 — convert Numbers spreadsheets → actual_fantasy_*.csv
# ---------------------------------------------------------------------------
echo ""
echo "=== [1/${STEPS}] Converting Numbers spreadsheets ==="
conda run --no-capture-output -n "$CONDA_ENV" \
    python -m src.convert_numbers

if [[ "$INFERENCE" -eq 0 ]]; then
    # -----------------------------------------------------------------------
    # Step 2 — update f1db submodule
    # -----------------------------------------------------------------------
    echo ""
    echo "=== [2/5] Updating f1db submodule ==="
    git -C f1db fetch upstream && \
        git -C f1db checkout add-docker && \
        git -C f1db rebase upstream/main && \
        git -C f1db push origin add-docker -f

    # -----------------------------------------------------------------------
    # Step 3 — rebuild f1db Docker image and copy artifacts
    # -----------------------------------------------------------------------
    echo ""
    echo "=== [3/5] Building f1db Docker image ==="
    ./scripts/build_f1db

    # -----------------------------------------------------------------------
    # Step 4 — rebuild processed feature CSVs
    # -----------------------------------------------------------------------
    echo ""
    echo "=== [4/5] Rebuilding processed data ==="
    conda run --no-capture-output -n "$CONDA_ENV" \
        python -m src.data_processing
fi

# ---------------------------------------------------------------------------
# Final step — generate EV predictions + optimize team
# ---------------------------------------------------------------------------
echo ""
echo "=== [${STEPS}/${STEPS}] Generating EV predictions and optimising team ==="
mkdir -p "$PROJECT_ROOT/data/ev_reports"

PREDICT_ARGS=""
if [[ -n "$YEAR" && -n "$ROUND" ]]; then
    PREDICT_ARGS="--year $YEAR --round $ROUND"
fi
if [[ "$INFERENCE" -eq 1 ]]; then
    PREDICT_ARGS="$PREDICT_ARGS --from-checkpoint"
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

if [[ -n "$TOP_N" ]]; then
    OPT_FLAGS=(--hedge none --top-n "$TOP_N")
    MODE_LABEL="top ${TOP_N}"
else
    OPT_FLAGS=(--hedge anchor)
    MODE_LABEL="hedge slate"
fi

echo ""
echo "=== Optimising team (${MODE_LABEL}, budget \$${BUDGET}M) ==="
conda run --no-capture-output -n "$CONDA_ENV" \
    python src/optimize_lp.py "$EV_REPORT" "$BUDGET" "${OPT_FLAGS[@]}"

echo ""
echo "Done. EV report: $EV_REPORT"
