import pandas as pd
import numpy as np
from src.model import F1FantasyPredictor
from pathlib import Path

script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "data"


def detect_current_round(d_df: pd.DataFrame) -> tuple[int, int]:
    """
    Return (year, round) for the next round to predict.

    Preference order:
    1. First round in the data with FP features but no race result yet
       (fantasy_points is null for every driver in that round).
    2. If all rounds have results, infer the next round as max_round + 1
       in the most recent year — this covers the pre-practice baseline case
       where the round isn't in the processed data yet.
    """
    round_results = (
        d_df.groupby(['year', 'round'])['fantasy_points']
        .apply(lambda s: s.isna().all())
        .reset_index(name='no_result')
    )
    pending = round_results[round_results['no_result']].sort_values(['year', 'round'])
    if not pending.empty:
        row = pending.iloc[0]
        return int(row['year']), int(row['round'])

    # All rounds in processed data have results — infer next round
    latest_year  = int(d_df['year'].max())
    latest_round = int(d_df[d_df['year'] == latest_year]['round'].max())
    next_round   = latest_round + 1
    print(f"  All processed rounds have results. "
          f"Inferring next round: {latest_year} R{next_round} "
          f"(baseline prediction — no FP data yet).")
    return latest_year, next_round


_CONSTRUCTOR_REBRAND = {
    'kick-sauber': 'audi',   # rebranded for 2026 season
}

def _latest_costs(actual_path: Path, id_col: str) -> pd.DataFrame | None:
    """Return the most recent round's costs from actual_fantasy_*.csv.

    Applies constructor rebranding so that historical names (e.g. kick-sauber)
    resolve to the canonical 2026+ f1db IDs (e.g. audi).
    """
    if not actual_path.exists():
        return None
    df = pd.read_csv(actual_path).dropna(subset=['cost'])
    if df.empty:
        return None
    latest_year  = df['year'].max()
    latest_round = df[df['year'] == latest_year]['round'].max()
    result = df[(df['year'] == latest_year) & (df['round'] == latest_round)][[id_col, 'cost']].copy()
    if id_col == 'constructor_id':
        result[id_col] = result[id_col].replace(_CONSTRUCTOR_REBRAND)
    return result


def generate_ev_report(year: int | None = None, round_num: int | None = None):
    d_path = data_dir / "processed_fantasy_drivers.csv"
    c_path = data_dir / "processed_fantasy_constructors.csv"

    if not d_path.exists() or not c_path.exists():
        print("Driver or Constructor data files missing.")
        return None

    d_df = pd.read_csv(d_path)
    c_df = pd.read_csv(c_path)

    if year is None or round_num is None:
        year, round_num = detect_current_round(d_df)
        print(f"Auto-detected target round: {year} R{round_num}")

    # Latest known costs (fallback when the target round has no processed row)
    d_cost_df = _latest_costs(data_dir / "actual_fantasy_drivers.csv",     "driver_id")
    c_cost_df = _latest_costs(data_dir / "actual_fantasy_constructors.csv", "constructor_id")

    # 1. Predict Drivers
    d_train = d_df[
        (d_df['year'] < year) | ((d_df['year'] == year) & (d_df['round'] < round_num))
    ]
    d_predictor = F1FantasyPredictor(asset_type='driver')
    if not d_train.empty:
        d_predictor.train(d_train)
    d_preds = d_predictor.predict_next(d_df, year, round_num, cost_df=d_cost_df)

    # 2. Predict Constructors
    c_train = c_df[
        (c_df['year'] < year) | ((c_df['year'] == year) & (c_df['round'] < round_num))
    ].copy()

    if 'drivers_predicted_points_sum' not in c_train.columns:
        actual_sums = (
            d_df.groupby(['year', 'round', 'constructor_id'])['fantasy_points']
            .sum().reset_index()
            .rename(columns={'fantasy_points': 'drivers_predicted_points_sum'})
        )
        c_train = pd.merge(c_train, actual_sums, on=['year', 'round', 'constructor_id'], how='left')

    c_predictor = F1FantasyPredictor(asset_type='constructor')
    if not c_train.empty:
        c_predictor.train(c_train)

    driver_sums = d_preds.groupby('constructor_id')['predicted_points'].sum().to_dict()
    c_preds = c_predictor.predict_next(
        c_df, year, round_num,
        drivers_predicted_sum_map=driver_sums,
        cost_df=c_cost_df,
    )

    ev_df = pd.concat([d_preds, c_preds]).sort_values('predicted_points', ascending=False)

    ev_file = data_dir / "ev_reports" / f"ev_report_{year}_R{round_num}.csv"
    ev_file.parent.mkdir(parents=True, exist_ok=True)
    ev_df.to_csv(ev_file, index=False)
    print(f"EV Report saved to {ev_file}")

    print("\n--- Top Drivers (Expected Points) ---")
    drivers = ev_df[ev_df['asset_type'] == 'driver']
    print(drivers[['asset_id', 'predicted_points', 'cost']].to_string(index=False))

    print("\n--- Top Constructors (Expected Points) ---")
    constructors = ev_df[ev_df['asset_type'] == 'constructor']
    print(constructors[['asset_id', 'predicted_points', 'cost']].to_string(index=False))

    return ev_df, year, round_num


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate EV predictions for a race round")
    parser.add_argument("--year",  type=int, default=None,
                        help="Season year (auto-detected if omitted)")
    parser.add_argument("--round", type=int, default=None,
                        help="Round number (auto-detected if omitted)")
    args = parser.parse_args()
    generate_ev_report(args.year, args.round)
