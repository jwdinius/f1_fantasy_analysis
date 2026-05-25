"""
F1 Fantasy Analysis — Data Processing Pipeline

Produces:
  data/processed_fantasy_drivers.csv
  data/processed_fantasy_constructors.csv

Each row is one (asset, round) pair with:
  - fantasy_points: target variable (2025 scoring rules applied uniformly)
  - Current-round features: f1db Practice Gaps + FastF1 median gap / IQR
  - Training Window features: rolling aggregates over N prior rounds
    - Continuous (race_pos, quali_pos, pos_gained, fantasy_points,
                  fastest_pit_stop): median, min, max
    - Binary (dotd, fastest_lap, both_q3, one_q3, both_q2): sum

DNF/DSQ race positions are imputed with DNF_DSQ_IMPUTE (25) before rolling.
Training Window width is configurable via --window (default 3).
"""

import argparse
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

from src.database import F1DataManager
from src.scoring import F1FantasyScorer2025

# ---------------------------------------------------------------------------
# FastF1 cache
# ---------------------------------------------------------------------------
_cache_dir = Path(".fastf1")
_cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(_cache_dir)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONSTRUCTOR_REBRANDING = {
    'racing-point': 'aston-martin',
    'alphatauri': 'racing-bulls',
    'toro-rosso': 'racing-bulls',
    'rb': 'racing-bulls',
    'alfa-romeo': 'kick-sauber',
    'sauber': 'kick-sauber',
    'renault': 'alpine',
}

# Imputed position for DNF / DSQ — above the 20-car grid, scores negatively
DNF_DSQ_IMPUTE = 25

DEFAULT_WINDOW = 3

DRIVER_CONTINUOUS_COLS = ['race_pos', 'quali_pos', 'pos_gained', 'fantasy_points']
DRIVER_BINARY_COLS = ['dotd', 'fastest_lap']
CONSTRUCTOR_CONTINUOUS_COLS = ['fastest_pit_stop']
CONSTRUCTOR_BINARY_COLS = ['both_q3', 'one_q3', 'both_q2']

# Cost CSV team-name → canonical constructor_id (after rebranding)
CONSTRUCTOR_COST_MAP = {
    'Red Bull': 'red-bull',
    'McLaren': 'mclaren',
    'Ferrari': 'ferrari',
    'Mercedes': 'mercedes',
    'Aston Martin': 'aston-martin',
    'Alpine': 'alpine',
    'Williams': 'williams',
    'AlphaTauri': 'racing-bulls',
    'RB': 'racing-bulls',
    'Racing Bulls': 'racing-bulls',
    'Alfa Romeo': 'kick-sauber',
    'Kick Sauber': 'kick-sauber',
    'Haas': 'haas',
    'Haas F1 Team': 'haas',
}

# Overrides for driver last names that are ambiguous or differ from f1db
LAST_NAME_OVERRIDES = {
    'Hamilton': 'lewis-hamilton',
    'Schumacher': 'mick-schumacher',
    'Verstappen': 'max-verstappen',
    'Sainz': 'carlos-sainz-jr',
    'Zhou': 'guanyu-zhou',
    'Magnussen': 'kevin-magnussen',
    'Hulkenberg': 'nico-hulkenberg',
}

# ---------------------------------------------------------------------------
# Cost loading
# ---------------------------------------------------------------------------

def _load_cost_csv(path: Path) -> pd.DataFrame:
    """Read a cost CSV, dropping the trailing Average column if present."""
    df = pd.read_csv(path, index_col=0)
    if 'Average' in df.columns:
        df = df.drop(columns=['Average'])
    return df


def load_costs(year: int, db: F1DataManager):
    """
    Returns (driver_costs, constructor_costs) as dicts keyed by
    (id, round_num) → cost in millions.

    Cost data is only available from 2022 onwards; returns empty dicts
    for earlier years.
    """
    project_root = Path(__file__).resolve().parent.parent
    driver_cost_path = project_root / "data" / "fantasy_csv" / f"Drivers-Cost-{year}.csv"
    team_cost_path = project_root / "data" / "fantasy_csv" / f"Teams-Cost-{year}.csv"

    if not driver_cost_path.exists():
        return {}, {}

    rounds = db.query(
        f"SELECT round FROM race WHERE year = {year} ORDER BY round"
    )['round'].tolist()

    drivers_df = db.query("SELECT id, last_name FROM driver")
    last_name_to_id = {row['last_name']: row['id'] for _, row in drivers_df.iterrows()}
    last_name_to_id.update(LAST_NAME_OVERRIDES)

    driver_costs = {}
    d_csv = _load_cost_csv(driver_cost_path)
    num_rounds = min(len(d_csv.columns), len(rounds))
    for name, row in d_csv.iterrows():
        d_id = last_name_to_id.get(str(name).strip())
        if not d_id:
            continue
        for col_idx in range(num_rounds):
            val = row.iloc[col_idx]
            if pd.notna(val):
                driver_costs[(d_id, rounds[col_idx])] = float(val)

    constructor_costs = {}
    if team_cost_path.exists():
        t_csv = _load_cost_csv(team_cost_path)
        for name, row in t_csv.iterrows():
            c_id = CONSTRUCTOR_COST_MAP.get(str(name).strip())
            if not c_id:
                continue
            for col_idx in range(num_rounds):
                val = row.iloc[col_idx]
                if pd.notna(val):
                    constructor_costs[(c_id, rounds[col_idx])] = float(val)

    return driver_costs, constructor_costs


# ---------------------------------------------------------------------------
# FastF1 feature extraction
# ---------------------------------------------------------------------------

def fetch_fastf1_session_stats(year: int, round_num: int, session_type: str) -> pd.DataFrame:
    """
    Fetch per-driver median Practice Gap and IQR for one session.

    Returns a DataFrame with columns:
      driver_number (str), median_gap (sec), iqr (sec)

    Returns an empty DataFrame on any failure (missing cache, old season, etc.).
    session_type: 'FP1', 'FP2', or 'FP3'
    """
    try:
        session = fastf1.get_session(year, round_num, session_type)
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps.dropna(subset=['LapTime'])
        if laps.empty:
            return pd.DataFrame()

        fastest_lap_sec = laps['LapTime'].min().total_seconds()
        stats = []
        for drv_code in laps['Driver'].unique():
            try:
                drv_laps = laps[laps['Driver'] == drv_code]['LapTime'].dt.total_seconds()
                drv_info = session.get_driver(drv_code)
                d_num = str(int(drv_info['DriverNumber']))
                stats.append({
                    'driver_number': d_num,
                    'median_gap': drv_laps.median() - fastest_lap_sec,
                    'iqr': float(drv_laps.quantile(0.75) - drv_laps.quantile(0.25)),
                })
            except Exception:
                continue
        return pd.DataFrame(stats)
    except Exception:
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Raw data extraction + scoring
# ---------------------------------------------------------------------------

def _get_session_row(results_df: pd.DataFrame, driver_id: str):
    """Return the first matching row as a Series, or None."""
    rows = results_df[results_df['driver_id'] == driver_id]
    return rows.iloc[0] if not rows.empty else None


def _quali_bonus_flags(team_q_positions: list) -> tuple:
    """
    Return (both_q3, one_q3, both_q2) binary flags from a list of
    qualifying position numbers.
    """
    valid = [p for p in team_q_positions if p is not None and pd.notna(p) and p > 0]
    if not valid:
        return 0, 0, 0
    min_q = min(valid)
    max_q = max(valid) if len(valid) > 1 else 99
    both_q3 = 1 if min_q <= 10 and max_q <= 10 else 0
    one_q3 = 1 if min_q <= 10 and max_q > 10 else 0
    both_q2 = 1 if min_q > 10 and max_q <= 15 else 0
    return both_q3, one_q3, both_q2


def get_all_rows(db: F1DataManager, year: int) -> tuple:
    """
    Extract per-driver and per-constructor rows for all rounds in year.
    Returns (driver_rows, constructor_rows) as lists of dicts.
    FastF1 columns are set to NaN here and filled in by the caller.
    """
    scorer = F1FantasyScorer2025()
    races = db.query(f"SELECT * FROM race WHERE year = {year} ORDER BY round")
    driver_rows = []
    constructor_rows = []

    for _, race in races.iterrows():
        race_id = race['id']
        round_num = race['round']
        print(f"  R{round_num}", end='... ', flush=True)

        race_results = db.query(
            f"SELECT * FROM race_result WHERE race_id = {race_id}"
        )
        if race_results.empty:
            print("no results, skipping")
            continue

        quali_results = db.query(
            f"SELECT * FROM qualifying_result WHERE race_id = {race_id}"
        )
        sprint_results = db.query(
            f"SELECT * FROM sprint_race_result WHERE race_id = {race_id}"
        )
        fp1_results = db.query(
            f"SELECT * FROM free_practice_1_result WHERE race_id = {race_id}"
        )
        fp2_results = db.query(
            f"SELECT * FROM free_practice_2_result WHERE race_id = {race_id}"
        )
        fp3_results = db.query(
            f"SELECT * FROM free_practice_3_result WHERE race_id = {race_id}"
        )
        pit_data = db.query(
            f"SELECT * FROM race_data WHERE race_id = {race_id} AND type = 'PIT_STOP'"
        )
        dotd_data = db.query(
            f"SELECT * FROM race_data WHERE race_id = {race_id}"
            f" AND type = 'DRIVER_OF_THE_DAY_RESULT'"
        )

        # DotD winner (highest vote percentage)
        dotd_driver_id = None
        if not dotd_data.empty:
            valid_dotd = dotd_data.dropna(subset=['driver_of_the_day_percentage'])
            if not valid_dotd.empty:
                dotd_driver_id = valid_dotd.loc[
                    valid_dotd['driver_of_the_day_percentage'].idxmax(), 'driver_id'
                ]

        # Fastest pit stop time across all teams (for constructor bonus)
        fastest_overall_millis = None
        if not pit_data.empty:
            times = pit_data['pit_stop_time_millis'].dropna()
            if not times.empty:
                fastest_overall_millis = times.min()

        # --- Driver rows ---
        driver_contribs: dict = {}
        driver_dsq: dict = {}

        for _, r in race_results.iterrows():
            d_id = r['driver_id']
            c_id = CONSTRUCTOR_REBRANDING.get(r['constructor_id'], r['constructor_id'])

            q_res = _get_session_row(quali_results, d_id)
            s_res = _get_session_row(sprint_results, d_id)

            total, contrib, is_dsq = scorer.calculate_driver_score(
                r, s_res, q_res, dotd_driver_id
            )
            driver_contribs[d_id] = contrib
            driver_dsq[d_id] = is_dsq

            # Practice Gaps from f1db (best lap vs session fastest, in seconds)
            def _fp_gap(fp_df):
                rows = fp_df[fp_df['driver_id'] == d_id]
                if rows.empty:
                    return np.nan
                millis = rows.iloc[0]['gap_millis']
                return millis / 1000.0 if pd.notna(millis) else np.nan

            race_pos = r['position_number'] if pd.notna(r['position_number']) else np.nan
            quali_pos = (
                q_res['position_number']
                if q_res is not None and pd.notna(q_res['position_number'])
                else np.nan
            )
            pos_gained = r.get('positions_gained', 0)
            pos_gained = float(pos_gained) if pd.notna(pos_gained) else 0.0

            driver_rows.append({
                'year': year,
                'round': round_num,
                'driver_id': d_id,
                'constructor_id': c_id,
                'driver_number': str(r['driver_number']),
                'fantasy_points': float(total),
                'race_pos': race_pos,
                'quali_pos': quali_pos,
                'pos_gained': pos_gained,
                'dotd': 1 if d_id == dotd_driver_id else 0,
                'fastest_lap': 1 if r.get('fastest_lap', 0) else 0,
                'fp1_gap_sec': _fp_gap(fp1_results),
                'fp2_gap_sec': _fp_gap(fp2_results),
                'fp3_gap_sec': _fp_gap(fp3_results),
                # FastF1 features — filled in after this loop
                'lap_time_median_gap_fp1': np.nan,
                'lap_time_iqr_fp1': np.nan,
                'lap_time_median_gap_fp2': np.nan,
                'lap_time_iqr_fp2': np.nan,
                'lap_time_median_gap_fp3': np.nan,
                'lap_time_iqr_fp3': np.nan,
                'cost': np.nan,
            })

        # --- Constructor rows ---
        for c_id_raw in race_results['constructor_id'].unique():
            c_id = CONSTRUCTOR_REBRANDING.get(c_id_raw, c_id_raw)
            team_driver_ids = race_results[
                race_results['constructor_id'] == c_id_raw
            ]['driver_id'].tolist()

            contrib_sum = sum(driver_contribs.get(d, 0) for d in team_driver_ids)
            dsq_list = [driver_dsq.get(d, False) for d in team_driver_ids]

            team_q = quali_results[quali_results['constructor_id'] == c_id_raw]
            team_q_pos = team_q['position_number'].tolist()

            team_pit_rows = pit_data[pit_data['constructor_id'] == c_id_raw]
            team_pit_times = [
                t / 1000.0
                for t in team_pit_rows['pit_stop_time_millis'].dropna().tolist()
            ]
            fastest_pit = min(team_pit_times) if team_pit_times else np.nan

            is_fastest_team = (
                fastest_overall_millis is not None
                and bool(team_pit_rows['pit_stop_time_millis'].dropna().tolist())
                and team_pit_rows['pit_stop_time_millis'].dropna().min()
                == fastest_overall_millis
            )

            c_score = scorer.calculate_constructor_score(
                contrib_sum, team_q_pos, team_pit_times, is_fastest_team, dsq_list
            )
            both_q3, one_q3, both_q2 = _quali_bonus_flags(team_q_pos)

            constructor_rows.append({
                'year': year,
                'round': round_num,
                'constructor_id': c_id,
                'fantasy_points': float(c_score),
                'fastest_pit_stop': fastest_pit,
                'both_q3': both_q3,
                'one_q3': one_q3,
                'both_q2': both_q2,
                'cost': np.nan,
            })

        print("done")

    return driver_rows, constructor_rows


# ---------------------------------------------------------------------------
# FastF1 enrichment (batch — called once per round after f1db extraction)
# ---------------------------------------------------------------------------

def enrich_with_fastf1(driver_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (year, round) in driver_df, fetch FP1/FP2/FP3 stats from FastF1
    and fill in lap_time_median_gap_fp* and lap_time_iqr_fp* columns.
    Merges on driver_number.
    """
    for (year, round_num), group in driver_df.groupby(['year', 'round']):
        for session_type, suffix in [('FP1', 'fp1'), ('FP2', 'fp2'), ('FP3', 'fp3')]:
            stats = fetch_fastf1_session_stats(int(year), int(round_num), session_type)
            if stats.empty:
                continue
            stats = stats.rename(columns={
                'median_gap': f'lap_time_median_gap_{suffix}',
                'iqr': f'lap_time_iqr_{suffix}',
            })
            mask = (driver_df['year'] == year) & (driver_df['round'] == round_num)
            merged = driver_df.loc[mask].merge(
                stats, on='driver_number', how='left', suffixes=('', '_ff1')
            )
            driver_df.loc[mask, f'lap_time_median_gap_{suffix}'] = merged[
                f'lap_time_median_gap_{suffix}_ff1'
            ].values
            driver_df.loc[mask, f'lap_time_iqr_{suffix}'] = merged[
                f'lap_time_iqr_{suffix}_ff1'
            ].values
    return driver_df


# ---------------------------------------------------------------------------
# Cost enrichment
# ---------------------------------------------------------------------------

def enrich_with_costs(driver_df: pd.DataFrame, constructor_df: pd.DataFrame,
                      db: F1DataManager) -> tuple:
    """Fill in the cost column for all rows, year by year."""
    for year in driver_df['year'].unique():
        d_costs, c_costs = load_costs(int(year), db)
        for (d_id, rnd), cost in d_costs.items():
            mask = (
                (driver_df['year'] == year)
                & (driver_df['round'] == rnd)
                & (driver_df['driver_id'] == d_id)
            )
            driver_df.loc[mask, 'cost'] = cost
        for (c_id, rnd), cost in c_costs.items():
            mask = (
                (constructor_df['year'] == year)
                & (constructor_df['round'] == rnd)
                & (constructor_df['constructor_id'] == c_id)
            )
            constructor_df.loc[mask, 'cost'] = cost
    return driver_df, constructor_df


# ---------------------------------------------------------------------------
# Rolling window aggregates (Training Window features)
# ---------------------------------------------------------------------------

def add_rolling_features(df: pd.DataFrame, id_col: str,
                          continuous_cols: list, binary_cols: list,
                          window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    """
    For each asset (identified by id_col), add rolling aggregates over the
    N rounds immediately prior to each row.

    Continuous columns → _rolling_median, _rolling_min, _rolling_max
    Binary columns     → _rolling_sum

    race_pos nulls (DNF/DSQ) are imputed with DNF_DSQ_IMPUTE before rolling.
    pos_gained nulls are treated as 0 (no positions gained).
    """
    df = df.sort_values([id_col, 'year', 'round']).reset_index(drop=True)

    for col in continuous_cols:
        # Build the working series with imputation
        work = df[col].copy()
        if col == 'race_pos':
            work = work.fillna(DNF_DSQ_IMPUTE)
        elif col == 'pos_gained':
            work = work.fillna(0.0)

        df[col + '_work'] = work

        for agg_name in ('median', 'min', 'max'):
            out_col = f'{col}_rolling_{agg_name}'
            df[out_col] = (
                df.groupby(id_col)[col + '_work']
                .transform(
                    lambda x, a=agg_name: (
                        x.shift(1).rolling(window, min_periods=1).agg(a)
                    )
                )
            )

        df.drop(columns=[col + '_work'], inplace=True)

    for col in binary_cols:
        df[f'{col}_rolling_sum'] = (
            df.groupby(id_col)[col]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).sum())
        )

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(window: int = DEFAULT_WINDOW):
    db = F1DataManager()
    all_driver_rows = []
    all_constructor_rows = []

    for year in range(2015, 2026):
        print(f"\nProcessing {year}...")
        try:
            d_rows, c_rows = get_all_rows(db, year)
            all_driver_rows.extend(d_rows)
            all_constructor_rows.extend(c_rows)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    if not all_driver_rows:
        print("CRITICAL: No driver data collected.")
        return

    driver_df = pd.DataFrame(all_driver_rows)
    constructor_df = pd.DataFrame(all_constructor_rows)

    print("\nEnriching with costs...")
    driver_df, constructor_df = enrich_with_costs(driver_df, constructor_df, db)

    print("Enriching with FastF1 Practice Gap features...")
    driver_df = enrich_with_fastf1(driver_df)

    print(f"Computing Training Window rolling features (window={window})...")
    driver_df = add_rolling_features(
        driver_df, id_col='driver_id',
        continuous_cols=DRIVER_CONTINUOUS_COLS,
        binary_cols=DRIVER_BINARY_COLS,
        window=window,
    )
    constructor_df = add_rolling_features(
        constructor_df, id_col='constructor_id',
        continuous_cols=CONSTRUCTOR_CONTINUOUS_COLS,
        binary_cols=CONSTRUCTOR_BINARY_COLS,
        window=window,
    )

    # Drop working columns used internally; keep only model-relevant columns
    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "data"

    driver_out = out_dir / "processed_fantasy_drivers.csv"
    constructor_out = out_dir / "processed_fantasy_constructors.csv"

    driver_df.drop(columns=['driver_number'], inplace=True, errors='ignore')
    driver_df.to_csv(driver_out, index=False)
    constructor_df.to_csv(constructor_out, index=False)

    print(f"\nSaved {len(driver_df)} driver rows → {driver_out}")
    print(f"Saved {len(constructor_df)} constructor rows → {constructor_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1 Fantasy data processing pipeline")
    parser.add_argument(
        "--window", type=int, default=DEFAULT_WINDOW,
        help=f"Training Window width in rounds (default: {DEFAULT_WINDOW})"
    )
    args = parser.parse_args()
    main(window=args.window)
