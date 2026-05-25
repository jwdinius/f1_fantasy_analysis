"""
One-shot model training with chronological 70-15-15 split.

Splits all available rounds in time order (no shuffling), trains both the
Driver and Constructor EV predictors with early stopping, saves them to
models/, and writes train/validation loss curves to data/plots/.

Usage:
    PYTHONPATH=. python src/train_model.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error

from src.model import F1FantasyPredictor

TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# test = remaining 0.15

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
PLOTS_DIR    = DATA_DIR / "plots"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chronological_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split df by round boundaries in chronological order."""
    rounds = (
        df[['year', 'round']].drop_duplicates()
        .sort_values(['year', 'round'])
        .reset_index(drop=True)
    )
    n       = len(rounds)
    n_train = int(n * TRAIN_FRAC)
    n_val   = int(n * VAL_FRAC)

    train_rounds = rounds.iloc[:n_train]
    val_rounds   = rounds.iloc[n_train : n_train + n_val]
    test_rounds  = rounds.iloc[n_train + n_val :]

    def _filter(rnd_set):
        return df.merge(rnd_set, on=['year', 'round'], how='inner').reset_index(drop=True)

    train_df = _filter(train_rounds)
    val_df   = _filter(val_rounds)
    test_df  = _filter(test_rounds)

    t_start, t_end = train_rounds.iloc[0],  train_rounds.iloc[-1]
    v_start, v_end = val_rounds.iloc[0],    val_rounds.iloc[-1]
    s_start, s_end = test_rounds.iloc[0],   test_rounds.iloc[-1]
    print(f"  Train : {len(train_rounds):3d} rounds  "
          f"({int(t_start['year'])} R{int(t_start['round'])} → "
          f"{int(t_end['year'])} R{int(t_end['round'])})  {len(train_df)} rows")
    print(f"  Val   : {len(val_rounds):3d} rounds  "
          f"({int(v_start['year'])} R{int(v_start['round'])} → "
          f"{int(v_end['year'])} R{int(v_end['round'])})  {len(val_df)} rows")
    print(f"  Test  : {len(test_rounds):3d} rounds  "
          f"({int(s_start['year'])} R{int(s_start['round'])} → "
          f"{int(s_end['year'])} R{int(s_end['round'])})  {len(test_df)} rows")

    return train_df, val_df, test_df


def recency_weights(df: pd.DataFrame, half_life_years: float) -> np.ndarray:
    """
    Exponential decay sample weights: w = 2^(-age / half_life).
    Age is measured in fractional years from the most recent (year, round) in df.
    """
    rounds_per_year = df.groupby('year')['round'].max().mean()
    df_frac = df['year'] + df['round'] / rounds_per_year
    age = df_frac.max() - df_frac
    return np.power(2.0, -age / half_life_years).values


def plot_loss_curves(evals_result: dict, asset_type: str, output_path: Path):
    train_rmse = evals_result.get('validation_0', {}).get('rmse', [])
    val_rmse   = evals_result.get('validation_1', {}).get('rmse', [])
    if not train_rmse:
        print(f"  No eval results to plot for {asset_type}.")
        return

    best_round = int(np.argmin(val_rmse)) + 1
    rounds = range(1, len(train_rmse) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(rounds, train_rmse, label='Train RMSE', linewidth=1.5)
    ax.plot(rounds, val_rmse,   label='Validation RMSE', linewidth=1.5)
    ax.axvline(best_round, color='grey', linestyle='--', linewidth=1,
               label=f'Best round ({best_round})')
    ax.set_xlabel('Boosting Round')
    ax.set_ylabel('RMSE (Fantasy Points)')
    ax.set_title(f'{asset_type.capitalize()} EV Model — Train vs Validation Loss')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Loss curve → {output_path}")


def evaluate_test(predictor: F1FantasyPredictor, test_df: pd.DataFrame,
                  asset_type: str, output_path: Path):
    """Compute test RMSE and per-round Spearman rank correlation, plot the latter."""
    test_clean = test_df.dropna(subset=['fantasy_points']).copy()
    X_test = test_clean[predictor.feature_cols].fillna(-1)
    y_test = test_clean['fantasy_points']
    preds  = predictor.model.predict(X_test)

    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))

    test_clean['predicted'] = preds
    correlations, round_labels = [], []
    for (year, rnd), grp in test_clean.groupby(['year', 'round']):
        if len(grp) < 2:
            continue
        rho, _ = spearmanr(grp['predicted'], grp['fantasy_points'])
        if not np.isnan(rho):
            correlations.append(rho)
            round_labels.append(f"{int(year)}\nR{int(rnd)}")

    mean_rho = float(np.mean(correlations))
    std_rho  = float(np.std(correlations))

    # Plot per-round Spearman ρ
    fig, ax = plt.subplots(figsize=(max(10, len(correlations) * 0.4), 4))
    bar_colors = ['steelblue' if r >= 0 else 'tomato' for r in correlations]
    ax.bar(range(len(correlations)), correlations, color=bar_colors, alpha=0.8)
    ax.axhline(mean_rho, color='black', linestyle='--', linewidth=1.2,
               label=f'Mean ρ = {mean_rho:.3f}')
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_xticks(range(len(correlations)))
    ax.set_xticklabels(round_labels, fontsize=7)
    ax.set_ylabel('Spearman ρ')
    ax.set_title(f'{asset_type.capitalize()} EV Model — Per-Round Ranking Correlation (test set)')
    ax.set_ylim(-1, 1)
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Spearman plot → {output_path}")

    return rmse, mean_rho, std_rho


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(min_year: int = 2015, half_life: float = 3.0):
    d_df = pd.read_csv(DATA_DIR / "processed_fantasy_drivers.csv")
    c_df = pd.read_csv(DATA_DIR / "processed_fantasy_constructors.csv")

    if min_year > 2015:
        print(f"Restricting to {min_year}+ (dropping {len(d_df[d_df['year'] < min_year])} driver rows, "
              f"{len(c_df[c_df['year'] < min_year])} constructor rows)")
        d_df = d_df[d_df['year'] >= min_year].reset_index(drop=True)
        c_df = c_df[c_df['year'] >= min_year].reset_index(drop=True)

    # Attach actual driver point sums to constructor data (proxy for
    # drivers_predicted_points_sum during static training).
    actual_driver_sums = (
        d_df.groupby(['year', 'round', 'constructor_id'])['fantasy_points']
        .sum().reset_index()
        .rename(columns={'fantasy_points': 'drivers_predicted_points_sum'})
    )
    c_df = pd.merge(c_df, actual_driver_sums, on=['year', 'round', 'constructor_id'], how='left')

    # --- Driver model ---
    print("\n=== Driver Model ===")
    d_train, d_val, d_test = chronological_split(d_df)

    d_weights = recency_weights(d_train, half_life)
    print(f"  Recency weights — half-life {half_life}y  "
          f"min={d_weights.min():.3f}  max={d_weights.max():.3f}")

    d_predictor = F1FantasyPredictor(asset_type='driver')
    d_predictor.train(d_train, val_df=d_val, sample_weight=pd.Series(d_weights, index=d_train.index))

    plot_loss_curves(
        d_predictor.evals_result_,
        asset_type='driver',
        output_path=PLOTS_DIR / 'driver_loss_curves.png',
    )

    d_rmse, d_rho, d_rho_std = evaluate_test(
        d_predictor, d_test, 'driver',
        PLOTS_DIR / 'driver_spearman.png',
    )
    print(f"  Driver test RMSE          : {d_rmse:.2f} fantasy points")
    print(f"  Driver mean Spearman ρ    : {d_rho:.3f}  (std {d_rho_std:.3f})")

    d_predictor.save(MODELS_DIR / 'driver_predictor.joblib')

    # --- Constructor model ---
    print("\n=== Constructor Model ===")
    c_train, c_val, c_test = chronological_split(c_df)

    c_weights = recency_weights(c_train, half_life)
    print(f"  Recency weights — half-life {half_life}y  "
          f"min={c_weights.min():.3f}  max={c_weights.max():.3f}")

    c_predictor = F1FantasyPredictor(asset_type='constructor')
    c_predictor.train(c_train, val_df=c_val, sample_weight=pd.Series(c_weights, index=c_train.index))

    plot_loss_curves(
        c_predictor.evals_result_,
        asset_type='constructor',
        output_path=PLOTS_DIR / 'constructor_loss_curves.png',
    )

    c_rmse, c_rho, c_rho_std = evaluate_test(
        c_predictor, c_test, 'constructor',
        PLOTS_DIR / 'constructor_spearman.png',
    )
    print(f"  Constructor test RMSE          : {c_rmse:.2f} fantasy points")
    print(f"  Constructor mean Spearman ρ    : {c_rho:.3f}  (std {c_rho_std:.3f})")

    c_predictor.save(MODELS_DIR / 'constructor_predictor.joblib')

    print("\nDone.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-year", type=int, default=2015,
                        help="Exclude rounds before this year (default: 2015)")
    parser.add_argument("--half-life", type=float, default=3.0,
                        help="Recency weight half-life in years (default: 3.0)")
    args = parser.parse_args()
    main(min_year=args.min_year, half_life=args.half_life)
