"""
Walk-forward evaluation of the EV model against Game Scores (2023–2025).

Training: Proxy Scores from processed_fantasy_*.csv (all prior rounds).
Ground truth: Game Scores from actual_fantasy_*.csv (inner-joined per round).

Outputs:
  data/eval_driver_results.csv
  data/eval_constructor_results.csv
  data/plots/eval_driver_spearman.png
  data/plots/eval_constructor_spearman.png
  data/plots/eval_driver_residuals.png

See docs/adr/0005-game-score-evaluation-against-proxy-score-training.md.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.model import F1FantasyPredictor

EVAL_YEARS   = (2023, 2024, 2025)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
PLOTS_DIR    = DATA_DIR / "plots"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_driver_sums(c_df: pd.DataFrame, d_df: pd.DataFrame) -> pd.DataFrame:
    """Attach proxy driver point sums as drivers_predicted_points_sum."""
    sums = (
        d_df.groupby(['year', 'round', 'constructor_id'])['fantasy_points']
        .sum().reset_index()
        .rename(columns={'fantasy_points': 'drivers_predicted_points_sum'})
    )
    return pd.merge(c_df, sums, on=['year', 'round', 'constructor_id'], how='left')


def _plot_spearman(correlations: list, round_labels: list, mean_rho: float,
                   asset_type: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(max(10, len(correlations) * 0.45), 4))
    colors = ['steelblue' if r >= 0 else 'tomato' for r in correlations]
    ax.bar(range(len(correlations)), correlations, color=colors, alpha=0.8)
    ax.axhline(mean_rho, color='black', linestyle='--', linewidth=1.2,
               label=f'Mean ρ = {mean_rho:.3f}')
    ax.axhline(0, color='grey', linewidth=0.8)
    ax.set_xticks(range(len(correlations)))
    ax.set_xticklabels(round_labels, fontsize=6)
    ax.set_ylabel('Spearman ρ')
    ax.set_ylim(-1, 1)
    ax.set_title(f'{asset_type.capitalize()} EV Model — Per-Round Ranking Correlation '
                 f'vs Game Scores ({EVAL_YEARS[0]}–{EVAL_YEARS[-1]})')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Spearman plot → {output_path}")


def _plot_residuals(results_df: pd.DataFrame, id_col: str,
                    asset_type: str, output_path: Path):
    fig, ax = plt.subplots(figsize=(15, 10))
    assets = sorted(results_df[id_col].unique())
    cmap   = plt.get_cmap('tab20')
    for i, asset in enumerate(assets):
        data = results_df[results_df[id_col] == asset].sort_values(['year', 'round'])
        ax.plot(range(len(data)), data['residual'], 'o-',
                color=cmap(i % 20), alpha=0.6, markersize=5, label=asset)
    ax.axhline(0, color='black', linewidth=1.5)
    ax.set_xlabel('Round (chronological)')
    ax.set_ylabel('Residual — Predicted EV minus Game Score (pts)')
    ax.set_title(f'{asset_type.capitalize()} EV Model — Residuals vs Game Scores '
                 f'({EVAL_YEARS[0]}–{EVAL_YEARS[-1]})')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='small')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Residuals plot → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def evaluate():
    proxy_d = pd.read_csv(DATA_DIR / "processed_fantasy_drivers.csv")
    proxy_c = pd.read_csv(DATA_DIR / "processed_fantasy_constructors.csv")
    actual_d = pd.read_csv(DATA_DIR / "actual_fantasy_drivers.csv")
    actual_c = pd.read_csv(DATA_DIR / "actual_fantasy_constructors.csv")

    proxy_c_with_sums = _add_driver_sums(proxy_c, proxy_d)

    eval_rounds = (
        proxy_d[proxy_d['year'].isin(EVAL_YEARS)][['year', 'round']]
        .drop_duplicates()
        .sort_values(['year', 'round'])
        .values.tolist()
    )
    print(f"Evaluating {len(eval_rounds)} rounds ({EVAL_YEARS[0]}–{EVAL_YEARS[-1]}) "
          f"against Game Scores...")

    d_results, c_results = [], []
    d_corrs, d_labels    = [], []
    c_corrs, c_labels    = [], []

    for year, rnd in eval_rounds:
        # Walk-forward training filter: strictly prior rounds only
        d_train = proxy_d[
            (proxy_d['year'] < year) |
            ((proxy_d['year'] == year) & (proxy_d['round'] < rnd))
        ]
        c_train = proxy_c_with_sums[
            (proxy_c_with_sums['year'] < year) |
            ((proxy_c_with_sums['year'] == year) & (proxy_c_with_sums['round'] < rnd))
        ]

        # --- Driver model ---
        d_pred = F1FantasyPredictor(asset_type='driver')
        if not d_train.empty:
            d_pred.train(d_train)
        d_preds = d_pred.predict_next(proxy_d, year, rnd)

        driver_ev_sums = d_preds.groupby('constructor_id')['predicted_points'].sum().to_dict()

        # Join predictions with Game Scores (inner — drops phantom entries)
        d_game = actual_d[
            (actual_d['year'] == year) & (actual_d['round'] == rnd)
        ][['driver_id', 'actual_points']]
        d_merged = pd.merge(
            d_preds, d_game,
            left_on='asset_id', right_on='driver_id', how='inner'
        )
        d_merged['year']     = year
        d_merged['round']    = rnd
        d_merged['residual'] = d_merged['predicted_points'] - d_merged['actual_points']
        d_results.append(d_merged)

        if len(d_merged) >= 2:
            rho, _ = spearmanr(d_merged['predicted_points'], d_merged['actual_points'])
            if not np.isnan(rho):
                d_corrs.append(rho)
                d_labels.append(f"{year}\nR{rnd}")

        # --- Constructor model ---
        c_pred = F1FantasyPredictor(asset_type='constructor')
        if not c_train.empty:
            c_pred.train(c_train)
        c_preds = c_pred.predict_next(
            proxy_c, year, rnd, drivers_predicted_sum_map=driver_ev_sums
        )

        c_game = actual_c[
            (actual_c['year'] == year) & (actual_c['round'] == rnd)
        ][['constructor_id', 'actual_points']]
        c_merged = pd.merge(
            c_preds, c_game,
            left_on='asset_id', right_on='constructor_id', how='inner'
        )
        c_merged['year']     = year
        c_merged['round']    = rnd
        c_merged['residual'] = c_merged['predicted_points'] - c_merged['actual_points']
        c_results.append(c_merged)

        if len(c_merged) >= 2:
            rho, _ = spearmanr(c_merged['predicted_points'], c_merged['actual_points'])
            if not np.isnan(rho):
                c_corrs.append(rho)
                c_labels.append(f"{year}\nR{rnd}")

    print("Evaluation complete.\n")

    all_d = pd.concat(d_results, ignore_index=True)
    all_c = pd.concat(c_results, ignore_index=True)

    all_d.to_csv(DATA_DIR / "eval_driver_results.csv",      index=False)
    all_c.to_csv(DATA_DIR / "eval_constructor_results.csv", index=False)

    # Spearman summary
    d_mean_rho = float(np.mean(d_corrs))
    c_mean_rho = float(np.mean(c_corrs))
    d_rmse = float(np.sqrt((all_d['residual'] ** 2).mean()))
    c_rmse = float(np.sqrt((all_c['residual'] ** 2).mean()))

    print(f"Driver      — mean Spearman ρ : {d_mean_rho:.3f}  "
          f"(std {np.std(d_corrs):.3f})   RMSE vs Game Score: {d_rmse:.2f} pts")
    print(f"Constructor — mean Spearman ρ : {c_mean_rho:.3f}  "
          f"(std {np.std(c_corrs):.3f})   RMSE vs Game Score: {c_rmse:.2f} pts")

    _plot_spearman(d_corrs, d_labels, d_mean_rho, 'driver',
                   PLOTS_DIR / 'eval_driver_spearman.png')
    _plot_spearman(c_corrs, c_labels, c_mean_rho, 'constructor',
                   PLOTS_DIR / 'eval_constructor_spearman.png')
    _plot_residuals(all_d, 'driver_id',      'driver',
                    PLOTS_DIR / 'eval_driver_residuals.png')


if __name__ == "__main__":
    evaluate()
