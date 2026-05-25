# Rolling Window Aggregates Over Per-Round Lag Columns

The original feature engineering approach stored each Training Window slot as a separate column (e.g., `race_pos_n_1`, `race_pos_n_2`, `race_pos_n_3`), producing a feature vector whose width grows linearly with window size N. We replaced this with rolling aggregates — median, min, and max for continuous features; sum for binary features — producing a fixed-width feature vector regardless of N.

This makes window width a tunable hyperparameter: the same pipeline can be evaluated at N=3, 5, 7, etc. without any schema changes, enabling direct comparative analysis. Median is preferred over mean because sample sizes within a window are small (≤7) and a single outlier result (e.g., a DNF) would disproportionately shift an arithmetic mean; median, min, and max together capture central tendency, floor, and ceiling performance more robustly.

DNF and DSQ results, which carry no valid finishing position, are imputed with 25 (above the maximum grid position of 20) before aggregation, reflecting that these outcomes score negatively in the fantasy game.

## Considered Options

- **Per-round lag columns** (`*_n_1`, `*_n_2`, `*_n_3`) — rejected because the feature vector widens with N, making window-width comparison require retraining on incompatible schemas, and because raw per-round values expose the model to position-in-window ordering artefacts.
- **Rolling mean** — rejected in favour of median/min/max because arithmetic mean is unstable for small samples and obscures the range of recent performance.
