# Model Selection: Gradient Boosted Trees (XGBoost)

The EV model must regress `fantasy_points` from a feature vector that mixes ordinal integers (race and qualifying positions, rolling aggregates thereof), continuous floats (practice lap-time gaps and IQRs), binary indicators (DotD, fastest lap, qualifying bonus flags), and asset cost. The dataset is moderately small — roughly 4,700 driver rows and 2,350 constructor rows spanning 2015–2025 — and contains structured missing data: FastF1 features are absent for rounds before 2019, and cost features are absent before 2022. Walk-forward training windows for early 2025 rounds may contain as few as a few hundred usable rows.

We use **XGBoost** (gradient boosted decision trees) as the regression model for both the Driver and Constructor EV predictors.

Tree-based models partition the feature space via axis-aligned splits, so ordinal positions, continuous gaps, and binary flags all receive the same treatment without scaling, encoding, or imputation preprocessing — a split on `fp1_gap_sec` and a split on `race_pos_rolling_median` are structurally identical. XGBoost additionally learns an optimal default branch direction for missing values during training, which means NaN FastF1 features in pre-2019 rows are handled natively rather than being imputed with a fixed sentinel that could bias the model.

The fantasy scoring function is non-linear: the points difference between P1 and P2 is much larger than between P9 and P10, and bonuses (DotD, fastest lap, qualifying tier) create discrete jumps. Tree splits capture these thresholds directly without requiring manual feature engineering of interaction terms or polynomial expansions. Feature importance scores from the trained model are also useful for iterative feature engineering — identifying which practice session or rolling aggregate is most predictive.

## Considered Options

- **Ridge / Lasso regression** — rejected because the relationship between positions and fantasy points is non-linear and would require manual construction of interaction terms and polynomial features to approximate the scoring thresholds. Also sensitive to the unscaled missing-value sentinel used for NaN features.

- **Random Forest** — viable alternative with comparable accuracy on tabular data. Rejected primarily because Random Forest does not natively handle missing values: it requires imputation before splitting, meaning the NaN pattern in FastF1 features must be explicitly encoded rather than learned. Also, Random Forest averages independent trees and cannot iteratively correct high-residual examples the way boosting can, which matters when the training window is small and a few anomalous rounds (safety car finishes, first-lap incidents) dominate the error.

- **Multi-layer perceptron (MLP)** — rejected due to dataset size. Walk-forward training windows for the first few 2025 rounds may contain only a few hundred examples, which is insufficient to reliably train a network with hidden layers. MLPs also require feature scaling and offer no native missing-value handling. Interpretability via feature importance is not available.

- **LightGBM** — functionally equivalent to XGBoost for this use case. XGBoost is preferred because it is already integrated into the codebase and the dataset is small enough that LightGBM's speed advantage is immaterial.
