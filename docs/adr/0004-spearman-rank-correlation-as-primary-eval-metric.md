# Spearman Rank Correlation as Primary Evaluation Metric

The EV model's output is consumed by the LP optimizer, which selects a fantasy Team by ranking assets against each other within a budget constraint. What matters for that use case is not whether the model predicts the correct absolute score for each asset, but whether it correctly orders assets — identifying which drivers and constructors will score more than others in a given round.

RMSE measures absolute prediction error uniformly across all assets and rounds. It is dominated by high-variance events that are fundamentally unforeseeable at Team Lock time: first-lap incidents, safety-car beneficiaries, and driver-of-the-day votes. A model that always predicts the per-round mean achieves a driver RMSE of 16.92, while a trained model achieves 14.64 — only 13% better than the null model. Evaluated by RMSE alone, the driver model appears nearly useless.

Per-round Spearman rank correlation (ρ) measures whether the model's ranking of assets within a round agrees with the actual finishing order of fantasy scores. Evaluated on the test set (2024 R13 – 2025 R24), the driver model achieves a mean ρ of 0.514 (std 0.192) and the constructor model achieves a mean ρ of 0.956 (std 0.046). A random ranker would score ρ ≈ 0; both models are substantially above that floor. The constructor model reliably orders constructors by score, and the driver model correctly orders the field more often than not — enough to provide a consistent advantage when selecting from the top of the ranking.

Spearman ρ is computed per round (not across the full dataset) and then averaged, because the optimizer operates on one round at a time. A model that ranks assets correctly in 30 rounds but badly in 6 is more useful than the pooled correlation suggests; the per-round distribution (captured by the std) exposes this variance.

RMSE is retained as a secondary diagnostic — a large RMSE relative to score variance can indicate data quality problems or target leakage — but it does not gate model acceptance.

## Considered Options

- **RMSE only** — rejected as the primary metric because it is insensitive to ranking and dominated by irreducible variance from unpredictable race events. A model can achieve low RMSE by regressing toward the mean without producing useful rankings.

- **Per-round Spearman ρ (chosen)** — directly measures what the optimizer needs. Robust to outlier scores because it operates on ranks, not values. Computed per round and averaged to match the optimizer's decision granularity.

- **Kendall's τ** — equivalent in spirit to Spearman ρ for this use case. Spearman preferred for familiarity and direct interpretability (ρ = 1 is a perfect ranking, ρ = 0 is random).

- **NDCG (Normalized Discounted Cumulative Gain)** — appropriate when top-ranked assets matter more than lower-ranked ones, which is true here (the optimizer picks 5 drivers and 2 constructors, so only the top ~25% of the driver field matters). Not adopted in the initial implementation because it requires defining a relevance scale from raw fantasy scores, adding a tunable parameter. Revisit if top-of-ranking accuracy proves more important than overall ranking quality.
