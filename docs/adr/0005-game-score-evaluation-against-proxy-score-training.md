# Game Score Evaluation Against Proxy Score Training

The EV model is trained on Proxy Scores — fantasy points computed by `F1FantasyScorer2025` applied uniformly to f1db event data back to 2015. Game Scores (the points officially awarded by the F1 Fantasy platform each round) are available from 2022 onward but were not used as a training target. Despite measuring the same underlying race events, the two are correlated estimates rather than identical quantities: Game Scores reflect steward decisions, eligibility edge cases, and platform rounding that the proxy formula does not capture.

Evaluating the model against its own training signal (Proxy Scores) understates real-world error because it hides the formula gap between what the model predicts and what a fantasy player actually receives. End-to-end evaluation against Game Scores compounds both error sources — model prediction error and proxy approximation error — and therefore measures the true performance a player would experience. This is the honest evaluation target.

The evaluation window is 2023–2025: three complete seasons with full Game Score and cost coverage. 2022 is excluded because cost data is absent, and cost is a feature in the model. The current 2026 season is excluded from the fixed evaluation window but can be appended incrementally as rounds complete.

Training remains on Proxy Scores for all seasons back to 2015. Switching to Game Score targets for 2023+ would cut training volume by ~60% (25 seasons to 3) for a marginal gain in target accuracy, which empirically degraded performance. The training/evaluation target mismatch is deliberate: volume and temporal breadth in training outweigh the benefit of a more accurate but narrow training signal.

At evaluation time, driver–ground-truth joins are performed as inner joins on `(year, round, driver_id)`. Drivers appearing in Game Score data for rounds they did not compete in (zero-point spreadsheet padding) are naturally excluded because they are absent from the proxy data for those rounds.

## Considered Options

- **Evaluate against Proxy Scores** — rejected as the primary metric because it flatters the model by hiding the formula gap. Retained only as a secondary diagnostic for isolating model prediction error from proxy approximation error.

- **Evaluate against Game Scores, 2023–2025 (chosen)** — honest end-to-end evaluation. Compounds both error sources, which matches the real-world experience of a fantasy player acting on the model's recommendations.

- **Evaluate against Game Scores, 2022–2025** — rejected because 2022 cost data is absent, making the model partially feature-blind for those rounds and confounding the evaluation.

- **Train on Game Scores for 2023+ seasons** — rejected because the volume reduction empirically increased test RMSE. Revisit if Game Score coverage extends back beyond 2022.
