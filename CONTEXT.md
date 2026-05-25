# F1 Fantasy Analysis

A system for predicting F1 Fantasy points per asset and selecting the optimal team within game constraints.

## Language

**Round** (also: Race):
A single race weekend within a season, identified by its sequential number (e.g., Round 7 of 2025). The atomic unit of team selection — one EV prediction and one optimal team are produced per round. Some rounds include a Sprint race, which adds additional scoring opportunities but does not make the weekend a separate round.
_Avoid_: Event, Grand Prix

**Expected Value (EV)**:
The model's predicted fantasy points for an asset in a given race round. Computed using only data available at team lock (after FP3, before qualifying): current-round practice session gaps plus historical lags of race performance, qualifying performance, positions gained, and Driver of the Day vote percentages. The fantasy score being predicted accounts for finishing position, positions gained (overtakes), Driver of the Day bonus, fastest lap, and qualifying order.
_Avoid_: Score, prediction, forecast

**Practice Gap**:
The delta between a Driver's best lap time in a given practice session (FP1, FP2, or FP3) and the session's overall fastest lap. A smaller gap indicates stronger pace. Used as the primary current-round feature for EV prediction.
_Avoid_: Lap delta, practice time, session gap

**Team Lock**:
The point in a race weekend after which a fantasy Team selection is frozen for that round. Occurs after FP3 and before qualifying. Defines the hard boundary on what data may be used for EV prediction — only data from FP3 and earlier is valid input.
_Avoid_: Deadline, cutoff, lock-in

**Constructor Rebranding**:
The canonicalization of a Constructor's identity across seasons when its commercial name changes but the physical team is continuous. Historical data for rebranded Constructors is preserved under the canonical (current) name so the EV model can use it as training data.
_Avoid_: Team rename, alias, name mapping

**Budget Cap**:
The maximum total cost allowed for a fantasy Team in a given round. The official F1 Fantasy game cap is $100M; the optimizer accepts this as a configurable parameter defaulting to $100M. All asset costs are denominated in millions.
_Avoid_: Salary cap, cost limit, budget

**Walk-Forward Validation**:
The training strategy used by the EV model. For each target round, the model trains exclusively on rounds that occurred before it — across all prior seasons and earlier rounds in the current season. This prevents data leakage and mirrors the real-world constraint that future results are unknown at team lock time.
_Avoid_: Cross-validation, backtesting, time-series split

**Driver of the Day (DotD)**:
A post-race fan vote awarding +10 fantasy points to the winning Driver. A Driver's historical DotD vote percentage (rolling 5-race window) is used as a predictive feature in the EV model. DotD points are excluded from Constructor scoring.
_Avoid_: Fan vote, popularity bonus

**Team**:
The fantasy selection for a given round — exactly 5 Drivers and 2 Constructors, subject to a total cost at or under the budget cap. Distinct from a Constructor, which is an F1 entity.
_Avoid_: Roster, lineup, squad

**Constructor**:
A Formula 1 team, selectable as a single asset in the F1 Fantasy game. Each Constructor has exactly 2 Driver assets and 1 Constructor asset affiliated with it — a total of 3 selectable assets. A fantasy team may include any combination of these, up to all 3.
_Avoid_: Team (ambiguous — prefer Constructor for the F1 entity, Team for the fantasy selection)

**Game Score**:
The fantasy points officially awarded by the F1 Fantasy platform for a given asset in a given round. Computed by the platform's proprietary formula and reflects steward decisions, eligibility edge cases, and post-race adjustments. Available for 2022 onwards from weekly draw records. The ground truth for evaluating real-world prediction performance.
_Avoid_: Actual score, real score, true score

**Proxy Score**:
The fantasy points computed by `F1FantasyScorer2025` applied uniformly to f1db event data. A correlated estimate of the Game Score using a publicly documented scoring formula. Used as the training target for the EV model because it can be reconstructed consistently back to 2015, whereas Game Scores are only available from 2022. Not identical to the Game Score — differences arise from steward decisions, platform rounding, and rule interpretations not captured in the formula.
_Avoid_: Actual score, computed score, synthetic score

**Training Window**:
The N most recent prior rounds whose performance data are included as features for a given EV prediction. Currently set to N=3 rounds. Captures recent form for each asset without over-weighting distant history.
_Avoid_: Lag features, lookback, lag window

**Asset**:
A selectable unit in the F1 Fantasy game — either a Driver or a Constructor. Each asset type has its own scoring schema: Driver scores are based on race/qualifying/sprint performance; Constructor scores are derived partly from their drivers' combined points (excluding Driver of the Day) and partly from team-level metrics (qualifying performance and pit stop performance).
_Avoid_: Player, participant, entity, unit
