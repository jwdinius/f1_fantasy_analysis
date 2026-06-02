# Walk-Forward Validation for EV Model Training

F1 race results are strictly time-ordered and serially correlated — a driver's recent form, constructor momentum, and circuit characteristics all carry forward across rounds. Using random k-fold cross-validation would allow the model to train on future rounds and predict past ones, leaking information that would never be available at Team Lock time and producing optimistically biased accuracy estimates. We use walk-forward validation instead: for each target round, the model trains only on rounds that occurred strictly before it. This mirrors the real-world constraint exactly and produces honest out-of-sample error estimates.

## Considered Options

- **Random k-fold cross-validation** — rejected because it allows future data to appear in the training set, violating the Team Lock boundary and inflating apparent model accuracy.
- **Single train/test split** — rejected because it wastes historical data and produces a single point estimate of accuracy rather than a distribution across rounds.
