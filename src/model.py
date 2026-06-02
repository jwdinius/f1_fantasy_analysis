import pandas as pd
import numpy as np
import xgboost as xgb
import itertools

# 2026 Canonical Constructor Mapping
DRIVER_TO_CONSTRUCTOR = {
    'lando-norris':       'mclaren',
    'oscar-piastri':      'mclaren',
    'max-verstappen':     'red-bull',
    'liam-lawson':        'red-bull',
    'george-russell':     'mercedes',
    'kimi-antonelli':     'mercedes',
    'charles-leclerc':    'ferrari',
    'lewis-hamilton':     'ferrari',
    'alexander-albon':    'williams',
    'carlos-sainz-jr':    'williams',
    'fernando-alonso':    'aston-martin',
    'lance-stroll':       'aston-martin',
    'pierre-gasly':       'alpine',
    'franco-colapinto':   'alpine',
    'yuki-tsunoda':       'racing-bulls',
    'isack-hadjar':       'racing-bulls',
    'arvid-lindblad':     'racing-bulls',
    'esteban-ocon':       'haas',
    'oliver-bearman':     'haas',
    'nico-hulkenberg':    'audi',
    'gabriel-bortoleto':  'audi',
    'sergio-perez':       'cadillac',
    'valtteri-bottas':    'cadillac',
}

_DRIVER_CURRENT_ROUND_FEATURES = [
    'fp1_gap_sec', 'fp2_gap_sec', 'fp3_gap_sec',
    'lap_time_median_gap_fp1', 'lap_time_iqr_fp1',
    'lap_time_median_gap_fp2', 'lap_time_iqr_fp2',
    'lap_time_median_gap_fp3', 'lap_time_iqr_fp3',
    'cost',
]

_CONSTRUCTOR_CURRENT_ROUND_FEATURES = [
    'drivers_predicted_points_sum',
    'cost',
]


class F1FantasyPredictor:
    def __init__(self, asset_type='driver'):
        self.asset_type = asset_type
        self.model = xgb.XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=5,
            random_state=42,
        )
        self.is_trained = False
        self.feature_cols = []
        self.evals_result_ = {}

    def prepare_features(self, df):
        rolling_cols = [c for c in df.columns if '_rolling_' in c]
        if self.asset_type == 'driver':
            base_cols = [c for c in _DRIVER_CURRENT_ROUND_FEATURES if c in df.columns]
        else:
            base_cols = [c for c in _CONSTRUCTOR_CURRENT_ROUND_FEATURES if c in df.columns]
        self.feature_cols = base_cols + rolling_cols
        if not self.feature_cols:
            print(f"WARNING: No features found for {self.asset_type}! Columns: {df.columns.tolist()}")
        return df

    def train(self, df, val_df=None, sample_weight=None):
        df = self.prepare_features(df)
        train_df = df.dropna(subset=['fantasy_points'])
        if train_df.empty or not self.feature_cols:
            print(f"WARNING: Nothing to train for {self.asset_type}.")
            return
        X = train_df[self.feature_cols].fillna(-1)
        y = train_df['fantasy_points']

        # Align sample weights to the rows that survived dropna
        sw = None
        if sample_weight is not None:
            sw = sample_weight[train_df.index]

        if val_df is not None:
            val_df = self.prepare_features(val_df)
            val_clean = val_df.dropna(subset=['fantasy_points'])
            X_val = val_clean[self.feature_cols].fillna(-1)
            y_val = val_clean['fantasy_points']
            # XGBoost 3.x requires early_stopping_rounds in the constructor,
            # so swap in a fresh instance for this path only.
            self.model = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=3,
                min_child_weight=5,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=5,
                random_state=42,
                early_stopping_rounds=30,
            )
            self.model.fit(
                X, y,
                sample_weight=sw,
                eval_set=[(X, y), (X_val, y_val)],
                verbose=False,
            )
            self.evals_result_ = self.model.evals_result_
        else:
            self.model.fit(X, y, sample_weight=sw)

        self.is_trained = True
        best = getattr(self.model, 'best_iteration', None)
        suffix = f", best round {best}" if best is not None else ""
        print(f"Model ({self.asset_type}) trained on {len(X)} rows, "
              f"{len(self.feature_cols)} features{suffix}.")

    def save(self, path):
        import joblib
        joblib.dump(self, path)
        print(f"Saved {self.asset_type} predictor → {path}")

    @classmethod
    def load(cls, path):
        import joblib
        return joblib.load(path)

    def predict_next(self, df, year, round_num, drivers_predicted_sum_map=None,
                     cost_df: pd.DataFrame | None = None):
        current_round = df[(df['year'] == year) & (df['round'] == round_num)].copy()
        if current_round.empty:
            # No feature row for this round yet (e.g. pre-practice baseline).
            # Use the most recent prior round's rolling features as a proxy and
            # blank out all practice-session columns (fp*_gap_sec, lap_time_*).
            prior = df[
                (df['year'] < year) | ((df['year'] == year) & (df['round'] < round_num))
            ]
            if prior.empty:
                return pd.DataFrame()
            latest_year  = prior['year'].max()
            latest_round = prior[prior['year'] == latest_year]['round'].max()
            current_round = prior[
                (prior['year'] == latest_year) & (prior['round'] == latest_round)
            ].copy()
            current_round['year']  = year
            current_round['round'] = round_num
            fp_cols = [c for c in current_round.columns
                       if any(tag in c for tag in ('fp1', 'fp2', 'fp3', 'gap_sec', 'lap_time'))]
            current_round[fp_cols] = np.nan
            # Overwrite costs with the caller-supplied cost_df when available
            if cost_df is not None:
                id_col = 'driver_id' if self.asset_type == 'driver' else 'constructor_id'
                cost_map = cost_df.set_index(id_col)['cost'].to_dict()
                current_round['cost'] = current_round[id_col].map(cost_map)
            print(f"  ({self.asset_type}) No feature row for {year} R{round_num} — "
                  f"using {latest_year} R{latest_round} rolling features as baseline.")
        if self.asset_type == 'constructor' and drivers_predicted_sum_map is not None:
            current_round['drivers_predicted_points_sum'] = (
                current_round['constructor_id'].map(drivers_predicted_sum_map).fillna(0)
            )
        # feature_cols are locked in by train(); only derive here if never trained
        if not self.feature_cols:
            self.prepare_features(current_round)
        X_curr = current_round[self.feature_cols].fillna(-1)
        preds = self.model.predict(X_curr) if self.is_trained else np.zeros(len(X_curr))

        if self.asset_type == 'driver':
            results = current_round[['driver_id', 'constructor_id', 'cost']].copy()
            results = results.rename(columns={'driver_id': 'asset_id'})
            results['asset_type'] = 'driver'
        else:
            results = current_round[['constructor_id', 'cost']].copy()
            results = results.rename(columns={'constructor_id': 'asset_id'})
            results['asset_type'] = 'constructor'
            results['constructor_id'] = results['asset_id']

        results['predicted_points'] = preds
        return results


def optimize_team(predictions, budget=100.0, top_n=1):
    drivers = predictions[predictions['asset_type'] == 'driver'].to_dict('records')
    constructors = predictions[predictions['asset_type'] == 'constructor'].to_dict('records')
    drivers = [d for d in drivers if not pd.isna(d['cost']) and d['cost'] > 0]
    constructors = [c for c in constructors if not pd.isna(c['cost']) and c['cost'] > 0]
    if len(drivers) < 5 or len(constructors) < 2:
        return None
    all_teams = []
    for d_comb in itertools.combinations(drivers, 5):
        d_cost = sum(d['cost'] for d in d_comb)
        if d_cost > budget:
            continue
        for c_comb in itertools.combinations(constructors, 2):
            total_cost = d_cost + sum(c['cost'] for c in c_comb)
            if total_cost <= budget:
                counts = {}
                for d in d_comb:
                    cons = DRIVER_TO_CONSTRUCTOR.get(d['asset_id'], 'unknown')
                    counts[cons] = counts.get(cons, 0) + 1
                for c in c_comb:
                    counts[c['asset_id']] = counts.get(c['asset_id'], 0) + 1
                if any(v > 3 for v in counts.values()):
                    continue
                total_points = (
                    sum(d['predicted_points'] for d in d_comb)
                    + sum(c['predicted_points'] for c in c_comb)
                )
                all_teams.append({
                    'drivers': [d['asset_id'] for d in d_comb],
                    'constructors': [c['asset_id'] for c in c_comb],
                    'total_cost': total_cost,
                    'predicted_points': total_points,
                })
    all_teams.sort(key=lambda x: x['predicted_points'], reverse=True)
    if not all_teams:
        return None
    return all_teams[:top_n] if top_n > 1 else all_teams[0]


if __name__ == "__main__":
    d_df = pd.read_csv("data/processed_fantasy_drivers.csv")
    c_df = pd.read_csv("data/processed_fantasy_constructors.csv")
    d_predictor = F1FantasyPredictor(asset_type='driver')
    d_predictor.train(d_df)
    d_preds = d_predictor.predict_next(d_df, 2025, 1)
    driver_sums = d_preds.groupby('constructor_id')['predicted_points'].sum().to_dict()
    c_predictor = F1FantasyPredictor(asset_type='constructor')
    actual_sums = (
        d_df.groupby(['year', 'round', 'constructor_id'])['fantasy_points']
        .sum().reset_index()
        .rename(columns={'fantasy_points': 'drivers_predicted_points_sum'})
    )
    c_df = pd.merge(c_df, actual_sums, on=['year', 'round', 'constructor_id'], how='left')
    c_predictor.train(c_df)
    c_preds = c_predictor.predict_next(c_df, 2025, 1, drivers_predicted_sum_map=driver_sums)
    all_preds = pd.concat([d_preds, c_preds])
    team = optimize_team(all_preds)
    if team:
        print(f"\nOptimal Team (Budget: ${team['total_cost']:.1f}M, Expected Points: {team['predicted_points']:.1f})")
        print("Drivers:", team['drivers'])
        print("Constructors:", team['constructors'])
