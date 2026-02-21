import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import itertools

# 2025 Driver-Constructor Mapping
DRIVER_TO_CONSTRUCTOR = {
    'lando-norris': 'mclaren',
    'max-verstappen': 'red-bull',
    'george-russell': 'mercedes',
    'andrea-kimi-antonelli': 'mercedes',
    'alexander-albon': 'williams',
    'lance-stroll': 'aston-martin',
    'nico-hulkenberg': 'sauber',
    'charles-leclerc': 'ferrari',
    'oscar-piastri': 'mclaren',
    'lewis-hamilton': 'ferrari',
    'pierre-gasly': 'alpine',
    'yuki-tsunoda': 'rb',
    'esteban-ocon': 'haas',
    'oliver-bearman': 'haas',
    'liam-lawson': 'red-bull',
    'gabriel-bortoleto': 'sauber',
    'fernando-alonso': 'aston-martin',
    'carlos-sainz-jr': 'williams',
    'jack-doohan': 'alpine',
    'isack-hadjar': 'rb'
}

class F1FantasyPredictor:
    def __init__(self, n_lags=3):
        self.n_lags = n_lags
        self.model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        self.is_trained = False
        self.feature_cols = []

    def prepare_features(self, df):
        df = df.sort_values(['asset_id', 'year', 'round'])
        
        # Lagged features (historical performance)
        for i in range(1, self.n_lags + 1):
            df[f'points_lag_{i}'] = df.groupby('asset_id')['points'].shift(i)
        
        # Current round features (Practice & Testing)
        # These are available BEFORE the race, so they are not lagged
        # lap_time_median_gap_fp1, lap_time_median_gap_fp2, lap_time_median_gap_fp3, lap_time_median_gap_test
        current_round_features = [
            'lap_time_median_gap_fp1', 'lap_time_std_fp1',
            'lap_time_median_gap_fp2', 'lap_time_std_fp2',
            'lap_time_median_gap_fp3', 'lap_time_std_fp3',
            'lap_time_median_gap_test', 'lap_time_std_test'
        ]
        
        # Also include historical race pace (lagged)
        for i in range(1, self.n_lags + 1):
            if 'lap_time_median_gap_r' in df.columns:
                df[f'race_gap_lag_{i}'] = df.groupby('asset_id')['lap_time_median_gap_r'].shift(i)
                df[f'race_std_lag_{i}'] = df.groupby('asset_id')['lap_time_std_r'].shift(i)
        
        df['is_driver'] = (df['asset_type'] == 'driver').astype(int)
        return df

    def train(self, df):
        featured_df = self.prepare_features(df)
        
        lag_cols = [f'points_lag_{i}' for i in range(1, self.n_lags + 1)]
        current_cols = [
            'lap_time_median_gap_fp1', 'lap_time_std_fp1',
            'lap_time_median_gap_fp2', 'lap_time_std_fp2',
            'lap_time_median_gap_fp3', 'lap_time_std_fp3',
            'lap_time_median_gap_test', 'lap_time_std_test'
        ]
        # Filter current_cols to only those that exist in df
        current_cols = [c for c in current_cols if c in featured_df.columns]
        
        race_lag_cols = []
        for i in range(1, self.n_lags + 1):
            if f'race_gap_lag_{i}' in featured_df.columns:
                race_lag_cols.append(f'race_gap_lag_{i}')
                race_lag_cols.append(f'race_std_lag_{i}')
            
        self.feature_cols = lag_cols + current_cols + race_lag_cols + ['cost', 'is_driver']
        
        # We need at least points to train
        train_df = featured_df.dropna(subset=['points'])
        # Fill NaNs in features with a neutral value (e.g., 0 or mean)
        # For XGBoost, it handles NaNs, but sometimes explicit filling is better
        X = train_df[self.feature_cols].fillna(0)
        y = train_df['points']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        self.is_trained = True
        preds = self.model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        print(f"Model trained with XGBoost. Features: {len(self.feature_cols)}. Test RMSE: {np.sqrt(mse):.2f}")
        
    def predict_next(self, df, year, round_num):
        featured_df = self.prepare_features(df)
        assets = featured_df['asset_id'].unique()
        predictions = []
        
        for asset in assets:
            asset_data = featured_df[featured_df['asset_id'] == asset].sort_values(['year', 'round'])
            current_race = asset_data[(asset_data['year'] == year) & (asset_data['round'] == round_num)]
            if current_race.empty: continue
            
            # Use pre-prepared features from featured_df for the specific row
            X_curr = current_race[self.feature_cols].fillna(0)
            pred_points = self.model.predict(X_curr)[0]
            
            predictions.append({
                'asset_id': asset, 
                'asset_type': current_race['asset_type'].iloc[0], 
                'predicted_points': pred_points, 
                'cost': current_race['cost'].iloc[0]
            })
            
        return pd.DataFrame(predictions)

def optimize_team(predictions, budget=100.0):
    drivers = predictions[predictions['asset_type'] == 'driver'].to_dict('records')
    constructors = predictions[predictions['asset_type'] == 'constructor'].to_dict('records')
    
    best_team = None
    best_score = -1
    
    for d_comb in itertools.combinations(drivers, 5):
        d_cost = sum(d['cost'] for d in d_comb)
        if d_cost > budget: continue
        
        for c_comb in itertools.combinations(constructors, 2):
            total_cost = d_cost + sum(c['cost'] for c in c_comb)
            if total_cost <= budget:
                # 3-asset rule: check each constructor count
                counts = {}
                # Drivers
                for d in d_comb:
                    cons = DRIVER_TO_CONSTRUCTOR.get(d['asset_id'], 'unknown')
                    counts[cons] = counts.get(cons, 0) + 1
                # Constructors themselves count as 1 asset for that team
                for c in c_comb:
                    counts[c['asset_id']] = counts.get(c['asset_id'], 0) + 1
                
                if any(v > 3 for v in counts.values()):
                    continue
                
                total_points = sum(d['predicted_points'] for d in d_comb) + sum(c['predicted_points'] for c in c_comb)
                if total_points > best_score:
                    best_score = total_points
                    best_team = {'drivers': d_comb, 'constructors': c_comb, 'total_cost': total_cost, 'predicted_points': total_points}
            
    return best_team

if __name__ == "__main__":
    df = pd.read_csv("data/processed_fantasy.csv")
    print("\n--- Final Recommendation (N=3) ---")
    predictor = F1FantasyPredictor(n_lags=3)
    predictor.train(df)
    preds = predictor.predict_next(df, 2025, 1)
    team = optimize_team(preds)
    if team:
        print(f"Optimal Team (Budget: ${team['total_cost']:.1f}M, Predicted Points: {team['predicted_points']:.1f}):")
        print("Drivers:", [d['asset_id'] for d in team['drivers']])
        print("Constructors:", [c['asset_id'] for c in team['constructors']])
