import pandas as pd
import numpy as np
import sys
from src.model import F1FantasyPredictor
from pathlib import Path

# get this file's path
script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "data"

def generate_ev_report(year, round_num):
    d_path = data_dir / "processed_fantasy_drivers.csv"
    c_path = data_dir / "processed_fantasy_constructors.csv"
    
    if not d_path.exists() or not c_path.exists():
        print("Driver or Constructor data files missing.")
        return None

    d_df = pd.read_csv(d_path)
    c_df = pd.read_csv(c_path)
    
    # 1. Predict Drivers
    d_train = d_df[(d_df['year'] < year) | ((d_df['year'] == year) & (d_df['round'] < round_num))]
    d_predictor = F1FantasyPredictor(asset_type='driver')
    if not d_train.empty:
        d_predictor.train(d_train)
    d_preds = d_predictor.predict_next(d_df, year, round_num)
    
    # 2. Predict Constructors
    c_train = c_df[(c_df['year'] < year) | ((c_df['year'] == year) & (c_df['round'] < round_num))].copy()
    
    # Add ACTUAL sum of driver points as proxy for 'predicted_sum' during training
    # This matches the user's request to use predicted sums during prediction.
    if 'drivers_predicted_points_sum' not in c_train.columns:
        actual_sums = (
            d_df.groupby(['year', 'round', 'constructor_id'])['fantasy_points']
            .sum().reset_index()
            .rename(columns={'fantasy_points': 'drivers_predicted_points_sum'})
        )
        c_train = pd.merge(c_train, actual_sums, on=['year', 'round', 'constructor_id'], how='left')

    c_predictor = F1FantasyPredictor(asset_type='constructor')
    if not c_train.empty:
        c_predictor.train(c_train)
        
    # For the target prediction, use the SUM of the DRIVER PREDICTIONS we just made
    driver_sums = d_preds.groupby('constructor_id')['predicted_points'].sum().to_dict()
    c_preds = c_predictor.predict_next(c_df, year, round_num, drivers_predicted_sum_map=driver_sums)
    
    ev_df = pd.concat([d_preds, c_preds]).sort_values('predicted_points', ascending=False)
    
    ev_file = data_dir / "ev_reports" / f"ev_report_{year}_R{round_num}.csv"
    ev_df.to_csv(ev_file, index=False)
    print(f"EV Report saved to {ev_file}")
    
    print("\n--- Top Drivers (Expected Points) ---")
    drivers = ev_df[ev_df['asset_type'] == 'driver']
    print(drivers[['asset_id', 'predicted_points', 'cost']].to_string(index=False))
    
    print("\n--- Top Constructors (Expected Points) ---")
    constructors = ev_df[ev_df['asset_type'] == 'constructor']
    print(constructors[['asset_id', 'predicted_points', 'cost']].to_string(index=False))
    
    return ev_df

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/predict_ev.py <year> <round>")
    else:
        generate_ev_report(int(sys.argv[1]), int(sys.argv[2]))
