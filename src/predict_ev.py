import pandas as pd
import numpy as np
import sys
from src.model import F1FantasyPredictor
from pathlib import Path

# get this file's path
script_dir = Path(__file__).resolve().parent
data_dir = script_dir.parent / "data"

def generate_ev_report(year, round_num):
    df = pd.read_csv(f"{data_dir}/processed_fantasy.csv")
    predictor = F1FantasyPredictor(n_lags=3)
    
    # Train on all data PRIOR to the target year/round
    train_data = df[
        (df['year'] < year) | 
        ((df['year'] == year) & (df['round'] < round_num))
    ]
    
    if train_data.empty:
        print(f"No training data available before {year} R{round_num}")
        return
        
    predictor.train(train_data)
    
    # Predict for the current round
    ev_df = predictor.predict_next(df, year, round_num)
    ev_df = ev_df.sort_values('predicted_points', ascending=False)
    
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
