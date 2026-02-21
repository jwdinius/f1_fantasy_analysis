import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.model import F1FantasyPredictor
import os

def evaluate_2025_season():
    df = pd.read_csv("data/processed_fantasy.csv")
    
    results = []
    rounds = sorted(df[df['year'] == 2025]['round'].unique())
    
    print(f"Starting out-of-sample evaluation for 2025 ({len(rounds)} rounds)...")
    
    for r_num in rounds:
        print(f"  Processing Round {r_num}...")
        predictor = F1FantasyPredictor(n_lags=3)
        
        # Training set: All data before this specific round
        train_data = df[
            (df['year'] < 2025) | 
            ((df['year'] == 2025) & (df['round'] < r_num))
        ]
        
        if train_data.empty:
            continue
            
        # Train the model
        predictor.train(train_data)
        
        # Predict for the current round
        preds = predictor.predict_next(df, 2025, r_num)
        
        # Join with actual results
        actuals = df[(df['year'] == 2025) & (df['round'] == r_num)][['asset_id', 'points']]
        merged = pd.merge(preds, actuals, on='asset_id', how='inner')
        merged['round'] = r_num
        merged['residual'] = merged['predicted_points'] - merged['points']
        results.append(merged)
        
    print("Evaluation complete.")
    all_results = pd.concat(results)
    
    # Filter for drivers only
    driver_results = all_results[all_results['asset_type'] == 'driver']
    
    # Save results for reproducibility
    all_results.to_csv("data/eval_2025_results.csv", index=False)
    
    # Plotting Residuals
    plt.figure(figsize=(15, 10))
    drivers = sorted(driver_results['asset_id'].unique())
    
    # Assign a color to each driver
    cmap = plt.get_cmap('tab20')
    colors = {driver: cmap(i % 20) for i, driver in enumerate(drivers)}
    
    for driver in drivers:
        driver_data = driver_results[driver_results['asset_id'] == driver].sort_values('round')
        color = colors[driver]
        
        # Plot Residuals (difference)
        plt.plot(driver_data['round'], driver_data['residual'], 'o-', 
                 color=color, alpha=0.7, markersize=6, label=driver)

    # Reference line at 0 (Perfect prediction)
    plt.axhline(0, color='black', linestyle='-', linewidth=2, alpha=0.8)
    
    plt.title("F1 Fantasy 2025: Prediction Residuals (Predicted - Actual)")
    plt.xlabel("Round")
    plt.ylabel("Residual (Points)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Create legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize='small')
    
    plt.tight_layout()
    plt.savefig("data/eval_2025_residuals.png")
    print("Residual plot saved to data/eval_2025_residuals.png")
    
    # Summary Metrics
    mean_error = np.mean(driver_results['residual'])
    rmse = np.sqrt(np.mean(driver_results['residual']**2))
    print(f"Overall 2025 Driver Mean Error (Bias): {mean_error:.2f}")
    print(f"Overall 2025 Driver RMSE: {rmse:.2f}")

if __name__ == "__main__":
    evaluate_2025_season()
