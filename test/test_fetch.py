import pandas as pd
import numpy as np
from pathlib import Path
import fastf1
from src.database import F1DataManager
import src.data_processing as dp

# Enable fastf1 cache
cache_dir = Path(".fastf1")
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

def test_single_round():
    year = 2024
    round_num = 1
    print(f"Testing FastF1 fetch for {year} R{round_num}...")
    stats = dp.fetch_fastf1_stats(year, round_num, 'R')
    if not stats.empty:
        print("Successfully fetched stats!")
        print(stats.head())
    else:
        print("Failed to fetch stats or no data found.")

if __name__ == "__main__":
    test_single_round()
