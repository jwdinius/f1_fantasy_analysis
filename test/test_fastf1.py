import fastf1
import pandas as pd
from pathlib import Path

# Enable cache
cache_dir = Path(".fastf1")
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

def test_fastf1_data():
    year = 2024
    round_num = 1
    session = fastf1.get_session(year, round_num, 'R')
    session.load(laps=True)
    
    laps = session.laps
    if laps.empty:
        print("No laps found.")
        return

    # Filter out in/out laps or laps without time
    laps = laps.dropna(subset=['LapTime'])
    
    # Session fastest lap
    fastest_lap_session = laps['LapTime'].min()
    print(f"Fastest lap of the session: {fastest_lap_session}")

    # Driver-level stats
    driver_stats = []
    for driver in laps['Driver'].unique():
        driver_laps = laps[laps['Driver'] == driver]
        lap_times = driver_laps['LapTime'].dt.total_seconds()
        
        # Gap to fastest lap (on average or per lap)
        # Maybe average gap to the session fastest lap for all valid laps?
        gaps = lap_times - fastest_lap_session.total_seconds()
        
        driver_stats.append({
            'Driver': driver,
            'lap_time_median': lap_times.median(),
            'lap_time_std': lap_times.std(),
            'lap_time_iqr': lap_times.quantile(0.75) - lap_times.quantile(0.25),
            'lap_time_median_gap': gaps.median()
        })
    
    df_stats = pd.DataFrame(driver_stats)
    print(df_stats.head())

if __name__ == "__main__":
    test_fastf1_data()
