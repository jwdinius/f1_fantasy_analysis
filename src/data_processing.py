import pandas as pd
import numpy as np
from pathlib import Path
import fastf1
import sys
from src.database import F1DataManager
from src.scoring import F1FantasyScorer2025

# Enable fastf1 cache
cache_dir = Path(".fastf1")
cache_dir.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

def get_mapping(db):
    # Driver mapping: last_name -> database_id
    drivers = db.query("SELECT id, last_name FROM driver")
    driver_map = {row['last_name']: row['id'] for _, row in drivers.iterrows()}
    # Special cases for fantasy CSV naming
    driver_map['Sargent'] = 'logan-sargeant'
    driver_map['De Vries'] = 'nyck-de-vries'
    driver_map['Hulkenberg'] = 'nico-hulkenberg'
    driver_map['Tsunoda'] = 'yuki-tsunoda'
    driver_map['Sainz'] = 'carlos-sainz-jr'
    driver_map['Verstappen'] = 'max-verstappen'
    driver_map['Perez'] = 'sergio-perez'
    driver_map['Hamilton'] = 'lewis-hamilton'
    driver_map['Russell'] = 'george-russell'
    driver_map['Leclerc'] = 'charles-leclerc'
    driver_map['Norris'] = 'lando-norris'
    driver_map['Piastri'] = 'oscar-piastri'
    driver_map['Alonso'] = 'fernando-alonso'
    driver_map['Stroll'] = 'lance-stroll'
    driver_map['Gasly'] = 'pierre-gasly'
    driver_map['Ocon'] = 'esteban-ocon'
    driver_map['Albon'] = 'alexander-albon'
    driver_map['Bottas'] = 'valtteri-bottas'
    driver_map['Zhou'] = 'guanyu-zhou'
    driver_map['Magnussen'] = 'kevin-magnussen'
    driver_map['Ricciardo'] = 'daniel-ricciardo'
    driver_map['Lawson'] = 'liam-lawson'
    driver_map['Bearman'] = 'oliver-bearman'
    driver_map['Colapinto'] = 'franco-colapinto'
    driver_map['Doohan'] = 'jack-doohan'
    driver_map['Bortoleto'] = 'gabriel-bortoleto'
    driver_map['Antonelli'] = 'andrea-kimi-antonelli'
    driver_map['Hadjar'] = 'isack-hadjar'

    # Constructor mapping: name -> database_id
    constructors = db.query("SELECT id, name FROM constructor")
    cons_map = {row['name']: row['id'] for _, row in constructors.iterrows()}
    # Special cases for fantasy naming
    cons_map['Red Bull'] = 'red-bull'
    cons_map['Mercedes'] = 'mercedes'
    cons_map['Ferrari'] = 'ferrari'
    cons_map['McLaren'] = 'mclaren'
    cons_map['Aston Martin'] = 'aston-martin'
    cons_map['Alpine'] = 'alpine'
    cons_map['Williams'] = 'williams'
    cons_map['RB'] = 'rb'
    cons_map['Alpha Tauri'] = 'alphatauri'
    cons_map['Haas'] = 'haas'
    cons_map['Sauber'] = 'sauber'
    cons_map['Alfa Romeo'] = 'alfa-romeo'
    cons_map['Stake'] = 'sauber'

    return driver_map, cons_map

def get_fastf1_driver_mapping(session):
    """Maps FastF1 driver abbreviations to our database driver IDs."""
    # This is a bit tricky as FastF1 uses abbreviations like 'VER', 'HAM'.
    # We can try to match them using the session's driver info.
    mapping = {}
    for drv_code in session.laps['Driver'].unique():
        try:
            drv_info = session.get_driver(drv_code)
            last_name = drv_info['LastName']
            # Heuristic: match last name to our IDs
            # This is not perfect but should work for most
            # We'll use a hardcoded fallback for common mismatches
            full_name_lower = f"{drv_info['FirstName']} {drv_info['LastName']}".lower().replace(" ", "-")
            if 'verstappen' in full_name_lower: mapping[drv_code] = 'max-verstappen'
            elif 'perez' in full_name_lower: mapping[drv_code] = 'sergio-perez'
            elif 'hamilton' in full_name_lower: mapping[drv_code] = 'lewis-hamilton'
            elif 'russell' in full_name_lower: mapping[drv_code] = 'george-russell'
            elif 'leclerc' in full_name_lower: mapping[drv_code] = 'charles-leclerc'
            elif 'sainz' in full_name_lower: mapping[drv_code] = 'carlos-sainz-jr'
            elif 'norris' in full_name_lower: mapping[drv_code] = 'lando-norris'
            elif 'piastri' in full_name_lower: mapping[drv_code] = 'oscar-piastri'
            elif 'alonso' in full_name_lower: mapping[drv_code] = 'fernando-alonso'
            elif 'stroll' in full_name_lower: mapping[drv_code] = 'lance-stroll'
            elif 'gasly' in full_name_lower: mapping[drv_code] = 'pierre-gasly'
            elif 'ocon' in full_name_lower: mapping[drv_code] = 'esteban-ocon'
            elif 'albon' in full_name_lower: mapping[drv_code] = 'alexander-albon'
            elif 'sargeant' in full_name_lower: mapping[drv_code] = 'logan-sargeant'
            elif 'tsunoda' in full_name_lower: mapping[drv_code] = 'yuki-tsunoda'
            elif 'ricciardo' in full_name_lower: mapping[drv_code] = 'daniel-ricciardo'
            elif 'bottas' in full_name_lower: mapping[drv_code] = 'valtteri-bottas'
            elif 'zhou' in full_name_lower: mapping[drv_code] = 'guanyu-zhou'
            elif 'magnussen' in full_name_lower: mapping[drv_code] = 'kevin-magnussen'
            elif 'hulkenberg' in full_name_lower: mapping[drv_code] = 'nico-hulkenberg'
            elif 'lawson' in full_name_lower: mapping[drv_code] = 'liam-lawson'
            elif 'bearman' in full_name_lower: mapping[drv_code] = 'oliver-bearman'
            elif 'colapinto' in full_name_lower: mapping[drv_code] = 'franco-colapinto'
            elif 'doohan' in full_name_lower: mapping[drv_code] = 'jack-doohan'
            elif 'bortoleto' in full_name_lower: mapping[drv_code] = 'gabriel-bortoleto'
            elif 'antonelli' in full_name_lower: mapping[drv_code] = 'andrea-kimi-antonelli'
            elif 'hadjar' in full_name_lower: mapping[drv_code] = 'isack-hadjar'
            else:
                mapping[drv_code] = full_name_lower
        except:
            continue
    return mapping

def fetch_fastf1_stats(year, round_num, session_type='R'):
    """
    session_type: 'R' for Race, 'Q' for Qualifying, 'FP1', 'FP2', 'FP3'
    For preseason testing, round_num is ignored if we use get_testing_session.
    """
    try:
        if isinstance(round_num, str) and 'test' in round_num.lower():
            # Handle preseason testing (simplified: just get Day 3 of Test 1)
            session = fastf1.get_testing_session(year, 1, 3)
        else:
            session = fastf1.get_session(year, round_num, session_type)
            
        session.load(laps=True, telemetry=False, weather=False, messages=False)
        laps = session.laps
        if laps.empty: return pd.DataFrame()
        
        laps = laps.dropna(subset=['LapTime'])
        if laps.empty: return pd.DataFrame()
        
        fastest_lap_session = laps['LapTime'].min().total_seconds()
        driver_mapping = get_fastf1_driver_mapping(session)
        
        stats = []
        for drv_code in laps['Driver'].unique():
            drv_laps = laps[laps['Driver'] == drv_code]
            lap_times = drv_laps['LapTime'].dt.total_seconds()
            gaps = lap_times - fastest_lap_session
            
            suffix = f"_{session_type.lower()}" if not ('test' in str(round_num).lower()) else "_test"
            stats.append({
                'asset_id': driver_mapping.get(drv_code, drv_code),
                f'lap_time_median_gap{suffix}': gaps.median(),
                f'lap_time_std{suffix}': lap_times.std(),
                f'lap_time_iqr{suffix}': lap_times.quantile(0.75) - lap_times.quantile(0.25)
            })
        
        df_stats = pd.DataFrame(stats)
        
        # Aggregate for constructors
        cons_stats = []
        for drv_code in laps['Driver'].unique():
            try:
                drv_info = session.get_driver(drv_code)
                team_name = drv_info['TeamName']
                cons_id = team_name.lower().replace(" ", "-")
                if 'red-bull' in cons_id: cons_id = 'red-bull'
                elif 'rb' == cons_id: cons_id = 'rb'
                
                drv_id = driver_mapping.get(drv_code, drv_code)
                drv_row = df_stats[df_stats['asset_id'] == drv_id]
                if not drv_row.empty:
                    suffix = f"_{session_type.lower()}" if not ('test' in str(round_num).lower()) else "_test"
                    cons_stats.append({
                        'asset_id': cons_id,
                        f'lap_time_median_gap{suffix}': drv_row[f'lap_time_median_gap{suffix}'].iloc[0],
                        f'lap_time_std{suffix}': drv_row[f'lap_time_std{suffix}'].iloc[0],
                        f'lap_time_iqr{suffix}': drv_row[f'lap_time_iqr{suffix}'].iloc[0]
                    })
            except: continue
        
        if cons_stats:
            df_cons_stats = pd.DataFrame(cons_stats).groupby('asset_id').mean().reset_index()
            df_cons_stats['asset_type'] = 'constructor'
            df_stats['asset_type'] = 'driver'
            return pd.concat([df_stats, df_cons_stats])
        return df_stats
    except Exception as e:
        print(f"  Error fetching FastF1 {session_type} for {year} R{round_num}: {e}")
        sys.stdout.flush()
        return pd.DataFrame()

def load_costs(year, driver_map, cons_map):
    base_path = Path("data/fantasy_csv")
    d_cost_file = base_path / f"Drivers-Cost-{year}.csv"
    c_cost_file = base_path / f"Teams-Cost-{year}.csv"
    
    if not d_cost_file.exists() or not c_cost_file.exists():
        return pd.DataFrame()

    d_df = pd.read_csv(d_cost_file, index_col=0)
    c_df = pd.read_csv(c_cost_file, index_col=0)
    
    # Remove 'Average' column if exists
    if 'Average' in d_df.columns: d_df = d_df.drop(columns=['Average'])
    if 'Average' in c_df.columns: c_df = c_df.drop(columns=['Average'])
    
    # Map rows to IDs
    d_df.index = d_df.index.map(lambda x: driver_map.get(x, x))
    c_df.index = c_df.index.map(lambda x: cons_map.get(x, x))
    
    # Melt into long form
    d_melted = d_df.reset_index().melt(id_vars='index', var_name='race_name', value_name='cost')
    d_melted.rename(columns={'index': 'asset_id'}, inplace=True)
    d_melted['asset_type'] = 'driver'
    
    c_melted = c_df.reset_index().melt(id_vars='index', var_name='race_name', value_name='cost')
    c_melted.rename(columns={'index': 'asset_id'}, inplace=True)
    c_melted['asset_type'] = 'constructor'
    
    df = pd.concat([d_melted, c_melted])
    df['year'] = year
    return df

def calculate_all_scores(db, year):
    scorer = F1FantasyScorer2025()
    races = db.query(f"SELECT * FROM race WHERE year = {year} ORDER BY round")
    all_scores = []
    
    for _, race in races.iterrows():
        race_id = race['id']
        results = db.query(f"SELECT * FROM race_result WHERE race_id = {race_id}")
        if results.empty: continue
        
        quali = db.query(f"SELECT * FROM qualifying_result WHERE race_id = {race_id}")
        sprint = db.query(f"SELECT * FROM sprint_race_result WHERE race_id = {race_id}")
        pit_stops = db.query(f"SELECT * FROM pit_stop WHERE race_id = {race_id}")
        
        dotd_row = results[results['driver_of_the_day'] == 1]
        dotd_driver_id = dotd_row['driver_id'].iloc[0] if not dotd_row.empty else None
        
        fastest_overall_stop = pit_stops['time_millis'].min() if not pit_stops.empty else None
        
        driver_cons_contribs = {}
        driver_dsq_status = {}
        
        for _, res in results.iterrows():
            d_id = res['driver_id']
            q_res = quali[quali['driver_id'] == d_id].iloc[0] if not quali[quali['driver_id'] == d_id].empty else None
            s_res = sprint[sprint['driver_id'] == d_id].iloc[0] if not sprint[sprint['driver_id'] == d_id].empty else None
            
            total, contrib, is_dsq = scorer.calculate_driver_score(res, s_res, q_res, dotd_driver_id)
            
            driver_cons_contribs[d_id] = contrib
            driver_dsq_status[d_id] = is_dsq
            
            all_scores.append({
                'year': year, 'round': race['round'], 'asset_id': d_id, 'asset_type': 'driver', 'points': total
            })

        # Constructors
        for c_id in results['constructor_id'].unique():
            team_res = results[results['constructor_id'] == c_id]
            team_drivers = team_res['driver_id'].tolist()
            contrib_sum = sum(driver_cons_contribs.get(d, 0) for d in team_drivers)
            dsq_list = [driver_dsq_status.get(d, False) for d in team_drivers]
            team_q_pos = quali[quali['constructor_id'] == c_id]['position_number'].tolist()
            team_pits = pit_stops[pit_stops['constructor_id'] == c_id]['time_millis'].tolist()
            team_pits_sec = [t / 1000.0 for t in team_pits if not pd.isna(t)]
            
            is_fastest_team = False
            if fastest_overall_stop and team_pits and min(team_pits) == fastest_overall_stop:
                is_fastest_team = True
                    
            c_score = scorer.calculate_constructor_score(contrib_sum, team_q_pos, team_pits_sec, is_fastest_team, dsq_list)
            
            all_scores.append({
                'year': year, 'round': race['round'], 'asset_id': c_id, 'asset_type': 'constructor', 'points': c_score
            })
            
    return pd.DataFrame(all_scores)

def main():
    db = F1DataManager()
    d_map, c_map = get_mapping(db)
    
    all_data = []
    # Note: 2025 might not have race data yet depending on when this is run
    for year in [2023, 2024, 2025]:
        print(f"Processing {year}...")
        sys.stdout.flush()
        costs = load_costs(year, d_map, c_map)
        if costs.empty: continue
        
        points = calculate_all_scores(db, year)
        
        # Map race names to rounds for costs
        races = db.query(f"SELECT round, official_name FROM race WHERE year = {year}")
        unique_races_in_cost = costs['race_name'].unique()
        race_to_round = {name: i+1 for i, name in enumerate(unique_races_in_cost)}
        costs['round'] = costs['race_name'].map(race_to_round)
        
        merged = pd.merge(costs, points, on=['year', 'round', 'asset_id', 'asset_type'], how='left')
        
        # Preseason Testing (only once per year)
        print(f"  Fetching Preseason Testing for {year}...")
        sys.stdout.flush()
        test_stats = fetch_fastf1_stats(year, 'test')
        if not test_stats.empty:
            test_stats['year'] = year
            # Map testing to the first round for feature availability
            test_stats['round'] = 1 
            merged = pd.merge(merged, test_stats, on=['year', 'round', 'asset_id', 'asset_type'], how='left')

        # Practice & Race Stats per round
        for round_num in merged['round'].unique():
            if pd.isna(round_num): continue
            r_num = int(round_num)
            for s_type in ['FP1', 'FP2', 'FP3', 'R']:
                print(f"  Fetching FastF1 {s_type} for {year} Round {r_num}...")
                sys.stdout.flush()
                stats = fetch_fastf1_stats(year, r_num, s_type)
                if not stats.empty:
                    stats['year'] = year
                    stats['round'] = r_num
                    # Avoid duplicate columns if we re-run or merge multiple times
                    cols_to_use = [c for c in stats.columns if c not in merged.columns or c in ['year', 'round', 'asset_id', 'asset_type']]
                    merged = pd.merge(merged, stats[cols_to_use], on=['year', 'round', 'asset_id', 'asset_type'], how='left')
            
        all_data.append(merged)
        
    final_df = pd.concat(all_data)
    final_df.to_csv("data/processed_fantasy.csv", index=False)
    print("Saved to data/processed_fantasy.csv")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
