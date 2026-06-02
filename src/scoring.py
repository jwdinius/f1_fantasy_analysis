import pandas as pd
import numpy as np

class F1FantasyScorer2025:
    """
    Implements Official 2025 Rules for calculating point totals.
    
    https://fantasy.formula1.com/en/game-rules
    """
    
    def __init__(self):
        # 2025 Points Map
        self.race_points = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
        self.sprint_points = {1: 8, 2: 7, 3: 6, 4: 5, 5: 4, 6: 3, 7: 2, 8: 1}
        self.quali_points = {1: 10, 2: 9, 3: 8, 4: 7, 5: 6, 6: 5, 7: 4, 8: 3, 9: 2, 10: 1}

    def calculate_driver_score(self, race_res, sprint_res, quali_res, dotd_driver_id):
        """
        Calculates individual driver value for a single Grand Prix weekend.
        Returns: (total_driver_score, constructor_contribution, is_dsq)
        """
        score = 0
        cons_contrib = 0
        is_dsq_weekend = False
        
        # --- QUALIFYING ---
        if quali_res is not None and not quali_res.empty:
            q_pos = quali_res['position_number']
            q_text = str(quali_res.get('position_text', ''))
            
            if q_text in ['DSQ', 'EX', 'NC'] or pd.isna(q_pos):
                score -= 5
                is_dsq_weekend = (q_text == 'DSQ')
            else:
                score += self.quali_points.get(int(q_pos), 0)
        
        # --- SPRINT (If applicable) ---
        if sprint_res is not None and not sprint_res.empty:
            s_pos = sprint_res['position_number']
            s_text = str(sprint_res.get('position_text', ''))
            
            if s_text in ['DSQ', 'EX'] or pd.isna(s_pos):
                score -= 20
                if s_text == 'DSQ': is_dsq_weekend = True
            else:
                # Finishing Points
                score += self.sprint_points.get(int(s_pos), 0)
                # Positions Gained (proxy for overtakes + pos gained rules)
                # F1 Fantasy 2025: 1pt per pos gained, 1pt per overtake. 
                # We use 2 * positions_gained as a proxy if positive, else just the loss.
                pg = sprint_res.get('positions_gained', 0)
                if pg > 0:
                    score += (pg * 2) 
                else:
                    score += pg # Negative points for lost positions
            
            # Sprint Fastest Lap (+5 in 2025)
            if sprint_res.get('fastest_lap', False):
                score += 5

        # --- RACE ---
        r_pos = race_res['position_number']
        r_text = str(race_res.get('position_text', ''))
        
        if r_text in ['DSQ', 'EX'] or pd.isna(r_pos):
            score -= 20
            if r_text == 'DSQ': is_dsq_weekend = True
        else:
            # Finishing Points
            score += self.race_points.get(int(r_pos), 0)
            # Positions Gained + Overtakes Proxy
            pg = race_res.get('positions_gained', 0)
            if pg > 0:
                score += (pg * 2)
            else:
                score += pg
                
        # Fastest Lap (+10 in 2025 Race)
        if race_res.get('fastest_lap', False):
            score += 10
            
        cons_contrib = score
        
        # Driver of the Day (+10) - DOES NOT count for constructor
        if race_res['driver_id'] == dotd_driver_id:
            score += 10
            
        return score, cons_contrib, is_dsq_weekend

    def calculate_constructor_score(self, driver_contrib_sum, team_quali_positions, team_pit_stops, is_fastest_lap_of_race, is_dsq_drivers):
        """
        is_fastest_lap_of_race: bool, if the team had the fastest stop of all teams
        is_dsq_drivers: list of bools, if driver 1 or 2 was DSQ'd
        """
        score = driver_contrib_sum
        
        # --- QUALIFYING BONUSES (Based on reaching Q2/Q3) ---
        valid_q = [p for p in team_quali_positions if p is not None and p > 0]
        if valid_q:
            min_q = min(valid_q)
            max_q = max(valid_q) if len(valid_q) > 1 else 99
            
            if min_q <= 10 and max_q <= 10:
                score += 10 # Both Q3
            elif min_q <= 10:
                score += 5  # One Q3
            elif min_q <= 15 and max_q <= 15:
                score += 3  # Both Q2
            elif min_q <= 15:
                score += 1  # One Q2
            else:
                score -= 1  # Neither Q2
        
        # --- PIT STOP POINTS ---
        fastest_stop = None
        if team_pit_stops:
            fastest_stop = min(team_pit_stops)
            # Scale
            if fastest_stop < 2.0: score += 20
            elif fastest_stop <= 2.19: score += 10
            elif fastest_stop <= 2.49: score += 5
            elif fastest_stop <= 2.99: score += 2
            
            # Fastest of race bonus
            if is_fastest_lap_of_race:
                score += 5
            
            # World Record bonus (sub 1.80s)
            if fastest_stop < 1.80:
                score += 15
        
        # --- DSQ PENALTIES ---
        # "Additional -5 for Quali and -10 for Sprint/Race added to constructor"
        for dsq in is_dsq_drivers:
            if dsq:
                score -= 15 # (5 + 10)
            
        return score

def main():
    from src.database import F1DataManager
    try:
        db = F1DataManager()
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    scorer = F1FantasyScorer2025()
    races = db.query("SELECT * FROM race WHERE year = 2025 ORDER BY round")
    all_scores = []
    
    for _, race in races.iterrows():
        race_id = race['id']
        print(f"Processing Round {race['round']}: {race['official_name']}...")
        
        results = db.query(f"SELECT * FROM race_result WHERE race_id = {race_id}")
        quali = db.query(f"SELECT * FROM qualifying_result WHERE race_id = {race_id}")
        sprint = db.query(f"SELECT * FROM sprint_race_result WHERE race_id = {race_id}")
        pit_stops = db.query(f"SELECT * FROM pit_stop WHERE race_id = {race_id}")
        
        dotd_row = results[results['driver_of_the_day'] == 1]
        dotd_driver_id = dotd_row['driver_id'].iloc[0] if not dotd_row.empty else None
        
        # Determine global fastest pit stop for the +5 bonus
        fastest_overall_stop = pit_stops['time_millis'].min() if not pit_stops.empty else None
        
        driver_total_scores = {}
        driver_cons_contribs = {}
        driver_dsq_status = {}
        
        for _, res in results.iterrows():
            d_id = res['driver_id']
            q_res = quali[quali['driver_id'] == d_id].iloc[0] if not quali[quali['driver_id'] == d_id].empty else None
            s_res = sprint[sprint['driver_id'] == d_id].iloc[0] if not sprint[sprint['driver_id'] == d_id].empty else None
            
            total, contrib, is_dsq = scorer.calculate_driver_score(res, s_res, q_res, dotd_driver_id)
            
            driver_total_scores[d_id] = total
            driver_cons_contribs[d_id] = contrib
            driver_dsq_status[d_id] = is_dsq
            
            all_scores.append({
                'round': race['round'], 'type': 'driver', 'id': d_id, 'score': total
            })

        # Constructors
        for c_id in results['constructor_id'].unique():
            team_res = results[results['constructor_id'] == c_id]
            team_drivers = team_res['driver_id'].tolist()
            
            contrib_sum = sum(driver_cons_contribs.get(d, 0) for d in team_drivers)
            dsq_list = [driver_dsq_status.get(d, False) for d in team_drivers]
            
            team_q_df = quali[quali['constructor_id'] == c_id]
            team_q_pos = team_q_df['position_number'].tolist()
            
            team_pits = pit_stops[pit_stops['constructor_id'] == c_id]['time_millis'].tolist()
            team_pits_sec = [t / 1000.0 for t in team_pits if not pd.isna(t)]
            
            is_fastest_team = False
            if fastest_overall_stop and team_pits:
                if min(team_pits) == fastest_overall_stop:
                    is_fastest_team = True
                    
            c_score = scorer.calculate_constructor_score(
                contrib_sum, team_q_pos, team_pits_sec, is_fastest_team, dsq_list
            )
            
            all_scores.append({
                'round': race['round'], 'type': 'constructor', 'id': c_id, 'score': c_score
            })
            
    df = pd.DataFrame(all_scores)
    print("\n" + "="*40 + "\nREFINED 2025 FANTASY STANDINGS\n" + "="*40)
    
    print("\nDrivers:")
    print(df[df['type']=='driver'].groupby('id')['score'].sum().sort_values(ascending=False))
    
    print("\nConstructors:")
    print(df[df['type']=='constructor'].groupby('id')['score'].sum().sort_values(ascending=False))

if __name__ == "__main__":
    main()