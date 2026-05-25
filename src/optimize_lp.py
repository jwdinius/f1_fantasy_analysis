import pandas as pd
import itertools
import sys

# 2025 Canonical Constructor Mapping for the 3-asset rule
DRIVER_TO_CONSTRUCTOR = {
    'lando-norris': 'mclaren',
    'oscar-piastri': 'mclaren',
    'max-verstappen': 'red-bull',
    'liam-lawson': 'red-bull',
    'george-russell': 'mercedes',
    'andrea-kimi-antonelli': 'mercedes',
    'charles-leclerc': 'ferrari',
    'lewis-hamilton': 'ferrari',
    'alexander-albon': 'williams',
    'carlos-sainz-jr': 'williams',
    'fernando-alonso': 'aston-martin',
    'lance-stroll': 'aston-martin',
    'pierre-gasly': 'alpine',
    'jack-doohan': 'alpine',
    'yuki-tsunoda': 'racing-bulls',
    'isack-hadjar': 'racing-bulls',
    'esteban-ocon': 'haas',
    'oliver-bearman': 'haas',
    'nico-hulkenberg': 'kick-sauber',
    'gabriel-bortoleto': 'kick-sauber'
}

def solve_knapsack(ev_file, budget=100.0, cost_overrides=None, top_n=1):
    df = pd.read_csv(ev_file)
    
    # Apply cost overrides if provided
    if cost_overrides:
        for asset, new_cost in cost_overrides.items():
            df.loc[df['asset_id'] == asset, 'cost'] = new_cost
            
    # CRITICAL: Filter out assets that are not competing (NaN or 0 cost)
    df = df.dropna(subset=['cost'])
    df = df[df['cost'] > 0]
    
    drivers = df[df['asset_type'] == 'driver'].to_dict('records')
    constructors = df[df['asset_type'] == 'constructor'].to_dict('records')
    
    if len(drivers) < 5 or len(constructors) < 2:
        print(f"Warning: Not enough active assets found in {ev_file} (Drivers: {len(drivers)}, Constructors: {len(constructors)})")
        return None
    
    all_teams = []
    
    # 5 drivers
    print(f"Iterating through driver combinations ({len(drivers)} active drivers)...")
    for d_comb in itertools.combinations(drivers, 5):
        d_cost = sum(d['cost'] for d in d_comb)
        if d_cost > budget:
            continue
            
        # 2 constructors
        for c_comb in itertools.combinations(constructors, 2):
            total_cost = d_cost + sum(c['cost'] for c in c_comb)
            if total_cost > budget:
                continue
            
            # 3-asset rule validation
            counts = {}
            # Drivers' teams
            for d in d_comb:
                team = DRIVER_TO_CONSTRUCTOR.get(d['asset_id'], 'unknown')
                counts[team] = counts.get(team, 0) + 1
            # Constructors count as 1 asset for their own team
            for c in c_comb:
                c_id = c['asset_id']
                counts[c_id] = counts.get(c_id, 0) + 1
            
            if any(v > 3 for v in counts.values()):
                continue
            
            total_points = sum(d['predicted_points'] for d in d_comb) + sum(c['predicted_points'] for c in c_comb)
            
            team_result = {
                'drivers': [d['asset_id'] for d in d_comb],
                'constructors': [c['asset_id'] for c in c_comb],
                'total_cost': total_cost,
                'predicted_points': total_points
            }
            all_teams.append(team_result)
                
    # Sort and return top N
    all_teams.sort(key=lambda x: x['predicted_points'], reverse=True)
    return all_teams[:top_n] if top_n > 1 else (all_teams[0] if all_teams else None)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="F1 Fantasy team optimizer")
    parser.add_argument("ev_report", help="Path to EV report CSV")
    parser.add_argument("budget", nargs='?', type=float, default=100.0,
                        help="Budget cap in millions (default: 100.0)")
    parser.add_argument("--top-n", type=int, default=1,
                        help="Number of top teams to display (default: 1)")
    args = parser.parse_args()

    results = solve_knapsack(args.ev_report, args.budget, top_n=args.top_n)

    if not results:
        print("No valid team found within budget.")
    elif isinstance(results, list):
        for i, team in enumerate(results, 1):
            print(f"\n--- Team #{i} ---")
            print(f"Budget Used: ${team['total_cost']:.1f}M / ${args.budget:.1f}M")
            print(f"Expected Points: {team['predicted_points']:.2f}")
            print(f"Drivers: {team['drivers']}")
            print(f"Constructors: {team['constructors']}")
    else:
        print("--- Optimal F1 Fantasy Team ---")
        print(f"Budget Used: ${results['total_cost']:.1f}M / ${args.budget:.1f}M")
        print(f"Expected Points: {results['predicted_points']:.2f}")
        print(f"Drivers: {results['drivers']}")
        print(f"Constructors: {results['constructors']}")
