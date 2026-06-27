import pandas as pd
import itertools
import sys

# 2026 Canonical Constructor Mapping for the 3-asset rule
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


def _load_ev_data(ev_file, cost_overrides=None):
    df = pd.read_csv(ev_file)
    if cost_overrides:
        for asset, new_cost in cost_overrides.items():
            df.loc[df['asset_id'] == asset, 'cost'] = new_cost
    df = df.dropna(subset=['cost'])
    df = df[df['cost'] > 0]
    drivers = df[df['asset_type'] == 'driver'].to_dict('records')
    constructors = df[df['asset_type'] == 'constructor'].to_dict('records')
    return drivers, constructors


def _enumerate_teams(drivers, constructors, budget):
    all_teams = []
    for d_comb in itertools.combinations(drivers, 5):
        d_cost = sum(d['cost'] for d in d_comb)
        if d_cost > budget:
            continue
        for c_comb in itertools.combinations(constructors, 2):
            total_cost = d_cost + sum(c['cost'] for c in c_comb)
            if total_cost > budget:
                continue

            counts = {}
            for d in d_comb:
                team = DRIVER_TO_CONSTRUCTOR.get(d['asset_id'], 'unknown')
                counts[team] = counts.get(team, 0) + 1
            for c in c_comb:
                counts[c['asset_id']] = counts.get(c['asset_id'], 0) + 1
            if any(v > 3 for v in counts.values()):
                continue

            total_points = sum(d['predicted_points'] for d in d_comb) + sum(c['predicted_points'] for c in c_comb)
            all_teams.append({
                'drivers': [d['asset_id'] for d in d_comb],
                'driver_evs': [d['predicted_points'] for d in d_comb],
                'constructors': [c['asset_id'] for c in c_comb],
                'total_cost': total_cost,
                'predicted_points': total_points,
            })

    all_teams.sort(key=lambda x: x['predicted_points'], reverse=True)
    return all_teams


def solve_knapsack(ev_file, budget=100.0, cost_overrides=None, top_n=1):
    drivers, constructors = _load_ev_data(ev_file, cost_overrides)
    if len(drivers) < 5 or len(constructors) < 2:
        print(f"Warning: Not enough active assets found in {ev_file} (Drivers: {len(drivers)}, Constructors: {len(constructors)})")
        return None

    print(f"Iterating through driver combinations ({len(drivers)} active drivers)...")
    teams = _enumerate_teams(drivers, constructors, budget)
    if not teams:
        return None

    return teams[:top_n] if top_n > 1 else teams[0]


def _same_roster(t1, t2):
    return (set(t1['drivers']) == set(t2['drivers'])
            and set(t1['constructors']) == set(t2['constructors']))


def select_hedge_slate(ev_file, budget=100.0, cost_overrides=None):
    """Return a structured hedge slate against single-driver tail risk.

    The slate hedges the top two highest-EV drivers in the headline team:
      - Headline:    max-EV team, no exclusion.
      - Hedge-D1:    max-EV team that excludes D1 (highest-EV driver in Headline).
      - Hedge-D2:    max-EV team that excludes D2 (second-highest-EV driver in Headline).

    If Hedge-D1 and Hedge-D2 land on the same roster (common when D1 and D2 are
    teammates and the 3-asset cascade forces both swaps together), they're
    merged into a single Hedge-both row labelled with both excluded anchors.

    Returns a list of (label, excluded_anchor_str_or_None, team_dict_or_None).
    """
    drivers, constructors = _load_ev_data(ev_file, cost_overrides)
    if len(drivers) < 5 or len(constructors) < 2:
        print(f"Warning: Not enough active assets found in {ev_file} (Drivers: {len(drivers)}, Constructors: {len(constructors)})")
        return []

    print(f"Iterating through driver combinations ({len(drivers)} active drivers)...")
    teams = _enumerate_teams(drivers, constructors, budget)
    if not teams:
        return []

    headline = teams[0]

    drivers_by_ev = sorted(
        zip(headline['drivers'], headline['driver_evs']),
        key=lambda x: x[1],
        reverse=True,
    )
    if len(drivers_by_ev) < 2:
        return [('Headline', None, headline)]

    d1_id, _ = drivers_by_ev[0]
    d2_id, _ = drivers_by_ev[1]

    hedge_d1 = next((t for t in teams if d1_id not in t['drivers']), None)
    hedge_d2 = next((t for t in teams if d2_id not in t['drivers']), None)

    slate = [('Headline', None, headline)]

    if (hedge_d1 is not None and hedge_d2 is not None
            and _same_roster(hedge_d1, hedge_d2)):
        slate.append(('Hedge-both', f'{d1_id} + {d2_id}', hedge_d1))
    else:
        slate.append(('Hedge-D1', d1_id, hedge_d1))
        slate.append(('Hedge-D2', d2_id, hedge_d2))

    return slate


def _print_team(team, budget):
    print(f"Budget Used: ${team['total_cost']:.1f}M / ${budget:.1f}M")
    print(f"Expected Points: {team['predicted_points']:.2f}")
    print(f"Drivers: {team['drivers']}")
    print(f"Constructors: {team['constructors']}")


def _print_slate(slate, budget):
    for label, excluded, team in slate:
        print(f"\n--- {label} ---")
        if excluded is not None:
            print(f"Excludes anchor(s): {excluded}")
        if team is None:
            print("(no feasible team for this exclusion)")
            continue
        _print_team(team, budget)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="F1 Fantasy team optimizer")
    parser.add_argument("ev_report", help="Path to EV report CSV")
    parser.add_argument("budget", nargs='?', type=float, default=100.0,
                        help="Budget cap in millions (default: 100.0)")
    parser.add_argument("--hedge", choices=['anchor', 'none'], default='anchor',
                        help="Output mode. 'anchor' (default): 2-3 row slate "
                             "hedging the top-2 EV drivers in the headline team. "
                             "'none': flat top-N ranking by EV.")
    parser.add_argument("--top-n", type=int, default=1,
                        help="(--hedge none only) Number of top teams to display.")
    args = parser.parse_args()

    if args.hedge == 'anchor':
        if args.top_n != 1:
            print("Note: --top-n is ignored in --hedge anchor mode.", file=sys.stderr)
        slate = select_hedge_slate(args.ev_report, args.budget)
        if not slate:
            print("No valid team found within budget.")
        else:
            _print_slate(slate, args.budget)
    else:
        results = solve_knapsack(args.ev_report, args.budget, top_n=args.top_n)
        if not results:
            print("No valid team found within budget.")
        elif isinstance(results, list):
            for i, team in enumerate(results, 1):
                print(f"\n--- Team #{i} ---")
                _print_team(team, args.budget)
        else:
            print("--- Optimal F1 Fantasy Team ---")
            _print_team(results, args.budget)
