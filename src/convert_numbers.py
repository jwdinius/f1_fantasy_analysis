"""
Convert data/numbers/*.numbers to long-format CSVs.

Outputs:
  data/actual_fantasy_drivers.csv
  data/actual_fantasy_constructors.csv

Columns: year, round, driver_id / constructor_id, actual_points, cost
Round numbers are derived from column position (1-indexed), not circuit name.
2022 data is included but cost/points are largely absent for that season.
"""

from pathlib import Path

import numbers_parser
import pandas as pd

DATA_DIR    = Path("data")
NUMBERS_DIR = DATA_DIR / "numbers"

DRIVER_NAME_TO_ID = {
    'Albon':        'alexander-albon',
    'Alonso':       'fernando-alonso',
    'Antonelli':    'kimi-antonelli',
    'Bearman':      'oliver-bearman',
    'Bortoleto':    'gabriel-bortoleto',
    'Bottas':       'valtteri-bottas',
    'Colapinto':    'franco-colapinto',
    'Doohan':       'jack-doohan',
    'Gasly':        'pierre-gasly',
    'Hadjar':       'isack-hadjar',
    'Hamilton':     'lewis-hamilton',
    'Hulkenberg':   'nico-hulkenberg',
    'Latifi':       'nicholas-latifi',
    'Lawson':       'liam-lawson',
    'Leclerc':      'charles-leclerc',
    'Lindblad':     'arvid-lindblad',
    'Magnussen':    'kevin-magnussen',
    'Norris':       'lando-norris',
    'Ocon':         'esteban-ocon',
    'Perez':        'sergio-perez',
    'Piastri':      'oscar-piastri',
    'Ricciardo':    'daniel-ricciardo',
    'Russell':      'george-russell',
    'Sainz':        'carlos-sainz-jr',
    'Sargent':      'logan-sargeant',    # misspelled in source
    'Schumacher':   'mick-schumacher',
    'Stroll':       'lance-stroll',
    'Tsunoda':      'yuki-tsunoda',
    'Verstappen':   'max-verstappen',
    'Vettel':       'sebastian-vettel',
    'Zhou':         'guanyu-zhou',
}

TEAM_NAME_TO_ID = {
    'Alfa Romeo':   'kick-sauber',      # rebranded 2024
    'Alpha Tauri':  'racing-bulls',     # rebranded 2024
    'Alpine':       'alpine',
    'Aston Martin': 'aston-martin',
    'Ferrari':      'ferrari',
    'Haas':         'haas',
    'Mclaren':      'mclaren',          # capitalisation in source
    'Mercedes':     'mercedes',
    'Red Bull':     'red-bull',
    'Williams':     'williams',
}


def _sheet_to_long(doc_path: Path, name_map: dict, id_col: str) -> pd.DataFrame:
    """
    Read one .numbers file and return a long-format DataFrame with columns:
      year, round, <id_col>, value
    Round = 1-based column index (ignoring the trailing Average column).
    """
    doc = numbers_parser.Document(str(doc_path))
    rows_out = []

    for sheet in doc.sheets:
        try:
            year = int(sheet.name)
        except ValueError:
            continue

        raw = list(sheet.tables[0].rows(values_only=True))
        if not raw:
            continue

        # Last column is Average — drop it; column 0 is the name label
        n_data_cols = len(raw[0]) - 2   # subtract name col and Average col

        for data_row in raw[1:]:
            raw_name = data_row[0]
            if raw_name is None:
                continue
            canonical = name_map.get(str(raw_name).strip())
            if canonical is None:
                print(f"  WARNING: unmapped name '{raw_name}' in {doc_path.stem} {year}")
                continue

            for round_num in range(1, n_data_cols + 1):
                val = data_row[round_num]
                if val is not None:
                    rows_out.append({
                        'year':  year,
                        'round': round_num,
                        id_col:  canonical,
                        'value': float(val),
                    })

    return pd.DataFrame(rows_out)


def main():
    # --- Drivers ---
    print("Converting Drivers-Points...")
    d_pts = _sheet_to_long(
        NUMBERS_DIR / "Drivers-Points.numbers",
        DRIVER_NAME_TO_ID, 'driver_id',
    ).rename(columns={'value': 'actual_points'})

    print("Converting Drivers-Cost...")
    d_cost = _sheet_to_long(
        NUMBERS_DIR / "Drivers-Cost.numbers",
        DRIVER_NAME_TO_ID, 'driver_id',
    ).rename(columns={'value': 'cost'})

    d_df = pd.merge(d_pts, d_cost, on=['year', 'round', 'driver_id'], how='outer')
    d_df = d_df.sort_values(['year', 'round', 'driver_id']).reset_index(drop=True)
    out_path = DATA_DIR / "actual_fantasy_drivers.csv"
    d_df.to_csv(out_path, index=False)
    print(f"  Saved {len(d_df)} rows → {out_path}")
    print(f"  Years: {sorted(d_df['year'].unique())}")
    print(f"  Rows with actual_points: {d_df['actual_points'].notna().sum()}")
    print(f"  Rows with cost: {d_df['cost'].notna().sum()}")

    # --- Constructors ---
    print("\nConverting Teams-Points...")
    c_pts = _sheet_to_long(
        NUMBERS_DIR / "Teams-Points.numbers",
        TEAM_NAME_TO_ID, 'constructor_id',
    ).rename(columns={'value': 'actual_points'})

    print("Converting Teams-Cost...")
    c_cost = _sheet_to_long(
        NUMBERS_DIR / "Teams-Cost.numbers",
        TEAM_NAME_TO_ID, 'constructor_id',
    ).rename(columns={'value': 'cost'})

    c_df = pd.merge(c_pts, c_cost, on=['year', 'round', 'constructor_id'], how='outer')
    c_df = c_df.sort_values(['year', 'round', 'constructor_id']).reset_index(drop=True)
    out_path = DATA_DIR / "actual_fantasy_constructors.csv"
    c_df.to_csv(out_path, index=False)
    print(f"  Saved {len(c_df)} rows → {out_path}")
    print(f"  Years: {sorted(c_df['year'].unique())}")
    print(f"  Rows with actual_points: {c_df['actual_points'].notna().sum()}")
    print(f"  Rows with cost: {c_df['cost'].notna().sum()}")


if __name__ == "__main__":
    main()
