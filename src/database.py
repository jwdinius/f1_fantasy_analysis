import pandas as pd
from sqlalchemy import create_engine
from pathlib import Path

class F1DataManager:
    def __init__(self):
        # Path setup using pathlib
        # .parent gets 'src/', .parent.parent gets the project root
        self.project_root = Path(__file__).resolve().parent.parent
        self.db_path = self.project_root / "data" / "sqlite" / "f1db.db"
        
        # Check if the database file exists
        if not self.db_path.exists():
            raise FileNotFoundError(f"F1 database not found at: {self.db_path}")
            
        # SQLite connection string (3 slashes for relative, 4 for absolute)
        # .as_posix() ensures the path string uses forward slashes for the URI
        self.database_url = f"sqlite:///{self.db_path.resolve().as_posix()}"
        self.engine = create_engine(self.database_url)

    def get_driver_stats(self):
        """Query to test the connection using the 'drivers' table."""
        query = "SELECT full_name, total_race_wins, total_podiums, total_points FROM driver"
        return pd.read_sql(query, self.engine)

    def get_constructor_standings(self):
        """Pulls constructor data based on your schema."""
        query = "SELECT name, total_championship_wins, total_race_wins FROM constructor"
        return pd.read_sql(query, self.engine)
    
    def query(self, query_string):
        return pd.read_sql(query_string, self.engine)


if __name__ == "__main__":
    try:
        db = F1DataManager()
        df = db.get_driver_stats()
        print("Successfully connected to the F1 Database!")
        print("View of Driver Stats:")
        print(df.head())
        print("View of Constructor Stats:")
        df = db.get_constructor_standings()
        print(df.head())
    except FileNotFoundError as e:
        print(f"Error: {e}")