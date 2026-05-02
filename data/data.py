import os
import pandas as pd
import sqlite3 as sqlite

class Storage:
    def __init__(self, dataset_raw_path: str, dataset_db_path: str, table_name: sqlite):
        self.ds_raw_path = dataset_raw_path
        self.ds_db_path = dataset_db_path
        self.table = table_name

    def init(self):
        if not os.path.exists(self.ds_db_path):
            self._create_database_from_scv()

    def _create_database_from_scv(self) -> None:
        if not os.path.exists(self.ds_raw_path):
            raise FileNotFoundError(f"Not found {self.ds_raw_path}")
        
        df = pd.read_csv(self.ds_raw_path)
        with sqlite.connect(self.ds_db_path) as connect:
            df.to_sql(self.table, connect, if_exists="replace", index=None)

    def load_data(self):
        if not os.path.exists(self.ds_db_path):
            raise FileNotFoundError(f"Not found {self.ds_db_path}")
        
        with sqlite.connect(self.ds_db_path) as connect:
            return pd.read_sql(f"SELECT * FROM {self.table}", connect)