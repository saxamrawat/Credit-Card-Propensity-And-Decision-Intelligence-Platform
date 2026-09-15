# Load Raw Data

# Libraries
from pathlib import Path
import pandas as pd

DATA_PATH = Path("data/raw/default of credit card clients.xls")

def load_raw_data(path : Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_excel(path, engine="xlrd", header=[0, 1])