# Load Raw Data

# Libraries
from pathlib import Path
import pandas as pd

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = PROJECT_ROOT / "data/raw/default of credit card clients.xls"

def load_raw_data(path : Path = DATA_PATH) -> pd.DataFrame:
    return pd.read_excel(path, engine="xlrd", header=[0, 1])