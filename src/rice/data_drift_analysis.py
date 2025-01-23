import pandas as pd
from pathlib import Path

# Antag, at scriptet kører fra src/rice/
reference_data_path = Path('../../data/processed/reference_data.csv')
current_data_path = Path('../../data/processed/current_data.csv')

reference_data = pd.read_csv(reference_data_path)
current_data = pd.read_csv(current_data_path)