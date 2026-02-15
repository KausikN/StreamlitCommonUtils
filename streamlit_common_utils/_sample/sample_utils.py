# Imports
import numpy as np
from typing import Dict

# Main Functions
# Basic Dataset Functions
def load_csv(path: str) -> pd.DataFrame:
    '''
    Load a CSV file into a pandas DataFrame.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    '''
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    return pd.read_csv(path)