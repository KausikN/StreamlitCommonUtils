# Imports
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict

# Main Functions
# Basic Dataset Functions
def load_csv(path: str) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    
    return pd.read_csv(path)

def save_csv(dataset: pd.DataFrame, path: str) -> None:
    """
    Save a pandas DataFrame to a CSV file.

    Args:
        dataset (pd.DataFrame): DataFrame to save.
        path (str): Destination CSV file path.
    """
    path = Path(path)
    dataset.to_csv(path, index=False)

# Datatype Functions
def autodetect_column_types(data: pd.DataFrame) -> Dict[str, str]:
    """
    Automatically detect the type of each column in a DataFrame.
    Types detected: 'int', 'float', 'bool', 'datetime', 'text', 'unknown'

    Args:
        data (pd.DataFrame): Input dataset.

    Returns:
        Dict[str, str]: Column name → detected type
    """
    column_types = {}

    for col in data.columns:
        series = data[col].dropna()  # ignore NaNs for type detection
        if series.empty:
            column_types[col] = "unknown"
            continue

        # Boolean detection
        if series.dropna().isin([0, 1, True, False]).all():
            column_types[col] = "bool"
            continue

        # Integer detection
        if pd.api.types.is_integer_dtype(series):
            column_types[col] = "int"
            continue

        # Float detection
        if pd.api.types.is_float_dtype(series):
            column_types[col] = "float"
            continue

        # Datetime detection
        try:
            pd.to_datetime(series, errors="raise")
            column_types[col] = "datetime"
            continue
        except Exception:
            pass

        # Fallback to text
        column_types[col] = "text"

    return column_types

def is_categorizable(
    data: pd.DataFrame, 
    col: str, 
    max_unique: int = None, 
    max_fraction: float = None
) -> bool:
    """
    Determine whether a column is suitable for categorical treatment.

    Args:
        data (pd.DataFrame): The dataset.
        col (str): Column name to check.
        max_unique (int, optional): Maximum number of unique values to automatically consider as categorical. Defaults to None.
        max_fraction (float, optional): Maximum fraction of unique values relative to total rows to consider as categorical. Defaults to None.

    Returns:
        bool: True if the column can be treated as categorical, False otherwise.
    """
    series = data[col]
    n_unique = series.nunique()
    n_total = len(series)

    if max_unique is not None and n_unique <= max_unique:
        return True
    if max_fraction is not None and (n_unique / n_total) <= max_fraction:
        return True

    return False
