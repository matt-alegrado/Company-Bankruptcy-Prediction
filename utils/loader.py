import pandas as pd
from pathlib import Path
from .common import COLUMNS_MAPPING, INPUT_DIR, ZSCORE_DESCRIPTIONS

def load_data(filepath: str | Path,
              mapping: dict | None = COLUMNS_MAPPING,
              columns_to_keep: dict | None = None,
              verbose: bool = True) -> pd.DataFrame:
    """
    Load csv file from filepath, apply columns mapping and columns to keep

    Args:
        filepath        : csv file path
        mapping         : column mapping dict
        columns_to_keep : columns to keep
        verbose         : print log

    Return:
        pd.DataFrame    : dataframe after processing

    Raises:
        FileNotFoundError: if file doesn't exist
        RuntimeError: if csv parse failed
        KeyError: if columns_to_keep has missing column
    """
    filepath = INPUT_DIR / filepath
    if not filepath.exists():
        raise FileNotFoundError(f"File doesn't exist: {filepath}")
    if not filepath.is_file():
        raise ValueError(f"Path is not a file: {filepath}")
    
    mapping = mapping or COLUMNS_MAPPING

    try:
        df = pd.read_csv(filepath, dtype=str)
    except Exception as e:
        raise RuntimeError(f"Fail to read file: {e}")
    
    df = df.rename(columns=mapping)

    if columns_to_keep:
        missing = [c for c in columns_to_keep if c not in df.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")
        df = df[columns_to_keep]
    
    if verbose:
        print(f"Loaded: {df.shape} from {filepath}")
    
    return df

def basic_info(df: pd.DataFrame):
    """
    Print Dataframe basic info

    Args:
        df : input dataframe
    """
    print("Missing value counts:")
    print(df.isna().sum())
    print("\nDataset info:")
    print(df.describe())

def transfer_z_col_to_name(df: pd.DataFrame,
                           mapping: dict | None = ZSCORE_DESCRIPTIONS) -> pd.DataFrame:
    """
    Map z-score column to real financial feature name.

    Args:
        df       : input dataframe
        mapping  : mapping between x_i (e.g., 'X1') to its real descriptive feature name (e.g., 'working_capital_to_total_assets').

    Return:
        pd.DataFrame      : dataframe with columns renamed to feature names.
    """

    rename_map = {}
    for z_col, feature_name in mapping.items():
        if z_col in df.columns and isinstance(feature_name, str):
            rename_map[z_col] = feature_name
    
    # rename the columns in a copy of the DataFrame
    df_renamed = df.rename(columns=rename_map)
    
    return df_renamed
