import pandas as pd
import numpy as np
from .common import MISSING_DROP_THRESHOLD, WINSOR_LOWER, WINSOR_UPPER, ZSCORE_COLS_MAPPING, ZSCORE_DESCRIPTIONS
from difflib import get_close_matches

def unify_target_column(
    df: pd.DataFrame,
    column: str,
    mapping: dict | None = None,
    rename: str | None = None,
) -> pd.DataFrame:
    """
    Unify target variable to 0/1 (0=Negative, 1=Positive)
    
    Args:
        df           : input dataframe  
        column       : target column
        mapping      : custom mapping rule
    
    Returns:
        pd.DataFrame : dataframe after processing 

    Raises:
        KeyError     : if mapping column missing
    """
    if column not in df.columns:
        raise KeyError(f"{column} doesn't exist")
    
    df = df.copy()
    
    col = df[column].astype(str).str.strip().str.lower()
    unique_vals = col.unique().tolist()
    
    auto_mapping = {
        'alive': 0, 'survived': 0, 'non-bankrupt': 0, 'no': 0, 'n': 0, 'false': 0, '0': 0,
        'failed': 1, 'dead': 1, 'bankrupt': 1, 'yes': 1, 'y': 1, 'true': 1, '1': 1,
    }
    
    # custom mapping
    final_mapping = mapping or auto_mapping
    
    # check unmapped
    unmapped = [v for v in unique_vals if v not in final_mapping]
    if unmapped:
        print(f"Warning: Unmapped value exists: {unmapped}")
    
    df[column] = col.map(final_mapping)
    
    df[column] = pd.to_numeric(df[column], errors='coerce')
   
    if rename:
        df = df.rename(columns={column: rename})

    return df


def clean_data(df: pd.DataFrame)-> pd.DataFrame:
    """
    Clean the dataframe using following steps
        - Fix messy column names to make it easier to work with
        - Convert everything to numbers
        - Remove duplicate entries
        - Drop columns with too many missing values  
        - Clip extreme outliers, this stops crazy ratios from breaking things
        - Remove zero variance columns

    Args:
        df        : dataframe to be cleaned

    Return:
        pd.Dataframe    : dataframe after cleaning
    """
    df = df.copy()
    
    # Fix column names a bit
    df.columns = (
        df.columns.str.strip()
                .str.replace('[^A-Za-z0-9]+','_', regex=True)
                .str.strip('_')
                .str.lower()
    )

    # Additional cleaning
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.drop_duplicates()
    df = df.dropna(axis=1, thresh=int((1 - MISSING_DROP_THRESHOLD) * len(df)))

    # Clip outliers
    for col in df.columns:
        if col == "bankrupt":
            continue
        lo, hi = df[col].quantile(WINSOR_LOWER), df[col].quantile(WINSOR_UPPER)
        df[col] = np.clip(df[col], lo, hi)

    # Drop columns with no variation
    df = df.loc[:, df.std() > 0]

    return df

def match_ratio_columns(df: pd.DataFrame, 
                        target_keywords: dict, 
                        threshold: float = 0.6,
                        manual_mapping: dict | None = None) -> dict:
    """Fuzzy match column names to target keywords, prioritizing manual map."""
    matched_cols = {}
    
    # apply manual mapping first
    if manual_mapping:
        for required_kw, actual_col in manual_mapping.items():
            if actual_col in df.columns:
                matched_cols[required_kw.lower()] = actual_col

    # identify all required keywords not yet matched
    required_keywords = set()
    for components in target_keywords.values():
        for keyword in components.values():
            required_kw_lower = keyword.lower()
            if required_kw_lower not in matched_cols:
                required_keywords.add(required_kw_lower)

    # fuzzy match remaining keywords
    df_cols_lower = [col.lower() for col in df.columns]
    
    for required_kw in required_keywords:
        matches = get_close_matches(required_kw, df_cols_lower, n=1, cutoff=threshold)
        
        if matches:
            original_col = df.columns[df.columns.str.lower() == matches[0]][0]
            matched_cols[required_kw] = original_col

    return matched_cols

def zscore_transform(df: pd.DataFrame, 
                     computed: bool = False,
                     verbose: bool | None= False,
                     manual_col_map: dict | None = None) -> pd.DataFrame:
    """
    Transform a dataframe to keep only the five z-score indicators needed for bankruptcy prediction:
    X1 = Working Capital / Total Assets
    X2 = Retained Earnings / Total Assets
    X3 = EBIT / Total Assets
    X4 = Market Value of Equity / Book Value of Total Debt
    X5 = Sales / Total Assets

    Args:
        df             : input dataframe (raw data or precomputed ratios)
        computed       : if True, dataframe already contains ratios; otherwise calculate from raw data
        target_cols    : formula for computing z-score components
        verbose        : print matching result 
        manual_col_map : Custom mapping for financial keywords to actual column names
    Returns:
        pd.DataFrame : dataframe containing columns X1-X5 and bankrupt
    """
    df = df.copy()
    required_cols = ["X1", "X2", "X3", "X4", "X5", "bankrupt"]
    
    if computed:
        target_cols = ZSCORE_DESCRIPTIONS

        matched_x_cols = {}
        df_cols_lower = df.columns.str.lower()
        
        if manual_col_map:
            for x_key, description in target_cols.items():
                if description in manual_col_map and manual_col_map[description] in df.columns:
                    matched_x_cols[x_key] = manual_col_map[description]

        # use descriptive names for fuzzy matching
        for x_key, description in target_cols.items():
            # skip manual mapping
            if x_key in matched_x_cols:
                continue
            matches = get_close_matches(description.lower(), df_cols_lower, n=1, cutoff=0.7)
            if matches:
                # get the original column name based on the lowercase match
                original_col = df.columns[df_cols_lower == matches[0]][0]
                matched_x_cols[x_key] = original_col

        if verbose:
            print("--- Computed Ratios Matching Results (using descriptions) ---")
            for k in ZSCORE_DESCRIPTIONS.keys():
                print(f"{k} ({ZSCORE_DESCRIPTIONS[k]}) matched to column: {matched_x_cols.get(k, 'Not Found')}")

        try:
            rename_map = {v: k for k, v in matched_x_cols.items()}
            # select and rename columns, raising error if key columns are missing
            df_z = df.rename(columns=rename_map).filter(required_cols)
        except KeyError:
            # re-raise if filtering fails, indicating required columns are missing
            missing_cols = [col for col in required_cols if col not in rename_map.values()]
            raise KeyError(f"Error selecting computed ratios: missing Z-Score columns {missing_cols}. Check matching results.")

    else:
        # calculate ratios from raw data
        target_cols = ZSCORE_COLS_MAPPING

        # match all required financial components using fuzzy matching and manual overrides
        all_matched_cols = match_ratio_columns(df, target_cols, manual_mapping=manual_col_map)

        if verbose:
            print("--- Raw Data Components Matching Results ---")
            for required_kw, matched_col in all_matched_cols.items():
                 print(f"Required keyword '{required_kw}' matched to column: {matched_col}")

        # check for completeness of required components
        required_raw_keywords = {kw.lower() for components in target_cols.values() for kw in components.values()}
        missing_keywords = [kw for kw in required_raw_keywords if kw not in all_matched_cols]
        
        if missing_keywords:
            raise KeyError(f"Missing required columns for keywords: {missing_keywords}. Check manual_col_map.")

        # calculate z-score ratios (X1-X5)
        try:
            # helper function to get the actual column name from the matched results
            def get_col(x_key, comp_key):
                required_kw = target_cols[x_key][comp_key].lower()
                return all_matched_cols[required_kw]

            df_z = pd.DataFrame({
                # X1 = (current_assets - current_liabilities) / total_assets
                "X1": (df[get_col("X1", "numerator_comp1")] - 
                       df[get_col("X1", "numerator_comp2")]) / 
                       df[get_col("X1", "denominator")],
                       
                # X2 = retained_earnings / total_assets
                "X2": df[get_col("X2", "numerator")] / df[get_col("X2", "denominator")],
                      
                # X3 = ebit / total_assets
                "X3": df[get_col("X3", "numerator")] / df[get_col("X3", "denominator")],
                      
                # X4 = market_value_equity / total_debt
                "X4": df[get_col("X4", "numerator")] / df[get_col("X4", "denominator")], 
                      
                # X5 = sales / total_assets
                "X5": df[get_col("X5", "numerator")] / df[get_col("X5", "denominator")],
                      
                "bankrupt": df["bankrupt"]
            })
        except KeyError as e:
            raise KeyError(f"Error accessing columns during calculation: {e}. Check matching results.")

    return df_z

def remove_outlier(df: pd.DataFrame, 
                       cols: list = None, 
                       threshold: float = 10.0,
                       verbose: bool = True) -> pd.DataFrame:
    """
    Remove extreme values from a dataframe by absolute value threshold.
    
    Args:
        df        : input dataframe
        cols      : list of columns to apply; if None, all numeric columns except 'bankrupt'
        threshold : absolute value threshold; any value with abs(value) > threshold will be clipped
        verbose   : whether to print info
        
    Returns:
        pd.DataFrame : dataframe with outliers clipped
    """
    df = df.copy()
    cols = cols or [c for c in df.select_dtypes(include=np.number).columns if c != 'bankrupt']
    
    for col in cols:
        num_low = (df[col] < -threshold).sum()
        num_high = (df[col] > threshold).sum()
        if verbose:
            print(f"{col}: clipping {num_low} values below {-threshold}, {num_high} values above {threshold}")
        df[col] = np.clip(df[col], -threshold, threshold)
        
    return df
