import joblib
import os
import pandas as pd
import sys
import argparse
import numpy as np 

# Configuration and Model Loading
MODEL_DIR = "models"
MODEL_PATH        = os.path.join(MODEL_DIR, "final_model.pkl")
FEATURES_PATH     = os.path.join(MODEL_DIR, "feature_columns.pkl")
METADATA_PATH     = os.path.join(MODEL_DIR, "model_metadata.pkl")
SAMPLE_PATH       = os.path.join(MODEL_DIR, "sample_input_for_testing.csv")

try:
    model           = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    metadata        = joblib.load(METADATA_PATH)
except FileNotFoundError:
    print(f"ERROR: Model or features not found in '{MODEL_DIR}'. Try executing notebook first!")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: An unexpected error occurred during model loading: {e}")
    sys.exit(1)

best_threshold  = metadata.get('best_threshold', 0.5) # Default value for robustness

def predict_early_warning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes Early Warning System flags based on Random Forest model and a simple rule.
    """
    
    if df is None or df.empty:
        print("Warning: Input DataFrame is empty. Returning empty result.")
        return pd.DataFrame()

    df_out = df.copy()

    # Model Prediction 
    missing_features = [col for col in feature_columns if col not in df_out.columns]
    if missing_features:
        print("\nWarning: Missing columns required for Random Forest prediction:")
        print(f"   Missing {len(missing_features)} features, RF prediction will be NaN.")
        df_out["ews_probability"] = float('nan')
        df_out["ews_flag"]   = False
    else:
        try:
            df_out["ews_probability"] = model.predict_proba(df_out[feature_columns])[:, 1]
            df_out["ews_flag"]   = df_out["ews_probability"] >= best_threshold
        except Exception as e:
            print(f"ERROR during model prediction: {e}. Setting RF flags to False.")
            df_out["ews_probability"] = float('nan')
            df_out["ews_flag"]   = False
    

    return df_out

def get_input(feature: str, default_val: float) -> float:
    """Prompts user for a single feature value, handling non-numeric input."""
    while True:
        prompt = f"Enter value for '{feature}' (default: {default_val}): "
        user_input = input(prompt).strip()
        if not user_input:
            return default_val
        try:
            return float(user_input)
        except ValueError:
            print("Invalid input. Please enter a number.")

def interactive_mode():
    """Guides user through entering features and runs prediction."""
    print("\n" + "="*60)
    print("✨ EARLY WARNING SYSTEM — INTERACTIVE MODE ✨")
    print("Please enter the required feature values for one record.")
    print("Press ENTER to accept the default value (0.0).")
    print("="*60)

    data = {}
    for feature in feature_columns:
        # A simple default value for interactive mode to keep it moving
        default_val = 0.0 
        value = get_input(feature, default_val)
        data[feature] = [value]
    
    try:
        input_df = pd.DataFrame(data)
    except Exception as e:
        print(f"ERROR: Could not create DataFrame from inputs: {e}")
        return

    # Run prediction
    result = predict_early_warning(input_df)

    # Display results for interactive mode
    print("\n" + "="*60)
    print("PREDICTION RESULT")
    print("="*60)
    if not result.empty:
        print(result[[
            "ews_probability", 
            "ews_flag", 
        ]].to_string(index=False))
    print("="*60)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Applies column name cleaning rules."""
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace('[^A-Za-z0-9]+', '_', regex=True)
    df.columns = df.columns.str.strip('_')
    df.columns = df.columns.str.lower()
    return df

def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Early Warning System prediction script.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--file', 
        type=str, 
        help=f"Path to input CSV file. Example: --file data.csv"
    )
    group.add_argument(
        '-i', '--interactive', 
        action='store_true', 
        help="Enter interactive mode to input one record manually."
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    if args.interactive:
        interactive_mode()
        return

    # File or Sample mode
    input_path = args.file
    df = None

    if input_path:
        if not os.path.isfile(input_path):
            print(f"ERROR: File not found → {input_path}")
            sys.exit(1)
        print(f"Loading data from: {input_path}")
        try: 
            df = pd.read_csv(input_path)
            df = normalize_columns(df)
            if df.empty:
                print("Warning: Input file is empty.")
                sys.exit(0)
        except Exception as e:
            print(f"ERROR: Could not read CSV file ({e}).")
            sys.exit(1)
    else:
        if not os.path.isfile(SAMPLE_PATH):
            print(f"ERROR: No input file specified and sample file not found → {SAMPLE_PATH}")
            sys.exit(1)
        print("No input file provided → using saved **reference sample** for testing.")
        try:
            df = pd.read_csv(SAMPLE_PATH)
        except Exception as e:
            print(f"ERROR: Could not read sample file ({e}).")
            sys.exit(1)

    # Run prediction
    result = predict_early_warning(df)

    if result.empty:
        return

    # Save output automatically
    os.makedirs("output", exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"output/EWS_results_{timestamp}.csv"
    result.to_csv(output_file, index=False)

    # Pretty summary
    n_high = result["ews_flag"].sum()
    print("\n" + "="*60)
    print("EARLY WARNING SYSTEM — BATCH COMPLETE")
    print("="*60)
    print(f"Records processed   : {len(result):,}")
    print(f"High-risk customers : {n_high:,} ({n_high/len(result):.2%})")
    print(f"Results saved to    : {output_file}")
    if n_high > 0:
        flagged_file = f"output/FLAGGED_only_{timestamp}.csv"
        print(f"Flagged records only: {flagged_file}")
    print("="*60)

    # Show first few rows safely
    show_n = min(10, len(result))
    if show_n > 0:
        cols = ["ews_probability", "ews_flag"]
        avail = [c for c in cols if c in result.columns]
        print(f"\nFirst {show_n} prediction(s):")
        print(result[avail].head(show_n).to_string(index=False))


if __name__ == "__main__":
    main()
