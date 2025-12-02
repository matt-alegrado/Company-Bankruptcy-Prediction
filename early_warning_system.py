import joblib
import os
import pandas as pd
import sys

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
    print("Model not found! Try executing notebook first!")
    sys.exit(1)

best_threshold  = metadata['best_threshold']
PROFIT_THR = metadata['per_share_net_profit_before_tax_yuan_thr']
QUICK_THR = metadata['quick_ratio_thr']

def predict_early_warning(df: pd.DataFrame = None,
                          profit_col: str = "per_share_net_profit_before_tax_yuan",
                          quick_col: str = "quick_ratio"
                          ) -> pd.DataFrame:
    """
    Returns df with 
        ews_probability : Random Forest risk score
        ews_flag_model     : True if probability ≥ best_threshold
        ews_flag_rule   : True if the approved 2-feature rule triggers
        ews_high_risk   : final flag (OR)
    """
    
    df = df.copy()
    
    missing_features = [col for col in feature_columns if col not in df.columns]
    if missing_features:
        print("ERROR: Missing columns required for Random Forest prediction:")
        print(f"   Missing {len(missing_features)} features:")
        for col in missing_features:   
            print(f"     - {col}")
        print(f"\nExpected {len(feature_columns)} features. Available: {len([col for col in feature_columns if col in df.columns])}")
        print("Hint: Required features are saved in models/feature_columns.pkl")
        print("     You can see full list with: joblib.load('models/feature_columns.pkl')")
        
        # Fall back gracefully: set RF prediction to NaN
        df["ews_probability"] = float('nan')
        df["ews_flag_model"]     = False
    else:
        df["ews_probability"] = model.predict_proba(df[feature_columns])[:, 1]
        df["ews_flag_model"]     = df["ews_probability"] >= best_threshold

    # Exact reproduction of the surrogate tree
    df["ews_flag_rule"] = (df[profit_col] <= PROFIT_THR) & (df[quick_col] <= QUICK_THR)

    # Final decision
    df["ews_high_risk"] = df["ews_flag_model"] | df["ews_flag_rule"]
    
    return df

def main():
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        if not os.path.isfile(input_path):
            print(f"Error: File not found → {input_path}")
            sys.exit(1)
        print(f"Loading data from: {input_path}")
        try: 
            df = pd.read_csv(input_path)
        except Exception as e:
            print(f"Warning: Could not read CSV ({e}).")
            sys.exit(1)
    else:
        print("No input file provided → using saved reference sample for testing")
        df = pd.read_csv(SAMPLE_PATH)

    # Run prediction
    result = predict_early_warning(df)

    # Save output automatically
    os.makedirs("output", exist_ok=True)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"output/EWS_results_{timestamp}.csv"
    result.to_csv(output_file, index=False)

    # Pretty summary
    n_high = result["ews_high_risk"].sum()
    print("\n" + "="*60)
    print("EARLY WARNING SYSTEM — COMPLETE")
    print("="*60)
    print(f"Records processed   : {len(result):,}")
    print(f"High-risk customers : {n_high:,} ({n_high/len(result):.2%})")
    print(f"Results saved to    : {output_file}")
    if n_high > 0:
        flagged_file = f"output/FLAGGED_only_{timestamp}.csv"
        result[result["ews_high_risk"]].to_csv(flagged_file, index=False)
        print(f"Flagged records only: {flagged_file}")
    print("="*60)

    # Show first few rows safely
    show_n = min(10, len(result))
    if show_n > 0:
        cols = ["ews_probability", "ews_flag_rf", "ews_flag_rule", "ews_high_risk", "ews_reason"]
        avail = [c for c in cols if c in result.columns]
        print(f"\nFirst {show_n} prediction(s):")
        print(result[avail].head(show_n).to_string(index=False))


if __name__ == "__main__":
    main()