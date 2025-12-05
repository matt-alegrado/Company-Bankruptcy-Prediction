from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.3
VIF_THRESHOLD = 5
MISSING_DROP_THRESHOLD = 0.40
WINSOR_LOWER = 0.02
WINSOR_UPPER = 0.98
CORRELATION_THRESH = 0.15

PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

COLUMNS_MAPPING = {
    'X1':  'Current Assets',
    'X2':  'Cost of Goods Sold', 
    'X3':  'Depreciation and Amortization',
    'X4':  'EBITDA',
    'X5':  'Inventory',
    'X6':  'Net Income',
    'X7':  'Total Receivables', 
    'X8':  'Market Value',
    'X9':  'Net Sales',
    'X10':  'Total Assets',
    'X11':  'Total Long-term Debt',
    'X12':  'EBIT', 
    'X13':  'Gross Profit',
    'X14':  'Total Current Liabilities',
    'X15':  'Retained Earnings',
    'X16':  'Total Revenue',
    'X17':  'Total Liabilities', 
    'X18':  'Total Operating Expenses',
}

# using standard accounting name here
ZSCORE_COLS_MAPPING = {
    # Working Capital (current_assets - current_liabilities) / Total Assets
    "X1": {
        "numerator_comp1": "current_assets", 
        "numerator_comp2": "current_liabilities", 
        "denominator": "total_assets"
    },
    # Retained Earnings / Total Assets
    "X2": {
        "numerator": "retained_earnings", 
        "denominator": "total_assets"
    },
    # EBIT / Total Assets
    "X3": {
        "numerator": "ebit", 
        "denominator": "total_assets"
    },
    # Market Value of Equity / Total Debt
    "X4": {
        "numerator": "market_value_equity", 
        "denominator": "total_debt"
    },
    # Sales / Total Assets
    "X5": {
        "numerator": "sales", 
        "denominator": "total_assets"
    }
}

ZSCORE_DESCRIPTIONS = {
    "X1": "working_capital_to_total_assets",
    "X2": "retained_earnings_to_total_assets",
    "X3": "ebit_to_total_assets",
    "X4": "market_value_equity_to_total_debt",
    "X5": "sales_to_total_assets"
}

print(f"utils loaded | data path: {INPUT_DIR}")
