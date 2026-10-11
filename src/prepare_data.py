
import pandas as pd
from pathlib import Path

# File paths
INPUT_FILE = Path("data/raw/transactions.csv")
OUTPUT_FILE = Path("data/processed/transactions_processed.csv")

# Create output directory if it does not exist
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Columns required for fraud detection
FEATURE_COLUMNS = [
    "step",
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
]

TARGET_COLUMN = "isFraud"

def main():
    print("Loading PaySim transaction data...")

    # Read only the columns needed for modeling
    df = pd.read_csv(
        INPUT_FILE,
        usecols=FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    print(f"Original rows: {len(df):,}")

    # Remove rows with missing required values
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN]).copy()

    # Calculate useful transaction features
    df["origin_balance_change"] = (
        df["oldbalanceOrg"] - df["newbalanceOrig"]
    )

    df["destination_balance_change"] = (
        df["newbalanceDest"] - df["oldbalanceDest"]
    )

    # Encode transaction type as numeric columns
    df = pd.get_dummies(df, columns=["type"], dtype=int)

    # Keep the fraud label separate from input features
    df[TARGET_COLUMN] = df[TARGET_COLUMN].astype(int)

    # Save the processed dataset
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Processed rows: {len(df):,}")
    print(f"Fraudulent transactions: {df[TARGET_COLUMN].sum():,}")
    print(f"Saved processed data to: {OUTPUT_FILE}")
    print("Data preparation completed successfully!")

if __name__ == "__main__":
    main()
