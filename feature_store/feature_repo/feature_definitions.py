
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "transactions.csv"

# Store the generated Feast data on D: because C: is full.
FEAST_DATA_PATH = Path(os.environ.get("FEAST_DATA_PATH", r"D:\FraudDetectionData\feature_repo\data"))
OUTPUT_FILE = FEAST_DATA_PATH / "fraud_transactions.parquet"

FEAST_DATA_PATH.mkdir(parents=True, exist_ok=True)

SIMULATION_START = datetime(2020, 1, 1, tzinfo=timezone.utc)

# Read the source CSV in batches to limit memory usage.
CHUNK_SIZE = 100_000


# --------------------------------------------------
# Prepare the Feast offline feature source
# --------------------------------------------------

def prepare_feast_data() -> Path:
    """Convert PaySim transactions to Feast-compatible Parquet."""

    columns = [
        "step",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    feast_columns = [
        "event_timestamp",
        "step",
        "nameOrig",
        "nameDest",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "origin_balance_change",
        "destination_balance_change",
    ]

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {RAW_DATA_PATH}"
        )

    # Import here so the script gives a clear error if unavailable.
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError(
            "PyArrow is required. Install it with: "
            "python -m pip install pyarrow"
        ) from exc

    # Never overwrite an existing output without explicit permission.
    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}\n"
            "Inspect or move it before running this script again."
        )

    print(f"Input:  {RAW_DATA_PATH}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Chunk size: {CHUNK_SIZE:,} rows")

    writer = None
    total_rows = 0

    try:
        for chunk_number, df in enumerate(
            pd.read_csv(
                RAW_DATA_PATH,
                usecols=columns,
                chunksize=CHUNK_SIZE,
            ),
            start=1,
        ):
            df = df.dropna(subset=columns).copy()

            if df.empty:
                continue

            df["origin_balance_change"] = (
                df["oldbalanceOrg"] - df["newbalanceOrig"]
            )

            df["destination_balance_change"] = (
                df["newbalanceDest"] - df["oldbalanceDest"]
            )

            # PaySim step is simulation time, not an actual timestamp.
            df["event_timestamp"] = (
                pd.Timestamp(SIMULATION_START)
                + pd.to_timedelta(df["step"], unit="h")
            )

            df = df[feast_columns]

            table = pa.Table.from_pandas(
                df,
                preserve_index=False,
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    OUTPUT_FILE,
                    table.schema,
                    compression="snappy",
                )

            writer.write_table(table)

            total_rows += len(df)
            print(
                f"Chunk {chunk_number}: "
                f"{len(df):,} rows written "
                f"(total {total_rows:,})"
            )

        if writer is None:
            raise ValueError(
                "No usable transactions were found in the input CSV."
            )

    except Exception:
        if writer is not None:
            writer.close()
            writer = None

        # Remove only the incomplete file created by this run.
        if OUTPUT_FILE.exists():
            OUTPUT_FILE.unlink()

        raise

    finally:
        if writer is not None:
            writer.close()

    print(f"Prepared {total_rows:,} transactions.")
    print(f"Parquet saved to: {OUTPUT_FILE}")

    return OUTPUT_FILE


# --------------------------------------------------
# Feast entities
# --------------------------------------------------

origin_account = Entity(
    name="origin_account",
    join_keys=["nameOrig"],
    value_type=__import__("feast").ValueType.STRING,
    description="Origin account identifier from PaySim",
)

destination_account = Entity(
    name="destination_account",
    join_keys=["nameDest"],
    value_type=__import__("feast").ValueType.STRING,
    description="Destination account identifier from PaySim",
)


# --------------------------------------------------
# Shared offline source
# --------------------------------------------------

transactions_source = FileSource(
    name="paysim_transactions_source",
    path=str(OUTPUT_FILE),
    timestamp_field="event_timestamp",
)


# --------------------------------------------------
# Origin-account features
# --------------------------------------------------

origin_transaction_features = FeatureView(
    name="origin_transaction_features",
    entities=[origin_account],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="amount", dtype=Float64),
        Field(name="oldbalanceOrg", dtype=Float64),
        Field(name="newbalanceOrig", dtype=Float64),
        Field(name="origin_balance_change", dtype=Float64),
        Field(name="step", dtype=Int64),
    ],
    source=transactions_source,
    online=True,
)


# --------------------------------------------------
# Destination-account features
# --------------------------------------------------

destination_transaction_features = FeatureView(
    name="destination_transaction_features",
    entities=[destination_account],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="amount", dtype=Float64),
        Field(name="oldbalanceDest", dtype=Float64),
        Field(name="newbalanceDest", dtype=Float64),
        Field(name="destination_balance_change", dtype=Float64),
        Field(name="step", dtype=Int64),
    ],
    source=transactions_source,
    online=True,
)


if __name__ == "__main__":
    prepare_feast_data()

