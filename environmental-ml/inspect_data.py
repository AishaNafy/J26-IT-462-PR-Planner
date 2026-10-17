
from pathlib import Path
import pandas as pd

FILES = [
    Path("data/pond_iot_2023_raw.csv"),
    Path("data/pond_iot_2023.csv"),
]

for file in FILES:
    print("\n" + "=" * 65)
    print(f"FILE: {file}")

    if not file.exists():
        print("ERROR: File not found.")
        continue

    try:
        df = pd.read_csv(file)
    except Exception as error:
        print(f"ERROR reading CSV: {error}")
        continue

    print(f"Rows: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nMissing values:\n{df.isna().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")

    if "created_date" not in df.columns:
        print("ERROR: created_date column not found.")
        continue

    df["created_date"] = pd.to_datetime(
        df["created_date"], errors="coerce"
    )

    print(
        "Invalid timestamps:",
        f"{df['created_date'].isna().sum():,}"
    )
    print("First timestamp:", df["created_date"].min())
    print("Last timestamp:", df["created_date"].max())
    print(
        "Unique timestamps:",
        f"{df['created_date'].nunique():,}"
    )

    for col in ["water_pH", "TDS", "water_temp"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")

            print(f"\n--- {col} ---")
            print(values.describe().round(3))
            print(
                "Invalid numeric values:",
                f"{values.isna().sum():,}"
            )
        else:
            print(f"\nColumn not found: {col}")

    print("\nFirst 5 rows:")
    print(df.head(5).to_string(index=False))
