
from pathlib import Path
import pandas as pd

files = [
    Path("data/pond_iot_2023_raw.csv"),
    Path("data/pond_iot_2023.csv"),
]

for file in files:
    print("\n" + "=" * 65)
    print("FILE:", file.name)

    df = pd.read_csv(file)
    df["created_date"] = pd.to_datetime(
        df["created_date"], errors="coerce"
    )

    df = df.dropna(subset=["created_date"])
    df = df.sort_values("created_date")

    # Number of records recorded at each timestamp
    counts = df.groupby("created_date").size()

    print("\nRecords per timestamp:")
    print(counts.describe().round(2))
    print("Timestamps with multiple records:", (counts > 1).sum())

    print("\nExamples of timestamps with multiple records:")
    print(counts[counts > 1].head(10).to_string())

    # Time gaps between unique timestamps
    times = pd.Series(
        df["created_date"].drop_duplicates().sort_values()
    )
    gaps = times.diff().dropna().dt.total_seconds() / 60

    print("\nTime gaps between unique timestamps (minutes):")
    print(gaps.describe(
        percentiles=[0.25, 0.50, 0.75, 0.90, 0.99]
    ).round(2))

    print("Largest 10 time gaps (minutes):")
    print(gaps.nlargest(10).to_string(index=False))

    # Suspicious pH readings: flag for review, do not delete
    ph = pd.to_numeric(df["water_pH"], errors="coerce")

    print("\npH readings below 0:", (ph < 0).sum())
    print("pH readings above 14:", (ph > 14).sum())

    print("\nSample records around the first repeated timestamp:")
    repeated = counts[counts > 1]

    if not repeated.empty:
        timestamp = repeated.index[0]
        print(
            df[df["created_date"] == timestamp]
            .to_string(index=False)
        )
