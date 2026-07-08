"""
Converts raw CSVs to Parquet. We use Parquet (not because it's trendy) so
DuckDB can read it lazily and fast, and so we're forced to nail down a schema
early instead of letting pandas silently infer wrong types.
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_and_convert(filename: str, dtypes: dict | None = None) -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / filename, dtype=dtypes)

    out_path = OUT_DIR / filename.replace(".csv", ".parquet")
    try:
        df.to_parquet(out_path, index=False)
    except ImportError:
        # pyarrow not installed in this environment -- fall back to CSV so the
        # rest of the pipeline still has something to read. Install pyarrow
        # (see requirements.txt) to get real Parquet output.
        out_path = OUT_DIR / filename
        df.to_csv(out_path, index=False)

    print(f"Converted {filename} -> {out_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    load_and_convert("application_train.csv", dtypes={"SK_ID_CURR": "int64", "TARGET": "int64"})
    load_and_convert("bureau.csv", dtypes={"SK_ID_CURR": "int64"})
    load_and_convert("previous_application.csv", dtypes={"SK_ID_CURR": "int64"})
