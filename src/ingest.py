"""
Download the Kaggle dataset and save to /app/data/raw.csv.
Auth: set KAGGLE_API_TOKEN in the environment (e.g. KGAT_xxx).
"""
import os
import shutil

import kagglehub
import pandas as pd

RAW_CSV = "/app/data/raw.csv"
DATASET_SLUG = "desalegngeb/german-fintech-companies"
KNOWN_FILE = "German_FinTechCompanies.csv"


def ingest() -> None:
    print(f"[ingest] Downloading: {DATASET_SLUG}")
    path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"[ingest] Downloaded to: {path}")

    files = os.listdir(path)
    csv_files = [f for f in files if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {path}. Files: {files}")

    chosen = KNOWN_FILE if KNOWN_FILE in csv_files else csv_files[0]
    os.makedirs(os.path.dirname(RAW_CSV), exist_ok=True)
    shutil.copy2(os.path.join(path, chosen), RAW_CSV)

    df = pd.read_csv(RAW_CSV)
    print(f"[ingest] Saved {len(df)} rows, {len(df.columns)} columns -> {RAW_CSV}")
    print(f"[ingest] Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    ingest()
