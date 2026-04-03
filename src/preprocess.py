"""
Load raw.csv, build features and target.

Target: has_former_name
  1 = company has a non-null "Former name" (was rebranded/restructured)
  0 = otherwise

Feature strategy:
  - Drop ID, Name, Former name (identifiers / target source)
  - Label-encode categoricals (works well with RandomForest)
  - Fill nulls: median for numerics, mode for categoricals
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

RAW_CSV = "/app/data/raw.csv"
DROP_COLS = ["ID", "Name", "Former name"]


def load_and_prepare(csv_path: str = RAW_CSV) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)

    # Build binary target
    y = df["Former name"].fillna("").str.strip().ne("").astype(int)
    print(f"[preprocess] Target distribution:\n{y.value_counts().to_string()}")

    # Drop identifier and target-source columns
    feature_df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = feature_df.select_dtypes(include=["object"]).columns.tolist()

    # Fill missing values
    for col in num_cols:
        feature_df[col] = feature_df[col].fillna(feature_df[col].median())
    for col in cat_cols:
        mode = feature_df[col].mode()
        feature_df[col] = feature_df[col].fillna(mode.iloc[0] if not mode.empty else "UNKNOWN")

    # Label-encode categoricals
    le = LabelEncoder()
    for col in cat_cols:
        feature_df[col] = le.fit_transform(feature_df[col].astype(str))

    print(f"[preprocess] Feature matrix: {feature_df.shape}")
    return feature_df, y


if __name__ == "__main__":
    load_and_prepare()
