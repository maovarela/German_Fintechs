import kagglehub
import os
import pandas as pd

# Download the dataset and discover files
path = kagglehub.dataset_download("desalegngeb/german-fintech-companies")
print("Downloaded to:", path)
print("Files found:", os.listdir(path))

# Load the first CSV found
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
if csv_files:
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    print("First 5 records:")
    print(df.head())
else:
    print("No CSV files found. All files:", os.listdir(path))
