import pandas as pd
import csv

df = pd.read_csv("dataset.csv", dtype=str)

# Remove duplicate prompts
df = df.drop_duplicates(subset=["prompt"])

# Ensure every column is string
df = df.astype(str)

# Save with quotes around every field
df.to_csv(
    "dataset.csv",
    index=False,
    quoting=csv.QUOTE_ALL
)

print(df.head())
print(df.dtypes)
print(df["category"].value_counts())
print(f"\nTotal samples after deduplication: {len(df)}")