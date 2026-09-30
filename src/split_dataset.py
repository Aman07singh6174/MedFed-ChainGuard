import pandas as pd
import numpy as np

df = pd.read_csv("data/parkinsons_updrs.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

splits = np.array_split(df, 3)

for i, split in enumerate(splits):
    split.to_csv(f"data/hospital_{i+1}.csv", index=False)
    print(f"hospital_{i+1}.csv: {len(split)} rows")