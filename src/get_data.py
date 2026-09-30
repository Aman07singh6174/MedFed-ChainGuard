from ucimlrepo import fetch_ucirepo
import pandas as pd

# Fetch the Parkinsons Telemonitoring dataset (id=189)
parkinsons = fetch_ucirepo(id=189)

X = parkinsons.data.features
y = parkinsons.data.targets

# Combine features and targets, save as CSV
df = pd.concat([X, y], axis=1)
df.to_csv("data/parkinsons_updrs.csv", index=False)

print("Saved! Shape:", df.shape)
print(df.head())

