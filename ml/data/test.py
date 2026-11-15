from curses import window

import pandas as pd

# load data
df = pd.read_csv("ml/data/location_aware_gis_leakage_dataset.csv")

# select only 4 columns
df = df[["Pressure", "Flow_Rate", "Temperature", "Vibration"]]

# rename (important)
df = pd.DataFrame(window, columns=["pressure", "flow_rate", "temperature", "vibration"])

# CLEAN
df = df.dropna()
df = df.drop_duplicates()

# FEATURES
df["pressure_diff"] = df["pressure"].diff()
df["flow_diff"] = df["flow_rate"].diff()
df["rolling_mean"] = df["pressure"].rolling(10).mean()
df["rolling_std"] = df["pressure"].rolling(10).std()
df["pressure_rate"] = df["pressure"].pct_change()

df.fillna(0, inplace=True)

# OUTPUT
print(df.head())