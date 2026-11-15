def create_features(df):
    df["pressure_diff"] = df["pressure"].diff()
    df["flow_diff"] = df["flow_rate"].diff()
    df["rolling_mean"] = df["pressure"].rolling(10).mean()
    df["rolling_std"] = df["pressure"].rolling(10).std()
    df["pressure_rate"] = df["pressure"].pct_change()
    df.fillna(0, inplace=True)
    return df