import pandas as pd

def load_data(path):
    df = pd.read_csv(path)
    df = df[["Pressure", "Flow_Rate", "Temperature", "Vibration"]]
    df.columns = ["pressure", "flow_rate", "temperature", "vibration"]
    return df