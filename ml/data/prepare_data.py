from data.data_loader import load_data
from data.clean_data import clean_data
from features.feature_engineering import create_features

def prepare_pipeline(path):
    df = load_data(path)
    df = clean_data(df)
    df = create_features(df)
    return df