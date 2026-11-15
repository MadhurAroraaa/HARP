from sklearn.ensemble import IsolationForest

def train_model(data):
    model = IsolationForest(contamination=0.05, random_state=42)
    model.fit(data)
    return model

def detect_anomalies(model, data):
    preds = model.predict(data)
    return preds