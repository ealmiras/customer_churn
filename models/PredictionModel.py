import joblib

artifact = joblib.load("models/churn_model_rf.pkl")

model = artifact["model"]
threshold = artifact["threshold"]

def predict_churn(X, model, threshold):
    proba = model.predict_proba(X)[:, 1]
    return (proba >= threshold).astype(int), proba

# The model and threshold can now be used for predictions