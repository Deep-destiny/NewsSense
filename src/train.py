import pickle
import os
from sklearn.linear_model import LogisticRegression

MODEL_PATH = "models/sentiment_model.pkl"

def train_model(df):
    if df.empty:
        return
    X = df[['avg_sentiment']] # Simplified feature
    y = df['target']
    model = LogisticRegression()
    model.fit(X, y)
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)

def load_model_and_predict():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    pred = model.predict([[0.5]])[0]
    label = "Positive" if pred == 1 else "Negative"
    return {"label": label, "confidence": 0.75}
