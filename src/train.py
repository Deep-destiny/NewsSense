import pickle
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

MODEL_PATH = "models/sentiment_model.pkl"

def train_model(df):
    """
    Trains a Logistic Regression classifier on financial sentiment features
    to forecast directional market probability.
    """
    if df.empty or 'target' not in df.columns:
        print("[!] Dataset empty or missing target column. Training aborted.")
        return None

    feature_col = 'sentiment_feature' if 'sentiment_feature' in df.columns else df.columns[0]
    X = df[[feature_col]]
    y = df['target']

    # Initialize and train regularized Logistic Regression
    model = LogisticRegression(C=1.0, solver='lbfgs', random_state=42)
    model.fit(X, y)

    # Evaluate train accuracy
    preds = model.predict(X)
    acc = accuracy_score(y, preds)

    # Serialize trained artifact
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump({"model": model, "feature_col": feature_col}, f)

    print(f"[OK] Model trained successfully on {len(df)} sessions. Training Accuracy: {acc * 100:.2f}%")
    return model

def predict_market_trend(sentiment_score=0.5):
    """
    Dynamically predicts directional market trend and calibrated probability
    using the real sentiment score produced by the FinBERT inference engine.
    """
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        with open(MODEL_PATH, 'rb') as f:
            artifact = pickle.load(f)

        # Handle backward compatibility if artifact is a raw model or a dict
        model = artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact

        # Clamp input score between 0.0 and 1.0
        score = float(np.clip(sentiment_score, 0.0, 1.0))

        # Predict class probabilities using DataFrame with feature column to avoid feature name warnings
        input_df = pd.DataFrame([[score]], columns=[artifact.get("feature_col", "sentiment_feature")])
        probas = model.predict_proba(input_df)[0]
        prob_bearish = float(probas[0])
        prob_bullish = float(probas[1])

        # Dynamic classification based on probability threshold
        if prob_bullish >= 0.52:
            label = "Bullish"
            confidence = prob_bullish
        elif prob_bearish >= 0.52:
            label = "Bearish"
            confidence = prob_bearish
        else:
            label = "Neutral"
            confidence = max(prob_bullish, prob_bearish)

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "prob_bullish": round(prob_bullish, 4),
            "prob_bearish": round(prob_bearish, 4)
        }
    except Exception as e:
        print(f"[!] Prediction inference error: {e}")
        return None

# Alias for backwards compatibility
load_model_and_predict = predict_market_trend
