from src.dataset import build_daily_dataset
from src.train import train_model

if __name__ == "__main__":
    print("Fetching historical data and building dataset...")
    df = build_daily_dataset("^NSEI", lookback_days=60)
    print("Training Logistic Regression model...")
    train_model(df)
    print("Pipeline complete. Model saved to models/sentiment_model.pkl")
