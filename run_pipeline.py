import time
from src.dataset import build_daily_dataset
from src.train import train_model

def execute_pipeline():
    print("=" * 60)
    print("  NEWSSENSE: FINANCIAL SENTIMENT & ML TRAINING PIPELINE")
    print("=" * 60)
    start_time = time.time()

    print("\n[1/3] Ingesting 7 years of historical OHLCV data from Yahoo Finance (^NSEI)...")
    dataset = build_daily_dataset(ticker="^NSEI", lookback_days=2555)
    print(f"      Retrieved {len(dataset)} trading sessions with engineered sentiment features.")

    print("\n[2/3] Fitting regularized Logistic Regression classifier...")
    model = train_model(dataset)

    elapsed = time.time() - start_time
    print(f"\n[3/3] Serialization complete in {elapsed:.2f}s.")
    print("      Model saved to 'models/sentiment_model.pkl' - Ready for inference.\n")
    print("=" * 60)

if __name__ == "__main__":
    execute_pipeline()
