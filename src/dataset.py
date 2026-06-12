import yfinance as yf
import pandas as pd
import datetime

def build_daily_dataset(ticker="^NSEI", lookback_days=30):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=lookback_days)
    data = yf.download(ticker, start=start, end=end, progress=False)
    data['avg_sentiment'] = 0.5 # Placeholder for dummy training
    # Target: 1 if close > open, else 0
    data['target'] = (data['Close'] > data['Open']).astype(int)
    return data.dropna()
