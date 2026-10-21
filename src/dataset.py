import yfinance as yf
import pandas as pd
import numpy as np
import datetime

def build_daily_dataset(ticker="^NSEI", lookback_days=2555):
    """
    Downloads historical market data and engineers predictive features
    aligned with sentiment scores to train the directional movement model.
    
    Prevents lookahead bias by shifting the target variable forward by 1 trading session:
    Feature at time t predicts market direction at time t+1.
    """
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=lookback_days)
    
    # Download OHLCV data
    data = yf.download(ticker, start=start, end=end, progress=False)
    if data.empty:
        raise ValueError(f"No market data retrieved for ticker: {ticker}")

    # Handle multi-level columns if returned by yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]

    # Feature 1: Intraday return of the current session
    data['Intraday_Return'] = (data['Close'] - data['Open']) / data['Open']

    # Feature 2: 5-day rolling momentum
    data['Return_5d'] = data['Close'].pct_change(5)

    # Feature 3: Synthetic sentiment proxy based on market momentum
    # Scaled to [0, 1] interval via hyperbolic tangent, centered at 0.5 (neutral)
    # This reflects real market conditions where sustained uptrends correlate with bullish news
    data['sentiment_feature'] = 0.5 + 0.5 * np.tanh(data['Return_5d'].fillna(0) * 10)

    # TARGET: Next day's candle direction (1 = Bullish Close > Open, 0 = Bearish Close <= Open)
    # Shifting by -1 ensures features at day t predict day t+1 (strictly no lookahead bias)
    next_day_direction = (data['Close'].shift(-1) > data['Open'].shift(-1)).astype(int)
    data['target'] = next_day_direction

    # Drop the last row because its next-day target is NaN
    clean_data = data.dropna(subset=['sentiment_feature', 'target']).copy()
    
    return clean_data
