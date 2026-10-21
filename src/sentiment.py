import pandas as pd
from transformers import pipeline
import logging

# Suppress verbose transformers logging
logging.getLogger("transformers").setLevel(logging.ERROR)

# Global pipeline cache (Singleton pattern to prevent re-loading 440MB weights on every query)
_pipeline_cache = None

def get_sentiment_pipeline():
    """Lazily initialize and cache the FinBERT sentiment analysis pipeline."""
    global _pipeline_cache
    if _pipeline_cache is None:
        _pipeline_cache = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
    return _pipeline_cache

def analyze_sentiment_batch(news_items):
    """
    Analyzes sentiment for a batch of news items using FinBERT.
    Returns:
        df: Pandas DataFrame containing titles, links, timestamps, sentiment labels, and confidence scores.
        metrics: Dictionary with aggregated sentiment metrics (bullish/bearish ratio, net sentiment score).
    """
    df = pd.DataFrame(news_items)
    if df.empty or 'title' not in df.columns:
        return df, {"net_score": 0.5, "positive_pct": 0, "negative_pct": 0, "neutral_pct": 0}

    sentiment_pipe = get_sentiment_pipeline()
    titles = df['title'].tolist()
    
    # Run batch inference
    results = sentiment_pipe(titles)

    df['sentiment'] = [res['label'].capitalize() for res in results]
    df['score'] = [round(float(res['score']), 4) for res in results]

    # Calculate numerical polarity weights: Positive = +1 * score, Negative = -1 * score, Neutral = 0
    weights = []
    for sentiment, score in zip(df['sentiment'], df['score']):
        if sentiment == 'Positive':
            weights.append(score)
        elif sentiment == 'Negative':
            weights.append(-score)
        else:
            weights.append(0.0)
    df['polarity'] = weights

    # Compute batch-level summary metrics
    total_articles = len(df)
    pos_count = (df['sentiment'] == 'Positive').sum()
    neg_count = (df['sentiment'] == 'Negative').sum()
    neu_count = (df['sentiment'] == 'Neutral').sum()

    # Net normalized sentiment score scaled between 0.0 (extreme bearish) and 1.0 (extreme bullish)
    # 0.5 represents perfectly neutral
    mean_polarity = df['polarity'].mean() if total_articles > 0 else 0.0
    net_score = round(float(0.5 + (mean_polarity * 0.5)), 4)
    net_score = max(0.0, min(1.0, net_score))

    metrics = {
        "total": total_articles,
        "positive_count": int(pos_count),
        "negative_count": int(neg_count),
        "neutral_count": int(neu_count),
        "positive_pct": round(float((pos_count / total_articles) * 100), 1) if total_articles > 0 else 0,
        "negative_pct": round(float((neg_count / total_articles) * 100), 1) if total_articles > 0 else 0,
        "neutral_pct": round(float((neu_count / total_articles) * 100), 1) if total_articles > 0 else 0,
        "net_score": net_score
    }

    return df, metrics
