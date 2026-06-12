import pandas as pd
from transformers import pipeline
import logging

# Suppress some transformers warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

def analyze_sentiment_batch(news_items):
    df = pd.DataFrame(news_items)
    if df.empty:
        return df
    
    # Initialize pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-tone")
    results = sentiment_pipeline(df['title'].tolist())
    
    df['sentiment'] = [res['label'].capitalize() for res in results]
    df['score'] = [res['score'] for res in results]
    
    return df
