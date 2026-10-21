import feedparser
import urllib.parse
import re

def clean_headline(title):
    """
    Strips news agency attribution suffixes (e.g. ' - Economic Times')
    to ensure FinBERT evaluates purely the financial event.
    """
    if not title:
        return ""
    # Remove trailing ' - Publisher Name'
    cleaned = re.sub(r'\s*-\s*[^-]+$', '', title).strip()
    return cleaned if cleaned else title

def fetch_news(query, max_results=20):
    """
    Fetches real-time financial news headlines via Google News RSS feed.
    Enforces a strict 7-day lookback window and returns structured metadata.
    """
    if not query or not query.strip():
        query = "NIFTY"

    encoded_query = urllib.parse.quote_plus(query.strip())
    url = f"https://news.google.com/rss/search?q={encoded_query}+when:7d&hl=en-IN&gl=IN&ceid=IN:en"
    
    try:
        feed = feedparser.parse(url)
        articles = []
        for entry in feed.entries[:max_results]:
            clean_title = clean_headline(entry.title)
            articles.append({
                "title": clean_title,
                "raw_title": entry.title,
                "link": entry.link,
                "published": entry.published if hasattr(entry, 'published') else None
            })
        return articles
    except Exception as e:
        print(f"[!] Error fetching RSS news: {e}")
        return []
