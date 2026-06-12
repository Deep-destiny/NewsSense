import feedparser

def fetch_news(query, max_results=20):
    url = f"https://news.google.com/rss/search?q={query}+when:7d"
    feed = feedparser.parse(url)
    articles = []
    for entry in feed.entries[:max_results]:
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published if hasattr(entry, 'published') else None
        })
    return articles
