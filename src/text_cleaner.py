import feedparser
import pandas as pd
from datetime import datetime

def fetch_rss_titles(rss_urls, max_entries=1000):
    """
    Récupère les titres depuis une liste de flux RSS et retourne un DataFrame avec date, titre, source.
    """
    all_titles = []
    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries[:max_entries]:
            title = entry.title
            published = entry.get('published')
            try:
                date = datetime.strptime(str(published)[:16], '%a, %d %b %Y') if published else None
            except Exception:
                date = None
            if title and date:
                all_titles.append({'date': date, 'title': title, 'source': url})
    return pd.DataFrame(all_titles).dropna()
