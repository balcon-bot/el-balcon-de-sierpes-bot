import feedparser
from bs4 import BeautifulSoup

def fetch_latest_news(rss_url):
    try:
        feed = feedparser.parse(rss_url)

        if not feed.entries:
            return None

        entry = feed.entries[0]

        soup = BeautifulSoup(entry.summary, "html.parser")
        clean_summary = soup.get_text()

        if len(clean_summary) > 250:
            clean_summary = clean_summary[:247] + "..."

        return {
            'title': entry.title,
            'summary': clean_summary,
            'link': entry.link
        }

    except Exception as e:
        print(f"❌ Error extrayendo noticias: {e}")
        return None
