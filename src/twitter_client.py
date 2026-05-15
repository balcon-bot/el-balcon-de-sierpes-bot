import tweepy
import os
from dotenv import load_dotenv

load_dotenv()

class TwitterBot:
    def __init__(self):
        self.client = tweepy.Client(
            consumer_key=os.getenv("TWITTER_API_KEY"),
            consumer_secret=os.getenv("TWITTER_API_SECRET"),
            access_token=os.getenv("TWITTER_ACCESS_TOKEN"),
            access_token_secret=os.getenv("TWITTER_ACCESS_SECRET")
        )

    def post_tweet_with_link(self, text, discord_url):
        full_text = text + chr(10) + chr(10) + "🖼️ Ver Cartel:" + chr(10) + discord_url

        if len(full_text) > 280:
            allowed = 280 - len(discord_url) - 25
            full_text = text[:allowed] + "..." + chr(10) + chr(10) + "🖼️ Ver Cartel:" + chr(10) + discord_url

        try:
            self.client.create_tweet(text=full_text)
            print("✅ Tweet publicado correctamente.")
            return True
        except Exception as e:
            print("❌ Error en Twitter: " + str(e))
            return False
