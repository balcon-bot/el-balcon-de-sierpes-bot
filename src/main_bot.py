import os
import subprocess
from dotenv import load_dotenv

from style_engine import StyleEngine
from image_generator import create_cofrade_card
from news_fetcher import fetch_latest_news
from twitter_client import TwitterBot
from state_manager import StateManager

load_dotenv()

def main():
    print("🕯️ Iniciando El Balcón de Sierpes Bot...")

    state_mgr = StateManager()

    rss_url = os.getenv(
        'NEWS_RSS_URL',
        'https://www.abc.es/sevilla/semana-santa/rss/'
    )

    news_data = fetch_latest_news(rss_url)

    if not news_data:
        print("⚠️ Sin noticias o error de conexión.")
        return

    current_title = news_data['title']

    print(f"📰 Detectada: {current_title[:50]}...")

    if state_mgr.is_duplicate(current_title):
        print("🔄 DUPLICADO: Omitiendo publicación.")
        return

    print("✨ NUEVA: Generando contenido...")

    style_bot = StyleEngine()

    text_content = style_bot.generate_post_content(news_data)

    image_path = create_cofrade_card(
        title=current_title,
        subtitle="Actualidad Cofrade"
    )

    with open("last_post_data.txt", "w", encoding="utf-8") as f:
        f.write(text_content + "
")
        f.write(image_path)

    subprocess.run(["python", "src/discord_sender.py"])

    msg_id = "latest"

    if os.path.exists("last_msg_id.txt"):
        with open("last_msg_id.txt", "r") as f:
            msg_id = f.read().strip()

    guild_id = os.getenv('DISCORD_GUILD_ID')
    chan_id = os.getenv('DISCORD_CHANNEL_ID')

    discord_url = f"https://discord.com/channels/{guild_id}/{chan_id}/{msg_id}"

    tw_bot = TwitterBot()
    tw_bot.post_tweet_with_link(text_content, discord_url)

    state_mgr.update_last_title(current_title)

    print("🏁 Ciclo completado.")

if __name__ == "__main__":
    main()
