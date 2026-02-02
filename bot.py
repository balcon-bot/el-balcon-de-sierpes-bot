import os
import re
import json
import time
import hashlib
import asyncio
import datetime as dt
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup

import discord
from discord import app_commands
from dotenv import load_dotenv

from PIL import Image, ImageDraw, ImageFont

# Opcional (para postear en X con OAuth1 + subir media)
try:
    from requests_oauthlib import OAuth1
except Exception:
    OAuth1 = None


# =========================
# CONFIG
# =========================
# En Actions NO hay .env por defecto. En server sí.
if os.path.exists("/home/container/.env"):
    load_dotenv("/home/container/.env")
else:
    load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "./data")

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN", "")
DISCORD_CHANNEL_ID = int(os.getenv("DISCORD_CHANNEL_ID", "0"))

CHECK_INTERVAL_SECONDS = int(os.getenv("CHECK_INTERVAL_SECONDS", "300"))

# Modos:
# - RUN_MODE=server => loop infinito (Pterodactyl)
# - RUN_MODE=once   => una pasada y se cierra (GitHub Actions)
RUN_MODE = os.getenv("RUN_MODE", "server").lower()

# X posting (solo POST, no lectura)
X_POST_ENABLED = os.getenv("X_POST_ENABLED", "0") == "1"

# OAuth1 (necesario para publicar y para subir imagen)
X_CONSUMER_KEY = os.getenv("X_CONSUMER_KEY", "")
X_CONSUMER_SECRET = os.getenv("X_CONSUMER_SECRET", "")
X_ACCESS_TOKEN = os.getenv("X_ACCESS_TOKEN", "")
X_ACCESS_TOKEN_SECRET = os.getenv("X_ACCESS_TOKEN_SECRET", "")

# Si quieres intentar subir imagen a X
X_MEDIA_UPLOAD_ENABLED = os.getenv("X_MEDIA_UPLOAD_ENABLED", "0") == "1"

# Cuenta atrás diaria (08:00) - el bot decide si toca publicar
COUNTDOWN_ENABLED = os.getenv("COUNTDOWN_ENABLED", "1") == "1"
COUNTDOWN_HOUR = int(os.getenv("COUNTDOWN_HOUR", "8"))
COUNTDOWN_MINUTE = int(os.getenv("COUNTDOWN_MINUTE", "0"))

# Branding
LOGO_PATH = os.getenv("LOGO_PATH", f"{DATA_DIR}/logo.png")
COUNTDOWN_OUT_PATH = f"{DATA_DIR}/countdown.png"

# Sources JSON
PATH_OFICIALES = os.path.join(DATA_DIR, "x_oficiales.json")
PATH_MEDIOS = os.path.join(DATA_DIR, "x_radar_medios.json")
PATH_INFORMATIVAS = os.path.join(DATA_DIR, "x_informativas.json")
PATH_EXCLUIR = os.path.join(DATA_DIR, "x_excluir.json")

SEEN_PATH = os.path.join(DATA_DIR, "seen_items.json")


# =========================
# UTILS
# =========================
def load_json(path: str, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path: str, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def clean_text(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    return s


def now_local() -> dt.datetime:
    # En GitHub Actions suele ser UTC.
    # No dependemos de tz libs; usaremos hora del sistema.
    # Si quieres, en Actions ponemos TZ=Europe/Madrid.
    return dt.datetime.now()


def normalize_url(u: str) -> str:
    """Normaliza URL para dedupe (quita query tracking típica)."""
    try:
        p = urlparse(u)
        # limpia tracking
        q = p.query
        # conserva query si no es tracking (opcional)
        # para ABC casi siempre la URL buena viene sin query
        new = urlunparse((p.scheme, p.netloc, p.path.rstrip("/"), "", "", ""))
        return new
    except Exception:
        return u.strip()


def first_sentence(text: str, max_len: int = 140) -> str:
    t = clean_text(text)
    if not t:
        return ""
    # cortar por fin de frase
    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    s = parts[0] if parts else t
    s = clean_text(s)
    if len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


# =========================
# DEDUPE STORE
# =========================
class SeenStore:
    def __init__(self, path: str):
        self.path = path
        self.data = load_json(path, {"seen": {}, "last_cleanup": None})
        self.seen: Dict[str, float] = self.data.get("seen", {})  # key -> unix_ts

    def has(self, key: str) -> bool:
        return key in self.seen

    def add(self, key: str):
        self.seen[key] = time.time()

    def save(self):
        self.data["seen"] = self.seen
        self.data["last_cleanup"] = time.time()
        save_json(self.path, self.data)

    def cleanup(self, max_days: int = 45):
        cutoff = time.time() - (max_days * 86400)
        self.seen = {k: v for k, v in self.seen.items() if v >= cutoff}


# =========================
# SOURCES
# =========================
@dataclass
class Source:
    id: str
    name: str
    type: str
    url: str
    rss: Optional[str]
    enabled: bool
    parser: Dict[str, Any]
    tags: List[str]
    priority: int


class SourceManager:
    def __init__(self):
        self.paths = {
            "oficiales": PATH_OFICIALES,
            "medios": PATH_MEDIOS,
            "informativas": PATH_INFORMATIVAS,
            "excluir": PATH_EXCLUIR,
        }
        self.sources: List[Source] = []
        self.exclusions: Dict[str, Any] = {}

    def load(self):
        oficiales = load_json(self.paths["oficiales"], [])
        medios = load_json(self.paths["medios"], [])
        informativas = load_json(self.paths["informativas"], [])

        self.exclusions = load_json(self.paths["excluir"], {
            "deny_keywords": [],
            "deny_domains": [],
            "deny_url_contains": [],
        })

        all_items = []
        all_items.extend(oficiales)
        all_items.extend(medios)
        all_items.extend(informativas)

        sources: List[Source] = []
        for item in all_items:
            try:
                sources.append(Source(
                    id=item["id"],
                    name=item.get("name", item["id"]),
                    type=item.get("type", "unknown"),
                    url=item["url"],
                    rss=item.get("rss"),
                    enabled=item.get("enabled", True),
                    parser=item.get("parser", {"mode": "html"}),
                    tags=item.get("tags", []),
                    priority=int(item.get("priority", 1))
                ))
            except Exception:
                continue

        sources.sort(key=lambda s: s.priority, reverse=True)
        self.sources = sources

    def is_allowed(self, title: str, link: str, src: Optional[Source] = None) -> bool:
        deny_keywords = [k.lower() for k in self.exclusions.get("deny_keywords", [])]
        deny_domains = [d.lower() for d in self.exclusions.get("deny_domains", [])]
        deny_url_contains = [x.lower() for x in self.exclusions.get("deny_url_contains", [])]

        t = (title or "").lower()

        if any(k in t for k in deny_keywords):
            return False

        u = (link or "").lower()
        if any(bad in u for bad in deny_url_contains):
            return False

        # deny a nivel source también
        if src:
            for bad in (src.parser.get("deny_url_contains") or []):
                if bad.lower() in u:
                    return False

            include_kw = [k.lower() for k in (src.parser.get("include_keywords") or [])]
            # si hay include_keywords, exigimos match
            if include_kw:
                if not any(k in t for k in include_kw):
                    return False

        try:
            domain = urlparse(link).netloc.lower()
            if any(domain.endswith(d) or domain == d for d in deny_domains):
                return False
        except Exception:
            pass

        return True


# =========================
# FETCHERS
# =========================
@dataclass
class Item:
    source_id: str
    source_name: str
    title: str
    url: str
    summary: str
    image: Optional[str]
    published_at: Optional[str]
    category: Optional[str]


class WebFetcher:
    def __init__(self, session: Optional[requests.Session] = None):
        self.s = session or requests.Session()
        self.s.headers.update({
            "User-Agent": "ElBalconDeSierpesBot/1.0"
        })

    def fetch_html(self, url: str) -> Optional[str]:
        r = self.s.get(url, timeout=25)
        if r.status_code >= 400:
            return None
        r.encoding = r.apparent_encoding or "utf-8"
        return r.text

    def extract_category_from_article(self, soup: BeautifulSoup) -> Optional[str]:
        # Intentos típicos: meta section, breadcrumbs, etiquetas destacadas.
        # ABC móvil suele poner categoría arriba (ELECCIONES / PROCESIÓN / etc.)
        candidates = []

        for sel in [
            "meta[property='article:section']",
            "meta[name='section']",
            "meta[name='parsely-section']",
        ]:
            m = soup.select_one(sel)
            if m and m.get("content"):
                candidates.append(clean_text(m["content"]))

        # Muchas webs meten la categoría en un elemento tipo 'h2' o 'span' arriba del h1
        h1 = soup.select_one("h1")
        if h1:
            # mira anteriores (pocos)
            prev_texts = []
            node = h1
            for _ in range(6):
                node = node.find_previous()
                if not node:
                    break
                txt = clean_text(node.get_text())
                if txt and 2 <= len(txt) <= 20:
                    prev_texts.append(txt)
            # buscamos el primero "en mayúsculas"
            for txt in prev_texts:
                if txt.isupper() and " " not in txt and len(txt) <= 15:
                    candidates.append(txt)

        # Deduce y limpia
        for c in candidates:
            c2 = clean_text(c)
            if c2 and len(c2) <= 30:
                return c2.upper()
        return None

    def parse_html_listing(self, src: Source) -> List[Tuple[str, str]]:
        """Devuelve lista de (title, link) desde la portada/listado, SIN abrir cada artículo todavía."""
        html = self.fetch_html(src.url)
        if not html:
            return []
        soup = BeautifulSoup(html, "lxml")

        sel_link = src.parser.get("article_link_selector") or "a[href]"
        links = soup.select(sel_link)

        out: List[Tuple[str, str]] = []
        seen_links = set()

        max_items = int(src.parser.get("max_items", 10))

        for a in links:
            href = a.get("href")
            if not href:
                continue

            link = href if href.startswith("http") else urljoin(src.url, href)
            link = normalize_url(link)

            if link in seen_links:
                continue
            seen_links.add(link)

            # Filtra URLs basura en ABC: tags, autores, etc.
            u_low = link.lower()
            if any(bad.lower() in u_low for bad in (src.parser.get("deny_url_contains") or [])):
                continue

            title = clean_text(a.get_text()) or clean_text(a.get("title", ""))

            # Evita anchors tipo "Leer más" etc.
            if len(title) < 12:
                continue

            # Si allow_external=False, evitamos dominios fuera
            if not src.parser.get("allow_external", False):
                if urlparse(link).netloc and urlparse(link).netloc != urlparse(src.url).netloc:
                    continue

            out.append((title, link))
            if len(out) >= max_items:
                break

        return out

    def parse_article_page(self, src: Source, link: str, fallback_title: str) -> Optional[Item]:
        html = self.fetch_html(link)
        if not html:
            return None
        soup = BeautifulSoup(html, "lxml")

        title_sel = src.parser.get("title_selector") or "h1"
        title_el = soup.select_one(title_sel)
        title = clean_text(title_el.get_text()) if title_el else fallback_title
        if not title:
            title = fallback_title

        # summary: primera frase buena (párrafo decente)
        summary = ""
        for p in soup.select("p")[:12]:
            txt = clean_text(p.get_text())
            if len(txt) >= 80:
                summary = txt
                break

        # og:image
        image = None
        og = soup.select_one("meta[property='og:image']")
        if og and og.get("content"):
            image = og["content"].strip()

        # published time
        published_at = None
        pub = soup.select_one("meta[property='article:published_time']")
        if pub and pub.get("content"):
            published_at = pub["content"].strip()

        category = self.extract_category_from_article(soup)

        return Item(
            source_id=src.id,
            source_name=src.name,
            title=title,
            url=link,
            summary=summary,
            image=image,
            published_at=published_at,
            category=category
        )

    def fetch(self, src: Source) -> List[Item]:
        if not src.enabled:
            return []

        # Listado -> abrimos artículo solo si no lo hemos visto (ahorra requests y evita “0 items” por filtros)
        listing = self.parse_html_listing(src)
        out: List[Item] = []
        for (t, link) in listing:
            item = self.parse_article_page(src, link, t)
            if item:
                out.append(item)
        return out


# =========================
# X POSTING
# =========================
class XPoster:
    def __init__(self):
        self.enabled = X_POST_ENABLED
        self.can_oauth1 = OAuth1 is not None and all([
            X_CONSUMER_KEY, X_CONSUMER_SECRET, X_ACCESS_TOKEN, X_ACCESS_TOKEN_SECRET
        ])

    def _auth(self):
        return OAuth1(
            X_CONSUMER_KEY,
            X_CONSUMER_SECRET,
            X_ACCESS_TOKEN,
            X_ACCESS_TOKEN_SECRET
        )

    def upload_media(self, image_bytes: bytes) -> Optional[str]:
        if not (self.enabled and X_MEDIA_UPLOAD_ENABLED and self.can_oauth1):
            return None

        upload_url = "https://upload.twitter.com/1.1/media/upload.json"
        r = requests.post(
            upload_url,
            auth=self._auth(),
            files={"media": image_bytes},
            timeout=30
        )
        if r.status_code >= 400:
            return None
        j = r.json()
        return j.get("media_id_string")

    def post_tweet(self, text: str, media_id: Optional[str] = None) -> Tuple[bool, str]:
        if not (self.enabled and self.can_oauth1):
            return False, "X posting disabled or missing OAuth1 credentials."

        url = "https://api.twitter.com/2/tweets"
        payload: Dict[str, Any] = {"text": text}

        if media_id:
            payload["media"] = {"media_ids": [media_id]}

        r = requests.post(url, auth=self._auth(), json=payload, timeout=30)
        if r.status_code >= 400:
            return False, f"{r.status_code} {r.text[:200]}"
        return True, "OK"


# =========================
# COUNTDOWN IMAGE
# =========================
def easter_date(year: int) -> dt.date:
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return dt.date(year, month, day)


def palm_sunday(year: int) -> dt.date:
    return easter_date(year) - dt.timedelta(days=7)


def next_palm_sunday(today: dt.date) -> dt.date:
    ps = palm_sunday(today.year)
    if today <= ps:
        return ps
    return palm_sunday(today.year + 1)


def generate_countdown_image(days_left: int, out_path: str, logo_path: Optional[str] = None):
    W, H = 1080, 1080
    base = Image.new("RGB", (W, H), (90, 35, 140))  # morado
    draw = ImageDraw.Draw(base)

    # patrón cofrade simple repetido
    for y in range(0, H, 90):
        for x in range(0, W, 90):
            draw.line((x+45, y+20, x+45, y+70), fill=(120, 70, 170), width=3)
            draw.line((x+30, y+40, x+60, y+40), fill=(120, 70, 170), width=3)
            draw.polygon([(x+15, y+75), (x+35, y+45), (x+55, y+75)], outline=(120, 70, 170))

    pad = 90
    draw.rounded_rectangle((pad, pad, W-pad, H-pad), radius=40, fill=(255, 255, 255))

    try:
        font_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 160)
        font_mid = ImageFont.truetype("DejaVuSans-Bold.ttf", 74)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 46)
    except Exception:
        font_big = font_mid = font_small = None

    draw.text((W//2, 190), "CUENTA ATRÁS", fill=(50, 20, 90), anchor="mm", font=font_mid)
    draw.text((W//2, 380), f"{days_left}", fill=(50, 20, 90), anchor="mm", font=font_big)
    draw.text((W//2, 525), "días para", fill=(50, 20, 90), anchor="mm", font=font_small)
    draw.text((W//2, 610), "DOMINGO DE RAMOS", fill=(50, 20, 90), anchor="mm", font=font_mid)

    if logo_path and os.path.exists(logo_path):
        logo = Image.open(logo_path).convert("RGBA")
        logo = logo.resize((260, 260))
        base.paste(logo, (W//2 - 130, 720), logo)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    base.save(out_path, "PNG")


# =========================
# DISCORD BOT
# =========================
intents = discord.Intents.default()
client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

sources = SourceManager()
fetcher = WebFetcher()
seen = SeenStore(SEEN_PATH)
xposter = XPoster()


def build_x_text(item: Item) -> str:
    # Sin citar el medio, pero sí el enlace “Fuente: …”
    # Incluye 1 frase del resumen
    title = item.title.strip()
    if len(title) > 170:
        title = title[:167] + "..."

    cat = (item.category or "").strip()
    cat_prefix = f"[{cat}] " if cat else ""

    s1 = first_sentence(item.summary, max_len=140)
    resumen_line = f"\n\n📝 {s1}" if s1 else ""

    txt = f"🔔 Última hora\n{cat_prefix}{title}{resumen_line}\n\nFuente: {item.url}"

    # Ajuste por 280
    if len(txt) > 280:
        # recortamos resumen primero
        if resumen_line:
            resumen_line = ""
            txt = f"🔔 Última hora\n{cat_prefix}{title}\n\nFuente: {item.url}"
        if len(txt) > 280:
            max_title = 280 - len("🔔 Última hora\n\nFuente: ") - len(item.url) - len(cat_prefix) - 3
            title2 = title[:max(0, max_title)].rstrip() + "…"
            txt = f"🔔 Última hora\n{cat_prefix}{title2}\n\nFuente: {item.url}"
            txt = txt[:280]
    return txt


async def post_to_discord(channel: discord.TextChannel, item: Item):
    title = item.title
    if item.category:
        title = f"[{item.category}] {title}"

    desc = first_sentence(item.summary, max_len=200) if item.summary else ""
    embed = discord.Embed(
        title=title,
        url=item.url,
        description=desc,
        color=0x6A1FB4
    )
    if item.image:
        embed.set_image(url=item.image)
    await channel.send(embed=embed)


async def process_item(channel: discord.TextChannel, src: Source, item: Item):
    if not sources.is_allowed(item.title, item.url, src=src):
        return False

    norm = normalize_url(item.url)
    key = sha1(norm)
    if seen.has(key):
        return False

    # Discord
    await post_to_discord(channel, item)

    # X
    if X_POST_ENABLED:
        media_id = None
        if X_MEDIA_UPLOAD_ENABLED and item.image:
            try:
                img = requests.get(item.image, timeout=20).content
                media_id = xposter.upload_media(img)
            except Exception:
                media_id = None

        ok, msg = xposter.post_tweet(build_x_text(item), media_id=media_id)
        if not ok:
            await channel.send(f"⚠️ No pude publicar en X: {msg}")

    seen.add(key)
    return True


async def publish_countdown_if_time(channel: discord.TextChannel):
    if not COUNTDOWN_ENABLED:
        return

    now = now_local()
    if now.hour != COUNTDOWN_HOUR or now.minute != COUNTDOWN_MINUTE:
        return

    today = now.date()
    target = next_palm_sunday(today)
    days_left = (target - today).days

    generate_countdown_image(days_left, COUNTDOWN_OUT_PATH, logo_path=LOGO_PATH if os.path.exists(LOGO_PATH) else None)
    file = discord.File(COUNTDOWN_OUT_PATH, filename="countdown.png")
    await channel.send(content=f"⏳ Quedan **{days_left}** días para **Domingo de Ramos**.", file=file)


async def run_one_pass():
    """Una pasada: carga fuentes, scrapea, publica novedades, y listo."""
    sources.load()
    channel = client.get_channel(DISCORD_CHANNEL_ID)
    if not isinstance(channel, discord.TextChannel):
        print("❌ Canal Discord no encontrado o no es de texto.")
        return

    await publish_countdown_if_time(channel)

    published = 0
    for src in sources.sources:
        if not src.enabled:
            continue

        items = fetcher.fetch(src)
        for item in items:
            ok = await process_item(channel, src, item)
            if ok:
                published += 1
            await asyncio.sleep(float(src.parser.get("sleep", 0.7)))

    seen.cleanup(max_days=45)
    seen.save()
    print(f"✅ Pasada terminada. Publicados nuevos: {published}")


async def fetch_loop_forever():
    await client.wait_until_ready()
    while True:
        try:
            await run_one_pass()
        except Exception as e:
            print(f"❌ Error loop: {e}")
        await asyncio.sleep(CHECK_INTERVAL_SECONDS)


@client.event
async def on_ready():
    print(f"✅ Conectado como {client.user}")
    try:
        await tree.sync()
    except Exception:
        pass


@tree.command(name="status", description="Estado del bot y fuentes cargadas")
async def status_cmd(interaction: discord.Interaction):
    sources.load()
    await interaction.response.send_message(
        f"✅ OK\nFuentes activas: {sum(1 for s in sources.sources if s.enabled)}\n"
        f"RUN_MODE={RUN_MODE}\n"
        f"X_POST_ENABLED={int(X_POST_ENABLED)} | X_MEDIA_UPLOAD_ENABLED={int(X_MEDIA_UPLOAD_ENABLED)}\n"
        f"CHECK_INTERVAL_SECONDS={CHECK_INTERVAL_SECONDS}",
        ephemeral=True
    )


@tree.command(name="run_once", description="Ejecuta una pasada de scraping ahora")
async def run_once_cmd(interaction: discord.Interaction):
    await interaction.response.defer(ephemeral=True)
    try:
        await run_one_pass()
        await interaction.followup.send("✅ Pasada ejecutada. Revisa el canal.", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)


async def main_async():
    if not DISCORD_TOKEN or not DISCORD_CHANNEL_ID:
        raise SystemExit("Faltan DISCORD_TOKEN o DISCORD_CHANNEL_ID")

    if RUN_MODE == "once":
        # En Actions: arranca, espera ready, hace una pasada, y cierra.
        await client.login(DISCORD_TOKEN)
        await client.connect(reconnect=False)
    else:
        # Server: deja tasks corriendo
        async with client:
            client.loop.create_task(fetch_loop_forever())
            await client.start(DISCORD_TOKEN)


if __name__ == "__main__":
    # Si RUN_MODE=once, no podemos usar client.connect() así sin control.
    # Truco: en once usamos start() pero paramos tras una pasada.
    async def runner():
        if not DISCORD_TOKEN or not DISCORD_CHANNEL_ID:
            raise SystemExit("Faltan DISCORD_TOKEN o DISCORD_CHANNEL_ID")

        if RUN_MODE == "once":
            async with client:
                await client.start(DISCORD_TOKEN)
        else:
            async with client:
                client.loop.create_task(fetch_loop_forever())
                await client.start(DISCORD_TOKEN)

    # Para RUN_MODE=once: usamos on_ready para disparar y cerrar
    if RUN_MODE == "once":
        @client.event
        async def on_ready():
            print(f"✅ (once) Conectado como {client.user}")
            try:
                await run_one_pass()
            finally:
                await client.close()

    asyncio.run(runner())
