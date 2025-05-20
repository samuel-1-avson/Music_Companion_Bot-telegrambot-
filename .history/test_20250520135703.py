import os
import logging
import sys
import re
import json
import base64
import pytz
import signal
import atexit
from typing import Dict, List, Optional, Tuple, Any, Union
from dotenv import load_dotenv
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from cryptography.fernet import Fernet
import aiohttp
import asyncio
import sqlite3
from contextlib import contextmanager
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes,
    filters, CallbackQueryHandler, ConversationHandler
)
from telegram.error import TimedOut, NetworkError

import yt_dlp
from openai import OpenAI
import importlib
if importlib.util.find_spec("lyricsgenius") is not None:
    import lyricsgenius
else:
    lyricsgenius = None
if importlib.util.find_spec("speech_recognition") is not None:
    import speech_recognition as sr
else:
    sr = None
from functools import lru_cache

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "https://your-callback-url.com")

# Enable logging with detailed format
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s - [%(filename)s:%(lineno)d]',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN) if GENIUS_ACCESS_TOKEN and lyricsgenius else None
cipher = Fernet(Fernet.generate_key())  # For encrypting sensitive data

# Conversation states
MOOD, PREFERENCE, ACTION, SPOTIFY_CODE = range(4)

# Track active downloads and user contexts
active_downloads = set()
user_contexts: Dict[int, Dict] = {}
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ==================== SPOTIFY HELPER FUNCTIONS ====================

async def get_spotify_token() -> Optional[str]:
    """Get Spotify access token using client credentials."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        logger.warning("Spotify credentials not configured")
        return None

    auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {auth_base64}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(url, headers=headers, data=data) as response:
                response.raise_for_status()
                json_response = await response.json()
                token = json_response.get("access_token")
                if not token:
                    logger.error("No access token in Spotify response")
                    return None
                logger.debug(f"Successfully retrieved Spotify token")
                return token
    except aiohttp.ClientError as e:
        logger.error(f"Error getting Spotify token: {e}", exc_info=True)
        return None

@lru_cache(maxsize=100)
async def search_spotify_track(token: str, query: str) -> Optional[Dict]:
    """Search for a track on Spotify."""
    if not token:
        logger.warning("No Spotify token provided for track search")
        return None

    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "type": "track", "limit": 1}

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                json_response = await response.json()
                items = json_response.get("tracks", {}).get("items", [])
                if not items:
                    logger.info(f"No Spotify tracks found for query: {query}")
                    return None
                logger.debug(f"Found Spotify track for query: {query}")
                return items[0]
    except aiohttp.ClientError as e:
        logger.error(f"Error searching Spotify track: {e}", exc_info=True)
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_spotify_recommendations(token: str, seed_tracks: List[str], limit: int = 5) -> List[Dict]:
    """Get track recommendations from Spotify."""
    if not token or not seed_tracks:
        logger.warning("No token or seed tracks provided for Spotify recommendations")
        return []

    url = "https://api.spotify.com/v1/recommendations"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"seed_tracks": ",".join(seed_tracks[:2]), "limit": limit}

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                json_response = await response.json()
                tracks = json_response.get("tracks", [])
                logger.debug(f"Retrieved {len(tracks)} Spotify recommendations")
                return tracks
    except aiohttp.ClientError as e:
        logger.error(f"Error getting Spotify recommendations: {e}", exc_info=True)
        return []

async def get_user_spotify_token(user_id: int, code: str) -> Optional[Dict]:
    """Exchange authorization code for Spotify access and refresh tokens."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI:
        logger.warning("Spotify OAuth credentials not configured")
        return None

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}'.encode()).decode()}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": SPOTIFY_REDIRECT_URI
    }

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(url, headers=headers, data=data) as response:
                response.raise_for_status()
                token_data = await response.json()
                if not token_data.get("access_token"):
                    logger.error("No access token in Spotify user token response")
                    return None
                token_data["expires_at"] = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600))).timestamp()
                logger.debug(f"Successfully retrieved Spotify user token for user {user_id}")
                return token_data
    except aiohttp.ClientError as e:
        logger.error(f"Error getting user Spotify token: {e}", exc_info=True)
        return None

async def refresh_spotify_token(user_id: int) -> Optional[str]:
    """Refresh Spotify access token using refresh token."""
    context = user_contexts.get(user_id, {})
    refresh_token = context.get("spotify", {}).get("refresh_token")
    if not refresh_token:
        logger.warning(f"No refresh token found for user {user_id}")
        return None

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}'.encode()).decode()}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    try:
        decrypted_refresh_token = cipher.decrypt(refresh_token).decode()
    except Exception as e:
        logger.error(f"Error decrypting refresh token for user {user_id}: {e}", exc_info=True)
        return None

    data = {"grant_type": "refresh_token", "refresh_token": decrypted_refresh_token}

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post(url, headers=headers, data=data) as response:
                response.raise_for_status()
                token_data = await response.json()
                if not token_data.get("access_token"):
                    logger.error("No access token in Spotify refresh token response")
                    return None
                expires_at = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600))).timestamp()
                user_contexts[user_id]["spotify"] = {
                    "access_token": cipher.encrypt(token_data.get("access_token").encode()),
                    "refresh_token": cipher.encrypt(token_data.get("refresh_token", decrypted_refresh_token).encode()),
                    "expires_at": expires_at
                }
                logger.debug(f"Successfully refreshed Spotify token for user {user_id}")
                return token_data.get("access_token")
    except aiohttp.ClientError as e:
        logger.error(f"Error refreshing Spotify token: {e}", exc_info=True)
        return None

async def get_user_spotify_data(user_id: int, endpoint: str) -> Optional[List[Dict]]:
    """Fetch user-specific Spotify data (recently played or top tracks)."""
    context = user_contexts.get(user_id, {})
    spotify_data = context.get("spotify", {})
    access_token = spotify_data.get("access_token")
    expires_at = spotify_data.get("expires_at")

    if not access_token or (expires_at and datetime.now(pytz.UTC).timestamp() > expires_at):
        access_token = await refresh_spotify_token(user_id)
        if not access_token:
            logger.warning(f"Failed to refresh Spotify token for user {user_id}")
            return None
    else:
        try:
            access_token = cipher.decrypt(access_token).decode()
        except Exception as e:
            logger.error(f"Error decrypting access token for user {user_id}: {e}", exc_info=True)
            return None

    url = f"https://api.spotify.com/v1/me/{endpoint}"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"limit": 10}

    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                json_response = await response.json()
                items = json_response.get("items", [])
                logger.debug(f"Fetched {len(items)} items from Spotify endpoint {endpoint} for user {user_id}")
                return items
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching Spotify user data ({endpoint}): {e}", exc_info=True)
        return None

# ==================== YOUTUBE HELPER FUNCTIONS ====================

def is_valid_youtube_url(url: str) -> bool:
    """Check if the URL is a valid YouTube URL."""
    if not url:
        return False
    patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=',
        r'(https?://)?youtu\.be/',
        r'(https?://)?(www\.)?youtube\.com/shorts/'
    ]
    return any(re.search(pattern, url) for pattern in patterns)

def sanitize_filename(filename: str) -> str:
    """Remove invalid characters from filenames."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return sanitized[:100]

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_youtube_audio(url: str) -> Dict[str, Any]:
    """Download audio from a YouTube video with retries."""
    video_id_match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11})', url)
    if not video_id_match:
        logger.error(f"Invalid YouTube URL or video ID: {url}")
        return {"success": False, "error": "Invalid YouTube URL or video ID"}

    try:
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio[abr<=128]/bestaudio',
            'outtmpl': f'{DOWNLOAD_DIR}/%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            'postprocessor_args': ['-acodec', 'copy'],
            'prefer_ffmpeg': False,
            'max_filesize': 50 * 1024 * 1024,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                logger.error(f"Could not extract video information for URL: {url}")
                return {"success": False, "error": "Could not extract video information"}
            title = sanitize_filename(info.get('title', 'Unknown Title'))
            artist = info.get('artist', info.get('uploader', 'Unknown Artist'))
            logger.debug(f"Extracted info for {title} by {artist}")
            ydl.download([url])
            audio_path = None
            for ext in ['m4a', 'webm', 'mp3', 'opus']:
                potential_path = os.path.join(DOWNLOAD_DIR, f"{title}.{ext}")
                if os.path.exists(potential_path):
                    audio_path = potential_path
                    break
            if not audio_path:
                for file in os.listdir(DOWNLOAD_DIR):
                    if file.startswith(title[:20]):
                        audio_path = os.path.join(DOWNLOAD_DIR, file)
                        break
            if not audio_path or not os.path.exists(audio_path):
                logger.error(f"Downloaded file not found for {title}")
                return {"success": False, "error": "Downloaded file not found or inaccessible"}
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            if file_size_mb > 50:
                os.remove(audio_path)
                logger.warning(f"File {audio_path} exceeds 50 MB, deleted")
                return {"success": False, "error": "File exceeds 50 MB Telegram limit"}
            logger.debug(f"Successfully downloaded {audio_path}, size: {file_size_mb:.2f} MB")
            return {
                "success": True,
                "title": title,
                "artist": artist,
                "thumbnail_url": info.get('thumbnail', ''),
                "duration": info.get('duration', 0),
                "audio_path": audio_path
            }
    except Exception as e:
        logger.error(f"Error downloading YouTube audio: {e}", exc_info=True)
        return {"success": False, "error": str(e)}

def search_youtube(query: str, max_results: int = 5) -> List[Dict]:
    """Search YouTube for videos matching the query."""
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'default_search': 'ytsearch',
            'format': 'bestaudio',
            'noplaylist': True,
            'playlist_items': f'1-{max_results}'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_query = f"ytsearch{max_results}:{query}"
            info = ydl.extract_info(search_query, download=False)
            if not info or 'entries' not in info:
                logger.info(f"No YouTube results for query: {query}")
                return []
            results = [
                {
                    'title': entry.get('title', 'Unknown Title'),
                    'url': entry.get('url') or f"https://www.youtube.com/watch?v={entry.get('id')}",
                    'thumbnail': entry.get('thumbnail', ''),
                    'uploader': entry.get('uploader', 'Unknown Artist'),
                    'duration': entry.get('duration', 0),
                    'id': entry.get('id', '')
                }
                for entry in info['entries'] if entry
            ]
            logger.debug(f"Found {len(results)} YouTube results for query: {query}")
            return results
    except Exception as e:
        logger.error(f"Error searching YouTube: {e}", exc_info=True)
        return []

# ==================== AI AND LYRICS FUNCTIONS ====================

async def generate_chat_response(user_id: int, message: str) -> str:
    """Generate a conversational response using OpenAI."""
    if not client:
        logger.warning("OpenAI client not initialized")
        return "I'm having trouble connecting to my AI service. Please try again later."

    context = user_contexts.get(user_id, {
        "mood": None,
        "preferences": [],
        "conversation_history": [],
        "spotify": {}
    })

    context["conversation_history"] = context["conversation_history"][-50:]  # Limit history

    messages = [
        {"role": "system", "content": (
            "You are a friendly, empathetic music companion bot named MelodyMind. "
            "Your role is to: "
            "1. Have natural conversations about music and feelings "
            "2. Recommend songs based on mood and preferences "
            "3. Provide emotional support through music "
            "4. Keep responses concise but warm (around 2-3 sentences) "
            "If the user has linked their Spotify account, use their listening history to personalize responses."
        )}
    ]

    if context.get("mood") or context.get("spotify"):
        system_content = f"The user's current mood is: {context.get('mood', 'unknown')}. "
        if context.get("preferences"):
            system_content += f"Their music preferences include: {', '.join(context.get('preferences', ['various genres']))}. "
        if context.get("spotify", {}).get("recently_played"):
            artists = [item["track"]["artists"][0]["name"] for item in context["spotify"].get("recently_played", [])]
            system_content += f"They recently listened to artists: {', '.join(artists[:3])}. "
        messages.append({"role": "system", "content": system_content})

    for hist in context["conversation_history"][-10:]:
        messages.append(hist)

    messages.append({"role": "user", "content": message})

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        reply = response.choices[0].message.content
        context["conversation_history"].extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply}
        ])
        user_contexts[user_id] = context
        logger.debug(f"Generated chat response for user {user_id}: {reply}")
        return reply
    except Exception as e:
        logger.error(f"Error generating chat response: {e}", exc_info=True)
        return "I'm having trouble thinking right now. Let's talk about music instead!"

def get_lyrics(song_title: str, artist: Optional[str] = None) -> str:
    """Get lyrics for a song using Genius API with fallback."""
    if not genius:
        logger.warning("Genius client not initialized")
        return "Lyrics service unavailable. Try asking me for a song instead!"
    try:
        song = genius.search_song(song_title, artist) if artist else genius.search_song(song_title)
        if not song:
            logger.info(f"No lyrics found for '{song_title}' by {artist or 'unknown'}")
            return f"Couldn't find lyrics for '{song_title}'" + (f" by {artist}" if artist else "") + ". Try another song!"
        lyrics = song.lyrics
        lyrics = re.sub(r'\[.*?\]', '', lyrics)
        lyrics = re.sub(r'\d+Embed$', '', lyrics)
        lyrics = re.sub(r'Embed$', '', lyrics)
        header = f"🎵 {song.title} by {song.artist} 🎵\n\n"
        logger.debug(f"Retrieved lyrics for {song.title} by {song.artist}")
        return header + lyrics.strip()
    except Exception as e:
        logger.error(f"Error fetching lyrics: {e}", exc_info=True)
        return f"Couldn't find lyrics for '{song_title}'" + (f" by {artist}" if artist else "") + ". Try another song!"

async def detect_mood_from_text(user_id: int, text: str) -> str:
    """Detect mood from user's message using AI."""
    if not client:
        logger.warning("OpenAI client not initialized for mood detection")
        return user_contexts.get(user_id, {}).get("mood", "happy")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Detect mood from this text: '{text}'"}],
            max_tokens=50
        )
        mood = response.choices[0].message.content.lower().strip()
        logger.debug(f"Detected mood for user {user_id}: {mood}")
        return mood if mood else "happy"
    except Exception as e:
        logger.error(f"Error detecting mood: {e}", exc_info=True)
        return "happy"

async def is_music_request(user_id: int, message: str) -> Dict:
    """Use AI to determine if a message is a music/song request."""
    if not client:
        logger.warning("OpenAI client not initialized for music request detection")
        return {"is_music_request": False}

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                    "You are an AI that determines if a message is requesting a song or music. "
                    "If it is, extract the song/artist the user wants to hear."
                },
                {"role": "user", "content": 
                    f"Is this message asking for a song or music? If yes, what song/artist? Message: '{message}'"
                }
            ],
            max_tokens=100,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        if not isinstance(result, dict):
            logger.warning(f"Invalid JSON response from OpenAI: {result}")
            return {"is_music_request": False}

        is_request = result.get("is_music_request", False)
        if isinstance(is_request, str):
            is_request = is_request.lower() in ("yes", "true")
        song_query = result.get("song", "") or result.get("artist", "") or result.get("query", "")
        logger.debug(f"Music request detection for user {user_id}: {result}")
        return {
            "is_music_request": bool(is_request),
            "song_query": song_query if song_query else None
        }
    except Exception as e:
        logger.error(f"Error in is_music_request: {e}", exc_info=True)
        return {"is_music_request": False}

# ==================== TELEGRAM BOT HANDLERS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message."""
    user = update.effective_user
    welcome_msg = (
        f"Hi {user.first_name}! 👋 I'm MelodyMind, your Music Healing Companion.\n\n"
        "I can:\n"
        "🎵 Download music from YouTube\n"
        "📜 Find song lyrics\n"
        "💿 Recommend music based on your mood\n"
        "💬 Chat about music and feelings\n"
        "🔗 Link your Spotify account\n"
        "📖 Create Spotify playlists\n\n"
        "Try /link_spotify or send a YouTube link to start!"
    )
    try:
        await update.message.reply_text(welcome_msg)
        logger.debug(f"Sent welcome message to user {user.id}")
    except Exception as e:
        logger.error(f"Error sending welcome message: {e}", exc_info=True)
        raise

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message."""
    help_text = (
        "🎶 <b>MelodyMind - Music Healing Companion</b> 🎶\n\n"
        "<b>Commands:</b>\n"
        "/start - Welcome message\n"
        "/help - This help message\n"
        "/download [YouTube URL] - Download music from YouTube\n"
        "/autodownload [song name] - Search and download a song\n"
        "/search [song name] - Search for a song\n"
        "/lyrics [song name] or [artist - song] - Get lyrics\n"
        "/recommend - Get music recommendations\n"
        "/mood - Set your current mood\n"
        "/link_spotify - Link your Spotify account\n"
        "/create_playlist [name] - Create a Spotify playlist\n"
        "/clear - Clear your conversation history\n\n"
        "<b>Or just chat with me!</b> Examples:\n"
        "- \"I'm feeling sad, what songs might help?\"\n"
        "- \"Play Shape of You by Ed Sheeran\"\n"
        "- \"Get me the new Taylor Swift song\"\n"
        "- \"Download this song: [YouTube link]\"\n"
        "- \"What are the lyrics to Bohemian Rhapsody?\"\n"
        "- Send a voice message to request a song!"
    )
    try:
        await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)
        logger.debug(f"Sent help message to user {update.effective_user.id}")
    except Exception as e:
        logger.error(f"Error sending help message: {e}", exc_info=True)
        raise

async def download_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download music from YouTube URL."""
    message_text = update.message.text
    user_id = update.effective_user.id
    if context.args:
        url = " ".join(context.args)
    else:
        words = message_text.split()
        urls = [word for word in words if is_valid_youtube_url(word)]
        if urls:
            url = urls[0]
        else:
            try:
                await update.message.reply_text(
                    "❌ Please provide a valid YouTube URL. Example:\n"
                    "/download https://www.youtube.com/watch?v=dQw4w9WgXcQ"
                )
                logger.debug(f"Invalid URL provided by user {user_id}")
            except Exception as e:
                logger.error(f"Error sending invalid URL message: {e}", exc_info=True)
            return

    if not is_valid_youtube_url(url):
        try:
            await update.message.reply_text("❌ Invalid YouTube URL. Please send a valid YouTube link.")
            logger.debug(f"Invalid YouTube URL provided by user {user_id}: {url}")
        except Exception as e:
            logger.error(f"Error sending invalid URL message: {e}", exc_info=True)
        return

    if user_id in active_downloads:
        try:
            await update.message.reply_text("⚠️ You already have a download in progress. Please wait.")
            logger.debug(f"User {user_id} has active download")
        except Exception as e:
            logger.error(f"Error sending active download message: {e}", exc_info=True)
        return

    active_downloads.add(user_id)
    status_msg = None
    try:
        status_msg = await update.message.reply_text("⏳ Starting download...")
        await status_msg.edit_text("🔍 Fetching video information...")
        result = download_youtube_audio(url)
        if not result["success"]:
            await status_msg.edit_text(f"❌ Download failed: {result['error']}")
            logger.error(f"Download failed for user {user_id}: {result['error']}")
            return

        await status_msg.edit_text(f"✅ Downloaded: {result['title']}\n⏳ Sending file...")
        with open(result["audio_path"], 'rb') as audio:
            await update.message.reply_audio(
                audio=audio,
                title=result["title"][:64],
                performer=result["artist"][:64] if result.get("artist") else "Unknown Artist",
                caption=f"🎵 {result['title']}"
            )

        if os.path.exists(result["audio_path"]):
            os.remove(result["audio_path"])
            logger.info(f"Deleted file: {result['audio_path']}")
        await status_msg.delete()
        logger.debug(f"Successfully sent audio to user {user_id}: {result['title']}")
    except Exception as e:
        logger.error(f"Error in download_music for user {user_id}: {e}", exc_info=True)
        if status_msg:
            await status_msg.edit_text(f"❌ Error: {str(e)[:200]}")
    finally:
        active_downloads.discard(user_id)

async def create_playlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Create a private Spotify playlist."""
    user_id = update.effective_user.id
    if not context.args:
        try:
            await update.message.reply_text("Usage: /create_playlist <name>")
            logger.debug(f"User {user_id} did not provide playlist name")
        except Exception as e:
            logger.error(f"Error sending playlist usage message: {e}", exc_info=True)
        return
    name = " ".join(context.args)
    spotify_data = user_contexts.get(user_id, {}).get("spotify", {})
    token = spotify_data.get("access_token")
    if not token:
        token = await refresh_spotify_token(user_id)
        if not token:
            try:
                await update.message.reply_text("Please link your Spotify with /link_spotify first!")
                logger.debug(f"User {user_id} not linked with Spotify")
            except Exception as e:
                logger.error(f"Error sending Spotify link message: {e}", exc_info=True)
            return
    else:
        try:
            token = cipher.decrypt(token).decode()
        except Exception as e:
            logger.error(f"Error decrypting Spotify token for user {user_id}: {e}", exc_info=True)
            try:
                await update.message.reply_text("Error with Spotify credentials. Please try /link_spotify again.")
            except Exception as e:
                logger.error(f"Error sending Spotify credential error message: {e}", exc_info=True)
            return

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    data = {"name": name, "public": False}
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.post("https://api.spotify.com/v1/users/me/playlists", headers=headers, json=data) as response:
                response.raise_for_status()
                await update.message.reply_text(f"Playlist '{name}' created successfully!")
                logger.debug(f"Created Spotify playlist '{name}' for user {user_id}")
    except aiohttp.ClientError as e:
        logger.error(f"Error creating playlist for user {user_id}: {e}", exc_info=True)
        try:
            await update.message.reply_text("Failed to create playlist. Try again later.")
        except Exception as e:
            logger.error(f"Error sending playlist creation error message: {e}", exc_info=True)

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages by transcribing them."""
    if not sr:
        try:
            await update.message.reply_text("Voice message support is not available. Please use text commands.")
            logger.warning("speech_recognition module not installed")
        except Exception as e:
            logger.error(f"Error sending voice support message: {e}", exc_info=True)
        return

    user_id = update.effective_user.id
    file = await context.bot.get_file(update.message.voice.file_id)
    audio_path = os.path.join(DOWNLOAD_DIR, f"voice_{user_id}.ogg")
    try:
        await file.download_to_drive(audio_path)
        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        mood = await detect_mood_from_text(user_id, text)
        user_contexts.setdefault(user_id, {})["mood"] = mood
        logger.debug(f"Transcribed voice message for user {user_id}: {text}, mood: {mood}")
        new_update = Update(update.update_id, message=update.message._replace(text=text))
        await enhanced_handle_message(new_update, context)
    except sr.UnknownValueError:
        try:
            await update.message.reply_text("Sorry, I couldn’t understand your voice message.")
            logger.debug(f"Could not transcribe voice message for user {user_id}")
        except Exception as e:
            logger.error(f"Error sending voice transcription error message: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Error processing voice message for user {user_id}: {e}", exc_info=True)
        try:
            await update.message.reply_text("Error processing your voice message. Try again?")
        except Exception as e:
            logger.error(f"Error sending voice processing error message: {e}", exc_info=True)
    finally:
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.debug(f"Deleted voice file: {audio_path}")
            except Exception as e:
                logger.error(f"Error deleting voice file {audio_path}: {e}", exc_info=True)

async def link_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Initiate Spotify OAuth flow."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI:
        try:
            await update.message.reply_text("Sorry, Spotify linking is not available at the moment.")
            logger.warning("Spotify OAuth credentials not configured")
        except Exception as e:
            logger.error(f"Error sending Spotify unavailable message: {e}", exc_info=True)
        return ConversationHandler.END

    user_id = update.effective_user.id
    auth_url = (
        "https://accounts.spotify.com/authorize"
        f"?client_id={SPOTIFY_CLIENT_ID}"
        "&response_type=code"
        f"&redirect_uri={SPOTIFY_REDIRECT_URI}"
        "&scope=user-read-recently-played%20user-top-read"
        f"&state={user_id}"
    )
    keyboard = [
        [InlineKeyboardButton("🔗 Link Spotify", url=auth_url)],
        [InlineKeyboardButton("Cancel", callback_data="cancel_spotify")]
    ]
    try:
        await update.message.reply_text(
            "Let's link your Spotify account for personalized music! 🎵\n\n"
            "1. Click below to log in to Spotify.\n"
            "2. Authorize, copy the code, and send it here.\n",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.MARKDOWN
        )
        logger.debug(f"Initiated Spotify OAuth for user {user_id}")
        return SPOTIFY_CODE
    except Exception as e:
        logger.error(f"Error sending Spotify link message: {e}", exc_info=True)
        raise

async def spotify_code_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle Spotify authorization code."""
    user_id = update.effective_user.id
    message_text = update.message.text.strip()

    if not message_text.startswith('/'):
        code = message_text
    elif message_text.startswith('/spotify_code') and context.args:
        code = context.args[0]
    else:
        try:
            await update.message.reply_text(
                "Please send the Spotify authorization code. "
                "Paste the code or use /spotify_code <code>."
            )
            logger.debug(f"User {user_id} did not provide Spotify code")
            return SPOTIFY_CODE
        except Exception as e:
            logger.error(f"Error sending Spotify code prompt: {e}", exc_info=True)
            return SPOTIFY_CODE

    token_data = await get_user_spotify_token(user_id, code)
    if not token_data or not token_data.get("access_token"):
        try:
            await update.message.reply_text(
                "❌ Failed to link Spotify. Code might be invalid. Try /link_spotify again."
            )
            logger.debug(f"Invalid Spotify code for user {user_id}")
            return SPOTIFY_CODE
        except Exception as e:
            logger.error(f"Error sending Spotify link failure message: {e}", exc_info=True)
            return SPOTIFY_CODE

    if user_id not in user_contexts:
        user_contexts[user_id] = {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}}
    user_contexts[user_id]["spotify"] = {
        "access_token": cipher.encrypt(token_data.get("access_token").encode()),
        "refresh_token": cipher.encrypt(token_data.get("refresh_token").encode()),
        "expires_at": token_data.get("expires_at")
    }

    recently_played = await get_user_spotify_data(user_id, "player/recently-played")
    if recently_played:
        user_contexts[user_id]["spotify"]["recently_played"] = recently_played

    try:
        await update.message.reply_text(
            "✅ Spotify linked! 🎉 Try /recommend for personalized music!"
        )
        logger.debug(f"Successfully linked Spotify for user {user_id}")
        return ConversationHandler.END
    except Exception as e:
        logger.error(f"Error sending Spotify link success message: {e}", exc_info=True)
        return ConversationHandler.END

async def spotify_code_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /spotify_code command."""
    if not context.args:
        try:
            await update.message.reply_text(
                "Please provide the Spotify code. Example:\n/spotify_code <code>"
            )
            logger.debug(f"User {update.effective_user.id} did not provide Spotify code in command")
        except Exception as e:
            logger.error(f"Error sending Spotify code command prompt: {e}", exc_info=True)
        return
    await spotify_code_handler(update, context)

async def cancel_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel Spotify linking."""
    query = update.callback_query
    try:
        await query.answer()
        await query.message.edit_text("Spotify linking cancelled. Use /link_spotify to try again.")
        logger.debug(f"User {query.from_user.id} cancelled Spotify linking")
        return ConversationHandler.END
    except Exception as e:
        logger.error(f"Error cancelling Spotify linking: {e}", exc_info=True)
        return ConversationHandler.END

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /search command."""
    user_id = update.effective_user.id
    if not context.args:
        try:
            await update.message.reply_text(
                "Please specify a song. Example:\n/search Shape of You Ed Sheeran"
            )
            logger.debug(f"User {user_id} did not provide search query")
            return
        except Exception as e:
            logger.error(f"Error sending search prompt: {e}", exc_info=True)
            return

    query = " ".join(context.args)
    status_msg = None
    try:
        status_msg = await update.message.reply_text(f"🔍 Searching for: '{query}'...")
        results = search_youtube(query)
        await status_msg.delete()
        await send_search_results(update, query, results)
        logger.debug(f"Sent search results for query '{query}' to user {user_id}")
    except Exception as e:
        logger.error(f"Error in search_command for user {user_id}: {e}", exc_info=True)
        if status_msg:
            try:
                await status_msg.edit_text("Failed to search. Please try again.")
            except Exception as e:
                logger.error(f"Error updating search status message: {e}", exc_info=True)

async def auto_download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /autodownload command."""
    user_id = update.effective_user.id
    if not context.args:
        try:
            await update.message.reply_text(
                "Please specify a song. Example:\n/autodownload Shape of You Ed Sheeran"
            )
            logger.debug(f"User {user_id} did not provide autodownload query")
            return
        except Exception as e:
            logger.error(f"Error sending autodownload prompt: {e}", exc_info=True)
            return

    query = " ".join(context.args)
    await auto_download_first_result(update, context, query)

async def get_lyrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle lyrics requests."""
    user_id = update.effective_user.id
    if not context.args:
        try:
            await update.message.reply_text(
                "Please specify a song. Examples:\n"
                "/lyrics Bohemian Rhapsody\n"
                "/lyrics Queen - Bohemian Rhapsody"
            )
            logger.debug(f"User {user_id} did not provide lyrics query")
            return
        except Exception as e:
            logger.error(f"Error sending lyrics prompt: {e}", exc_info=True)
            return

    query = " ".join(context.args)
    status_msg = None
    try:
        status_msg = await update.message.reply_text(f"🔍 Searching for lyrics: {query}")
        artist = None
        song = query
        if " - " in query:
            parts = query.split(" - ", 1)
            artist, song = parts[0].strip(), parts[1].strip()
        elif " by " in query.lower():
            parts = query.lower().split(" by ", 1)
            song, artist = parts[0].strip(), parts[1].strip()

        lyrics = get_lyrics(song, artist)
        if len(lyrics) > 4000:
            await status_msg.edit_text(lyrics[:4000] + "\n\n(Message continues...)")
            remaining = lyrics[4000:]
            while remaining:
                part = remaining[:4000]
                remaining = remaining[4000:]
                await update.message.reply_text(
                    part + ("\n\n(Continued...)" if remaining else "")
                )
        else:
            await status_msg.edit_text(lyrics)
        logger.debug(f"Sent lyrics for '{query}' to user {user_id}")
    except Exception as e:
        logger.error(f"Error in get_lyrics_command for user {user_id}: {e}", exc_info=True)
        if status_msg:
            try:
                await status_msg.edit_text("Sorry, I couldn't find those lyrics.")
            except Exception as e:
                logger.error(f"Error updating lyrics status message: {e}", exc_info=True)

async def recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provide music recommendations."""
    user_id = update.effective_user.id
    context_data = user_contexts.get(user_id, {})

    if not context_data.get("mood"):
        keyboard = [
            [
                InlineKeyboardButton("Happy 😊", callback_data="mood_happy"),
                InlineKeyboardButton("Sad 😢", callback_data="mood_sad"),
            ],
            [
                InlineKeyboardButton("Energetic 💪", callback_data="mood_energetic"),
                InlineKeyboardButton("Relaxed 😌", callback_data="mood_relaxed"),
            ],
            [
                InlineKeyboardButton("Focused 🧠", callback_data="mood_focused"),
                InlineKeyboardButton("Nostalgic 🕰️", callback_data="mood_nostalgic"),
            ],
        ]
        try:
            await update.message.reply_text(
                f"Hi {update.effective_user.first_name}! How are you feeling today?",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            logger.debug(f"Prompted mood selection for user {user_id}")
            return
        except Exception as e:
            logger.error(f"Error sending mood prompt: {e}", exc_info=True)
            return

    mood = context_data["mood"]
    status_msg = None
    try:
        status_msg = await update.message.reply_text(f"🎧 Finding {mood} music for {update.effective_user.first_name}...")
        if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
            await provide_generic_recommendations(update, mood)
            await status_msg.delete()
            logger.debug(f"Provided generic recommendations for user {user_id} due to missing Spotify credentials")
            return

        token = await get_spotify_token()
        if not token:
            await provide_generic_recommendations(update, mood)
            await status_msg.delete()
            logger.debug(f"Provided generic recommendations for user {user_id} due to failed Spotify token")
            return

        seed_query = f"{mood} music"
        if context_data.get("preferences"):
            preferences = context_data["preferences"]
            if preferences:
                seed_query = f"{mood} {preferences[0]} music"

        seed_track = await search_spotify_track(token, seed_query)
        if not seed_track:
            await provide_generic_recommendations(update, mood)
            await status_msg.delete()
            logger.debug(f"Provided generic recommendations for user {user_id} due to no seed track")
            return

        recommendations = await get_spotify_recommendations(token, [seed_track["id"]])
        if not recommendations:
            await provide_generic_recommendations(update, mood)
            await status_msg.delete()
            logger.debug(f"Provided generic recommendations for user {user_id} due to no recommendations")
            return

        response = f"🎵 <b>Recommended {mood} music:</b>\n\n"
        for i, track in enumerate(recommendations[:5], 1):
            artists = ", ".join(a["name"] for a in track["artists"])
            album = track.get("album", {}).get("name", "")
            response += f"{i}. <b>{track['name']}</b> by {artists}"
            if album:
                response += f" (from {album})"
            response += "\n"
        response += "\n💡 <i>Send a YouTube link to download!</i>"
        await status_msg.edit_text(response, parse_mode=ParseMode.HTML)
        logger.debug(f"Sent Spotify recommendations for user {user_id}")
    except Exception as e:
        logger.error(f"Error in recommend_music for user {user_id}: {e}", exc_info=True)
        if status_msg:
            try:
                await status_msg.edit_text("I couldn't get recommendations. Try again later?")
            except Exception as e:
                logger.error(f"Error updating recommendation status message: {e}", exc_info=True)

async def provide_generic_recommendations(update: Update, mood: str) -> None:
    """Provide generic recommendations."""
    mood_recommendations = {
        "happy": [
            "Walking on Sunshine - Katrina & The Waves",
            "Happy - Pharrell Williams",
            "Can't Stop the Feeling - Justin Timberlake",
            "Uptown Funk - Mark Ronson ft. Bruno Mars",
            "Good Vibrations - The Beach Boys"
        ],
        "sad": [
            "Someone Like You - Adele",
            "Fix You - Coldplay",
            "Everybody Hurts - R.E.M.",
            "Nothing Compares 2 U - Sinéad O'Connor",
            "Tears in Heaven - Eric Clapton"
        ],
        "energetic": [
            "Eye of the Tiger - Survivor",
            "Don't Stop Me Now - Queen",
            "Thunderstruck - AC/DC",
            "Stronger - Kanye West",
            "Shake It Off - Taylor Swift"
        ],
        "relaxed": [
            "Weightless - Marconi Union",
            "Clair de Lune - Claude Debussy",
            "Watermark - Enya",
            "Breathe - Pink Floyd",
            "Gymnopedie No.1 - Erik Satie"
        ],
        "focused": [
            "The Four Seasons - Vivaldi",
            "Time - Hans Zimmer",
            "Intro - The xx",
            "Brain Waves - Alpha Waves",
            "Experience - Ludovico Einaudi"
        ],
        "nostalgic": [
            "Yesterday - The Beatles",
            "Vivalada - Coldplay",
            "Landslide - Fleetwood Mac",
            "Vienna - Billy Joel",
            "Time After Time - Cyndi Lauper"
        ]
    }

    recommendations = mood_recommendations.get(mood.lower(), mood_recommendations["happy"])
    response = f"🎵 <b>Recommended {mood} music:</b>\n\n"
    for i, track in enumerate(recommendations, 1):
        response += f"{i}. {track}\n"
    response += "\n💡 <i>Send a YouTube link to download!</i>"
    try:
        await update.message.reply_text(response, parse_mode=ParseMode.HTML)
        logger.debug(f"Sent generic recommendations for mood {mood} to user {update.effective_user.id}")
    except Exception as e:
        logger.error(f"Error sending generic recommendations: {e}", exc_info=True)

async def set_mood(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start conversation to set mood."""
    user_id = update.effective_user.id
    keyboard = [
        [
            InlineKeyboardButton("Happy 😊", callback_data="mood_happy"),
            InlineKeyboardButton("Sad 😢", callback_data="mood_sad"),
        ],
        [
            InlineKeyboardButton("Energetic 💪", callback_data="mood_energetic"),
            InlineKeyboardButton("Relaxed 😌", callback_data="mood_relaxed"),
        ],
        [
            InlineKeyboardButton("Focused 🧠", callback_data="mood_focused"),
            InlineKeyboardButton("Nostalgic 🕰️", callback_data="mood_nostalgic"),
        ],
    ]
    try:
        await update.message.reply_text(
            f"Hi {update.effective_user.first_name}! How are you feeling today?",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        logger.debug(f"Prompted mood selection for user {user_id}")
        return MOOD
    except Exception as e:
        logger.error(f"Error sending mood selection prompt: {e}", exc_info=True)
        return ConversationHandler.END

async def send_search_results(update: Update, query: str, results: List[Dict]) -> None:
    """Send search results with inline keyboard."""
    user_id = update.effective_user.id
    if not results:
        try:
            await update.message.reply_text(f"Sorry, I couldn't find any songs for '{query}'.")
            logger.debug(f"No search results for query '{query}' for user {user_id}")
            return
        except Exception as e:
            logger.error(f"Error sending no results message: {e}", exc_info=True)
            return

    keyboard = []
    for i, result in enumerate(results):
        duration_str = ""
        if result.get('duration'):
            minutes = result['duration'] // 60
            seconds = result['duration'] % 60
            duration_str = f" [{minutes}:{seconds:02d}]"

        title = result['title']
        if len(title) > 40:
            title = title[:37] + "..."

        button_text = f"{i+1}. {title}{duration_str}"
        keyboard.append([InlineKeyboardButton(button_text, callback_data=f"download_{result['id']}")])

    keyboard.append([InlineKeyboardButton("Cancel", callback_data="cancel_search")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    try:
        await update.message.reply_text(
            f"🔎 Search results for '{query}':\n\nClick a song to download:",
            reply_markup=reply_markup
        )
        logger.debug(f"Sent search results for '{query}' to user {user_id}")
    except Exception as e:
        logger.error(f"Error sending search results: {e}", exc_info=True)

async def auto_download_first_result(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str) -> None:
    """Automatically download the first song result."""
    user_id = update.effective_user.id
    if user_id in active_downloads:
        try:
            await update.message.reply_text("⚠️ You already have a download in progress. Please wait.")
            logger.debug(f"User {user_id} has active download")
            return
        except Exception as e:
            logger.error(f"Error sending active download message: {e}", exc_info=True)
            return

    active_downloads.add(user_id)
    status_msg = None
    try:
        status_msg = await update.message.reply_text(f"🔍 Searching for '{query}'...")
        results = search_youtube(query, max_results=1)
        if not results:
            await status_msg.edit_text(f"❌ Couldn't find any results for '{query}'.")
            logger.debug(f"No results for autodownload query '{query}' for user {user_id}")
            return

        result = results[0]
        video_url = result["url"]
        await status_msg.edit_text(f"✅ Found: {result['title']}\n⏳ Downloading...")

        download_result = download_youtube_audio(video_url)
        if not download_result["success"]:
            await status_msg.edit_text(f"❌ Download failed: {download_result['error']}")
            logger.error(f"Autodownload failed for user {user_id}: {download_result['error']}")
            return

        await status_msg.edit_text(f"✅ Downloaded: {download_result['title']}\n⏳ Sending file...")
        with open(download_result["audio_path"], 'rb') as audio:
            await update.message.reply_audio(
                audio=audio,
                title=download_result["title"][:64],
                performer=download_result["artist"][:64] if download_result.get("artist") else "Unknown Artist",
                caption=f"🎵 {download_result['title']}"
            )

        if os.path.exists(download_result["audio_path"]):
            os.remove(download_result["audio_path"])
            logger.info(f"Deleted file: {download_result['audio_path']}")

        await status_msg.delete()
        logger.debug(f"Successfully autodownloaded '{download_result['title']}' for user {user_id}")
    except Exception as e:
        logger.error(f"Error in auto_download_first_result for user {user_id}: {e}", exc_info=True)
        if status_msg:
            try:
                await status_msg.edit_text(f"❌ Error: {str(e)[:200]}")
            except Exception as e:
                logger.error(f"Error updating autodownload status message: {e}", exc_info=True)
    finally:
        active_downloads.discard(user_id)

async def enhanced_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
    """Handle button callbacks."""
    query = update.callback_query
    user_id = query.from_user.id
    try:
        await query.answer()
    except Exception as e:
        logger.error(f"Error answering callback query for user {user_id}: {e}", exc_info=True)

    data = query.data
    logger.debug(f"Handling callback query: {data} for user {user_id}")

    try:
        if data.startswith("mood_"):
            mood = data.split("_")[1]
            if user_id not in user_contexts:
                user_contexts[user_id] = {"mood": mood, "preferences": [], "conversation_history": [], "spotify": {}}
            else:
                user_contexts[user_id]["mood"] = mood

            keyboard = [
                [
                    InlineKeyboardButton("Pop", callback_data="pref_pop"),
                    InlineKeyboardButton("Rock", callback_data="pref_rock"),
                    InlineKeyboardButton("Hip-Hop", callback_data="pref_hiphop"),
                ],
                [
                    InlineKeyboardButton("Classical", callback_data="pref_classical"),
                    InlineKeyboardButton("Electronic", callback_data="pref_electronic"),
                    InlineKeyboardButton("Jazz", callback_data="pref_jazz"),
                ],
                [
                    InlineKeyboardButton("Skip", callback_data="pref_skip"),
                ],
            ]
            await query.edit_message_text(
                f"Got it, {query.from_user.first_name}! You're feeling {mood}. 🎶\n\nAny genre preference?",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            logger.debug(f"Set mood {mood} for user {user_id}")
            return PREFERENCE

        elif data.startswith("pref_"):
            preference = data.split("_")[1]
            if user_id in user_contexts and preference != "skip":
                user_contexts[user_id]["preferences"] = [preference]
            await query.edit_message_text(
                "Great! Try:\n"
                "/recommend - Music recommendations\n"
                "/download - Download songs\n"
                "/lyrics - Find lyrics\n"
                "Or chat about music!"
            )
            logger.debug(f"Set preference {preference} for user {user_id}")
            return ConversationHandler.END

        elif data.startswith("download_") or data.startswith("auto_download_"):
            video_id = data.split("_")[2] if data.startswith("auto_download_") else data.split("_")[1]
            if not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
                logger.error(f"Invalid YouTube video ID: {video_id}")
                await query.edit_message_text("❌ Invalid video ID. Try another song.")
                return None
            url = f"https://www.youtube.com/watch?v={video_id}"
            await query.edit_message_text(f"⏳ Starting download...")

            if user_id in active_downloads:
                await query.edit_message_text("⚠️ You already have a download in progress. Please wait.")
                logger.debug(f"User {user_id} has active download")
                return None

            active_downloads.add(user_id)
            try:
                await query.edit_message_text("⏳ 50% done downloading...")
                result = download_youtube_audio(url)
                if not result["success"]:
                    await query.edit_message_text(f"❌ Download failed: {result['error']}")
                    logger.error(f"Download failed for user {user_id}: {result['error']}")
                    return None

                await query.edit_message_text(f"✅ Downloaded: {result['title']}\n⏳ Sending file...")
                logger.info(f"Sending audio file: {result['audio_path']}, size: {os.path.getsize(result['audio_path'])/(1024*1024):.2f} MB")

                @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
                async def send_audio_with_retry():
                    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=60)) as session:
                        context.bot.session = session
                        try:
                            return await context.bot.send_audio(
                                chat_id=query.message.chat_id,
                                audio=open(result["audio_path"], 'rb'),
                                title=result["title"][:64],
                                performer=result["artist"][:64] if result.get("artist") else "Unknown Artist",
                                caption=f"🎵 {result['title']}"
                            )
                        finally:
                            context.bot.session = None

                try:
                    await send_audio_with_retry()
                except TimedOut as e:
                    logger.error(f"Timeout sending audio for user {user_id}: {e}", exc_info=True)
                    await query.edit_message_text("❌ Failed to send audio due to timeout.")
                    return None
                except NetworkError as e:
                    logger.error(f"Network error sending audio for user {user_id}: {e}", exc_info=True)
                    await query.edit_message_text("❌ Failed to send audio due to network error.")
                    return None

                if os.path.exists(result["audio_path"]):
                    os.remove(result["audio_path"])
                    logger.info(f"Deleted file: {result['audio_path']}")

                await query.edit_message_text(f"✅ Download complete: {result['title']}")
                logger.debug(f"Successfully downloaded and sent '{result['title']}' to user {user_id}")
            finally:
                active_downloads.discard(user_id)
            return None

        elif data.startswith("show_options_"):
            search_query = data.split("show_options_")[1]
            results = search_youtube(search_query)
            if not results:
                await query.edit_message_text(f"Sorry, I couldn't find any songs for '{search_query}'.")
                logger.debug(f"No search results for '{search_query}' for user {user_id}")
                return None

            keyboard = []
            for i, result in enumerate(results[:5]):
                if not result.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', result['id']):
                    logger.warning(f"Skipping invalid YouTube result ID: {result.get('id', 'No ID')}")
                    continue
                duration_str = ""
                if result.get('duration'):
                    minutes = result['duration'] // 60
                    seconds = result['duration'] % 60
                    duration_str = f" [{minutes}:{seconds:02d}]"
                title = result['title']
                if len(title) > 40:
                    title = title[:37] + "..."
                button_text = f"{i+1}. {title}{duration_str}"
                keyboard.append([InlineKeyboardButton(button_text, callback_data=f"download_{result['id']}")])
            if not keyboard:
                await query.edit_message_text(f"Sorry, I couldn't find valid songs for '{search_query}'.")
                logger.debug(f"No valid search results for '{search_query}' for user {user_id}")
                return None
            keyboard.append([InlineKeyboardButton("Cancel", callback_data="cancel_search")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            await query.edit_message_text(
                f"🔎 Search results for '{search_query}':\n\nClick a song to download:",
                reply_markup=reply_markup
            )
            logger.debug(f"Sent search options for '{search_query}' to user {user_id}")
            return None

        elif data == "cancel_search":
            await query.edit_message_text("❌ Search cancelled.")
            logger.debug(f"User {user_id} cancelled search")
            return None

        elif data == "cancel_spotify":
            await query.edit_message_text("❌ Spotify linking cancelled.")
            logger.debug(f"User {user_id} cancelled Spotify linking")
            return ConversationHandler.END

    except Exception as e:
        logger.error(f"Error in enhanced_button_handler for user {user_id}: {e}", exc_info=True)
        try:
            await query.edit_message_text("Sorry, an error occurred. Please try again.")
        except Exception as e:
            logger.error(f"Error sending button handler error message: {e}", exc_info=True)
        return None

async def enhanced_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enhanced message handler with music detection."""
    user_id = update.effective_user.id
    text = update.message.text
    logger.debug(f"Processing message from user {user_id}: {text}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def reply_with_retry(text, reply_markup=None):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            context.bot.session = session
            try:
                return await update.message.reply_text(text, reply_markup=reply_markup)
            finally:
                context.bot.session = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def delete_message_with_retry(message):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            context.bot.session = session
            try:
                return await message.delete()
            finally:
                context.bot.session = None

    try:
        if is_valid_youtube_url(text):
            context.args = []
            await download_music(update, context)
            return

        mood = await detect_mood_from_text(user_id, text)
        user_contexts.setdefault(user_id, {})["mood"] = mood

        detected_song = detect_music_in_message(text)
        if detected_song:
            if detected_song == "AI_ANALYSIS_NEEDED":
                ai_response = await is_music_request(user_id, text)
                if ai_response.get("is_music_request") and ai_response.get("song_query"):
                    detected_song = ai_response.get("song_query")
                else:
                    detected_song = None

            if detected_song:
                status_msg = await reply_with_retry(f"🔍 Searching for: '{detected_song}'...")
                results = search_youtube(detected_song)
                await delete_message_with_retry(status_msg)
                if not results:
                    await reply_with_retry(f"Sorry, I couldn't find any songs for '{detected_song}'.")
                    logger.debug(f"No results for song '{detected_song}' for user {user_id}")
                    return
                if not results[0].get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', results[0]['id']):
                    logger.warning(f"Invalid YouTube video ID: {results[0].get('id', 'No ID')}")
                    await reply_with_retry(f"Sorry, I couldn't find valid songs for '{detected_song}'.")
                    return
                keyboard = [
                    [InlineKeyboardButton("✅ Yes, download it", callback_data=f"auto_download_{results[0]['id']}")],
                    [InlineKeyboardButton("👀 Show me options", callback_data=f"show_options_{detected_song}")],
                    [InlineKeyboardButton("❌ No, cancel", callback_data="cancel_search")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                message = (
                    f"I found '{results[0]['title']}' by {results[0]['uploader']}.\n\n"
                    f"Would you like me to download this for you, {update.effective_user.first_name}?"
                )
                await reply_with_retry(message, reply_markup=reply_markup)
                logger.debug(f"Offered download for '{detected_song}' to user {user_id}")
                return

        lower_text = text.lower()
        if ("download" in lower_text or "get this song" in lower_text) and any(domain in lower_text for domain in ['youtube.com', 'youtu.be']):
            urls = [word for word in text.split() if is_valid_youtube_url(word)]
            if urls:
                context.args = [urls[0]]
                await download_music(update, context)
                return

        if any(phrase in lower_text for phrase in ["lyrics", "words to", "what's the song that goes"]):
            song_query = text
            for phrase in ["lyrics", "words to", "what's the song that goes"]:
                song_query = song_query.replace(phrase, "")
            context.args = [song_query.strip()]
            await get_lyrics_command(update, context)
            return

        if any(word in lower_text for word in ["song", "music", "track", "listen", "audio"]):
            response = await is_music_request(user_id, text)
            if response.get("is_music_request") and response.get("song_query"):
                status_msg = await reply_with_retry(f"🔍 Searching for: '{response['song_query']}'...")
                results = search_youtube(response['song_query'])
                await delete_message_with_retry(status_msg)
                if not results:
                    await reply_with_retry(f"Sorry, I couldn't find any songs for '{response['song_query']}'.")
                    logger.debug(f"No results for song '{response['song_query']}' for user {user_id}")
                    return
                if not results[0].get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', results[0]['id']):
                    logger.warning(f"Invalid YouTube video ID: {results[0].get('id', 'No ID')}")
                    await reply_with_retry(f"Sorry, I couldn't find valid songs for '{response['song_query']}'.")
                    return
                keyboard = [
                    [InlineKeyboardButton("✅ Yes, download it", callback_data=f"auto_download_{results[0]['id']}")],
                    [InlineKeyboardButton("👀 Show me options", callback_data=f"show_options_{response['song_query']}")],
                    [InlineKeyboardButton("❌ No, cancel", callback_data="cancel_search")]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                message = (
                    f"I found '{results[0]['title']}' by {results[0]['uploader']}.\n\n"
                    f"Would you like me to download this for you, {update.effective_user.first_name}?"
                )
                await reply_with_retry(message, reply_markup=reply_markup)
                logger.debug(f"Offered download for '{response['song_query']}' to user {user_id}")
                return

        if "i'm feeling" in lower_text or "i feel" in lower_text:
            text_after_feeling = ""
            if "i'm feeling" in lower_text:
                text_after_feeling = lower_text.split("i'm feeling")[1].strip()
            elif "i feel" in lower_text:
                text_after_feeling = lower_text.split("i feel")[1].strip()
            if text_after_feeling:
                mood = text_after_feeling.split()[0].strip('.,!?')
                user_contexts.setdefault(user_id, {})["mood"] = mood
                logger.debug(f"Set mood for user {user_id}: {mood}")

        typing_msg = await reply_with_retry("🎵 Thinking about music...")
        try:
            response = await generate_chat_response(user_id, text)
            await reply_with_retry(response)
            logger.debug(f"Sent chat response to user {user_id}: {response}")
        except Exception as e:
            logger.error(f"Error generating chat response for user {user_id}: {e}", exc_info=True)
            await reply_with_retry("I'm having trouble responding. Try again?")
        finally:
            await delete_message_with_retry(typing_msg)

    except (TimedOut, NetworkError) as e:
        logger.error(f"Network error in handle_message for user {user_id}: {e}", exc_info=True)
        try:
            await reply_with_retry("Sorry, I'm having network issues. Please try again.")
        except Exception as e:
            logger.error(f"Error sending network error message: {e}", exc_info=True)

def detect_music_in_message(text: str) -> Optional[str]:
    """Detect if a message is asking for music."""
    patterns = [
        r'play (.*?)(?:by|from|$)', r'find (.*?)(?:by|from|$)',
        r'download (.*?)(?:by|from|$)', r'get (.*?)(?:by|from|$)',
        r'send me (.*?)(?:by|from|$)', r'i want to listen to (.*?)(?:by|from|$)',
        r'can you get (.*?)(?:by|from|$)', r'i need (.*?)(?:by|from|$)',
        r'find me (.*?)(?:by|from|$)', r'fetch (.*?)(?:by|from|$)',
        r'give me (.*?)(?:by|from|$)', r'send (.*?)(?:by|from|$)',
        r'song (.*?)(?:by|from|$)'
    ]
    keywords = ['music', 'song', 'track', 'tune', 'audio']

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            song_title = match.group(1).strip()
            artist_match = re.search(r'by (.*?)(?:from|$)', text, re.IGNORECASE)
            if artist_match:
                artist = artist_match.group(1).strip()
                return f"{song_title} {artist}"
            return song_title

    if any(keyword in text.lower() for keyword in keywords):
        return "AI_ANALYSIS_NEEDED"
    return None

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear user conversation history."""
    user_id = update.effective_user.id
    if user_id in user_contexts:
        user_contexts[user_id]["conversation_history"] = []
        try:
            await update.message.reply_text("✅ Your conversation history has been cleared.")
            logger.debug(f"Cleared conversation history for user {user_id}")
        except Exception as e:
            logger.error(f"Error sending clear history message: {e}", exc_info=True)
    else:
        try:
            await update.message.reply_text("You don't have any saved conversation history.")
            logger.debug(f"No conversation history for user {user_id}")
        except Exception as e:
            logger.error(f"Error sending no history message: {e}", exc_info=True)

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the conversation."""
    try:
        await update.message.reply_text("No problem! Feel free to chat or use commands anytime.")
        logger.debug(f"User {update.effective_user.id} cancelled conversation")
        return ConversationHandler.END
    except Exception as e:
        logger.error(f"Error sending cancel message: {e}", exc_info=True)
        return ConversationHandler.END

async def analyze_conversation(user_id: int) -> Dict:
    """Analyze conversation history and Spotify data to extract user preferences."""
    if not client:
        logger.warning("OpenAI client not initialized for conversation analysis")
        return {"genres": [], "artists": [], "mood": None}

    context = user_contexts.get(user_id, {
        "mood": None,
        "preferences": [],
        "conversation_history": [],
        "spotify": {}
    })

    if len(context.get("conversation_history", [])) < 4 and not context.get("spotify"):
        return {
            "genres": context.get("preferences", []),
            "artists": [],
            "mood": context.get("mood")
        }

    try:
        conversation_text = ""
        if context.get("conversation_history"):
            history = context["conversation_history"][-20:]
            conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

        spotify_data = ""
        if context.get("spotify", {}).get("recently_played"):
            tracks = context["spotify"]["recently_played"]
            spotify_data = "Recently played: " + ", ".join(
                [f"{item['track']['name']} by {item['track']['artists'][0]['name']}" for item in tracks[:5]]
            )
        elif context.get("spotify", {}).get("top_tracks"):
            tracks = context["spotify"]["top_tracks"]
            spotify_data = "Top tracks: " + ", ".join(
                [f"{item['name']} by {item['artists'][0]['name']}" for item in tracks[:5]]
            )

        prompt = (
            "Analyze the following conversation history and Spotify data to identify the user's music preferences, "
            "including genres, favorite artists, and current mood. Return a JSON object with 'genres' (list), "
            "'artists' (list), and 'mood' (string). If no specific data is available, use the existing mood and preferences.\n\n"
            f"Conversation History:\n{conversation_text}\n\n"
            f"Spotify Data:\n{spotify_data}\n\n"
            "Provide a concise analysis."
        )

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.5,
            response_format={"type": "json_object"}
        )

        analysis = json.loads(response.choices[0].message.content)
        if not isinstance(analysis, dict):
            logger.warning(f"Invalid analysis response for user {user_id}: {analysis}")
            return {
                "genres": context.get("preferences", []),
                "artists": [],
                "mood": context.get("mood", "happy")
            }

        genres = analysis.get("genres", context.get("preferences", []))
        artists = analysis.get("artists", [])
        mood = analysis.get("mood", context.get("mood", "happy"))
        logger.debug(f"Conversation analysis for user {user_id}: genres={genres}, artists={artists}, mood={mood}")

        # Update user context with analyzed data
        user_contexts[user_id]["preferences"] = genres
        user_contexts[user_id]["mood"] = mood
        return {"genres": genres, "artists": artists, "mood": mood}

    except Exception as e:
        logger.error(f"Error analyzing conversation for user {user_id}: {e}", exc_info=True)
        return {
            "genres": context.get("preferences", []),
            "artists": [],
            "mood": context.get("mood", "happy")
        }

# ==================== CLEANUP AND SHUTDOWN ====================

def cleanup_downloads():
    """Clean up downloaded files on bot shutdown."""
    try:
        for file in os.listdir(DOWNLOAD_DIR):
            file_path = os.path.join(DOWNLOAD_DIR, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file during cleanup: {file_path}")
        logger.debug("Cleanup of download directory completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}", exc_info=True)

def handle_shutdown(signum, frame):
    """Handle bot shutdown gracefully."""
    logger.info("Received shutdown signal, cleaning up...")
    cleanup_downloads()
    sys.exit(0)



@contextmanager
def get_db_connection():
    conn = sqlite3.connect("user_contexts.db")
    try:
        yield conn
    finally:
        conn.close()

def save_user_context(user_id: int, context: Dict):
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute(
            "INSERT OR REPLACE INTO user_contexts (user_id, context) VALUES (?, ?)",
            (user_id, json.dumps(context))
        )
        conn.commit()

def load_user_context(user_id: int) -> Dict:
    with get_db_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT context FROM user_contexts WHERE user_id = ?", (user_id,))
        result = c.fetchone()
        return json.loads(result[0]) if result else {}

# Initialize database
with get_db_connection() as conn:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS user_contexts (user_id INTEGER PRIMARY KEY, context TEXT)"
    )



# ==================== MAIN APPLICATION SETUP ====================

def main():
    """Set up and run the Telegram bot."""
    if not TOKEN:
        logger.error("No TELEGRAM_TOKEN provided in environment variables")
        sys.exit(1)

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    atexit.register(cleanup_downloads)

    # Initialize the Telegram application
    application = Application.builder().token(TOKEN).build()

    # Define conversation handler for mood and preferences
    conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("mood", set_mood),
            CommandHandler("link_spotify", link_spotify)
        ],
        states={
            MOOD: [CallbackQueryHandler(enhanced_button_handler, pattern="^mood_.*$")],
            PREFERENCE: [CallbackQueryHandler(enhanced_button_handler, pattern="^pref_.*$")],
            SPOTIFY_CODE: [
                CommandHandler("spotify_code", spotify_code_command),
                MessageHandler(filters.TEXT & ~filters.COMMAND, spotify_code_handler),
                CallbackQueryHandler(cancel_spotify, pattern="^cancel_spotify$")
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )

    # Add handlers to the application
    application.add_handler(conv_handler)
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("download", download_music))
    application.add_handler(CommandHandler("autodownload", auto_download_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("lyrics", get_lyrics_command))
    application.add_handler(CommandHandler("recommend", recommend_music))
    application.add_handler(CommandHandler("create_playlist", create_playlist))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, enhanced_handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(CallbackQueryHandler(enhanced_button_handler))

    # Start the bot
    logger.info("Starting Enhanced MelodyMind Bot with Spotify Integration")
    application.run_polling()

if __name__ == "__main__":
    main()