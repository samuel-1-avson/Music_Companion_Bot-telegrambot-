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
import speech_recognition as sr
from functools import lru_cache

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "https://your-callback-url.com")

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
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
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=data) as response:
                response.raise_for_status()
                return (await response.json()).get("access_token")
    except aiohttp.ClientError as e:
        logger.error(f"Error getting Spotify token: {e}")
        return None

@lru_cache(maxsize=100)
async def search_spotify_track(token: str, query: str) -> Optional[Dict]:
    """Search for a track on Spotify."""
    if not token:
        return None

    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "type": "track", "limit": 1}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                items = (await response.json()).get("tracks", {}).get("items", [])
                return items[0] if items else None
    except aiohttp.ClientError as e:
        logger.error(f"Error searching Spotify track: {e}")
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
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                return (await response.json()).get("tracks", [])
    except aiohttp.ClientError as e:
        logger.error(f"Error getting Spotify recommendations: {e}")
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
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=data) as response:
                response.raise_for_status()
                token_data = await response.json()
                token_data["expires_at"] = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600))).timestamp()
                return token_data
    except aiohttp.ClientError as e:
        logger.error(f"Error getting user Spotify token: {e}")
        return None

async def refresh_spotify_token(user_id: int) -> Optional[str]:
    """Refresh Spotify access token using refresh token."""
    context = user_contexts.get(user_id, {})
    refresh_token = context.get("spotify", {}).get("refresh_token")
    if not refresh_token:
        return None

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}'.encode()).decode()}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "refresh_token", "refresh_token": cipher.decrypt(refresh_token).decode()}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=data) as response:
                response.raise_for_status()
                token_data = await response.json()
                expires_at = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600))).timestamp()
                user_contexts[user_id]["spotify"] = {
                    "access_token": cipher.encrypt(token_data.get("access_token").encode()),
                    "refresh_token": cipher.encrypt(token_data.get("refresh_token", refresh_token).encode()),
                    "expires_at": expires_at
                }
                return token_data.get("access_token")
    except aiohttp.ClientError as e:
        logger.error(f"Error refreshing Spotify token: {e}")
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
            return None
    else:
        access_token = cipher.decrypt(access_token).decode()

    url = f"https://api.spotify.com/v1/me/{endpoint}"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"limit": 10}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                return (await response.json()).get("items", [])
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching Spotify user data ({endpoint}): {e}")
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
                return {"success": False, "error": "Could not extract video information"}
            title = sanitize_filename(info.get('title', 'Unknown Title'))
            artist = info.get('artist', info.get('uploader', 'Unknown Artist'))
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
                return {"success": False, "error": "Downloaded file not found or inaccessible"}
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            if file_size_mb > 50:
                os.remove(audio_path)
                return {"success": False, "error": "File exceeds 50 MB Telegram limit"}
            return {
                "success": True,
                "title": title,
                "artist": artist,
                "thumbnail_url": info.get('thumbnail', ''),
                "duration": info.get('duration', 0),
                "audio_path": audio_path
            }
    except Exception as e:
        logger.error(f"Error downloading YouTube audio: {e}")
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
                return []
            return [
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
    except Exception as e:
        logger.error(f"Error searching YouTube: {e}")
        return []

# ==================== AI AND LYRICS FUNCTIONS ====================

async def generate_chat_response(user_id: int, message: str) -> str:
    """Generate a conversational response using OpenAI."""
    if not client:
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
        return reply
    except Exception as e:
        logger.error(f"Error generating chat response: {e}")
        return "I'm having trouble thinking right now. Let's talk about music instead!"

def get_lyrics(song_title: str, artist: Optional[str] = None) -> str:
    """Get lyrics for a song using Genius API with fallback."""
    if not genius:
        return "Lyrics service unavailable. Try asking me for a song instead!"
    try:
        song = genius.search_song(song_title, artist) if artist else genius.search_song(song_title)
        if not song:
            return f"Couldn't find lyrics for '{song_title}'" + (f" by {artist}" if artist else "") + ". Try another song!"
        lyrics = song.lyrics
        lyrics = re.sub(r'\[.*?\]', '', lyrics)
        lyrics = re.sub(r'\d+Embed$', '', lyrics)
        lyrics = re.sub(r'Embed$', '', lyrics)
        header = f"🎵 {song.title} by {song.artist} 🎵\n\n"
        return header + lyrics.strip()
    except Exception as e:
        logger.error(f"Error fetching lyrics: {e}")
        return f"Couldn't find lyrics for '{song_title}'" + (f" by {artist}" if artist else "") + ". Try another song!"

async def detect_mood_from_text(user_id: int, text: str) -> str:
    """Detect mood from user's message using AI."""
    if not client:
        return user_contexts.get(user_id, {}).get("mood", "happy")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Detect mood from this text: '{text}'"}],
            max_tokens=50
        )
        mood = response.choices[0].message.content.lower().strip()
        return mood if mood else "happy"
    except Exception as e:
        logger.error(f"Error detecting mood: {e}")
        return "happy"

async def is_music_request(user_id: int, message: str) -> Dict:
    """Use AI to determine if a message is a music/song request."""
    if not client:
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
            return {"is_music_request": False}

        is_request = result.get("is_music_request", False)
        if isinstance(is_request, str):
            is_request = is_request.lower() in ("yes", "true")
        song_query = result.get("song", "") or result.get("artist", "") or result.get("query", "")
        return {
            "is_music_request": bool(is_request),
            "song_query": song_query if song_query else None
        }
    except Exception as e:
        logger.error(f"Error in is_music_request: {e}")
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
    await update.message.reply_text(welcome_msg)

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
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

async def download_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download music from YouTube URL."""
    message_text = update.message.text
    if context.args:
        url = " ".join(context.args)
    else:
        words = message_text.split()
        urls = [word for word in words if is_valid_youtube_url(word)]
        if urls:
            url = urls[0]
        else:
            await update.message.reply_text(
                "❌ Please provide a valid YouTube URL. Example:\n"
                "/download https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            )
            return

    if not is_valid_youtube_url(url):
        await update.message.reply_text("❌ Invalid YouTube URL. Please send a valid YouTube link.")
        return

    user_id = update.effective_user.id
    if user_id in active_downloads:
        await update.message.reply_text("⚠️ You already have a download in progress. Please wait.")
        return

    active_downloads.add(user_id)
    status_msg = await update.message.reply_text("⏳ Starting download...")

    try:
        await status_msg.edit_text("🔍 Fetching video information...")
        result = download_youtube_audio(url)
        if not result["success"]:
            await status_msg.edit_text(f"❌ Download failed: {result['error']}")
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
    except Exception as e:
        logger.error(f"Error in download_music: {e}")
        await status_msg.edit_text(f"❌ Error: {str(e)[:200]}")
    finally:
        active_downloads.discard(user_id)

async def create_playlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Create a private Spotify playlist."""
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Usage: /create_playlist <name>")
        return
    name = " ".join(context.args)
    spotify_data = user_contexts.get(user_id, {}).get("spotify", {})
    token = spotify_data.get("access_token")
    if not token:
        token = await refresh_spotify_token(user_id)
        if not token:
            await update.message.reply_text("Please link your Spotify with /link_spotify first!")
            return
    else:
        token = cipher.decrypt(token).decode()

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    data = {"name": name, "public": False}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post("https://api.spotify.com/v1/users/me/playlists", headers=headers, json=data) as response:
                response.raise_for_status()
                await update.message.reply_text(f"Playlist '{name}' created successfully!")
    except aiohttp.ClientError as e:
        logger.error(f"Error creating playlist: {e}")
        await update.message.reply_text("Failed to create playlist. Try again later.")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages by transcribing them."""
    user_id = update.effective_user.id
    file = await context.bot.get_file(update.message.voice.file_id)
    audio_path = os.path.join(DOWNLOAD_DIR, f"voice_{user_id}.ogg")
    await file.download_to_drive(audio_path)
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(audio_path) as source:
            audio = recognizer.record(source)
        text = recognizer.recognize_google(audio)
        mood = await detect_mood_from_text(user_id, text)
        user_contexts.setdefault(user_id, {})["mood"] = mood
        new_update = Update(update.update_id, message=update.message._replace(text=text))
        await enhanced_handle_message(new_update, context)
    except sr.UnknownValueError:
        await update.message.reply_text("Sorry, I couldn’t understand your voice message.")
    except Exception as e:
        logger.error(f"Error processing voice message: {e}")
        await update.message.reply_text("Error processing your voice message. Try again?")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

async def link_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Initiate Spotify OAuth flow."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI:
        await update.message.reply_text("Sorry, Spotify linking is not available at the moment.")
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
    await update.message.reply_text(
        "Let's link your Spotify account for personalized music! 🎵\n\n"
        "1. Click below to log in to Spotify.\n"
        "2. Authorize, copy the code, and send it here.\n",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN
    )
    return SPOTIFY_CODE

async def spotify_code_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle Spotify authorization code."""
    user_id = update.effective_user.id
    message_text = update.message.text.strip()

    if not message_text.startswith('/'):
        code = message_text
    elif message_text.startswith('/spotify_code') and context.args:
        code = context.args[0]
    else:
        await update.message.reply_text(
            "Please send the Spotify authorization code. "
            "Paste the code or use /spotify_code <code>."
        )
        return SPOTIFY_CODE

    token_data = await get_user_spotify_token(user_id, code)
    if not token_data or not token_data.get("access_token"):
        await update.message.reply_text(
            "❌ Failed to link Spotify. Code might be invalid. Try /link_spotify again."
        )
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

    await update.message.reply_text(
        "✅ Spotify linked! 🎉 Try /recommend for personalized music!"
    )
    return ConversationHandler.END

async def spotify_code_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /spotify_code command."""
    if not context.args:
        await update.message.reply_text(
            "Please provide the Spotify code. Example:\n/spotify_code <code>"
        )
        return
    await spotify_code_handler(update, context)

async def cancel_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel Spotify linking."""
    query = update.callback_query
    await query.answer()
    await query.message.edit_text("Spotify linking cancelled. Use /link_spotify to try again.")
    return ConversationHandler.END

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /search command."""
    if not context.args:
        await update.message.reply_text(
            "Please specify a song. Example:\n/search Shape of You Ed Sheeran"
        )
        return

    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Searching for: '{query}'...")
    results = search_youtube(query)
    await status_msg.delete()
    await send_search_results(update, query, results)

async def auto_download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /autodownload command."""
    if not context.args:
        await update.message.reply_text(
            "Please specify a song. Example:\n/autodownload Shape of You Ed Sheeran"
        )
        return

    query = " ".join(context.args)
    await auto_download_first_result(update, context, query)

async def get_lyrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle lyrics requests."""
    if not context.args:
        await update.message.reply_text(
            "Please specify a song. Examples:\n"
            "/lyrics Bohemian Rhapsody\n"
            "/lyrics Queen - Bohemian Rhapsody"
        )
        return

    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Searching for lyrics: {query}")

    try:
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
    except Exception as e:
        logger.error(f"Error in get_lyrics_command: {e}")
        await status_msg.edit_text("Sorry, I couldn't find those lyrics.")

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
        await update.message.reply_text(
            f"Hi {update.effective_user.first_name}! How are you feeling today?",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return

    mood = context_data["mood"]
    await update.message.reply_text(f"🎧 Finding {mood} music for {update.effective_user.first_name}...")

    try:
        if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
            await provide_generic_recommendations(update, mood)
            return

        token = await get_spotify_token()
        if not token:
            await provide_generic_recommendations(update, mood)
            return

        seed_query = f"{mood} music"
        if context_data.get("preferences"):
            preferences = context_data["preferences"]
            if preferences:
                seed_query = f"{mood} {preferences[0]} music"

        seed_track = await search_spotify_track(token, seed_query)
        if not seed_track:
            await provide_generic_recommendations(update, mood)
            return

        recommendations = await get_spotify_recommendations(token, [seed_track["id"]])
        if not recommendations:
            await provide_generic_recommendations(update, mood)
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
        await update.message.reply_text(response, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Error in recommend_music: {e}")
        await update.message.reply_text("I couldn't get recommendations. Try again later?")

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
    await update.message.reply_text(response, parse_mode=ParseMode.HTML)

async def set_mood(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start conversation to set mood."""
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
    await update.message.reply_text(
        f"Hi {update.effective_user.first_name}! How are you feeling today?",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return MOOD

async def send_search_results(update: Update, query: str, results: List[Dict]) -> None:
    """Send search results with inline keyboard."""
    if not results:
        await update.message.reply_text(f"Sorry, I couldn't find any songs for '{query}'.")
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
    await update.message.reply_text(
        f"🔎 Search results for '{query}':\n\nClick a song to download:",
        reply_markup=reply_markup
    )

async def auto_download_first_result(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str) -> None:
    """Automatically download the first song result."""
    user_id = update.effective_user.id

    if user_id in active_downloads:
        await update.message.reply_text("⚠️ You already have a download in progress. Please wait.")
        return

    active_downloads.add(user_id)
    status_msg = await update.message.reply_text(f"🔍 Searching for '{query}'...")

    try:
        results = search_youtube(query, max_results=1)
        if not results:
            await status_msg.edit_text(f"❌ Couldn't find any results for '{query}'.")
            return

        result = results[0]
        video_url = result["url"]
        await status_msg.edit_text(f"✅ Found: {result['title']}\n⏳ Downloading...")

        download_result = download_youtube_audio(video_url)
        if not download_result["success"]:
            await status_msg.edit_text(f"❌ Download failed: {download_result['error']}")
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
    except Exception as e:
        logger.error(f"Error in auto_download_first_result: {e}")
        await status_msg.edit_text(f"❌ Error: {str(e)[:200]}")
    finally:
        active_downloads.discard(user_id)

async def enhanced_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
    """Handle button callbacks."""
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = query.from_user.id

    logger.debug(f"Handling callback query: {data} for user {user_id}")

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
            return None

        active_downloads.add(user_id)
        try:
            await query.edit_message_text("⏳ 50% done downloading...")
            result = download_youtube_audio(url)
            if not result["success"]:
                await query.edit_message_text(f"❌ Download failed: {result['error']}")
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
                logger.error(f"Timeout sending audio: {e}")
                await query.edit_message_text("❌ Failed to send audio due to timeout.")
                return None
            except NetworkError as e:
                logger.error(f"Network error sending audio: {e}")
                await query.edit_message_text("❌ Failed to send audio due to network error.")
                return None

            if os.path.exists(result["audio_path"]):
                os.remove(result["audio_path"])
                logger.info(f"Deleted file: {result['audio_path']}")

            await query.edit_message_text(f"✅ Download complete: {result['title']}")
        except Exception as e:
            logger.error(f"Error in button handler for video {video_id}: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)[:200]}")
        finally:
            active_downloads.discard(user_id)
        return None

    elif data.startswith("show_options_"):
        search_query = data.split("show_options_")[1]
        results = search_youtube(search_query)
        if not results:
            await query.edit_message_text(f"Sorry, I couldn't find any songs for '{search_query}'.")
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
            return None
        keyboard.append([InlineKeyboardButton("Cancel", callback_data="cancel_search")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            f"🔎 Search results for '{search_query}':\n\nClick a song to download:",
            reply_markup=reply_markup
        )
        return None

    elif data == "cancel_search":
        await query.edit_message_text("❌ Search cancelled.")
        return None

    elif data == "cancel_spotify":
        await query.edit_message_text("❌ Spotify linking cancelled.")
        return ConversationHandler.END

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
        except Exception as e:
            logger.error(f"Error generating chat response: {e}")
            await reply_with_retry("I'm having trouble responding. Try again?")
        finally:
            await delete_message_with_retry(typing_msg)

    except (TimedOut, NetworkError) as e:
        logger.error(f"Network error in handle_message: {e}")
        await reply_with_retry("Sorry, I'm having network issues. Please try again.")

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
        await update.message.reply_text("✅ Your conversation history has been cleared.")
    else:
        await update.message.reply_text("You don't have any saved conversation history.")

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the conversation."""
    await update.message.reply_text("No problem! Feel free to chat or use commands anytime.")
    return ConversationHandler.END

async def analyze_conversation(user_id: int) -> Dict:
    """Analyze conversation history and Spotify data."""
    if not client:
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

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                    "Analyze conversation and Spotify history to extract music preferences. "
                    "Return genres, artists, and mood in JSON."
                },
                {"role": "user", "content": 
                    f"Conversation:\n{conversation_text}\n\nSpotify Data:\n{spotify_data}"
                }
            ],
            max_tokens=150,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        result = json.loads(response.choices[0].message.content)
        if not isinstance(result, dict):
            return {"genres": context.get("preferences", []), "artists": [], "mood": context.get("mood")}

        genres = result.get("genres", [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split(",")]
        artists = result.get("artists", [])
        if isinstance(artists, str):
            artists = [a.strip() for a in artists.split(",")]
        mood = result.get("mood")

        if genres:
            context["preferences"] = genres[:3]
        if mood and not context.get("mood"):
            context["mood"] = mood
        user_contexts[user_id] = context

        return {
            "genres": genres,
            "artists": artists,
            "mood": mood
        }
    except Exception as e:
        logger.error(f"Error in analyze_conversation: {e}")
        return {"genres": context.get("preferences", []), "artists": [], "mood": context.get("mood")}

async def smart_recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provide smarter music recommendations."""
    user_id = update.effective_user.id
    status_msg = await update.message.reply_text(f"🎧 Finding personalized music for {update.effective_user.first_name}...")

    try:
        if user_contexts.get(user_id, {}).get("spotify"):
            recently_played = await get_user_spotify_data(user_id, "player/recently-played")
            if recently_played:
                user_contexts[user_id]["spotify"]["recently_played"] = recently_played
            top_tracks = await get_user_spotify_data(user_id, "top/tracks")
            if top_tracks:
                user_contexts[user_id]["spotify"]["top_tracks"] = top_tracks

        analysis = await analyze_conversation(user_id)
        if not analysis.get("mood"):
            await status_msg.delete()
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
            await update.message.reply_text(
                f"Hi {update.effective_user.first_name}! How are you feeling today?",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return

        mood = analysis.get("mood")
        genres = analysis.get("genres", [])
        artists = analysis.get("artists", [])

        search_query = mood if mood else "music"
        if genres:
            search_query += f" {genres[0]} music"
        if artists:
            search_query += f" like {artists[0]}"

        await status_msg.edit_text(f"🔍 Searching for {search_query}...")

        token = await get_spotify_token()
        seed_track_ids = []
        if user_contexts.get(user_id, {}).get("spotify", {}).get("recently_played"):
            tracks = user_contexts[user_id]["spotify"]["recently_played"]
            seed_track_ids = [track["track"]["id"] for track in tracks[:2] if track.get("track", {}).get("id")]
        elif user_contexts.get(user_id, {}).get("spotify", {}).get("top_tracks"):
            tracks = user_contexts[user_id]["spotify"]["top_tracks"]
            seed_track_ids = [track["id"] for track in tracks[:2] if track.get("id")]

        if token and seed_track_ids:
            headers = {"Authorization": f"Bearer {token}"}
            valid_seed_track_ids = []
            async with aiohttp.ClientSession() as session:
                for track_id in seed_track_ids:
                    try:
                        async with session.get(f"https://api.spotify.com/v1/tracks/{track_id}", headers=headers) as response:
                            response.raise_for_status()
                            valid_seed_track_ids.append(track_id)
                    except aiohttp.ClientError as e:
                        logger.warning(f"Invalid Spotify track ID {track_id}: {e}")
                        continue

            if valid_seed_track_ids:
                recommendations = await get_spotify_recommendations(token, valid_seed_track_ids[:2])
                if recommendations:
                    response = f"🎵 <b>Recommended music for you:</b>\n\n"
                    for i, track in enumerate(recommendations[:5], 1):
                        artists_text = ", ".join(a["name"] for a in track["artists"])
                        album = track.get("album", {}).get("name", "")
                        response += f"{i}. <b>{track['name']}</b> by {artists_text}"
                        if album:
                            response += f" (from {album})"
                        response += "\n"
                    response += "\n💡 <i>Send the song name to download!</i>"
                    await status_msg.edit_text(response, parse_mode=ParseMode.HTML)
                    return

        results = search_youtube(search_query, max_results=5)
        if results:
            response = f"🎵 <b>Recommended music for you:</b>\n\n"
            keyboard = []
            for i, result in enumerate(results, 1):
                if not result.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', result['id']):
                    logger.warning(f"Skipping invalid YouTube result ID: {result.get('id', 'No ID')}")
                    continue
                duration = result.get('duration', 0)
                duration_min = int(duration // 60)
                duration_sec = int(duration % 60)
                duration_str = f"[{duration_min}:{duration_sec:02d}]"
                response += f"{i}. <b>{result['title']}</b> - {result['uploader']} {duration_str}\n"
                button_text = f"Download: {result['title'][:30]}..." if len(result['title']) > 30 else f"Download: {result['title']}"
                keyboard.append([InlineKeyboardButton(button_text, callback_data=f"download_{result['id']}")])
            if not keyboard:
                await status_msg.delete()
                await provide_generic_recommendations(update, mood if mood else "happy")
                return
            reply_markup = InlineKeyboardMarkup(keyboard)
            await status_msg.edit_text(response, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
        else:
            await status_msg.delete()
            await provide_generic_recommendations(update, mood if mood else "happy")

    except Exception as e:
        logger.error(f"Error in smart_recommend_music: {e}")
        await status_msg.edit_text("I couldn't get recommendations. Try again later?")

async def handle_error(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors."""
    logger.error(f"Update {update} caused error {context.error}")
    if update and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "Sorry, something went wrong. Please try again later."
            )
        except (TimedOut, NetworkError) as e:
            logger.error(f"Failed to send error message: {e}")

def cleanup_downloads() -> None:
    """Clean up temporary files."""
    try:
        if os.path.exists(DOWNLOAD_DIR):
            for file in os.listdir(DOWNLOAD_DIR):
                file_path = os.path.join(DOWNLOAD_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleaned up download directory")
    except Exception as e:
        logger.error(f"Error cleaning up downloads: {e}")

def signal_handler(sig, frame) -> None:
    """Handle termination signals."""
    logger.info("Received termination signal, cleaning up...")
    cleanup_downloads()
    sys.exit(0)

def main() -> None:
    """Start the bot."""
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("download", download_music))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("autodownload", auto_download_command))
    application.add_handler(CommandHandler("lyrics", get_lyrics_command))
    application.add_handler(CommandHandler("recommend", smart_recommend_music))
    application.add_handler(CommandHandler("create_playlist", create_playlist))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(CommandHandler("spotify_code", spotify_code_command))

    spotify_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("link_spotify", link_spotify)],
        states={
            SPOTIFY_CODE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, spotify_code_handler),
                CommandHandler("spotify_code", spotify_code_handler),
                CallbackQueryHandler(cancel_spotify, pattern="cancel_spotify")
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )
    application.add_handler(spotify_conv_handler)

    mood_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("mood", set_mood)],
        states={
            MOOD: [CallbackQueryHandler(enhanced_button_handler)],
            PREFERENCE: [CallbackQueryHandler(enhanced_button_handler)],
            ACTION: [CallbackQueryHandler(enhanced_button_handler)]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )
    application.add_handler(mood_conv_handler)

    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(CallbackQueryHandler(enhanced_button_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, enhanced_handle_message))
    application.add_error_handler(handle_error)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_downloads)

    logger.info("Starting Enhanced MelodyMind Bot with Spotify Integration")
    application.run_polling()

if __name__ == "__main__":
    main()