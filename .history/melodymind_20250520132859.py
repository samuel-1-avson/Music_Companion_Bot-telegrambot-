import os
import logging
import requests
import re
import json
import base64
import pytz
from typing import Dict, List, Optional, Any, Union
from dotenv import load_dotenv
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes,
    filters, CallbackQueryHandler, ConversationHandler
)

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

# Conversation states
MOOD, PREFERENCE, ACTION, SPOTIFY_CODE = range(4)

# Global variables
active_downloads = set()
user_contexts: Dict[int, Dict] = {}
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ### Spotify Helper Functions

def get_spotify_token() -> Optional[str]:
    """Get Spotify access token using client credentials for non-user-specific requests."""
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
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting Spotify token: {e}")
        return None

@lru_cache(maxsize=100)
def search_spotify_track(token: str, query: str) -> Optional[Dict]:
    """Search for a track on Spotify."""
    if not token:
        return None

    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "type": "track", "limit": 1}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        items = response.json().get("tracks", {}).get("items", [])
        return items[0] if items else None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error searching Spotify track: {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_spotify_recommendations(token: str, seed_tracks: List[str], limit: int = 5) -> List[Dict]:
    """Get track recommendations from Spotify."""
    if not token or not seed_tracks:
        logger.warning("No token or seed tracks provided for Spotify recommendations")
        return []

    url = "https://api.spotify.com/v1/recommendations"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"seed_tracks": ",".join(seed_tracks[:2]), "limit": limit}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json().get("tracks", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting Spotify recommendations: {e}")
        return []

def refresh_spotify_token(user_id: int) -> Optional[str]:
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
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token}

    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        token_data = response.json()
        expires_at = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600))).timestamp()
        user_contexts[user_id]["spotify"] = {
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token", refresh_token),
            "expires_at": expires_at
        }
        return token_data.get("access_token")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error refreshing Spotify token: {e}")
        return None

def get_user_spotify_token(user_id: int, code: str) -> Optional[Dict]:
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
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        token_data = response.json()
        token_data["expires_at"] = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600))).timestamp()
        return token_data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting user Spotify token: {e}")
        return None

def get_user_spotify_data(user_id: int, endpoint: str) -> Optional[List[Dict]]:
    """Fetch user-specific Spotify data (recently played or top tracks)."""
    context = user_contexts.get(user_id, {})
    spotify_data = context.get("spotify", {})
    access_token = spotify_data.get("access_token")
    expires_at = spotify_data.get("expires_at")

    if not access_token or (expires_at and datetime.now(pytz.UTC).timestamp() > expires_at):
        access_token = refresh_spotify_token(user_id)
        if not access_token:
            return None

    url = f"https://api.spotify.com/v1/me/{endpoint}"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"limit": 10}

    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json().get("items", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Spotify user data ({endpoint}): {e}")
        return None

# ### YouTube Helper Functions

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

# ### AI and Lyrics Functions

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

    context["conversation_history"] = context["conversation_history"][-50:]

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

# ### Telegram Bot Handlers

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
        "🔗 Link your Spotify account\n\n"
        "Try /link_spotify or send a YouTube link to start!"
    )
    await update.message.reply_text(welcome_msg)

async def download_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download music from YouTube URL."""
    message_text = update.message.text
    url = " ".join(context.args) if context.args else next((word for word in message_text.split() if is_valid_youtube_url(word)), None)
    if not url or not is_valid_youtube_url(url):
        await update.message.reply_text("❌ Please provide a valid YouTube URL.\nExample: /download https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        return

    user_id = update.effective_user.id
    if user_id in active_downloads:
        await update.message.reply_text("⚠️ You already have a download in progress. Please wait.")
        return

    active_downloads.add(user_id)
    status_msg = await update.message.reply_text("⏳ Starting download...")

    try:
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
        os.remove(result["audio_path"])
        await status_msg.delete()
    except Exception as e:
        logger.error(f"Error in download_music: {e}")
        await status_msg.edit_text(f"❌ Error: {str(e)[:200]}")
    finally:
        active_downloads.discard(user_id)

async def smart_recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provide personalized music recommendations."""
    user_id = update.effective_user.id
    token = user_contexts.get(user_id, {}).get("spotify", {}).get("access_token") or get_spotify_token()
    if not token and user_contexts.get(user_id, {}).get("spotify"):
        token = refresh_spotify_token(user_id)

    mood = user_contexts.get(user_id, {}).get("mood", "happy")
    status_msg = await update.message.reply_text(f"🎧 Finding {mood} music recommendations...")

    try:
        seed_query = f"{mood} music"
        if token:
            seed_track = search_spotify_track(token, seed_query)
            recommendations = get_spotify_recommendations(token, [seed_track["id"]]) if seed_track else []
            if recommendations:
                response = f"🎵 **Recommended {mood} music:**\n\n"
                for i, track in enumerate(recommendations[:5], 1):
                    artists = ", ".join(a["name"] for a in track["artists"])
                    response += f"{i}. **{track['name']}** by {artists}\n"
                await status_msg.edit_text(response, parse_mode=ParseMode.MARKDOWN)
                return
        results = search_youtube(seed_query)
        response = f"🎵 **Recommended {mood} music:**\n\n"
        keyboard = [[InlineKeyboardButton(f"Download: {r['title'][:30]}...", callback_data=f"download_{r['id']}")] for r in results[:5]]
        await status_msg.edit_text(response + "\n".join(f"{i}. **{r['title']}** - {r['uploader']}" for i, r in enumerate(results[:5], 1)), reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Error in smart_recommend_music: {e}")
        await status_msg.edit_text("❌ Couldn’t fetch recommendations. Try again later.")

async def create_playlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Create a private Spotify playlist."""
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Usage: /create_playlist <name>")
        return
    name = " ".join(context.args)
    token = user_contexts.get(user_id, {}).get("spotify", {}).get("access_token") or refresh_spotify_token(user_id)
    if not token:
        await update.message.reply_text("Please link your Spotify with /link_spotify first!")
        return
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    data = {"name": name, "public": False}
    try:
        response = requests.post("https://api.spotify.com/v1/users/me/playlists", headers=headers, json=data)
        response.raise_for_status()
        await update.message.reply_text(f"Playlist '{name}' created successfully!")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error creating playlist: {e}")
        await update.message.reply_text("Failed to create playlist. Try again later.")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages by transcribing them."""
    file = await context.bot.get_file(update.message.voice.file_id)
    audio_path = os.path.join(DOWNLOAD_DIR, "voice.ogg")
    await file.download_to_drive(audio_path)
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        new_update = Update(update.update_id, message=update.message._replace(text=text))
        await enhanced_handle_message(new_update, context)
    except sr.UnknownValueError:
        await update.message.reply_text("Sorry, I couldn’t understand your voice message.")
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

async def enhanced_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages with mood detection and music requests."""
    user_id = update.effective_user.id
    text = update.message.text

    if is_valid_youtube_url(text):
        await download_music(update, context)
        return

    mood = await detect_mood_from_text(user_id, text)
    user_contexts.setdefault(user_id, {})["mood"] = mood

    response = await generate_chat_response(user_id, text)
    await update.message.reply_text(response)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline keyboard button presses."""
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = query.from_user.id

    if data.startswith("download_"):
        video_id = data.split("_")[1]
        url = f"https://www.youtube.com/watch?v={video_id}"
        active_downloads.add(user_id)
        await query.edit_text("⏳ Downloading...")
        try:
            result = download_youtube_audio(url)
            if result["success"]:
                with open(result["audio_path"], 'rb') as audio:
                    await context.bot.send_audio(
                        chat_id=query.message.chat_id,
                        audio=audio,
                        title=result["title"][:64],
                        performer=result["artist"][:64] if result.get("artist") else "Unknown Artist"
                    )
                os.remove(result["audio_path"])
                await query.edit_text(f"✅ Downloaded: {result['title']}")
            else:
                await query.edit_text(f"❌ Error: {result['error']}")
        finally:
            active_downloads.discard(user_id)

# ### Main Function

def main() -> None:
    """Start the MelodyMind bot."""
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("download", download_music))
    application.add_handler(CommandHandler("recommend", smart_recommend_music))
    application.add_handler(CommandHandler("create_playlist", create_playlist))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, enhanced_handle_message))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(CallbackQueryHandler(button_handler))

    logger.info("Starting MelodyMind Bot")
    application.run_polling()

if __name__ == "__main__":
    main()