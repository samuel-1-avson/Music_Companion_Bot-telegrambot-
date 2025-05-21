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
    filters, CallbackQueryHandler, ConversationHandler, AIORateLimiter
)
from telegram.error import TimedOut, NetworkError

import yt_dlp
from openai import OpenAI
import importlib
if importlib.util.find_spec("lyricsgenius") is not None:
    import lyricsgenius
else:
    lyricsgenius = None
if importlib.util.find_spec("async_lru") is not None:
    from async_lru import alru_cache
else:
    # Fallback for environments where async_lru might not be installed
    # This won't cache but allows the code to run.
    def alru_cache(maxsize=128, typed=False):
        def decorator(func):
            return func
        return decorator
    logging.warning("async_lru not found. Spotify search results will not be cached efficiently. Pip install async-lru.")

import speech_recognition as sr

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "https://your-callback-url.com") # Must match Spotify Dev Dashboard
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=15, retries=3) if GENIUS_ACCESS_TOKEN and lyricsgenius else None # Added timeout and retries for genius

# Encryption Cipher
if ENCRYPTION_KEY:
    try:
        cipher_key_bytes = base64.urlsafe_b64decode(ENCRYPTION_KEY.encode())
        cipher = Fernet(cipher_key_bytes)
        logger.info("Successfully loaded ENCRYPTION_KEY.")
    except Exception as e:
        logger.warning(f"Invalid ENCRYPTION_KEY format: {e}. Generating a new one for this session.")
        cipher_key_bytes = Fernet.generate_key()
        logger.warning(f"Generated new ENCRYPTION_KEY for this session: {base64.urlsafe_b64encode(cipher_key_bytes).decode()}")
        logger.warning("Spotify tokens will NOT persist across restarts unless a static ENCRYPTION_KEY is correctly set AND user_contexts are persisted.")
        cipher = Fernet(cipher_key_bytes)
else:
    cipher_key_bytes = Fernet.generate_key()
    logger.warning("ENCRYPTION_KEY not set. Generating a new one for this session.")
    logger.warning(f"If you want Spotify links to persist across restarts (requires persisting user_contexts), set this as ENCRYPTION_KEY: {base64.urlsafe_b64encode(cipher_key_bytes).decode()}")
    logger.warning("Currently, user_contexts (including Spotify tokens) are in-memory and will be lost on restart.")
    cipher = Fernet(cipher_key_bytes)


# Conversation states
MOOD, PREFERENCE, ACTION, SPOTIFY_CODE = range(4)

# Callback Data Prefixes / Constants
CB_MOOD_PREFIX = "mood_"
CB_PREFERENCE_PREFIX = "pref_"
CB_DOWNLOAD_PREFIX = "download_"
CB_AUTO_DOWNLOAD_PREFIX = "auto_download_"
CB_SHOW_OPTIONS_PREFIX = "show_options_"
CB_CANCEL_SEARCH = "cancel_search"
CB_CANCEL_SPOTIFY = "cancel_spotify"


# Track active downloads and user contexts
active_downloads = set() # Tracks user_ids with active downloads
user_contexts: Dict[int, Dict] = {} # In-memory user context store
logger.warning("User contexts are stored in-memory and will be lost on bot restart.")
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=10) # 10 seconds default for Spotify API calls

# ==================== SPOTIFY HELPER FUNCTIONS ====================

async def get_spotify_token() -> Optional[str]:
    """Get Spotify access token using client credentials."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        logger.warning("Spotify client credentials not configured for client-credentials flow.")
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
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url, headers=headers, data=data) as response:
                response.raise_for_status()
                return (await response.json()).get("access_token")
    except aiohttp.ClientError as e:
        logger.error(f"Error getting Spotify client_credentials token: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting Spotify client_credentials token: {e}")
        return None

@alru_cache(maxsize=100)
async def search_spotify_track(token: str, query: str) -> Optional[Dict]:
    """Search for a track on Spotify. Cached."""
    if not token:
        return None

    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "type": "track", "limit": 1}

    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                items = (await response.json()).get("tracks", {}).get("items", [])
                return items[0] if items else None
    except aiohttp.ClientError as e:
        logger.error(f"Error searching Spotify track '{query}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error searching Spotify track '{query}': {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_spotify_recommendations(token: str, seed_tracks: List[str], limit: int = 5, seed_genres: Optional[List[str]] = None, seed_artists: Optional[List[str]] = None) -> List[Dict]:
    """Get track recommendations from Spotify."""
    if not token:
        logger.warning("No token provided for Spotify recommendations.")
        return []
    
    params = {"limit": limit}
    seed_count = 0
    if seed_tracks:
        params["seed_tracks"] = ",".join(seed_tracks[:max(0, 5-seed_count)])
        seed_count += len(params["seed_tracks"].split(','))
    if seed_genres and seed_count < 5:
        params["seed_genres"] = ",".join(seed_genres[:max(0, 5-seed_count)])
        seed_count += len(params["seed_genres"].split(','))
    if seed_artists and seed_count < 5:
        params["seed_artists"] = ",".join(seed_artists[:max(0, 5-seed_count)])
        seed_count += len(params["seed_artists"].split(','))

    if seed_count == 0:
        logger.warning("No seeds (tracks, genres, artists) provided for Spotify recommendations.")
        return []

    url = "https://api.spotify.com/v1/recommendations"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                return (await response.json()).get("tracks", [])
    except aiohttp.ClientError as e:
        logger.error(f"Error getting Spotify recommendations (params: {params}): {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error getting Spotify recommendations (params: {params}): {e}")
        return []

async def get_user_spotify_token(user_id: int, code: str) -> Optional[Dict]:
    """Exchange authorization code for Spotify access and refresh tokens."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI:
        logger.warning("Spotify OAuth credentials (client_id, client_secret, redirect_uri) not fully configured.")
        return None

    url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": SPOTIFY_REDIRECT_URI
    }

    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url, headers=headers, data=data) as response:
                response.raise_for_status()
                token_data = await response.json()
                token_data["expires_at"] = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600) - 60)).timestamp() # -60s buffer
                return token_data
    except aiohttp.ClientError as e:
        logger.error(f"Error getting user Spotify token for user {user_id} with code: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting user Spotify token for user {user_id}: {e}")
        return None

async def refresh_spotify_token(user_id: int) -> Optional[str]:
    """Refresh Spotify access token using refresh token."""
    context = user_contexts.get(user_id, {})
    encrypted_refresh_token_bytes = context.get("spotify", {}).get("refresh_token")

    if not encrypted_refresh_token_bytes:
        logger.warning(f"No refresh token found for user {user_id} to refresh Spotify token.")
        return None
    
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET: 
        logger.error("Cannot refresh Spotify token: Client ID or Secret not configured.")
        return None

    try:
        refresh_token_str = cipher.decrypt(encrypted_refresh_token_bytes).decode()
    except Exception as e:
        logger.error(f"Failed to decrypt refresh token for user {user_id}: {e}. Re-authentication required.")
        if "spotify" in user_contexts.get(user_id, {}):
            user_contexts[user_id]["spotify"] = {}
        return None


    url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token_str}

    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url, headers=headers, data=data) as response:
                response.raise_for_status()
                token_data = await response.json()

                new_access_token = token_data.get("access_token")
                if not new_access_token:
                    logger.error(f"Refresh token grant did not return new access_token for user {user_id}")
                    return None
                
                expires_at = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600) - 60)).timestamp() # -60s buffer
                new_refresh_token_str = token_data.get("refresh_token", refresh_token_str)

                user_contexts[user_id]["spotify"]["access_token"] = cipher.encrypt(new_access_token.encode())
                user_contexts[user_id]["spotify"]["refresh_token"] = cipher.encrypt(new_refresh_token_str.encode())
                user_contexts[user_id]["spotify"]["expires_at"] = expires_at
                
                return new_access_token
    except aiohttp.ClientError as e:
        logger.error(f"Error refreshing Spotify token for user {user_id}: {e}")
        if hasattr(e, 'status') and e.status == 400: 
             logger.error(f"Spotify refresh token for user {user_id} might be revoked. Re-authentication needed.")
             if "spotify" in user_contexts.get(user_id, {}):
                user_contexts[user_id]["spotify"] = {} 
        return None
    except Exception as e:
        logger.error(f"Unexpected error refreshing Spotify token for user {user_id}: {e}")
        return None

async def get_user_spotify_access_token(user_id: int) -> Optional[str]:
    """Helper to get a valid access token for a user, refreshing if necessary."""
    context = user_contexts.get(user_id, {})
    spotify_data = context.get("spotify", {})
    encrypted_access_token_bytes = spotify_data.get("access_token")
    expires_at = spotify_data.get("expires_at")

    if not encrypted_access_token_bytes or \
       (expires_at and datetime.now(pytz.UTC).timestamp() > expires_at):
        logger.info(f"Access token for user {user_id} is missing or expired, attempting refresh.")
        return await refresh_spotify_token(user_id)
    
    try:
        return cipher.decrypt(encrypted_access_token_bytes).decode()
    except Exception as e:
        logger.error(f"Failed to decrypt access token for user {user_id}: {e}. Attempting refresh.")
        return await refresh_spotify_token(user_id)


async def get_user_spotify_data(user_id: int, endpoint: str, params: Optional[Dict] = None) -> Optional[List[Dict]]:
    """Fetch user-specific Spotify data (e.g., 'player/recently-played', 'top/tracks')."""
    access_token = await get_user_spotify_access_token(user_id)
    if not access_token:
        logger.warning(f"Could not obtain Spotify access token for user {user_id} to fetch {endpoint}.")
        return None

    url = f"https://api.spotify.com/v1/me/{endpoint}"
    headers = {"Authorization": f"Bearer {access_token}"}
    request_params = {"limit": 10, **(params or {})} 

    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.get(url, headers=headers, params=request_params) as response:
                response.raise_for_status()
                return (await response.json()).get("items", [])
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching Spotify user data ({endpoint}) for user {user_id}: {e}")
        if hasattr(e, 'status') and e.status == 401: 
            logger.info(f"Spotify token unauthorized for user {user_id} for {endpoint}.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching Spotify user data ({endpoint}) for user {user_id}: {e}")
        return None

# ==================== YOUTUBE HELPER FUNCTIONS ====================

def is_valid_youtube_url(url: str) -> bool:
    """Check if the URL is a valid YouTube URL."""
    if not url:
        return False
    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    ]
    return any(re.search(pattern, url) for pattern in patterns)

def sanitize_filename(filename: str) -> str:
    """Remove invalid characters from filenames for display or metadata."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return sanitized[:100]

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5)) # Reduced retries for faster fail
def download_youtube_audio_sync(url: str) -> Dict[str, Any]: 
    """Download audio from a YouTube video. This is a BLOCKING function."""
    logger.info(f"Attempting to download audio from: {url}")
    
    video_id_match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11})', url)
    video_id = video_id_match.group(1) if video_id_match else "UnknownID"

    try:
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best', 
            'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            'max_filesize': 50 * 1024 * 1024, 
            'writethumbnail': False, # Changed to False unless path used
            'restrictfilenames': True, # For safer filenames
            'sleep_interval_requests': 1, # Slow down if making many requests
            'sleep_interval':1,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False) 
            if not info:
                logger.error(f"Could not extract video information for {url} (ID: {video_id})")
                return {"success": False, "error": "Could not extract video information"}

            display_title = sanitize_filename(info.get('title', 'Unknown Title'))
            artist = sanitize_filename(info.get('artist', info.get('uploader', 'Unknown Artist')))
            expected_audio_path = ydl.prepare_filename(info)

            logger.info(f"Downloading '{display_title}' to '{expected_audio_path}'")
            ydl.extract_info(url, download=True)

            if not os.path.exists(expected_audio_path):
                # Simplified fallback: Check for common extensions if prepare_filename was imperfect
                base_name_no_ext, _ = os.path.splitext(expected_audio_path)
                possible_exts = ['m4a', 'mp3', 'webm', 'ogg', 'opus'] # Common audio extensions yt-dlp might use
                found_path = None
                for ext_attempt in possible_exts:
                    potential_path = f"{base_name_no_ext}.{ext_attempt}"
                    if os.path.exists(potential_path):
                        found_path = potential_path
                        logger.info(f"File found at path: {found_path} (original expected: {expected_audio_path})")
                        break
                if not found_path:
                    logger.error(f"Downloaded file not found. Expected: {expected_audio_path} or similar variants for {url}")
                    return {"success": False, "error": "Downloaded file not found"}
                expected_audio_path = found_path


            file_size_mb = os.path.getsize(expected_audio_path) / (1024 * 1024)
            if file_size_mb > 50.5: # Small buffer for check
                os.remove(expected_audio_path)
                logger.warning(f"File '{display_title}' exceeded 50MB limit ({file_size_mb:.2f}MB), removing.")
                return {"success": False, "error": "File exceeds 50 MB Telegram limit after download"}
            
            thumbnail_url = info.get('thumbnail') 

            return {
                "success": True,
                "title": display_title,
                "artist": artist,
                "thumbnail_url": thumbnail_url, 
                "duration": info.get('duration', 0),
                "audio_path": expected_audio_path
            }
    except yt_dlp.utils.DownloadError as de:
        logger.error(f"yt-dlp DownloadError for {url} (ID: {video_id}): {de}")
        error_msg = str(de)
        if "Unsupported URL" in error_msg: return {"success": False, "error": "Unsupported URL."}
        if "Video unavailable" in error_msg: return {"success": False, "error": "Video unavailable."}
        if "is not available" in error_msg: return {"success": False, "error": "Video not available."}
        if "Private video" in error_msg: return {"success": False, "error": "This is a private video."}
        return {"success": False, "error": f"Download failed: {error_msg[:100]}"} 
    except Exception as e:
        logger.error(f"Generic error downloading YouTube audio {url} (ID: {video_id}): {e}", exc_info=True)
        return {"success": False, "error": f"An unexpected error occurred: {str(e)[:100]}"}

def search_youtube_sync(query: str, max_results: int = 5) -> List[Dict]: 
    """Search YouTube for videos matching the query. This is a BLOCKING function."""
    logger.info(f"Searching YouTube for: '{query}' with max_results={max_results}")
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'discard_in_playlist', 
            'default_search': f'ytsearch{max_results}', 
            'noplaylist': True, 
            'sleep_interval_requests': 1,
            'sleep_interval': 1,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            
            if not info or 'entries' not in info:
                logger.warning(f"No YouTube search results for query: '{query}'")
                return []
            
            results = []
            for entry in info['entries']:
                if not entry: continue 
                results.append({
                    'title': entry.get('title', 'Unknown Title'),
                    'url': entry.get('webpage_url') or entry.get('url') or f"https://www.youtube.com/watch?v={entry.get('id')}", # Prefer webpage_url
                    'thumbnail': entry.get('thumbnail') or (entry.get('thumbnails')[0]['url'] if entry.get('thumbnails') else ''),
                    'uploader': entry.get('uploader', 'Unknown Artist'),
                    'duration': entry.get('duration', 0),
                    'id': entry.get('id', '') 
                })
            logger.info(f"Found {len(results)} results for '{query}'")
            return results
            
    except yt_dlp.utils.DownloadError as de:
        logger.error(f"yt-dlp DownloadError during YouTube search for '{query}': {de}")
        return[] 
    except Exception as e:
        logger.error(f"Error searching YouTube for '{query}': {e}", exc_info=True)
        return []

# ==================== AI AND LYRICS FUNCTIONS ====================

async def generate_chat_response(user_id: int, message: str) -> str:
    """Generate a conversational response using OpenAI."""
    if not client:
        return "I'm having trouble connecting to my AI service. Please try again later."

    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    context = user_contexts[user_id]
    context.setdefault("conversation_history", []) 

    context["conversation_history"] = context["conversation_history"][-20:]  # Shorter history for less token usage

    system_prompt = (
        "You are MelodyMind, a friendly, empathetic music companion bot. Focus on brief, warm, natural conversation about music and feelings. "
        "If the user asks for music, guide them to use commands or suggest you can search if they name a song. "
        "Use user context (mood, prefs, Spotify artists) subtly to personalize. Keep replies to 2-3 sentences. "
        "Do not suggest commands like /download unless the user explicitly asks how to get a song."
    )
    messages = [{"role": "system", "content": system_prompt}]

    context_summary_parts = []
    if context.get("mood"):
        context_summary_parts.append(f"Mood: {context.get('mood')}.")
    if context.get("preferences"):
        context_summary_parts.append(f"Prefs: {', '.join(context.get('preferences'))}.")
    
    if "spotify" in context and context["spotify"].get("recently_played"):
        try:
            artists = list(set(item["track"]["artists"][0]["name"] for item in context["spotify"]["recently_played"][:3] if item.get("track") and item["track"].get("artists")))
            if artists:
                context_summary_parts.append(f"Listens to: {', '.join(artists)}.")
        except Exception: 
            pass 
            
    if context_summary_parts:
        messages.append({"role": "system", "content": "User Info: " + " ".join(context_summary_parts)})


    for hist_msg in context["conversation_history"][-6:]: # Last 3 exchanges
        messages.append(hist_msg)
    messages.append({"role": "user", "content": message})

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=100, # Reduced max_tokens
            temperature=0.75
        )
        reply = response.choices[0].message.content.strip()
        context["conversation_history"].extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply}
        ])
        return reply
    except Exception as e:
        logger.error(f"Error generating chat response for user {user_id}: {e}")
        return "I'm having a little trouble thinking of a reply right now. Maybe we can talk about your favorite song instead?"

def get_lyrics_sync(song_title: str, artist: Optional[str] = None) -> str: 
    """Get lyrics for a song using Genius API. This is a BLOCKING function."""
    if not genius:
        return "Lyrics service is currently unavailable."
    logger.info(f"Fetching lyrics for song: '{song_title}' by artist: '{artist or 'Any'}'")
    try:
        if artist:
            song = genius.search_song(song_title, artist)
        else:
            song = genius.search_song(song_title)
            
        if not song:
            err_msg = f"Sorry, I couldn't find lyrics for '<b>{song_title}</b>'"
            if artist: err_msg += f" by '<i>{artist}</i>'"
            err_msg += ". Please check the spelling or try another song!"
            logger.warning(f"No lyrics found for '{song_title}' by '{artist or 'Any'}'")
            return err_msg
        
        lyrics = song.lyrics
        lyrics = re.sub(r'\s*\[.*?\]\s*', '\n', lyrics)  # Replace [Chorus] with newline, better flow
        lyrics = re.sub(r'\d*Embed$', '', lyrics.strip()) 
        lyrics = re.sub(r'^\S*Lyrics', '', lyrics.strip(), flags=re.IGNORECASE) 
        lyrics = re.sub(r'\n{3,}', '\n\n', lyrics) # Reduce multiple newlines to double
        lyrics = lyrics.strip()

        if not lyrics: 
            logger.warning(f"Lyrics found for '{song.title}' but were empty after cleaning.")
            return f"Lyrics for '<b>{song.title}</b>' by <i>{song.artist}</i> seem to be empty or missing. Try another?"

        header = f"🎵 <b>{song.title}</b> by <i>{song.artist}</i> 🎵\n\n"
        return header + lyrics
    except Exception as e: 
        logger.error(f"Error fetching lyrics for '{song_title}' from Genius: {e}", exc_info=True)
        return f"I encountered an issue trying to fetch lyrics for '<b>{song_title}</b>'. Please try again later."


async def detect_mood_from_text(user_id: int, text: str) -> str:
    """Detect mood from user's message using AI."""
    if not client:
        return user_contexts.get(user_id, {}).get("mood", "neutral") 
    logger.debug(f"Detecting mood from text for user {user_id}: '{text[:50]}...'")
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a mood detection AI. Analyze text and return a single dominant mood (happy, sad, anxious, excited, calm, angry, energetic, relaxed, focused, nostalgic, or neutral if unclear)."},
                      {"role": "user", "content": f"Text: '{text}'"}],
            max_tokens=10, 
            temperature=0.2 # Lower for more deterministic mood
        )
        mood_raw = response.choices[0].message.content.lower().strip().replace(".", "")
        # Normalize common variations
        mood_map = {
            "positive": "happy", "negative": "sad", "joyful": "happy", "depressed": "sad",
            "chill": "relaxed", "stressed": "anxious", "hyper": "energetic", "peaceful": "calm"
        }
        mood = mood_map.get(mood_raw, mood_raw)
        
        valid_moods = ["happy", "sad", "anxious", "excited", "calm", "angry", "neutral", "energetic", "relaxed", "focused", "nostalgic"]
        if mood in valid_moods:
            logger.info(f"Detected mood for user {user_id}: '{mood}' from raw '{mood_raw}'")
            return mood
        else: 
            logger.warning(f"Unexpected mood from AI: '{mood_raw}'. Defaulting to neutral.")
            return "neutral"

    except Exception as e:
        logger.error(f"Error detecting mood for user {user_id}: {e}")
        return user_contexts.get(user_id, {}).get("mood", "neutral") 


async def is_music_request(user_id: int, message: str) -> Dict:
    """Use AI to determine if a message is a music/song request and extract query."""
    if not client:
        return {"is_music_request": False, "song_query": None}

    logger.debug(f"AI: Checking if '{message[:50]}...' is music request for user {user_id}")
    try:
        prompt_messages = [
            {"role": "system", "content": 
                "You are an AI that analyzes user messages for specific music requests. "
                "Respond in JSON with 'is_music_request' (boolean) and 'song_query' (string containing song title/artist, or null). "
                "Focus on explicit requests like 'play X by Y', 'download Z', 'find music A'. General music chat or mood expression is NOT a specific song request unless they name something specific they want *now*."
            },
            {"role": "user", "content": f"Analyze message: '{message}'"}
        ]
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo-0125", 
            messages=prompt_messages,
            max_tokens=80, # Shorter response
            temperature=0.05, # Very low temp for this task
            response_format={"type": "json_object"}
        )

        result_str = response.choices[0].message.content
        result = json.loads(result_str)

        if not isinstance(result, dict): 
            logger.error(f"AI music request (user {user_id}) returned non-dict: {result_str}")
            return {"is_music_request": False, "song_query": None}

        is_request = result.get("is_music_request", False)
        if isinstance(is_request, str):
            is_request = is_request.lower() in ("yes", "true")
        
        song_query = result.get("song_query")
        if not isinstance(song_query, str) or not song_query.strip():
            song_query = None 

        logger.info(f"AI music request (user {user_id}): is_request={is_request}, query='{song_query}' for msg: '{message[:30]}'")
        return {
            "is_music_request": bool(is_request),
            "song_query": song_query
        }
    except json.JSONDecodeError as jde:
        logger.error(f"AI music request JSON (user {user_id}) decode error: {jde}. Raw: {response.choices[0].message.content if 'response' in locals() and response.choices else 'N/A'}")
        return {"is_music_request": False, "song_query": None}
    except Exception as e:
        logger.error(f"Error in AI is_music_request for user {user_id}: {e}", exc_info=False) # exc_info false for brevity unless debugging
        return {"is_music_request": False, "song_query": None}


# ==================== TELEGRAM BOT HANDLERS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message."""
    user = update.effective_user
    user_contexts.setdefault(user.id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})

    welcome_msg = (
        f"Hi {user.first_name}! 👋 I'm MelodyMind, your Music Healing Companion.\n\n"
        "I can help you:\n"
        "🎵 Download music (send a YouTube link or ask, e.g., 'play despacito')\n"
        "📜 Find lyrics (e.g., `/lyrics Bohemian Rhapsody`)\n"
        "💿 Get music recommendations (try `/recommend` or `/mood`)\n"
        "💬 Chat about music or how you're feeling!\n"
        "🔗 Link Spotify for personalized vibes (`/link_spotify`)\n"
        "📖 Create Spotify playlists (`/create_playlist My Favs`)\n\n"
        "Type `/help` for all commands, or just start chatting!"
    )
    await update.message.reply_text(welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message."""
    help_text = (
        "🎶 <b>MelodyMind - Your Music Companion</b> 🎶\n\n"
        "<b>Commands:</b>\n"
        "/start - Welcome message\n"
        "/help - This help guide\n"
        "/download <code>[YT URL]</code> - Download from YouTube link\n"
        "/autodownload <code>[song]</code> - Search & download top result\n"
        "/search <code>[song]</code> - YouTube search with options\n"
        "/lyrics <code>[song]</code> or <code>[artist - song]</code> - Get lyrics\n"
        "/recommend - Personalized music recommendations\n"
        "/mood - Set your mood for recommendations\n"
        "/link_spotify - Connect Spotify account\n"
        "/create_playlist <code>[name]</code> - New private Spotify playlist\n"
        "/clear - Clear our chat history\n\n"
        "<b>Chat With Me!</b>\n"
        "You can also just talk to me:\n"
        "- \"I'm feeling sad.\"\n"
        "- \"Play 'Shape of You' by Ed Sheeran\"\n"
        "- \"Find lyrics for Hotel California\"\n"
        "- Send a YouTube link to download.\n"
        "- Send a voice message!\n\n"
        "Let the music flow! 🎵"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

async def download_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download music from YouTube URL (called by command or direct URL message)."""
    message_text = update.message.text
    url = ""

    if context.args: 
        url = " ".join(context.args) # Args may contain spaces if user typed /download some url part1 part2
    else: 
        urls_in_message = [word for word in message_text.split() if is_valid_youtube_url(word)]
        if urls_in_message:
            url = urls_in_message[0]
        else: # Should ideally not be reached if direct URL routes here correctly
            await update.message.reply_text(
                "❌ Please provide a valid YouTube URL with `/download` or send the link directly."
            )
            return

    if not is_valid_youtube_url(url):
        await update.message.reply_text("❌ That doesn't look like a valid YouTube URL.")
        return

    user_id = update.effective_user.id
    if user_id in active_downloads:
        await update.message.reply_text("⚠️ One download at a time, please! Yours is in progress. 😊")
        return

    active_downloads.add(user_id)
    status_msg = await update.message.reply_text("⏳ Starting download... hold tight!")

    try:
        await status_msg.edit_text("🔍 Fetching info & preparing download...")
        result = await asyncio.to_thread(download_youtube_audio_sync, url)
        
        if not result["success"]:
            error_message = result.get('error', 'Unknown download error.')
            await status_msg.edit_text(f"❌ Download failed: {error_message}")
            return

        await status_msg.edit_text(f"✅ Downloaded: <b>{result['title']}</b>\n⏳ Sending you the audio...", parse_mode=ParseMode.HTML)
        
        audio_path = result["audio_path"]
        with open(audio_path, 'rb') as audio_file:
            logger.info(f"Sending audio '{result['title']}' (user: {user_id}). Path: {audio_path}")
            await update.message.reply_audio(
                audio=audio_file,
                title=result["title"][:64], 
                performer=result["artist"][:64] if result.get("artist") else "Unknown", 
                caption=f"🎵 {result['title']}",
                duration=result.get('duration'),
            )

        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"Deleted temp file: {audio_path}")
            except OSError as e:
                logger.error(f"Error deleting temp file {audio_path}: {e}")
        try:
            await status_msg.delete()
        except: pass # If already deleted or other issue, just log implicitly by not failing

    except TimedOut:
        logger.error(f"Telegram API timeout during download (user {user_id}, url: {url})")
        await status_msg.edit_text("❌ Telegram timeout. Please try again.")
    except NetworkError as ne:
        logger.error(f"Telegram API network error during download (user {user_id}, url: {url}): {ne}")
        await status_msg.edit_text(f"❌ Telegram network error: {ne}. Try again.")
    except Exception as e:
        logger.error(f"Unexpected error in download_music (user {user_id}, url: {url}): {e}", exc_info=True)
        try: 
            await status_msg.edit_text(f"❌ An unexpected error: {str(e)[:100]}.")
        except Exception:
            pass 
    finally:
        active_downloads.discard(user_id)


async def create_playlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Create a private Spotify playlist."""
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Name your playlist: `/create_playlist <Your Playlist Name>`")
        return
    
    playlist_name = " ".join(context.args)
    logger.info(f"User {user_id} creating Spotify playlist: '{playlist_name}'")

    access_token = await get_user_spotify_access_token(user_id)
    if not access_token:
        await update.message.reply_text(
            "I need Spotify access. 😥 Please link your account via /link_spotify."
        )
        return

    user_profile_url = "https://api.spotify.com/v1/me"
    headers_auth = {"Authorization": f"Bearer {access_token}"}
    spotify_user_id_from_api = None # Renamed to avoid clash
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.get(user_profile_url, headers=headers_auth) as response:
                response.raise_for_status()
                spotify_user_id_from_api = (await response.json()).get("id")
        if not spotify_user_id_from_api:
            logger.error(f"Could not get Spotify user ID for Telegram user {user_id}.")
            await update.message.reply_text("Sorry, couldn't get your Spotify profile ID.")
            return
    except aiohttp.ClientError as e:
        logger.error(f"API error fetching Spotify profile (user {user_id}): {e}")
        await update.message.reply_text("Issue fetching your Spotify profile. Try again.")
        return
    
    playlist_creation_url = f"https://api.spotify.com/v1/users/{spotify_user_id_from_api}/playlists"
    headers_create = {**headers_auth, "Content-Type": "application/json"}
    payload = {"name": playlist_name, "public": False, "description": "Created with MelodyMind Bot @ " + datetime.now().strftime("%Y-%m-%d %H:%M")}

    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(playlist_creation_url, headers=headers_create, json=payload) as response:
                response.raise_for_status()
                playlist_data = await response.json()
                playlist_url = playlist_data.get("external_urls", {}).get("spotify", "#")
                logger.info(f"Playlist '{playlist_name}' created (user {user_id}). URL: {playlist_url}")
                await update.message.reply_text(
                    f"🎉 Playlist '<b>{playlist_name}</b>' created!\n"
                    f"View: {playlist_url}",
                    parse_mode=ParseMode.HTML, disable_web_page_preview=True
                )
    except aiohttp.ClientError as e:
        status = getattr(e, 'status', 'N/A')
        message_detail = getattr(e, 'message', str(e)) # Get a more detailed message if available
        logger.error(f"API error creating playlist '{playlist_name}' (user {user_id}): {status} - {message_detail}")
        await update.message.reply_text(f"Oops! Failed to create playlist (Error {status}: {message_detail[:100]}).")
    except Exception as e:
        logger.error(f"Unexpected error creating playlist (user {user_id}): {e}", exc_info=True)
        await update.message.reply_text("Unexpected error creating playlist.")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages by transcribing them."""
    if not update.message or not update.message.voice:
        return

    user_id = update.effective_user.id
    logger.info(f"Voice message from user {user_id}")
    
    voice_file = await context.bot.get_file(update.message.voice.file_id)
    temp_ogg_path = os.path.join(DOWNLOAD_DIR, f"voice_{user_id}_{update.message.message_id}.ogg")
    
    await voice_file.download_to_drive(temp_ogg_path)
    logger.debug(f"Voice (user {user_id}) downloaded to {temp_ogg_path}")

    recognizer = sr.Recognizer()
    transcribed_text = None
    try:
        def _transcribe_audio_sync(): # Inner sync function for asyncio.to_thread
            with sr.AudioFile(temp_ogg_path) as source:
                audio_data = recognizer.record(source)  
            try:
                return recognizer.recognize_google(audio_data) 
            except sr.UnknownValueError:
                logger.warning(f"SR: Google could not understand audio (user {user_id})")
                return None
            except sr.RequestError as req_e:
                logger.error(f"SR: Google request failed (user {user_id}); {req_e}")
                return "ERROR_REQUEST"

        transcribed_text = await asyncio.to_thread(_transcribe_audio_sync)

        if transcribed_text == "ERROR_REQUEST":
            await update.message.reply_text("Sorry, voice recognition service error. Please type or try voice later.")
        elif transcribed_text:
            logger.info(f"Voice (user {user_id}) transcribed: '{transcribed_text}'")
            await update.message.reply_text(f"🎤 I heard: \"<i>{transcribed_text}</i>\"\nProcessing...", parse_mode=ParseMode.HTML)
            
            context.user_data['_voice_original_message'] = update.message 
            fake_message = update.message._replace(text=transcribed_text, voice=None) 
            fake_update = Update(update.update_id, message=fake_message)
            await enhanced_handle_message(fake_update, context)
        else:
            await update.message.reply_text("Hmm, I couldn't catch that. Try speaking clearly, or type your message? 😊")

    except Exception as e:
        logger.error(f"Error processing voice (user {user_id}): {e}", exc_info=True)
        await update.message.reply_text("Oops! Error with voice message. Please try again.")
    finally:
        if os.path.exists(temp_ogg_path):
            try:
                os.remove(temp_ogg_path)
                logger.debug(f"Deleted temp voice file: {temp_ogg_path}")
            except OSError as e:
                logger.error(f"Error deleting temp voice file {temp_ogg_path}: {e}")


async def link_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Initiate Spotify OAuth flow."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI:
        await update.message.reply_text("Sorry, Spotify linking isn't configured by admin. 😥")
        return ConversationHandler.END
    if SPOTIFY_REDIRECT_URI == "https://your-callback-url.com":
         await update.message.reply_text(
            "⚠️ Spotify redirect URI is a placeholder. Linking may need manual code copy from URL params."
        )

    user_id = update.effective_user.id
    scopes = "user-read-recently-played user-top-read playlist-modify-private"
    auth_url = (
        "https://accounts.spotify.com/authorize"
        f"?client_id={SPOTIFY_CLIENT_ID}"
        "&response_type=code"
        f"&redirect_uri={SPOTIFY_REDIRECT_URI}"
        f"&scope={scopes.replace(' ', '%20')}" 
        f"&state={user_id}" 
    )
    keyboard = [
        [InlineKeyboardButton("🔗 Link My Spotify", url=auth_url)],
        [InlineKeyboardButton("Cancel", callback_data=CB_CANCEL_SPOTIFY)]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Let's link Spotify for personalized music! 🎵\n\n"
        "1. Click below to go to Spotify.\n"
        "2. Authorize, then Spotify redirects you. From that page's URL, copy the `code` value.\n"
        "   (URL: `https://your-redirect-uri/?code=A_LONG_CODE&state=...` - get `A_LONG_CODE`)\n"
        "3. Send that code back to me here.\n\n"
        "If issues, ensure admin set redirect URI in Spotify Developer Dashboard correctly.",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN 
    )
    return SPOTIFY_CODE

async def spotify_code_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle Spotify authorization code from the user."""
    user_id = update.effective_user.id
    message_text = update.message.text.strip()
    
    code_to_use = None
    if message_text.startswith('/spotify_code') and context.args: 
        code_to_use = context.args[0]
    elif not message_text.startswith('/'): 
        code_to_use = message_text
    
    # Basic code validation (length, chars) could be added
    if not code_to_use or len(code_to_use) < 50: # Arbitrary short length check
        await update.message.reply_text(
            "That code seems too short or is missing. Please send the full Spotify code you copied, or use `/spotify_code YOUR_CODE`."
        )
        return SPOTIFY_CODE 

    status_msg = await update.message.reply_text("⏳ Validating your Spotify code...")
    token_data = await get_user_spotify_token(user_id, code_to_use)

    if not token_data or not token_data.get("access_token"):
        await status_msg.edit_text(
            "❌ Failed to link Spotify. Code might be invalid/expired or config issue. "
            "Try /link_spotify again. Ensure you copy `code` parameter correctly."
        )
        return SPOTIFY_CODE 

    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    
    user_contexts[user_id]["spotify"] = {
        "access_token": cipher.encrypt(token_data["access_token"].encode()),
        "refresh_token": cipher.encrypt(token_data["refresh_token"].encode()), 
        "expires_at": token_data["expires_at"]
    }
    logger.info(f"Spotify linked for user {user_id}.")

    recently_played = await get_user_spotify_data(user_id, "player/recently-played", params={"limit": 3})
    rp_info = ""
    if recently_played and len(recently_played) > 0 :
        try:
            first_rp_artist = recently_played[0]['track']['artists'][0]['name']
            rp_info = f" I see you recently enjoyed some {first_rp_artist}!"
        except (KeyError, IndexError, TypeError):
            pass # silent fail if structure unexpected

    await status_msg.edit_text(
        f"✅ Spotify linked! 🎉{rp_info} Try /recommend for personalized music!"
    )
    return ConversationHandler.END

async def spotify_code_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
    """Handle /spotify_code command. If called globally, it attempts to process the code."""
    if not context.args:
        await update.message.reply_text(
            "Please provide the Spotify authorization code after the command. Example:\n`/spotify_code YOUR_CODE_HERE`"
        )
        # If this command is configured in a state (it is for SPOTIFY_CODE state of spotify_conv_handler),
        # returning the state value keeps the conversation in that state.
        # If called globally and not as part of an active conversation, its return may not matter much
        # unless it's an entry point to a new conversation.
        # The actual logic inside spotify_code_handler determines the real outcome (END or stay in SPOTIFY_CODE).
        # Returning SPOTIFY_CODE here when no args, implies if it was *somehow* reached while conv active & no args, stay.
        # It is safer for the specific handler (spotify_code_handler) to always determine the state.
        # However, this func is registered globally and as a state handler.
        # The one in the state map (spotify_code_handler for CommandHandler('spotify_code', ...)) will take precedence if state is active.
        # So this global one handles the case when no conversation is active or not in SPOTIFY_CODE state.
        # Let's just make it return None if no args, and delegate if args are present.
        return None 
    
    # Pass the handling to the main spotify_code_handler which understands state returns
    return await spotify_code_handler(update, context)


async def cancel_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel Spotify linking."""
    query = update.callback_query
    await query.answer() 
    await query.edit_message_text("Spotify linking cancelled. Try again with /link_spotify. 👍")
    return ConversationHandler.END


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /search command for YouTube."""
    if not context.args:
        await update.message.reply_text("What song to search? Example:\n`/search Shape of You Ed Sheeran`")
        return

    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Searching YouTube for: '<i>{query}</i>'...", parse_mode=ParseMode.HTML)
    
    results = await asyncio.to_thread(search_youtube_sync, query, max_results=5)
    
    try:
        await status_msg.delete() 
    except Exception: pass # Message might have already been interacted with

    await send_search_results(update, query, results)

async def auto_download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /autodownload command: searches YouTube and downloads the first result."""
    if not context.args:
        await update.message.reply_text("What song to auto-download? Example:\n`/autodownload Imagine Dragons Believer`")
        return

    query = " ".join(context.args)
    await auto_download_first_result(update, context, query)

async def get_lyrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle lyrics requests via /lyrics command."""
    if not context.args:
        await update.message.reply_text(
            "Song for lyrics? Examples:\n"
            "`/lyrics Bohemian Rhapsody`\n"
            "`/lyrics Queen - Bohemian Rhapsody`"
        )
        return

    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Searching lyrics: \"<i>{query}</i>\"...", parse_mode=ParseMode.HTML)

    try:
        artist = None
        song_title = query
        if " - " in query:
            parts = query.split(" - ", 1)
            artist, song_title = parts[0].strip(), parts[1].strip()
        elif " by " in query.lower(): 
            match = re.search(r'^(.*?)\s+by\s+(.*?)$', query, re.IGNORECASE)
            if match:
                song_title, artist = match.group(1).strip(), match.group(2).strip()
        
        logger.info(f"Lyrics query: song='{song_title}', artist='{artist}'")
        
        lyrics_text = await asyncio.to_thread(get_lyrics_sync, song_title, artist)
        
        max_len = 4090 # Keep some buffer from 4096
        if len(lyrics_text) > max_len:
            first_chunk = lyrics_text[:max_len]
            # Find last double newline to make a clean break
            cut_point = first_chunk.rfind('\n\n')
            if cut_point == -1 : cut_point = first_chunk.rfind('\n') # fallback to single newline
            if cut_point == -1 or cut_point < max_len - 500 : cut_point = max_len # If no good cut point, just cut
            
            await status_msg.edit_text(f"{lyrics_text[:cut_point]}\n\n<small>(Lyrics too long, continued below)</small>", parse_mode=ParseMode.HTML)
            remaining_lyrics = lyrics_text[cut_point:]
            
            while remaining_lyrics:
                chunk_to_send = remaining_lyrics[:max_len]
                remaining_lyrics = remaining_lyrics[max_len:]
                
                cut_point_chunk = chunk_to_send.rfind('\n\n')
                if cut_point_chunk == -1 : cut_point_chunk = chunk_to_send.rfind('\n')
                if cut_point_chunk == -1 or cut_point_chunk < max_len - 500 : cut_point_chunk = max_len

                final_chunk_part = chunk_to_send[:cut_point_chunk]
                remaining_lyrics = chunk_to_send[cut_point_chunk:] + remaining_lyrics


                chunk_message = final_chunk_part + ("\n\n<small>(...continued)</small>" if remaining_lyrics.strip() else "")
                if chunk_message.strip(): # Only send if there's content
                    await update.message.reply_text(chunk_message, parse_mode=ParseMode.HTML)
        else:
            await status_msg.edit_text(lyrics_text, parse_mode=ParseMode.HTML)

    except Exception as e: 
        logger.error(f"Error in get_lyrics_command (query '{query}'): {e}", exc_info=True)
        await status_msg.edit_text("Sorry, unexpected hiccup fetching lyrics. 😕 Try again.")


async def recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provide music recommendations. (Alias for smart_recommend_music now)."""
    await smart_recommend_music(update, context)


async def provide_generic_recommendations(update: Update, mood: str, chat_id_override: Optional[int] = None) -> None:
    """Provide generic, hardcoded recommendations as a fallback."""
    logger.info(f"Providing generic recommendations for mood: {mood}")
    target_chat_id = chat_id_override or update.effective_chat.id

    mood_recommendations = {
        "happy": ["Uptown Funk - Mark Ronson", "Happy - Pharrell Williams", "Walking on Sunshine - Katrina & The Waves"],
        "sad": ["Someone Like You - Adele", "Hallelujah - Leonard Cohen (Jeff Buckley version)", "Fix You - Coldplay"],
        "energetic": ["Don't Stop Me Now - Queen", "Thunderstruck - AC/DC", "Can't Stop the Feeling! - Justin Timberlake"],
        "relaxed": ["Weightless - Marconi Union", "Clair de Lune - Debussy", "Orinoco Flow - Enya"],
        "focused": ["The Four Seasons - Vivaldi", "Time - Hans Zimmer", "Ambient 1: Music for Airports - Brian Eno"],
        "nostalgic": ["Bohemian Rhapsody - Queen", "Yesterday - The Beatles", "Wonderwall - Oasis"],
        "neutral": ["Three Little Birds - Bob Marley", "Here Comes The Sun - The Beatles", "What a Wonderful World - Louis Armstrong"]
    }

    chosen_mood_key = mood.lower()
    if chosen_mood_key not in mood_recommendations:
        logger.warning(f"Generic mood '{mood}' not in list, defaulting to neutral.")
        chosen_mood_key = "neutral" 
        
    recommendations = mood_recommendations.get(chosen_mood_key) # Default to neutral list if mood still not found (should not happen)
    response_text = f"🎵 Some general **{mood.capitalize()}** vibes for you:\n\n"
    for i, track in enumerate(recommendations, 1):
        response_text += f"{i}. {track}\n"
    response_text += "\n💡 <i>You can ask me to search or download any!</i>"
    
    await context.bot.send_message(chat_id=target_chat_id, text=response_text, parse_mode=ParseMode.HTML)


async def set_mood(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start conversation to set mood, part of a ConversationHandler."""
    user = update.effective_user
    user_contexts.setdefault(user.id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})

    keyboard = [
        [InlineKeyboardButton("Happy 😊", callback_data=f"{CB_MOOD_PREFIX}happy"),
         InlineKeyboardButton("Sad 😢", callback_data=f"{CB_MOOD_PREFIX}sad")],
        [InlineKeyboardButton("Energetic 💪", callback_data=f"{CB_MOOD_PREFIX}energetic"),
         InlineKeyboardButton("Relaxed 😌", callback_data=f"{CB_MOOD_PREFIX}relaxed")],
        [InlineKeyboardButton("Focused 🧠", callback_data=f"{CB_MOOD_PREFIX}focused"),
         InlineKeyboardButton("Nostalgic 🕰️", callback_data=f"{CB_MOOD_PREFIX}nostalgic")],
         [InlineKeyboardButton("Neutral / Other", callback_data=f"{CB_MOOD_PREFIX}neutral")],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    current_mood = user_contexts[user.id].get("mood")
    prompt_text = f"Hi {user.first_name}! "
    if current_mood and current_mood != "neutral":
        prompt_text += f"Your mood is **{current_mood}**. Change it or how are you feeling now?"
    else:
        prompt_text += "How are you feeling today?"
    
    # If called from a command, reply. If from a callback that was edited, might need to send new.
    if update.callback_query: # If from a previous button, edit message
        await update.callback_query.edit_message_text(prompt_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
    else: # If from /mood command
        await update.message.reply_text(prompt_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
    return MOOD 

async def send_search_results(update: Update, query: str, results: List[Dict]) -> None:
    """Send YouTube search results with inline keyboard for download."""
    target_chat_id = update.effective_chat.id
    
    if not results:
        await update.message.reply_text(f"😕 Sorry, no YouTube results for '<i>{query}</i>'. Try different keywords?", parse_mode=ParseMode.HTML)
        return

    keyboard_rows = []
    response_text_header = f"🔎 YouTube results for '<i>{query}</i>':\n\n"
    
    valid_results_count = 0
    for i, result in enumerate(results[:5]): 
        if not result.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', result['id']):
            logger.warning(f"Skipping invalid YouTube result ID: {result.get('id', 'N/A')}")
            continue
        valid_results_count +=1

        duration_str = ""
        if result.get('duration') and isinstance(result['duration'], (int, float)) and result['duration'] > 0:
            try:
                minutes = int(result['duration'] // 60)
                seconds = int(result['duration'] % 60)
                duration_str = f" [{minutes}:{seconds:02d}]"
            except TypeError: 
                duration_str = ""
        
        title = result.get('title', 'Unknown Title')
        button_display_title = (title[:35] + "...") if len(title) > 38 else title # Max button text length
        button_text = f"[{valid_results_count}] {button_display_title}{duration_str}"
        
        response_text_header += f"{valid_results_count}. <b>{title}</b> by <i>{result.get('uploader', 'N/A')}</i>{duration_str}\n"
        keyboard_rows.append([InlineKeyboardButton(button_text, callback_data=f"{CB_DOWNLOAD_PREFIX}{result['id']}")])

    if not keyboard_rows: 
        await update.message.reply_text(f"😕 Found YouTube results for '<i>{query}</i>', but had issues creating download options.", parse_mode=ParseMode.HTML)
        return

    keyboard_rows.append([InlineKeyboardButton("Cancel Search", callback_data=CB_CANCEL_SEARCH)])
    reply_markup = InlineKeyboardMarkup(keyboard_rows)
    
    final_text = response_text_header + "\nClick a song to download its audio:"
    await update.message.reply_text(final_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

async def auto_download_first_result(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str, original_message_id_to_edit: Optional[int] = None) -> None:
    """Search YouTube, then automatically download the first valid song result."""
    user_id = update.effective_user.id

    if user_id in active_downloads:
        # If original_message_id_to_edit, it's likely a button press, so edit that message
        if original_message_id_to_edit:
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=original_message_id_to_edit, text="Hold on! You already have a download in progress. Let that finish first. 😊", reply_markup=None)
        else: # From /autodownload command
            await update.message.reply_text("Hold on! You already have a download in progress. Let that finish first. 😊")
        return

    active_downloads.add(user_id)
    status_msg = None # Will hold the message object to edit

    try:
        if original_message_id_to_edit:
            # This means it was likely called from a button. Edit that button's message.
            status_msg = await context.bot.edit_message_text(chat_id=update.effective_chat.id, message_id=original_message_id_to_edit, 
                                             text=f"🔍 Okay, looking for '<i>{query}</i>' to download...", 
                                             parse_mode=ParseMode.HTML, reply_markup=None)
        else: # Called from /autodownload command
            status_msg = await update.message.reply_text(f"🔍 Okay, looking for '<i>{query}</i>' to download...", parse_mode=ParseMode.HTML)

        results = await asyncio.to_thread(search_youtube_sync, query, max_results=1) 
        if not results or not results[0].get('id') or not is_valid_youtube_url(results[0].get('url', '')):
            await status_msg.edit_text(f"❌ Oops! Couldn't find a downloadable track for '<i>{query}</i>'. Maybe try `/search {query}` for more options?", parse_mode=ParseMode.HTML)
            return

        top_result = results[0]
        video_url = top_result["url"]
        video_title = top_result.get("title", "this track")
        
        await status_msg.edit_text(f"✅ Found: <b>{video_title}</b>.\n⏳ Downloading... (this can take a moment!)", parse_mode=ParseMode.HTML)

        download_result = await asyncio.to_thread(download_youtube_audio_sync, video_url)
        
        if not download_result["success"]:
            error_message = download_result.get('error', 'Unknown download error.')
            await status_msg.edit_text(f"❌ Download failed for <b>{video_title}</b>: {error_message}", parse_mode=ParseMode.HTML)
            return

        # After successful download, the status_msg is about to be deleted. Send file, then delete status_msg.
        audio_path = download_result["audio_path"]
        await status_msg.edit_text(f"✅ Downloaded: <b>{download_result['title']}</b>.\n✅ Sending the audio file now...", parse_mode=ParseMode.HTML)

        with open(audio_path, 'rb') as audio_file:
            logger.info(f"Auto-DL: Sending '{download_result['title']}' (user {user_id}). Path: {audio_path}")
            await context.bot.send_audio( # Always send as new message for audio
                chat_id=update.effective_chat.id,
                audio=audio_file,
                title=download_result["title"][:64],
                performer=download_result["artist"][:64] if download_result.get("artist") else "Unknown Artist",
                caption=f"🎵 Here's: {download_result['title']}",
                duration=download_result.get('duration')
            )

        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"Temp file deleted after auto-DL: {audio_path}")
            except OSError as e:
                logger.error(f"Error deleting temp file (auto-DL) {audio_path}: {e}")
        
        try:
            await status_msg.delete() # Clean up the final status message
        except Exception: pass
    
    except (TimedOut, NetworkError) as net_err:
        logger.error(f"Telegram API/Network error (auto-DL user {user_id}, query '{query}'): {net_err}")
        if status_msg: 
           try: await status_msg.edit_text(f"❌ Network problem with '<i>{query}</i>'. Try again.", parse_mode=ParseMode.HTML)
           except: pass
    except Exception as e:
        logger.error(f"Unexpected error (auto-DL user {user_id}, query '{query}'): {e}", exc_info=True)
        if status_msg:
            try: await status_msg.edit_text(f"❌ Unexpected error with '<i>{query}</i>'. My apologies!", parse_mode=ParseMode.HTML)
            except: pass
    finally:
        active_downloads.discard(user_id)


async def send_audio_via_bot(bot, chat_id, audio_path, title, performer, caption, duration):
    """Helper to send audio, using PTB's built-in mechanisms."""
    with open(audio_path, 'rb') as audio_file_obj: # Ensure correct variable name
        await bot.send_audio(
            chat_id=chat_id,
            audio=audio_file_obj, # Pass the file object
            title=title[:64],
            performer=performer[:64] if performer else "Unknown",
            caption=caption,
            duration=duration
        )

async def enhanced_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
    """Handle button callbacks from inline keyboards."""
    query = update.callback_query
    await query.answer() 
    
    data = query.data
    user_id = query.from_user.id
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    
    logger.debug(f"Button: '{data}' for user {user_id}")

    if data.startswith(CB_MOOD_PREFIX):
        mood = data[len(CB_MOOD_PREFIX):]
        user_contexts[user_id]["mood"] = mood
        logger.info(f"User {user_id} set mood: {mood}")

        keyboard = [ 
            [InlineKeyboardButton("Pop", callback_data=f"{CB_PREFERENCE_PREFIX}pop"),
             InlineKeyboardButton("Rock", callback_data=f"{CB_PREFERENCE_PREFIX}rock"),
             InlineKeyboardButton("Hip-Hop", callback_data=f"{CB_PREFERENCE_PREFIX}hiphop")],
            [InlineKeyboardButton("Electronic", callback_data=f"{CB_PREFERENCE_PREFIX}electronic"),
             InlineKeyboardButton("Classical", callback_data=f"{CB_PREFERENCE_PREFIX}classical"),
             InlineKeyboardButton("Jazz", callback_data=f"{CB_PREFERENCE_PREFIX}jazz")],
            [InlineKeyboardButton("Folk", callback_data=f"{CB_PREFERENCE_PREFIX}folk"),
             InlineKeyboardButton("R&B", callback_data=f"{CB_PREFERENCE_PREFIX}rnb"),
             InlineKeyboardButton("Any/Surprise!", callback_data=f"{CB_PREFERENCE_PREFIX}any")],
            [InlineKeyboardButton("Skip Genre", callback_data=f"{CB_PREFERENCE_PREFIX}skip")],
        ]
        await query.edit_message_text(
            f"Got it, {query.from_user.first_name}! You're feeling {mood}. 🎶\nAny genre preference for music now?",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return PREFERENCE 

    elif data.startswith(CB_PREFERENCE_PREFIX):
        preference = data[len(CB_PREFERENCE_PREFIX):]
        msg_text = ""
        if preference == "skip" or preference == "any":
            user_contexts[user_id]["preferences"] = [] 
            msg_text = "Alright! I'll keep that in mind for recommendations."
        else:
            user_contexts[user_id]["preferences"] = [preference] 
            msg_text = f"Great choice! {preference.capitalize()} it is. "
        logger.info(f"User {user_id} set preference: {preference}")
        
        msg_text += " Try:\n➡️ `/recommend`\n➡️ `/search [song]`\n➡️ Or just chat!"
        await query.edit_message_text(msg_text)
        return ConversationHandler.END 

    elif data.startswith(CB_DOWNLOAD_PREFIX): # For direct download from search result etc.
        video_id = data[len(CB_DOWNLOAD_PREFIX):]
        # Use auto_download_first_result logic, but with a known video_id (becomes the query)
        # This reuses the download and send logic nicely
        # We pass query.message.message_id so it edits the search results message
        if not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
             logger.error(f"Invalid YT ID from button: '{video_id}'")
             await query.edit_message_text("❌ Invalid video ID. Try searching again.")
             return None
        # To use auto_download_first_result, the 'query' it searches for becomes the direct URL.
        youtube_direct_url = f"https://www.youtube.com/watch?v={video_id}"
        await auto_download_first_result(update, context, query=youtube_direct_url, original_message_id_to_edit=query.message.message_id)
        return None # No further state transition

    elif data.startswith(CB_AUTO_DOWNLOAD_PREFIX): # From the "Yes, download it" type buttons
        video_id_or_query = data[len(CB_AUTO_DOWNLOAD_PREFIX):] # This could be an ID or a query string now
        
        if re.match(r'^[0-9A-Za-z_-]{11}$', video_id_or_query): # It's a video ID
            youtube_url_for_auto = f"https://www.youtube.com/watch?v={video_id_or_query}"
            await auto_download_first_result(update, context, query=youtube_url_for_auto, original_message_id_to_edit=query.message.message_id)
        else: # It's a query string that was stored in callback
            await auto_download_first_result(update, context, query=video_id_or_query, original_message_id_to_edit=query.message.message_id)
        return None

    elif data.startswith(CB_SHOW_OPTIONS_PREFIX):
        search_query_str = data[len(CB_SHOW_OPTIONS_PREFIX):]
        if not search_query_str:
            await query.edit_message_text("Cannot show options, original query missing.")
            return None
        
        await query.edit_message_text(f"🔍 Showing more YouTube options for '<i>{search_query_str}</i>'...", parse_mode=ParseMode.HTML, reply_markup=None)
        results = await asyncio.to_thread(search_youtube_sync, search_query_str, max_results=5)
        
        try: # Delete the "I found X, download or show options?" msg
             await query.message.delete() 
        except Exception as e:
            logger.warning(f"Could not delete previous message before showing options: {e}")

        # send_search_results expects an 'Update' object. Create a mock or adapt.
        # For simplicity, call a variant or use current query.message as basis.
        await send_search_results(Update(query.update_id, message=query.message), search_query_str, results)
        return None

    elif data == CB_CANCEL_SEARCH:
        await query.edit_message_text("❌ Search cancelled. Feel free to try another!")
        return None
    elif data == CB_CANCEL_SPOTIFY: 
        await query.edit_message_text("Spotify linking cancelled. Use /link_spotify anytime.")
        return ConversationHandler.END 

    logger.warning(f"Unhandled callback data: {data} (user {user_id})")
    return None 


async def enhanced_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enhanced message handler: URL, music keywords, AI chat."""
    if not update.message or not update.message.text: 
        return

    user_id = update.effective_user.id
    text = update.message.text.strip()
    logger.debug(f"Msg (user {user_id}): '{text[:100]}'")

    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})

    # 1. YouTube URLs direct to download
    if is_valid_youtube_url(text):
        logger.info(f"User {user_id} sent YouTube URL: {text}")
        # download_music expects URL either in context.args or as the full message text.
        # To simplify, we can make download_music also parse it directly from update.message.text if context.args is empty.
        # The current download_music is structured to handle /download <url> or being called with text that IS the url.
        # This should work:
        temp_context = ContextTypes.DEFAULT_TYPE(application=context.application, chat_id=user_id, user_id=user_id, bot=context.bot, args=[text])
        await download_music(Update(update.update_id, message=update.message._replace(text=text)), temp_context)
        return

    # 2. AI-based mood detection (subtle background update)
    # Avoid frequent mood re-detection unless significant text. Maybe only if text > N words.
    if len(text.split()) > 2: # Only detect mood on slightly longer messages
        detected_mood_on_message = await detect_mood_from_text(user_id, text)
        if detected_mood_on_message and detected_mood_on_message != "neutral" and detected_mood_on_message != user_contexts[user_id].get("mood"):
            user_contexts[user_id]["mood"] = detected_mood_on_message
            logger.debug(f"Passive mood update for user {user_id} to: {detected_mood_on_message} based on: '{text[:30]}'")


    # 3. AI music request detection for song/artist queries
    # We rely more on AI now to determine if "play X", "song Y" is a request.
    ai_music_eval = await is_music_request(user_id, text)
    if ai_music_eval.get("is_music_request") and ai_music_eval.get("song_query"):
        music_query = ai_music_eval["song_query"]
        status_msg = await update.message.reply_text(f"🎵 You're looking for '<i>{music_query}</i>'? Let me check...", parse_mode=ParseMode.HTML)
        results = await asyncio.to_thread(search_youtube_sync, music_query, max_results=1)
        
        if results and results[0].get('id') and re.match(r'^[0-9A-Za-z_-]{11}$', results[0]['id']):
            top_res = results[0]
            keyboard = [
                [InlineKeyboardButton(f"✅ Yes, download '{top_res['title'][:20]}...'", callback_data=f"{CB_AUTO_DOWNLOAD_PREFIX}{top_res['id']}")],
                [InlineKeyboardButton("👀 More options", callback_data=f"{CB_SHOW_OPTIONS_PREFIX}{music_query}")], 
                [InlineKeyboardButton("❌ No, cancel", callback_data=CB_CANCEL_SEARCH)]
            ]
            await status_msg.edit_text(
                f"I found: <b>{top_res['title']}</b> by <i>{top_res.get('uploader', 'N/A')}</i>.\nDownload or see more?",
                reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML
            )
        else:
            await status_msg.edit_text(f"😕 Couldn't find a specific track for '<i>{music_query}</i>'. Try `/search {music_query}` for more options or be more specific!", parse_mode=ParseMode.HTML)
        return

    # 4. Lyrics request detection (more explicit trigger, e.g. "lyrics for X")
    lyrics_keywords_precise = ["lyrics for", "lyrics to", "get lyrics", "find lyrics for"]
    lyrics_query = None
    text_lower = text.lower()
    for keyword in lyrics_keywords_precise:
        if text_lower.startswith(keyword):
            lyrics_query_candidate = text[len(keyword):].strip()
            if lyrics_query_candidate: # Make sure there's something after keyword
                lyrics_query = lyrics_query_candidate
                logger.info(f"Heuristic lyrics request detected: '{lyrics_query}' for user {user_id}")
                break
    if lyrics_query:
        # Prepare mock context.args for get_lyrics_command
        # For lyrics "Song by Artist", get_lyrics_command will parse it.
        await get_lyrics_command(update, ContextTypes.DEFAULT_TYPE(application=context.application, chat_id=user_id, user_id=user_id, bot=context.bot, args=lyrics_query.split()))
        return

    # 5. Fallback to general AI chat
    # Add a small delay if AI is about to be called, so "thinking" message feels natural
    await asyncio.sleep(0.2) 
    typing_msg = await update.message.reply_text("<i>MelodyMind is thinking...</i> 🎶", parse_mode=ParseMode.HTML)
    try:
        response_text = await generate_chat_response(user_id, text)
        await typing_msg.edit_text(response_text) # Edits the "thinking..." message
    except (TimedOut, NetworkError) as net_err:
        logger.error(f"Network error (AI chat user {user_id}): {net_err}")
        await typing_msg.edit_text("Sorry, network hiccup. Could you try again?")
    except Exception as e:
        logger.error(f"Error in AI chat response (user {user_id}): {e}", exc_info=True)
        await typing_msg.edit_text("A little tangled up right now! 😅 Let's try later.")


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear user's conversation history with the bot."""
    user_id = update.effective_user.id
    if user_id in user_contexts and "conversation_history" in user_contexts[user_id] and user_contexts[user_id]["conversation_history"]:
        user_contexts[user_id]["conversation_history"] = []
        logger.info(f"Cleared conversation history for user {user_id}")
        await update.message.reply_text("✅ Our chat history is cleared.")
    else:
        await update.message.reply_text("No chat history to clear yet! 😊")

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Generic cancel handler for ConversationHandlers."""
    await update.message.reply_text("Okay, cancelled. Chat or use commands anytime! 👍")
    return ConversationHandler.END

async def analyze_conversation(user_id: int) -> Dict:
    """Analyze conversation history and Spotify data using AI for preferences."""
    default_return = {"genres": user_contexts.get(user_id, {}).get("preferences", []), 
                      "artists": [], 
                      "mood": user_contexts.get(user_id, {}).get("mood")}
    if not client: 
        return default_return

    user_ctx = user_contexts.get(user_id, {})
    user_ctx.setdefault("preferences", [])
    user_ctx.setdefault("conversation_history", [])
    user_ctx.setdefault("spotify", {}).setdefault("recently_played", [])
    user_ctx["spotify"].setdefault("top_tracks", [])

    if len(user_ctx["conversation_history"]) < 2 and not user_ctx["spotify"]["recently_played"] and not user_ctx["spotify"]["top_tracks"]: # Stricter check for data
        logger.info(f"Insufficient data for AI analysis (user {user_id}).")
        return default_return

    logger.info(f"AI conversation analysis for user {user_id}")
    try:
        history_summary_parts = [f"{msg['role']}: {msg['content'][:80]}" for msg in user_ctx["conversation_history"][-6:]] 
        conversation_text_summary = "\n".join(history_summary_parts)

        spotify_summary_parts = []
        if user_ctx["spotify"]["recently_played"]:
            try:
                tracks = user_ctx["spotify"]["recently_played"]
                spotify_summary_parts.append("Recently played: " + ", ".join(
                    [f"'{item['track']['name']}' by {item['track']['artists'][0]['name']}" for item in tracks[:2] if item.get("track")] 
                ))
            except: pass 
        if user_ctx["spotify"]["top_tracks"]:
            try:
                tracks = user_ctx["spotify"]["top_tracks"]
                spotify_summary_parts.append("Top tracks: " + ", ".join(
                    [f"'{item['name']}' by {item['artists'][0]['name']}" for item in tracks[:2] if item.get("artists")] 
                ))
            except: pass
        spotify_summary = ". ".join(spotify_summary_parts)
        
        if not conversation_text_summary and not spotify_summary:
             logger.info(f"Not enough text/spotify summary for AI analysis (user {user_id})")
             return default_return

        prompt_user_content = (
            f"Conversation Summary:\n{conversation_text_summary}\n\n"
            f"Spotify Data Summary (if any):\n{spotify_summary}\n\n"
            f"User's set mood: {user_ctx.get('mood', 'Not set')}\n"
            f"User's set preferences: {', '.join(user_ctx.get('preferences',[])) if user_ctx.get('preferences') else 'Not set'}"
        )
        prompt_messages = [
            {"role": "system", "content": 
                "Analyze user's chat and Spotify data. Infer music preferences (genres, artists) and current/recent mood. "
                "JSON output: 'genres' (list, max 2), 'artists' (list, max 2 relevant from data/chat), 'mood' (string, or null). "
                "Prioritize explicit statements. If no strong signals, return empty lists or null mood."
            },
            {"role": "user", "content": prompt_user_content }
        ]

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo-0125", 
            messages=prompt_messages,
            max_tokens=150, # Reduced tokens
            temperature=0.1, # Lower temp
            response_format={"type": "json_object"}
        )

        result_str = response.choices[0].message.content
        result = json.loads(result_str)

        if not isinstance(result, dict):
            logger.error(f"AI analysis (user {user_id}) non-dict: {result_str}")
            return default_return

        inferred_genres = result.get("genres", [])
        if isinstance(inferred_genres, str): inferred_genres = [g.strip() for g in inferred_genres.split(",") if g.strip()]
        if not isinstance(inferred_genres, list): inferred_genres = []
        
        inferred_artists = result.get("artists", [])
        if isinstance(inferred_artists, str): inferred_artists = [a.strip() for a in inferred_artists.split(",") if a.strip()]
        if not isinstance(inferred_artists, list): inferred_artists = []

        inferred_mood = result.get("mood")
        if not isinstance(inferred_mood, str) or not inferred_mood.strip() or inferred_mood.lower() == "null": inferred_mood = None # Treat "null" string as None
        
        # Update user_contexts with AI inferred data (carefully)
        # Only update if AI is confident or user hasn't set.
        if inferred_genres and (not user_ctx.get("preferences") or set(inferred_genres) != set(user_ctx.get("preferences",[]))):
            user_ctx["preferences"] = list(set(inferred_genres[:2])) 
        
        if inferred_mood and (inferred_mood != user_ctx.get("mood") or not user_ctx.get("mood")):
            user_ctx["mood"] = inferred_mood

        logger.info(f"AI analysis (user {user_id}) output: Genres={user_ctx['preferences']}, Mood={user_ctx['mood']}, Artists={inferred_artists[:2]}")
        return {
            "genres": user_ctx["preferences"], 
            "artists": inferred_artists[:2], 
            "mood": user_ctx["mood"]
        }

    except json.JSONDecodeError as jde:
        logger.error(f"AI analysis JSON decode error (user {user_id}): {jde}. Raw: {response.choices[0].message.content if 'response' in locals() and response.choices else 'N/A'}")
    except Exception as e:
        logger.error(f"Error in AI analyze_conversation for user {user_id}: {e}", exc_info=False)
    
    return default_return

async def smart_recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provide smarter music recommendations."""
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    status_msg = await update.message.reply_text(f"🎵 Thinking of music for you, {user_name}...")

    try:
        if user_contexts.get(user_id, {}).get("spotify", {}).get("access_token"): 
            logger.info(f"Fetching latest Spotify data (user {user_id}) for smart recs.")
            rp_task = get_user_spotify_data(user_id, "player/recently-played", params={"limit": 5}) # Fetch more for context
            tt_task = get_user_spotify_data(user_id, "top/tracks", params={"limit": 5, "time_range": "short_term"})
            recently_played, top_tracks = await asyncio.gather(rp_task, tt_task)
            if recently_played: user_contexts[user_id]["spotify"]["recently_played"] = recently_played
            if top_tracks: user_contexts[user_id]["spotify"]["top_tracks"] = top_tracks
        
        analysis = await analyze_conversation(user_id)
        current_mood = analysis.get("mood")
        
        if not current_mood or current_mood == "neutral": # neutral considered as "not set" for direct recs
            await status_msg.delete()
            logger.info(f"Mood not clear (user {user_id}), prompting for smart recs.")
            await set_mood(update, context) # Call the mood setting conv handler
            return

        await status_msg.edit_text(f"Okay {user_name}, for your **{current_mood}** mood...\nLooking for recommendations... 🎧", parse_mode=ParseMode.MARKDOWN)

        seed_track_ids, seed_artist_ids, seed_genre_list = [], [], analysis.get("genres", [])
        
        spotify_user_ctx = user_contexts.get(user_id, {}).get("spotify", {})
        if spotify_user_ctx.get("access_token"):
            # Prioritize AI-identified artists if they match Spotify history or are just good seeds
            ai_artists = analysis.get("artists", [])
            if ai_artists and spotify_user_ctx.get("top_tracks"): # Try to find IDs for AI artists from top tracks
                for art_name in ai_artists:
                    for track in spotify_user_ctx["top_tracks"]:
                        if any(a['name'].lower() == art_name.lower() for a in track.get('artists',[])):
                            seed_artist_ids.append(track['artists'][0]['id']) # Add first artist ID
                            break # Found artist
                    if len(seed_artist_ids) >=2 : break # Max 2 artist seeds
            
            if not seed_artist_ids and spotify_user_ctx.get("recently_played"): # Fallback to recent tracks
                seed_track_ids.extend([t["track"]["id"] for t in spotify_user_ctx["recently_played"][:2] if t.get("track") and t["track"].get("id")])

        spotify_client_token = await get_spotify_token()
        if spotify_client_token and (seed_track_ids or seed_artist_ids or seed_genre_list):
            logger.info(f"Spotify API recs (user {user_id}) seeds: tracks={seed_track_ids}, artists={seed_artist_ids}, genres={seed_genre_list}")
            
            recs = await get_spotify_recommendations(spotify_client_token, 
                                                     seed_tracks=seed_track_ids[:2], 
                                                     seed_genres=seed_genre_list[:1], 
                                                     seed_artists=seed_artist_ids[:2], limit=5)
            
            if recs:
                resp_html = f"🎵 Spotify recs for your **{current_mood}** mood, {user_name}:\n\n"
                kb_spotify = []
                for i, track in enumerate(recs, 1):
                    artists_str = ", ".join(a["name"] for a in track.get("artists",[]))
                    album_str = track.get("album", {}).get("name", "")
                    track_info = f"<b>{track['name']}</b> by <i>{artists_str}</i>" + (f" ({album_str})" if album_str else "")
                    resp_html += f"{i}. {track_info}\n"
                    yt_query = f"{track['name']} {artists_str}"
                    kb_spotify.append([InlineKeyboardButton(f"YT: {track['name'][:20]}...", callback_data=f"{CB_SHOW_OPTIONS_PREFIX}{yt_query}")])
                resp_html += "\n💡 <i>Click to search on YouTube.</i>"
                await status_msg.edit_text(resp_html, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(kb_spotify))
                return

        # Fallback: YouTube search
        yt_query_parts = [current_mood]
        if seed_genre_list: yt_query_parts.append(seed_genre_list[0]) 
        ai_artists_from_analysis = analysis.get("artists", [])
        if ai_artists_from_analysis: yt_query_parts.append(f"like {ai_artists_from_analysis[0]}") 
        youtube_search_query = " ".join(yt_query_parts) + " music"
        logger.info(f"YouTube fallback recs (user {user_id}) query: '{youtube_search_query}'")
        await status_msg.edit_text(f"Searching YouTube for **{current_mood}** tracks like '<i>{youtube_search_query}</i>'...", parse_mode=ParseMode.HTML)

        yt_results = await asyncio.to_thread(search_youtube_sync, youtube_search_query, max_results=5)
        if yt_results:
            resp_html_yt = f"🎵 YouTube suggestions for **{current_mood}** mood, {user_name}:\n\n"
            kb_yt = []
            valid_yt_count = 0
            for i, res in enumerate(yt_results, 1):
                if not res.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', res['id']): continue
                valid_yt_count +=1
                dur = res.get('duration', 0)
                dur_str = ""
                if dur and isinstance(dur, (int,float)) and dur > 0: 
                    m, s = divmod(int(dur), 60)
                    dur_str = f" [{m}:{s:02d}]"
                resp_html_yt += f"{valid_yt_count}. <b>{res['title']}</b> - <i>{res.get('uploader', 'N/A')}</i>{dur_str}\n"
                btn_title = res['title'][:30] + "..." if len(res['title']) > 30 else res['title']
                kb_yt.append([InlineKeyboardButton(f"DL: {btn_title}", callback_data=f"{CB_DOWNLOAD_PREFIX}{res['id']}")])
            
            if not kb_yt: 
                 await status_msg.delete()
                 await provide_generic_recommendations(update, current_mood, chat_id_override=user_id)
                 return
            resp_html_yt += "\n💡 <i>Click to download audio.</i>"
            await status_msg.edit_text(resp_html_yt, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(kb_yt))
        else: 
            logger.info(f"No YT results for '{youtube_search_query}', generic recs for {current_mood}.")
            await status_msg.delete() 
            await provide_generic_recommendations(update, current_mood, chat_id_override=user_id)

    except Exception as e:
        logger.error(f"Error in smart_recommend_music (user {user_id}): {e}", exc_info=True)
        await status_msg.edit_text(f"Snag finding recs, {user_name}! 😥 Try again later.")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates and send user-friendly message."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    
    # User-facing error message
    error_message_text = "😓 Oops! Something went wrong. My developers have been notified. Please try again in a bit!"
    if isinstance(context.error, TimedOut):
        error_message_text = "🐢 Things are a bit slow, the operation timed out. Please try again!"
    elif isinstance(context.error, NetworkError):
         error_message_text = "📡 I'm having trouble connecting. Please check your connection or try again later."

    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(error_message_text)
        except Exception as e_reply:
            logger.error(f"Failed to send error reply to user: {e_reply}")
    elif isinstance(update, Update) and update.callback_query:
        try:
            await update.callback_query.message.reply_text(error_message_text) # Reply to the message associated with callback
        except Exception as e_reply_cb:
            logger.error(f"Failed to send error reply for callback to user: {e_reply_cb}")


def cleanup_downloads_atexit() -> None:
    """Clean up temporary audio files from DOWNLOAD_DIR on exit."""
    logger.info("Cleaning up temporary download files at exit...")
    cleaned_count = 0
    try:
        if os.path.exists(DOWNLOAD_DIR):
            for item_name in os.listdir(DOWNLOAD_DIR):
                item_path = os.path.join(DOWNLOAD_DIR, item_name)
                try:
                    if os.path.isfile(item_path) and (item_path.endswith((".m4a", ".mp3", ".webm", ".ogg", ".opus")) or "voice_" in item_name): # Be more specific
                        os.remove(item_path)
                        cleaned_count +=1
                except Exception as e:
                    logger.error(f"Failed to remove {item_path}: {e}")
            if cleaned_count > 0:
                logger.info(f"Cleaned {cleaned_count} file(s) from '{DOWNLOAD_DIR}'.")
            else:
                logger.info(f"No specific temp files to clean in '{DOWNLOAD_DIR}'.")
        else:
            logger.info(f"Download directory '{DOWNLOAD_DIR}' not found, no cleanup needed at exit.")
    except Exception as e:
        logger.error(f"Error during atexit cleanup of downloads directory: {e}")

def signal_exit_handler(sig, frame) -> None:
    """Handle termination signals gracefully."""
    logger.info(f"Received signal {sig}, preparing for graceful exit...")
    # cleanup_downloads_atexit() is registered with atexit, so it should run on normal sys.exit().
    # Explicitly call if worried about all atexit scenarios.
    if sig in [signal.SIGINT, signal.SIGTERM]: # If it's a kill signal, ensure cleanup attempt
        cleanup_downloads_atexit() 
    sys.exit(0)

def main() -> None:
    """Start the bot."""
    application = (
        Application.builder()
        .token(TOKEN)
        .connect_timeout(15.0)  # Increased
        .read_timeout(30.0)     
        .write_timeout(45.0)    
        .pool_timeout(120.0)    # Increased for potentially larger files / slower connections
        .rate_limiter(AIORateLimiter(overall_max_rate=20, max_retries=3)) # Stricter rate limits
        .build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("download", download_music))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("autodownload", auto_download_command))
    application.add_handler(CommandHandler("lyrics", get_lyrics_command))
    application.add_handler(CommandHandler("recommend", smart_recommend_music))
    application.add_handler(CommandHandler("create_playlist", create_playlist))
    application.add_handler(CommandHandler("clear", clear_history))
    # Global /spotify_code handler (if called outside conversation)
    application.add_handler(CommandHandler("spotify_code", spotify_code_command))


    spotify_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("link_spotify", link_spotify)],
        states={
            SPOTIFY_CODE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, spotify_code_handler),
                # This CommandHandler for /spotify_code within the state will take precedence
                # if the conversation is in SPOTIFY_CODE state.
                CommandHandler("spotify_code", spotify_code_handler), 
                CallbackQueryHandler(cancel_spotify, pattern=f"^{CB_CANCEL_SPOTIFY}$")
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        conversation_timeout=timedelta(minutes=10).total_seconds() 
    )
    application.add_handler(spotify_conv_handler)

    mood_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("mood", set_mood)],
        states={
            MOOD: [CallbackQueryHandler(enhanced_button_handler, pattern=f"^{CB_MOOD_PREFIX}")],
            PREFERENCE: [CallbackQueryHandler(enhanced_button_handler, pattern=f"^{CB_PREFERENCE_PREFIX}")],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        conversation_timeout=timedelta(minutes=5).total_seconds()
    )
    application.add_handler(mood_conv_handler)

    application.add_handler(MessageHandler(filters.VOICE & ~filters.COMMAND, handle_voice))
    application.add_handler(CallbackQueryHandler(enhanced_button_handler)) 
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, enhanced_handle_message))
    
    application.add_error_handler(error_handler)

    signal.signal(signal.SIGINT, signal_exit_handler) 
    signal.signal(signal.SIGTERM, signal_exit_handler) 
    atexit.register(cleanup_downloads_atexit)

    logger.info("🚀 Starting MelodyMind Bot... Connecting to Telegram.")
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True) # Drop pending if any from previous crashes
    except Exception as e:
        logger.critical(f"Bot polling failed to start or crashed critically: {e}", exc_info=True)
    finally:
        logger.info(" MelodyMind Bot has shut down gracefully.")

if __name__ == "__main__":
    if not TOKEN:
        logger.critical("TELEGRAM_TOKEN is MISSING. Bot cannot start.")
        sys.exit(1)
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set. AI features will be degraded or disabled.")
    # Add more checks for SPOTIFY keys if essential for core functionality on startup.
    
    main()