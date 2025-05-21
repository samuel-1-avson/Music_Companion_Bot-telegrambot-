\
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
    def alru_cache(maxsize=128, typed=False): # type: ignore
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
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=15, retries=3) if GENIUS_ACCESS_TOKEN and lyricsgenius else None

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
active_downloads: set[int] = set() # Tracks user_ids with active downloads
user_contexts: Dict[int, Dict] = {} # In-memory user context store
logger.warning("User contexts are stored in-memory and will be lost on bot restart. Consider implementing persistence (e.g., pickle, shelve, DB).")
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20) # Increased default timeout

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
        logger.error(f"Unexpected error getting Spotify client_credentials token: {e}", exc_info=True)
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
        logger.error(f"Unexpected error searching Spotify track '{query}': {e}", exc_info=True)
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def get_spotify_recommendations(token: str, seed_tracks: List[str], limit: int = 5, seed_genres: Optional[List[str]] = None, seed_artists: Optional[List[str]] = None) -> List[Dict]:
    """Get track recommendations from Spotify."""
    if not token:
        logger.warning("No token provided for Spotify recommendations.")
        return []
    
    params: Dict[str, Any] = {"limit": limit}
    seed_count = 0
    if seed_tracks:
        current_seed_values = seed_tracks[:max(0, 5-seed_count)]
        if current_seed_values:
            params["seed_tracks"] = ",".join(current_seed_values)
            seed_count += len(current_seed_values)
    if seed_genres and seed_count < 5:
        current_seed_values = seed_genres[:max(0, 5-seed_count)]
        if current_seed_values:
            params["seed_genres"] = ",".join(current_seed_values)
            seed_count += len(current_seed_values)
    if seed_artists and seed_count < 5:
        current_seed_values = seed_artists[:max(0, 5-seed_count)]
        if current_seed_values:
            params["seed_artists"] = ",".join(current_seed_values)
            seed_count += len(current_seed_values)


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
        logger.error(f"Unexpected error getting Spotify recommendations (params: {params}): {e}", exc_info=True)
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
        logger.error(f"Unexpected error getting user Spotify token for user {user_id}: {e}", exc_info=True)
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
        if "spotify" in user_contexts.get(user_id, {}): # Check if "spotify" key exists before trying to set it.
             if isinstance(user_contexts[user_id], dict): # Ensure user_contexts[user_id] is a dict
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
                
                expires_at = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600) - 60)).timestamp() 
                new_refresh_token_str = token_data.get("refresh_token", refresh_token_str) # Spotify may issue a new refresh token

                # Ensure structure before assignment
                user_contexts.setdefault(user_id, {}).setdefault("spotify", {})
                user_contexts[user_id]["spotify"]["access_token"] = cipher.encrypt(new_access_token.encode())
                user_contexts[user_id]["spotify"]["refresh_token"] = cipher.encrypt(new_refresh_token_str.encode())
                user_contexts[user_id]["spotify"]["expires_at"] = expires_at
                
                return new_access_token
    except aiohttp.ClientError as e:
        logger.error(f"Error refreshing Spotify token for user {user_id}: {e}")
        if hasattr(e, 'status') and e.status == 400: # HTTP 400 can mean 'invalid_grant' (e.g. revoked token)
             logger.error(f"Spotify refresh token for user {user_id} might be revoked. Response: {getattr(e, 'message', '')}. Re-authentication needed.")
             if "spotify" in user_contexts.get(user_id, {}):
                user_contexts[user_id]["spotify"] = {} 
        return None
    except Exception as e:
        logger.error(f"Unexpected error refreshing Spotify token for user {user_id}: {e}", exc_info=True)
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
    request_params: Dict[str, Any] = {"limit": 10, **(params or {})} 

    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.get(url, headers=headers, params=request_params) as response:
                response.raise_for_status()
                return (await response.json()).get("items", [])
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching Spotify user data ('{endpoint}') for user {user_id}: {e}")
        if hasattr(e, 'status') and e.status == 401: 
            logger.info(f"Spotify token unauthorized for user {user_id} for '{endpoint}'. May need re-link.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching Spotify user data ('{endpoint}') for user {user_id}: {e}", exc_info=True)
        return None

# ==================== YOUTUBE HELPER FUNCTIONS ====================

def is_valid_youtube_url(url: str) -> bool:
    """Check if the URL is a valid YouTube URL."""
    if not url:
        return False
    # Regex to match various YouTube URL formats including shorts
    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    ]
    return any(re.search(pattern, url) for pattern in patterns)

def sanitize_filename(filename: str) -> str:
    """Remove invalid characters from filenames for display or metadata."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return sanitized[:100] # Truncate to a reasonable length

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
def download_youtube_audio_sync(url: str) -> Dict[str, Any]: 
    """Download audio from a YouTube video. This is a BLOCKING function."""
    logger.info(f"Attempting to download audio from: {url}")
    
    video_id_match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11})', url)
    video_id = video_id_match.group(1) if video_id_match else "UnknownID"
    unique_filename_base = f"{video_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}"


    try:
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best', 
            'outtmpl': os.path.join(DOWNLOAD_DIR, f'{unique_filename_base}.%(ext)s'), # Unique temp name
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            'max_filesize': 50 * 1024 * 1024, # 50MB limit (Telegram's typical limit for bots sending files)
            'writethumbnail': False, 
            'restrictfilenames': True, 
            'sleep_interval_requests': 1, 
            'sleep_interval':1,
        }
        final_audio_path = None
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False) 
            if not info:
                logger.error(f"Could not extract video information for {url} (ID: {video_id})")
                return {"success": False, "error": "Could not extract video information"}

            # Use title and artist from metadata for display, but unique_filename_base for actual file
            display_title = sanitize_filename(info.get('title', 'Unknown Title'))
            artist = sanitize_filename(info.get('artist', info.get('uploader', 'Unknown Artist')))
            
            # This will download to a filename like 'VideoID_Timestamp.m4a' (or other extension)
            logger.info(f"Downloading '{display_title}' by '{artist}' (ID: {video_id})")
            download_info = ydl.extract_info(url, download=True) # This will perform the download
            
            # Find the actual downloaded file path from 'requested_downloads' or by listing dir for unique_filename_base.*
            # yt-dlp >= 2023.06.22 should populate 'filepath' in 'requested_downloads' correctly.
            # For older versions or fallback:
            if download_info and 'requested_downloads' in download_info and download_info['requested_downloads']:
                 final_audio_path = download_info['requested_downloads'][0].get('filepath')

            if not final_audio_path or not os.path.exists(final_audio_path):
                # Fallback: Search for file starting with unique_filename_base in DOWNLOAD_DIR
                for f_name in os.listdir(DOWNLOAD_DIR):
                    if f_name.startswith(unique_filename_base):
                        final_audio_path = os.path.join(DOWNLOAD_DIR, f_name)
                        logger.info(f"File found by fallback search: {final_audio_path}")
                        break
                if not final_audio_path or not os.path.exists(final_audio_path):
                    logger.error(f"Downloaded file not found for base '{unique_filename_base}' from {url}")
                    return {"success": False, "error": "Downloaded file not found after download process."}
            
            file_size_mb = os.path.getsize(final_audio_path) / (1024 * 1024)
            if file_size_mb > 50.5: # Small buffer for Telegram's 50MB limit
                os.remove(final_audio_path)
                logger.warning(f"File '{display_title}' exceeded 50MB limit ({file_size_mb:.2f}MB), removing: {final_audio_path}")
                return {"success": False, "error": "File exceeds 50 MB Telegram limit after download"}
            
            thumbnail_url = info.get('thumbnail') 

            return {
                "success": True,
                "title": display_title,
                "artist": artist,
                "thumbnail_url": thumbnail_url, 
                "duration": info.get('duration', 0),
                "audio_path": final_audio_path # This is the actual path
            }
    except yt_dlp.utils.DownloadError as de:
        logger.error(f"yt-dlp DownloadError for {url} (ID: {video_id}): {de}")
        error_msg = str(de)
        if "Unsupported URL" in error_msg: return {"success": False, "error": "Unsupported URL."}
        if "Video unavailable" in error_msg: return {"success": False, "error": "Video unavailable."}
        if "is not available" in error_msg: return {"success": False, "error": "Video not available."}
        if "Private video" in error_msg: return {"success": False, "error": "This is a private video."}
        if "Premiere" in error_msg: return {"success": False, "error": "Cannot download waiting Premieres."}
        # Add more specific yt-dlp error checks if needed
        return {"success": False, "error": f"Download failed: {error_msg[:100]}"} # Truncate long errors
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
            'extract_flat': 'discard_in_playlist', # Important: get individual videos if search matches a playlist
            'default_search': f'ytsearch{max_results}', 
            'noplaylist': True, 
            'sleep_interval_requests': 1, # Be polite to YouTube
            'sleep_interval': 1,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False) # download=False implies search
            
            if not info or 'entries' not in info:
                logger.warning(f"No YouTube search results for query: '{query}'")
                return []
            
            results = []
            for entry in info['entries']:
                if not entry: continue # Skip empty entries
                # Ensure basic validity of an entry as a video result
                video_id = entry.get('id')
                if not video_id or not entry.get('title') or not (entry.get('webpage_url') or entry.get('url')):
                    logger.debug(f"Skipping malformed search entry for '{query}': {entry.get('title', 'No Title')}")
                    continue
                
                results.append({
                    'title': entry.get('title', 'Unknown Title'),
                    'url': entry.get('webpage_url') or entry.get('url') or f"https://www.youtube.com/watch?v={video_id}",
                    'thumbnail': entry.get('thumbnail') or (entry.get('thumbnails')[0]['url'] if entry.get('thumbnails') else None),
                    'uploader': entry.get('uploader', 'Unknown Artist'),
                    'duration': entry.get('duration', 0),
                    'id': video_id 
                })
            logger.info(f"Found {len(results)} results for '{query}'")
            return results
            
    except yt_dlp.utils.DownloadError as de: # This can happen if search itself fails for some reason
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

    context["conversation_history"] = context["conversation_history"][-12:]  # Last 6 exchanges for context

    system_prompt = (
        "You are MelodyMind, a friendly, empathetic music companion bot. Focus on brief (1-3 sentences), warm, natural conversation about music and feelings. "
        "If the user asks for music, guide them to use commands (like /search or /autodownload song name) or suggest you can search if they name a song. "
        "Use user context (mood, prefs, Spotify artists) subtly to personalize. "
        "Do not suggest specific commands unless the user explicitly asks how to get a song or asks for help."
    )
    messages = [{"role": "system", "content": system_prompt}]

    context_summary_parts = []
    if context.get("mood"):
        context_summary_parts.append(f"User's mood: {context.get('mood')}.")
    if context.get("preferences"):
        context_summary_parts.append(f"User's preferred genres: {', '.join(context.get('preferences'))}.")
    
    if "spotify" in context and context["spotify"].get("recently_played"):
        try:
            artists = list(set(item["track"]["artists"][0]["name"] for item in context["spotify"]["recently_played"][:3] if item.get("track") and item["track"].get("artists")))
            if artists:
                context_summary_parts.append(f"User recently listened to artists like: {', '.join(artists)}.")
        except Exception: 
            pass # Silent fail on parsing Spotify data for summary
            
    if context_summary_parts:
        messages.append({"role": "system", "content": "Relevant User Info: " + " ".join(context_summary_parts)})

    for hist_msg in context["conversation_history"]:
        messages.append(hist_msg)
    messages.append({"role": "user", "content": message})

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4-turbo",
            messages=messages,
            max_tokens=150, # Increased from 100
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
        # Genius client handles retries based on its init params
        if artist:
            song = genius.search_song(song_title, artist)
        else:
            song = genius.search_song(song_title)
            
        if not song:
            err_msg = f"Sorry, I couldn't find lyrics for '<b>{song_title}</b>'"
            if artist: err_msg += f" by <i>{artist}</i>"
            err_msg += ". Please check the spelling or try another song!"
            logger.warning(f"No lyrics found for '{song_title}' by '{artist or 'Any'}'")
            return err_msg
        
        lyrics = song.lyrics
        # Pre-process to remove instrumental/intro/outro markers on their own lines, etc.
        lyrics = re.sub(r'^\s*\[(Verse|Chorus|Bridge|Intro|Outro|Pre-Chorus|Hook|Instrumental|Guitar Solo)[^\]]*\]\s*$', '', lyrics, flags=re.MULTILINE | re.IGNORECASE)
        lyrics = re.sub(r'\s*\[.*?\]\s*', '\n', lyrics)  # General metadata tags replaced by newline
        lyrics = re.sub(r'\d*Embed$', '', lyrics.strip()) 
        lyrics = re.sub(r'^\S*Lyrics', '', lyrics.strip(), flags=re.IGNORECASE) # Remove "SongTitleLyrics"
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
        return user_contexts.get(user_id, {}).get("mood", "neutral") # Return current mood or neutral
    logger.debug(f"Detecting mood from text for user {user_id}: '{text[:50]}...'")
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a mood detection AI. Analyze text and return a single dominant mood (happy, sad, anxious, excited, calm, angry, energetic, relaxed, focused, nostalgic, or neutral if unclear). Respond with only the mood word."},
                      {"role": "user", "content": f"Text: '{text}'"}],
            max_tokens=10, 
            temperature=0.2
        )
        mood_raw = response.choices[0].message.content.lower().strip().replace(".", "")
        mood_map = { # Simple normalization
            "positive": "happy", "negative": "sad", "joyful": "happy", "depressed": "sad",
            "chill": "relaxed", "stressed": "anxious", "hyper": "energetic", "peaceful": "calm"
        }
        mood = mood_map.get(mood_raw, mood_raw)
        
        valid_moods = ["happy", "sad", "anxious", "excited", "calm", "angry", "neutral", "energetic", "relaxed", "focused", "nostalgic"]
        if mood in valid_moods:
            logger.info(f"Detected mood for user {user_id}: '{mood}' (from raw: '{mood_raw}')")
            return mood
        else: 
            logger.warning(f"Unexpected mood from AI: '{mood_raw}' for text '{text}'. Defaulting to neutral.")
            return "neutral"

    except Exception as e:
        logger.error(f"Error detecting mood for user {user_id}: {e}")
        return user_contexts.get(user_id, {}).get("mood", "neutral") # Fallback


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
                "Focus on explicit requests like 'play X by Y', 'download Z', 'find song A'. General music chat or mood expression is NOT a specific song request unless they name something specific they want *now*. If they ask 'can you play X?', that is a request."
            },
            {"role": "user", "content": f"Analyze message: '{message}'"}
        ]
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo-0125", # Good model for JSON mode
            messages=prompt_messages,
            max_tokens=80,
            temperature=0.05, # Low temperature for factual extraction
            response_format={"type": "json_object"}
        )

        result_str = response.choices[0].message.content
        result = json.loads(result_str)

        if not isinstance(result, dict): 
            logger.error(f"AI music request (user {user_id}) returned non-dict: {result_str}")
            return {"is_music_request": False, "song_query": None}

        is_request = result.get("is_music_request", False)
        # Robust boolean conversion
        if isinstance(is_request, str): is_request = is_request.lower() in ("true", "yes", "1")
        
        song_query = result.get("song_query")
        if not isinstance(song_query, str) or not song_query.strip(): song_query = None 

        logger.info(f"AI music request (user {user_id}): is_request={is_request}, query='{song_query}' for msg: '{message[:30]}...'")
        return {
            "is_music_request": bool(is_request),
            "song_query": song_query
        }
    except json.JSONDecodeError as jde:
        logger.error(f"AI music request JSON (user {user_id}) decode error: {jde}. Raw: {response.choices[0].message.content if 'response' in locals() and response.choices else 'N/A'}")
        return {"is_music_request": False, "song_query": None}
    except Exception as e:
        logger.error(f"Error in AI is_music_request for user {user_id}: {e}", exc_info=True) # exc_info=True for debugging
        return {"is_music_request": False, "song_query": None}


# ==================== TELEGRAM BOT HANDLERS ====================

async def send_audio_via_bot(bot, chat_id, audio_path, title, performer, caption, duration, reply_to_message_id=None):
    """Helper to send audio using PTB."""
    try:
        with open(audio_path, 'rb') as audio_file_obj:
            logger.info(f"Sending audio '{title}' to chat {chat_id}. Path: {audio_path}")
            await bot.send_audio(
                chat_id=chat_id,
                audio=audio_file_obj,
                title=title[:64], # Telegram's internal limits might be around this
                performer=performer[:64] if performer else "Unknown Artist",
                caption=caption,
                duration=duration,
                reply_to_message_id=reply_to_message_id,
                parse_mode=ParseMode.HTML
            )
    except FileNotFoundError:
        logger.error(f"Audio file not found for sending: {audio_path}")
        await bot.send_message(chat_id, "Sorry, there was an issue preparing the audio file. It might have been removed.")
    except Exception as e:
        logger.error(f"Error sending audio file {audio_path} to chat {chat_id}: {e}", exc_info=True)
        await bot.send_message(chat_id, "Sorry, an error occurred while sending the audio file.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message."""
    user = update.effective_user
    user_contexts.setdefault(user.id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})

    welcome_msg = (
        f"Hi {user.first_name}! 👋 I'm MelodyMind, your Music Healing Companion.\n\n"
        "I can help you:\n"
        "🎵 <b>Download music:</b> Send a YouTube link or ask (e.g., 'play despacito')\n"
        "📜 <b>Find lyrics:</b> Try `/lyrics Bohemian Rhapsody`\n"
        "💿 <b>Get music recommendations:</b> Try `/recommend` or `/mood`\n"
        "💬 <b>Chat</b> about music or how you're feeling!\n"
        "🔗 <b>Link Spotify</b> for personalized vibes with `/link_spotify`\n"
        "📖 <b>Create Spotify playlists</b> (e.g. `/create_playlist My Favs`)\n\n"
        "Type `/help` for all commands, or just start chatting!"
    )
    await update.message.reply_text(welcome_msg, parse_mode=ParseMode.HTML)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message."""
    help_text = (
        "🎶 <b>MelodyMind - Your Music Companion</b> 🎶\n\n"
        "<b>Commands:</b>\n"
        "/start - Welcome message\n"
        "/help - This help guide\n"
        "/download <code>&lt;YouTube URL&gt;</code> - Download from YouTube link\n"
        "/autodownload <code>&lt;song name&gt;</code> - Search & download top result\n"
        "/search <code>&lt;song name&gt;</code> - YouTube search with options\n"
        "/lyrics <code>&lt;song&gt;</code> or <code>&lt;artist - song&gt;</code> - Get lyrics\n"
        "/recommend - Personalized music recommendations\n"
        "/mood - Set your mood for recommendations\n"
        "/link_spotify - Connect Spotify account\n"
        "/create_playlist <code>&lt;name&gt;</code> - New private Spotify playlist\n"
        "/clear - Clear our chat history\n\n"
        "<b>Chat With Me!</b>\n"
        "You can also just talk to me:\n"
        "- \"I'm feeling sad.\"\n"
        "- \"Play 'Shape of You' by Ed Sheeran\"\n"
        "- \"Find lyrics for Hotel California\"\n"
        "- Send a YouTube link directly to download.\n"
        "- Send a voice message!\n\n"
        "Let the music flow! 🎵"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

async def download_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download music from YouTube URL (called by command or direct URL message)."""
    if not update.message: return
    message_text = update.message.text if update.message.text else ""
    url_to_download = ""

    if context.args: 
        url_to_download = " ".join(context.args) # Handle potential spaces if user types `/download url part1 part2`
    else: 
        # Try to find URL in message text (for direct URL messages)
        urls_in_message = [word for word in message_text.split() if is_valid_youtube_url(word)]
        if urls_in_message:
            url_to_download = urls_in_message[0]
        else:
            await update.message.reply_text(
                "❌ Please provide a valid YouTube URL with `/download <URL>` or send the link directly."
            )
            return

    if not is_valid_youtube_url(url_to_download):
        await update.message.reply_text("❌ That doesn't look like a valid YouTube URL. Please check and try again.")
        return

    user_id = update.effective_user.id
    if user_id in active_downloads:
        await update.message.reply_text("⚠️ One download at a time, please! Your current download is still in progress. 😊")
        return

    active_downloads.add(user_id)
    status_msg = await update.message.reply_text("⏳ Starting download... please wait!", parse_mode=ParseMode.HTML)

    try:
        await status_msg.edit_text("🔍 Fetching video info & preparing audio download...", parse_mode=ParseMode.HTML)
        result = await asyncio.to_thread(download_youtube_audio_sync, url_to_download)
        
        if not result["success"]:
            error_message = result.get('error', 'Unknown download error.')
            await status_msg.edit_text(f"❌ Download failed: {error_message}", parse_mode=ParseMode.HTML)
            return

        await status_msg.edit_text(f"✅ Downloaded: <b>{result['title']}</b>\n⏳ Sending you the audio file...", parse_mode=ParseMode.HTML)
        
        await send_audio_via_bot(
            bot=context.bot,
            chat_id=update.effective_chat.id,
            audio_path=result["audio_path"],
            title=result["title"],
            performer=result.get("artist"),
            caption=f"🎵 Here's your track: <b>{result['title']}</b>",
            duration=result.get("duration"),
            reply_to_message_id=update.message.message_id
        )

        if os.path.exists(result["audio_path"]):
            try:
                os.remove(result["audio_path"])
                logger.info(f"Deleted temp file: {result['audio_path']} (user: {user_id})")
            except OSError as e:
                logger.error(f"Error deleting temp file {result['audio_path']} (user: {user_id}): {e}")
        try:
            await status_msg.delete()
        except Exception: pass 

    except TimedOut:
        logger.error(f"Telegram API timeout during download (user {user_id}, url: {url_to_download})")
        if status_msg: await status_msg.edit_text("❌ Telegram API timeout. Please try downloading again.")
    except NetworkError as ne:
        logger.error(f"Telegram API network error during download (user {user_id}, url: {url_to_download}): {ne}")
        if status_msg: await status_msg.edit_text(f"❌ Telegram network error: {str(ne)[:100]}. Please try again.")
    except Exception as e:
        logger.error(f"Unexpected error in download_music (user {user_id}, url: {url_to_download}): {e}", exc_info=True)
        try: 
            if status_msg: await status_msg.edit_text(f"❌ An unexpected error occurred during download: {str(e)[:100]}.")
        except Exception:
            logger.error(f"Failed to send final error message for download to user {user_id}")
    finally:
        active_downloads.discard(user_id)


async def create_playlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Create a private Spotify playlist."""
    if not update.message or not update.effective_user: return
    user_id = update.effective_user.id

    if not context.args:
        await update.message.reply_text("Please specify a name for your playlist. Example: `/create_playlist My Awesome Mix`", parse_mode=ParseMode.HTML)
        return
    
    playlist_name = " ".join(context.args)
    logger.info(f"User {user_id} attempting to create Spotify playlist: '{playlist_name}'")

    access_token = await get_user_spotify_access_token(user_id)
    if not access_token:
        await update.message.reply_text(
            "I need access to your Spotify account to create playlists. 😥 "
            "Please link your account first using the /link_spotify command.",
            parse_mode=ParseMode.HTML
        )
        return

    # Get Spotify User ID
    user_profile_url = "https://api.spotify.com/v1/me"
    headers_auth = {"Authorization": f"Bearer {access_token}"}
    spotify_user_id_from_api = None
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.get(user_profile_url, headers=headers_auth) as response:
                response.raise_for_status()
                spotify_user_id_from_api = (await response.json()).get("id")
        if not spotify_user_id_from_api:
            logger.error(f"Could not retrieve Spotify user ID for Telegram user {user_id}.")
            await update.message.reply_text("Sorry, I couldn't get your Spotify profile ID. This is needed to create playlists.")
            return
    except aiohttp.ClientError as e:
        logger.error(f"API error fetching Spotify profile for user {user_id}: {e}")
        await update.message.reply_text("There was an issue fetching your Spotify profile. Please try again.")
        return
    except Exception as e:
        logger.error(f"Unexpected error fetching Spotify profile for user {user_id}: {e}", exc_info=True)
        await update.message.reply_text("An unexpected error occurred while fetching your Spotify profile.")
        return

    # Create Playlist
    playlist_creation_url = f"https://api.spotify.com/v1/users/{spotify_user_id_from_api}/playlists"
    headers_create = {**headers_auth, "Content-Type": "application/json"}
    payload = {"name": playlist_name, "public": False, "description": f"Created with MelodyMind Bot @ {datetime.now(pytz.UTC).strftime('%Y-%m-%d %H:%M %Z')}"}

    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(playlist_creation_url, headers=headers_create, json=payload) as response:
                response.raise_for_status()
                playlist_data = await response.json()
                playlist_url = playlist_data.get("external_urls", {}).get("spotify", "#") # Get public URL
                logger.info(f"Playlist '{playlist_name}' created successfully for user {user_id}. URL: {playlist_url}")
                await update.message.reply_text(
                    f"🎉 Playlist '<b>{playlist_name}</b>' created successfully!\n"
                    f"You can view it here: {playlist_url}",
                    parse_mode=ParseMode.HTML, disable_web_page_preview=True
                )
    except aiohttp.ClientError as e:
        status = getattr(e, 'status', 'N/A')
        message_detail = getattr(e, 'message', str(e)) # Get a more detailed message if available
        logger.error(f"API error creating playlist '{playlist_name}' for user {user_id}: {status} - {message_detail}")
        await update.message.reply_text(f"Oops! Failed to create playlist (Spotify Error {status}: {message_detail[:100]}). Please try again.")
    except Exception as e:
        logger.error(f"Unexpected error creating playlist '{playlist_name}' for user {user_id}: {e}", exc_info=True)
        await update.message.reply_text("An unexpected error occurred while creating the playlist.")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages by transcribing them."""
    if not update.message or not update.message.voice or not update.effective_user:
        return

    user_id = update.effective_user.id
    logger.info(f"Received voice message from user {user_id}")
    
    try:
        voice_file = await context.bot.get_file(update.message.voice.file_id)
        # Generate a unique filename for the temporary OGG file
        temp_ogg_path = os.path.join(DOWNLOAD_DIR, f"voice_{user_id}_{update.message.message_id}_{datetime.now().timestamp()}.ogg")
        
        await voice_file.download_to_drive(temp_ogg_path)
        logger.debug(f"Voice message from user {user_id} downloaded to {temp_ogg_path}")

        recognizer = sr.Recognizer()
        transcribed_text = None

        def _transcribe_audio_sync_wrapper(): # Wrapper for asyncio.to_thread
            with sr.AudioFile(temp_ogg_path) as source:
                audio_data = recognizer.record(source)  # Read the entire audio file
            try:
                return recognizer.recognize_google(audio_data) # Using Google Web Speech API
            except sr.UnknownValueError:
                logger.warning(f"SpeechRecognition (Google): Could not understand audio from user {user_id}")
                return None # Indicates audio was not understood
            except sr.RequestError as req_e:
                logger.error(f"SpeechRecognition (Google): API request failed for user {user_id}; {req_e}")
                return "ERROR_REQUEST" # Indicates API error

        transcribed_text = await asyncio.to_thread(_transcribe_audio_sync_wrapper)

        if transcribed_text == "ERROR_REQUEST":
            await update.message.reply_text("Sorry, there was an issue with the voice recognition service. Please type your message or try voice again later.")
        elif transcribed_text:
            logger.info(f"Voice message from user {user_id} transcribed: '{transcribed_text}'")
            await update.message.reply_text(f"🎤 I heard: \"<i>{transcribed_text}</i>\"\nLet me process that...", parse_mode=ParseMode.HTML)
            
            # Create a fake Update object with the transcribed text to pass to message handler
            # This is a way to reuse the existing text message handling logic
            context.user_data['_voice_original_message'] = update.message # Store original for reference if needed
            fake_message = update.message._replace(text=transcribed_text, voice=None) # Create new message obj
            fake_update = Update(update.update_id, message=fake_message)
            await enhanced_handle_message(fake_update, context) # Process as if it were a text message
        else: # Transcribed_text is None (couldn't understand)
            await update.message.reply_text("Hmm, I couldn't quite catch that. Could you try speaking more clearly, or perhaps type your message? 😊")

    except Exception as e:
        logger.error(f"Error processing voice message from user {user_id}: {e}", exc_info=True)
        await update.message.reply_text("Oops! Something went wrong while processing your voice message. Please try again.")
    finally:
        # Clean up the temporary OGG file
        if 'temp_ogg_path' in locals() and os.path.exists(temp_ogg_path): # Ensure var defined and file exists
            try:
                os.remove(temp_ogg_path)
                logger.debug(f"Deleted temporary voice file: {temp_ogg_path}")
            except OSError as e_del:
                logger.error(f"Error deleting temporary voice file {temp_ogg_path}: {e_del}")


async def link_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Initiate Spotify OAuth flow."""
    if not update.message or not update.effective_user: return ConversationHandler.END # Should not happen

    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI:
        await update.message.reply_text("Sorry, Spotify linking isn't configured by the bot administrator. 😥")
        return ConversationHandler.END
    
    if SPOTIFY_REDIRECT_URI == "https://your-callback-url.com": # Default placeholder
         await update.message.reply_text(
            "⚠️ <b>Important:</b> The Spotify redirect URI is currently a placeholder. "
            "After authorizing on Spotify, you will need to manually copy the '<code>code</code>' parameter "
            "from your browser's address bar (from the redirected page's URL) and send it back to me here. "
            "The bot admin should configure this properly for a smoother experience.",
            parse_mode=ParseMode.HTML
        )

    user_id = update.effective_user.id
    scopes = "user-read-recently-played user-top-read playlist-modify-private" # Define scopes required
    auth_url = (
        "https://accounts.spotify.com/authorize"
        f"?client_id={SPOTIFY_CLIENT_ID}"
        "&response_type=code"
        f"&redirect_uri={SPOTIFY_REDIRECT_URI}"
        f"&scope={scopes.replace(' ', '%20')}" # URL-encode scopes
        f"&state={user_id}" # Use user_id as state to associate code with user
    )
    keyboard = [
        [InlineKeyboardButton("🔗 Link My Spotify Account", url=auth_url)],
        [InlineKeyboardButton("Cancel Linking", callback_data=CB_CANCEL_SPOTIFY)]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    instructions = (
        "Let's link your Spotify account for a personalized music experience! 🎵\n\n"
        "1. Click the button below to go to Spotify and authorize MelodyMind.\n"
        "2. After authorization, Spotify will redirect you. From that page's URL in your browser, "
        "carefully copy the value of the '<code>code</code>' parameter.\n"
        "   (Example URL: <code>https://your-redirect-uri/?code=A_VERY_LONG_CODE_HERE&state=...</code> - you need '<code>A_VERY_LONG_CODE_HERE</code>')\n"
        "3. Send that <b>entire code</b> back to me here as a message.\n\n"
        "<i>If you encounter issues, please ensure the bot admin has correctly set up the redirect URI in the Spotify Developer Dashboard to match the one I'm expecting.</i>"
    )
    await update.message.reply_text(
        instructions,
        reply_markup=reply_markup,
        parse_mode=ParseMode.HTML 
    )
    return SPOTIFY_CODE

async def spotify_code_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle Spotify authorization code from the user."""
    if not update.message or not update.effective_user : return SPOTIFY_CODE # Stay in state if bad update
    user_id = update.effective_user.id
    message_text = update.message.text.strip() if update.message.text else ""
    
    code_to_use = None
    # Handle direct command `/spotify_code YOUR_CODE` or just pasting the code
    if message_text.startswith('/spotify_code') and context.args: 
        code_to_use = context.args[0]
    elif not message_text.startswith('/'): # Assume raw code pasted
        code_to_use = message_text
    
    # Basic validation of the code (Spotify codes are quite long)
    if not code_to_use or len(code_to_use) < 50: # Arbitrary short length check
        await update.message.reply_text(
            "That Spotify authorization code seems too short or is missing. "
            "Please send the <b>full</b> code you copied from the redirect URL, or use <code>/spotify_code YOUR_CODE</code>.",
            parse_mode=ParseMode.HTML
        )
        return SPOTIFY_CODE # Stay in the same state to await a valid code

    status_msg = await update.message.reply_text("⏳ Validating your Spotify authorization code...")
    token_data = await get_user_spotify_token(user_id, code_to_use)

    if not token_data or not token_data.get("access_token") or not token_data.get("refresh_token"):
        await status_msg.edit_text(
            "❌ Failed to link your Spotify account. The code might be invalid, expired, or there might be a configuration issue. "
            "Please try the /link_spotify command again. Ensure you copy the '<code>code</code>' parameter value correctly and completely.",
            parse_mode=ParseMode.HTML
        )
        return SPOTIFY_CODE # Stay in state, user needs to try again or provide new code

    # Successfully got tokens, store them (encrypted)
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    # Ensure 'spotify' dict exists
    user_contexts[user_id].setdefault("spotify", {})

    user_contexts[user_id]["spotify"]["access_token"] = cipher.encrypt(token_data["access_token"].encode())
    user_contexts[user_id]["spotify"]["refresh_token"] = cipher.encrypt(token_data["refresh_token"].encode()) 
    user_contexts[user_id]["spotify"]["expires_at"] = token_data["expires_at"]
    logger.info(f"Spotify account successfully linked for user {user_id}.")

    # Try to fetch some immediate personalized info as feedback
    rp_info_str = ""
    recently_played = await get_user_spotify_data(user_id, "player/recently-played", params={"limit": 1})
    if recently_played and len(recently_played) > 0:
        try:
            first_rp_artist = recently_played[0]['track']['artists'][0]['name']
            first_rp_track = recently_played[0]['track']['name']
            rp_info_str = f" I see you recently enjoyed '<i>{first_rp_track}</i>' by <b>{first_rp_artist}</b>!"
        except (KeyError, IndexError, TypeError) as e:
            logger.debug(f"Couldn't parse recently played for welcome msg (user {user_id}): {e}")
            pass 

    await status_msg.edit_text(
        f"✅ Spotify successfully linked! 🎉{rp_info_str}\n"
        f"You can now use `/recommend` for personalized music suggestions or `/create_playlist`.",
        parse_mode=ParseMode.HTML
    )
    return ConversationHandler.END

async def spotify_code_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
    """Global handler for /spotify_code if called outside conversation or as entry."""
    if not update.message or not update.effective_user: return None

    if not context.args:
        await update.message.reply_text(
            "Please provide the Spotify authorization code after the command.\n"
            "Example: <code>/spotify_code YOUR_CODE_HERE</code>",
            parse_mode=ParseMode.HTML
        )
        # This global handler should not manage conversation states itself
        # unless it's an entry point TO a conversation.
        # If this is intended to push users INTO the SPOTIFY_CODE state, 
        # it would need to be an entry_point in the ConversationHandler.
        # As a standalone command that shares logic with a state handler,
        # it should just execute its part or guide the user.
        return None # No state change from global context.
    
    # Call the main handler. It returns a state, but globally it means less unless we enter a conv.
    # For now, it will just process or fail, and if in conv, state transition correctly.
    return await spotify_code_handler(update, context)


async def cancel_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel Spotify linking process."""
    query = update.callback_query
    if query:
        await query.answer() 
        await query.edit_message_text("Spotify linking process cancelled. You can try again anytime with /link_spotify. 👍")
    # If somehow called without a query (e.g. via /cancel during spotify linking)
    elif update.message:
        await update.message.reply_text("Spotify linking process cancelled. You can try again anytime with /link_spotify. 👍")
    return ConversationHandler.END


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /search command for YouTube."""
    if not update.message: return
    if not context.args:
        await update.message.reply_text("What song or video would you like to search for on YouTube? Example:\n<code>/search Shape of You Ed Sheeran</code>", parse_mode=ParseMode.HTML)
        return

    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Searching YouTube for: '<i>{query}</i>'...", parse_mode=ParseMode.HTML)
    
    results = await asyncio.to_thread(search_youtube_sync, query, max_results=5)
    
    try:
        await status_msg.delete() 
    except Exception: pass # Message might have already been deleted or interacted with

    await send_search_results(update, query, results)

async def auto_download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /autodownload command: searches YouTube and downloads the first result."""
    if not update.message: return
    if not context.args:
        await update.message.reply_text("What song would you like to auto-download? Example:\n<code>/autodownload Imagine Dragons Believer</code>", parse_mode=ParseMode.HTML)
        return

    query = " ".join(context.args)
    # Pass update.message.message_id so that auto_download_first_result can reply to it.
    await auto_download_first_result(update, context, query, original_message_to_reply_to=update.message.message_id)


async def get_lyrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle lyrics requests via /lyrics command."""
    if not update.message: return
    if not context.args:
        await update.message.reply_text(
            "Which song's lyrics are you looking for? Examples:\n"
            "<code>/lyrics Bohemian Rhapsody</code>\n"
            "<code>/lyrics Queen - Bohemian Rhapsody</code>\n"
            "<code>/lyrics lyrics for Hotel California by The Eagles</code> (informal)",
            parse_mode=ParseMode.HTML
        )
        return

    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Searching for lyrics for: \"<i>{query}</i>\"...", parse_mode=ParseMode.HTML)

    try:
        artist = None
        song_title = query
        # Attempt to parse "Artist - Song" or "Song by Artist"
        if " - " in query:
            parts = query.split(" - ", 1)
            artist, song_title = parts[0].strip(), parts[1].strip()
        elif " by " in query.lower(): # More flexible "by" parsing
            match = re.search(r'^(.*?)\s+by\s+(.*?)$', query, re.IGNORECASE)
            if match:
                song_title, artist = match.group(1).strip(), match.group(2).strip()
        
        logger.info(f"Lyrics command processing: song='{song_title}', artist='{artist}' (original query: '{query}')")
        
        lyrics_text = await asyncio.to_thread(get_lyrics_sync, song_title, artist)
        
        max_len = 4090 # Telegram message length limit (slightly less for safety)
        if len(lyrics_text) > max_len:
            current_message = status_msg # Start with the status message to edit
            sent_chars = 0
            
            parts = []
            while len(lyrics_text) > 0:
                chunk = lyrics_text[:max_len]
                lyrics_text = lyrics_text[max_len:]

                if len(lyrics_text) > 0: # If there's more to come
                    # Try to break at a double newline for cleaner separation
                    cut_point = chunk.rfind('\n\n')
                    if cut_point == -1 or cut_point < max_len - 500: # If no good \n\n or too early, try single \n
                        cut_point = chunk.rfind('\n')
                    if cut_point == -1 or cut_point < max_len - 300 : # If still no good \n, just cut at max_len
                        cut_point = max_len
                    
                    parts.append(chunk[:cut_point])
                    lyrics_text = chunk[cut_point:] + lyrics_text # Prepend remainder to next chunk
                else:
                    parts.append(chunk)

            for i, part_text in enumerate(parts):
                is_last_part = (i == len(parts) - 1)
                text_to_send = part_text
                if not is_last_part:
                    text_to_send += "\n\n<small><i>(lyrics continued below...)</i></small>"
                
                if i == 0: # First part, edit status message
                    await current_message.edit_text(text_to_send, parse_mode=ParseMode.HTML)
                else: # Subsequent parts, send new message
                    await update.message.reply_text(text_to_send, parse_mode=ParseMode.HTML)
        else: # Lyrics fit in one message
            await status_msg.edit_text(lyrics_text, parse_mode=ParseMode.HTML)

    except Exception as e: 
        logger.error(f"Error in get_lyrics_command (query '{query}'): {e}", exc_info=True)
        if status_msg: await status_msg.edit_text("Sorry, an unexpected hiccup occurred while fetching lyrics. 😕 Please try again.")


async def recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provide music recommendations. (Alias for smart_recommend_music)."""
    await smart_recommend_music(update, context)


async def provide_generic_recommendations(update: Update, mood: str, chat_id_override: Optional[int] = None) -> None:
    """Provide generic, hardcoded recommendations as a fallback."""
    logger.info(f"Providing generic recommendations for mood: {mood} to chat ID {chat_id_override or update.effective_chat.id}")
    target_chat_id = chat_id_override or update.effective_chat.id # Ensure chat_id is defined

    mood_recommendations = {
        "happy": ["Uptown Funk - Mark Ronson", "Happy - Pharrell Williams", "Walking on Sunshine - Katrina & The Waves"],
        "sad": ["Someone Like You - Adele", "Hallelujah - Leonard Cohen (Jeff Buckley version)", "Fix You - Coldplay"],
        "energetic": ["Don't Stop Me Now - Queen", "Thunderstruck - AC/DC", "Can't Stop the Feeling! - Justin Timberlake"],
        "relaxed": ["Weightless - Marconi Union", "Clair de Lune - Debussy", "Orinoco Flow - Enya"],
        "focused": ["The Four Seasons - Vivaldi (Max Richter recomposed)", "Time - Hans Zimmer", "Ambient 1: Music for Airports - Brian Eno"],
        "nostalgic": ["Bohemian Rhapsody - Queen", "Yesterday - The Beatles", "Wonderwall - Oasis"],
        "default": ["Three Little Birds - Bob Marley", "Here Comes The Sun - The Beatles", "What a Wonderful World - Louis Armstrong"]
    }

    chosen_mood_key = mood.lower()
    if chosen_mood_key not in mood_recommendations:
        logger.warning(f"Generic mood '{mood}' not in list, defaulting to 'default' recommendations.")
        chosen_mood_key = "default" 
        
    recommendations = mood_recommendations.get(chosen_mood_key, mood_recommendations["default"]) 
    response_text = f"🎵 Here are some general song suggestions for a <b>{mood.capitalize()}</b> vibe:\n\n"
    for i, track in enumerate(recommendations, 1):
        response_text += f"{i}. {track}\n"
    response_text += "\n💡 <i>You can ask me to search or download any of these! (e.g., /autodownload {track example})</i>"
    
    await context.bot.send_message(chat_id=target_chat_id, text=response_text, parse_mode=ParseMode.HTML)


async def set_mood(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start conversation to set mood, part of a ConversationHandler."""
    if not update.effective_user : return ConversationHandler.END # Should not happen

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
        prompt_text += f"Your current mood is set to <b>{current_mood}</b>. How are you feeling now, or would you like to change it?"
    else:
        prompt_text += "How are you feeling today?"
    
    if update.callback_query: 
        await update.callback_query.answer()
        await update.callback_query.edit_message_text(prompt_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    elif update.message: 
        await update.message.reply_text(prompt_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)
    return MOOD 

async def send_search_results(update: Update, query: str, results: List[Dict]) -> None:
    """Send YouTube search results with inline keyboard for download."""
    # Ensure there's a message to reply to or context to send a new message
    message_to_reply_to = update.message or (update.callback_query and update.callback_query.message)
    if not message_to_reply_to or not update.effective_chat:
        logger.error("send_search_results: Cannot determine chat to send results to.")
        return
        
    target_chat_id = update.effective_chat.id
    
    if not results:
        await context.bot.send_message(chat_id=target_chat_id, text=f"😕 Sorry, no YouTube results for '<i>{query}</i>'. Try different keywords or check your spelling?", parse_mode=ParseMode.HTML)
        return

    keyboard_rows = []
    response_text_header = f"🔎 YouTube search results for '<i>{query}</i>':\n\n"
    
    valid_results_count = 0
    for i, result in enumerate(results[:5]): # Show top 5
        # Validate essential fields
        video_id = result.get('id')
        title = result.get('title')
        if not video_id or not title or not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
            logger.warning(f"Skipping invalid YouTube search result item (ID: {video_id}, Title: {title}) for query '{query}'")
            continue
        valid_results_count +=1

        duration_str = ""
        duration_seconds = result.get('duration')
        if duration_seconds and isinstance(duration_seconds, (int, float)) and duration_seconds > 0:
            try:
                minutes = int(duration_seconds // 60)
                seconds = int(duration_seconds % 60)
                duration_str = f" [{minutes}:{seconds:02d}]"
            except TypeError: duration_str = "" # Should not happen with type check
        
        # Ensure button text is within Telegram's limits (approx 64 bytes, but play safe)
        button_display_title = (title[:35] + "...") if len(title) > 38 else title
        button_text = f"[{valid_results_count}] {button_display_title}{duration_str}"
        
        response_text_header += f"{valid_results_count}. <b>{title}</b> by <i>{result.get('uploader', 'N/A')}</i>{duration_str}\n"
        keyboard_rows.append([InlineKeyboardButton(button_text, callback_data=f"{CB_DOWNLOAD_PREFIX}{video_id}")])

    if not keyboard_rows: # Should only happen if all results were invalid
        await context.bot.send_message(chat_id=target_chat_id, text=f"😕 Found some YouTube results for '<i>{query}</i>', but had trouble processing them. Please try again.", parse_mode=ParseMode.HTML)
        return

    keyboard_rows.append([InlineKeyboardButton("Cancel Search", callback_data=CB_CANCEL_SEARCH)])
    reply_markup = InlineKeyboardMarkup(keyboard_rows)
    
    final_text = response_text_header + "\nClick a song from the list above to download its audio:"
    await context.bot.send_message(chat_id=target_chat_id, text=final_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)


async def auto_download_first_result(
    update: Update, 
    context: ContextTypes.DEFAULT_TYPE, 
    query: str, 
    original_message_id_to_edit: Optional[int] = None,
    original_message_to_reply_to: Optional[int] = None # For replying to user's initial command
) -> None:
    """Search YouTube, then automatically download the first valid song result."""
    if not update.effective_user or not update.effective_chat: return

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id

    if user_id in active_downloads:
        message_text = "Hold on! You already have a download in progress. Let that finish first. 😊"
        if original_message_id_to_edit: # Coming from a button press
            await context.bot.edit_message_text(chat_id=chat_id, message_id=original_message_id_to_edit, text=message_text, reply_markup=None, parse_mode=ParseMode.HTML)
        elif original_message_to_reply_to: # Coming from a command
             await context.bot.send_message(chat_id=chat_id, text=message_text, reply_to_message_id=original_message_to_reply_to, parse_mode=ParseMode.HTML)
        else: # Fallback
             await context.bot.send_message(chat_id=chat_id, text=message_text, parse_mode=ParseMode.HTML)
        return

    active_downloads.add(user_id)
    status_msg = None 

    try:
        # Determine initial status message action (edit or send new)
        search_status_text = f"🔍 Okay, searching for '<i>{query}</i>' to download..."
        if original_message_id_to_edit:
            status_msg = await context.bot.edit_message_text(chat_id=chat_id, message_id=original_message_id_to_edit, 
                                             text=search_status_text, parse_mode=ParseMode.HTML, reply_markup=None)
        elif original_message_to_reply_to: # If called from /autodownload
            status_msg = await context.bot.send_message(chat_id=chat_id, text=search_status_text, 
                                             reply_to_message_id=original_message_to_reply_to, parse_mode=ParseMode.HTML)
        else: # Fallback if no specific message to edit/reply to
            status_msg = await context.bot.send_message(chat_id=chat_id, text=search_status_text, parse_mode=ParseMode.HTML)

        # Search for the track. If query is already a URL, search_youtube_sync might directly use it or search with it.
        # If query is an ID, it needs to be formatted as URL for download_youtube_audio_sync below.
        video_url_to_download = ""
        video_title_for_status = query # Default to query for status messages

        if is_valid_youtube_url(query): # Query is already a URL (e.g., from button click with direct URL)
            video_url_to_download = query
            # Attempt to get title for better status message
            try:
                temp_info = await asyncio.to_thread(yt_dlp.YoutubeDL({'quiet':True, 'no_warnings':True}).extract_info, query, download=False)
                if temp_info and temp_info.get('title'):
                    video_title_for_status = temp_info.get('title', query)
            except Exception:
                logger.debug(f"Could not pre-fetch title for URL {query} in auto-DL, using URL as title.")
        else: # Query is a search term
            results = await asyncio.to_thread(search_youtube_sync, query, max_results=1) 
            if not results or not results[0].get('url') or not is_valid_youtube_url(results[0]['url']):
                await status_msg.edit_text(f"❌ Oops! Couldn't find a downloadable track for '<i>{query}</i>'. Maybe try <code>/search {query}</code> for more options?", parse_mode=ParseMode.HTML)
                return
            top_result = results[0]
            video_url_to_download = top_result["url"]
            video_title_for_status = top_result.get("title", query)
        
        await status_msg.edit_text(f"✅ Found: <b>{video_title_for_status}</b>.\n⏳ Preparing download... (this can take a moment!)", parse_mode=ParseMode.HTML)

        download_result = await asyncio.to_thread(download_youtube_audio_sync, video_url_to_download)
        
        if not download_result["success"]:
            error_message = download_result.get('error', 'Unknown download error.')
            await status_msg.edit_text(f"❌ Download failed for <b>{video_title_for_status}</b>: {error_message}", parse_mode=ParseMode.HTML)
            return

        await status_msg.edit_text(f"✅ Downloaded: <b>{download_result['title']}</b>.\n✅ Sending the audio file now...", parse_mode=ParseMode.HTML)
        
        await send_audio_via_bot(
            bot=context.bot,
            chat_id=chat_id,
            audio_path=download_result["audio_path"],
            title=download_result["title"],
            performer=download_result.get("artist"),
            caption=f"🎵 Here's your track: <b>{download_result['title']}</b>",
            duration=download_result.get("duration"),
            reply_to_message_id= original_message_to_reply_to or (update.callback_query and update.callback_query.message.message_id if update.callback_query else None) or status_msg.message_id
        )

        if os.path.exists(download_result["audio_path"]):
            try:
                os.remove(download_result["audio_path"])
                logger.info(f"Temp file deleted after auto-DL: {download_result['audio_path']} (user {user_id})")
            except OSError as e:
                logger.error(f"Error deleting temp file (auto-DL) {download_result['audio_path']} (user {user_id}): {e}")
        
        try:
            await status_msg.delete() # Clean up the final status message
        except Exception: pass # It might have already been deleted or other issues
    
    except (TimedOut, NetworkError) as net_err:
        logger.error(f"Telegram API/Network error during auto-download (user {user_id}, query '{query}'): {net_err}")
        if status_msg: 
           try: await status_msg.edit_text(f"❌ A network problem occurred while processing '<i>{query}</i>'. Please try again.", parse_mode=ParseMode.HTML)
           except: pass # Ignore error editing if message is gone
    except Exception as e:
        logger.error(f"Unexpected error during auto-download (user {user_id}, query '{query}'): {e}", exc_info=True)
        if status_msg:
            try: await status_msg.edit_text(f"❌ An unexpected error occurred while processing '<i>{query}</i>'. My apologies! Please try again.", parse_mode=ParseMode.HTML)
            except: pass # Ignore error editing if message is gone
    finally:
        active_downloads.discard(user_id)


async def enhanced_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
    """Handle button callbacks from inline keyboards."""
    query = update.callback_query
    if not query or not query.from_user: return None # Should not happen
    await query.answer() 
    
    data = query.data
    user_id = query.from_user.id
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    
    logger.debug(f"Button callback: '{data}' for user {user_id} (message_id: {query.message.message_id if query.message else 'N/A'})")

    if data.startswith(CB_MOOD_PREFIX):
        mood = data[len(CB_MOOD_PREFIX):]
        user_contexts[user_id]["mood"] = mood
        logger.info(f"User {user_id} set mood via button to: {mood}")

        keyboard = [ 
            [InlineKeyboardButton("Pop", callback_data=f"{CB_PREFERENCE_PREFIX}pop"),
             InlineKeyboardButton("Rock", callback_data=f"{CB_PREFERENCE_PREFIX}rock"),
             InlineKeyboardButton("Hip-Hop", callback_data=f"{CB_PREFERENCE_PREFIX}hiphop")],
            [InlineKeyboardButton("Electronic", callback_data=f"{CB_PREFERENCE_PREFIX}electronic"),
             InlineKeyboardButton("Classical", callback_data=f"{CB_PREFERENCE_PREFIX}classical"),
             InlineKeyboardButton("Jazz", callback_data=f"{CB_PREFERENCE_PREFIX}jazz")],
            [InlineKeyboardButton("Folk/Acoustic", callback_data=f"{CB_PREFERENCE_PREFIX}folk"), # Changed slightly
             InlineKeyboardButton("R&B/Soul", callback_data=f"{CB_PREFERENCE_PREFIX}rnb"), # Changed slightly
             InlineKeyboardButton("Any/Surprise Me!", callback_data=f"{CB_PREFERENCE_PREFIX}any")],
            [InlineKeyboardButton("➡️ Skip Genre / Use Mood Only", callback_data=f"{CB_PREFERENCE_PREFIX}skip")],
        ]
        await query.edit_message_text(
            f"Got it, {query.from_user.first_name}! You're feeling <b>{mood}</b>. 🎶\nAny particular genre you'd like for music recommendations now, or should I just go by mood?",
            reply_markup=InlineKeyboardMarkup(keyboard),
            parse_mode=ParseMode.HTML
        )
        return PREFERENCE 

    elif data.startswith(CB_PREFERENCE_PREFIX):
        preference = data[len(CB_PREFERENCE_PREFIX):]
        msg_text = ""
        if preference == "skip" or preference == "any":
            user_contexts[user_id]["preferences"] = [] # Clear specific genre preferences
            msg_text = "Alright! I'll keep your mood in mind for recommendations."
        else:
            user_contexts[user_id]["preferences"] = [preference] # Set this specific genre
            msg_text = f"Great choice! <b>{preference.capitalize()}</b> it is. "
        logger.info(f"User {user_id} set preference via button to: {preference}")
        
        msg_text += "\n\nYou can now:\n➡️ Try `/recommend` for tailored suggestions.\n➡️ Use `/search [song name]` to find specific tracks.\n➡️ Or just continue chatting with me!"
        await query.edit_message_text(msg_text, parse_mode=ParseMode.HTML)
        return ConversationHandler.END 

    elif data.startswith(CB_DOWNLOAD_PREFIX): # Download chosen from a list of search results
        video_id = data[len(CB_DOWNLOAD_PREFIX):]
        if not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
             logger.error(f"Invalid YouTube ID '{video_id}' from CB_DOWNLOAD_PREFIX button (user {user_id}).")
             await query.edit_message_text("❌ Error: Invalid video ID provided. Please try searching again.", reply_markup=None)
             return None
        
        youtube_direct_url = f"https://www.youtube.com/watch?v={video_id}"
        # original_message_id_to_edit is the message with the search result buttons
        await auto_download_first_result(update, context, query=youtube_direct_url, original_message_id_to_edit=query.message.message_id if query.message else None)
        return None 

    elif data.startswith(CB_AUTO_DOWNLOAD_PREFIX): # From "Yes, download [this specific track]" type buttons
        video_id_or_query = data[len(CB_AUTO_DOWNLOAD_PREFIX):] 
        
        target_url_for_auto_dl = ""
        if re.match(r'^[0-9A-Za-z_-]{11}$', video_id_or_query): # It's a video ID
            target_url_for_auto_dl = f"https://www.youtube.com/watch?v={video_id_or_query}"
        else: # It's a query string (less likely from this specific prefix now, but good to handle)
            target_url_for_auto_dl = video_id_or_query # auto_download_first_result handles search then download

        # original_message_id_to_edit is the message with "Download X or show more options?"
        await auto_download_first_result(update, context, query=target_url_for_auto_dl, original_message_id_to_edit=query.message.message_id if query.message else None)
        return None

    elif data.startswith(CB_SHOW_OPTIONS_PREFIX): # User clicked "Show more options"
        search_query_str = data[len(CB_SHOW_OPTIONS_PREFIX):]
        if not search_query_str:
            logger.warning(f"CB_SHOW_OPTIONS_PREFIX (user {user_id}): Original query missing from callback data.")
            await query.edit_message_text("Cannot show options, the original search query was lost. Please try searching again with `/search`.", reply_markup=None)
            return None
        
        # Edit the current message ("I found X, download or show options?") to "Searching..."
        await query.edit_message_text(f"🔍 Okay, fetching more YouTube options for '<i>{search_query_str}</i>'...", parse_mode=ParseMode.HTML, reply_markup=None)
        
        results = await asyncio.to_thread(search_youtube_sync, search_query_str, max_results=5)
        
        # We want send_search_results to send a *new* message with the list, not edit.
        # The original query.message (the "Download X or more options?" one) effectively gets replaced by the search results list.
        # send_search_results needs an Update-like object; the `update` object from the callback query handler is fine.
        # It will use update.effective_chat.id to send the new message.
        await send_search_results(update, search_query_str, results)
        return None

    elif data == CB_CANCEL_SEARCH:
        await query.edit_message_text("❌ Search cancelled. Feel free to try another search or command!", reply_markup=None)
        return None
    elif data == CB_CANCEL_SPOTIFY: 
        # This callback should ideally only be handled by the spotify_conv_handler's fallback or state handler.
        # If it reaches here, it means it wasn't caught by the ConversationHandler.
        # This could happen if the conversation timed out but the button was still active.
        logger.info(f"CB_CANCEL_SPOTIFY (user {user_id}) handled by general button handler (likely outside active conversation).")
        await query.edit_message_text("Spotify linking cancelled. You can use /link_spotify to try again anytime.", reply_markup=None)
        # Return END to ensure any stray conversation context (if any) is terminated.
        return ConversationHandler.END 

    logger.warning(f"Unhandled callback data: '{data}' from user {user_id} (message_id: {query.message.message_id if query.message else 'N/A'})")
    try:
        await query.edit_message_text("Sorry, there was an issue with that button or the action has expired.", reply_markup=None)
    except Exception as e_edit:
        logger.error(f"Failed to edit message for unhandled callback: {e_edit}")
    return None # No state transition for unhandled or general cancels here


async def enhanced_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enhanced message handler: Handles YouTube URLs, AI-based music/lyrics requests, and general chat."""
    if not update.message or not update.message.text or not update.effective_user: 
        return # Ignore empty messages or messages without user context

    user_id = update.effective_user.id
    text = update.message.text.strip()
    logger.debug(f"Received message from user {user_id}: '{text[:100]}...'")

    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})

    # 1. Direct YouTube URL for Download
    if is_valid_youtube_url(text):
        logger.info(f"User {user_id} sent a YouTube URL directly: {text}")
        # Pass the update directly to download_music, it will extract the URL
        # download_music handles creating its own context.args if needed from update.message.text
        await download_music(update, context)
        return

    # 2. AI-based Mood Detection (passive, for context)
    # Only run on messages with a few words to avoid over-processing short interactions.
    if len(text.split()) > 3: 
        current_mood = user_contexts[user_id].get("mood", "neutral")
        # Schedule mood detection, don't await it here to keep message flow fast
        async def _update_mood():
            detected_mood = await detect_mood_from_text(user_id, text)
            if detected_mood and detected_mood != "neutral" and detected_mood != current_mood :
                user_contexts[user_id]["mood"] = detected_mood
                logger.info(f"Passive mood update for user {user_id} to: {detected_mood} (was: {current_mood}) based on: '{text[:30]}'")
        asyncio.create_task(_update_mood())


    # 3. AI Music Request Detection (e.g., "play song X", "download Y by Z")
    ai_music_eval = await is_music_request(user_id, text)
    if ai_music_eval.get("is_music_request") and ai_music_eval.get("song_query"):
        music_query = ai_music_eval["song_query"]
        # User explicitly asked for music by name. Offer to download first result or show options.
        status_msg = await update.message.reply_text(f"🎵 You're looking for '<i>{music_query}</i>'? Let me check YouTube for that...", parse_mode=ParseMode.HTML)
        
        # Search YouTube for the top result
        results = await asyncio.to_thread(search_youtube_sync, music_query, max_results=1)
        
        if results and results[0].get('id') and re.match(r'^[0-9A-Za-z_-]{11}$', results[0]['id']):
            top_res = results[0]
            title = top_res.get('title', 'this track')
            uploader = top_res.get('uploader', 'Unknown Artist')
            
            # Ensure callback_data length limits are respected (Telegram: 1-64 bytes)
            # video ID (11 chars) + prefix (18 chars for CB_AUTO_DOWNLOAD_PREFIX) is fine.
            # For query, it could be long. Truncate if necessary, or better, rely on ID.
            # Since CB_SHOW_OPTIONS_PREFIX takes a query, we pass the original validated music_query.
            # Ensure music_query is not excessively long for callback_data.
            cb_music_query = music_query[:30] # Truncate query for callback data safety

            keyboard = [
                [InlineKeyboardButton(f"✅ Yes, download '{title[:20]}...'", callback_data=f"{CB_AUTO_DOWNLOAD_PREFIX}{top_res['id']}")],
                [InlineKeyboardButton("👀 Show more search options", callback_data=f"{CB_SHOW_OPTIONS_PREFIX}{cb_music_query}")], 
                [InlineKeyboardButton("❌ No, cancel this", callback_data=CB_CANCEL_SEARCH)]
            ]
            await status_msg.edit_text(
                f"I found this on YouTube:\n<b>{title}</b> by <i>{uploader}</i>.\n\nWould you like to download the audio, or see more search options?",
                reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML
            )
        else:
            await status_msg.edit_text(f"😕 I couldn't find a specific downloadable track for '<i>{music_query}</i>' with a quick search. "
                                       f"You could try being more specific or use <code>/search {music_query}</code> to see a list of results.", parse_mode=ParseMode.HTML)
        return

    # 4. Heuristic Lyrics Request Detection (e.g., "lyrics for X", "what are the lyrics to Y")
    # This is a simpler, non-AI check for common lyrics phrases.
    lyrics_keywords_precise = ["lyrics for", "lyrics to", "get lyrics for", "find lyrics for", "what are the lyrics to", "song lyrics for"]
    lyrics_query_candidate = None
    text_lower = text.lower()
    for keyword in lyrics_keywords_precise:
        if text_lower.startswith(keyword):
            # Extract text after the keyword
            potential_query = text[len(keyword):].strip()
            if potential_query: # Ensure there's something after the keyword
                lyrics_query_candidate = potential_query
                logger.info(f"Heuristic lyrics request detected for user {user_id}: '{lyrics_query_candidate}' (full text: '{text_lower}')")
                break
    
    if lyrics_query_candidate:
        # Pass the extracted query to the /lyrics command handler logic
        # Create a minimal context with args for get_lyrics_command
        # This assumes get_lyrics_command can parse "song title by artist" from its args.
        # We use .split() which might not be ideal if artist/song has many words.
        # get_lyrics_command has its own more robust parsing of "by" and "-".
        # For this direct invocation, pass the full string as a single arg for now or refined split.
        mock_args = [lyrics_query_candidate] # Treat the whole thing as query. get_lyrics_command will parse.
        temp_context_for_lyrics = ContextTypes.DEFAULT_TYPE(application=context.application, chat_id=user_id, user_id=user_id, bot=context.bot, args=mock_args)
        await get_lyrics_command(update, temp_context_for_lyrics) # Use the actual update for message reply context
        return

    # 5. Fallback to General AI Chat
    # Show a "thinking" message for a better UX, as AI can take a moment
    # Random small delay can make it feel more natural than instant "thinking"
    await asyncio.sleep(0.1 + (os.urandom(1)[0] / 255.0) * 0.3) # Delay 0.1 to 0.4s
    typing_msg = await update.message.reply_text("<i>MelodyMind is thinking...</i> 🎶", parse_mode=ParseMode.HTML)
    try:
        response_text = await generate_chat_response(user_id, text)
        await typing_msg.edit_text(response_text, parse_mode=ParseMode.HTML) # HTML if AI response might contain it
    except (TimedOut, NetworkError) as net_err:
        logger.error(f"Network error during AI chat for user {user_id}: {net_err}")
        await typing_msg.edit_text("Sorry, I'm having a bit of a network hiccup. Could you try saying that again?", parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Error in AI chat response generation for user {user_id}: {e}", exc_info=True)
        await typing_msg.edit_text("A little tangled up right now! 😅 My circuits are a bit fuzzy. Let's try that again later or try a command like /help.", parse_mode=ParseMode.HTML)


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear user's conversation history with the bot."""
    if not update.message or not update.effective_user: return
    user_id = update.effective_user.id
    
    cleared = False
    if user_id in user_contexts:
        if "conversation_history" in user_contexts[user_id] and user_contexts[user_id]["conversation_history"]:
            user_contexts[user_id]["conversation_history"] = []
            cleared = True
        # Optionally, clear mood and preferences too, or keep them. For now, just history.
        # user_contexts[user_id]["mood"] = None
        # user_contexts[user_id]["preferences"] = []

    if cleared:
        logger.info(f"Cleared conversation history for user {user_id}")
        await update.message.reply_text("✅ Our chat history has been cleared. Your mood and preferences (if set) are still remembered.")
    else:
        await update.message.reply_text("There's no chat history to clear yet! Feel free to start a conversation. 😊")

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Generic cancel handler for ConversationHandlers."""
    reply_text = "Okay, the current operation has been cancelled. What would you like to do next? You can chat or use commands like /help anytime! 👍"
    if update.message:
        await update.message.reply_text(reply_text)
    elif update.callback_query:
        await update.callback_query.answer("Operation cancelled.")
        try: # Try to edit the message the button was on
            await update.callback_query.edit_message_text(reply_text, reply_markup=None)
        except Exception: # If edit fails (e.g. message too old, or no text to edit)
            if update.effective_chat: # Send a new message in the chat
                 await context.bot.send_message(chat_id=update.effective_chat.id, text=reply_text)
    
    # Clear any temporary user_data specific to this conversation if needed
    # context.user_data.pop('some_temp_conv_state', None)
    return ConversationHandler.END

async def analyze_conversation(user_id: int) -> Dict:
    """Analyze conversation history and Spotify data using AI for preferences."""
    default_return = {"genres": user_contexts.get(user_id, {}).get("preferences", []), 
                      "artists": [], 
                      "mood": user_contexts.get(user_id, {}).get("mood")}
    if not client: 
        logger.warning(f"OpenAI client not available for analyze_conversation (user {user_id}).")
        return default_return

    # Ensure necessary keys exist in user_contexts
    user_ctx = user_contexts.setdefault(user_id, {})
    user_ctx.setdefault("preferences", [])
    user_ctx.setdefault("conversation_history", [])
    user_ctx.setdefault("spotify", {}).setdefault("recently_played", [])
    user_ctx["spotify"].setdefault("top_tracks", [])

    # Require some data to be present for meaningful analysis
    if len(user_ctx["conversation_history"]) < 2 and \
       not user_ctx["spotify"]["recently_played"] and \
       not user_ctx["spotify"]["top_tracks"] and \
       not user_ctx.get("mood") and \
       not user_ctx.get("preferences"):
        logger.info(f"Insufficient data for AI analysis (user {user_id}). Using existing context.")
        return default_return

    logger.info(f"Performing AI conversation/data analysis for user {user_id}")
    try:
        # Summarize conversation history (last few exchanges)
        history_summary_parts = [f"{msg['role']}: {msg['content'][:100]}" for msg in user_ctx["conversation_history"][-6:]] # Limit chars per message
        conversation_text_summary = "\n".join(history_summary_parts) if history_summary_parts else "No conversation history."

        # Summarize Spotify data
        spotify_summary_parts = []
        if user_ctx["spotify"]["recently_played"]:
            try:
                tracks = user_ctx["spotify"]["recently_played"]
                rp_summary = ", ".join(
                    [f"'{item['track']['name'][:30]}' by {item['track']['artists'][0]['name'][:20]}" for item in tracks[:3] if item.get("track")] 
                )
                if rp_summary: spotify_summary_parts.append("Recently played: " + rp_summary)
            except Exception as e_rp: logger.debug(f"Error summarizing recently_played for AI (user {user_id}): {e_rp}")
        if user_ctx["spotify"]["top_tracks"]:
            try:
                tracks = user_ctx["spotify"]["top_tracks"]
                tt_summary = ", ".join(
                    [f"'{item['name'][:30]}' by {item['artists'][0]['name'][:20]}" for item in tracks[:3] if item.get("artists")]
                )
                if tt_summary: spotify_summary_parts.append("Top tracks: " + tt_summary)
            except Exception as e_tt: logger.debug(f"Error summarizing top_tracks for AI (user {user_id}): {e_tt}")
        spotify_summary = ". ".join(spotify_summary_parts) if spotify_summary_parts else "No Spotify data."
        
        prompt_user_content = (
            f"Conversation Snippets:\n{conversation_text_summary}\n\n"
            f"Spotify Listening Summary:\n{spotify_summary}\n\n"
            f"User's explicitly set current mood: {user_ctx.get('mood', 'Not set')}\n"
            f"User's explicitly set genre preferences: {', '.join(user_ctx.get('preferences',[])) if user_ctx.get('preferences') else 'Not set'}"
        )
        prompt_messages = [
            {"role": "system", "content": 
                "Analyze the user's recent chat messages, Spotify listening habits, and explicitly set mood/preferences. "
                "Infer up to 2 relevant music genres and up to 2 relevant artists based on this data. "
                "Also, confirm or infer the user's most likely current overall mood. "
                "Prioritize explicit statements or strong signals. If data is sparse or contradictory, return empty lists or null for mood. "
                "Provide output in JSON format with keys: 'genres' (list of strings, max 2), 'artists' (list of strings, max 2), 'mood' (string, one of: happy, sad, anxious, excited, calm, angry, energetic, relaxed, focused, nostalgic, or null if unclear/neutral)."
            },
            {"role": "user", "content": prompt_user_content }
        ]

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo-0125", # JSON mode capable
            messages=prompt_messages,
            max_tokens=150, 
            temperature=0.1, # Low temp for more deterministic analysis
            response_format={"type": "json_object"}
        )

        result_str = response.choices[0].message.content
        result = json.loads(result_str)

        if not isinstance(result, dict):
            logger.error(f"AI analysis (user {user_id}) returned non-dict: {result_str}")
            return default_return

        inferred_genres = result.get("genres", [])
        if isinstance(inferred_genres, str): inferred_genres = [g.strip().lower() for g in inferred_genres.split(",") if g.strip()]
        if not isinstance(inferred_genres, list): inferred_genres = []
        
        inferred_artists = result.get("artists", [])
        if isinstance(inferred_artists, str): inferred_artists = [a.strip() for a in inferred_artists.split(",") if a.strip()]
        if not isinstance(inferred_artists, list): inferred_artists = []

        inferred_mood_raw = result.get("mood")
        inferred_mood = None
        if isinstance(inferred_mood_raw, str) and inferred_mood_raw.strip().lower() not in ["null", "none", ""]:
            valid_moods_ai = ["happy", "sad", "anxious", "excited", "calm", "angry", "neutral", "energetic", "relaxed", "focused", "nostalgic"]
            if inferred_mood_raw.lower() in valid_moods_ai:
                inferred_mood = inferred_mood_raw.lower()
        
        # Update user_contexts based on AI analysis, potentially overriding or augmenting existing.
        # Give preference to user's explicit settings if AI is very different or unsure.
        # For this iteration, let's merge: use AI if user hasn't set, or if AI has high confidence (not modeled here).
        # Simple merge: If AI found genres and user has none, use AI's. If AI found mood and user has none, use AI's.
        final_genres = list(set(user_ctx.get("preferences", []) + inferred_genres[:2])) # Combine and keep unique, max 2 from AI
        final_mood = user_ctx.get("mood") or inferred_mood or "neutral" # Prioritize user's explicit mood
        
        # Don't directly modify user_contexts['preferences'] or ['mood'] here; this function is for analysis.
        # The calling function (smart_recommend_music) will use this analysis.

        analyzed_data = {
            "genres": final_genres[:2], # Max 2 genres overall
            "artists": inferred_artists[:2], # Max 2 artists
            "mood": final_mood
        }
        logger.info(f"AI analysis (user {user_id}) completed. Result: {analyzed_data}")
        return analyzed_data

    except json.JSONDecodeError as jde:
        logger.error(f"AI analysis JSON decode error (user {user_id}): {jde}. Raw output: {response.choices[0].message.content if 'response' in locals() and response.choices else 'N/A'}")
    except Exception as e:
        logger.error(f"Error in AI analyze_conversation for user {user_id}: {e}", exc_info=True)
    
    return default_return # Fallback to default context if any error


async def smart_recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provide smarter music recommendations based on context and Spotify data."""
    if not update.message or not update.effective_user: return

    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    status_msg = await update.message.reply_text(f"🎵 Thinking of some music just for you, {user_name}...")

    try:
        # Refresh Spotify data if linked, non-blocking to initial analysis
        async def _fetch_spotify_data():
            if user_contexts.get(user_id, {}).get("spotify", {}).get("access_token"): 
                logger.info(f"Fetching latest Spotify data (user {user_id}) for smart recommendations.")
                rp_task = get_user_spotify_data(user_id, "player/recently-played", params={"limit": 5})
                tt_task = get_user_spotify_data(user_id, "top/tracks", params={"limit": 5, "time_range": "short_term"})
                recently_played_data, top_tracks_data = await asyncio.gather(rp_task, tt_task)
                if recently_played_data: user_contexts.setdefault(user_id, {}).setdefault("spotify", {})["recently_played"] = recently_played_data
                if top_tracks_data: user_contexts.setdefault(user_id, {}).setdefault("spotify", {})["top_tracks"] = top_tracks_data
        asyncio.create_task(_fetch_spotify_data())
        
        # Analyze existing context (mood, preferences, history, existing Spotify data)
        analysis_results = await analyze_conversation(user_id)
        current_mood = analysis_results.get("mood")
        
        # If mood is not set or neutral, prompt the user to set it via the /mood conversation.
        if not current_mood or current_mood == "neutral":
            await status_msg.delete() # Delete "Thinking..." message
            logger.info(f"User {user_id} has no clear mood for recommendations, initiating /mood conversation.")
            # set_mood returns the initial state for the conversation handler.
            # It needs `update` which should be the original command message update.
            await set_mood(update, context) 
            return

        await status_msg.edit_text(f"Okay {user_name}, you're feeling <b>{current_mood}</b>. Let's find some music...\nLooking for recommendations based on your vibe and preferences... 🎧", parse_mode=ParseMode.HTML)

        seed_track_ids, seed_artist_ids, seed_genre_list = [], [], analysis_results.get("genres", [])
        ai_inferred_artists = analysis_results.get("artists", []) # Artists inferred by AI

        spotify_user_ctx_data = user_contexts.get(user_id, {}).get("spotify", {})
        
        # Prioritize seeds: 1. Explicitly liked artists (if any match recent/top on Spotify for ID), 2. Recent tracks, 3. Top artists (not yet explicitly used here but could be).
        # Use AI inferred artists as seed if available and Spotify linked.
        spotify_client_token_for_general_api = await get_spotify_token()

        if ai_inferred_artists and spotify_client_token_for_general_api:
            for art_name in ai_inferred_artists:
                # Try to get Spotify ID for AI inferred artist name
                artist_search_result = await search_spotify_track(spotify_client_token_for_general_api, f"artist:{art_name}") # Search for artist type explicitly
                if artist_search_result and artist_search_result.get("artists") and artist_search_result["artists"][0].get("id"):
                     seed_artist_ids.append(artist_search_result["artists"][0]["id"])
                if len(seed_artist_ids) >= 2: break # Max 2 artist seeds

        if not seed_artist_ids and spotify_user_ctx_data.get("recently_played"): # Fallback to recent tracks if no AI artists match
            seed_track_ids.extend([
                t["track"]["id"] for t in spotify_user_ctx_data["recently_played"][:2] 
                if t.get("track") and t["track"].get("id") and isinstance(t["track"]["id"], str)
            ])
        
        # Attempt Spotify API recommendations
        if spotify_client_token_for_general_api and (seed_track_ids or seed_artist_ids or seed_genre_list):
            logger.info(f"Requesting Spotify API recommendations (user {user_id}) with seeds: tracks={seed_track_ids}, artists={seed_artist_ids}, genres={seed_genre_list}")
            
            spotify_recs = await get_spotify_recommendations(
                spotify_client_token_for_general_api, 
                seed_tracks=seed_track_ids[:2], 
                seed_genres=seed_genre_list[:1], # Max 1 genre seed to keep variety
                seed_artists=seed_artist_ids[:2], 
                limit=5
            )
            
            if spotify_recs:
                resp_html = f"🎵 Based on your <b>{current_mood}</b> mood & preferences, Spotify suggests:\n\n"
                kb_spotify_recs = []
                for i, track in enumerate(spotify_recs, 1):
                    artists_str = ", ".join([a["name"] for a in track.get("artists",[])])
                    track_info = f"<b>{track['name']}</b> by <i>{artists_str}</i>"
                    if track.get("album", {}).get("name"): track_info += f" (from <i>{track['album']['name']}</i>)"
                    resp_html += f"{i}. {track_info}\n"
                    
                    # Create a YouTube search query for this Spotify track
                    yt_query = f"{track['name']} {artists_str}"
                    cb_yt_query = yt_query[:40] # Truncate for callback data limit
                    kb_spotify_recs.append([InlineKeyboardButton(f"YT Search: {track['name'][:20]}...", callback_data=f"{CB_SHOW_OPTIONS_PREFIX}{cb_yt_query}")])
                
                resp_html += "\n💡 <i>Click a track to search for it on YouTube.</i>"
                await status_msg.edit_text(resp_html, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(kb_spotify_recs))
                return

        # Fallback: YouTube search if Spotify recs fail or not enough seeds
        logger.info(f"Spotify recs failed or no seeds for user {user_id}. Falling back to YouTube search.")
        yt_query_parts = [current_mood] # Start with mood
        if seed_genre_list: yt_query_parts.append(seed_genre_list[0]) # Add primary genre
        if ai_inferred_artists: yt_query_parts.append(f"like {ai_inferred_artists[0]}") # Add an inferred artist
        elif spotify_user_ctx_data.get("recently_played"): # Fallback to recent artist if no AI artist
            try:
                rp_artist = spotify_user_ctx_data["recently_played"][0]['track']['artists'][0]['name']
                yt_query_parts.append(f"like {rp_artist}")
            except: pass
        
        youtube_search_query = " ".join(yt_query_parts) + " music playlist" # Add "playlist" for broader, mood-fitting results
        logger.info(f"YouTube fallback recommendation search (user {user_id}) with query: '{youtube_search_query}'")
        await status_msg.edit_text(f"Searching YouTube for <b>{current_mood}</b> tracks matching '<i>{youtube_search_query[:50]}...</i>'", parse_mode=ParseMode.HTML)

        yt_results = await asyncio.to_thread(search_youtube_sync, youtube_search_query, max_results=5)
        if yt_results:
            resp_html_yt = f"🎵 Here are some YouTube suggestions for your <b>{current_mood}</b> mood, {user_name}:\n\n"
            kb_yt_recs = []
            valid_yt_count = 0
            for res in yt_results:
                if not res.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', res['id']): continue # Skip invalid
                valid_yt_count +=1
                
                dur = res.get('duration', 0)
                dur_str = ""
                if dur and isinstance(dur, (int,float)) and dur > 0: 
                    m, s = divmod(int(dur), 60)
                    dur_str = f" [{m}:{s:02d}]"
                resp_html_yt += f"{valid_yt_count}. <b>{res['title']}</b> - <i>{res.get('uploader', 'N/A')}</i>{dur_str}\n"
                
                btn_title_text = res['title'][:30] + "..." if len(res['title']) > 33 else res['title']
                kb_yt_recs.append([InlineKeyboardButton(f"DL: {btn_title_text}", callback_data=f"{CB_DOWNLOAD_PREFIX}{res['id']}")])
            
            if not kb_yt_recs: # All YT results were invalid somehow
                 logger.info(f"No valid YouTube results for '{youtube_search_query}', providing generic recommendations.")
                 await status_msg.delete()
                 await provide_generic_recommendations(update, current_mood, chat_id_override=user_id)
                 return

            resp_html_yt += "\n💡 <i>Click a track to download its audio.</i>"
            await status_msg.edit_text(resp_html_yt, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(kb_yt_recs))
        else: 
            logger.info(f"No YouTube results for '{youtube_search_query}', providing generic recommendations for mood {current_mood} (user {user_id}).")
            await status_msg.delete() 
            await provide_generic_recommendations(update, current_mood, chat_id_override=user_id) # Use current update for bot context

    except Exception as e:
        logger.error(f"Error in smart_recommend_music for user {user_id}: {e}", exc_info=True)
        if status_msg:
            try: await status_msg.edit_text(f"Oh dear, I ran into a snag trying to find recommendations for you, {user_name}! 😥 Perhaps try again in a little bit?", parse_mode=ParseMode.HTML)
            except: pass # Avoid error in error handling


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates and send a user-friendly message."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    
    # Default user-facing error message
    error_message_text = "😓 Oops! Something went a bit sideways on my end. My human supervisors have been notified. Please try again in a moment!"

    # More specific messages for common errors
    if isinstance(context.error, TimedOut):
        error_message_text = "🐢 Things are a bit slow right now, and the operation timed out. Could you please try that again?"
    elif isinstance(context.error, NetworkError):
         error_message_text = "📡 I seem to be having trouble connecting to the wider web. Please check if Telegram is having issues, or try again in a bit."
    # Add more specific error types if you notice them occurring frequently
    # elif isinstance(context.error, telegram.error.BadRequest):
    #     error_message_text = "Hmm, that request doesn't look quite right. If you were using a command, maybe check /help?"

    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(error_message_text, parse_mode=ParseMode.HTML)
        except Exception as e_reply:
            logger.error(f"Failed to send error reply to user via effective_message: {e_reply}")
    elif isinstance(update, Update) and update.callback_query and update.callback_query.message:
        try: # Try to send reply in the chat of the callback query
            await update.callback_query.message.reply_text(error_message_text, parse_mode=ParseMode.HTML)
        except Exception as e_reply_cb:
            logger.error(f"Failed to send error reply for callback to user: {e_reply_cb}")
    else: # Fallback if update object structure is not as expected
        logger.info(f"Error handler called with non-standard update type: {type(update)}. Context error: {context.error}")


def cleanup_downloads_atexit() -> None:
    """Clean up temporary audio files from DOWNLOAD_DIR on exit."""
    logger.info("Cleaning up temporary download files at exit...")
    cleaned_count = 0
    try:
        if os.path.exists(DOWNLOAD_DIR):
            for item_name in os.listdir(DOWNLOAD_DIR):
                item_path = os.path.join(DOWNLOAD_DIR, item_name)
                try:
                    # Be more specific: target files generated by this bot
                    # (e.g., common audio extensions, or specific filename patterns like voice_ or videoID_)
                    is_temp_audio = item_path.endswith((".m4a", ".mp3", ".webm", ".ogg", ".opus", ".aac"))
                    is_temp_voice = "voice_" in item_name and item_path.endswith(".ogg")
                    is_temp_ytdl = re.match(r"^[a-zA-Z0-9_-]{11}_\d{14}\..+", item_name) # Matches videoID_timestamp.ext

                    if os.path.isfile(item_path) and (is_temp_audio or is_temp_voice or is_temp_ytdl):
                        os.remove(item_path)
                        cleaned_count +=1
                        logger.debug(f"Cleaned temp file: {item_path}")
                except Exception as e_file_remove:
                    logger.error(f"Failed to remove temporary file {item_path} during cleanup: {e_file_remove}")
            if cleaned_count > 0:
                logger.info(f"Successfully cleaned {cleaned_count} temporary file(s) from '{DOWNLOAD_DIR}'.")
            else:
                logger.info(f"No temporary files matching cleanup criteria found in '{DOWNLOAD_DIR}'.")
        else:
            logger.info(f"Download directory '{DOWNLOAD_DIR}' not found, no file cleanup needed at exit.")
    except Exception as e_cleanup_dir:
        logger.error(f"Error during atexit cleanup of downloads directory '{DOWNLOAD_DIR}': {e_cleanup_dir}")

def signal_exit_handler(sig, frame) -> None:
    """Handle termination signals gracefully."""
    logger.info(f"Received signal {sig}, preparing for graceful shutdown...")
    # cleanup_downloads_atexit() is registered with atexit, so it will run on sys.exit().
    # However, explicit call can be good insurance for forceful signals.
    if sig in [signal.SIGINT, signal.SIGTERM]:
        logger.info("Ensuring download cleanup is attempted before exit due to signal.")
        cleanup_downloads_atexit() # Attempt cleanup now
    sys.exit(0) # This will trigger other atexit registered functions too.

def main() -> None:
    """Start the bot."""
    # Configure bot application with increased timeouts and rate limiting
    application = (
        Application.builder()
        .token(TOKEN) # Assumes TOKEN is valid and checked before main()
        .connect_timeout(20.0)  
        .read_timeout(40.0)     
        .write_timeout(60.0)    
        .pool_timeout(180.0)    
        .rate_limiter(AIORateLimiter(overall_max_rate=20, max_retries=3)) # Default PTB overall is 30. Max retries is good.
        .build()
    )

    # --- Command Handlers ---
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("download", download_music))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("autodownload", auto_download_command))
    application.add_handler(CommandHandler("lyrics", get_lyrics_command))
    application.add_handler(CommandHandler("recommend", smart_recommend_music)) # Smart recommendations
    application.add_handler(CommandHandler("create_playlist", create_playlist))
    application.add_handler(CommandHandler("clear", clear_history))
    # Global /spotify_code handler (if called outside conversation or as alternative entry)
    # Note: spotify_code_handler itself determines state transitions
    application.add_handler(CommandHandler("spotify_code", spotify_code_command))


    # --- Conversation Handlers ---
    # Spotify Linking Conversation
    spotify_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("link_spotify", link_spotify)],
        states={
            SPOTIFY_CODE: [
                # Handles text messages (pasted code) in SPOTIFY_CODE state
                MessageHandler(filters.TEXT & ~filters.COMMAND, spotify_code_handler),
                # Handles `/spotify_code YOUR_CODE` command in SPOTIFY_CODE state (preferred over global)
                CommandHandler("spotify_code", spotify_code_handler), 
                # Handles "Cancel" button callback specifically for Spotify linking
                CallbackQueryHandler(cancel_spotify, pattern=f"^{CB_CANCEL_SPOTIFY}$") 
            ]
        },
        fallbacks=[
            CommandHandler("cancel", cancel), # Generic /cancel command
            CallbackQueryHandler(cancel_spotify, pattern=f"^{CB_CANCEL_SPOTIFY}$") # Ensure cancel button always works
        ],
        conversation_timeout=timedelta(minutes=10).total_seconds() 
    )
    application.add_handler(spotify_conv_handler)

    # Mood & Preference Setting Conversation
    mood_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("mood", set_mood)],
        states={
            MOOD: [CallbackQueryHandler(enhanced_button_handler, pattern=f"^{CB_MOOD_PREFIX}")], # Handles mood buttons
            PREFERENCE: [CallbackQueryHandler(enhanced_button_handler, pattern=f"^{CB_PREFERENCE_PREFIX}")], # Handles preference buttons
        },
        fallbacks=[CommandHandler("cancel", cancel)], # Generic /cancel command
        conversation_timeout=timedelta(minutes=5).total_seconds()
    )
    application.add_handler(mood_conv_handler)

    # --- Message and Callback Handlers (Order can be important) ---
    application.add_handler(MessageHandler(filters.VOICE & ~filters.COMMAND, handle_voice))
    
    # General CallbackQueryHandler for other buttons (download, search options, etc.)
    # This should come AFTER specific conversation handler callbacks if patterns overlap.
    # Current patterns (CB_MOOD_PREFIX, CB_PREFERENCE_PREFIX, CB_CANCEL_SPOTIFY) are distinct enough.
    application.add_handler(CallbackQueryHandler(enhanced_button_handler)) 
    
    # General text message handler (for chat, URL detection, AI requests) - should be one of the last.
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, enhanced_handle_message))
    
    # Error handler
    application.add_error_handler(error_handler)

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_exit_handler) # Ctrl+C
    signal.signal(signal.SIGTERM, signal_exit_handler) # Kill signal
    atexit.register(cleanup_downloads_atexit) # Register cleanup for normal exit

    logger.info("🚀 Starting MelodyMind Bot... Attempting to connect to Telegram.")
    try:
        # Run the bot until an interrupt
        application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
    except TimedOut:
        logger.critical("Bot timed out connecting to Telegram. Check network or token.", exc_info=True)
    except NetworkError as ne:
         logger.critical(f"Network error starting bot: {ne}. Check network.", exc_info=True)
    except Exception as e:
        logger.critical(f"Bot polling failed to start or crashed critically: {e}", exc_info=True)
    finally:
        logger.info(" MelodyMind Bot has shut down.")


if __name__ == "__main__":
    if not TOKEN:
        logger.critical("FATAL: TELEGRAM_TOKEN is not set in the environment. Bot cannot start.")
        sys.exit(1)
    if not OPENAI_API_KEY:
        logger.warning("WARNING: OPENAI_API_KEY not set. AI-powered features (chat, music request detection, mood analysis) will be disabled or degraded.")
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        logger.warning("WARNING: SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET not set. Spotify general search/recommendation features may be limited. User-specific Spotify features require these AND linking.")
    if not GENIUS_ACCESS_TOKEN:
        logger.warning("WARNING: GENIUS_ACCESS_TOKEN not set. Lyrics functionality will be disabled.")
    else:
        if not lyricsgenius: # Check if library failed to import
            logger.warning("WARNING: lyricsgenius library not found/imported, but GENIUS_ACCESS_TOKEN is set. Lyrics functionality will be disabled. Install with 'pip install lyricsgenius'.")
    
    if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET and SPOTIFY_REDIRECT_URI == "https://your-callback-url.com":
        logger.warning("WARNING: SPOTIFY_REDIRECT_URI is set to the default placeholder 'https://your-callback-url.com'. "
                       "Spotify user linking (/link_spotify) will require manual code copying from the URL. "
                       "For a smoother experience, set a proper redirect URI in your .env and Spotify Developer Dashboard.")

    main()
