
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
# from functools import lru_cache # Replaced by async_lru for async functions

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
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN) if GENIUS_ACCESS_TOKEN and lyricsgenius else None

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
async def get_spotify_recommendations(token: str, seed_tracks: List[str], limit: int = 5) -> List[Dict]:
    """Get track recommendations from Spotify."""
    if not token or not seed_tracks:
        logger.warning("No token or seed tracks provided for Spotify recommendations.")
        return []

    url = "https://api.spotify.com/v1/recommendations"
    headers = {"Authorization": f"Bearer {token}"}
    # Spotify API allows up to 5 seed entities (tracks, artists, genres). Use up to 2 seed_tracks.
    params = {"seed_tracks": ",".join(seed_tracks[:2]), "limit": limit}


    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                return (await response.json()).get("tracks", [])
    except aiohttp.ClientError as e:
        logger.error(f"Error getting Spotify recommendations (seeds: {seed_tracks}): {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error getting Spotify recommendations (seeds: {seed_tracks}): {e}")
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
                # Calculate expiry timestamp immediately
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
    
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET: # Required for refreshing
        logger.error("Cannot refresh Spotify token: Client ID or Secret not configured.")
        return None

    try:
        refresh_token_str = cipher.decrypt(encrypted_refresh_token_bytes).decode()
    except Exception as e:
        logger.error(f"Failed to decrypt refresh token for user {user_id}: {e}. Re-authentication required.")
        # Potentially clear bad token data here
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
                
                # Spotify may or may not return a new refresh token. If it does, use it.
                new_refresh_token_str = token_data.get("refresh_token", refresh_token_str)

                user_contexts[user_id]["spotify"]["access_token"] = cipher.encrypt(new_access_token.encode())
                user_contexts[user_id]["spotify"]["refresh_token"] = cipher.encrypt(new_refresh_token_str.encode())
                user_contexts[user_id]["spotify"]["expires_at"] = expires_at
                
                return new_access_token
    except aiohttp.ClientError as e:
        logger.error(f"Error refreshing Spotify token for user {user_id}: {e}")
        if e.status == 400: # Bad request, often due to invalid refresh token
             logger.error(f"Spotify refresh token for user {user_id} might be revoked. Re-authentication needed.")
             if "spotify" in user_contexts.get(user_id, {}):
                user_contexts[user_id]["spotify"] = {} # Clear outdated token info
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
    request_params = {"limit": 10, **(params or {})} # Default limit 10, allow override

    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.get(url, headers=headers, params=request_params) as response:
                response.raise_for_status()
                return (await response.json()).get("items", [])
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching Spotify user data ({endpoint}) for user {user_id}: {e}")
        if e.status == 401: # Unauthorized
            logger.info(f"Spotify token unauthorized for user {user_id}, attempting refresh (might be handled by get_user_spotify_access_token on next call).")
            # Forcing a refresh here might lead to loops if refresh also fails.
            # Better to let get_user_spotify_access_token handle it on subsequent calls.
        return None
    except Exception as e:
        logger.error(f"Unexpected error fetching Spotify user data ({endpoint}) for user {user_id}: {e}")
        return None

# ==================== YOUTUBE HELPER FUNCTIONS ====================

def is_valid_youtube_url(url: str) -> bool:
    """Check if the URL is a valid YouTube URL."""
    if not url:
        return False
    # Simpler and more common patterns
    patterns = [
        r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    ]
    return any(re.search(pattern, url) for pattern in patterns)

def sanitize_filename(filename: str) -> str:
    """Remove invalid characters from filenames for display or metadata, NOT for yt-dlp paths directly."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return sanitized[:100] # Truncate for safety/display

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def download_youtube_audio_sync(url: str) -> Dict[str, Any]: # Renamed to indicate it's synchronous for clarity
    """Download audio from a YouTube video with retries. This is a BLOCKING function."""
    logger.info(f"Attempting to download audio from: {url}")
    
    # Extract video ID for logging/potential use, though not strictly needed for yt-dlp itself if URL is valid
    video_id_match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11})', url)
    video_id = video_id_match.group(1) if video_id_match else "UnknownID"

    try:
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best', # Prioritize m4a, then best audio
            'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'),
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            # 'postprocessors': [{ # Usually not needed if format selector is good
            #     'key': 'FFmpegExtractAudio',
            #     'preferredcodec': 'm4a',
            # }],
            'max_filesize': 50 * 1024 * 1024, # 50MB
            'writethumbnail': True, # To get thumbnail path if needed later
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False) # Get info first
            if not info:
                logger.error(f"Could not extract video information for {url} (ID: {video_id})")
                return {"success": False, "error": "Could not extract video information"}

            # Use title from info for metadata, sanitize it for display purposes
            display_title = sanitize_filename(info.get('title', 'Unknown Title'))
            artist = info.get('artist', info.get('uploader', 'Unknown Artist'))
            
            # This gets the filename yt-dlp *will* use based on outtmpl and info
            # yt-dlp handles its own path sanitization for the filename.
            expected_audio_path = ydl.prepare_filename(info)

            logger.info(f"Downloading '{display_title}' to '{expected_audio_path}'")
            ydl.extract_info(url, download=True) # Perform the actual download
            # info = ydl.download([url]) # Alternative download call if needed. extract_info(download=True) is often preferred.


            if not os.path.exists(expected_audio_path):
                # Fallback if prepare_filename's extension differs slightly or casing
                # This is less likely if 'format' forces an extension or prepare_filename is accurate
                found_path = None
                base_name_no_ext, _ = os.path.splitext(expected_audio_path)
                for f in os.listdir(DOWNLOAD_DIR):
                    if os.path.splitext(f)[0] == os.path.basename(base_name_no_ext): # Match filename part
                        found_path = os.path.join(DOWNLOAD_DIR, f)
                        logger.info(f"File found at slightly different path: {found_path}")
                        break
                if not found_path:
                    logger.error(f"Downloaded file not found at expected path: {expected_audio_path} for {url}")
                    return {"success": False, "error": "Downloaded file not found or inaccessible"}
                expected_audio_path = found_path


            file_size_mb = os.path.getsize(expected_audio_path) / (1024 * 1024)
            if file_size_mb > 50: # Double check, though ydl_opts should handle it
                os.remove(expected_audio_path)
                logger.warning(f"File '{display_title}' exceeded 50MB limit ({file_size_mb:.2f}MB), removing.")
                return {"success": False, "error": "File exceeds 50 MB Telegram limit after download"}
            
            thumbnail_path = info.get('thumbnail') # URL
            # If writethumbnail was true, thumbnail might be local, check info.get('thumbnails')[-1]['filepath'] or similar

            return {
                "success": True,
                "title": display_title,
                "artist": artist,
                "thumbnail_url": thumbnail_path, # This is URL, not local path typically from 'thumbnail' key
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
        return {"success": False, "error": f"Download failed: {error_msg[:100]}"} # Keep it concise
    except Exception as e:
        logger.error(f"Generic error downloading YouTube audio {url} (ID: {video_id}): {e}", exc_info=True)
        return {"success": False, "error": f"An unexpected error occurred: {str(e)[:100]}"}

def search_youtube_sync(query: str, max_results: int = 5) -> List[Dict]: # Renamed
    """Search YouTube for videos matching the query. This is a BLOCKING function."""
    logger.info(f"Searching YouTube for: '{query}' with max_results={max_results}")
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'discard_in_playlist', # Faster for searches, gets playlist entries
            'default_search': f'ytsearch{max_results}', # Use ytsearchN: query directly
            # 'format': 'bestaudio', # Not needed for search usually, extract_flat handles it
            'noplaylist': True, # Should be true, search isn't a playlist context
            # 'playlist_items': f'1-{max_results}' # Covered by ytsearchN
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # For ytsearchN, query itself becomes the URL argument for extract_info
            info = ydl.extract_info(query, download=False)
            
            if not info or 'entries' not in info:
                logger.warning(f"No YouTube search results for query: '{query}'")
                return []
            
            results = []
            for entry in info['entries']:
                if not entry: continue # Skip if entry is None or empty
                results.append({
                    'title': entry.get('title', 'Unknown Title'),
                    'url': entry.get('url') or f"https://www.youtube.com/watch?v={entry.get('id')}",
                    'thumbnail': entry.get('thumbnail') or (entry.get('thumbnails')[0]['url'] if entry.get('thumbnails') else ''),
                    'uploader': entry.get('uploader', 'Unknown Artist'),
                    'duration': entry.get('duration', 0),
                    'id': entry.get('id', '') # Must have ID
                })
            logger.info(f"Found {len(results)} results for '{query}'")
            return results
            
    except yt_dlp.utils.DownloadError as de:
        logger.error(f"yt-dlp DownloadError during YouTube search for '{query}': {de}")
        return[] # Typically indicates search itself failed, not just no results
    except Exception as e:
        logger.error(f"Error searching YouTube for '{query}': {e}", exc_info=True)
        return []

# ==================== AI AND LYRICS FUNCTIONS ====================

async def generate_chat_response(user_id: int, message: str) -> str:
    """Generate a conversational response using OpenAI."""
    if not client:
        return "I'm having trouble connecting to my AI service. Please try again later."

    # Ensure basic context structure exists
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    context = user_contexts[user_id]
    context.setdefault("conversation_history", []) # Ensure history list exists

    context["conversation_history"] = context["conversation_history"][-50:]  # Limit history

    system_prompt = (
        "You are a friendly, empathetic music companion bot named MelodyMind. "
        "Your role is to: "
        "1. Have natural conversations about music and feelings. "
        "2. Recommend songs based on mood and preferences (if known). "
        "3. Provide emotional support through music-related chat. "
        "4. Keep responses concise but warm (around 2-3 sentences typically). "
        "If the user has linked their Spotify account and data is provided, use their listening history (e.g., recently played artists) to personalize responses if relevant. "
        "Do not explicitly mention analyzing their Spotify data unless it's natural. Just use it to sound more informed. "
        "Do not suggest commands like /download unless the user explicitly asks how to get a song. Focus on conversational interaction."
    )
    messages = [{"role": "system", "content": system_prompt}]

    # Add context summary as a system message for better grounding
    context_summary_parts = []
    if context.get("mood"):
        context_summary_parts.append(f"Current user mood: {context.get('mood')}.")
    if context.get("preferences"):
        context_summary_parts.append(f"User music preferences: {', '.join(context.get('preferences'))}.")
    
    # Spotify derived context (if available and not too noisy)
    if "spotify" in context and context["spotify"].get("recently_played"):
        try:
            artists = list(set(item["track"]["artists"][0]["name"] for item in context["spotify"]["recently_played"][:5] if item.get("track"))) # Top 5 unique recent artists
            if artists:
                context_summary_parts.append(f"User recently listened to artists like: {', '.join(artists)}.")
        except Exception: # Guard against unexpected data structure
            pass 
            
    if context_summary_parts:
        messages.append({"role": "system", "content": "User context summary: " + " ".join(context_summary_parts)})


    for hist_msg in context["conversation_history"][-10:]: # Last 5 exchanges
        messages.append(hist_msg)
    messages.append({"role": "user", "content": message})

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        reply = response.choices[0].message.content.strip()
        context["conversation_history"].extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply}
        ])
        # user_contexts[user_id] = context # Context is modified in-place
        return reply
    except Exception as e:
        logger.error(f"Error generating chat response for user {user_id}: {e}")
        return "I'm having a little trouble thinking of a reply right now. Maybe we can talk about your favorite song instead?"

def get_lyrics_sync(song_title: str, artist: Optional[str] = None) -> str: # Renamed
    """Get lyrics for a song using Genius API. This is a BLOCKING function."""
    if not genius:
        return "Lyrics service is currently unavailable. I can't fetch lyrics right now."
    logger.info(f"Fetching lyrics for song: '{song_title}' by artist: '{artist or 'Any'}'")
    try:
        # Genius library might perform network I/O, better to run in thread for async context
        if artist:
            song = genius.search_song(song_title, artist)
        else:
            song = genius.search_song(song_title)
            
        if not song:
            err_msg = f"Sorry, I couldn't find lyrics for '{song_title}'"
            if artist: err_msg += f" by '{artist}'"
            err_msg += ". Please check the spelling or try another song!"
            logger.warning(f"No lyrics found for '{song_title}' by '{artist or 'Any'}'")
            return err_msg
        
        lyrics = song.lyrics
        # Clean up common Genius artifacts
        lyrics = re.sub(r'\[.*?\]', '', lyrics)  # Remove [Chorus], [Verse], etc.
        lyrics = re.sub(r'\d*Embed$', '', lyrics.strip()) # Remove NNNEmbed footer
        lyrics = re.sub(r'^\S*Lyrics', '', lyrics.strip()) # Remove SongTitleLyrics header if present
        lyrics = lyrics.strip()

        if not lyrics: # After cleaning, if lyrics are empty
            logger.warning(f"Lyrics found for '{song.title}' but were empty after cleaning.")
            return f"Lyrics for '{song.title}' by {song.artist} seem to be empty or missing. Try another song?"

        header = f"🎵 **{song.title}** by *{song.artist}* 🎵\n\n"
        return header + lyrics
    except Exception as e: # Catching broad exception from Genius library
        logger.error(f"Error fetching lyrics for '{song_title}' from Genius: {e}", exc_info=True)
        return f"I encountered an issue trying to fetch lyrics for '{song_title}'. Please try again later."


async def detect_mood_from_text(user_id: int, text: str) -> str:
    """Detect mood from user's message using AI."""
    if not client:
        return user_contexts.get(user_id, {}).get("mood", "neutral") # Default to neutral if AI off
    logger.debug(f"Detecting mood from text for user {user_id}: '{text[:50]}...'")
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "You are a mood detection AI. Analyze the following text and return a single dominant mood word (e.g., happy, sad, anxious, excited, calm, angry, neutral). If unsure, return 'neutral'."},
                      {"role": "user", "content": f"Detect mood from this text: '{text}'"}],
            max_tokens=10, # Short response
            temperature=0.3
        )
        mood = response.choices[0].message.content.lower().strip().replace(".", "")
        # Basic validation, could be more robust
        valid_moods = ["happy", "sad", "anxious", "excited", "calm", "angry", "neutral", "energetic", "relaxed", "focused", "nostalgic"] # Add more if needed
        if mood in valid_moods:
            logger.info(f"Detected mood for user {user_id}: '{mood}'")
            return mood
        else: # If OpenAI returns something unexpected
            logger.warning(f"Unexpected mood from AI: '{mood}'. Defaulting to neutral.")
            return "neutral"

    except Exception as e:
        logger.error(f"Error detecting mood for user {user_id}: {e}")
        return user_contexts.get(user_id, {}).get("mood", "neutral") # Fallback to existing or neutral


async def is_music_request(user_id: int, message: str) -> Dict:
    """Use AI to determine if a message is a music/song request and extract query."""
    if not client:
        return {"is_music_request": False, "song_query": None}

    logger.debug(f"Checking if message is music request for user {user_id}: '{message[:50]}...'")
    try:
        # The function_call / tools API would be more robust here, but for simplicity with JSON mode:
        prompt_messages = [
            {"role": "system", "content": 
                "You are an AI that analyzes user messages. Determine if the message is primarily a request for a specific song, music by an artist, or a music download. "
                "If it is, respond in JSON format with two keys: 'is_music_request' (boolean) and 'song_query' (string, containing the song title and artist if specified, or general music query, null if not a music request). "
                "Focus on explicit requests like 'play song X by Y', 'download Z', 'find music like A'. If it's a general chat about music or mood, it's not a specific song request unless they name something."
            },
            {"role": "user", "content": f"Analyze this message: '{message}'"}
        ]
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo-0125", # Model that reliably supports JSON mode
            messages=prompt_messages,
            max_tokens=100,
            temperature=0.1, # Low temperature for more deterministic output
            response_format={"type": "json_object"}
        )

        result_str = response.choices[0].message.content
        logger.debug(f"AI music request analysis raw response: {result_str}")
        result = json.loads(result_str)

        if not isinstance(result, dict): # Should not happen with JSON mode but good check
            logger.error(f"AI music request analysis returned non-dict: {result}")
            return {"is_music_request": False, "song_query": None}

        is_request = result.get("is_music_request", False)
        # Ensure boolean, some models might return "true" as string in older setups
        if isinstance(is_request, str):
            is_request = is_request.lower() in ("yes", "true")
        
        song_query = result.get("song_query")
        if not isinstance(song_query, str) or not song_query.strip():
            song_query = None # Ensure None if empty or not string

        logger.info(f"AI music request analysis for user {user_id}: is_request={is_request}, query='{song_query}'")
        return {
            "is_music_request": bool(is_request),
            "song_query": song_query
        }
    except json.JSONDecodeError as jde:
        logger.error(f"Failed to decode JSON from AI music request analysis: {jde}. Raw: {response.choices[0].message.content if 'response' in locals() else 'N/A'}")
        return {"is_music_request": False, "song_query": None}
    except Exception as e:
        logger.error(f"Error in AI is_music_request for user {user_id}: {e}", exc_info=True)
        return {"is_music_request": False, "song_query": None}


# ==================== TELEGRAM BOT HANDLERS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message."""
    user = update.effective_user
    # Initialize user context if not exists
    user_contexts.setdefault(user.id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})

    welcome_msg = (
        f"Hi {user.first_name}! 👋 I'm MelodyMind, your Music Healing Companion.\n\n"
        "I can help you:\n"
        "🎵 Download music from YouTube (just send a link or ask!)\n"
        "📜 Find song lyrics (e.g., `/lyrics Bohemian Rhapsody`)\n"
        "💿 Get music recommendations based on your mood (try `/recommend`)\n"
        "💬 Chat about music, how you're feeling, or anything on your mind!\n"
        "🔗 Link your Spotify account for more personalized experiences (use `/link_spotify`)\n"
        "📖 Create private Spotify playlists (e.g., `/create_playlist My Chill Mix`)\n\n"
        "Type `/help` for a full list of commands, or just start chatting!"
    )
    await update.message.reply_text(welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message."""
    help_text = (
        "🎶 <b>MelodyMind - Your Music Healing Companion</b> 🎶\n\n"
        "I'm here to chat, find music, and hopefully brighten your day!\n\n"
        "<b>Core Commands:</b>\n"
        "/start - Shows the welcome message.\n"
        "/help - Displays this help message.\n"
        "/download <code>[YouTube URL]</code> - Downloads audio from a YouTube link.\n"
        "   <i>(e.g., /download https://youtu.be/dQw4w9WgXcQ)</i>\n"
        "/autodownload <code>[song name/artist]</code> - Searches and downloads the top result.\n"
        "   <i>(e.g., /autodownload Shape of You Ed Sheeran)</i>\n"
        "/search <code>[song name/artist]</code> - Searches YouTube and shows download options.\n"
        "   <i>(e.g., /search Imagine Dragons Believer)</i>\n"
        "/lyrics <code>[song name]</code> or <code>[artist - song]</code> - Fetches song lyrics.\n"
        "   <i>(e.g., /lyrics Hotel California or /lyrics Eagles - Hotel California)</i>\n"
        "/recommend - Get personalized music recommendations. Uses Spotify data if linked, or asks your mood.\n"
        "/mood - Set or update your current mood to tailor recommendations.\n"
        "/link_spotify - Connect your Spotify account for enhanced features.\n"
        "/create_playlist <code>[playlist name]</code> - Creates a new private Spotify playlist (requires linked Spotify).\n"
        "/clear - Clears your conversation history with me.\n\n"
        "<b>Chat With Me!</b>\n"
        "You can also just talk to me naturally. Try things like:\n"
        "- \"I'm feeling a bit down today.\"\n"
        "- \"Can you play 'Bohemian Rhapsody' by Queen?\"\n"
        "- \"What are some good songs for studying?\"\n"
        "- Send a YouTube link directly to download it.\n"
        "- Send a voice message to request a song or chat!\n\n"
        "Let the music play! 🎵"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

async def download_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download music from YouTube URL provided in /download command or as a direct message."""
    message_text = update.message.text
    url = ""

    if context.args: # For /download <url>
        url = " ".join(context.args)
    else: # For direct message that is a URL
        urls_in_message = [word for word in message_text.split() if is_valid_youtube_url(word)]
        if urls_in_message:
            url = urls_in_message[0]
        else:
            await update.message.reply_text(
                "❌ To download, please provide a valid YouTube URL. Example:\n"
                "/download https://www.youtube.com/watch?v=dQw4w9WgXcQ\n"
                "Or just send me a YouTube link directly!"
            )
            return

    if not is_valid_youtube_url(url):
        await update.message.reply_text("❌ That doesn't look like a valid YouTube URL. Please send a valid YouTube link.")
        return

    user_id = update.effective_user.id
    if user_id in active_downloads:
        await update.message.reply_text("⚠️ You already have a download in progress. Please be patient! 😊")
        return

    active_downloads.add(user_id)
    status_msg = await update.message.reply_text("⏳ Starting download... This might take a moment.")

    try:
        await status_msg.edit_text("🔍 Fetching video information and preparing download...")
        # Run the blocking download function in a separate thread
        result = await asyncio.to_thread(download_youtube_audio_sync, url)
        
        if not result["success"]:
            error_message = result.get('error', 'Unknown download error.')
            await status_msg.edit_text(f"❌ Download failed: {error_message}")
            return

        await status_msg.edit_text(f"✅ Downloaded: {result['title']}\n⏳ Now sending the audio file to you...")
        
        audio_path = result["audio_path"]
        with open(audio_path, 'rb') as audio_file:
            logger.info(f"Sending audio '{result['title']}' to user {user_id}. Path: {audio_path}")
            await update.message.reply_audio(
                audio=audio_file,
                title=result["title"][:64], # Telegram API limit for title
                performer=result["artist"][:64] if result.get("artist") else "Unknown Artist", # Telegram API limit for performer
                caption=f"🎵 {result['title']}",
                duration=result.get('duration'),
                # thumbnail=result.get('thumbnail_path') # If thumbnail downloaded locally
            )

        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"Successfully deleted temporary file: {audio_path}")
            except OSError as e:
                logger.error(f"Error deleting temporary file {audio_path}: {e}")
        await status_msg.delete()
    except TimedOut:
        logger.error(f"Telegram API timeout during download process for user {user_id}, url: {url}")
        await status_msg.edit_text("❌ The operation timed out while communicating with Telegram. Please try again.")
    except NetworkError as ne:
        logger.error(f"Telegram API network error during download for user {user_id}, url: {url}: {ne}")
        await status_msg.edit_text(f"❌ A network error occurred with Telegram: {ne}. Please try again.")
    except Exception as e:
        logger.error(f"Unexpected error in download_music handler for user {user_id}, url: {url}: {e}", exc_info=True)
        try: # Try to inform user, but this might also fail if bot is unstable
            await status_msg.edit_text(f"❌ An unexpected error occurred: {str(e)[:100]}. My developers have been notified.")
        except Exception:
            pass # If sending status fails, nothing more to do here
    finally:
        active_downloads.discard(user_id)


async def create_playlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Create a private Spotify playlist."""
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Please provide a name for your playlist. Usage: `/create_playlist <Your Playlist Name>`")
        return
    
    playlist_name = " ".join(context.args)
    logger.info(f"User {user_id} attempting to create Spotify playlist: '{playlist_name}'")

    access_token = await get_user_spotify_access_token(user_id)
    if not access_token:
        await update.message.reply_text(
            "I couldn't access your Spotify account. 😥 Please link your Spotify account first using /link_spotify or re-link if you're having issues."
        )
        return

    # Get Spotify User ID first (required for creating playlist for the user)
    user_profile_url = "https://api.spotify.com/v1/me"
    headers_auth = {"Authorization": f"Bearer {access_token}"}
    spotify_user_id = None
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.get(user_profile_url, headers=headers_auth) as response:
                response.raise_for_status()
                spotify_user_id = (await response.json()).get("id")
        if not spotify_user_id:
            logger.error(f"Could not retrieve Spotify user ID for Telegram user {user_id}.")
            await update.message.reply_text("Sorry, I couldn't fetch your Spotify profile ID. Can't create the playlist right now.")
            return
    except aiohttp.ClientError as e:
        logger.error(f"API error fetching Spotify profile for user {user_id}: {e}")
        await update.message.reply_text("Sorry, there was an issue fetching your Spotify profile. Please try again.")
        return
    
    playlist_creation_url = f"https://api.spotify.com/v1/users/{spotify_user_id}/playlists"
    headers_create = {**headers_auth, "Content-Type": "application/json"}
    payload = {"name": playlist_name, "public": False, "description": "Created with MelodyMind Bot"}

    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(playlist_creation_url, headers=headers_create, json=payload) as response:
                response.raise_for_status()
                playlist_data = await response.json()
                playlist_url = playlist_data.get("external_urls", {}).get("spotify", "#")
                logger.info(f"Playlist '{playlist_name}' created successfully for user {user_id}. URL: {playlist_url}")
                await update.message.reply_text(
                    f"🎉 Playlist '{playlist_name}' created successfully on Spotify!\n"
                    f"You can find it here: {playlist_url}",
                    disable_web_page_preview=True
                )
    except aiohttp.ClientError as e:
        logger.error(f"API error creating Spotify playlist '{playlist_name}' for user {user_id}: {e}")
        status = getattr(e, 'status', 'Unknown')
        message = getattr(e, 'message', str(e))
        await update.message.reply_text(f"Oops! Failed to create playlist on Spotify (Error {status}: {message[:100]}). Please try again later.")
    except Exception as e:
        logger.error(f"Unexpected error creating Spotify playlist for user {user_id}: {e}", exc_info=True)
        await update.message.reply_text("An unexpected error occurred while creating the playlist. Sorry about that!")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle voice messages by transcribing them and processing as text."""
    if not update.message or not update.message.voice:
        return

    user_id = update.effective_user.id
    logger.info(f"Received voice message from user {user_id}")
    
    voice_file = await context.bot.get_file(update.message.voice.file_id)
    # Using a unique filename to avoid conflicts if multiple users send voice simultaneously
    temp_ogg_path = os.path.join(DOWNLOAD_DIR, f"voice_{user_id}_{update.message.message_id}.ogg")
    
    await voice_file.download_to_drive(temp_ogg_path)
    logger.debug(f"Voice message for user {user_id} downloaded to {temp_ogg_path}")

    # Speech-to-text is blocking, so run in a thread
    recognizer = sr.Recognizer()
    transcribed_text = None
    try:
        def _transcribe_audio():
            with sr.AudioFile(temp_ogg_path) as source:
                audio_data = recognizer.record(source)  # Read the entire audio file
            try:
                return recognizer.recognize_google(audio_data) # Using Google Web Speech API
            except sr.UnknownValueError:
                logger.warning(f"Google Web Speech API could not understand audio from user {user_id}")
                return None
            except sr.RequestError as e:
                logger.error(f"Could not request results from Google Web Speech API for user {user_id}; {e}")
                return "ERROR_REQUEST"

        transcribed_text = await asyncio.to_thread(_transcribe_audio)

        if transcribed_text == "ERROR_REQUEST":
            await update.message.reply_text("Sorry, I'm having trouble connecting to the voice recognition service right now. Please try again later or type your message.")
        elif transcribed_text:
            logger.info(f"Voice message from user {user_id} transcribed as: '{transcribed_text}'")
            await update.message.reply_text(f"🎤 I heard: \"<i>{transcribed_text}</i>\"\nLet me see...", parse_mode=ParseMode.HTML)
            
            # Simulate a new text message update to pass to the main message handler
            # This is a bit of a workaround; a more direct routing might be cleaner
            # Ensure the original message is available in context if needed
            context.user_data['_voice_original_message'] = update.message 
            fake_message = update.message._replace(text=transcribed_text, voice=None) # Remove voice to avoid reprocessing
            fake_update = Update(update.update_id, message=fake_message)
            await enhanced_handle_message(fake_update, context)
        else:
            await update.message.reply_text("Hmm, I couldn't quite catch that. Could you try speaking again, or type your message? 😊")

    except Exception as e:
        logger.error(f"Error processing voice message for user {user_id}: {e}", exc_info=True)
        await update.message.reply_text("Oops! Something went wrong while processing your voice message. Please try again.")
    finally:
        # Clean up the downloaded temporary ogg file
        if os.path.exists(temp_ogg_path):
            try:
                os.remove(temp_ogg_path)
                logger.debug(f"Deleted temporary voice file: {temp_ogg_path}")
            except OSError as e:
                logger.error(f"Error deleting temporary voice file {temp_ogg_path}: {e}")


async def link_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Initiate Spotify OAuth flow."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI:
        await update.message.reply_text(
            "Sorry, the Spotify linking feature is not configured by the bot admin. 😥"
        )
        return ConversationHandler.END
    if SPOTIFY_REDIRECT_URI == "https://your-callback-url.com": # Default placeholder
         await update.message.reply_text(
            "⚠️ The Spotify redirect URI is set to a placeholder. Linking might not work as expected "
            "unless you manually copy the code from the redirected URL's query parameters."
        )

    user_id = update.effective_user.id
    # Scopes: user-read-recently-played, user-top-read, playlist-modify-private (for /create_playlist)
    # Ensure playlist-modify-public if you add public playlist creation
    scopes = "user-read-recently-played user-top-read playlist-modify-private"
    auth_url = (
        "https://accounts.spotify.com/authorize"
        f"?client_id={SPOTIFY_CLIENT_ID}"
        "&response_type=code"
        f"&redirect_uri={SPOTIFY_REDIRECT_URI}"
        f"&scope={scopes.replace(' ', '%20')}" # URL encode scopes
        f"&state={user_id}"  # Using user_id in state can be a simple check, though more robust state mechanisms exist
    )
    keyboard = [
        [InlineKeyboardButton("🔗 Link My Spotify Account", url=auth_url)],
        [InlineKeyboardButton("Cancel", callback_data=CB_CANCEL_SPOTIFY)]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        "Let's link your Spotify account for personalized music experiences! 🎵\n\n"
        "1. Click the button below to go to Spotify and authorize MelodyMind.\n"
        "2. After authorizing, Spotify will redirect you. From the redirected page's URL, copy the `code` value.\n"
        "   (It looks like `https://your-redirect-uri.com/?code=A_LONG_STRING_OF_CHARS&state=...` - you need `A_LONG_STRING_OF_CHARS`)\n"
        "3. Send that code back to me here.\n\n"
        "If you see an error page from Spotify, ensure the bot admin has configured the redirect URI correctly in Spotify's Developer Dashboard.",
        reply_markup=reply_markup,
        parse_mode=ParseMode.MARKDOWN # Using Markdown for clarity with the example URL
    )
    return SPOTIFY_CODE

async def spotify_code_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle Spotify authorization code from the user."""
    user_id = update.effective_user.id
    message_text = update.message.text.strip()
    
    code_to_use = None
    if message_text.startswith('/spotify_code') and context.args: # From /spotify_code <arg>
        code_to_use = context.args[0]
    elif not message_text.startswith('/'): # Pasted code directly
        code_to_use = message_text
    # Could also parse if user pastes full redirect URL, e.g. extract 'code=' param

    if not code_to_use:
        await update.message.reply_text(
            "Please send me the Spotify authorization code you copied from the redirect URL.\n"
            "You can paste the code directly, or use `/spotify_code YOUR_CODE`."
        )
        return SPOTIFY_CODE # Stay in this state

    status_msg = await update.message.reply_text("⏳ Validating your Spotify code...")
    token_data = await get_user_spotify_token(user_id, code_to_use)

    if not token_data or not token_data.get("access_token"):
        await status_msg.edit_text(
            "❌ Failed to link Spotify. The code might be invalid, expired, or there was a configuration issue. "
            "Please try /link_spotify again. Make sure you copy the `code` parameter correctly from the URL."
        )
        return SPOTIFY_CODE # Stay, allow user to try again with a new code

    # Ensure context structure
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    
    # Encrypt and store tokens
    user_contexts[user_id]["spotify"] = {
        "access_token": cipher.encrypt(token_data["access_token"].encode()),
        "refresh_token": cipher.encrypt(token_data["refresh_token"].encode()), # refresh_token must exist here
        "expires_at": token_data["expires_at"]
    }
    logger.info(f"Spotify account successfully linked for user {user_id}.")

    # Optionally, fetch some initial data to confirm & personalize
    recently_played = await get_user_spotify_data(user_id, "player/recently-played", params={"limit": 5})
    if recently_played:
        user_contexts[user_id]["spotify"]["recently_played"] = recently_played
        # Could mention a recently played artist here as confirmation

    await status_msg.edit_text(
        "✅ Spotify linked successfully! 🎉 I can now give you more personalized recommendations. Try /recommend!"
    )
    return ConversationHandler.END

async def spotify_code_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
    """Handle /spotify_code command. Mostly a passthrough to spotify_code_handler logic."""
    # This command might be invoked outside the ConversationHandler context.
    # For simplicity, let's assume if it's called, we want to process it like a direct code entry.
    # The main `spotify_code_handler` function will deal with actual processing.
    # To ensure it flows into the conversation handler if active, one might need to be careful
    # For now, this directly processes if args are provided.
    if not context.args:
        await update.message.reply_text(
            "Please provide the Spotify authorization code after the command. Example:\n`/spotify_code YOUR_CODE_HERE`"
        )
        # If in SPOTIFY_CODE state, remain there. If not, it's just a mis-command.
        return SPOTIFY_CODE if context. Espírito Santo == SPOTIFY_CODE else None 
    return await spotify_code_handler(update, context) # Re-use the logic


async def cancel_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel Spotify linking."""
    query = update.callback_query
    await query.answer() # Acknowledge callback
    await query.edit_message_text("Spotify linking process cancelled. You can always try again using /link_spotify. 👍")
    return ConversationHandler.END


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /search command for YouTube."""
    if not context.args:
        await update.message.reply_text(
            "What song are you looking for? Please tell me! Example:\n`/search Shape of You Ed Sheeran`"
        )
        return

    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Searching YouTube for: '{query}'...")
    
    results = await asyncio.to_thread(search_youtube_sync, query, max_results=5)
    
    await status_msg.delete() # Clean up "Searching..." message
    await send_search_results(update, query, results)

async def auto_download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /autodownload command: searches YouTube and downloads the first result."""
    if not context.args:
        await update.message.reply_text(
            "What song should I download for you? Example:\n`/autodownload Imagine Dragons Believer`"
        )
        return

    query = " ".join(context.args)
    # This will directly use the download logic, effectively simulating clicking first result.
    # The 'download_music' handler is more suited for direct URL, so we use auto_download_first_result
    await auto_download_first_result(update, context, query)

async def get_lyrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle lyrics requests via /lyrics command."""
    if not context.args:
        await update.message.reply_text(
            "Please tell me the song (and optionally artist) for the lyrics. Examples:\n"
            "`/lyrics Bohemian Rhapsody`\n"
            "`/lyrics Queen - Bohemian Rhapsody`"
        )
        return

    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Searching for lyrics for: \"<i>{query}</i>\"...", parse_mode=ParseMode.HTML)

    try:
        artist = None
        song_title = query
        # Try to parse "Artist - Song" or "Song by Artist"
        if " - " in query:
            parts = query.split(" - ", 1)
            artist, song_title = parts[0].strip(), parts[1].strip()
        elif " by " in query.lower(): # Case-insensitive "by"
            # Regex to be more robust with "by" in titles
            match = re.search(r'^(.*?)\s+by\s+(.*?)$', query, re.IGNORECASE)
            if match:
                song_title, artist = match.group(1).strip(), match.group(2).strip()
        
        logger.info(f"Processed lyrics query: song='{song_title}', artist='{artist}'")
        
        lyrics_text = await asyncio.to_thread(get_lyrics_sync, song_title, artist)
        
        # Send lyrics, potentially in chunks if too long
        if len(lyrics_text) > 4000: # Telegram message length limit ~4096
            await status_msg.edit_text(f"{lyrics_text[:4000]}\n\n<small>(Lyrics too long, continued below)</small>", parse_mode=ParseMode.HTML)
            remaining_lyrics = lyrics_text[4000:]
            while remaining_lyrics:
                chunk = remaining_lyrics[:4000]
                remaining_lyrics = remaining_lyrics[4000:]
                # Add continuation note if there's still more
                chunk_message = chunk + ("\n\n<small>(...continued)</small>" if remaining_lyrics else "")
                await update.message.reply_text(chunk_message, parse_mode=ParseMode.HTML)
        else:
            await status_msg.edit_text(lyrics_text, parse_mode=ParseMode.HTML) # Use HTML for bold/italic from get_lyrics_sync

    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error in get_lyrics_command for query '{query}': {e}", exc_info=True)
        await status_msg.edit_text("Sorry, an unexpected hiccup occurred while fetching lyrics. 😕 Please try again.")


async def recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provide music recommendations. (Alias for smart_recommend_music now)."""
    await smart_recommend_music(update, context)


async def provide_generic_recommendations(update: Update, mood: str) -> None:
    """Provide generic, hardcoded recommendations as a fallback."""
    logger.info(f"Providing generic recommendations for mood: {mood}")
    mood_recommendations = {
        "happy": [
            "Walking on Sunshine - Katrina & The Waves", "Happy - Pharrell Williams", "Can't Stop the Feeling - Justin Timberlake",
            "Uptown Funk - Mark Ronson ft. Bruno Mars", "Good Vibrations - The Beach Boys"
        ],
        "sad": [
            "Someone Like You - Adele", "Fix You - Coldplay", "Everybody Hurts - R.E.M.",
            "Nothing Compares 2 U - Sinéad O'Connor", "Tears in Heaven - Eric Clapton"
        ],
        "energetic": [
            "Eye of the Tiger - Survivor", "Don't Stop Me Now - Queen", "Thunderstruck - AC/DC",
            "Stronger - Kanye West", "Shake It Off - Taylor Swift"
        ],
        "relaxed": [
            "Weightless - Marconi Union", "Clair de Lune - Claude Debussy", "Watermark - Enya",
            "Breathe (In The Air) - Pink Floyd", "Gymnopédie No.1 - Erik Satie"
        ],
        "focused": [
            "The Four Seasons - Vivaldi (Antonio Vivaldi)", "Time - Hans Zimmer", "Intro - The xx",
            "Alpha Waves - Brain Power", "Experience - Ludovico Einaudi"
        ],
        "nostalgic": [
            "Yesterday - The Beatles", "Viva la Vida - Coldplay", "Landslide - Fleetwood Mac", # Corrected Vivalada
            "Vienna - Billy Joel", "Time After Time - Cyndi Lauper"
        ]
    }

    chosen_mood_list = mood.lower()
    if chosen_mood_list not in mood_recommendations:
        logger.warning(f"Mood '{mood}' not in generic list, defaulting to happy.")
        chosen_mood_list = "happy" # Fallback mood
        
    recommendations = mood_recommendations.get(chosen_mood_list, mood_recommendations["happy"])
    response_text = f"🎵 Here are some general **{mood.capitalize()}** vibes for you:\n\n"
    for i, track in enumerate(recommendations, 1):
        response_text += f"{i}. {track}\n"
    response_text += "\n💡 <i>You can ask me to search or download any of these, or try `/recommend` again for more tailored suggestions!</i>"
    
    await update.message.reply_text(response_text, parse_mode=ParseMode.HTML)


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
    if current_mood:
        prompt_text += f"Your current mood is set to **{current_mood}**. Want to change it or how are you feeling now?"
    else:
        prompt_text += "How are you feeling today?"
    
    await update.message.reply_text(prompt_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)
    return MOOD # Next state in ConversationHandler

async def send_search_results(update: Update, query: str, results: List[Dict]) -> None:
    """Send YouTube search results with inline keyboard for download."""
    if not results:
        await update.message.reply_text(f"😕 Sorry, I couldn't find any YouTube results for '<i>{query}</i>'. Try different keywords?", parse_mode=ParseMode.HTML)
        return

    keyboard = []
    response_text = f"🔎 Here's what I found on YouTube for '<i>{query}</i>':\n\n"
    
    for i, result in enumerate(results[:5]): # Show top 5 results
        if not result.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', result['id']):
            logger.warning(f"Skipping invalid YouTube result ID: {result.get('id', 'No ID')}")
            continue

        duration_str = ""
        if result.get('duration'):
            try:
                minutes = int(result['duration'] // 60)
                seconds = int(result['duration'] % 60)
                duration_str = f" [{minutes}:{seconds:02d}]"
            except TypeError: # If duration is not a number
                duration_str = ""
        
        title = result['title']
        # Truncate button text if too long for Telegram
        button_display_title = (title[:35] + "...") if len(title) > 38 else title
        button_text = f"[{i+1}] {button_display_title}{duration_str}"
        
        # Add to response text for clarity
        response_text += f"{i+1}. <b>{title}</b> by <i>{result.get('uploader', 'N/A')}</i>{duration_str}\n"
        
        keyboard.append([InlineKeyboardButton(button_text, callback_data=f"{CB_DOWNLOAD_PREFIX}{result['id']}")])

    if not keyboard: # If all results had invalid IDs
        await update.message.reply_text(f"😕 I found some YouTube results for '<i>{query}</i>', but couldn't create download options for them. Sorry!", parse_mode=ParseMode.HTML)
        return

    keyboard.append([InlineKeyboardButton("Cancel Search", callback_data=CB_CANCEL_SEARCH)])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    response_text += "\nClick a song to download its audio:"
    await update.message.reply_text(response_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML)

async def auto_download_first_result(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str, original_message_id: Optional[int] = None) -> None:
    """Search YouTube, then automatically download the first valid song result."""
    user_id = update.effective_user.id

    if user_id in active_downloads:
        await update.message.reply_text("Hold on! You already have a download in progress. Let that finish first. 😊")
        return

    active_downloads.add(user_id)
    reply_func = context.bot.edit_message_text if original_message_id else update.message.reply_text
    chat_id_to_use = update.effective_chat.id

    status_msg = None
    if original_message_id: # Editing an existing message (e.g., from button click)
        status_msg = await reply_func(chat_id=chat_id_to_use, message_id=original_message_id, text=f"🔍 Okay, searching for '<i>{query}</i>' to download...", parse_mode=ParseMode.HTML)
    else: # Replying to a new command
        status_msg = await reply_func(f"🔍 Okay, searching for '<i>{query}</i>' to download...", parse_mode=ParseMode.HTML)

    try:
        results = await asyncio.to_thread(search_youtube_sync, query, max_results=1) # Get only the top result
        if not results or not results[0].get('id') or not is_valid_youtube_url(results[0].get('url', '')):
            await status_msg.edit_text(f"❌ Oops! I couldn't find a downloadable track for '<i>{query}</i>'. Maybe try a more specific search?", parse_mode=ParseMode.HTML)
            return

        top_result = results[0]
        video_url = top_result["url"]
        video_title = top_result.get("title", "this track")
        
        await status_msg.edit_text(f"✅ Found: <b>{video_title}</b>.\n⏳ Now downloading... this might take a moment!", parse_mode=ParseMode.HTML)

        download_result = await asyncio.to_thread(download_youtube_audio_sync, video_url)
        
        if not download_result["success"]:
            error_message = download_result.get('error', 'Unknown download error.')
            await status_msg.edit_text(f"❌ Download failed for <b>{video_title}</b>: {error_message}", parse_mode=ParseMode.HTML)
            return

        await status_msg.edit_text(f"✅ Downloaded: <b>{download_result['title']}</b>.\n⏳ Sending the audio file...", parse_mode=ParseMode.HTML)
        
        audio_path = download_result["audio_path"]
        with open(audio_path, 'rb') as audio_file:
            logger.info(f"Auto-downloading and sending '{download_result['title']}' to user {user_id}. Path: {audio_path}")
            # Send as a new message, not edit
            await context.bot.send_audio(
                chat_id=chat_id_to_use,
                audio=audio_file,
                title=download_result["title"][:64],
                performer=download_result["artist"][:64] if download_result.get("artist") else "Unknown Artist",
                caption=f"🎵 Here's your requested track: {download_result['title']}",
                duration=download_result.get('duration')
            )

        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"Successfully deleted temp file after auto-download: {audio_path}")
            except OSError as e:
                logger.error(f"Error deleting temp file after auto-download {audio_path}: {e}")
        
        await status_msg.delete() # Clean up the status message
    
    except (TimedOut, NetworkError) as net_err:
        logger.error(f"Telegram API or Network error during auto-download for user {user_id}, query '{query}': {net_err}")
        if status_msg: # Try to update status if possible
           try:
               await status_msg.edit_text(f"❌ A network problem occurred. Please try again for '<i>{query}</i>'.", parse_mode=ParseMode.HTML)
           except: pass
    except Exception as e:
        logger.error(f"Unexpected error in auto_download_first_result for user {user_id}, query '{query}': {e}", exc_info=True)
        if status_msg:
            try:
                await status_msg.edit_text(f"❌ An unexpected error stopped the download of '<i>{query}</i>'. My apologies!", parse_mode=ParseMode.HTML)
            except: pass
    finally:
        active_downloads.discard(user_id)

# Refactored send_audio_with_retry from original (was complex, now relies on PTB defaults)
async def send_audio_via_bot(bot, chat_id, audio_path, title, performer, caption, duration):
    """Helper to send audio, assuming PTB's internal retries and timeouts are configured."""
    with open(audio_path, 'rb') as audio_file:
        await bot.send_audio(
            chat_id=chat_id,
            audio=audio_file,
            title=title[:64],
            performer=performer[:64] if performer else "Unknown Artist",
            caption=caption,
            duration=duration
        )

async def enhanced_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
    """Handle button callbacks from inline keyboards."""
    query = update.callback_query
    await query.answer() # Acknowledge the button press
    
    data = query.data
    user_id = query.from_user.id
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    
    logger.debug(f"Handling callback query: '{data}' for user {user_id}")

    # Mood setting
    if data.startswith(CB_MOOD_PREFIX):
        mood = data[len(CB_MOOD_PREFIX):]
        user_contexts[user_id]["mood"] = mood
        logger.info(f"User {user_id} set mood to: {mood}")

        keyboard = [ # Genre preference keyboard
            [InlineKeyboardButton("Pop", callback_data=f"{CB_PREFERENCE_PREFIX}pop"),
             InlineKeyboardButton("Rock", callback_data=f"{CB_PREFERENCE_PREFIX}rock"),
             InlineKeyboardButton("Hip-Hop/Rap", callback_data=f"{CB_PREFERENCE_PREFIX}hiphop")],
            [InlineKeyboardButton("Electronic/Dance", callback_data=f"{CB_PREFERENCE_PREFIX}electronic"),
             InlineKeyboardButton("Classical", callback_data=f"{CB_PREFERENCE_PREFIX}classical"),
             InlineKeyboardButton("Jazz/Blues", callback_data=f"{CB_PREFERENCE_PREFIX}jazz")],
            [InlineKeyboardButton("Folk/Acoustic", callback_data=f"{CB_PREFERENCE_PREFIX}folk"),
             InlineKeyboardButton("R&B/Soul", callback_data=f"{CB_PREFERENCE_PREFIX}rnb"),
             InlineKeyboardButton("Any / Surprise Me!", callback_data=f"{CB_PREFERENCE_PREFIX}any")],
            [InlineKeyboardButton("Skip Genre", callback_data=f"{CB_PREFERENCE_PREFIX}skip")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            f"Got it, you're feeling {mood}! 🎶\n\nTo help me recommend music, do you have any genre preferences right now?",
            reply_markup=reply_markup
        )
        return PREFERENCE # Next state in mood conversation

    # Preference setting
    elif data.startswith(CB_PREFERENCE_PREFIX):
        preference = data[len(CB_PREFERENCE_PREFIX):]
        if preference == "skip" or preference == "any":
            user_contexts[user_id]["preferences"] = [] # Clear or set to "any"
            message_text = "Alright! I'll keep that in mind. "
        else:
            # Allow multiple preferences in future? For now, one.
            user_contexts[user_id]["preferences"] = [preference] 
            message_text = f"Great choice! {preference.capitalize()} it is. "
        logger.info(f"User {user_id} set preference to: {preference}")
        
        message_text += "Feel free to:\n" \
                        "➡️ `/recommend` for music suggestions\n" \
                        "➡️ `/search [song]` or `/autodownload [song]`\n" \
                        "➡️ Just chat with me about music or anything else!"
        await query.edit_message_text(message_text)
        return ConversationHandler.END # End mood conversation

    # Download from search result / recommendation
    elif data.startswith(CB_DOWNLOAD_PREFIX) or data.startswith(CB_AUTO_DOWNLOAD_PREFIX):
        is_auto = data.startswith(CB_AUTO_DOWNLOAD_PREFIX)
        video_id = data.split("_")[-1] # Assumes ID is last part, e.g. download_VIDEOID or auto_download_VIDEOID

        if not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
            logger.error(f"Invalid YouTube video ID from callback: '{video_id}'")
            await query.edit_message_text("❌ Oops! That video ID seems invalid. Please try searching again.")
            return None

        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        
        if user_id in active_downloads:
            await query.edit_message_text("⚠️ You have another download active. Please wait for it to complete!")
            return None

        active_downloads.add(user_id)
        # Edit original message that had the button
        await query.edit_message_text(f"⏳ Preparing to download audio from YouTube...\nThis might take a moment.", reply_markup=None) 
        
        try:
            result = await asyncio.to_thread(download_youtube_audio_sync, youtube_url)
            if not result["success"]:
                error_msg = result.get('error', 'Unknown download error.')
                # Inform via a new message since edit_message_text can be tricky after initial edits
                await context.bot.send_message(chat_id=query.message.chat_id, text=f"❌ Download failed for the selected video: {error_msg}")
                return None # No further action from button handler

            # Sending as new message to avoid editing issues if download takes long
            await context.bot.send_message(chat_id=query.message.chat_id, text=f"✅ Downloaded: <b>{result['title']}</b>\n⏳ Sending you the audio file now...", parse_mode=ParseMode.HTML)
            
            await send_audio_via_bot(
                context.bot, 
                query.message.chat_id, 
                result["audio_path"], 
                result["title"], 
                result.get("artist"), 
                f"🎵 Here's your downloaded track: {result['title']}",
                result.get("duration")
            )
            logger.info(f"Sent audio '{result['title']}' to user {user_id} via button callback.")

            if os.path.exists(result["audio_path"]):
                try:
                    os.remove(result["audio_path"])
                    logger.info(f"Deleted temp file after button download: {result['audio_path']}")
                except OSError as e:
                    logger.error(f"Error deleting temp file {result['audio_path']}: {e}")
            # Try to delete the "Preparing to download..." message if it's still there.
            # This is tricky as query.message might be stale.
            # For simplicity, we don't try to delete the "Preparing" message if successful send happened.

        except (TimedOut, NetworkError) as net_err:
            logger.error(f"Telegram API/Network error during button download for user {user_id}, video_id {video_id}: {net_err}")
            await context.bot.send_message(chat_id=query.message.chat_id, text="❌ A network issue occurred while sending the file. Please try downloading again.")
        except Exception as e:
            logger.error(f"Error in button download handler for user {user_id}, video_id {video_id}: {e}", exc_info=True)
            await context.bot.send_message(chat_id=query.message.chat_id, text="❌ An unexpected error occurred with that download. Sorry about that!")
        finally:
            active_downloads.discard(user_id)
        return None # End callback processing for downloads

    # Show options (re-search based on original query string)
    elif data.startswith(CB_SHOW_OPTIONS_PREFIX):
        search_query_str = data[len(CB_SHOW_OPTIONS_PREFIX):]
        if not search_query_str:
            await query.edit_message_text("Cannot show options, original query missing.")
            return None
        
        await query.edit_message_text(f"🔍 Okay, showing more YouTube options for '<i>{search_query_str}</i>'...", parse_mode=ParseMode.HTML, reply_markup=None)
        results = await asyncio.to_thread(search_youtube_sync, search_query_str, max_results=5)
        # This effectively replaces the original message with search results
        # To do this cleanly, we delete the old and send new. query.message.delete() and then send_search_results.
        # Simpler: treat this as a new search from user's perspective.
        try:
             await query.message.delete() # Delete the "I found X, download or show options?" msg
        except Exception as e:
            logger.warning(f"Could not delete previous message before showing options: {e}")

        # Send new search results message (pass `query.message` from the callback as `update` for send_search_results)
        # Need to construct a compatible Update object or modify send_search_results signature
        # Simpler for now: just reply in chat.
        fake_update_for_search = Update(update_id=query.update_id, message=query.message) # Use query.message as base
        await send_search_results(fake_update_for_search, search_query_str, results)
        return None

    # Cancel search/Spotify
    elif data == CB_CANCEL_SEARCH:
        await query.edit_message_text("❌ Search cancelled. Feel free to try another search or chat!")
        return None
    elif data == CB_CANCEL_SPOTIFY: # For Spotify ConvHandler button
        await query.edit_message_text("Spotify linking cancelled. Use /link_spotify anytime to try again.")
        return ConversationHandler.END # Ensure it ends the conv

    logger.warning(f"Unhandled callback data: {data} for user {user_id}")
    return None # Default case if no match


async def enhanced_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enhanced message handler with music detection, URL handling, and AI chat."""
    if not update.message or not update.message.text: # Ignore empty messages or non-text
        return

    user_id = update.effective_user.id
    text = update.message.text.strip()
    logger.debug(f"Processing text message from user {user_id}: '{text[:100]}'") # Log truncated text

    # Initialize user context if it's their first message handled here
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})

    # 1. Handle YouTube URLs directly for download
    if is_valid_youtube_url(text):
        logger.info(f"User {user_id} sent a YouTube URL directly: {text}")
        # Pass to download_music, it expects context.args or parsable text for URL
        # To make it simpler, create a fake context.args
        mock_context_args = text.split() # Simplistic; assumes URL is the only thing or first.
        await download_music(update, ContextTypes.DEFAULT_TYPE(application=context.application, chat_id=user_id, user_id=user_id, bot=context.bot, args=mock_context_args))
        return

    # 2. AI-based mood detection (subtle, doesn't announce)
    # This is done somewhat passively now, better to integrate into conversation analysis
    # For now, a quick mood update can be useful
    detected_mood_on_message = await detect_mood_from_text(user_id, text)
    if detected_mood_on_message and detected_mood_on_message != "neutral": # Don't overwrite good moods with neutral
        user_contexts[user_id]["mood"] = detected_mood_on_message
        logger.debug(f"Passive mood update for user {user_id} to: {detected_mood_on_message}")

    # 3. Music request detection (using simpler regex first, then AI if needed)
    # Simplified regex logic first from original `detect_music_in_message`
    simple_music_keywords = ['play', 'find song', 'download', 'get song', 'listen to', 'search for song']
    music_query = None
    for keyword in simple_music_keywords:
        if text.lower().startswith(keyword):
            music_query = text[len(keyword):].strip()
            break
    
    if not music_query and any(kw in text.lower() for kw in ["song", "music", "track", "artist", "album"]): # General music mention
        # Use AI for more nuanced detection if simple keyword not found but music terms exist
        ai_music_check = await is_music_request(user_id, text)
        if ai_music_check.get("is_music_request") and ai_music_check.get("song_query"):
            music_query = ai_music_check["song_query"]
            logger.info(f"AI detected music request: '{music_query}' for user {user_id}")

    if music_query:
        status_msg = await update.message.reply_text(f"🎵 Got it! You're looking for '<i>{music_query}</i>'. One moment...", parse_mode=ParseMode.HTML)
        results = await asyncio.to_thread(search_youtube_sync, music_query, max_results=1)
        
        if results and results[0].get('id') and re.match(r'^[0-9A-Za-z_-]{11}$', results[0]['id']):
            top_result = results[0]
            keyboard = [
                [InlineKeyboardButton(f"✅ Yes, download '{top_result['title'][:20]}...'", callback_data=f"{CB_AUTO_DOWNLOAD_PREFIX}{top_result['id']}")],
                [InlineKeyboardButton("👀 Show me more options", callback_data=f"{CB_SHOW_OPTIONS_PREFIX}{music_query}")], # Pass original query
                [InlineKeyboardButton("❌ No, that's not it", callback_data=CB_CANCEL_SEARCH)]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            uploader = top_result.get('uploader', 'Unknown Artist')
            await status_msg.edit_text(
                f"I found: <b>{top_result['title']}</b> by <i>{uploader}</i>.\n\n"
                f"Would you like me to download this, or show more options?",
                reply_markup=reply_markup, parse_mode=ParseMode.HTML
            )
        else:
            await status_msg.edit_text(f"😕 Sorry, I couldn't find a specific track for '<i>{music_query}</i>' on YouTube. You can try being more specific or use `/search` command.", parse_mode=ParseMode.HTML)
        return

    # 4. Lyrics request detection (simple heuristic)
    lyrics_keywords = ["lyrics for", "words to", "what are the lyrics to"]
    lyrics_query = None
    for keyword in lyrics_keywords:
        if keyword in text.lower():
            # Attempt to extract song title after keyword
            # This is very basic, /lyrics command is more reliable
            # Example: "what are the lyrics to bohemian rhapsody" -> "bohemian rhapsody"
            parts = text.lower().split(keyword, 1)
            if len(parts) > 1 and parts[1].strip():
                lyrics_query = parts[1].strip()
                logger.info(f"Heuristic lyrics request: '{lyrics_query}' for user {user_id}")
                break
    if lyrics_query:
        # Fake context.args for get_lyrics_command
        mock_context_args_lyrics = lyrics_query.split() # This is crude
        await get_lyrics_command(update, ContextTypes.DEFAULT_TYPE(application=context.application, chat_id=user_id, user_id=user_id, bot=context.bot, args=mock_context_args_lyrics))
        return

    # 5. Fallback to general AI chat response
    typing_msg = await update.message.reply_text("<i>MelodyMind is thinking...</i> 🎶", parse_mode=ParseMode.HTML)
    try:
        response_text = await generate_chat_response(user_id, text)
        await typing_msg.edit_text(response_text)
    except (TimedOut, NetworkError) as net_err:
        logger.error(f"Network error during AI chat response generation for user {user_id}: {net_err}")
        await typing_msg.edit_text("Sorry, I'm having a bit of trouble connecting. Could you try saying that again in a moment?")
    except Exception as e:
        logger.error(f"Error generating AI chat response for user {user_id}: {e}", exc_info=True)
        await typing_msg.edit_text("I seem to be a bit tangled up at the moment! 😅 Let's try that conversation again later.")


def detect_music_in_message(text: str) -> Optional[str]: # Kept for reference or potential future use
    """Detect if a message is asking for music using regex. (Simpler version now in enhanced_handle_message)"""
    patterns = [
        r'play (.*?)(?:by|from|$)', r'find song (.*?)(?:by|from|$)',
        r'download (.*?)(?:by|from|$)', r'get (.*?)(?:by|from|$)',
        r'send me (.*?)(?:by|from|$)', r'i want to listen to (.*?)(?:by|from|$)',
        r'can you get (.*?)(?:by|from|$)', r'i need the song (.*?)(?:by|from|$)',
        r'find me the track (.*?)(?:by|from|$)', r'fetch (.*?)(?:by|from|$)',
        r'give me (.*?)(?:by|from|$)', r'search for song (.*?)(?:by|from|$)'
    ] # Removed generic "song", "send" which are too broad

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            song_title_match = match.group(1).strip()
            # Try to capture artist if "by artist_name" follows
            artist_pattern = rf'{re.escape(song_title_match)}\s+by\s+(.*?)(?:from|$)'
            artist_match_obj = re.search(artist_pattern, text, re.IGNORECASE)
            if artist_match_obj:
                artist_name = artist_match_obj.group(1).strip()
                logger.debug(f"Regex music detection: '{song_title_match} by {artist_name}'")
                return f"{song_title_match} {artist_name}" # Combine for search query
            logger.debug(f"Regex music detection: '{song_title_match}'")
            return song_title_match

    # If no specific pattern matches, but contains music-related keywords, flag for AI.
    # This part is now handled by is_music_request AI call more directly.
    # keywords_for_ai_check = ['music', 'song', 'track', 'tune', 'audio', 'artist', 'album']
    # if any(keyword in text.lower() for keyword in keywords_for_ai_check):
    #    logger.debug(f"Regex: General music terms found, suggest AI check.")
    #    return "AI_ANALYSIS_NEEDED" # Signal that AI should make the call
    return None

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear user's conversation history with the bot."""
    user_id = update.effective_user.id
    if user_id in user_contexts and "conversation_history" in user_contexts[user_id]:
        user_contexts[user_id]["conversation_history"] = []
        logger.info(f"Cleared conversation history for user {user_id}")
        await update.message.reply_text("✅ Your conversation history with me has been cleared.")
    else:
        await update.message.reply_text("You don't have any conversation history with me to clear yet! 😊")

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Generic cancel handler for ConversationHandlers."""
    await update.message.reply_text("Okay, cancelled whatever we were doing. Feel free to use commands or chat anytime! 👍")
    return ConversationHandler.END

async def analyze_conversation(user_id: int) -> Dict:
    """Analyze conversation history and Spotify data using AI for preferences."""
    if not client: # AI not available
        return {"genres": user_contexts.get(user_id, {}).get("preferences", []), 
                "artists": [], 
                "mood": user_contexts.get(user_id, {}).get("mood")}

    user_ctx = user_contexts.get(user_id, {})
    # Ensure default structures to avoid KeyErrors
    user_ctx.setdefault("preferences", [])
    user_ctx.setdefault("conversation_history", [])
    user_ctx.setdefault("spotify", {})
    user_ctx["spotify"].setdefault("recently_played", [])
    user_ctx["spotify"].setdefault("top_tracks", [])

    # Basic check: if very little data, don't call AI.
    if len(user_ctx.get("conversation_history", [])) < 3 and \
       not user_ctx.get("spotify", {}).get("recently_played") and \
       not user_ctx.get("spotify", {}).get("top_tracks"):
        logger.info(f"Insufficient data for AI conversation analysis for user {user_id}.")
        return {"genres": user_ctx.get("preferences", []), "artists": [], "mood": user_ctx.get("mood")}

    logger.info(f"Performing AI conversation analysis for user {user_id}")
    try:
        conversation_text_summary = ""
        if user_ctx["conversation_history"]:
            # Summarize last 10 messages (5 user, 5 bot typically)
            history_summary = [f"{msg['role']}: {msg['content'][:100]}" for msg in user_ctx["conversation_history"][-10:]] # Truncate long messages
            conversation_text_summary = "\n".join(history_summary)

        spotify_summary = ""
        # Recently Played (more current indicator)
        if user_ctx["spotify"]["recently_played"]:
            try:
                tracks = user_ctx["spotify"]["recently_played"]
                spotify_summary += "Recently played tracks: " + ", ".join(
                    [f"'{item['track']['name']}' by {item['track']['artists'][0]['name']}" for item in tracks[:3] if item.get("track")] # Top 3
                ) + ". "
            except: pass # Gracefully handle if structure is not as expected
        # Top Tracks (longer term preference)
        if user_ctx["spotify"]["top_tracks"]:
            try:
                tracks = user_ctx["spotify"]["top_tracks"]
                spotify_summary += "User's top tracks: " + ", ".join(
                    [f"'{item['name']}' by {item['artists'][0]['name']}" for item in tracks[:3] if item.get("artists")] # Top 3
                ) + "."
            except: pass
        
        if not conversation_text_summary and not spotify_summary: # Still not enough info
             logger.info(f"Not enough text/spotify summary for AI analysis for user {user_id}")
             return {"genres": user_ctx.get("preferences", []), "artists": [], "mood": user_ctx.get("mood")}


        prompt_messages = [
            {"role": "system", "content": 
                "You are an AI analyzing a user's conversation with a music bot and their Spotify listening data. "
                "Your goal is to infer their musical preferences (genres, artists) and current/recent mood. "
                "Respond in JSON format with three keys: 'genres' (list of strings, up to 3 most prominent), 'artists' (list of strings, up to 3 relevant artists), and 'mood' (single string, or null if not clear). "
                "Prioritize information from recent messages and explicit statements. Use Spotify data to confirm or refine. If no strong signals, return empty lists or null mood."
            },
            {"role": "user", "content": 
                f"Conversation Summary:\n{conversation_text_summary}\n\nSpotify Data Summary:\n{spotify_summary}\n\nUser's explicitly set mood (if any): {user_ctx.get('mood')}\nUser's explicitly set preferences (if any): {', '.join(user_ctx.get('preferences',[]))}"
            }
        ]

        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo-0125", # JSON mode model
            messages=prompt_messages,
            max_tokens=200,
            temperature=0.2,
            response_format={"type": "json_object"}
        )

        result_str = response.choices[0].message.content
        logger.debug(f"AI analysis raw response for user {user_id}: {result_str}")
        result = json.loads(result_str)

        if not isinstance(result, dict):
            logger.error(f"AI analysis for user {user_id} returned non-dict: {result}")
            return {"genres": user_ctx.get("preferences", []), "artists": [], "mood": user_ctx.get("mood")}

        # Process and validate results
        inferred_genres = result.get("genres", [])
        if isinstance(inferred_genres, str): inferred_genres = [g.strip() for g in inferred_genres.split(",") if g.strip()]
        if not isinstance(inferred_genres, list): inferred_genres = []
        
        inferred_artists = result.get("artists", [])
        if isinstance(inferred_artists, str): inferred_artists = [a.strip() for a in inferred_artists.split(",") if a.strip()]
        if not isinstance(inferred_artists, list): inferred_artists = []

        inferred_mood = result.get("mood")
        if not isinstance(inferred_mood, str) or not inferred_mood.strip(): inferred_mood = None
        
        # Update user_contexts with AI inferred data if it's more specific
        # Prioritize existing explicit settings unless AI has strong new signals.
        if inferred_genres and not user_ctx.get("preferences"): # If user hasn't set any, use AI's
            user_ctx["preferences"] = list(set(inferred_genres[:3])) # Limit to 3 unique
        
        # Mood is more transient, update if AI has one and it differs or user has no mood set
        if inferred_mood and (inferred_mood != user_ctx.get("mood") or not user_ctx.get("mood")):
            user_ctx["mood"] = inferred_mood

        logger.info(f"AI analysis for user {user_id} results: Genres={user_ctx['preferences']}, Mood={user_ctx['mood']}, Artists mentioned={inferred_artists}")
        return {
            "genres": user_ctx["preferences"], # Return updated context
            "artists": inferred_artists[:3], # Return AI artists (can be used for seed)
            "mood": user_ctx["mood"]
        }

    except json.JSONDecodeError as jde:
        logger.error(f"AI analysis JSON decode error for user {user_id}: {jde}. Raw: {response.choices[0].message.content if 'response' in locals() else 'N/A'}")
    except Exception as e:
        logger.error(f"Error in AI analyze_conversation for user {user_id}: {e}", exc_info=True)
    
    # Fallback to existing context if AI analysis fails
    return {"genres": user_ctx.get("preferences", []), "artists": [], "mood": user_ctx.get("mood")}

async def smart_recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provide smarter music recommendations using conversation analysis and Spotify data."""
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name
    status_msg = await update.message.reply_text(f"🎵 Thinking of some great music for you, {user_name}...")

    try:
        # Refresh Spotify data if linked (recently played, top tracks)
        # This updates user_contexts[user_id]["spotify"]
        if user_contexts.get(user_id, {}).get("spotify", {}).get("access_token"): # Check if Spotify is linked
            logger.info(f"Fetching latest Spotify data for user {user_id} for smart recommendations.")
            # Run these in parallel to speed up
            rp_task = get_user_spotify_data(user_id, "player/recently-played", params={"limit": 10})
            tt_task = get_user_spotify_data(user_id, "top/tracks", params={"limit": 10, "time_range": "short_term"}) # Short term top tracks
            
            recently_played, top_tracks = await asyncio.gather(rp_task, tt_task)

            if recently_played:
                user_contexts[user_id]["spotify"]["recently_played"] = recently_played
            if top_tracks:
                user_contexts[user_id]["spotify"]["top_tracks"] = top_tracks
        
        # Analyze conversation and existing context (mood, prefs, Spotify data)
        analysis = await analyze_conversation(user_id)
        
        current_mood = analysis.get("mood")
        # If mood is still not determined, ask the user explicitly
        if not current_mood or current_mood == "neutral":
            await status_msg.delete()
            # Use the set_mood ConversationHandler entry point
            # We need to ensure `set_mood` is designed to be called this way (it is, via CommandHandler)
            # Here we call its logic directly or trigger the conv handler if possible.
            # For simplicity, redirect to /mood command or its logic:
            logger.info(f"Mood not determined for user {user_id}, prompting for mood for recommendations.")
            # Forward to set_mood state via its command for now (simplest integration)
            # This effectively makes /recommend ask for mood if not set.
            await set_mood(update, context) 
            return

        await status_msg.edit_text(f"Okay {user_name}, based on your mood of **{current_mood}** (and other vibes!)... \nLooking for recommendations... 🎧", parse_mode=ParseMode.MARKDOWN)

        # Prepare for Spotify API or YouTube search
        seed_track_ids = []
        seed_artist_ids = [] # Spotify API can also use seed_artists
        seed_genres = analysis.get("genres", []) # And seed_genres

        # Use Spotify recently played/top tracks for seeds if available
        spotify_ctx = user_contexts.get(user_id, {}).get("spotify", {})
        if spotify_ctx.get("access_token"): # If Spotify linked
            # Priority to explicitly mentioned artists from analysis if they exist in user's Spotify history.
            # For now, simple: use recent/top tracks as primary seeds.
            if spotify_ctx.get("recently_played"):
                seed_track_ids.extend([
                    track["track"]["id"] for track in spotify_ctx["recently_played"][:2] 
                    if track.get("track") and track["track"].get("id")
                ])
            if not seed_track_ids and spotify_ctx.get("top_tracks"): # If no recent, try top
                 seed_track_ids.extend([
                    track["id"] for track in spotify_ctx["top_tracks"][:2] 
                    if track.get("id")
                ])
        
        spotify_client_token = await get_spotify_token() # For client-credentials Spotify calls

        # Try Spotify recommendations if we have seeds and client token
        if seed_track_ids and spotify_client_token:
            logger.info(f"Attempting Spotify API recommendations for user {user_id} with seeds: tracks={seed_track_ids}, genres={seed_genres}")
            # TODO: Potentially add artist seeds too if available from 'analysis.get("artists")' by searching their IDs.
            # Make sure seed_tracks are valid (some user tracks might not be usable as seeds, rare)
            # For now, assume they are generally valid. Max 5 seeds total.
            
            recommendations = await get_spotify_recommendations(spotify_client_token, seed_track_ids[:2], limit=5) # Use up to 2 track seeds
            
            if recommendations:
                response_html = f"🎵 Tailored Spotify recommendations for your **{current_mood}** mood, {user_name}:\n\n"
                keyboard_spotify = []
                for i, track in enumerate(recommendations, 1):
                    artists_text = ", ".join(a["name"] for a in track["artists"])
                    album = track.get("album", {}).get("name", "")
                    track_info_text = f"<b>{track['name']}</b> by <i>{artists_text}</i>"
                    if album: track_info_text += f" (from {album})"
                    response_html += f"{i}. {track_info_text}\n"
                    # Create a search query for YouTube based on this Spotify track for download button
                    yt_search_query = f"{track['name']} {artists_text}"
                    keyboard_spotify.append([InlineKeyboardButton(f" YT Search: {track['name'][:20]}...", callback_data=f"{CB_SHOW_OPTIONS_PREFIX}{yt_search_query}")])

                response_html += "\n💡 <i>Click to search on YouTube, or ask me to play something else!</i>"
                await status_msg.edit_text(response_html, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard_spotify))
                return

        # Fallback: YouTube search based on mood, genres, and AI-identified artists
        yt_query_parts = [current_mood]
        if seed_genres: yt_query_parts.append(seed_genres[0]) # Add primary genre
        ai_artists = analysis.get("artists", [])
        if ai_artists: yt_query_parts.append(f"music like {ai_artists[0]}") # Add an artist likeness
        
        youtube_search_query = " ".join(yt_query_parts) + " music"
        logger.info(f"Falling back to YouTube search for recommendations for user {user_id} with query: '{youtube_search_query}'")
        await status_msg.edit_text(f"Searching YouTube for some **{current_mood}** tracks like '<i>{youtube_search_query}</i>'...", parse_mode=ParseMode.HTML)

        yt_results = await asyncio.to_thread(search_youtube_sync, youtube_search_query, max_results=5)
        if yt_results:
            response_html_yt = f"🎵 Some YouTube suggestions for your **{current_mood}** mood, {user_name}:\n\n"
            keyboard_yt = []
            for i, result in enumerate(yt_results, 1):
                if not result.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', result['id']): continue
                duration_obj = result.get('duration', 0)
                duration_str = ""
                if duration_obj: 
                    try:
                        mins, secs = divmod(int(duration_obj), 60)
                        duration_str = f" [{mins}:{secs:02d}]"
                    except: pass # Ignore duration formatting errors
                
                response_html_yt += f"{i}. <b>{result['title']}</b> - <i>{result.get('uploader', 'N/A')}</i>{duration_str}\n"
                button_title = result['title'][:30] + "..." if len(result['title']) > 30 else result['title']
                keyboard_yt.append([InlineKeyboardButton(f"Download: {button_title}", callback_data=f"{CB_DOWNLOAD_PREFIX}{result['id']}")])
            
            if not keyboard_yt: # No valid results after filtering
                 await status_msg.delete()
                 await provide_generic_recommendations(update, current_mood)
                 return

            response_html_yt += "\n💡 <i>Click to download audio directly!</i>"
            await status_msg.edit_text(response_html_yt, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard_yt))
        else: # Final fallback: generic hardcoded list
            logger.info(f"No YouTube results for '{youtube_search_query}', providing generic recommendations for {current_mood}.")
            await status_msg.delete() # Clear "Searching..."
            await provide_generic_recommendations(update, current_mood)

    except Exception as e:
        logger.error(f"Error in smart_recommend_music for user {user_id}: {e}", exc_info=True)
        await status_msg.edit_text(f"Oh no, {user_name}! I ran into a snag trying to find recommendations. 😥 Please try again in a bit.")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    
    # Optionally, send a message to the user if it's an update-related error
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "😓 Oops! Something went wrong on my end. My developers are looking into it. Please try again later."
            )
        except Exception as e:
            logger.error(f"Failed to send error message to user: {e}")

def cleanup_downloads_atexit() -> None:
    """Clean up temporary audio files from DOWNLOAD_DIR on exit."""
    logger.info("Cleaning up temporary download files...")
    cleaned_count = 0
    try:
        if os.path.exists(DOWNLOAD_DIR):
            for item_name in os.listdir(DOWNLOAD_DIR):
                item_path = os.path.join(DOWNLOAD_DIR, item_name)
                try:
                    if os.path.isfile(item_path): # or os.path.islink(item_path)
                        os.remove(item_path)
                        cleaned_count +=1
                    # Optionally, clean empty subdirs if any were created:
                    # elif os.path.isdir(item_path): shutil.rmtree(item_path)
                except Exception as e:
                    logger.error(f"Failed to remove {item_path}: {e}")
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} file(s) from '{DOWNLOAD_DIR}'.")
            else:
                logger.info(f"No files to clean in '{DOWNLOAD_DIR}'.")
        else:
            logger.info(f"Download directory '{DOWNLOAD_DIR}' not found, no cleanup needed.")
    except Exception as e:
        logger.error(f"Error during atexit cleanup of downloads directory: {e}")

def signal_exit_handler(sig, frame) -> None:
    """Handle termination signals gracefully."""
    logger.info(f"Received signal {sig}, preparing to exit...")
    # cleanup_downloads_atexit() is registered with atexit, so it will run.
    # Additional cleanup specific to signal can go here if needed.
    sys.exit(0)

def main() -> None:
    """Start the bot."""
    # PTB application builder with timeouts and rate limiter
    application = (
        Application.builder()
        .token(TOKEN)
        .connect_timeout(10.0)  # Seconds
        .read_timeout(20.0)     # Seconds
        .write_timeout(30.0)    # Seconds
        .pool_timeout(60.0)     # Seconds (for file uploads/downloads by bot)
        .rate_limiter(AIORateLimiter()) # Basic rate limiting
        .build()
    )

    # === Command Handlers ===
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("download", download_music)) # Also handles direct URL messages via logic in enhanced_handle_message
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("autodownload", auto_download_command))
    application.add_handler(CommandHandler("lyrics", get_lyrics_command))
    application.add_handler(CommandHandler("recommend", smart_recommend_music)) # smart_recommend is now default
    application.add_handler(CommandHandler("create_playlist", create_playlist))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(CommandHandler("spotify_code", spotify_code_command)) # Can be called outside conv

    # === Conversation Handlers ===
    # Spotify Linking Conversation
    spotify_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("link_spotify", link_spotify)],
        states={
            SPOTIFY_CODE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, spotify_code_handler), # Pasted code
                CommandHandler("spotify_code", spotify_code_handler), # Via /spotify_code cmd
                CallbackQueryHandler(cancel_spotify, pattern=f"^{CB_CANCEL_SPOTIFY}$") # Cancel button
            ]
        },
        fallbacks=[CommandHandler("cancel", cancel)], # Generic /cancel command
        # Potentially add conversation timeout
        # conversation_timeout=timedelta(minutes=5).total_seconds() 
    )
    application.add_handler(spotify_conv_handler)

    # Mood Setting Conversation
    mood_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("mood", set_mood)],
        states={
            MOOD: [CallbackQueryHandler(enhanced_button_handler, pattern=f"^{CB_MOOD_PREFIX}")],
            PREFERENCE: [CallbackQueryHandler(enhanced_button_handler, pattern=f"^{CB_PREFERENCE_PREFIX}")],
            # ACTION state was defined but not used; removed for clarity unless needed.
        },
        fallbacks=[CommandHandler("cancel", cancel)],
        # conversation_timeout=timedelta(minutes=3).total_seconds()
    )
    application.add_handler(mood_conv_handler)

    # === Message and Callback Handlers ===
    application.add_handler(MessageHandler(filters.VOICE & ~filters.COMMAND, handle_voice))
    # Generic CallbackQueryHandler MUST be after ConversationHandler ones with specific patterns
    # to avoid intercepting their callbacks.
    application.add_handler(CallbackQueryHandler(enhanced_button_handler)) 
    # Text message handler (ensure it's last for text messages to not override commands)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, enhanced_handle_message))
    
    # Error Handler
    application.add_error_handler(error_handler)

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_exit_handler) # Ctrl+C
    signal.signal(signal.SIGTERM, signal_exit_handler) # Kill/system shutdown

    # Register cleanup for downloads directory on normal exit
    atexit.register(cleanup_downloads_atexit)

    logger.info("🚀 Starting MelodyMind Bot...")
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.critical(f"Bot polling failed to start or crashed: {e}", exc_info=True)
    finally:
        logger.info(" MelodyMind Bot has shut down.")
        # cleanup_downloads_atexit() will run due to atexit registration

if __name__ == "__main__":
    # Pre-run checks for essential env vars
    if not TOKEN:
        logger.critical("TELEGRAM_TOKEN environment variable not set. Bot cannot start.")
        sys.exit(1)
    if not OPENAI_API_KEY:
        logger.warning("OPENAI_API_KEY not set. AI features will be disabled.")
    # Other keys can be checked too if considered critical for startup.
    
    main()