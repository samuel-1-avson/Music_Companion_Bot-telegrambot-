import os
import logging
import requests
import sys
import re
import json
import base64
import pytz
import signal
import atexit
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dotenv import load_dotenv
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential
from telegram.error import TimedOut, NetworkError
import httpx
import asyncio
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
# Telegram imports
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes,
    filters, CallbackQueryHandler, ConversationHandler
)
from functools import lru_cache
# API clients
import yt_dlp
from openai import OpenAI
import importlib
if importlib.util.find_spec("lyricsgenius") is not None:
    import lyricsgenius
else:
    lyricsgenius = None

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "https://your-callback-url.com") # Default for warning

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=15, retries=2, verbose=False) if GENIUS_ACCESS_TOKEN and lyricsgenius else None

# Conversation states
MOOD, PREFERENCE, SPOTIFY_CODE = range(3) # Removed ACTION state

# Track active downloads and user contexts
active_downloads: set[int] = set()
user_contexts: Dict[int, Dict[str, Any]] = {}
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
download_lock = asyncio.Lock() # Lock for active_downloads

# Constants
MAX_FILE_SIZE_TELEGRAM = 50 * 1024 * 1024  # 50 MB
MAX_AUDIO_TITLE_LENGTH = 64
SPOTIFY_RECOMMENDATION_LIMIT = 5
SPOTIFY_SEED_TRACK_LIMIT = 2 # Spotify API allows up to 5 seed items (artists, genres, tracks)
SPOTIFY_USER_DATA_LIMIT = 10
YOUTUBE_SEARCH_MAX_RESULTS = 5


# ==================== SPOTIFY HELPER FUNCTIONS (Synchronous cores) ====================
# These will be wrapped by async functions for use in the async bot

def _get_spotify_token_sync() -> Optional[str]:
    """Get Spotify access token using client credentials (synchronous)."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        logger.warning("Spotify client credentials not configured")
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
        response = requests.post(url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting Spotify client token: {e}")
        return None

def _search_spotify_track_sync(token: str, query: str) -> Optional[Dict]:
    """Search for a track on Spotify (synchronous)."""
    if not token:
        return None

    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "type": "track", "limit": 1}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        items = response.json().get("tracks", {}).get("items", [])
        return items[0] if items else None
    except (requests.exceptions.RequestException, IndexError) as e:
        logger.error(f"Error searching Spotify track '{query}': {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def _get_spotify_recommendations_sync(token: str, seed_tracks: List[str], limit: int = SPOTIFY_RECOMMENDATION_LIMIT) -> List[Dict]:
    """Get track recommendations from Spotify (synchronous)."""
    if not token or not seed_tracks:
        logger.warning("No token or seed tracks provided for Spotify recommendations")
        return []

    url = "https://api.spotify.com/v1/recommendations"
    headers = {"Authorization": f"Bearer {token}"}
    # Ensure we don't exceed Spotify's limit for seed tracks (usually 5 combined seeds)
    params = {"seed_tracks": ",".join(seed_tracks[:SPOTIFY_SEED_TRACK_LIMIT]), "limit": limit}


    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        return response.json().get("tracks", [])
    except requests.exceptions.HTTPError as http_error:
        logger.warning(f"Spotify recommendations failed for seed tracks: {seed_tracks}. Status: {http_error.response.status_code}. Response: {http_error.response.text if http_error.response else 'No response'}")
        return []
    except requests.exceptions.RequestException as req_error:
        logger.error(f"Error getting Spotify recommendations: {req_error}")
        return []

def _get_user_spotify_token_sync(code: str) -> Optional[Dict]:
    """Exchange authorization code for Spotify access and refresh tokens (synchronous)."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI or SPOTIFY_REDIRECT_URI == "https://your-callback-url.com":
        logger.warning("Spotify OAuth credentials or redirect URI not configured properly")
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
        response = requests.post(url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        token_data = response.json()
        token_data["expires_at"] = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600))).timestamp()
        return token_data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting user Spotify token with code: {e}")
        return None

def _refresh_spotify_token_sync(user_id: int) -> Optional[str]:
    """Refresh Spotify access token using refresh token (synchronous)."""
    context = user_contexts.get(user_id, {})
    spotify_info = context.get("spotify", {})
    refresh_token = spotify_info.get("refresh_token")

    if not refresh_token:
        logger.warning(f"No refresh token found for user {user_id}. Clearing their Spotify context.")
        if user_id in user_contexts and "spotify" in user_contexts[user_id]:
            user_contexts[user_id]["spotify"] = {}
        return None

    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        logger.warning("Spotify client credentials for token refresh not configured.")
        return None

    url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    headers = {
        "Authorization": f"Basic {auth_header}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token}

    try:
        response = requests.post(url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        token_data = response.json()
        expires_at = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600))).timestamp()
        
        # Ensure user_contexts[user_id] and user_contexts[user_id]["spotify"] exist
        if user_id not in user_contexts: user_contexts[user_id] = {}
        if "spotify" not in user_contexts[user_id]: user_contexts[user_id]["spotify"] = {}

        user_contexts[user_id]["spotify"].update({
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token", refresh_token), # Keep old if not in response
            "expires_at": expires_at
        })
        return token_data.get("access_token")
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 400: # Bad Request, often invalid refresh token
            logger.error(f"Invalid refresh token for user {user_id} (HTTP 400): {e}. Clearing Spotify context.")
            if user_id in user_contexts and "spotify" in user_contexts[user_id]:
                 user_contexts[user_id]["spotify"] = {}
        else:
            logger.error(f"HTTP error refreshing Spotify token for user {user_id}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error refreshing Spotify token for user {user_id}: {e}")
        return None

def _get_user_spotify_data_sync(user_id: int, endpoint: str) -> Optional[List[Dict]]:
    """Fetch user-specific Spotify data (synchronous)."""
    context = user_contexts.get(user_id, {})
    spotify_data = context.get("spotify", {})
    access_token = spotify_data.get("access_token")
    expires_at = spotify_data.get("expires_at")

    if not access_token or (expires_at and datetime.now(pytz.UTC).timestamp() > expires_at):
        logger.info(f"Access token for user {user_id} expired or missing, attempting refresh for endpoint {endpoint}.")
        access_token = _refresh_spotify_token_sync(user_id) # Call sync version
        if not access_token:
            logger.warning(f"Failed to refresh token for user {user_id} for endpoint {endpoint}.")
            return None

    url = f"https://api.spotify.com/v1/me/{endpoint}"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"limit": SPOTIFY_USER_DATA_LIMIT}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("items", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Spotify user data ({endpoint}) for user {user_id}: {e}")
        return None

def _get_user_spotify_playlists_sync(user_id: int) -> Optional[List[Dict]]:
    """Fetch user's Spotify playlists (synchronous)."""
    # This function essentially does the same as _get_user_spotify_data_sync with a different endpoint.
    # It could be merged, but keeping separate for clarity if params differ later.
    return _get_user_spotify_data_sync(user_id, "playlists")


# ==================== ASYNC WRAPPERS for SPOTIFY HELPERS ====================
async def get_spotify_token() -> Optional[str]:
    return await asyncio.to_thread(_get_spotify_token_sync)

async def search_spotify_track(token: str, query: str) -> Optional[Dict]:
    return await asyncio.to_thread(_search_spotify_track_sync, token, query)

async def get_spotify_recommendations(token: str, seed_tracks: List[str], limit: int = SPOTIFY_RECOMMENDATION_LIMIT) -> List[Dict]:
    return await asyncio.to_thread(_get_spotify_recommendations_sync, token, seed_tracks, limit)

async def get_user_spotify_token(code: str) -> Optional[Dict]:
    return await asyncio.to_thread(_get_user_spotify_token_sync, code)

async def refresh_spotify_token(user_id: int) -> Optional[str]:
    return await asyncio.to_thread(_refresh_spotify_token_sync, user_id)

async def get_user_spotify_data(user_id: int, endpoint: str) -> Optional[List[Dict]]:
    return await asyncio.to_thread(_get_user_spotify_data_sync, user_id, endpoint)

async def get_user_spotify_playlists(user_id: int) -> Optional[List[Dict]]:
    return await asyncio.to_thread(_get_user_spotify_playlists_sync, user_id)


# ==================== YOUTUBE HELPER FUNCTIONS ====================

def is_valid_youtube_url(url: str) -> bool:
    """Check if the URL is a valid YouTube URL."""
    if not url:
        return False
    patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=([0-9A-Za-z_-]{11})',
        r'(https?://)?youtu\.be/([0-9A-Za-z_-]{11})',
        r'(https?://)?(www\.)?youtube\.com/shorts/([0-9A-Za-z_-]{11})'
    ]
    return any(re.search(pattern, url) for pattern in patterns)

def sanitize_filename(filename: str) -> str:
    """Remove invalid characters from filenames and limit length."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return sanitized[:150] # Max filename length can be an issue

def _download_youtube_audio_sync(url: str) -> Dict[str, Any]:
    """Download audio from a YouTube video (synchronous)."""
    video_id_match = re.search(r'(?:v=|/|shorts/)([0-9A-Za-z_-]{11})', url) # Adjusted regex for shorts
    if not video_id_match:
        logger.error(f"Invalid YouTube URL or video ID: {url}")
        return {"success": False, "error": "Invalid YouTube URL or video ID"}

    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio[abr<=128]/bestaudio',
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'postprocessor_args': ['-acodec', 'copy'], # For m4a, this copies the AAC stream
        'prefer_ffmpeg': True, # Prefer ffmpeg if available for reliability
        'max_filesize': MAX_FILE_SIZE_TELEGRAM + (5 * 1024 * 1024), # Download slightly larger, then check precise
        'retries': 3,
        'fragment_retries': 3,
        'socket_timeout': 30, # Socket timeout for yt-dlp
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Attempting to extract info for: {url}")
            info = ydl.extract_info(url, download=False)
            if not info:
                logger.warning(f"Could not extract video information for URL: {url}. Info was: {info}")
                return {"success": False, "error": "Could not extract video information"}

            title = sanitize_filename(info.get('title', 'Unknown_Title'))
            artist = info.get('artist', info.get('uploader', 'Unknown_Artist')) # uploader as fallback for artist
            
            # Update outtmpl with sanitized title to ensure consistent filename
            # yt-dlp uses the template for the final filename, so we need to ensure it's correct before download.
            # This is tricky as ydl_opts are set at instantiation. We'll rely on sanitize_filename for the dict return
            # and yt-dlp's internal sanitization for the actual download filename (which is usually good).
            # The key is that our constructed `potential_path` uses our `title`.

            ydl_opts_download = ydl_opts.copy()
            # Use a temporary filename pattern that yt-dlp will use for downloading
            # Then we find it based on our sanitized title. This is a bit indirect.
            # A more robust way might be to use a fixed ID-based name, then rename.
            # For now, let's assume yt-dlp's %(title)s.%(ext)s with its sanitization is close enough.

            logger.info(f"Starting download for: {title} from {url}")
            ydl.download([url]) # This uses the ydl_opts from instantiation.

            audio_path = None
            # Search for the downloaded file. yt-dlp might use slightly different sanitization.
            # Best practice: have yt-dlp output to a predictable ID-based name, then rename.
            # For now, this search based on our sanitized title might work if yt-dlp's sanitization is similar.
            # A more robust approach:
            # 1. Set outtmpl to something like f'{DOWNLOAD_DIR}/{info["id"]}.%(ext)s'
            # 2. After download, rename f'{DOWNLOAD_DIR}/{info["id"]}.{info["ext"]}' to f'{DOWNLOAD_DIR}/{title}.{info["ext"]}'
            # Let's stick to the current approach for now and refine if it becomes problematic.

            downloaded_title_from_info = info.get('title', 'Unknown_Title') # Use title from info dict for path construction
            base_filename = sanitize_filename(downloaded_title_from_info)

            for ext in ['m4a', 'webm', 'mp3', 'opus', 'ogg']: # Added ogg
                potential_path = os.path.join(DOWNLOAD_DIR, f"{base_filename}.{ext}")
                if os.path.exists(potential_path):
                    audio_path = potential_path
                    break
            
            if not audio_path:
                # Fallback: list directory and find most recent matching common audio extension
                logger.warning(f"Could not find downloaded file with exact sanitized name '{base_filename}'. Trying to find latest audio file.")
                files = [os.path.join(DOWNLOAD_DIR, f) for f in os.listdir(DOWNLOAD_DIR)]
                files = [f for f in files if os.path.isfile(f) and f.lower().endswith(('.m4a', '.mp3', '.webm', '.opus', '.ogg'))]
                if files:
                    audio_path = max(files, key=os.path.getctime) # Get most recent
                    logger.info(f"Found fallback audio path: {audio_path}")
                else:
                    logger.error(f"Downloaded file not found for title '{title}' (sanitized: '{base_filename}')")
                    return {"success": False, "error": "Downloaded file not found. It might have an unexpected name or extension."}

            file_size_bytes = os.path.getsize(audio_path)
            if file_size_bytes > MAX_FILE_SIZE_TELEGRAM:
                logger.error(f"File too large: {file_size_bytes / (1024*1024):.2f} MB exceeds Telegram's limit for '{title}'")
                if os.path.exists(audio_path): os.remove(audio_path) # Clean up large file
                return {"success": False, "error": f"File too large for Telegram (max {MAX_FILE_SIZE_TELEGRAM/(1024*1024)} MB)"}
            
            return {
                "success": True,
                "title": downloaded_title_from_info, # Use the title from yt-dlp info
                "artist": artist,
                "thumbnail_url": info.get('thumbnail', ''),
                "duration": info.get('duration', 0),
                "audio_path": audio_path
            }
    except yt_dlp.utils.DownloadError as e:
        # Regex to check for common "video unavailable" messages
        if re.search(r'ideo unavailable|Private video|removed by the user', str(e), re.IGNORECASE):
            logger.warning(f"YouTube download error (video unavailable): {e} for {url}")
            return {"success": False, "error": "Video is unavailable (private, deleted, or restricted)."}
        logger.error(f"YouTube download error: {e} for {url}")
        return {"success": False, "error": f"Download failed. This video might be region-restricted or have other issues."}
    except Exception as e:
        logger.error(f"Unexpected error downloading YouTube audio for {url}: {e}", exc_info=True)
        return {"success": False, "error": "An unexpected error occurred during download"}

async def download_youtube_audio(url: str) -> Dict[str, Any]:
    return await asyncio.to_thread(_download_youtube_audio_sync, url)

@lru_cache(maxsize=128) # Cache search results
def _search_youtube_sync(query: str, max_results: int = YOUTUBE_SEARCH_MAX_RESULTS) -> List[Dict[str, Any]]:
    """Search YouTube for videos matching the query (synchronous, cached)."""
    query = sanitize_input(query) # Sanitize query before search
    logger.info(f"Searching YouTube (sync, cached) for: {query}")
    try:
        ydl_opts: Dict[str, Any] = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist', # Recommended for searches
            'default_search': f'ytsearch{max_results}', # ytsearchN:query
            'format': 'bestaudio', # Helps get duration if available
            'noplaylist': True, # Ensure we don't process actual playlists if a URL is given by mistake
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # search_query = f"ytsearch{max_results}:{query}" # No longer needed due to default_search
            info = ydl.extract_info(query, download=False) # Pass query directly
            
            if not info or 'entries' not in info or not info['entries']:
                logger.warning(f"No YouTube search results for query: {query}")
                return []
            
            results: List[Dict[str, Any]] = []
            for entry in info['entries']:
                if entry and entry.get('id'): # Ensure entry and ID exist
                    results.append({
                        'title': entry.get('title', 'Unknown Title'),
                        'url': entry.get('webpage_url') or f"https://www.youtube.com/watch?v={entry['id']}",
                        'thumbnail': entry.get('thumbnail', ''),
                        'uploader': entry.get('uploader', entry.get('channel', 'Unknown Artist')),
                        'duration': entry.get('duration', 0),
                        'id': entry['id']
                    })
            return results
    except Exception as e:
        logger.error(f"Error searching YouTube for '{query}': {e}", exc_info=True)
        return []

async def search_youtube(query: str, max_results: int = YOUTUBE_SEARCH_MAX_RESULTS) -> List[Dict[str, Any]]:
    return await asyncio.to_thread(_search_youtube_sync, query, max_results)


# ==================== LYRICS FUNCTION ====================
@lru_cache(maxsize=64)
def _get_lyrics_sync(song_title: str, artist_name: Optional[str] = None) -> str:
    """Fetch lyrics using LyricsGenius (synchronous, cached)."""
    if not genius:
        return "Lyrics functionality is not configured (Genius client missing)."
    
    song_title = sanitize_input(song_title)
    if artist_name:
        artist_name = sanitize_input(artist_name)

    logger.info(f"Searching lyrics (sync, cached) for: {song_title} by {artist_name or 'Any Artist'}")
    try:
        if artist_name:
            song = genius.search_song(song_title, artist_name)
        else:
            song = genius.search_song(song_title)
        
        if song and song.lyrics:
            lyrics_text = song.lyrics
            # Clean up common annotations and excessive newlines
            lyrics_text = re.sub(r'\n*\[[^\]]*?\]\n*', '\n', lyrics_text) # Remove [Chorus], [Verse], etc. more carefully
            lyrics_text = re.sub(r'\d*EmbedShare URLCopyEmbedCopy', '', lyrics_text) # Remove Genius footer
            lyrics_text = re.sub(r'\d*Embed$', '', lyrics_text).strip() # Remove numEmbed
            # Attempt to remove "Song Title Lyrics" header if present and artist name is also there
            if artist_name:
                 lyrics_text = re.sub(rf'^{re.escape(song.title)}\sLyrics(\[.*?\])?\s*by\s*{re.escape(song.artist)}\s*', '', lyrics_text, flags=re.IGNORECASE | re.MULTILINE).strip()
            lyrics_text = re.sub(rf'^{re.escape(song.title)}\sLyrics(\[.*?\])?\s*', '', lyrics_text, flags=re.IGNORECASE | re.MULTILINE).strip()


            # Remove song/artist N Contributors etc. if they are at the very start
            lyrics_text = re.sub(r'^.*(?:Lyrics|contributors)\s*?(\n|$)', '', lyrics_text, count=1, flags=re.IGNORECASE).strip()
            lyrics_text = re.sub(r'\n{3,}', '\n\n', lyrics_text).strip() # Normalize multiple newlines

            if not lyrics_text or len(lyrics_text) < 20: # If lyrics are too short after cleaning
                 return f"Lyrics for '{song.title}' seem to be unavailable or it's instrumental."
            return lyrics_text
        else:
            return f"Couldn't find lyrics for '{song_title}'{f' by {artist_name}' if artist_name else ''}."
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout searching lyrics for '{song_title}' by '{artist_name}'")
        return "Sorry, the lyrics search timed out. Please try again."
    except Exception as e:
        logger.error(f"Error fetching lyrics for '{song_title}' by '{artist_name}': {e}", exc_info=True)
        return "An error occurred while fetching lyrics."

async def get_lyrics(song_title: str, artist_name: Optional[str] = None) -> str:
    return await asyncio.to_thread(_get_lyrics_sync, song_title, artist_name)


# ==================== AI CONVERSATION FUNCTIONS ====================

async def generate_chat_response(user_id: int, message: str) -> str:
    """Generate a conversational response using OpenAI."""
    if not client:
        return "I'm having trouble connecting to my AI service. Please try again later."

    message = sanitize_input(message)
    context = user_contexts.get(user_id, {
        "mood": None, "preferences": [], "conversation_history": [], "spotify": {}
    })
    if user_id not in user_contexts: user_contexts[user_id] = context # Ensure it's set

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": (
            "You are MelodyMind, a friendly, empathetic music companion bot. "
            "Your primary goal is to discuss music, help users find songs, and offer music-based emotional support. "
            "Keep responses concise (2-4 sentences), warm, and engaging. "
            "If the user has linked Spotify, subtly use their listening history (mood, artists, genres) to personalize responses. "
            "Do not explicitly state you are using their Spotify data unless relevant to a direct question about it. "
            "Focus on music discovery, lyrics, and positive interaction."
        )}
    ]

    # Add context from user_contexts
    system_hints = []
    if context.get("mood"): system_hints.append(f"User's current mood: {context['mood']}.")
    if context.get("preferences"): system_hints.append(f"User's genre preferences: {', '.join(context['preferences'])}.")
    if context.get("spotify"):
        recent_artists = [item["track"]["artists"][0]["name"] for item in context["spotify"].get("recently_played", [])[:3] if item.get("track")]
        if recent_artists: system_hints.append(f"User recently listened to: {', '.join(recent_artists)}.")
    if system_hints:
        messages.append({"role": "system", "content": "Background: " + " ".join(system_hints)})


    # Limit conversation history
    context["conversation_history"] = context.get("conversation_history", [])[-10:] # Last 5 pairs
    for hist_msg in context["conversation_history"]:
        messages.append(hist_msg)
    messages.append({"role": "user", "content": message})

    try:
        response = await client.chat.completions.create( # Use await for async OpenAI client
            model="gpt-3.5-turbo",
            messages=messages, # type: ignore because OpenAI's type hint for messages is broader
            max_tokens=150,
            temperature=0.7
        )
        reply = response.choices[0].message.content
        if reply:
            context["conversation_history"].extend([
                {"role": "user", "content": message},
                {"role": "assistant", "content": reply}
            ])
            user_contexts[user_id] = context # Save updated context
            return reply
        return "I'm not sure how to respond to that. Can we talk about music?"
    except Exception as e:
        logger.error(f"Error generating chat response for user {user_id}: {e}", exc_info=True)
        return "I'm having a little trouble thinking right now. Maybe we can find some music instead?"

async def analyze_conversation(user_id: int) -> Dict[str, Any]:
    """Analyze conversation history and Spotify data to extract music preferences using AI."""
    if not client:
        return {"genres": [], "artists": [], "mood": None}

    context = user_contexts.get(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    if user_id not in user_contexts: user_contexts[user_id] = context

    # If not enough data, return current context
    if len(context.get("conversation_history", [])) < 2 and not context.get("spotify"):
        return {"genres": context.get("preferences", []), "artists": [], "mood": context.get("mood")}

    conversation_text = ""
    if context.get("conversation_history"):
        history = context["conversation_history"][-10:] # Last 5 interactions
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])

    spotify_data_summary = "User's Spotify data: "
    spotify_hints = []
    if context.get("spotify"):
        if context["spotify"].get("recently_played"):
            tracks = context["spotify"]["recently_played"][:3]
            spotify_hints.append("Recently played: " + ", ".join([f"{item['track']['name']} by {item['track']['artists'][0]['name']}" for item in tracks if item.get("track")]))
        if context["spotify"].get("top_tracks"):
            tracks = context["spotify"]["top_tracks"][:3]
            spotify_hints.append("Top tracks: " + ", ".join([f"{item['name']} by {item['artists'][0]['name']}" for item in tracks]))
    spotify_data_summary += "; ".join(spotify_hints) if spotify_hints else "Not available or not significant."


    prompt_messages = [
        {"role": "system", "content":
            "You are an AI analyzing a user's conversation with a music bot and their Spotify data. "
            "Your goal is to extract their current mood, preferred music genres, and liked artists. "
            "Return the analysis as a JSON object with keys: 'mood' (string or null), 'genres' (list of strings), 'artists' (list of strings). "
            "Prioritize recent information from the conversation. If Spotify data contradicts recent conversation, conversation mood takes precedence. "
            "If unsure, return empty lists or null for mood."},
        {"role": "user", "content":
            f"User's stated mood (if any): {context.get('mood', 'Not explicitly stated by user yet')}\n"
            f"User's stated genre preferences (if any): {', '.join(context.get('preferences', ['None stated']))}\n"
            f"Conversation History:\n{conversation_text if conversation_text else 'No conversation history yet.'}\n\n"
            f"Spotify Data Summary:\n{spotify_data_summary}"}
    ]

    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo-0125", # Specify model that supports JSON mode well
            messages=prompt_messages, # type: ignore
            max_tokens=200,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        result_str = response.choices[0].message.content
        if not result_str:
            raise ValueError("Empty response from AI for conversation analysis.")
        
        result = json.loads(result_str)
        if not isinstance(result, dict):
            raise ValueError("AI analysis result is not a dictionary.")

        # Validate and use results
        analysis: Dict[str, Any] = {
            "mood": result.get("mood") if isinstance(result.get("mood"), str) else context.get("mood"),
            "genres": result.get("genres", []) if isinstance(result.get("genres"), list) else [],
            "artists": result.get("artists", []) if isinstance(result.get("artists"), list) else []
        }
        
        # Update user_contexts with refined preferences from AI
        if analysis["genres"]: context["preferences"] = list(set(context.get("preferences", []) + analysis["genres"]))[:3] # Add new, keep unique, limit
        if analysis["mood"] and (not context.get("mood") or context.get("mood") != analysis["mood"]): # Update mood if AI found one or it's different
             context["mood"] = analysis["mood"]
        user_contexts[user_id] = context

        return analysis
    except Exception as e:
        logger.error(f"Error in analyze_conversation for user {user_id}: {e}", exc_info=True)
        return {"genres": context.get("preferences", []), "artists": [], "mood": context.get("mood")} # Fallback

async def is_music_request(user_id: int, message: str) -> Dict[str, Any]:
    """Use AI to determine if a message is a music/song request and extract query."""
    if not client:
        return {"is_music_request": False, "song_query": None}

    message = sanitize_input(message)
    try:
        response = await client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[ # type: ignore
                {"role": "system", "content":
                    "You are an AI that determines if a user's message is a request for a song or music (e.g., 'play a song', 'find some music', 'I want to hear X by Y'). "
                    "If it is a request, extract the song title and artist if possible, otherwise just the general query. "
                    "Respond in JSON format with two keys: 'is_music_request' (boolean) and 'song_query' (string, or null if not a music request or no specific query)."},
                {"role": "user", "content": f"Analyze this message: '{message}'"}
            ],
            max_tokens=80,
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        result_str = response.choices[0].message.content
        if not result_str:
             raise ValueError("Empty response from AI for music request check.")
        
        result = json.loads(result_str)
        if not isinstance(result, dict):
            raise ValueError("AI music request check result is not a dictionary.")

        is_request = result.get("is_music_request", False)
        if isinstance(is_request, str): # Handle string "true"/"false"
            is_request = is_request.lower() in ("yes", "true")
        
        song_query = result.get("song_query")
        if not isinstance(song_query, str) or not song_query.strip():
            song_query = None
            
        return {"is_music_request": bool(is_request), "song_query": song_query}
    except Exception as e:
        logger.error(f"Error in is_music_request for user {user_id}, message '{message}': {e}", exc_info=True)
        return {"is_music_request": False, "song_query": None}


# ==================== MUSIC DETECTION (Rule-based Pre-filter) ====================

def detect_music_in_message(text: str) -> Optional[str]:
    """Detect if a message is likely asking for music using regex (faster than AI for simple cases)."""
    text = text.lower()
    # Patterns for direct requests like "play song by artist"
    patterns = [
        r'play\s+(.+?)(?:\s+by\s+(.+))?$',
        r'find\s+(?:me\s+|me\s+a\s+)?song\s+(.+?)(?:\s+by\s+(.+))?$',
        r'download\s+(.+?)(?:\s+by\s+(.+))?$',
        r'get\s+(?:me\s+)?(.+?)(?:\s+by\s+(.+))?$',
        r'i\s+want\s+to\s+listen\s+to\s+(.+?)(?:\s+by\s+(.+))?$',
        r'can\s+you\s+find\s+(.+?)(?:\s+by\s+(.+))?$',
    ]
    # Keywords that might indicate a music request, needing AI for confirmation
    general_music_keywords = ['music', 'song', 'track', 'tune', 'audio', 'listen to something']

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            song_part = match.group(1).strip()
            artist_part = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else None
            if artist_part:
                # Avoid cases where "by" is part of the song title itself incorrectly captured
                if artist_part in ["the", "a", "me"] and len(artist_part.split()) == 1: # Heuristic
                    return song_part 
                return f"{song_part} {artist_part}"
            return song_part

    # If no direct pattern matched, check for general keywords
    if any(keyword in text for keyword in general_music_keywords):
        # Check for negative context for keywords like "not looking for music"
        if not re.search(r'(not|don\'t|do not)\s+(want|looking for|need)\s+.*?(music|song|track)', text):
            return "AI_ANALYSIS_NEEDED" # Signal that AI should check more thoroughly
    return None


# ==================== TELEGRAM BOT HANDLERS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    welcome_msg = (
        f"Hi {user.first_name}! 👋 I'm MelodyMind, your Music Healing Companion.\n\n"
        "I can:\n"
        "🎵 Download music: Send a YouTube link or ask for a song (e.g., 'play Shape of You')\n"
        "📜 Find lyrics: Use /lyrics [song name] or ask naturally.\n"
        "💿 Recommend music: Use /recommend or tell me your mood.\n"
        "🔗 Link Spotify: Use /link_spotify for personalized recommendations.\n\n"
        "How can I help you with music today?"
    )
    await update.message.reply_text(welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "🎶 <b>MelodyMind - Music Healing Companion</b> 🎶\n\n"
        "<b>Core Commands:</b>\n"
        "/start - Welcome message & quick start\n"
        "/help - This help message\n"
        "/download [YouTube URL or Song Name] - Download music\n"
        # "/autodownload [song name] - Search & download the top result (Deprecated, use /download or natural language)\n"
        "/search [song name] - Search YouTube for a song & get options\n"
        "/lyrics [song name] or [artist - song] - Get lyrics for a song\n"
        "/recommend - Get personalized music recommendations\n"
        "/mood - Set your current mood for better recommendations\n"
        "/link_spotify - Connect your Spotify account\n"
        "/clear - Clear your conversation history with me\n\n"
        "<b>Natural Language Examples:</b>\n"
        "- \"I'm feeling happy, suggest some upbeat songs!\"\n"
        "- \"Play Yesterday by The Beatles\"\n"
        "- \"What are the lyrics to Bohemian Rhapsody?\"\n"
        "- \"Send me this song: [YouTube link]\"\n\n"
        "Just chat with me, I'll try my best to understand!"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6),
       retry=lambda retry_state: isinstance(retry_state.outcome.exception(), (TimedOut, NetworkError)))
async def reply_message_with_retry(message: Any, text: str, **kwargs: Any) -> Any: # Can be Update.message or Query.message
    return await message.reply_text(text, **kwargs)

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6),
       retry=lambda retry_state: isinstance(retry_state.outcome.exception(), (TimedOut, NetworkError)))
async def edit_message_with_retry(message: Any, text: str, **kwargs: Any) -> Any: # Query.message
    return await message.edit_text(text, **kwargs)

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=6),
       retry=lambda retry_state: isinstance(retry_state.outcome.exception(), (TimedOut, NetworkError)))
async def send_audio_with_retry(bot: Any, chat_id: int, audio: Any, **kwargs: Any) -> Any:
    return await bot.send_audio(chat_id=chat_id, audio=audio, **kwargs)

async def download_music_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, url_or_query: Optional[str] = None) -> None:
    """Handles music download requests, either from a URL or a search query."""
    if not update.message:
        logger.warning("download_music_handler called without update.message")
        return

    user_id = update.effective_user.id
    
    # Determine if it's a URL or a search query
    is_url = False
    if url_or_query and is_valid_youtube_url(url_or_query):
        final_query = url_or_query
        is_url = True
    elif context.args:
        text_input = " ".join(context.args)
        if is_valid_youtube_url(text_input):
            final_query = text_input
            is_url = True
        else:
            final_query = text_input # Treat as search query
    elif url_or_query: # Came from natural language processing as a query
        final_query = url_or_query
    else: # Should not happen if called from /download or natural language with query
        await reply_message_with_retry(update.message, "Please specify a YouTube URL or a song name to download.")
        return

    async with download_lock:
        if user_id in active_downloads:
            await reply_message_with_retry(update.message, "⚠️ You already have a download in progress. Please wait.")
            return
        active_downloads.add(user_id)

    status_msg = None
    try:
        if is_url:
            status_msg = await reply_message_with_retry(update.message, f"⏳ Starting download for URL...")
            video_url = final_query
        else: # It's a search query
            status_msg = await reply_message_with_retry(update.message, f"🔍 Searching for '{final_query}' to download...")
            results = await search_youtube(final_query, max_results=1)
            if not results or not results[0].get('url'):
                await edit_message_with_retry(status_msg, f"❌ Couldn't find any direct matches for '{final_query}'. Try /search for more options.")
                return
            video_url = results[0]["url"]
            await edit_message_with_retry(status_msg, f"✅ Found: {results[0]['title']}. ⏳ Downloading...")

        download_result = await download_youtube_audio(video_url)

        if not download_result["success"]:
            error_message = download_result.get("error", "Unknown download error.")
            await edit_message_with_retry(status_msg, f"❌ Download failed: {error_message}")
            return

        await edit_message_with_retry(status_msg, f"✅ Downloaded: {download_result['title']}\n⏳ Preparing to send file...")
        
        with open(download_result["audio_path"], 'rb') as audio_file:
            sent_message = await send_audio_with_retry(
                context.bot,
                chat_id=update.effective_chat.id,
                audio=audio_file,
                title=download_result["title"][:MAX_AUDIO_TITLE_LENGTH],
                performer=download_result.get("artist", "Unknown Artist")[:MAX_AUDIO_TITLE_LENGTH],
                caption=f"🎵 {download_result['title']}",
                duration=download_result.get('duration', 0)
            )
        if status_msg: await status_msg.delete() # Delete status message after sending audio

    except (TimedOut, NetworkError) as te:
        logger.error(f"Telegram API timeout/network error during download process for user {user_id}: {te}")
        if status_msg: await edit_message_with_retry(status_msg, "⚠️ A network issue occurred. Please try again.")
        else: await reply_message_with_retry(update.message, "⚠️ A network issue occurred. Please try again.")
    except Exception as e:
        logger.error(f"Error in download_music_handler for user {user_id}, query '{final_query}': {e}", exc_info=True)
        if status_msg: await edit_message_with_retry(status_msg, "❌ An unexpected error occurred. Please try again.")
        else: await reply_message_with_retry(update.message, "❌ An unexpected error occurred. Please try again.")
    finally:
        async with download_lock:
            if user_id in active_downloads:
                active_downloads.remove(user_id)
        # Clean up downloaded file
        if 'download_result' in locals() and download_result.get("success") and download_result.get("audio_path"):
            if os.path.exists(download_result["audio_path"]):
                try:
                    os.remove(download_result["audio_path"])
                    logger.info(f"Cleaned up file: {download_result['audio_path']}")
                except OSError as e_os:
                    logger.error(f"Error deleting file {download_result['audio_path']}: {e_os}")


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not context.args:
        await reply_message_with_retry(update.message, "Please specify what you're looking for. Example:\n/search Shape of You Ed Sheeran")
        return

    query = " ".join(context.args)
    status_msg = await reply_message_with_retry(update.message, f"🔍 Searching YouTube for: '{query}'...")
    
    results = await search_youtube(query, max_results=YOUTUBE_SEARCH_MAX_RESULTS)
    
    if not results:
        await edit_message_with_retry(status_msg, f"Sorry, I couldn't find any songs for '{query}'.")
        return

    keyboard_buttons = []
    response_text = f"🔎 Search results for '{query}':\n\n"
    for i, result in enumerate(results):
        if not result.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', result['id']):
            logger.warning(f"Skipping invalid YouTube result ID in search: {result.get('id', 'No ID')}")
            continue
        
        duration_str = ""
        if result.get('duration') and isinstance(result['duration'], (int, float)) and result['duration'] > 0:
            minutes = int(result['duration'] // 60)
            seconds = int(result['duration'] % 60)
            duration_str = f" [{minutes}:{seconds:02d}]"
        
        title = result['title']
        title_display = (title[:35] + "...") if len(title) > 38 else title
        
        response_text += f"{i+1}. {title}{duration_str} (by {result.get('uploader', 'Unknown')})\n" # Add to text for non-button users
        keyboard_buttons.append([InlineKeyboardButton(f"[{i+1}] {title_display}{duration_str}", callback_data=f"download_{result['id']}")])

    if not keyboard_buttons: # All results were invalid
        await edit_message_with_retry(status_msg, f"Sorry, I found some items for '{query}', but none seem to be valid songs I can process.")
        return

    keyboard_buttons.append([InlineKeyboardButton("Cancel Search", callback_data="cancel_search")])
    reply_markup = InlineKeyboardMarkup(keyboard_buttons)
    await edit_message_with_retry(status_msg, response_text + "\nClick a button to download:", reply_markup=reply_markup)


async def get_lyrics_command_handler(update: Update, context: ContextTypes.DEFAULT_TYPE, query_override: Optional[str] = None) -> None:
    """Handles lyrics requests from command or natural language."""
    if not update.message: return

    query = query_override
    if not query: # If not called from natural language processing
        if not context.args:
            await reply_message_with_retry(update.message,
                "Please specify a song. Examples:\n"
                "/lyrics Bohemian Rhapsody\n"
                "/lyrics Queen - Bohemian Rhapsody"
            )
            return
        query = " ".join(context.args)

    if not query: # Should not happen if logic is correct
        await reply_message_with_retry(update.message, "Missing song query for lyrics.")
        return

    query = sanitize_input(query)
    status_msg = await reply_message_with_retry(update.message, f"🔍 Searching for lyrics: {query}")

    try:
        artist: Optional[str] = None
        song_title: str = query
        # Try to parse "Artist - Song" or "Song by Artist"
        if " - " in query:
            parts = query.split(" - ", 1)
            if len(parts) == 2: artist, song_title = parts[0].strip(), parts[1].strip()
        elif " by " in query.lower(): # "song by artist"
            parts = re.split(r'\s+by\s+', query, maxsplit=1, flags=re.IGNORECASE)
            if len(parts) == 2: song_title, artist = parts[0].strip(), parts[1].strip()
        
        lyrics_text = await get_lyrics(song_title, artist)

        if not lyrics_text or "not found" in lyrics_text.lower() or "unavailable" in lyrics_text.lower() or "not configured" in lyrics_text.lower():
            await edit_message_with_retry(status_msg, lyrics_text) # Show the "not found" message from get_lyrics
            return

        # Split long lyrics for Telegram message limits (4096 chars)
        max_len = 4000 # Leave some room for formatting and "continued" messages
        if len(lyrics_text) > max_len:
            await edit_message_with_retry(status_msg, lyrics_text[:max_len] + "\n\n<i>(Lyrics continue in next part...)</i>", parse_mode=ParseMode.HTML)
            remaining_lyrics = lyrics_text[max_len:]
            while remaining_lyrics:
                part = remaining_lyrics[:max_len]
                remaining_lyrics = remaining_lyrics[max_len:]
                await reply_message_with_retry(update.message, part + (("\n\n<i>(Lyrics continue...)</i>" if remaining_lyrics else "")), parse_mode=ParseMode.HTML)
        else:
            await edit_message_with_retry(status_msg, lyrics_text, parse_mode=ParseMode.HTML)

    except Exception as e:
        logger.error(f"Error in get_lyrics_command_handler for query '{query}': {e}", exc_info=True)
        await edit_message_with_retry(status_msg, "Sorry, an unexpected error occurred while fetching lyrics.")


async def recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    user_id = update.effective_user.id
    status_msg = await reply_message_with_retry(update.message, "🎧 Finding personalized music recommendations...")

    try:
        # Ensure user context exists
        if user_id not in user_contexts:
            user_contexts[user_id] = {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}}
        
        # Update Spotify data if linked
        if user_contexts[user_id].get("spotify") and user_contexts[user_id]["spotify"].get("access_token"):
            logger.info(f"Fetching latest Spotify data for user {user_id} for recommendation.")
            recently_played = await get_user_spotify_data(user_id, "player/recently-played")
            if recently_played is not None: user_contexts[user_id]["spotify"]["recently_played"] = recently_played
            
            top_tracks_data = await get_user_spotify_data(user_id, "top/tracks")
            if top_tracks_data is not None: user_contexts[user_id]["spotify"]["top_tracks"] = top_tracks_data
            
            # Playlists usually don't change that fast, maybe fetch less often or on demand
            # playlists = await get_user_spotify_playlists(user_id)
            # if playlists: user_contexts[user_id]["spotify"]["playlists"] = playlists

        # Analyze conversation for mood/genre hints
        ai_analysis = await analyze_conversation(user_id)
        mood = ai_analysis.get("mood") or user_contexts[user_id].get("mood")
        
        if not mood: # If no mood from AI or context, prompt user
            await status_msg.delete()
            await set_mood(update, context) # Re-use the /mood command's start
            return

        genres = list(set(ai_analysis.get("genres", []) + user_contexts[user_id].get("preferences", [])))
        artists_seed = ai_analysis.get("artists", [])

        # Prepare seeds for Spotify recommendation
        seed_track_ids: List[str] = []
        seed_artist_ids: List[str] = []
        seed_genres: List[str] = genres[:1] # Limit seed genres for Spotify API

        # Spotify seeds from user's listening history
        spotify_user_ctx = user_contexts[user_id].get("spotify", {})
        if spotify_user_ctx.get("recently_played"):
            seed_track_ids.extend([track["track"]["id"] for track in spotify_user_ctx["recently_played"][:SPOTIFY_SEED_TRACK_LIMIT] if track.get("track", {}).get("id")])
        if not seed_track_ids and spotify_user_ctx.get("top_tracks"):
            seed_track_ids.extend([track["id"] for track in spotify_user_ctx["top_tracks"][:SPOTIFY_SEED_TRACK_LIMIT] if track.get("id")])
        
        # Fallback: if AI suggested artists, try to get their Spotify IDs for seeding
        client_spotify_token = await get_spotify_token()
        if not seed_track_ids and artists_seed and client_spotify_token:
            for artist_name in artists_seed[:1]: # Use first artist seed
                artist_info = await search_spotify_track(client_spotify_token, f"artist:{artist_name}") # Search for artist
                if artist_info and artist_info.get('artists') and artist_info['artists'][0].get('id'):
                    seed_artist_ids.append(artist_info['artists'][0]['id'])
                    break # Found one artist seed

        # Attempt Spotify API recommendation (uses client token)
        recommendations: List[Dict] = []
        if client_spotify_token and (seed_track_ids or seed_artist_ids or seed_genres):
            # Spotify API allows a mix of up to 5 seed artists, tracks, and genres
            combined_seeds_params: Dict[str, str] = {}
            if seed_track_ids: combined_seeds_params["seed_tracks"] = ",".join(seed_track_ids[:SPOTIFY_SEED_TRACK_LIMIT])
            if seed_artist_ids: combined_seeds_params["seed_artists"] = ",".join(seed_artist_ids[:SPOTIFY_SEED_TRACK_LIMIT]) # Max 5 total seeds
            if seed_genres: combined_seeds_params["seed_genres"] = ",".join(seed_genres[:max(0, SPOTIFY_SEED_TRACK_LIMIT - len(seed_track_ids) - len(seed_artist_ids))]) # Fill remaining seed slots

            if combined_seeds_params: # Check if we have any seeds at all
                 # Temp using _get_spotify_recommendations_sync directly as it's complex to pass Dict[str,str] for params
                 # This should be _get_spotify_recommendations_sync(client_spotify_token, seed_tracks=[], limit=SPOTIFY_RECOMMENDATION_LIMIT, **combined_seeds_params)
                 # For now, let's simplify and use only track seeds if available, then fallback.
                 if seed_track_ids:
                    recommendations = await get_spotify_recommendations(client_spotify_token, seed_track_ids)


        if recommendations:
            response = f"🎵 <b>Personalized Spotify Recommendations for a '{mood}' mood:</b>\n\n"
            for i, track in enumerate(recommendations[:SPOTIFY_RECOMMENDATION_LIMIT], 1):
                artists_text = ", ".join(a["name"] for a in track.get("artists", []))
                album = track.get("album", {}).get("name", "")
                response += f"{i}. <b>{track['name']}</b> by {artists_text}"
                if album: response += f" (from <i>{album}</i>)"
                response += f" <a href='{track.get('external_urls', {}).get('spotify', '#')}'>Listen on Spotify</a>\n"
            response += "\n💡 <i>You can ask me to download these by name!</i>"
            await edit_message_with_retry(status_msg, response, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            return

        # Fallback to YouTube search based on mood and genres/artists
        search_query_parts = [mood]
        if genres: search_query_parts.append(genres[0]) # Add primary genre
        search_query_parts.append("music")
        if artists_seed: search_query_parts.extend(["like", artists_seed[0]]) # Add primary artist
        
        search_query = sanitize_input(" ".join(search_query_parts))
        logger.info(f"Falling back to YouTube search for recommendations with query: {search_query}")
        yt_results = await search_youtube(search_query, max_results=YOUTUBE_SEARCH_MAX_RESULTS)
        
        if yt_results:
            response = f"🎵 <b>Music suggestions for a '{mood}' mood (from YouTube):</b>\n\n"
            keyboard = []
            for i, result in enumerate(yt_results[:YOUTUBE_SEARCH_MAX_RESULTS], 1):
                if not result.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', result['id']): continue
                
                duration_str = ""
                if result.get('duration') and result['duration'] > 0:
                    minutes, seconds = divmod(int(result['duration']), 60)
                    duration_str = f" [{minutes}:{seconds:02d}]"
                
                response += f"{i}. <b>{result['title']}</b> by {result.get('uploader', 'Unknown')}{duration_str}\n"
                button_text = f"Download: {result['title'][:25]}..." if len(result['title']) > 28 else f"Download: {result['title']}"
                keyboard.append([InlineKeyboardButton(button_text, callback_data=f"download_{result['id']}")])
            
            if not keyboard: # All results were invalid
                await status_msg.delete()
                await provide_generic_recommendations(update, mood)
                return

            await edit_message_with_retry(status_msg, response, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard))
        else: # No YouTube results either
            await status_msg.delete() # Delete "Finding..."
            await provide_generic_recommendations(update, mood) # Use placeholder recommendations

    except Exception as e:
        logger.error(f"Error in recommend_music for user {user_id}: {e}", exc_info=True)
        await edit_message_with_retry(status_msg, "I couldn't get personalized recommendations right now. Please try /mood or /link_spotify first, or try again later.")


async def provide_generic_recommendations(update: Update, mood: str) -> None:
    """Provide generic, hardcoded recommendations when APIs fail or no data."""
    if not update.message: return
    mood = mood.lower()
    # ... (provide_generic_recommendations implementation remains largely the same)
    mood_recommendations = {
        "happy": [
            "Walking on Sunshine - Katrina & The Waves", "Happy - Pharrell Williams",
            "Can't Stop the Feeling - Justin Timberlake", "Uptown Funk - Mark Ronson ft. Bruno Mars",
        ],
        "sad": [
            "Someone Like You - Adele", "Fix You - Coldplay", "Hallelujah - Leonard Cohen (Jeff Buckley version)",
            "Everybody Hurts - R.E.M.",
        ],
        "energetic": [
            "Eye of the Tiger - Survivor", "Don't Stop Me Now - Queen", "Thunderstruck - AC/DC",
            "Can't Hold Us - Macklemore & Ryan Lewis",
        ],
        "relaxed": [
            "Weightless - Marconi Union", "Clair de Lune - Claude Debussy (Classical)", "Watermark - Enya",
            "Teardrop - Massive Attack",
        ],
        "focused": [
            "The Four Seasons - Vivaldi (Classical)", "Time - Hans Zimmer (Soundtrack)", "Experience - Ludovico Einaudi (Modern Classical)",
            "Ambient 1: Music for Airports - Brian Eno",
        ],
        "nostalgic": [
            "Yesterday - The Beatles", "Wonderwall - Oasis", "Landslide - Fleetwood Mac",
            "Bohemian Rhapsody - Queen",
        ]
    }
    recommendations = mood_recommendations.get(mood, mood_recommendations["happy"]) # Default to happy
    response_text = f"🎵 <b>Some general music ideas for a '{mood}' mood:</b>\n\n"
    for i, track in enumerate(recommendations, 1):
        response_text += f"{i}. {track}\n"
    response_text += "\n💡 <i>You can ask me to download these by name, or send a YouTube link!</i>"
    await reply_message_with_retry(update.message, response_text, parse_mode=ParseMode.HTML)


async def set_mood(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start conversation to set user's mood. Entry point for mood conversation."""
    if not update.message: return ConversationHandler.END
    keyboard = [
        [InlineKeyboardButton("Happy 😊", callback_data="mood_happy"), InlineKeyboardButton("Sad 😢", callback_data="mood_sad")],
        [InlineKeyboardButton("Energetic 💪", callback_data="mood_energetic"), InlineKeyboardButton("Relaxed 😌", callback_data="mood_relaxed")],
        [InlineKeyboardButton("Focused 🧠", callback_data="mood_focused"), InlineKeyboardButton("Nostalgic 🕰️", callback_data="mood_nostalgic")],
        [InlineKeyboardButton("Cancel", callback_data="mood_cancel")]
    ]
    await reply_message_with_retry(update.message,
        "How are you feeling today? This helps me recommend better music for you.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return MOOD # Next state is MOOD to handle the callback

async def enhanced_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    """Handle various button callbacks (mood, preferences, downloads, search actions)."""
    query = update.callback_query
    if not query or not query.data or not query.message: # Basic validation
        logger.warning("enhanced_button_handler received an incomplete query object.")
        if query: await query.answer("Error: Invalid button press.")
        return None
        
    await query.answer() # Acknowledge button press
    data: str = query.data
    user_id = query.from_user.id

    logger.info(f"Button pressed by user {user_id}: {data}")

    # Ensure user_id context exists
    if user_id not in user_contexts:
        user_contexts[user_id] = {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}}

    # Mood selection
    if data.startswith("mood_"):
        if data == "mood_cancel":
            await edit_message_with_retry(query.message, "Mood selection cancelled.")
            return ConversationHandler.END
        
        mood = data.split("_")[1]
        user_contexts[user_id]["mood"] = mood
        logger.info(f"User {user_id} set mood to: {mood}")
        
        # Optional: Ask for genre preference (can be skipped)
        keyboard_prefs = [
            [InlineKeyboardButton("Pop", callback_data="pref_pop"), InlineKeyboardButton("Rock", callback_data="pref_rock"), InlineKeyboardButton("Hip-Hop", callback_data="pref_hiphop")],
            [InlineKeyboardButton("Electronic", callback_data="pref_electronic"), InlineKeyboardButton("Classical", callback_data="pref_classical"), InlineKeyboardButton("Jazz", callback_data="pref_jazz")],
            [InlineKeyboardButton("Anything is fine!", callback_data="pref_skip")],
            [InlineKeyboardButton("Cancel", callback_data="pref_cancel")]
        ]
        await edit_message_with_retry(query.message,
            f"Got it! You're feeling {mood}. 🎶\nAny specific music genre you're in the mood for? (Optional)",
            reply_markup=InlineKeyboardMarkup(keyboard_prefs)
        )
        return PREFERENCE # Next state for preference selection

    # Genre preference selection
    elif data.startswith("pref_"):
        if data == "pref_cancel":
            await edit_message_with_retry(query.message, "Preference selection cancelled. Your mood is still set.")
            return ConversationHandler.END # End mood conversation
        if data == "pref_skip":
            await edit_message_with_retry(query.message,
                f"Okay, mood set to {user_contexts[user_id].get('mood', 'your chosen mood')}! "
                "Try /recommend, or just chat with me about music!"
            )
            return ConversationHandler.END

        preference = data.split("_")[1]
        if "preferences" not in user_contexts[user_id]: user_contexts[user_id]["preferences"] = []
        if preference not in user_contexts[user_id]["preferences"]: # Add if not already there
            user_contexts[user_id]["preferences"].append(preference)
            user_contexts[user_id]["preferences"] = user_contexts[user_id]["preferences"][-3:] # Keep last 3
        
        logger.info(f"User {user_id} added preference: {preference}. Current prefs: {user_contexts[user_id]['preferences']}")
        await edit_message_with_retry(query.message,
            f"Great! Mood: {user_contexts[user_id].get('mood')}, Genre preference: {preference} noted.\n"
            "You can now /recommend, /download, or just chat!"
        )
        return ConversationHandler.END # End mood conversation

    # Download from button (e.g., search results, recommendations)
    elif data.startswith("download_") or data.startswith("auto_download_"): # auto_download_ is for direct first result
        # Extract video ID
        try:
            video_id = data.split("_", 1)[1] if data.startswith("download_") else data.split("_", 2)[2]
            if not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
                raise ValueError("Invalid video ID format")
        except (IndexError, ValueError) as e:
            logger.error(f"Invalid video_id in callback data '{data}': {e}")
            await edit_message_with_retry(query.message, "❌ Error: Invalid song data from button. Please try searching again.")
            return None

        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        
        async with download_lock:
            if user_id in active_downloads:
                # Can't edit message from another download, so just answer query
                await query.answer("⚠️ You already have a download in progress. Please wait.", show_alert=True)
                return None
            active_downloads.add(user_id)

        original_message_text = query.message.text # Save original text to restore or append
        await edit_message_with_retry(query.message, f"{original_message_text}\n\n⏳ Preparing to download song ID: {video_id}...")

        try:
            download_result = await download_youtube_audio(youtube_url)
            if not download_result["success"]:
                error_msg = download_result.get("error", "Unknown error")
                await edit_message_with_retry(query.message, f"❌ Download failed: {error_msg}\nOriginal query was:\n{original_message_text}")
                return None # Keep original message with error

            await edit_message_with_retry(query.message, f"✅ Downloaded: {download_result['title']}\n⏳ Sending file...")
            
            with open(download_result["audio_path"], 'rb') as audio_file:
                await send_audio_with_retry(
                    context.bot,
                    chat_id=query.message.chat_id,
                    audio=audio_file,
                    title=download_result["title"][:MAX_AUDIO_TITLE_LENGTH],
                    performer=download_result.get("artist", "Unknown Artist")[:MAX_AUDIO_TITLE_LENGTH],
                    caption=f"🎵 {download_result['title']}",
                    duration=download_result.get('duration', 0)
                )
            # After sending, edit the original message to indicate completion
            await edit_message_with_retry(query.message, f"✅ Download complete & sent: {download_result['title']}")

        except (TimedOut, NetworkError) as te_net:
            logger.error(f"Network error during button download for {video_id}: {te_net}")
            await edit_message_with_retry(query.message, f"⚠️ Network issue while processing your download for '{video_id}'. Please try again.\nOriginal query:\n{original_message_text}")
        except Exception as e_btn:
            logger.error(f"Error in button download handler for {video_id}: {e_btn}", exc_info=True)
            await edit_message_with_retry(query.message, f"❌ An unexpected error occurred for song '{video_id}'.\nOriginal query:\n{original_message_text}")
        finally:
            async with download_lock:
                if user_id in active_downloads: active_downloads.remove(user_id)
            if 'download_result' in locals() and download_result.get("audio_path") and os.path.exists(download_result["audio_path"]):
                try: os.remove(download_result["audio_path"])
                except OSError as e_os_del: logger.error(f"Error deleting file {download_result['audio_path']} after button download: {e_os_del}")
        return None


    # Show more search options (from natural language handler's prompt)
    elif data.startswith("show_options_"):
        search_query = data.split("show_options_", 1)[1]
        await edit_message_with_retry(query.message, f"🔍 Okay, showing more options for '{search_query}'...")
        
        # Call the main search command logic by creating a mock update/context
        # This is a bit of a hack; ideally, the search result display logic would be a separate function.
        # For now, let's just inform the user to use /search.
        # await search_command(update, context) # This won't work directly as update is CallbackQuery
        await query.message.reply_text(f"To see more options for '{search_query}', please use the command: \n`/search {search_query}`", parse_mode=ParseMode.MARKDOWN)
        await query.message.delete() # Delete the message with "Show options" button
        return None

    # Cancel search/action
    elif data == "cancel_search":
        await edit_message_with_retry(query.message, "❌ Action cancelled.")
        return None
    
    elif data == "cancel_spotify": # From Spotify linking conversation
        await edit_message_with_retry(query.message, "Spotify linking cancelled. Use /link_spotify anytime to try again.")
        return ConversationHandler.END

    return None # Default return if no specific state change needed


async def link_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Initiate Spotify OAuth flow."""
    if not update.message: return ConversationHandler.END
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI or SPOTIFY_REDIRECT_URI == "https://your-callback-url.com":
        await reply_message_with_retry(update.message, "Sorry, Spotify linking is not configured correctly by the bot admin.")
        return ConversationHandler.END

    user_id = update.effective_user.id
    # Scopes for what data the bot can access
    scopes = "user-read-recently-played user-top-read user-read-private playlist-read-private playlist-read-collaborative"
    auth_url = (
        "https://accounts.spotify.com/authorize"
        f"?client_id={SPOTIFY_CLIENT_ID}"
        "&response_type=code"
        f"&redirect_uri={SPOTIFY_REDIRECT_URI}"
        f"&scope={requests.utils.quote(scopes)}" # URL encode scopes
        f"&state={user_id}" # Optional: for security, verify state later
    )
    keyboard = [
        [InlineKeyboardButton("🔗 Link My Spotify Account", url=auth_url)],
        [InlineKeyboardButton("Cancel", callback_data="cancel_spotify_link_process")] # Different callback for conv fallback
    ]
    await reply_message_with_retry(update.message,
        "Let's link your Spotify account for personalized music magic! 🎵\n\n"
        "1. Click the button below to go to Spotify.\n"
        "2. Log in and authorize access for MelodyMind.\n"
        "3. After authorizing, Spotify will redirect you to a page. **Copy the entire URL from that page's address bar.**\n"
        "4. Paste the **full URL** back here to me.\n\n"
        "Ready? Click below to start:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True
    )
    return SPOTIFY_CODE # Next state: expecting the redirect URL or code

async def spotify_code_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle the redirect URL from Spotify OAuth, extract code, and get token."""
    if not update.message or not update.message.text: return SPOTIFY_CODE # Stay in this state
    
    user_id = update.effective_user.id
    redirected_url = update.message.text.strip()

    # Extract code from the redirected URL
    # Example: https://your-callback-url.com/?code=AQB...&state=123
    code_match = re.search(r'[?&]code=([^&]+)', redirected_url)
    if not code_match:
        await reply_message_with_retry(update.message,
            "Hmm, that doesn't look like the correct redirect URL from Spotify. "
            "Please make sure you copy the full URL from your browser's address bar after authorizing. "
            "It should contain '?code=...' in it. Or type /cancel_spotify to stop."
        )
        return SPOTIFY_CODE # Stay in this state, ask again

    auth_code = code_match.group(1)
    
    # Optional: Verify state parameter if you used one and stored it
    # state_match = re.search(r'[?&]state=([^&]+)', redirected_url)
    # if not state_match or state_match.group(1) != str(user_id):
    #     await update.message.reply_text("Security check failed (state mismatch). Please try linking again.")
    #     return ConversationHandler.END

    status_msg = await reply_message_with_retry(update.message, "Processing Spotify authorization...")
    token_data = await get_user_spotify_token(auth_code) # Async wrapper

    if not token_data or not token_data.get("access_token"):
        await edit_message_with_retry(status_msg,
            "❌ Failed to link Spotify account. The authorization code might be invalid or expired. "
            "Please try /link_spotify again to get a new link. Or type /cancel_spotify to stop."
        )
        return SPOTIFY_CODE # Stay, user might try paste again or cancel

    if user_id not in user_contexts:
        user_contexts[user_id] = {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}}
    
    user_contexts[user_id]["spotify"] = {
        "access_token": token_data["access_token"],
        "refresh_token": token_data.get("refresh_token"),
        "expires_at": token_data["expires_at"],
        "scopes": token_data.get("scope","").split() # Store granted scopes
    }
    logger.info(f"Spotify account linked for user {user_id}. Token expires at: {datetime.fromtimestamp(token_data['expires_at'], pytz.UTC)}")

    # Fetch initial data
    recently_played = await get_user_spotify_data(user_id, "player/recently-played")
    if recently_played: user_contexts[user_id]["spotify"]["recently_played"] = recently_played

    await edit_message_with_retry(status_msg,
        "✅ Spotify account linked successfully! 🎉\n"
        "I can now use your listening habits for even better recommendations. Try /recommend!"
    )
    return ConversationHandler.END

async def cancel_spotify_linking_process(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels the Spotify linking conversation."""
    query = update.callback_query
    if query: # Came from button
        await query.answer()
        await edit_message_with_retry(query.message, "Spotify linking process cancelled.")
    elif update.message: # Came from /cancel_spotify command
        await reply_message_with_retry(update.message, "Spotify linking process cancelled.")
    return ConversationHandler.END


async def enhanced_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Main message handler for non-command text."""
    if not update.message or not update.message.text: return
    
    user_id = update.effective_user.id
    text = sanitize_input(update.message.text)
    logger.info(f"Handling message from user {user_id}: '{text[:50]}...'")

    # Ensure user context exists
    if user_id not in user_contexts:
        user_contexts[user_id] = {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}}

    # 1. Direct YouTube URL
    if is_valid_youtube_url(text):
        logger.info(f"Detected YouTube URL from user {user_id}: {text}")
        await download_music_handler(update, context, url_or_query=text)
        return

    # 2. Rule-based music detection (quick check)
    detected_song_query_rulebased = detect_music_in_message(text)
    
    # 3. If rule-based needs AI or no match, try AI for music request detection
    music_request_ai_result: Optional[Dict[str, Any]] = None
    final_song_query: Optional[str] = None

    if detected_song_query_rulebased == "AI_ANALYSIS_NEEDED":
        logger.info(f"Rule-based detection suggests AI check for music request: '{text}'")
        music_request_ai_result = await is_music_request(user_id, text)
        if music_request_ai_result and music_request_ai_result.get("is_music_request") and music_request_ai_result.get("song_query"):
            final_song_query = music_request_ai_result["song_query"]
    elif detected_song_query_rulebased:
        final_song_query = detected_song_query_rulebased
    else: # No rule-based match, try AI as a general fallback for music query
        logger.info(f"No rule-based music match, trying AI for: '{text}'")
        music_request_ai_result = await is_music_request(user_id, text)
        if music_request_ai_result and music_request_ai_result.get("is_music_request") and music_request_ai_result.get("song_query"):
            final_song_query = music_request_ai_result["song_query"]

    # If a song query was identified (either by rules or AI)
    if final_song_query:
        logger.info(f"Identified song query '{final_song_query}' for user {user_id}.")
        # Ask for confirmation before auto-downloading the first result
        status_msg = await reply_message_with_retry(update.message, f"🔍 Searching for: '{final_song_query}' based on your message...")
        results = await search_youtube(final_song_query, max_results=1)
        
        if not results or not results[0].get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', results[0]['id']):
            await edit_message_with_retry(status_msg, f"Sorry, I couldn't find a clear match for '{final_song_query}'. Try being more specific or use /search.")
            return

        top_result = results[0]
        keyboard = [
            [InlineKeyboardButton(f"✅ Yes, download: {top_result['title'][:20]}...", callback_data=f"download_{top_result['id']}")],
            [InlineKeyboardButton("👀 Show more options", callback_data=f"show_options_{final_song_query}")], # Pass original query
            [InlineKeyboardButton("❌ No, cancel", callback_data="cancel_search")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        message_text = (
            f"I found '{top_result['title']}' by {top_result.get('uploader', 'Unknown')}.\n\n"
            f"Would you like me to download this, or show you more options?"
        )
        await edit_message_with_retry(status_msg, message_text, reply_markup=reply_markup)
        return

    # 4. Lyrics request detection (e.g., "what are the lyrics to...")
    lower_text = text.lower()
    lyrics_keywords = ["lyrics for", "words to", "what's the song that goes", "what are the lyrics to"]
    for keyword in lyrics_keywords:
        if keyword in lower_text:
            song_query_for_lyrics = text.lower().split(keyword, 1)[-1].strip()
            if song_query_for_lyrics:
                logger.info(f"Detected lyrics request for '{song_query_for_lyrics}' user {user_id}.")
                await get_lyrics_command_handler(update, context, query_override=song_query_for_lyrics)
                return

    # 5. Mood detection (e.g., "I'm feeling sad")
    mood_match = re.search(r"i(?:'m| am| feel|'ve been feeling)\s+(?P<mood>\w+)", lower_text)
    if mood_match:
        detected_mood = mood_match.group('mood')
        # Basic validation for common moods
        common_moods = ["happy", "sad", "energetic", "relaxed", "focused", "nostalgic", "calm", "excited", "blue", "down", "upbeat"]
        if detected_mood in common_moods:
            user_contexts[user_id]["mood"] = detected_mood
            logger.info(f"Detected mood '{detected_mood}' for user {user_id} from message.")
            await reply_message_with_retry(update.message, f"Got it, you're feeling {detected_mood}. If you'd like song recommendations for this mood, try /recommend or just ask!")
            # Don't return here, allow AI to generate a conversational response too.

    # 6. Fallback to general AI chat response
    typing_msg = await reply_message_with_retry(update.message, "<i>MelodyMind is thinking...</i>", parse_mode=ParseMode.HTML)
    try:
        ai_response_text = await generate_chat_response(user_id, text)
        await edit_message_with_retry(typing_msg, ai_response_text)
    except Exception as e_chat:
        logger.error(f"Error generating or sending AI chat response for user {user_id}: {e_chat}", exc_info=True)
        await edit_message_with_retry(typing_msg, "I'm having a little trouble responding right now. Let's try that again in a moment?")


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message: return
    user_id = update.effective_user.id
    if user_id in user_contexts:
        user_contexts[user_id]["conversation_history"] = []
        user_contexts[user_id]["mood"] = None # Also clear mood
        user_contexts[user_id]["preferences"] = [] # And preferences
        logger.info(f"Cleared conversation history, mood, and preferences for user {user_id}.")
        await reply_message_with_retry(update.message, "✅ Your conversation history, mood, and preferences with me have been cleared.")
    else:
        await reply_message_with_retry(update.message, "You don't have any saved conversation data with me yet.")


async def cancel_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Generic cancel handler for conversations."""
    if update.message:
        await reply_message_with_retry(update.message, "Okay, action cancelled. What would you like to do next?")
    elif update.callback_query and update.callback_query.message:
        await edit_message_with_retry(update.callback_query.message, "Action cancelled.")
    return ConversationHandler.END

# ==================== ERROR HANDLING ====================
async def handle_telegram_error(update: Optional[object], context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors caused by updates and send user-friendly message."""
    logger.error(f"Update: {update} caused error: {context.error}", exc_info=context.error)
    
    if isinstance(context.error, (TimedOut, NetworkError)):
        error_message = "Sorry, I'm having trouble connecting. Please try again in a moment! 🔌"
    elif isinstance(context.error, yt_dlp.utils.DownloadError):
         error_message = "There was an issue with a download. The video might be unavailable or restricted. 🙁"
    else:
        error_message = "Oops! Something went a bit sideways. 🛠️ I've noted the issue. Please try again later."

    if update and isinstance(update, Update) and update.effective_message:
        try:
            await reply_message_with_retry(update.effective_message, error_message)
        except Exception as e:
            logger.error(f"Failed to send error message to user: {e}")
            if update.effective_chat:
                try: await context.bot.send_message(update.effective_chat.id, error_message)
                except Exception as e_send: logger.error(f"Failed to send fallback error message: {e_send}")
    elif update and isinstance(update, Update) and update.effective_chat: # For errors not tied to a specific message
         try: await context.bot.send_message(update.effective_chat.id, error_message)
         except Exception as e_send_chat: logger.error(f"Failed to send error message to chat: {e_send_chat}")


# ==================== UTILITY & CLEANUP ====================
def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection and clean text. Limit length."""
    if not text: return ""
    # Remove control characters except for common whitespace like \n, \r, \t
    text = "".join(ch for ch in text if unicodedata.category(ch)[0]!="C" or ch in ('\n', '\r', '\t'))
    # Limit length
    return text.strip()[:500] # Increased limit for potentially longer inputs like URLs for Spotify

def cleanup_downloads_atexit() -> None:
    """Clean up any temporary files in the download directory on exit."""
    logger.info("MelodyMind shutting down. Cleaning up download directory...")
    try:
        if os.path.exists(DOWNLOAD_DIR):
            for item_name in os.listdir(DOWNLOAD_DIR):
                item_path = os.path.join(DOWNLOAD_DIR, item_name)
                try:
                    if os.path.isfile(item_path) or os.path.islink(item_path):
                        os.unlink(item_path)
                        logger.info(f"Removed temp file: {item_path}")
                    # Optionally, remove empty subdirectories if any were created
                    # elif os.path.isdir(item_path): shutil.rmtree(item_path)
                except Exception as e_clean:
                    logger.error(f"Error removing {item_path} during cleanup: {e_clean}")
            logger.info("Download directory cleanup complete.")
    except Exception as e:
        logger.error(f"Error during atexit cleanup_downloads: {e}")

def signal_handler_exit(sig: int, frame: Any) -> None:
    """Handle termination signals for graceful shutdown."""
    logger.info(f"Received signal {sig}, initiating graceful shutdown...")
    # cleanup_downloads_atexit() is registered with atexit, so it will run.
    # If there are async tasks to await, this is where you'd manage them,
    # but python-telegram-bot's run_polling blocks, so direct async cleanup here is tricky.
    # The JobQueue should be shut down by PTB itself.
    sys.exit(0)

# Python 3.8+ compatibility for unicodedata
import unicodedata


# ==================== MAIN FUNCTION ====================
def main() -> None:
    """Start the enhanced MelodyMind bot."""
    # Validate essential environment variables
    required_env_vars = ["TELEGRAM_TOKEN"]
    if not TOKEN:
        logger.critical("TELEGRAM_TOKEN is not set. Bot cannot start.")
        sys.exit(1)
    
    # Check for optional but recommended services
    if not OPENAI_API_KEY: logger.warning("OPENAI_API_KEY not set. AI chat features will be disabled.")
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET: logger.warning("SPOTIFY_CLIENT_ID or SPOTIFY_CLIENT_SECRET not set. Spotify features will be limited.")
    if SPOTIFY_REDIRECT_URI == "https://your-callback-url.com": logger.warning("SPOTIFY_REDIRECT_URI is set to default. Spotify account linking will likely fail.")
    if not GENIUS_ACCESS_TOKEN: logger.warning("GENIUS_ACCESS_TOKEN not set. Lyrics feature will be disabled.")


    application = Application.builder().token(TOKEN).build()

    # Command Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("download", download_music_handler)) # Unified download handler
    application.add_handler(CommandHandler("search", search_command))
    # application.add_handler(CommandHandler("autodownload", auto_download_command)) # Deprecated
    application.add_handler(CommandHandler("lyrics", get_lyrics_command_handler))
    application.add_handler(CommandHandler("recommend", recommend_music))
    application.add_handler(CommandHandler("clear", clear_history))
    # application.add_handler(CommandHandler("spotify_code", spotify_code_command)) # Handled by conv

    # Spotify Linking Conversation Handler
    spotify_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("link_spotify", link_spotify)],
        states={
            SPOTIFY_CODE: [MessageHandler(filters.TEXT & ~filters.COMMAND, spotify_code_handler)],
        },
        fallbacks=[
            CommandHandler("cancel_spotify", cancel_spotify_linking_process),
            CallbackQueryHandler(cancel_spotify_linking_process, pattern="^cancel_spotify_link_process$")
            ]
    )
    application.add_handler(spotify_conv_handler)

    # Mood Setting Conversation Handler
    mood_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("mood", set_mood)],
        states={
            MOOD: [CallbackQueryHandler(enhanced_button_handler, pattern="^mood_")],
            PREFERENCE: [CallbackQueryHandler(enhanced_button_handler, pattern="^pref_")],
        },
        fallbacks=[CommandHandler("cancel", cancel_conversation)] # Generic cancel
    )
    application.add_handler(mood_conv_handler)

    # General CallbackQuery Handler (for downloads from buttons, search cancellations etc.)
    # Ensure patterns are specific enough not to clash with ConversationHandler callbacks if using same handler function
    application.add_handler(CallbackQueryHandler(enhanced_button_handler, pattern="^(download_|auto_download_|show_options_|cancel_search$)"))

    # Message Handler (must be last among text-based handlers)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, enhanced_handle_message))
    
    # Error Handler
    application.add_error_handler(handle_telegram_error)

    # Setup signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler_exit)
    signal.signal(signal.SIGTERM, signal_handler_exit)
    atexit.register(cleanup_downloads_atexit)

    logger.info("Starting Enhanced MelodyMind Bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()