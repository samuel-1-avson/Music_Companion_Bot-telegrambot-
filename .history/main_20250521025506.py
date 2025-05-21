
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
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "https://your-callback-url.com")
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=15, retries=3, remove_section_headers=True) if GENIUS_ACCESS_TOKEN and lyricsgenius else None

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
        logger.warning("Spotify tokens will NOT persist across restarts unless ENCRYPTION_KEY is static AND user_contexts are persisted.")
        cipher = Fernet(cipher_key_bytes)
else:
    cipher_key_bytes = Fernet.generate_key()
    logger.warning("ENCRYPTION_KEY not set. Generating a new one for this session.")
    logger.warning(f"To persist Spotify links (requires persisting user_contexts), set this as ENCRYPTION_KEY: {base64.urlsafe_b64encode(cipher_key_bytes).decode()}")
    logger.warning("User_contexts (including Spotify tokens) are currently in-memory and lost on restart.")
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

active_downloads = set()
user_contexts: Dict[int, Dict] = {}
logger.warning("User contexts are stored in-memory and will be lost on bot restart.")
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=15) 

# ==================== SPOTIFY HELPER FUNCTIONS ====================
# These seemed fine, no obvious conjugation issues. Included for completeness if needed.

async def get_spotify_token() -> Optional[str]:
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        logger.warning("Spotify client credentials not configured.")
        return None
    auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
    url = "https://accounts.spotify.com/api/token"
    headers = {"Authorization": f"Basic {auth_base64}", "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials"}
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url, headers=headers, data=data) as response:
                response.raise_for_status()
                return (await response.json()).get("access_token")
    except aiohttp.ClientError as e:
        logger.error(f"Error getting Spotify client_credentials token: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in get_spotify_token: {e}")
    return None

@alru_cache(maxsize=100) # type: ignore
async def search_spotify_track(token: str, query: str) -> Optional[Dict]:
    if not token: return None
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
    except Exception as e:
        logger.error(f"Unexpected error searching Spotify track '{query}': {e}")
    return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=8))
async def get_spotify_recommendations(token: str, seed_tracks: Optional[List[str]] = None, 
                                      seed_genres: Optional[List[str]] = None, 
                                      seed_artists: Optional[List[str]] = None, 
                                      limit: int = 5) -> List[Dict]:
    if not token:
        logger.warning("No token for Spotify recommendations.")
        return []
    
    params: Dict[str, Any] = {"limit": limit}
    current_seed_count = 0
    
    if seed_tracks and current_seed_count < 5:
        tracks_to_add = seed_tracks[:max(0, 5 - current_seed_count)]
        if tracks_to_add:
            params["seed_tracks"] = ",".join(tracks_to_add)
            current_seed_count += len(tracks_to_add)
            
    if seed_genres and current_seed_count < 5:
        genres_to_add = seed_genres[:max(0, 5 - current_seed_count)]
        if genres_to_add:
            params["seed_genres"] = ",".join(genres_to_add)
            current_seed_count += len(genres_to_add)

    if seed_artists and current_seed_count < 5:
        artists_to_add = seed_artists[:max(0, 5 - current_seed_count)]
        if artists_to_add:
            params["seed_artists"] = ",".join(artists_to_add)
            current_seed_count += len(artists_to_add)

    if current_seed_count == 0:
        logger.warning("No valid seeds provided for Spotify recommendations.")
        return []

    url = "https://api.spotify.com/v1/recommendations"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                return (await response.json()).get("tracks", [])
    except aiohttp.ClientError as e: 
        err_msg = f"API Error: {e.status}, {e.message}" if hasattr(e, 'status') else str(e)
        logger.error(f"Error getting Spotify recommendations (params: {params}): {err_msg}, URL: {e.request_info.url if hasattr(e, 'request_info') else 'N/A'}")
    except Exception as e:
        logger.error(f"Unexpected error getting Spotify recommendations (params: {params}): {e}")
    return []

async def get_user_spotify_token(user_id: int, code: str) -> Optional[Dict]:
    if not all([SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI]):
        logger.warning("Spotify OAuth credentials not fully configured.")
        return None
    url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    headers = {"Authorization": f"Basic {auth_header}", "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "authorization_code", "code": code, "redirect_uri": SPOTIFY_REDIRECT_URI}
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url, headers=headers, data=data) as response:
                if response.status == 400: 
                    error_details = await response.json()
                    logger.error(f"Spotify Bad Request (user {user_id}, code exchange): {error_details.get('error_description', response.reason)}")
                    return None 
                response.raise_for_status()
                token_data = await response.json()
                token_data["expires_at"] = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600) - 120)).timestamp() 
                return token_data
    except aiohttp.ClientError as e:
        err_msg = f"API Error: {e.status}, {e.message}" if hasattr(e, 'status') else str(e)
        logger.error(f"Error getting user Spotify token (user {user_id}): {err_msg}, URL: {e.request_info.url if hasattr(e, 'request_info') else 'N/A'}")
    except Exception as e:
        logger.error(f"Unexpected error in get_user_spotify_token (user {user_id}): {e}")
    return None

async def refresh_spotify_token(user_id: int) -> Optional[str]:
    context = user_contexts.get(user_id, {})
    spotify_data = context.get("spotify", {})
    encrypted_refresh_token = spotify_data.get("refresh_token")
    if not encrypted_refresh_token:
        logger.warning(f"No refresh token for user {user_id}.")
        return None
    if not all([SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET]):
        logger.error("Cannot refresh Spotify token: Client ID/Secret missing.")
        return None
    try:
        refresh_token_str = cipher.decrypt(encrypted_refresh_token).decode()
    except Exception as e:
        logger.error(f"Failed to decrypt refresh token for user {user_id}: {e}. Re-auth needed.")
        spotify_data.clear() 
        return None
    url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    headers = {"Authorization": f"Basic {auth_header}", "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token_str}
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url, headers=headers, data=data) as response:
                if response.status == 400: 
                    error_details = await response.json()
                    logger.error(f"Spotify Bad Request (user {user_id}, token refresh): {error_details.get('error_description', response.reason)}. User needs to re-link.")
                    spotify_data.clear()
                    return None
                response.raise_for_status()
                token_data = await response.json()
                new_access_token = token_data.get("access_token")
                if not new_access_token:
                    logger.error(f"Refresh grant no new access_token (user {user_id})")
                    return None
                new_refresh_token_str = token_data.get("refresh_token", refresh_token_str) 
                spotify_data["access_token"] = cipher.encrypt(new_access_token.encode())
                spotify_data["refresh_token"] = cipher.encrypt(new_refresh_token_str.encode())
                spotify_data["expires_at"] = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600) - 120)).timestamp()
                return new_access_token
    except aiohttp.ClientError as e:
        err_msg = f"API Error: {e.status}, {e.message}" if hasattr(e, 'status') else str(e)
        logger.error(f"Error refreshing token (user {user_id}): {err_msg}")
    except Exception as e:
        logger.error(f"Unexpected error refreshing token (user {user_id}): {e}")
    return None

async def get_user_spotify_access_token(user_id: int) -> Optional[str]:
    spotify_data = user_contexts.get(user_id, {}).get("spotify", {})
    encrypted_token = spotify_data.get("access_token")
    expires_at = spotify_data.get("expires_at")
    if not encrypted_token or (expires_at and datetime.now(pytz.UTC).timestamp() > expires_at):
        logger.info(f"Spotify token expired/missing for user {user_id}, refreshing.")
        return await refresh_spotify_token(user_id)
    try:
        return cipher.decrypt(encrypted_token).decode()
    except Exception as e:
        logger.error(f"Failed to decrypt access token for user {user_id}: {e}. Refreshing.")
        return await refresh_spotify_token(user_id)

async def get_user_spotify_data(user_id: int, endpoint: str, params: Optional[Dict] = None) -> Optional[List[Dict]]:
    access_token = await get_user_spotify_access_token(user_id)
    if not access_token:
        logger.warning(f"No Spotify access token for user {user_id} to fetch {endpoint}.")
        return None
    url = f"https://api.spotify.com/v1/me/{endpoint}"
    headers = {"Authorization": f"Bearer {access_token}"}
    request_params = {"limit": 10,**(params or {})}
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.get(url, headers=headers, params=request_params) as response:
                response.raise_for_status()
                return (await response.json()).get("items", [])
    except aiohttp.ClientError as e:
        logger.error(f"Error fetching Spotify user data ({endpoint}, user {user_id}): Status {e.status if hasattr(e,'status') else 'N/A'}, {e.message if hasattr(e,'message') else e}")
    except Exception as e:
        logger.error(f"Unexpected error fetching Spotify data ({endpoint}, user {user_id}): {e}")
    return None

# ==================== YOUTUBE HELPER FUNCTIONS ====================

def is_valid_youtube_url(url: str) -> bool:
    if not url: return False
    return bool(re.search(r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})", url))

def sanitize_filename(filename: str) -> str:
    return re.sub(r'[\\/*?:"<>|]', "_", filename)[:100]

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1.5, min=2, max=6))
def download_youtube_audio_sync(url: str) -> Dict[str, Any]:
    logger.info(f"Downloading YT audio: {url}")
    video_id_match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11})', url)
    video_id = video_id_match.group(1) if video_id_match else "UnknownID"
    
    try:
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'),
            'quiet': True, 'no_warnings': True, 'noplaylist': True,
            'max_filesize': 50 * 1024 * 1024, 'restrictfilenames': True,
            'sleep_interval_requests': 1, 'sleep_interval': 1, # Be nice to YouTube
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                logger.error(f"YT: No info for {url} (ID: {video_id})")
                return {"success": False, "error": "Could not extract video information"}
            
            title = sanitize_filename(info.get('title', 'Unknown Title'))
            artist = sanitize_filename(info.get('artist', info.get('uploader', 'Unknown Artist')))
            expected_path = ydl.prepare_filename(info)
            
            logger.info(f"YT: Downloading '{title}' to '{expected_path}'")
            ydl.extract_info(url, download=True) # Actual download

            actual_path = expected_path 
            if not os.path.exists(actual_path): 
                base_name_from_expected = os.path.splitext(os.path.basename(expected_path))[0]
                found_actual = False
                for f_name_in_dir in os.listdir(DOWNLOAD_DIR):
                    if base_name_from_expected in f_name_in_dir: 
                        actual_path = os.path.join(DOWNLOAD_DIR, f_name_in_dir)
                        logger.info(f"YT: File found at modified path {actual_path}")
                        found_actual = True
                        break
                if not found_actual:
                    logger.error(f"YT: Downloaded file not found at {expected_path} or variants for {url}")
                    return {"success": False, "error": "Downloaded file not found post-download"}

            if os.path.getsize(actual_path) > 50.5 * 1024 * 1024: # Check size again
                os.remove(actual_path)
                logger.warning(f"YT: File '{title}' over 50MB, removed.")
                return {"success": False, "error": "File >50MB"}
            
            return {"success": True, "title": title, "artist": artist,
                    "thumbnail_url": info.get('thumbnail'), "duration": info.get('duration', 0),
                    "audio_path": actual_path}
    except yt_dlp.utils.DownloadError as de:
        err_str = str(de).lower() # Normalize for easier matching
        if "video unavailable" in err_str: err_msg = "Video unavailable."
        elif "private video" in err_str: err_msg = "Private video."
        elif " geo-restricted" in err_str or " geo restricted" in err_str: err_msg = "Video geo-restricted."
        else: err_msg = f"Download issue: {str(de)[:80]}" # Generic ytdlp error
        logger.error(f"YT DownloadError for {url}: {err_msg} (Full: {de})")
        return {"success": False, "error": err_msg}
    except Exception as e:
        logger.error(f"YT Generic DL error {url}: {e}", exc_info=False) # exc_info=False for brevity in logs unless debugging specific case
        return {"success": False, "error": f"Unexpected error: {str(e)[:80]}"}

def search_youtube_sync(query: str, max_results: int = 5) -> List[Dict]:
    logger.info(f"YT Search: '{query}', max_results={max_results}")
    try:
        ydl_opts = {
            'quiet': True, 'no_warnings': True, 'extract_flat': 'discard_in_playlist',
            'default_search': f'ytsearch{max_results}', 'noplaylist': True,
            'sleep_interval_requests': 1, 'sleep_interval': 1
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            if not info or 'entries' not in info:
                logger.warning(f"YT Search: No results for '{query}'")
                return []
            
            results = []
            for e in info['entries']:
                if not e or not e.get('id'): continue # Ensure entry and ID exist
                results.append({
                    'title': e.get('title', 'Unknown Title'),
                    'url': e.get('webpage_url') or e.get('url') or f"https://youtube.com/watch?v={e['id']}", # Prefer webpage_url
                    'thumbnail': e.get('thumbnail') or (e.get('thumbnails')[0]['url'] if e.get('thumbnails') else ''),
                    'uploader': e.get('uploader', 'Unknown Artist'),
                    'duration': e.get('duration', 0), 
                    'id': e['id'] 
                })
            logger.info(f"YT Search: Found {len(results)} for '{query}'")
            return results
            
    except yt_dlp.utils.DownloadError as de: # Specific to yt-dlp search issues
        logger.error(f"YT Search DownloadError for '{query}': {de}")
    except Exception as e: # Other potential errors
        logger.error(f"YT Search error for '{query}': {e}", exc_info=False)
    return []

# ==================== AI AND LYRICS FUNCTIONS ====================

async def generate_chat_response(user_id: int, message: str) -> str:
    if not client: return "AI service offline. Let's talk music another way?"
    ctx = user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    ctx.setdefault("conversation_history", [])
    ctx["conversation_history"] = ctx["conversation_history"][-10:] # Limit context to last 5 interactions
    
    system_prompt = ("MelodyMind: Friendly music bot. Brief, warm chat about music/feelings. "
                     "If user asks for specific music, guide them to use commands or suggest you can search if they name a song. "
                     "Use user context (mood, prefs, Spotify artists) subtly. 2-3 sentences. "
                     "If you are going to suggest commands, only suggest simple ones like /search or /recommend.")
    messages: List[Dict[str,str]] = [{"role": "system", "content": system_prompt}]
    
    # Contextual Summary for AI
    summary_parts = []
    if ctx.get("mood") and ctx["mood"] != "neutral": summary_parts.append(f"Current Mood: {ctx['mood']}.")
    if ctx.get("preferences"): summary_parts.append(f"Music Preferences: {', '.join(ctx['preferences'])}.")
    if "spotify" in ctx and ctx["spotify"].get("recently_played"):
        try:
            recent_artists = list(set(item["track"]["artists"][0]["name"] for item in ctx["spotify"]["recently_played"][:2] if item.get("track") and item["track"].get("artists")))
            if recent_artists: summary_parts.append(f"Recently Listened To: {', '.join(recent_artists)}.")
        except (KeyError, IndexError, TypeError): pass # Graceful failure
            
    if summary_parts:
        messages.append({"role": "system", "content": "User Context: " + " ".join(summary_parts)})
    
    messages.extend(ctx["conversation_history"][-6:]) # Last 3 user/assistant exchanges
    messages.append({"role": "user", "content": message})
    
    try:
        response = await asyncio.to_thread(client.chat.completions.create, model="gpt-3.5-turbo", messages=messages, max_tokens=100, temperature=0.75) # max_tokens increased slightly for naturalness
        reply = response.choices[0].message.content.strip()
        ctx["conversation_history"].extend([{"role": "user", "content": message}, {"role": "assistant", "content": reply}])
        return reply
    except Exception as e:
        logger.error(f"AI chat response error (user {user_id}): {e}")
        return "Hmm, my thoughts are a bit jumbled right now. How about your favorite song, or we can try a /recommend?"

def get_lyrics_sync(song_title: str, artist: Optional[str] = None) -> str:
    if not genius: return "Lyrics service offline."
    logger.info(f"Lyrics search: '{song_title}' by '{artist or 'Any'}'")
    try:
        # The remove_section_headers=True in genius client init should handle [Chorus] etc.
        song = genius.search_song(song_title, artist) if artist else genius.search_song(song_title)
        if not song:
            return f"No lyrics found for '<b>{song_title}</b>'{f' by <i>{artist}</i>' if artist else ''}. Check spelling?"
        
        lyrics = song.lyrics # Already cleaned by library hopefully
        # Minor additional cleanup if needed:
        lyrics = re.sub(r'\d*Embed$', '', lyrics, flags=re.IGNORECASE).strip()
        lyrics = re.sub(r'^\S*Lyrics', '', lyrics, flags=re.IGNORECASE).strip() # Remove "SongTitleLyrics" if library misses it
        lyrics = re.sub(r'\n{3,}', '\n\n', lyrics).strip() # Condense excessive newlines

        if not lyrics: return f"Lyrics for '<b>{song.title}</b>' by <i>{song.artist}</i> seem to be empty after fetching."
        return f"🎵 <b>{song.title}</b> by <i>{song.artist}</i> 🎵\n\n{lyrics}"
    except Exception as e: # Broad catch from lyricsgenius library
        logger.error(f"Genius lyrics error for ('{song_title}', artist '{artist}'): {e}", exc_info=False)
        return f"An issue occurred fetching lyrics for '<b>{song_title}</b>'. Please try again later."

async def detect_mood_from_text(user_id: int, text: str) -> str:
    if not client: return user_contexts.get(user_id, {}).get("mood", "neutral")
    logger.debug(f"AI Mood detect (user {user_id}): '{text[:40]}...'")
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create, model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Detect dominant mood from text: happy, sad, anxious, excited, calm, angry, energetic, relaxed, focused, nostalgic, or neutral if unclear. Single word."},
                      {"role": "user", "content": f"Text: '{text}'"}],
            max_tokens=8, temperature=0.1)
        mood_raw = response.choices[0].message.content.lower().strip().replace(".", "")
        # Simple map for common AI outputs
        mood_map = {"positive": "happy", "very happy": "happy", "negative": "sad", "depressed": "sad", 
                    "joyful": "happy", "chill": "relaxed", "stressed": "anxious", "excited!": "excited"}
        mood = mood_map.get(mood_raw, mood_raw) # Normalize if possible
        
        valid_moods = ["happy", "sad", "anxious", "excited", "calm", "angry", "neutral", "energetic", "relaxed", "focused", "nostalgic"]
        if mood in valid_moods:
            logger.info(f"AI Mood (user {user_id}): '{mood}' from raw '{mood_raw}' for text '{text[:30]}'")
            return mood
        logger.warning(f"AI Mood: Invalid mood '{mood_raw}' for user {user_id}, defaulting to neutral.")
        return "neutral"
    except Exception as e:
        logger.error(f"AI Mood detect error (user {user_id}): {e}")
        return user_contexts.get(user_id, {}).get("mood", "neutral")

async def is_music_request(user_id: int, message: str) -> Dict[str, Any]:
    if not client: return {"is_music_request": False, "song_query": None}
    logger.debug(f"AI MusicReq detect (user {user_id}): '{message[:40]}...'")
    try:
        # Simplified prompt for better focus
        prompt = ("Is this message a specific request to play or download music now? "
                  "JSON: 'is_music_request': boolean, 'song_query': string (song and artist if clear, or null). "
                  "Examples of requests: 'play X by Y', 'download song Z'. "
                  "General chat about music/moods is NOT a request unless they name a specific song/artist they want NOW.")
        response = await asyncio.to_thread(
            client.chat.completions.create, model="gpt-3.5-turbo-0125", # Ensure JSON mode model
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": f"Message: '{message}'"}],
            max_tokens=70, temperature=0.0, response_format={"type": "json_object"})
        
        content = response.choices[0].message.content
        if not content: # Handle empty content case
            logger.warning(f"AI MusicReq (user {user_id}): Empty content from OpenAI for msg: '{message[:30]}'")
            return {"is_music_request": False, "song_query": None}
            
        result = json.loads(content)
        
        is_req_raw = result.get("is_music_request", False)
        # Robust boolean conversion
        is_req = str(is_req_raw).lower() == "true" if isinstance(is_req_raw, (str, bool)) else False
        
        query_raw = result.get("song_query")
        # Ensure query is a string or None
        query = str(query_raw).strip() if isinstance(query_raw, str) and str(query_raw).strip() else None
        
        logger.info(f"AI MusicReq (user {user_id}): is_req={is_req}, query='{query}' for msg: '{message[:30]}'")
        return {"is_music_request": is_req, "song_query": query}
        
    except json.JSONDecodeError as jde:
        logger.error(f"AI MusicReq JSON decode error (user {user_id}): {jde}. Raw content: '{content if 'content' in locals() else 'N/A'}'")
    except Exception as e: 
        logger.error(f"AI MusicReq error (user {user_id}): {e}", exc_info=False)
    return {"is_music_request": False, "song_query": None}

# ==================== TELEGRAM BOT HANDLERS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
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
    help_text = (
        "🎶 <b>MelodyMind - Your Music Companion</b> 🎶\n\n"
        "<b>Commands:</b>\n"
        "/start - Welcome\n/help - This guide\n"
        "/download <code>[YT URL]</code> - DL from link\n"
        "/autodownload <code>[song]</code> - Search & DL top result\n"
        "/search <code>[song]</code> - YT search w/ options\n"
        "/lyrics <code>[song]</code> or <code>[artist - song]</code> - Get lyrics\n"
        "/recommend - Personalized music recs\n"
        "/mood - Set mood for recs\n"
        "/link_spotify - Connect Spotify\n"
        "/create_playlist <code>[name]</code> - New private Spotify playlist\n"
        "/clear - Clear our chat history\n\n"
        "<b>Chat With Me!</b> Just talk, e.g.:\n"
        "- \"I'm feeling sad.\"\n- \"Play 'Shape of You'\"\n"
        "- \"Lyrics for Hotel California\"\n- Send YT link or voice message!\n\n"
        "Let the music flow! 🎵"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

async def download_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message_text = update.message.text if update.message and update.message.text else ""
    url_to_download = ""

    if context.args: 
        url_to_download = " ".join(context.args) 
    elif message_text: # Logic to extract URL if not from /download args
        url_match = re.search(r"(https?:\/\/(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11}))", message_text)
        if url_match:
            url_to_download = url_match.group(0) # Get the full matched URL
    
    if not url_to_download or not is_valid_youtube_url(url_to_download):
        await update.message.reply_text("❌ Invalid or missing YouTube URL. Use `/download <URL>` or send a valid link directly.")
        return

    user_id = update.effective_user.id
    if user_id in active_downloads:
        await update.message.reply_text("⚠️ One download at a time, please! Yours is in progress.")
        return

    active_downloads.add(user_id)
    status_msg = await update.message.reply_text("⏳ Starting download... hold tight!")

    try:
        await status_msg.edit_text("🔍 Fetching info & preparing download...")
        result = await asyncio.to_thread(download_youtube_audio_sync, url_to_download)
        
        if not result["success"]:
            await status_msg.edit_text(f"❌ Download failed: {result.get('error', 'Unknown error')}")
            return

        await status_msg.edit_text(f"✅ DL'd: <b>{result['title']}</b>\n⏳ Sending audio...", parse_mode=ParseMode.HTML)
        
        audio_path = result["audio_path"]
        if not audio_path or not os.path.exists(audio_path):
             logger.error(f"Audio path invalid or file missing after download: {audio_path}")
             await status_msg.edit_text(f"❌ Error: Downloaded file not found on server.")
             return

        with open(audio_path, 'rb') as audio_file:
            logger.info(f"Sending '{result['title']}' (user: {user_id}). Path: {audio_path}, Size: {os.path.getsize(audio_path)/(1024*1024):.2f}MB")
            await update.message.reply_audio(
                audio=audio_file, title=result["title"][:64], 
                performer=result["artist"][:64] if result.get("artist") else "Unknown", 
                caption=f"🎵 {result['title']}", duration=result.get('duration'),
            )
        try:
            os.remove(audio_path)
            logger.info(f"Temp file deleted: {audio_path}")
        except OSError as e: logger.error(f"Error deleting temp file {audio_path}: {e}")
        try: await status_msg.delete()
        except: pass # Message might have already been handled or deleted
    except (TimedOut, NetworkError) as net_err:
        logger.error(f"Net/TG API error during DL (user {user_id}, url: {url_to_download}): {net_err}")
        try: await status_msg.edit_text(f"❌ Network/Telegram error: {net_err}. Try again.")
        except: pass # status_msg might be gone
    except Exception as e:
        logger.error(f"Unexpected error in download_music (user {user_id}, url: {url_to_download}): {e}", exc_info=True)
        try: await status_msg.edit_text(f"❌ Unexpected error: {str(e)[:80]}.")
        except: pass 
    finally:
        active_downloads.discard(user_id)

async def create_playlist(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text("Name your playlist: `/create_playlist <Your Playlist Name>`")
        return
    playlist_name = " ".join(context.args)
    logger.info(f"User {user_id} creating Spotify playlist: '{playlist_name}'")
    access_token = await get_user_spotify_access_token(user_id)
    if not access_token:
        await update.message.reply_text("I need Spotify access. 😥 Link via /link_spotify.")
        return

    user_profile_url = "https://api.spotify.com/v1/me"
    headers_auth = {"Authorization": f"Bearer {access_token}"}
    spotify_user_id_api = None # Renamed variable
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.get(user_profile_url, headers=headers_auth) as response:
                response.raise_for_status()
                spotify_user_id_api = (await response.json()).get("id") # Use the new name
        if not spotify_user_id_api:
            logger.error(f"Could not get Spotify user ID for Telegram user {user_id}.")
            await update.message.reply_text("Sorry, couldn't get your Spotify profile ID.")
            return
    except aiohttp.ClientError as e:
        logger.error(f"API error fetching Spotify profile (user {user_id}): {e}")
        await update.message.reply_text("Issue fetching your Spotify profile. Try again.")
        return
    
    playlist_creation_url = f"https://api.spotify.com/v1/users/{spotify_user_id_api}/playlists"
    headers_create = {**headers_auth, "Content-Type": "application/json"}
    payload = {"name": playlist_name, "public": False, "description": f"Created by MelodyMind Bot on {datetime.now().strftime('%Y-%m-%d')}"}
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(playlist_creation_url, headers=headers_create, json=payload) as response:
                response.raise_for_status()
                playlist_data = await response.json()
                playlist_url = playlist_data.get("external_urls", {}).get("spotify", "#")
                logger.info(f"Playlist '{playlist_name}' created (user {user_id}). URL: {playlist_url}")
                await update.message.reply_text(
                    f"🎉 Playlist '<b>{playlist_name}</b>' created!\nView: {playlist_url}",
                    parse_mode=ParseMode.HTML, disable_web_page_preview=True)
    except aiohttp.ClientError as e:
        status, msg_detail = (getattr(e, 'status', 'N/A'), getattr(e, 'message', str(e)))
        logger.error(f"API error creating playlist '{playlist_name}' (user {user_id}): {status} - {msg_detail}")
        await update.message.reply_text(f"Oops! Failed to create playlist (Error {status}).")
    except Exception as e:
        logger.error(f"Unexpected error creating playlist (user {user_id}): {e}", exc_info=True)
        await update.message.reply_text("Unexpected error creating playlist.")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.voice: return
    user_id = update.effective_user.id
    logger.info(f"Voice message from user {user_id}")
    voice_file = await context.bot.get_file(update.message.voice.file_id)
    # Unique path for temp file
    temp_ogg_path = os.path.join(DOWNLOAD_DIR, f"voice_{user_id}_{update.message.message_id}_{datetime.now().timestamp()}.ogg")
    await voice_file.download_to_drive(temp_ogg_path)
    
    recognizer = sr.Recognizer()
    transcribed_text = None
    try:
        def _transcribe_sync_inner(): # To pass to to_thread
            with sr.AudioFile(temp_ogg_path) as source: audio_data = recognizer.record(source)
            try: return recognizer.recognize_google(audio_data)
            except sr.UnknownValueError: logger.warning(f"SR: Google UnknownValue (user {user_id}) for file {temp_ogg_path}")
            except sr.RequestError as req_e: logger.error(f"SR: Google RequestError (user {user_id}); {req_e}"); return "ERROR_REQUEST"
            return None # Explicitly return None if not understood but no error
        
        transcribed_text = await asyncio.to_thread(_transcribe_sync_inner)

        if transcribed_text == "ERROR_REQUEST":
            await update.message.reply_text("Voice recognition service error. Please type or try later.")
        elif transcribed_text: # Transcribed successfully
            logger.info(f"Voice (user {user_id}) transcribed: '{transcribed_text}'")
            await update.message.reply_text(f"🎤 Heard: \"<i>{transcribed_text}</i>\"\nProcessing...", parse_mode=ParseMode.HTML)
            # Store original message if needed by other parts of enhanced_handle_message
            context.user_data['_voice_original_message'] = update.message 
            # Create a new Message object that looks like a text message
            fake_text_message = update.message._replace(text=transcribed_text, voice=None) # Remove 'voice' attr to avoid re-processing as voice
            # Create a new Update object with this fake message
            fake_text_update = Update(update.update_id, message=fake_text_message)
            await enhanced_handle_message(fake_text_update, context) # Process as if user typed it
        else: # Understood nothing or error already handled by _transcribe
            await update.message.reply_text("Couldn't quite catch that. Try speaking clearly, or type? 😊")
    except Exception as e:
        logger.error(f"Error processing voice (user {user_id}): {e}", exc_info=True)
        await update.message.reply_text("Oops! Error with voice message. Try again.")
    finally:
        if os.path.exists(temp_ogg_path):
            try: os.remove(temp_ogg_path)
            except OSError as e_del: logger.error(f"Error deleting temp voice file {temp_ogg_path}: {e_del}")

async def link_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not all([SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI]):
        await update.message.reply_text("Sorry, Spotify linking not configured by admin. 😥")
        return ConversationHandler.END
    if SPOTIFY_REDIRECT_URI == "https://your-callback-url.com": # Default placeholder value
         await update.message.reply_text("⚠️ Spotify redirect URI is a placeholder. Manual code copy from URL parameters will likely be needed after authorizing on Spotify.")
    
    user_id = update.effective_user.id
    scopes = "user-read-recently-played user-top-read playlist-modify-private" # Scopes for Spotify
    auth_url = (f"https://accounts.spotify.com/authorize?client_id={SPOTIFY_CLIENT_ID}"
                f"&response_type=code&redirect_uri={SPOTIFY_REDIRECT_URI}"
                f"&scope={scopes.replace(' ', '%20')}&state={user_id}") # State for verification
                
    keyboard_buttons = [ # Keyboard for linking
        [InlineKeyboardButton("🔗 Link My Spotify Account", url=auth_url)],
        [InlineKeyboardButton("Cancel Linking", callback_data=CB_CANCEL_SPOTIFY)]]
    
    reply_text_md = (
        "Let's link Spotify for personalized music! 🎵\n\n"
        "1. Click the button below to go to Spotify.\n"
        "2. Authorize this bot. Spotify will then redirect you.\n"
        "3. From the **redirected page's URL**, copy the `code` value.\n"
        "   (It looks like `YOUR_REDIRECT_URI/?code=A_VERY_LONG_CODE_STRING&state=...` - you need the `A_VERY_LONG_CODE_STRING` part)\n"
        "4. Send **only that code string** back to me here.\n\n"
        "_If you encounter issues, ensure the bot administrator has correctly configured the Redirect URI in Spotify's Developer Dashboard._")
        
    await update.message.reply_text(reply_text_md,
        reply_markup=InlineKeyboardMarkup(keyboard_buttons), parse_mode=ParseMode.MARKDOWN)
    return SPOTIFY_CODE # Next state in conversation

async def spotify_code_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    message_text = update.message.text if update.message and update.message.text else ""
    
    # Determine code_to_use
    code_to_use: Optional[str] = None
    if message_text.startswith('/spotify_code') and context.args: 
        code_to_use = context.args[0]
    elif not message_text.startswith('/'): # Assume direct paste of code
        code_to_use = message_text.strip()
    
    # Basic validation of the code received (Spotify codes are long)
    if not code_to_use or len(code_to_use) < 50: # Arbitrary minimum length
        await update.message.reply_text(
            "That Spotify code seems too short or is missing. Please paste the full code you copied, or use `/spotify_code YOUR_CODE`.")
        return SPOTIFY_CODE # Stay in this state to allow re-entry of code
        
    status_msg = await update.message.reply_text("⏳ Validating your Spotify authorization code...")
    token_data = await get_user_spotify_token(user_id, code_to_use)
    
    if not token_data or not token_data.get("access_token"):
        await status_msg.edit_text(
            "❌ Failed to link Spotify. The code might be invalid, expired, or there's a configuration issue (e.g., incorrect Redirect URI). "
            "Please try /link_spotify again. Make sure you copy the `code` parameter correctly from the redirect URL.")
        return SPOTIFY_CODE # Stay, allow user to try with a new code if they get one
        
    # Ensure user_contexts structure exists
    user_contexts.setdefault(user_id, {}).setdefault("spotify", {}) # type: ignore
    
    # Store encrypted tokens
    user_contexts[user_id]["spotify"].update({ # type: ignore
        "access_token": cipher.encrypt(token_data["access_token"].encode()),
        "refresh_token": cipher.encrypt(token_data["refresh_token"].encode()), # refresh_token MUST exist after auth_code grant
        "expires_at": token_data["expires_at"]
    })
    logger.info(f"Spotify account successfully linked for user {user_id}.")

    # Optionally, fetch some initial data to confirm & personalize
    rp_confirm = await get_user_spotify_data(user_id, "player/recently-played", params={"limit": 1})
    rp_info_text = ""
    if rp_confirm and len(rp_confirm) > 0 and rp_confirm[0].get("track"):
        try:
            first_rp_artist_name = rp_confirm[0]['track']['artists'][0]['name']
            rp_info_text = f" I see you recently enjoyed some music by {first_rp_artist_name}!"
        except (KeyError, IndexError, TypeError): pass # Silently ignore if data structure is off
            
    await status_msg.edit_text(
        f"✅ Spotify linked successfully! 🎉{rp_info_text} You can now get personalized recommendations with /recommend!"
    )
    return ConversationHandler.END

async def spotify_code_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
    if not context.args:
        await update.message.reply_text("Provide Spotify code after command: `/spotify_code YOUR_CODE_HERE`")
        # If this command is used outside an active SPOTIFY_CODE state, its return doesn't directly affect states.
        # If it *is* called when the SPOTIFY_CODE state handler (for this command name) is active,
        # then returning SPOTIFY_CODE here would be redundant, as spotify_code_handler is preferred by PTB state matching.
        # Let it be None for global call, let spotify_code_handler manage state returns when in conv.
        return None 
    return await spotify_code_handler(update, context) # Delegate to main handler

async def cancel_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Spotify linking cancelled. You can always try again using /link_spotify. 👍")
    return ConversationHandler.END

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("What song to search? Ex: `/search Shape of You`")
        return
    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 YT Search: '<i>{query}</i>'...", parse_mode=ParseMode.HTML)
    results = await asyncio.to_thread(search_youtube_sync, query, max_results=5)
    try: await status_msg.delete() # Clean up "Searching..." message
    except Exception: pass # If already gone or other issue
    await send_search_results(update, query, results)

async def auto_download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Song to auto-download? Ex: `/autodownload Believer Imagine Dragons`")
        return
    query_str = " ".join(context.args)
    await auto_download_first_result(update, context, query_str)

async def get_lyrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Song for lyrics? Examples:\n`/lyrics Bohemian Rhapsody` or `/lyrics Queen - Song Title`")
        return
    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Lyrics search: \"<i>{query}</i>\"...", parse_mode=ParseMode.HTML)
    try:
        artist, song_title = (None, query) # Default: query is song title
        # Basic parsing for "Artist - Song" or "Song by Artist"
        if " - " in query: 
            parts = query.split(" - ", 1)
            artist, song_title = parts[0].strip(), parts[1].strip()
        elif " by " in query.lower(): # Case-insensitive "by"
            match_by = re.search(r'^(.*?)\s+by\s+(.*?)$', query, re.IGNORECASE)
            if match_by:
                song_title, artist = match_by.group(1).strip(), match_by.group(2).strip()
        
        logger.info(f"Lyrics parsed: song='{song_title}', artist='{artist}'")
        lyrics_html = await asyncio.to_thread(get_lyrics_sync, song_title, artist)
        
        # Send lyrics, handling long messages by splitting
        MAX_TG_MSG_LEN = 4080 # Telegram's limit is 4096, give buffer for HTML and notes
        if len(lyrics_html) > MAX_TG_MSG_LEN:
            first_chunk_text = lyrics_html[:MAX_TG_MSG_LEN]
            # Try to find a clean break point (double newline)
            clean_cut_point = first_chunk_text.rfind('\n\n') 
            if clean_cut_point == -1 or clean_cut_point < MAX_TG_MSG_LEN - 1000 : # If no good double newline, try single, far back
                clean_cut_point = first_chunk_text.rfind('\n', 0, MAX_TG_MSG_LEN - 500) 
            if clean_cut_point == -1 or clean_cut_point < MAX_TG_MSG_LEN - 1500 : # Last resort: rough cut
                 clean_cut_point = MAX_TG_MSG_LEN - 100 
            
            await status_msg.edit_text(f"{lyrics_html[:clean_cut_point]}\n\n<small>(Lyrics continued below...)</small>", parse_mode=ParseMode.HTML)
            
            remaining_lyrics_text = lyrics_html[clean_cut_point:]
            while remaining_lyrics_text:
                current_part_to_send = remaining_lyrics_text[:MAX_TG_MSG_LEN]
                remaining_lyrics_text = remaining_lyrics_text[MAX_TG_MSG_LEN:]
                
                # For simplicity, don't try to fine-tune cuts for subsequent parts too much, just send.
                if current_part_to_send.strip(): # Only send if there's content
                    await update.message.reply_text(current_part_to_send + ("\n<small>(...more lyrics)</small>" if remaining_lyrics_text.strip() else ""), 
                                                    parse_mode=ParseMode.HTML)
        else: # Lyrics fit in one message
            await status_msg.edit_text(lyrics_html, parse_mode=ParseMode.HTML)

    except Exception as e: # Catch any other unexpected errors during processing
        logger.error(f"Error in get_lyrics_command (query '{query}'): {e}", exc_info=True)
        await status_msg.edit_text("Sorry, an unexpected hiccup occurred while fetching lyrics. 😕 Please try again.")

async def recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await smart_recommend_music(update, context) # Main recommendation logic

async def provide_generic_recommendations(update: Update, mood: str, chat_id_override: Optional[int] = None) -> None:
    logger.info(f"Providing generic recommendations for mood: {mood}")
    target_chat_id = chat_id_override if chat_id_override else update.effective_chat.id # type: ignore

    mood_recommendations_map = { # Using a map for clarity
        "happy": ["Uptown Funk - Mark Ronson", "Happy - Pharrell Williams", "Walking on Sunshine - Katrina & The Waves"],
        "sad": ["Someone Like You - Adele", "Hallelujah - Leonard Cohen (Jeff Buckley version preferably)", "Fix You - Coldplay"],
        "energetic": ["Don't Stop Me Now - Queen", "Thunderstruck - AC/DC", "Can't Stop the Feeling! - Justin Timberlake"],
        "relaxed": ["Weightless - Marconi Union", "Clair de Lune - Claude Debussy", "Orinoco Flow - Enya"],
        "focused": ["The Four Seasons - Vivaldi (Antonio Vivaldi)", "Time - Hans Zimmer", "Ambient 1: Music for Airports - Brian Eno"],
        "nostalgic": ["Bohemian Rhapsody - Queen", "Yesterday - The Beatles", "Wonderwall - Oasis"],
        "neutral": ["Three Little Birds - Bob Marley", "Here Comes The Sun - The Beatles", "What a Wonderful World - Louis Armstrong"]
    }
    chosen_mood_key = mood.lower()
    if chosen_mood_key not in mood_recommendations_map:
        logger.warning(f"Generic mood '{mood}' not in recommendation list, defaulting to neutral.")
        chosen_mood_key = "neutral" 
        
    recommendations_list = mood_recommendations_map.get(chosen_mood_key, mood_recommendations_map["neutral"]) 
    response_text_html = f"🎵 Some general **{mood.capitalize()}** vibes for you:\n\n"
    for i, track_info in enumerate(recommendations_list, 1):
        response_text_html += f"{i}. {track_info}\n"
    response_text_html += "\n💡 <i>You can ask me to search or download any of these!</i>"
    
    await context.bot.send_message(chat_id=target_chat_id, text=response_text_html, parse_mode=ParseMode.HTML)

async def set_mood(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    user_contexts.setdefault(user.id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    # Keyboard layout
    keyboard_layout_mood = [
        [InlineKeyboardButton("Happy 😊", callback_data=f"{CB_MOOD_PREFIX}happy"), InlineKeyboardButton("Sad 😢", callback_data=f"{CB_MOOD_PREFIX}sad")],
        [InlineKeyboardButton("Energetic 💪", callback_data=f"{CB_MOOD_PREFIX}energetic"), InlineKeyboardButton("Relaxed 😌", callback_data=f"{CB_MOOD_PREFIX}relaxed")],
        [InlineKeyboardButton("Focused 🧠", callback_data=f"{CB_MOOD_PREFIX}focused"), InlineKeyboardButton("Nostalgic 🕰️", callback_data=f"{CB_MOOD_PREFIX}nostalgic")],
        [InlineKeyboardButton("Neutral / Other", callback_data=f"{CB_MOOD_PREFIX}neutral")]]
    
    current_mood_val = user_contexts[user.id].get("mood")
    prompt_text_md = f"Hi {user.first_name}! "
    if current_mood_val and current_mood_val != "neutral": # Don't mention if it's neutral or not set
        prompt_text_md += f"Your current mood is set to **{current_mood_val}**. Want to change it or how are you feeling right now?"
    else:
        prompt_text_md += "How are you feeling today?"
    
    reply_markup_inline = InlineKeyboardMarkup(keyboard_layout_mood)
    if update.callback_query: # If this function was called from a button press (e.g. /recommend when mood not set)
        await update.callback_query.edit_message_text(prompt_text_md, reply_markup=reply_markup_inline, parse_mode=ParseMode.MARKDOWN)
    else: # If called from /mood command directly
        await update.message.reply_text(prompt_text_md, reply_markup=reply_markup_inline, parse_mode=ParseMode.MARKDOWN)
    return MOOD # Next state in ConversationHandler

async def send_search_results(update: Update, query: str, results: List[Dict]) -> None:
    # target_chat_id = update.effective_chat.id if update.effective_chat else None
    # if not target_chat_id and update.callback_query:
    #     target_chat_id = update.callback_query.message.chat_id
    # if not target_chat_id:
    #     logger.error("send_search_results: Could not determine target_chat_id.")
    #     return
    
    if not results:
        await update.message.reply_text(f"😕 Sorry, no YouTube results for '<i>{query}</i>'. Try different keywords?", parse_mode=ParseMode.HTML)
        return

    keyboard_rows_list = []
    response_header_text = f"🔎 YouTube results for '<i>{query}</i>':\n\n"
    valid_results_display_count = 0
    
    for result_item in results[:5]: # Max 5 results
        if not result_item.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', result_item['id']):
            logger.warning(f"send_search_results: Skipping invalid YouTube result ID: {result_item.get('id', 'N/A')}")
            continue
        valid_results_display_count +=1

        duration_val = result_item.get('duration', 0)
        duration_display_str = ""
        if duration_val and isinstance(duration_val, (int, float)) and duration_val > 0:
            try:
                minutes, seconds = divmod(int(duration_val), 60)
                duration_display_str = f" [{minutes}:{seconds:02d}]"
            except TypeError: pass # Should not happen with type check
        
        title_text = result_item.get('title', 'Unknown Title')
        # Truncate button text if too long for Telegram button (max ~64 bytes, play safe)
        button_title_display = (title_text[:33] + "...") if len(title_text) > 36 else title_text
        button_text_content = f"[{valid_results_display_count}] {button_title_display}{duration_display_str}"
        
        response_header_text += f"{valid_results_display_count}. <b>{title_text}</b> by <i>{result_item.get('uploader', 'N/A')}</i>{duration_display_str}\n"
        keyboard_rows_list.append([InlineKeyboardButton(button_text_content, callback_data=f"{CB_DOWNLOAD_PREFIX}{result_item['id']}")])

    if not keyboard_rows_list: # If all results had invalid IDs or were skipped
        await update.message.reply_text(f"😕 I found some YouTube results for '<i>{query}</i>', but had issues creating download options for them. Sorry about that!", parse_mode=ParseMode.HTML)
        return

    keyboard_rows_list.append([InlineKeyboardButton("Cancel Search", callback_data=CB_CANCEL_SEARCH)])
    reply_markup_obj = InlineKeyboardMarkup(keyboard_rows_list)
    
    final_response_text = response_header_text + "\nClick a song to download its audio:"
    await update.message.reply_text(final_response_text, reply_markup=reply_markup_obj, parse_mode=ParseMode.HTML)

async def auto_download_first_result(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str, original_message_id_to_edit: Optional[int] = None) -> None:
    user_id = update.effective_user.id
    chat_id_val = update.effective_chat.id if update.effective_chat else None
    if not chat_id_val:
        logger.error("auto_download_first_result: could not determine chat_id")
        return # Cannot proceed without chat_id

    if user_id in active_downloads:
        text_to_send = "Hold on! Another download is active. 😊"
        if original_message_id_to_edit:
            await context.bot.edit_message_text(text=text_to_send, chat_id=chat_id_val, message_id=original_message_id_to_edit, reply_markup=None)
        else:
            await update.message.reply_text(text_to_send) # type: ignore
        return

    active_downloads.add(user_id)
    status_msg_obj = None # Will hold the message object that is being edited or was replied to
    try:
        status_msg_text_html = f"🔍 Looking for '<i>{query}</i>' to download..."
        if original_message_id_to_edit:
            status_msg_obj = await context.bot.edit_message_text(
                text=status_msg_text_html, parse_mode=ParseMode.HTML, 
                chat_id=chat_id_val, message_id=original_message_id_to_edit, reply_markup=None)
        else:
            status_msg_obj = await update.message.reply_text(status_msg_text_html, parse_mode=ParseMode.HTML) # type: ignore

        # If query is already a URL, download_youtube_audio_sync will handle it directly
        # If query is a search term, search_youtube_sync then download_youtube_audio_sync for first result's URL
        video_url_to_process = query # Default: query is URL
        video_title_for_status = "selected track"

        if not is_valid_youtube_url(query): # If it's not a URL, it's a search term
            results = await asyncio.to_thread(search_youtube_sync, query, max_results=1)
            if not results or not results[0].get('id') or not is_valid_youtube_url(results[0].get('url','')):
                await status_msg_obj.edit_text(f"❌ Couldn't find a downloadable track for '<i>{query}</i>'. Try `/search {query}` for options?", parse_mode=ParseMode.HTML)
                return
            video_url_to_process = results[0]["url"]
            video_title_for_status = results[0].get("title", "this track")
            await status_msg_obj.edit_text(f"✅ Found: <b>{video_title_for_status}</b>.\n⏳ Downloading... (can take a moment!)", parse_mode=ParseMode.HTML)
        else: # Query was already a URL
            await status_msg_obj.edit_text(f"⏳ Downloading direct URL... (can take a moment!)", parse_mode=ParseMode.HTML)


        download_result_dict = await asyncio.to_thread(download_youtube_audio_sync, video_url_to_process)
        
        if not download_result_dict["success"]:
            final_video_title = download_result_dict.get('title', video_title_for_status) # Use title from download if available
            await status_msg_obj.edit_text(f"❌ Download failed for <b>{final_video_title}</b>: {download_result_dict.get('error', 'Unknown error')}", parse_mode=ParseMode.HTML)
            return
        
        await status_msg_obj.edit_text(f"✅ Downloaded: <b>{download_result_dict['title']}</b>.\n✅ Sending the audio file now...", parse_mode=ParseMode.HTML)
        
        audio_file_path = download_result_dict["audio_path"]
        with open(audio_file_path, 'rb') as audio_rb:
            logger.info(f"Auto-DL: Sending '{download_result_dict['title']}' (user {user_id}). Path: {audio_file_path}")
            # Send as a new message to the chat
            await context.bot.send_audio(
                chat_id=chat_id_val, audio=audio_rb,
                title=download_result_dict["title"][:64], 
                performer=download_result_dict["artist"][:64] if download_result_dict.get("artist") else "Unknown Artist",
                caption=f"🎵 Here's: {download_result_dict['title']}", 
                duration=download_result_dict.get('duration'))
                
        if os.path.exists(audio_file_path):
            try: os.remove(audio_file_path)
            except OSError as e_del_auto: logger.error(f"Error deleting temp file (auto-DL) {audio_file_path}: {e_del_auto}")
        
        try: await status_msg_obj.delete() # Clean up the "Downloading..." status message
        except: pass # If already deleted or other error, ignore for cleanup
            
    except (TimedOut, NetworkError) as net_err_auto:
        logger.error(f"Net/API error (auto-DL user {user_id}, query '{query}'): {net_err_auto}")
        if status_msg_obj: try: await status_msg_obj.edit_text(f"❌ Network issue with '<i>{query}</i>'. Please try again.", parse_mode=ParseMode.HTML)
        except: pass
    except Exception as e_auto:
        logger.error(f"Unexpected error in auto_download_first_result (user {user_id}, query '{query}'): {e_auto}", exc_info=True)
        if status_msg_obj: try: await status_msg_obj.edit_text(f"❌ An unexpected error occurred processing '<i>{query}</i>'. My apologies!", parse_mode=ParseMode.HTML)
        except: pass
    finally: 
        active_downloads.discard(user_id)

async def send_audio_via_bot(bot, chat_id, audio_path, title, performer, caption, duration):
    # Corrected variable name usage for opening file
    with open(audio_path, 'rb') as audio_file_object: 
        await bot.send_audio(chat_id=chat_id, audio=audio_file_object, title=title[:64],
            performer=performer[:64] if performer else "Unknown", caption=caption, duration=duration)

async def enhanced_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
    query = update.callback_query
    await query.answer() # Acknowledge button press immediately
    
    data: str = query.data if query.data else "" # Ensure data is not None
    user_id: int = query.from_user.id
    # Ensure user context exists
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    
    logger.debug(f"Button callback received: '{data}' for user {user_id}")

    # Mood setting flow
    if data.startswith(CB_MOOD_PREFIX):
        mood_selected = data[len(CB_MOOD_PREFIX):]
        user_contexts[user_id]["mood"] = mood_selected
        logger.info(f"User {user_id} selected mood: {mood_selected}")

        # Create keyboard for genre preference, expanded for readability
        genre_keyboard_rows = [
            [InlineKeyboardButton("Pop", callback_data=f"{CB_PREFERENCE_PREFIX}pop"),
             InlineKeyboardButton("Rock", callback_data=f"{CB_PREFERENCE_PREFIX}rock"),
             InlineKeyboardButton("Hip-Hop", callback_data=f"{CB_PREFERENCE_PREFIX}hiphop")],
            [InlineKeyboardButton("Electronic", callback_data=f"{CB_PREFERENCE_PREFIX}electronic"),
             InlineKeyboardButton("Classical", callback_data=f"{CB_PREFERENCE_PREFIX}classical"),
             InlineKeyboardButton("Jazz", callback_data=f"{CB_PREFERENCE_PREFIX}jazz")],
            [InlineKeyboardButton("Folk/Acoustic", callback_data=f"{CB_PREFERENCE_PREFIX}folk"),
             InlineKeyboardButton("R&B/Soul", callback_data=f"{CB_PREFERENCE_PREFIX}rnb"),
             InlineKeyboardButton("Any / Surprise Me!", callback_data=f"{CB_PREFERENCE_PREFIX}any")],
            [InlineKeyboardButton("Skip Genre Selection", callback_data=f"{CB_PREFERENCE_PREFIX}skip")],
        ]
        reply_markup_genres = InlineKeyboardMarkup(genre_keyboard_rows)
        await query.edit_message_text(
            f"Got it, {query.from_user.first_name}! You're feeling {mood_selected}. 🎶\nAny particular genre you're in the mood for?",
            reply_markup=reply_markup_genres)
        return PREFERENCE # Next state in mood conversation

    # Preference setting flow
    elif data.startswith(CB_PREFERENCE_PREFIX):
        preference_selected = data[len(CB_PREFERENCE_PREFIX):]
        response_message_text = ""
        if preference_selected in ["skip", "any"]: # Handle skip or any preference
            user_contexts[user_id]["preferences"] = [] # Clear preferences or mark as "any"
            response_message_text = "Alright! I'll keep that in mind for recommendations."
        else:
            user_contexts[user_id]["preferences"] = [preference_selected] # Set the selected preference
            response_message_text = f"Great choice! {preference_selected.capitalize()} it is. "
        logger.info(f"User {user_id} selected preference: {preference_selected}")
        
        response_message_text += " You can try:\n➡️ `/recommend` for music suggestions\n" \
                                 "➡️ `/search [song]` or `/autodownload [song]`\n" \
                                 "➡️ Or just continue chatting with me!"
        await query.edit_message_text(response_message_text)
        return ConversationHandler.END # End the mood/preference conversation

    # Download from search result or recommendation (CB_DOWNLOAD_PREFIX)
    elif data.startswith(CB_DOWNLOAD_PREFIX):
        video_id_to_download = data[len(CB_DOWNLOAD_PREFIX):]
        if not re.match(r'^[0-9A-Za-z_-]{11}$', video_id_to_download): # Validate video ID format
            logger.error(f"Invalid YouTube video ID from button CB_DOWNLOAD_PREFIX: '{video_id_to_download}'")
            await query.edit_message_text("❌ Oops! That video ID seems invalid. Please try searching again.", reply_markup=None)
            return None
        
        youtube_url_direct = f"https://www.youtube.com/watch?v={video_id_to_download}"
        
        # Directly call auto_download_first_result which now handles URLs properly, pass message_id to edit
        await auto_download_first_result(update, context, query=youtube_url_direct, original_message_id_to_edit=query.message.message_id)
        return None # No further state transition from this button type typically

    # Download from AI suggestion confirmation (CB_AUTO_DOWNLOAD_PREFIX)
    elif data.startswith(CB_AUTO_DOWNLOAD_PREFIX):
        # Data here could be a video ID or the original search query string if AI didn't pinpoint an ID
        content_after_prefix = data[len(CB_AUTO_DOWNLOAD_PREFIX):]
        
        query_for_auto_dl = ""
        if re.match(r'^[0-9A-Za-z_-]{11}$', content_after_prefix): # It's a video ID
            query_for_auto_dl = f"https://www.youtube.com/watch?v={content_after_prefix}"
        else: # It's likely the original search query string stored in callback
            query_for_auto_dl = content_after_prefix
            
        # Pass the message_id of the message with the button, so auto_download_first_result can edit it.
        await auto_download_first_result(update, context, query=query_for_auto_dl, original_message_id_to_edit=query.message.message_id)
        return None

    # Show more options after AI suggestion (CB_SHOW_OPTIONS_PREFIX)
    elif data.startswith(CB_SHOW_OPTIONS_PREFIX):
        original_search_query = data[len(CB_SHOW_OPTIONS_PREFIX):]
        if not original_search_query:
            await query.edit_message_text("Cannot show options, original query information is missing.", reply_markup=None)
            return None
        
        # Edit the current message to indicate action, then delete it and send new search results
        await query.edit_message_text(f"🔍 Okay, showing more YouTube options for '<i>{original_search_query}</i>'...", 
                                      parse_mode=ParseMode.HTML, reply_markup=None)
        
        youtube_results = await asyncio.to_thread(search_youtube_sync, original_search_query, max_results=5)
        
        try: # Attempt to delete the previous message (that had "show options" button)
            await query.message.delete() 
        except Exception as e_del_options:
            logger.warning(f"Could not delete previous message before showing CB_SHOW_OPTIONS results: {e_del_options}")

        # Send search results as a new message. Need to use a message object for send_search_results.
        # update.callback_query.message is the message the button was attached to.
        # It's okay to use it as the "update" for send_search_results, as it needs `effective_chat.id` and `reply_text`.
        await send_search_results(Update(query.update_id, message=query.message), original_search_query, youtube_results)
        return None

    # Cancel search or Spotify linking
    elif data == CB_CANCEL_SEARCH:
        await query.edit_message_text("❌ Search/Action cancelled. Feel free to try another command or chat!", reply_markup=None)
        return None
    elif data == CB_CANCEL_SPOTIFY: # For Spotify ConversationHandler
        await query.edit_message_text("Spotify linking process cancelled. You can use /link_spotify anytime to try again.", reply_markup=None)
        return ConversationHandler.END # Important for ending the conversation properly
    
    logger.warning(f"Unhandled callback data received: {data} from user {user_id}")
    return None # Default if no specific callback pattern matched


async def enhanced_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text: return # Ignore non-text or empty messages
    user_id, text_received = update.effective_user.id, update.message.text.strip()
    logger.debug(f"Msg from user {user_id}: '{text_received[:80]}'") # Log truncated message
    # Ensure user context base structure
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})

    # 1. Handle direct YouTube URLs for download
    if is_valid_youtube_url(text_received):
        logger.info(f"User {user_id} sent YouTube URL directly: {text_received}")
        # Call download_music, it will parse the URL from update.message.text
        await download_music(update, context)
        return

    # 2. Subtle passive mood detection from message content
    if len(text_received.split()) > 2: # Only on slightly longer messages
        newly_detected_mood = await detect_mood_from_text(user_id, text_received)
        # Update mood if it's new, not neutral, and different from current (or if current is neutral)
        current_user_mood = user_contexts[user_id].get("mood")
        if newly_detected_mood and newly_detected_mood != "neutral" and \
           (newly_detected_mood != current_user_mood or current_user_mood == "neutral"):
            user_contexts[user_id]["mood"] = newly_detected_mood
            logger.debug(f"Passive mood update for user {user_id} to: {newly_detected_mood} based on: '{text_received[:30]}'")

    # 3. AI-driven music request detection
    ai_music_analysis = await is_music_request(user_id, text_received)
    if ai_music_analysis.get("is_music_request") and ai_music_analysis.get("song_query"):
        song_query_from_ai = ai_music_analysis["song_query"]
        status_message_ai = await update.message.reply_text(f"🎵 You're looking for '<i>{song_query_from_ai}</i>'? Let me check that for you...", parse_mode=ParseMode.HTML)
        
        search_results_yt = await asyncio.to_thread(search_youtube_sync, song_query_from_ai, max_results=1) # Get top result
        
        if search_results_yt and search_results_yt[0].get('id') and re.match(r'^[0-9A-Za-z_-]{11}$', search_results_yt[0]['id']):
            top_youtube_result = search_results_yt[0]
            # Confirmation keyboard
            confirmation_keyboard_layout = [
                [InlineKeyboardButton(f"✅ Yes, download '{top_youtube_result['title'][:18]}...'", 
                                      callback_data=f"{CB_AUTO_DOWNLOAD_PREFIX}{top_youtube_result['id']}")],
                [InlineKeyboardButton("👀 Show me more options", 
                                      callback_data=f"{CB_SHOW_OPTIONS_PREFIX}{song_query_from_ai}")], # Pass original query string for broader search
                [InlineKeyboardButton("❌ No, that's not it / Cancel", 
                                      callback_data=CB_CANCEL_SEARCH)]]
            reply_markup_confirm = InlineKeyboardMarkup(confirmation_keyboard_layout)
            uploader_name = top_youtube_result.get('uploader', 'Unknown Artist')
            await status_message_ai.edit_text(
                f"I found: <b>{top_youtube_result['title']}</b> by <i>{uploader_name}</i>.\n\nWould you like me to download this for you, or show more search options?",
                reply_markup=reply_markup_confirm, parse_mode=ParseMode.HTML)
        else: # No clear top result or invalid ID
            await status_message_ai.edit_text(f"😕 Sorry, I couldn't immediately find a specific track for '<i>{song_query_from_ai}</i>' on YouTube. You could try being more specific or use the `/search {song_query_from_ai}` command for more results.", parse_mode=ParseMode.HTML)
        return # Handled as a music request

    # 4. Heuristic lyrics request detection (e.g., "lyrics for Bohemian Rhapsody")
    # More precise keywords to avoid false positives from general chat.
    precise_lyrics_keywords = ["lyrics for", "lyrics to", "what are the lyrics to", "find lyrics for", "get lyrics for"]
    text_lower_case = text_received.lower()
    lyrics_search_query = None
    for keyword_phrase in precise_lyrics_keywords:
        if text_lower_case.startswith(keyword_phrase):
            # Extract song title after keyword phrase
            potential_query = text_received[len(keyword_phrase):].strip()
            if potential_query: # Make sure there's something after the keyword
                lyrics_search_query = potential_query
                logger.info(f"Heuristic lyrics request detected: '{lyrics_search_query}' for user {user_id}")
                break 
    if lyrics_search_query:
        # Mock context.args for get_lyrics_command (which expects args)
        # The command itself will parse "Artist - Song" or "Song by Artist" from this string
        mock_args_for_lyrics = lyrics_search_query.split() # Simplistic split, command handles better
        await get_lyrics_command(update, ContextTypes.DEFAULT_TYPE(application=context.application, chat_id=user_id, user_id=user_id, bot=context.bot, args=mock_args_for_lyrics))
        return # Handled as a lyrics request

    # 5. Fallback to general AI conversational response
    await asyncio.sleep(0.1) # Tiny cosmetic delay before "thinking"
    thinking_message = await update.message.reply_text("<i>MelodyMind is thinking...</i> 🎶", parse_mode=ParseMode.HTML)
    try:
        ai_chat_response = await generate_chat_response(user_id, text_received)
        await thinking_message.edit_text(ai_chat_response) # Edit the "thinking..." message with the actual response
    except (TimedOut, NetworkError) as net_err_chat:
        logger.error(f"Network error during AI chat response generation for user {user_id}: {net_err_chat}")
        await thinking_message.edit_text("Sorry, I'm having a bit of trouble connecting right now. Could you try saying that again in a moment?")
    except Exception as e_chat:
        logger.error(f"Error generating AI chat response for user {user_id}: {e_chat}", exc_info=True)
        await thinking_message.edit_text("I seem to be a bit tangled up at the moment! 😅 Let's try that conversation again later, or you can use one of my commands like /help.")


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_contexts.get(user_id, {}).get("conversation_history"): # Check if history exists and is not empty
        user_contexts[user_id]["conversation_history"] = []
        logger.info(f"Cleared conversation history for user {user_id}")
        await update.message.reply_text("✅ Our chat history has been cleared.")
    else: 
        await update.message.reply_text("You don't have any conversation history with me to clear yet! 😊")

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # Check if update is from callback query or message for appropriate reply
    if update.callback_query:
        await update.callback_query.edit_message_text("Okay, action cancelled. Feel free to use commands or chat anytime! 👍", reply_markup=None)
    elif update.message:
        await update.message.reply_text("Okay, action cancelled. Feel free to use commands or chat anytime! 👍")
    return ConversationHandler.END

async def analyze_conversation(user_id: int) -> Dict[str, Any]:
    default_analysis = {"genres": user_contexts.get(user_id, {}).get("preferences", []), 
                        "artists": [], 
                        "mood": user_contexts.get(user_id, {}).get("mood")}
    if not client: return default_analysis # AI is off
    
    # Ensure base structure for context
    user_ctx = user_contexts.get(user_id, {})
    user_ctx.setdefault("preferences", [])
    user_ctx.setdefault("conversation_history", [])
    spotify_ctx = user_ctx.setdefault("spotify", {})
    spotify_ctx.setdefault("recently_played", [])
    spotify_ctx.setdefault("top_tracks", [])

    # Require some data to proceed with AI analysis
    if len(user_ctx["conversation_history"]) < 1 and not spotify_ctx["recently_played"] and not spotify_ctx["top_tracks"]:
        logger.info(f"AI Conversation Analysis: Insufficient data for user {user_id}.")
        return default_analysis
        
    logger.info(f"Performing AI conversation analysis for user {user_id}")
    try:
        # Create a concise summary of recent conversation
        history_summary_text = "\n".join([f"{msg['role']}: {msg['content'][:70]}" for msg in user_ctx["conversation_history"][-5:]]) # Last ~2-3 exchanges
        
        # Create a concise summary of Spotify data
        spotify_summary_parts_list = []
        if spotify_ctx["recently_played"]: 
            try: spotify_summary_parts_list.append("Recent listens: " + ", ".join([f"'{item['track']['name']}' by {item['track']['artists'][0]['name']}" for item in spotify_ctx["recently_played"][:2] if item.get("track")]))
            except: pass # Ignore malformed Spotify data
        if spotify_ctx["top_tracks"]:
            try: spotify_summary_parts_list.append("Top tracks include: " + ", ".join([f"'{item['name']}' by {item['artists'][0]['name']}" for item in spotify_ctx["top_tracks"][:2] if item.get("artists")]))
            except: pass
        spotify_data_summary = ". ".join(filter(None, spotify_summary_parts_list)) # Filter out empty strings if no data for a part
        
        # Skip AI if still not enough meaningful summary
        if not history_summary_text and not spotify_data_summary:
             logger.info(f"AI Analysis: Not enough summary content from text/spotify for user {user_id}")
             return default_analysis

        # Construct prompt for AI
        prompt_to_ai_user_content = (
            f"Conversation Summary (last few messages):\n{history_summary_text}\n\n"
            f"User's Spotify Data Summary (if available):\n{spotify_data_summary}\n\n"
            f"User's explicitly set mood (if any): {user_ctx.get('mood', 'Not set')}\n"
            f"User's explicitly set preferences (if any): {', '.join(user_ctx.get('preferences',[])) if user_ctx.get('preferences') else 'Not set'}" )
        
        ai_system_message = ("Analyze user's chat and Spotify data. Infer their musical preferences (genres, artists they like or might like based on context) and current/recent mood. "
                           "Respond strictly in JSON format with keys: 'genres' (list of up to 2 strings), 'artists' (list of up to 2 strings), 'mood' (single string, or null if not clear). "
                           "Prioritize explicit statements in conversation. Use Spotify data for confirmation or refinement. If no strong signals, return empty lists or null mood. Be concise.")
        
        ai_messages_payload = [{"role": "system", "content": ai_system_message}, {"role": "user", "content": prompt_to_ai_user_content }]
        
        # Call OpenAI API
        openai_response = await asyncio.to_thread(
            client.chat.completions.create, model="gpt-3.5-turbo-0125", # Model supporting JSON mode
            messages=ai_messages_payload, max_tokens=120, temperature=0.05, # Low temp for more deterministic JSON
            response_format={"type": "json_object"})
        
        # Parse and validate response
        ai_result_content = openai_response.choices[0].message.content
        if not ai_result_content: 
            logger.warning(f"AI analysis (user {user_id}) returned empty content.")
            return default_analysis
        parsed_ai_result = json.loads(ai_result_content)

        if not isinstance(parsed_ai_result, dict):
            logger.error(f"AI analysis (user {user_id}) did not return a dictionary: {ai_result_content}")
            return default_analysis
            
        # Extract and sanitize data from AI response
        inferred_genres_list = parsed_ai_result.get("genres", [])
        if isinstance(inferred_genres_list, str): inferred_genres_list = [g.strip() for g in inferred_genres_list.split(",") if g.strip()]
        if not isinstance(inferred_genres_list, list): inferred_genres_list = []
        
        inferred_artists_list = parsed_ai_result.get("artists", [])
        if isinstance(inferred_artists_list, str): inferred_artists_list = [a.strip() for a in inferred_artists_list.split(",") if a.strip()]
        if not isinstance(inferred_artists_list, list): inferred_artists_list = []

        inferred_mood_str = str(parsed_ai_result.get("mood","")).strip().lower() if isinstance(parsed_ai_result.get("mood"),str) else None
        if inferred_mood_str == "null": inferred_mood_str = None # Treat "null" string from AI as Python None

        # Update user_contexts carefully: only if AI provides new/different info and user hasn't explicitly set stronger prefs.
        if inferred_genres_list and (not user_ctx.get("preferences") or set(inferred_genres_list[:2]) != set(user_ctx.get("preferences",[])) ): # Only update if new or different
            user_ctx["preferences"] = list(set(inferred_genres_list[:2])) # Take top 2 unique genres
        
        if inferred_mood_str and (inferred_mood_str != user_ctx.get("mood") or not user_ctx.get("mood")): # Update mood if AI has one and it's new/different or not set
            user_ctx["mood"] = inferred_mood_str
        
        logger.info(f"AI analysis output for user {user_id}: Genres={user_ctx['preferences']}, Mood={user_ctx['mood']}, Suggested Artists={inferred_artists_list[:2]}")
        return {"genres": user_ctx["preferences"], "artists": inferred_artists_list[:2], "mood": user_ctx["mood"]}

    except json.JSONDecodeError as jde_analyze:
        logger.error(f"AI analyze_conversation JSON decode error (user {user_id}): {jde_analyze}. Raw content from AI: '{openai_response.choices[0].message.content if 'openai_response' in locals() and openai_response.choices else 'N/A'}'")
    except Exception as e_analyze: # Catch other errors from OpenAI call or processing
        logger.error(f"Error in AI analyze_conversation for user {user_id}: {e_analyze}", exc_info=False) # exc_info=False for brevity
    
    return default_analysis # Fallback to existing context if AI fails

async def smart_recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id, user_first_name = update.effective_user.id, update.effective_user.first_name
    status_msg_handler = await update.message.reply_text(f"🎵 Thinking of some great music for you, {user_first_name}...")
    try:
        # Optionally refresh Spotify data if linked and needed for better analysis
        if user_contexts.get(user_id, {}).get("spotify", {}).get("access_token"):
            logger.info(f"Fetching latest Spotify data for user {user_id} for smart recommendations.")
            # Run in parallel if these become slow
            recent_p = await get_user_spotify_data(user_id, "player/recently-played", params={"limit": 5})
            top_t = await get_user_spotify_data(user_id, "top/tracks", params={"limit": 5, "time_range": "short_term"})
            if recent_p: user_contexts[user_id]["spotify"]["recently_played"] = recent_p # type: ignore
            if top_t: user_contexts[user_id]["spotify"]["top_tracks"] = top_t # type: ignore
        
        # Get current user context through AI analysis
        user_analysis_data = await analyze_conversation(user_id)
        derived_mood = user_analysis_data.get("mood")
        
        # If mood is not clear, prompt the user using the set_mood conversation
        if not derived_mood or derived_mood == "neutral": # "neutral" considered as needing explicit input
            await status_msg_handler.delete() # Remove the "Thinking..." message
            logger.info(f"Mood unclear for user {user_id} (current: {derived_mood}), prompting for smart recommendations via set_mood.")
            await set_mood(update, context) # This will initiate the mood selection conversation
            return
        
        await status_msg_handler.edit_text(f"Okay {user_first_name}, based on your mood of **{derived_mood}** (and other vibes!)...\nLooking for recommendations... 🎧", 
                                     parse_mode=ParseMode.MARKDOWN)
        
        # Prepare seeds for Spotify recommendations
        spotify_seed_tracks: List[str] = []
        spotify_seed_artists_ids: List[str] = user_analysis_data.get("artists_ids", []) # Assuming analyze_conversation could return artist_ids
        user_genres: List[str] = user_analysis_data.get("genres", [])
        
        # !! IMPORTANT: Populate this with actual values from GET /v1/recommendations/available-genre-seeds !!
        # This mapping needs to be accurate for Spotify's API.
        spotify_genre_seed_map = {
            "pop": "pop", "rock": "rock", "hip-hop": "hip-hop", "hiphop": "hip-hop",
            "electronic": "electronic", "dance": "dance", "edm": "edm", "electro": "electro",
            "classical": "classical", "jazz": "jazz", "blues": "blues",
            "r&b": "r-n-b", "rnb": "r-n-b", "soul": "soul",
            "folk": "folk", "acoustic": "acoustic",
            "sad": "sad", "happy": "happy", "chill": "chill", # Moods can sometimes be genres
            # Add many more: e.g., metal, country, reggae, latin, indie, etc.
        }
        # Convert user-friendly genres to Spotify seed genres
        valid_spotify_seed_genres = [spotify_genre_seed_map.get(g.lower().replace(" ", "-")) for g in user_genres if spotify_genre_seed_map.get(g.lower().replace(" ", "-"))]
        valid_spotify_seed_genres = list(set(valid_spotify_seed_genres)) # Unique and filter None

        user_spotify_context = user_contexts.get(user_id, {}).get("spotify", {})
        if user_spotify_context.get("access_token"): # If Spotify is linked
            # Prefer recently played tracks as seeds if available
            if user_spotify_context.get("recently_played"):
                spotify_seed_tracks.extend([
                    track_item["track"]["id"] for track_item in user_spotify_context["recently_played"][:2] # Max 2 recent tracks
                    if track_item.get("track") and track_item["track"].get("id")
                ])
        
        # Attempt Spotify recommendations
        spotify_client_credentials_token = await get_spotify_token()
        if spotify_client_credentials_token and (spotify_seed_tracks or spotify_seed_artists_ids or valid_spotify_seed_genres):
            logger.info(f"Attempting Spotify API recommendations for user {user_id} with seeds: Tracks={spotify_seed_tracks}, Artists={spotify_seed_artists_ids}, Genres={valid_spotify_seed_genres}")
            
            spotify_recommendations = await get_spotify_recommendations(
                spotify_client_credentials_token, 
                seed_tracks=spotify_seed_tracks[:2], # Max 2 track seeds
                seed_genres=valid_spotify_seed_genres[:1], # Max 1 genre seed
                seed_artists=spotify_seed_artists_ids[:1], # Max 1 artist seed
                limit=5)
            
            if spotify_recommendations:
                response_html_spotify = f"🎵 Tailored Spotify recommendations for your **{derived_mood}** mood, {user_first_name}:\n\n"
                keyboard_spotify_recs = []
                for i, track_data in enumerate(spotify_recommendations, 1):
                    artists_names_str = ", ".join(a["name"] for a in track_data.get("artists",[]))
                    album_name_str = track_data.get("album", {}).get("name", "")
                    track_info_html = f"<b>{track_data['name']}</b> by <i>{artists_names_str}</i>"
                    if album_name_str: track_info_html += f" (from {album_name_str})"
                    response_html_spotify += f"{i}. {track_info_html}\n"
                    
                    # Create a YouTube search query for this Spotify track
                    youtube_search_query_for_track = f"{track_data['name']} {artists_names_str}"
                    # Truncate button text safely
                    btn_display_text = track_data['name'][:18] + "..." if len(track_data['name']) > 18 else track_data['name']
                    keyboard_spotify_recs.append([InlineKeyboardButton(f"YT Search: {btn_display_text}", 
                                                                       callback_data=f"{CB_SHOW_OPTIONS_PREFIX}{youtube_search_query_for_track}")])

                response_html_spotify += "\n💡 <i>Click a track to search for it on YouTube for download options.</i>"
                await status_msg_handler.edit_text(response_html_spotify, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard_spotify_recs))
                return # Exit after successful Spotify recommendations

        # Fallback: YouTube search based on mood, genres, and AI-identified artists
        youtube_query_parts = [derived_mood] # Start with mood
        if valid_spotify_seed_genres: youtube_query_parts.append(valid_spotify_seed_genres[0]) # Use mapped Spotify genre if available
        elif user_genres: youtube_query_parts.append(user_genres[0]) # Fallback to user's raw genre if no mapping found
        
        ai_suggested_artists = user_analysis_data.get("artists", []) # Artists suggested by AI (names, not IDs)
        if ai_suggested_artists: youtube_query_parts.append(f"music like {ai_suggested_artists[0]}")
        
        final_youtube_search_query = " ".join(youtube_query_parts) + " music" # Add "music" for better results
        logger.info(f"Falling back to YouTube search for recommendations (user {user_id}) with query: '{final_youtube_search_query}'")
        await status_msg_handler.edit_text(f"Searching YouTube for some **{derived_mood}** tracks based on '<i>{final_youtube_search_query}</i>'...", parse_mode=ParseMode.HTML)

        youtube_search_results_list = await asyncio.to_thread(search_youtube_sync, final_youtube_search_query, max_results=5)
        if youtube_search_results_list:
            response_html_youtube_fallback = f"🎵 Some YouTube suggestions for your **{derived_mood}** mood, {user_first_name}:\n\n"
            keyboard_youtube_fallback_buttons = []
            displayed_yt_count = 0
            for i, yt_result_item in enumerate(youtube_search_results_list, 1):
                if not yt_result_item.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', yt_result_item['id']): continue # Skip invalid
                displayed_yt_count += 1
                duration_value = yt_result_item.get('duration', 0)
                duration_display_str_yt = ""
                if duration_value and isinstance(duration_value,(int,float)) and duration_value > 0: 
                    mins_yt, secs_yt = divmod(int(duration_value), 60)
                    duration_display_str_yt = f" [{mins_yt}:{secs_yt:02d}]"
                
                response_html_youtube_fallback += f"{displayed_yt_count}. <b>{yt_result_item['title']}</b> - <i>{yt_result_item.get('uploader', 'N/A')}</i>{duration_display_str_yt}\n"
                # Truncate button title
                button_title_yt = yt_result_item['title'][:28]+"..." if len(yt_result_item['title'])>28 else yt_result_item['title']
                keyboard_youtube_fallback_buttons.append([InlineKeyboardButton(f"DL: {button_title_yt}",callback_data=f"{CB_DOWNLOAD_PREFIX}{yt_result_item['id']}")])
            
            if not keyboard_youtube_fallback_buttons: # No valid results after filtering
                 await status_msg_handler.delete() # Delete "Searching..."
                 await provide_generic_recommendations(update, derived_mood, chat_id_override=user_id) # type: ignore
                 return

            response_html_youtube_fallback += "\n💡 <i>Click any track to download its audio directly.</i>"
            await status_msg_handler.edit_text(response_html_youtube_fallback, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard_youtube_fallback_buttons))
        else: # Final fallback: generic hardcoded list if YouTube search also fails
            logger.info(f"No YouTube results for '{final_youtube_search_query}', providing generic recommendations for mood {derived_mood}.")
            await status_msg_handler.delete() # Clear "Searching..."
            await provide_generic_recommendations(update, derived_mood, chat_id_override=user_id) # type: ignore

    except Exception as e_smart_rec:
        logger.error(f"Error in smart_recommend_music for user {user_id}: {e_smart_rec}", exc_info=True)
        await status_msg_handler.edit_text(f"Oh no, {user_first_name}! I ran into a snag trying to find recommendations. 😥 Please try again in a bit.")


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    error_message_text_to_user = "😓 Oops! Something went wrong on my end. My developers have been notified. Please try again in a bit!"
    
    if isinstance(context.error, TimedOut):
        error_message_text_to_user = "🐢 Things are a bit slow right now, the operation timed out. Please try again!"
    elif isinstance(context.error, NetworkError):
        # Provide a more user-friendly network error message
        if "getaddrinfo failed" in str(context.error).lower():
            error_message_text_to_user = "📡 I'm having trouble connecting (DNS issue). Please check your internet connection or try again later."
        else:
            error_message_text_to_user = f"📡 I'm having network issues: {str(context.error)[:100]}. Please check your connection or try again later."

    # Try to send the error message to the user
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text(error_message_text_to_user)
        except Exception as e_reply_err:
            logger.error(f"Failed to send error reply to user (via message): {e_reply_err}")
    elif isinstance(update, Update) and update.callback_query and update.callback_query.message:
        # If it's a callback query, reply to the message the callback was attached to
        try:
            await update.callback_query.message.reply_text(error_message_text_to_user) 
        except Exception as e_reply_cb_err:
            logger.error(f"Failed to send error reply for callback to user (via callback message): {e_reply_cb_err}")


def cleanup_downloads_atexit() -> None:
    logger.info("Cleaning up temporary download files at exit...")
    cleaned_files_count = 0
    try:
        if os.path.exists(DOWNLOAD_DIR):
            for item_name_in_dir in os.listdir(DOWNLOAD_DIR):
                item_full_path = os.path.join(DOWNLOAD_DIR, item_name_in_dir)
                try:
                    # Be more specific about what to delete to avoid accidental deletion of other files
                    if os.path.isfile(item_full_path) and \
                       (any(item_name_in_dir.endswith(ext) for ext in [".m4a", ".mp3", ".webm", ".ogg", ".opus", ".tmp"]) or \
                        item_name_in_dir.startswith("voice_")): 
                        os.remove(item_full_path)
                        cleaned_files_count +=1
                except Exception as e_remove_file:
                    logger.error(f"Failed to remove temporary file {item_full_path}: {e_remove_file}")
            
            if cleaned_files_count > 0:
                logger.info(f"Successfully cleaned {cleaned_files_count} temporary file(s) from '{DOWNLOAD_DIR}'.")
            else:
                logger.info(f"No specific temporary files found to clean in '{DOWNLOAD_DIR}'.")
        else:
            logger.info(f"Download directory '{DOWNLOAD_DIR}' not found, no cleanup needed at exit.")
    except Exception as e_cleanup_dir:
        logger.error(f"Error during atexit cleanup of downloads directory: {e_cleanup_dir}")

def signal_exit_handler(sig, frame) -> None:
    logger.info(f"Received signal {sig}, preparing for graceful exit...")
    # cleanup_downloads_atexit() is registered with atexit, so it should run on normal sys.exit().
    # Call it explicitly here if there's concern atexit might not trigger for all signal types handled.
    if sig in [signal.SIGINT, signal.SIGTERM]: # Common termination signals
        cleanup_downloads_atexit() 
    sys.exit(0) # Exit after cleanup

def main() -> None:
    # Application Builder with adjusted timeouts and rate limiter
    app_builder = Application.builder().token(TOKEN)
    app_builder.connect_timeout(20.0) # Time to establish connection
    app_builder.read_timeout(30.0)    # Time to read data once connected
    app_builder.write_timeout(45.0)   # Time to write data
    app_builder.pool_timeout(180.0)   # Timeout for operations within the connection pool (e.g. file uploads)
    # Rate limiter: limits to 15 requests/sec overall, 5/sec for group chats, retries 2 times. Adjust as needed.
    app_builder.rate_limiter(AIORateLimiter(overall_max_rate=15, max_retries=2, group_max_rate=5))
    application = app_builder.build()

    # Define handlers in a list for easier management
    handlers_list = [
        CommandHandler("start", start), CommandHandler("help", help_command),
        CommandHandler("download", download_music), CommandHandler("search", search_command),
        CommandHandler("autodownload", auto_download_command), CommandHandler("lyrics", get_lyrics_command),
        CommandHandler("recommend", smart_recommend_music), CommandHandler("create_playlist", create_playlist),
        CommandHandler("clear", clear_history), 
        # Global /spotify_code handler (if called when not in SPOTIFY_CODE conversation state)
        CommandHandler("spotify_code", spotify_code_command),
        
        # Spotify Linking Conversation
        ConversationHandler(
            entry_points=[CommandHandler("link_spotify", link_spotify)],
            states={
                SPOTIFY_CODE: [
                    MessageHandler(filters.TEXT & ~filters.COMMAND, spotify_code_handler), # For pasted code
                    # This CommandHandler for /spotify_code inside state map takes precedence if conversation is active
                    CommandHandler("spotify_code", spotify_code_handler), 
                    CallbackQueryHandler(cancel_spotify, pattern=f"^{CB_CANCEL_SPOTIFY}$") # For cancel button
                ]
            },
            fallbacks=[CommandHandler("cancel", cancel)], # Generic cancel
            conversation_timeout=timedelta(minutes=10).total_seconds(), # 10 min timeout for Spotify code entry
            per_message=True # Handles callbacks on a per-message basis within the conversation
        ),
        
        # Mood Setting Conversation
        ConversationHandler(
            entry_points=[CommandHandler("mood", set_mood)],
            states={
                MOOD: [CallbackQueryHandler(enhanced_button_handler, pattern=f"^{CB_MOOD_PREFIX}")],
                PREFERENCE: [CallbackQueryHandler(enhanced_button_handler, pattern=f"^{CB_PREFERENCE_PREFIX}")]
            },
            fallbacks=[CommandHandler("cancel", cancel)],
            conversation_timeout=timedelta(minutes=5).total_seconds(), # 5 min timeout for mood/preference selection
            per_message=True
        ),
        
        # Other message/callback handlers
        MessageHandler(filters.VOICE & ~filters.COMMAND, handle_voice),
        # Generic CallbackQueryHandler MUST be after ConversationHandler to not override specific patterns
        CallbackQueryHandler(enhanced_button_handler), 
        # General text message handler (should be last among message handlers)
        MessageHandler(filters.TEXT & ~filters.COMMAND, enhanced_handle_message)
    ]
    # Add all defined handlers to the application
    for handler_item in handlers_list:
        application.add_handler(handler_item)
        
    # Add the error handler
    application.add_error_handler(error_handler)

    # Setup OS signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_exit_handler) # Handle Ctrl+C
    signal.signal(signal.SIGTERM, signal_exit_handler) # Handle kill/system shutdown signals
    # Register atexit cleanup for download directory (runs on normal Python interpreter exit)
    atexit.register(cleanup_downloads_atexit)

    logger.info("🚀 Starting MelodyMind Bot... Attempting to connect to Telegram.")
    try:
        # Start polling, drop pending updates if any from previous unclean shutdowns
        application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
    except Exception as e_polling: # Catch critical errors during polling startup/runtime
        logger.critical(f"Bot polling failed to start or crashed critically: {e_polling}", exc_info=True)
    finally:
        logger.info(" MelodyMind Bot has shut down.") # Log on normal shutdown or after crash

if __name__ == "__main__":
    # Pre-run environment variable checks
    if not TOKEN:
        logger.critical("FATAL: TELEGRAM_TOKEN environment variable is MISSING. Bot cannot start.")
        sys.exit(1) # Critical failure, exit
    if not OPENAI_API_KEY:
        logger.warning("WARNING: OPENAI_API_KEY not set. AI-related features will be degraded or disabled.")
    # Consider adding checks for SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET if Spotify linking is core to startup.
    
    main()
