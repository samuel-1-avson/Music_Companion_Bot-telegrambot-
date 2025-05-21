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
CB_AUTO_DOWNLOAD_PREFIX = "auto_download_" # Used when AI finds a song and user confirms download
CB_SHOW_OPTIONS_PREFIX = "show_options_" # Used when AI finds a song and user wants more options
CB_CANCEL_SEARCH = "cancel_search"
CB_CANCEL_SPOTIFY = "cancel_spotify"

active_downloads = set()
user_contexts: Dict[int, Dict] = {}
logger.warning("User contexts are stored in-memory and will be lost on bot restart.")
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=15) # Increased default timeout

# ==================== SPOTIFY HELPER FUNCTIONS ====================

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
    except aiohttp.ClientError as e: # Catching ClientResponseError specifically if possible
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
                if response.status == 400: # Specific check for Bad Request (often invalid code/redirect_uri)
                    error_details = await response.json()
                    logger.error(f"Spotify Bad Request (user {user_id}, code exchange): {error_details.get('error_description', response.reason)}")
                    return None # Propagate failure clearly
                response.raise_for_status()
                token_data = await response.json()
                token_data["expires_at"] = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600) - 120)).timestamp() # 2-min buffer
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
        spotify_data.clear() # Clear potentially corrupt token data
        return None
    url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}".encode()).decode()
    headers = {"Authorization": f"Basic {auth_header}", "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token_str}
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url, headers=headers, data=data) as response:
                if response.status == 400: # Invalid refresh token likely
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
                new_refresh_token_str = token_data.get("refresh_token", refresh_token_str) # Use new if provided
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
    video_id = (re.search(r'(?:v=|/)([0-9A-Za-z_-]{11})', url) or {}).get(1, "UnknownID")
    try:
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'),
            'quiet': True, 'no_warnings': True, 'noplaylist': True,
            'max_filesize': 50 * 1024 * 1024, 'restrictfilenames': True,
            'sleep_interval_requests': 1, 'sleep_interval': 1,
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
            ydl.extract_info(url, download=True)

            actual_path = expected_path # Assume prepare_filename is correct first
            if not os.path.exists(actual_path): # Fallback if filename slightly changed by ydl
                base_name, _ = os.path.splitext(os.path.basename(expected_path)) # Only filename part
                found = False
                for f_name in os.listdir(DOWNLOAD_DIR):
                    if base_name in f_name: # Simpler check: if original title base is in new filename
                        actual_path = os.path.join(DOWNLOAD_DIR, f_name)
                        logger.info(f"YT: File found at modified path {actual_path}")
                        found = True
                        break
                if not found:
                    logger.error(f"YT: Downloaded file not found at {expected_path} or variants for {url}")
                    return {"success": False, "error": "Downloaded file not found post-download"}

            if os.path.getsize(actual_path) > 50.5 * 1024 * 1024:
                os.remove(actual_path)
                logger.warning(f"YT: File '{title}' over 50MB, removed.")
                return {"success": False, "error": "File >50MB"}
            
            return {"success": True, "title": title, "artist": artist,
                    "thumbnail_url": info.get('thumbnail'), "duration": info.get('duration', 0),
                    "audio_path": actual_path}
    except yt_dlp.utils.DownloadError as de:
        err_str = str(de).lower()
        if "video unavailable" in err_str: err_msg = "Video unavailable."
        elif "private video" in err_str: err_msg = "Private video."
        elif " геогр" in err_str or "geo-restricted" in err_str: err_msg = "Video geo-restricted." # Russian for geography
        else: err_msg = f"Download issue: {str(de)[:80]}"
        logger.error(f"YT DownloadError for {url}: {err_msg}")
        return {"success": False, "error": err_msg}
    except Exception as e:
        logger.error(f"YT Generic DL error {url}: {e}", exc_info=False)
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
            results = [{
                'title': e.get('title', 'Unknown Title'),
                'url': e.get('webpage_url') or e.get('url') or f"https://youtube.com/watch?v={e.get('id')}",
                'thumbnail': e.get('thumbnail') or (e.get('thumbnails')[0]['url'] if e.get('thumbnails') else ''),
                'uploader': e.get('uploader', 'Unknown Artist'),
                'duration': e.get('duration', 0), 'id': e.get('id', '')
            } for e in info['entries'] if e and e.get('id')] # Ensure ID exists
            logger.info(f"YT Search: Found {len(results)} for '{query}'")
            return results
    except yt_dlp.utils.DownloadError as de:
        logger.error(f"YT Search DownloadError for '{query}': {de}")
    except Exception as e:
        logger.error(f"YT Search error for '{query}': {e}", exc_info=False)
    return []

# ==================== AI AND LYRICS FUNCTIONS ====================

async def generate_chat_response(user_id: int, message: str) -> str:
    if not client: return "AI service offline. Let's talk music another way?"
    ctx = user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    ctx.setdefault("conversation_history", [])
    ctx["conversation_history"] = ctx["conversation_history"][-10:] # Limit context
    system_prompt = ("MelodyMind: Friendly music bot. Brief, warm chat about music/feelings. "
                     "If asked for music, guide to commands or ask for song name to search. "
                     "Use mood/prefs/Spotify artists subtly. 2-3 sentences.")
    messages = [{"role": "system", "content": system_prompt}]
    summary = []
    if ctx.get("mood"): summary.append(f"Mood: {ctx['mood']}.")
    if ctx.get("preferences"): summary.append(f"Prefs: {', '.join(ctx['preferences'])}.")
    if "spotify" in ctx and ctx["spotify"].get("recently_played"):
        try:
            artists = list(set(item["track"]["artists"][0]["name"] for item in ctx["spotify"]["recently_played"][:2] if item.get("track") and item["track"].get("artists")))
            if artists: summary.append(f"Listens to: {', '.join(artists)}.")
        except: pass
    if summary: messages.append({"role": "system", "content": "User Info: " + " ".join(summary)})
    messages.extend(ctx["conversation_history"][-4:]) # Last 2 exchanges
    messages.append({"role": "user", "content": message})
    try:
        response = await asyncio.to_thread(client.chat.completions.create, model="gpt-3.5-turbo", messages=messages, max_tokens=90, temperature=0.7)
        reply = response.choices[0].message.content.strip()
        ctx["conversation_history"].extend([{"role": "user", "content": message}, {"role": "assistant", "content": reply}])
        return reply
    except Exception as e:
        logger.error(f"AI chat response error (user {user_id}): {e}")
        return "Hmm, my thoughts are a bit jumbled. How about your favorite song?"

def get_lyrics_sync(song_title: str, artist: Optional[str] = None) -> str:
    if not genius: return "Lyrics service offline."
    logger.info(f"Lyrics search: '{song_title}' by '{artist or 'Any'}'")
    try:
        song = genius.search_song(song_title, artist) if artist else genius.search_song(song_title)
        if not song:
            return f"No lyrics found for '<b>{song_title}</b>'{f' by <i>{artist}</i>' if artist else ''}. Check spelling?"
        lyrics = song.lyrics
        lyrics = re.sub(r'\s*\[.*?\]\s*', '\n', lyrics).strip()
        lyrics = re.sub(r'\d*Embed$', '', lyrics, flags=re.IGNORECASE).strip()
        lyrics = re.sub(r'^\S*Lyrics', '', lyrics, flags=re.IGNORECASE).strip()
        lyrics = re.sub(r'\n{3,}', '\n\n', lyrics).strip()
        if not lyrics: return f"Lyrics for '<b>{song.title}</b>' seem empty."
        return f"🎵 <b>{song.title}</b> by <i>{song.artist}</i> 🎵\n\n{lyrics}"
    except Exception as e:
        logger.error(f"Genius lyrics error ('{song_title}'): {e}", exc_info=False)
        return f"Issue fetching lyrics for '<b>{song_title}</b>'. Try later."

async def detect_mood_from_text(user_id: int, text: str) -> str:
    if not client: return user_contexts.get(user_id, {}).get("mood", "neutral")
    logger.debug(f"AI Mood detect (user {user_id}): '{text[:40]}...'")
    try:
        response = await asyncio.to_thread(
            client.chat.completions.create, model="gpt-3.5-turbo",
            messages=[{"role": "system", "content": "Detect dominant mood: happy, sad, anxious, excited, calm, angry, energetic, relaxed, focused, nostalgic, or neutral."},
                      {"role": "user", "content": f"Text: '{text}'"}],
            max_tokens=8, temperature=0.1)
        mood_raw = response.choices[0].message.content.lower().strip().replace(".", "")
        mood_map = {"positive": "happy", "negative": "sad", "joyful": "happy", "chill": "relaxed", "stressed": "anxious"}
        mood = mood_map.get(mood_raw, mood_raw)
        valid_moods = ["happy", "sad", "anxious", "excited", "calm", "angry", "neutral", "energetic", "relaxed", "focused", "nostalgic"]
        if mood in valid_moods:
            logger.info(f"AI Mood (user {user_id}): '{mood}' from '{mood_raw}'")
            return mood
        return "neutral"
    except Exception as e:
        logger.error(f"AI Mood detect error (user {user_id}): {e}")
        return user_contexts.get(user_id, {}).get("mood", "neutral")

async def is_music_request(user_id: int, message: str) -> Dict[str, Any]:
    if not client: return {"is_music_request": False, "song_query": None}
    logger.debug(f"AI MusicReq detect (user {user_id}): '{message[:40]}...'")
    try:
        prompt = ("Analyze if message is specific music request (play/download X by Y). JSON: "
                  "'is_music_request': bool, 'song_query': str/null. General music chat is NOT a request.")
        response = await asyncio.to_thread(
            client.chat.completions.create, model="gpt-3.5-turbo-0125",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": f"Msg: '{message}'"}],
            max_tokens=70, temperature=0.0, response_format={"type": "json_object"})
        result = json.loads(response.choices[0].message.content)
        is_req = result.get("is_music_request", False)
        is_req = str(is_req).lower() == "true" if isinstance(is_req, (str, bool)) else False # Robust bool check
        query = result.get("song_query")
        query = str(query).strip() if isinstance(query, str) and query.strip() else None
        logger.info(f"AI MusicReq (user {user_id}): is_req={is_req}, query='{query}' for msg: '{message[:30]}'")
        return {"is_music_request": is_req, "song_query": query}
    except Exception as e: # Broad catch for API/JSON errors
        logger.error(f"AI MusicReq error (user {user_id}): {e}", exc_info=False)
        return {"is_music_request": False, "song_query": None}

# ==================== TELEGRAM BOT HANDLERS ====================
# ... (start, help, download_music, create_playlist, handle_voice, link_spotify, spotify_code_handler, spotify_code_command, cancel_spotify, search_command, auto_download_command, get_lyrics_command are largely similar or had minor tweaks already)

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
        "/clear - Clear chat history\n\n"
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
    elif message_text:
        # Allow direct URL pasting to trigger download
        url_match = re.search(r"(https?:\/\/(?:www\.)?(?:youtube\.com\/(?:watch\?v=|embed\/|v\/|shorts\/)|youtu\.be\/)([a-zA-Z0-9_-]{11}))", message_text)
        if url_match:
            url_to_download = url_match.group(1)
    
    if not url_to_download or not is_valid_youtube_url(url_to_download):
        await update.message.reply_text("❌ Invalid or missing YouTube URL. Use `/download <URL>` or send a valid link.")
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
        
        audio_path = result["audio_path"] # This should be populated by download_youtube_audio_sync
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
        except: pass
    except (TimedOut, NetworkError) as net_err:
        logger.error(f"Net/TG API error during DL (user {user_id}, url: {url_to_download}): {net_err}")
        try: await status_msg.edit_text(f"❌ Network/Telegram error: {net_err}. Try again.")
        except: pass
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
    spotify_user_id_api = None
    try:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.get(user_profile_url, headers=headers_auth) as response:
                response.raise_for_status()
                spotify_user_id_api = (await response.json()).get("id")
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
    temp_ogg_path = os.path.join(DOWNLOAD_DIR, f"voice_{user_id}_{update.message.message_id}.ogg")
    await voice_file.download_to_drive(temp_ogg_path)
    
    recognizer = sr.Recognizer()
    transcribed_text = None
    try:
        def _transcribe_sync():
            with sr.AudioFile(temp_ogg_path) as source: audio_data = recognizer.record(source)
            try: return recognizer.recognize_google(audio_data)
            except sr.UnknownValueError: logger.warning(f"SR: Google UnknownValue (user {user_id})")
            except sr.RequestError as req_e: logger.error(f"SR: Google RequestError (user {user_id}); {req_e}"); return "ERROR_REQUEST"
            return None
        transcribed_text = await asyncio.to_thread(_transcribe_sync)

        if transcribed_text == "ERROR_REQUEST":
            await update.message.reply_text("Voice recognition service error. Please type or try later.")
        elif transcribed_text:
            logger.info(f"Voice (user {user_id}) transcribed: '{transcribed_text}'")
            await update.message.reply_text(f"🎤 Heard: \"<i>{transcribed_text}</i>\"\nProcessing...", parse_mode=ParseMode.HTML)
            context.user_data['_voice_original_message'] = update.message 
            fake_msg = update.message._replace(text=transcribed_text, voice=None) 
            await enhanced_handle_message(Update(update.update_id, message=fake_msg), context)
        else:
            await update.message.reply_text("Couldn't catch that. Try speaking clearly, or type? 😊")
    except Exception as e:
        logger.error(f"Error processing voice (user {user_id}): {e}", exc_info=True)
        await update.message.reply_text("Oops! Error with voice message. Try again.")
    finally:
        if os.path.exists(temp_ogg_path):
            try: os.remove(temp_ogg_path)
            except OSError as e: logger.error(f"Error deleting temp voice file {temp_ogg_path}: {e}")

async def link_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not all([SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI]):
        await update.message.reply_text("Sorry, Spotify linking not configured by admin. 😥")
        return ConversationHandler.END
    if SPOTIFY_REDIRECT_URI == "https://your-callback-url.com":
         await update.message.reply_text("⚠️ Spotify redirect URI is placeholder. Manual code copy likely needed.")
    user_id = update.effective_user.id
    scopes = "user-read-recently-played user-top-read playlist-modify-private"
    auth_url = (f"https://accounts.spotify.com/authorize?client_id={SPOTIFY_CLIENT_ID}"
                f"&response_type=code&redirect_uri={SPOTIFY_REDIRECT_URI}"
                f"&scope={scopes.replace(' ', '%20')}&state={user_id}")
    kb = [[InlineKeyboardButton("🔗 Link My Spotify", url=auth_url)], [InlineKeyboardButton("Cancel", callback_data=CB_CANCEL_SPOTIFY)]]
    await update.message.reply_text(
        "Let's link Spotify for personalized music! 🎵\n\n"
        "1. Click below.\n2. Authorize. Spotify redirects you. From redirected page URL, copy the `code` value.\n"
        "   (URL: `.../?code=A_LONG_CODE&state=...` - get `A_LONG_CODE`)\n"
        "3. Send that code back to me here.\n\n",
        reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.MARKDOWN)
    return SPOTIFY_CODE

async def spotify_code_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    code_to_use = (context.args[0] if context.args and update.message.text.startswith('/spotify_code') 
                   else update.message.text.strip() if not update.message.text.startswith('/') else None)
    if not code_to_use or len(code_to_use) < 30: # Basic check, actual codes are much longer
        await update.message.reply_text("Code seems short/missing. Paste full Spotify code or use `/spotify_code YOUR_CODE`.")
        return SPOTIFY_CODE 
    status_msg = await update.message.reply_text("⏳ Validating Spotify code...")
    token_data = await get_user_spotify_token(user_id, code_to_use)
    if not token_data or not token_data.get("access_token"):
        await status_msg.edit_text("❌ Failed to link Spotify. Code invalid/expired or config issue (check redirect URI). Try /link_spotify again.")
        return SPOTIFY_CODE 
    user_contexts.setdefault(user_id, {}).setdefault("spotify", {})
    user_contexts[user_id]["spotify"].update({
        "access_token": cipher.encrypt(token_data["access_token"].encode()),
        "refresh_token": cipher.encrypt(token_data["refresh_token"].encode()),
        "expires_at": token_data["expires_at"]})
    logger.info(f"Spotify linked for user {user_id}.")
    rp = await get_user_spotify_data(user_id, "player/recently-played", params={"limit": 1})
    rp_info = f" I see you recently enjoyed some {rp[0]['track']['artists'][0]['name']}!" if rp and rp[0].get("track") else ""
    await status_msg.edit_text(f"✅ Spotify linked! 🎉{rp_info} Try /recommend!")
    return ConversationHandler.END

async def spotify_code_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
    if not context.args:
        await update.message.reply_text("Provide Spotify code after command: `/spotify_code YOUR_CODE_HERE`")
        return None # Not in a conv state if called globally with no args.
    return await spotify_code_handler(update, context)

async def cancel_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Spotify linking cancelled. Try again with /link_spotify. 👍")
    return ConversationHandler.END

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("What song to search? Ex: `/search Shape of You`")
        return
    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 YT Search: '<i>{query}</i>'...", parse_mode=ParseMode.HTML)
    results = await asyncio.to_thread(search_youtube_sync, query, max_results=5)
    try: await status_msg.delete()
    except Exception: pass
    await send_search_results(update, query, results)

async def auto_download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Song to auto-download? Ex: `/autodownload Believer`")
        return
    await auto_download_first_result(update, context, " ".join(context.args))

async def get_lyrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Song for lyrics? Ex:\n`/lyrics Bohemian Rhapsody` or `/lyrics Queen - Song`")
        return
    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Lyrics search: \"<i>{query}</i>\"...", parse_mode=ParseMode.HTML)
    try:
        artist, song_title = (None, query)
        if " - " in query: artist, song_title = map(str.strip, query.split(" - ", 1))
        elif " by " in query.lower(): 
            m = re.search(r'^(.*?)\s+by\s+(.*?)$', query, re.IGNORECASE)
            if m: song_title, artist = m.group(1).strip(), m.group(2).strip()
        logger.info(f"Lyrics parsed: song='{song_title}', artist='{artist}'")
        lyrics = await asyncio.to_thread(get_lyrics_sync, song_title, artist)
        
        max_len = 4080 # Safer limit
        if len(lyrics) > max_len:
            first_part = lyrics[:max_len]
            cut = first_part.rfind('\n\n') 
            if cut == -1 or cut < max_len - 1000 : cut = first_part.rfind('\n', 0, max_len -200) # Try single newline further back
            if cut == -1 or cut < max_len - 1500 : cut = max_len - 100 # last resort rough cut if no good newline
            
            await status_msg.edit_text(f"{lyrics[:cut]}\n\n<small>(Continued below)</small>", parse_mode=ParseMode.HTML)
            remaining = lyrics[cut:]
            while remaining:
                part_to_send = remaining[:max_len]
                remaining = remaining[max_len:]
                # For simplicity, don't try to fine-tune cuts for subsequent parts too much.
                await update.message.reply_text(part_to_send + ("\n<small>(...more)</small>" if remaining else ""), parse_mode=ParseMode.HTML)
        else:
            await status_msg.edit_text(lyrics, parse_mode=ParseMode.HTML)
    except Exception as e:
        logger.error(f"Lyrics cmd error (query '{query}'): {e}", exc_info=True)
        await status_msg.edit_text("Sorry, unexpected hiccup fetching lyrics. 😕")

async def recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await smart_recommend_music(update, context)

async def provide_generic_recommendations(update: Update, mood: str, chat_id_override: Optional[int] = None) -> None:
    logger.info(f"Generic recs for mood: {mood}")
    target_id = chat_id_override or update.effective_chat.id
    mood_map = { # Shortened for brevity
        "happy": ["Uptown Funk - Mark Ronson", "Happy - Pharrell Williams"],
        "sad": ["Someone Like You - Adele", "Hallelujah - Jeff Buckley"],
        "energetic": ["Don't Stop Me Now - Queen", "Thunderstruck - AC/DC"],
        "relaxed": ["Weightless - Marconi Union", "Clair de Lune - Debussy"],
        "focused": ["The Four Seasons - Vivaldi", "Time - Hans Zimmer"],
        "nostalgic": ["Bohemian Rhapsody - Queen", "Wonderwall - Oasis"],
        "neutral": ["Three Little Birds - Bob Marley", "Here Comes The Sun - Beatles"]
    }
    key = mood.lower()
    if key not in mood_map: key = "neutral"
    recs = mood_map[key]
    txt = f"🎵 General **{mood.capitalize()}** vibes:\n\n" + "\n".join(f"{i+1}. {t}" for i,t in enumerate(recs)) + "\n\n💡 <i>Ask me to search/download any!</i>"
    await context.bot.send_message(chat_id=target_id, text=txt, parse_mode=ParseMode.HTML)

async def set_mood(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user = update.effective_user
    user_contexts.setdefault(user.id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    kb_layout = [
        [InlineKeyboardButton("Happy 😊", callback_data=f"{CB_MOOD_PREFIX}happy"), InlineKeyboardButton("Sad 😢", callback_data=f"{CB_MOOD_PREFIX}sad")],
        [InlineKeyboardButton("Energetic 💪", callback_data=f"{CB_MOOD_PREFIX}energetic"), InlineKeyboardButton("Relaxed 😌", callback_data=f"{CB_MOOD_PREFIX}relaxed")],
        [InlineKeyboardButton("Focused 🧠", callback_data=f"{CB_MOOD_PREFIX}focused"), InlineKeyboardButton("Nostalgic 🕰️", callback_data=f"{CB_MOOD_PREFIX}nostalgic")],
        [InlineKeyboardButton("Neutral / Other", callback_data=f"{CB_MOOD_PREFIX}neutral")]]
    cur_mood = user_contexts[user.id].get("mood")
    prompt = f"Hi {user.first_name}! " + (f"Your mood is **{cur_mood}**. Change it or how are you now?" if cur_mood and cur_mood!="neutral" else "How are you feeling today?")
    if update.callback_query: await update.callback_query.edit_message_text(prompt, reply_markup=InlineKeyboardMarkup(kb_layout), parse_mode=ParseMode.MARKDOWN)
    else: await update.message.reply_text(prompt, reply_markup=InlineKeyboardMarkup(kb_layout), parse_mode=ParseMode.MARKDOWN)
    return MOOD

async def send_search_results(update: Update, query: str, results: List[Dict]) -> None:
    if not results:
        await update.message.reply_text(f"😕 No YouTube results for '<i>{query}</i>'. Try other keywords?", parse_mode=ParseMode.HTML)
        return
    kb_rows, head_txt, count = [], f"🔎 YT results for '<i>{query}</i>':\n\n", 0
    for r in results[:5]:
        if not r.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', r['id']): continue
        count +=1
        dur_s = ""
        if r.get('duration') and isinstance(r['duration'], (int, float)) and r['duration'] > 0:
            m, s = divmod(int(r['duration']), 60); dur_s = f" [{m}:{s:02d}]"
        title, btn_title = r.get('title', 'N/A'), (r.get('title', 'N/A')[:33] + "...") if len(r.get('title', '')) > 36 else r.get('title', 'N/A')
        head_txt += f"{count}. <b>{title}</b> by <i>{r.get('uploader', 'N/A')}</i>{dur_s}\n"
        kb_rows.append([InlineKeyboardButton(f"[{count}] {btn_title}{dur_s}", callback_data=f"{CB_DOWNLOAD_PREFIX}{r['id']}")])
    if not kb_rows:
        await update.message.reply_text(f"😕 Found YT results for '<i>{query}</i>', but issues creating download options.", parse_mode=ParseMode.HTML)
        return
    kb_rows.append([InlineKeyboardButton("Cancel Search", callback_data=CB_CANCEL_SEARCH)])
    await update.message.reply_text(head_txt + "\nClick song to download audio:", reply_markup=InlineKeyboardMarkup(kb_rows), parse_mode=ParseMode.HTML)

async def auto_download_first_result(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str, original_message_id_to_edit: Optional[int] = None) -> None:
    user_id = update.effective_user.id
    if user_id in active_downloads:
        reply_target = context.bot.edit_message_text if original_message_id_to_edit else update.message.reply_text
        kwargs = {"chat_id": update.effective_chat.id, "message_id": original_message_id_to_edit, "reply_markup": None} if original_message_id_to_edit else {}
        await reply_target("Hold on! Another download is active. 😊", **kwargs)
        return
    active_downloads.add(user_id)
    status_msg = None
    try:
        edit_kwargs = {"chat_id": update.effective_chat.id, "message_id": original_message_id_to_edit, "reply_markup": None} if original_message_id_to_edit else {}
        status_msg_text = f"🔍 Looking for '<i>{query}</i>' to download..."
        status_msg = await (context.bot.edit_message_text(text=status_msg_text, parse_mode=ParseMode.HTML, **edit_kwargs) if original_message_id_to_edit 
                            else update.message.reply_text(status_msg_text, parse_mode=ParseMode.HTML))

        results = await asyncio.to_thread(search_youtube_sync, query, max_results=1)
        if not results or not results[0].get('id') or not is_valid_youtube_url(results[0].get('url','')): # also check URL
            await status_msg.edit_text(f"❌ Couldn't find downloadable for '<i>{query}</i>'. Try `/search {query}`?", parse_mode=ParseMode.HTML)
            return
        top_res, video_url, video_title = results[0], results[0]["url"], results[0].get("title", "this track")
        await status_msg.edit_text(f"✅ Found: <b>{video_title}</b>.\n⏳ Downloading... (can take a moment!)", parse_mode=ParseMode.HTML)
        dl_res = await asyncio.to_thread(download_youtube_audio_sync, video_url)
        if not dl_res["success"]:
            await status_msg.edit_text(f"❌ DL failed for <b>{video_title}</b>: {dl_res.get('error', 'Unknown')}", parse_mode=ParseMode.HTML)
            return
        await status_msg.edit_text(f"✅ DL'd: <b>{dl_res['title']}</b>.\n✅ Sending audio...", parse_mode=ParseMode.HTML)
        with open(dl_res["audio_path"], 'rb') as audio:
            logger.info(f"Auto-DL: Sending '{dl_res['title']}' (user {user_id}). Path: {dl_res['audio_path']}")
            await context.bot.send_audio(chat_id=update.effective_chat.id, audio=audio,
                title=dl_res["title"][:64], performer=dl_res["artist"][:64] if dl_res.get("artist") else "Unknown",
                caption=f"🎵 Here's: {dl_res['title']}", duration=dl_res.get('duration'))
        if os.path.exists(dl_res["audio_path"]):
            try: os.remove(dl_res["audio_path"])
            except OSError as e: logger.error(f"Error deleting temp (auto-DL) {dl_res['audio_path']}: {e}")
        try: await status_msg.delete()
        except: pass
    except (TimedOut, NetworkError) as net_err:
        logger.error(f"Net/API error (auto-DL user {user_id}, query '{query}'): {net_err}")
        if status_msg: try: await status_msg.edit_text(f"❌ Network issue with '<i>{query}</i>'. Try again.", parse_mode=ParseMode.HTML)
        except: pass
    except Exception as e:
        logger.error(f"Unexpected (auto-DL user {user_id}, query '{query}'): {e}", exc_info=True)
        if status_msg: try: await status_msg.edit_text(f"❌ Unexpected error for '<i>{query}</i>'. Apologies!", parse_mode=ParseMode.HTML)
        except: pass
    finally: active_downloads.discard(user_id)

async def send_audio_via_bot(bot, chat_id, audio_path, title, performer, caption, duration):
    with open(audio_path, 'rb') as audio_f:
        await bot.send_audio(chat_id=chat_id, audio=audio_f, title=title[:64],
            performer=performer[:64] if performer else "Unknown", caption=caption, duration=duration)

async def enhanced_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
    query = update.callback_query
    await query.answer()
    data, user_id = query.data, query.from_user.id
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    logger.debug(f"Button: '{data}' for user {user_id}")

    if data.startswith(CB_MOOD_PREFIX):
        mood = data[len(CB_MOOD_PREFIX):]
        user_contexts[user_id]["mood"] = mood
        logger.info(f"User {user_id} mood: {mood}")
        kb = [[InlineKeyboardButton(p,cdata=f"{CB_PREFERENCE_PREFIX}{p.lower().split('/')[0]}") for p in row] for row in 
              [["Pop", "Rock", "Hip-Hop"], ["Electronic", "Classical", "Jazz"], ["Folk", "R&B", "Any/Surprise!"]]]
        kb.append([InlineKeyboardButton("Skip Genre", callback_data=f"{CB_PREFERENCE_PREFIX}skip")])
        await query.edit_message_text(f"Got it, {query.from_user.first_name}! Feeling {mood}. 🎶\nGenre preference?", reply_markup=InlineKeyboardMarkup(kb))
        return PREFERENCE
    elif data.startswith(CB_PREFERENCE_PREFIX):
        pref = data[len(CB_PREFERENCE_PREFIX):]
        msg = ""
        if pref in ["skip", "any"]: user_contexts[user_id]["preferences"], msg = [], "Alright! I'll keep that in mind."
        else: user_contexts[user_id]["preferences"], msg = [pref], f"Great choice! {pref.capitalize()} it is."
        logger.info(f"User {user_id} pref: {pref}")
        await query.edit_message_text(msg + " Try:\n➡️ `/recommend`\n➡️ `/search [song]`\n➡️ Or just chat!")
        return ConversationHandler.END
    
    elif data.startswith(CB_DOWNLOAD_PREFIX): # Direct download by ID from search/recs
        video_id = data[len(CB_DOWNLOAD_PREFIX):]
        if not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
            logger.error(f"Invalid YT ID from button CB_DOWNLOAD_PREFIX: '{video_id}'")
            await query.edit_message_text("❌ Invalid video ID. Try search again.", reply_markup=None)
            return None
        
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        if user_id in active_downloads:
            await query.edit_message_text("⚠️ Another download active. Please wait!", reply_markup=None)
            return None
        active_downloads.add(user_id)
        status_msg = await query.edit_message_text(f"⏳ Preparing direct download from YouTube...\nThis might take a moment.", reply_markup=None)
        try:
            result = await asyncio.to_thread(download_youtube_audio_sync, youtube_url)
            if not result["success"]:
                err_msg = result.get('error', 'Unknown download error.')
                await status_msg.edit_text(f"❌ Download failed: {err_msg}", reply_markup=None) # Keep status message
                # No context.bot.send_message for new message on failure, edit existing one.
                return None
            await status_msg.edit_text(f"✅ DL'd: <b>{result['title']}</b>\n⏳ Sending audio...", parse_mode=ParseMode.HTML, reply_markup=None)
            await send_audio_via_bot(context.bot, query.message.chat_id, result["audio_path"], 
                                     result["title"], result.get("artist"), 
                                     f"🎵 Here's: {result['title']}", result.get("duration"))
            logger.info(f"Sent audio '{result['title']}' (user {user_id}) via button (direct ID).")
            if os.path.exists(result["audio_path"]):
                try: os.remove(result["audio_path"])
                except OSError as e: logger.error(f"Error deleting temp file {result['audio_path']}: {e}")
            try: await status_msg.delete() # Clean up original message after successful send
            except: pass
        except (TimedOut, NetworkError) as net_err:
            logger.error(f"Net error during button direct DL (user {user_id}, vid {video_id}): {net_err}")
            if status_msg : try: await status_msg.edit_text("❌ Network issue sending file. Try again.", reply_markup=None)
            except : pass
        except Exception as e:
            logger.error(f"Error in button direct DL (user {user_id}, vid {video_id}): {e}", exc_info=True)
            if status_msg : try: await status_msg.edit_text("❌ Unexpected download error.", reply_markup=None)
            except: pass
        finally: active_downloads.discard(user_id)
        return None
    
    elif data.startswith(CB_AUTO_DOWNLOAD_PREFIX): # From "Yes, download" after AI suggestion
        video_id_or_query = data[len(CB_AUTO_DOWNLOAD_PREFIX):]
        query_for_adl = f"https://www.youtube.com/watch?v={video_id_or_query}" if re.match(r'^[0-9A-Za-z_-]{11}$', video_id_or_query) else video_id_or_query
        await auto_download_first_result(update, context, query=query_for_adl, original_message_id_to_edit=query.message.message_id)
        return None
    
    elif data.startswith(CB_SHOW_OPTIONS_PREFIX): # From "Show more options" after AI suggestion
        search_query = data[len(CB_SHOW_OPTIONS_PREFIX):]
        if not search_query:
            await query.edit_message_text("Cannot show options, original query missing.", reply_markup=None)
            return None
        await query.edit_message_text(f"🔍 Showing YouTube options for '<i>{search_query}</i>'...", parse_mode=ParseMode.HTML, reply_markup=None)
        results = await asyncio.to_thread(search_youtube_sync, search_query, max_results=5)
        try: await query.message.delete() # Delete the AI suggestion message
        except Exception as e: logger.warning(f"Could not delete prev msg before CB_SHOW_OPTIONS: {e}")
        # Send search results as a new message in the chat
        await send_search_results(Update(query.update_id, message=query.message), search_query, results) # Use original message as base for sending reply
        return None
    
    elif data == CB_CANCEL_SEARCH:
        await query.edit_message_text("❌ Search cancelled. Anything else?", reply_markup=None)
        return None
    elif data == CB_CANCEL_SPOTIFY:
        await query.edit_message_text("Spotify linking cancelled. Use /link_spotify anytime.", reply_markup=None)
        return ConversationHandler.END
    
    logger.warning(f"Unhandled callback: {data} (user {user_id})")
    return None

async def enhanced_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text: return
    user_id, text = update.effective_user.id, update.message.text.strip()
    logger.debug(f"Msg (user {user_id}): '{text[:80]}'")
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})

    if is_valid_youtube_url(text):
        logger.info(f"User {user_id} sent YT URL: {text}")
        # Create a minimal context-like object for download_music as it expects .args or specific text structure
        # The `Update` object is passed directly, download_music can inspect `update.message.text`.
        await download_music(update, context) # Pass original update and context
        return

    if len(text.split()) > 2: # Passive mood update
        new_mood = await detect_mood_from_text(user_id, text)
        if new_mood and new_mood != "neutral" and new_mood != user_contexts[user_id].get("mood"):
            user_contexts[user_id]["mood"] = new_mood
            logger.debug(f"Passive mood update (user {user_id}): {new_mood} from '{text[:30]}'")

    ai_music_eval = await is_music_request(user_id, text)
    if ai_music_eval.get("is_music_request") and ai_music_eval.get("song_query"):
        query_str = ai_music_eval["song_query"]
        status_msg = await update.message.reply_text(f"🎵 Searching '<i>{query_str}</i>'...", parse_mode=ParseMode.HTML)
        results = await asyncio.to_thread(search_youtube_sync, query_str, max_results=1)
        if results and results[0].get('id') and re.match(r'^[0-9A-Za-z_-]{11}$', results[0]['id']):
            res = results[0]
            kb = [[InlineKeyboardButton(f"✅ Yes, DL '{res['title'][:18]}...'", callback_data=f"{CB_AUTO_DOWNLOAD_PREFIX}{res['id']}")],
                  [InlineKeyboardButton("👀 More options", callback_data=f"{CB_SHOW_OPTIONS_PREFIX}{query_str}")], 
                  [InlineKeyboardButton("❌ No, cancel", callback_data=CB_CANCEL_SEARCH)]]
            await status_msg.edit_text(f"Found: <b>{res['title']}</b> by <i>{res.get('uploader', 'N/A')}</i>.\nDownload or see more?",
                                       reply_markup=InlineKeyboardMarkup(kb), parse_mode=ParseMode.HTML)
        else:
            await status_msg.edit_text(f"😕 Couldn't find track for '<i>{query_str}</i>'. Try `/search {query_str}` for options?", parse_mode=ParseMode.HTML)
        return

    lyrics_keywords = ["lyrics for", "lyrics to", "get lyrics", "what are the lyrics to", "find lyrics for"] # More precise
    text_lower = text.lower()
    for kw in lyrics_keywords:
        if text_lower.startswith(kw):
            lyrics_q = text[len(kw):].strip()
            if lyrics_q: 
                logger.info(f"Heuristic lyrics req: '{lyrics_q}' (user {user_id})")
                await get_lyrics_command(update, ContextTypes.DEFAULT_TYPE(application=context.application, chat_id=user_id, user_id=user_id, bot=context.bot, args=lyrics_q.split()))
                return
    
    await asyncio.sleep(0.1) # Brief pause before AI thinking
    typing_msg = await update.message.reply_text("<i>MelodyMind is thinking...</i> 🎶", parse_mode=ParseMode.HTML)
    try:
        response = await generate_chat_response(user_id, text)
        await typing_msg.edit_text(response)
    except (TimedOut, NetworkError) as net_err:
        logger.error(f"Net error (AI chat user {user_id}): {net_err}")
        await typing_msg.edit_text("Network hiccup. Try again?")
    except Exception as e:
        logger.error(f"Error in AI chat response (user {user_id}): {e}", exc_info=True)
        await typing_msg.edit_text("A bit tangled up! 😅 Try later.")

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_contexts.get(user_id, {}).get("conversation_history"):
        user_contexts[user_id]["conversation_history"] = []
        logger.info(f"Cleared chat history for user {user_id}")
        await update.message.reply_text("✅ Our chat history is cleared.")
    else: await update.message.reply_text("No chat history to clear! 😊")

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Okay, cancelled. Chat or use commands anytime! 👍")
    return ConversationHandler.END

async def analyze_conversation(user_id: int) -> Dict[str, Any]:
    default = {"genres": user_contexts.get(user_id, {}).get("preferences", []), "artists": [], "mood": user_contexts.get(user_id, {}).get("mood")}
    if not client: return default
    ctx = user_contexts.get(user_id, {})
    ctx.setdefault("preferences", []); ctx.setdefault("conversation_history", []); ctx.setdefault("spotify", {}).setdefault("recently_played", []); ctx["spotify"].setdefault("top_tracks", [])
    if len(ctx["conversation_history"]) < 1 and not ctx["spotify"]["recently_played"] and not ctx["spotify"]["top_tracks"]: # Stricter
        logger.info(f"AI Analysis: Insufficient data (user {user_id}).")
        return default
    logger.info(f"AI conversation analysis for user {user_id}")
    try:
        hist_sum = "\n".join([f"{m['role']}: {m['content'][:70]}" for m in ctx["conversation_history"][-5:]]) # Shorter summary
        spot_sum_parts = []
        if ctx["spotify"]["recently_played"]: spot_sum_parts.append("Recent: " + ", ".join([f"'{i['track']['name']}' by {i['track']['artists'][0]['name']}" for i in ctx["spotify"]["recently_played"][:2] if i.get("track")]))
        if ctx["spotify"]["top_tracks"]: spot_sum_parts.append("Top: " + ", ".join([f"'{i['name']}' by {i['artists'][0]['name']}" for i in ctx["spotify"]["top_tracks"][:2] if i.get("artists")]))
        spot_sum = ". ".join(filter(None, spot_sum_parts))
        
        prompt_content = (f"Conv Summary:\n{hist_sum}\n\nSpotify (if any):\n{spot_sum}\n\n"
                          f"Set Mood: {ctx.get('mood', 'N/A')}\nSet Prefs: {', '.join(ctx.get('preferences',[])) or 'N/A'}")
        messages=[{"role": "system", "content": "Analyze chat/Spotify. JSON: 'genres' (list, max 2), 'artists' (list, max 2 from data/chat), 'mood' (str/null). Prioritize explicit statements. Empty/null if unclear."},
                  {"role": "user", "content": prompt_content }]
        resp = await asyncio.to_thread(client.chat.completions.create, model="gpt-3.5-turbo-0125", messages=messages, max_tokens=120, temperature=0.05, response_format={"type": "json_object"})
        res = json.loads(resp.choices[0].message.content)
        if not isinstance(res, dict): logger.error(f"AI analysis non-dict (user {user_id})"); return default
        
        genres = res.get("genres", []); artists = res.get("artists", [])
        mood = str(res.get("mood","")).strip().lower() if isinstance(res.get("mood"),str) else None
        if mood == "null": mood = None

        if isinstance(genres, str): genres = [g.strip() for g in genres.split(",") if g.strip()]
        if not isinstance(genres, list): genres = []
        if isinstance(artists, str): artists = [a.strip() for a in artists.split(",") if a.strip()]
        if not isinstance(artists, list): artists = []

        if genres and (not ctx.get("preferences") or set(genres[:2]) != set(ctx.get("preferences",[])) ): ctx["preferences"] = list(set(genres[:2]))
        if mood and (mood != ctx.get("mood") or not ctx.get("mood")): ctx["mood"] = mood
        
        logger.info(f"AI analysis out (user {user_id}): G={ctx['preferences']}, M={ctx['mood']}, A={artists[:2]}")
        return {"genres": ctx["preferences"], "artists": artists[:2], "mood": ctx["mood"]}
    except Exception as e: # Broad catch for API/JSON
        logger.error(f"AI analyze_conversation error (user {user_id}): {e}", exc_info=False)
    return default

async def smart_recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id, name = update.effective_user.id, update.effective_user.first_name
    status_msg = await update.message.reply_text(f"🎵 Thinking of music for {name}...")
    try:
        if user_contexts.get(user_id, {}).get("spotify", {}).get("access_token"):
            logger.info(f"Fetching latest Spotify (user {user_id}) for smart recs.")
            # Simplified: only fetch if spotify is linked, let analyze_conversation use what's there.
            # Could pre-fetch here too if needed.
        
        analysis = await analyze_conversation(user_id)
        current_mood = analysis.get("mood")
        if not current_mood or current_mood == "neutral":
            await status_msg.delete()
            logger.info(f"Mood unclear (user {user_id}), prompting for smart recs.")
            await set_mood(update, context) # Uses the ConversationHandler to ask for mood
            return
        
        await status_msg.edit_text(f"Okay {name}, for **{current_mood}** mood...\nFinding recommendations... 🎧", parse_mode=ParseMode.MARKDOWN)
        
        s_tracks, s_artists, s_genres = [], analysis.get("artists", []), analysis.get("genres", [])
        
        # Example: Map user-friendly genre names to Spotify seed genre IDs
        # !! IMPORTANT: Populate this with actual values from GET /v1/recommendations/available-genre-seeds !!
        spotify_genre_map = {
            "pop": "pop", "rock": "rock", "hip-hop": "hip-hop", "hiphop":"hip-hop", # Check if hiphop or hip-hop is correct
            "electronic": "electronic", "dance": "dance", "edm": "edm",
            "classical": "classical", "jazz": "jazz",
            "r&b": "r-n-b", "rnb": "r-n-b", "soul": "soul", # Be specific
            "folk": "folk", "acoustic": "acoustic",
            "sad": "sad" # "sad" is often a valid mood genre in Spotify
            # ... and many more from Spotify's list
        }
        mapped_spotify_genres = [spotify_genre_map.get(g.lower()) for g in s_genres if spotify_genre_map.get(g.lower())]

        spotify_ctx = user_contexts.get(user_id, {}).get("spotify", {})
        if spotify_ctx.get("access_token"): # If Spotify linked
            # Get track IDs from Spotify recently played for seeds
            if spotify_ctx.get("recently_played"):
                s_tracks.extend([t["track"]["id"] for t in spotify_ctx["recently_played"][:2] if t.get("track") and t["track"].get("id")])
        
        # Try Spotify recommendations if client token and any seeds available
        client_token = await get_spotify_token()
        if client_token and (s_tracks or s_artists or mapped_spotify_genres):
            logger.info(f"Spotify API recs (user {user_id}) seeds: T={s_tracks}, A={s_artists}, G={mapped_spotify_genres}")
            recs = await get_spotify_recommendations(client_token, seed_tracks=s_tracks[:2], seed_artists=s_artists[:1], seed_genres=mapped_spotify_genres[:1], limit=5)
            if recs:
                html = f"🎵 Spotify recs for your **{current_mood}** mood, {name}:\n\n"
                kb = []
                for i,t in enumerate(recs, 1):
                    arts, alb = ", ".join(a["name"] for a in t.get("artists",[])), t.get("album",{}).get("name","")
                    info = f"<b>{t['name']}</b> by <i>{arts}</i>" + (f" ({alb})" if alb else "")
                    html += f"{i}. {info}\n"; yt_q = f"{t['name']} {arts}"
                    kb.append([InlineKeyboardButton(f"YT: {t['name'][:18]}...", callback_data=f"{CB_SHOW_OPTIONS_PREFIX}{yt_q}")])
                html += "\n💡 <i>Click to search on YouTube.</i>"
                await status_msg.edit_text(html, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(kb))
                return
        
        # YouTube search fallback
        yt_q_parts = [current_mood]
        if mapped_spotify_genres: yt_q_parts.append(mapped_spotify_genres[0]) # Use the mapped Spotify genre
        elif s_genres: yt_q_parts.append(s_genres[0]) # Fallback to user's raw genre if no mapping
        if s_artists: yt_q_parts.append(f"like {s_artists[0]}")
        yt_search_q = " ".join(yt_q_parts) + " music"
        logger.info(f"YT fallback recs (user {user_id}) query: '{yt_search_q}'")
        await status_msg.edit_text(f"Searching YT for **{current_mood}** tracks like '<i>{yt_search_q}</i>'...", parse_mode=ParseMode.HTML)
        yt_res = await asyncio.to_thread(search_youtube_sync, yt_search_q, max_results=5)
        if yt_res:
            html_yt = f"🎵 YT suggestions for **{current_mood}** mood, {name}:\n\n"
            kb_yt, count = [], 0
            for i,r in enumerate(yt_res,1):
                if not r.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$',r['id']): continue
                count += 1; dur, dur_s = r.get('duration',0),""
                if dur and isinstance(dur,(int,float)) and dur > 0: m,s=divmod(int(dur),60); dur_s = f" [{m}:{s:02d}]"
                html_yt += f"{count}. <b>{r['title']}</b>-<i>{r.get('uploader','N/A')}</i>{dur_s}\n"
                btn_t = r['title'][:28]+"..." if len(r['title'])>28 else r['title']
                kb_yt.append([InlineKeyboardButton(f"DL: {btn_t}",callback_data=f"{CB_DOWNLOAD_PREFIX}{r['id']}")])
            if not kb_yt: await status_msg.delete(); await provide_generic_recommendations(update, current_mood, user_id); return
            html_yt += "\n💡 <i>Click to download audio.</i>"
            await status_msg.edit_text(html_yt, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(kb_yt))
        else:
            logger.info(f"No YT results for '{yt_search_q}', generic recs for {current_mood}.")
            await status_msg.delete(); await provide_generic_recommendations(update, current_mood, user_id)
    except Exception as e:
        logger.error(f"Error in smart_recommend_music (user {user_id}): {e}", exc_info=True)
        await status_msg.edit_text(f"Snag finding recs, {name}! 😥 Try again later.")

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(msg="Exception while handling an update:", exc_info=context.error)
    err_msg_text = "😓 Oops! Something went wrong. My devs are notified. Try again in a bit!"
    if isinstance(context.error, TimedOut): err_msg_text = "🐢 Operation timed out. Try again!"
    elif isinstance(context.error, NetworkError): err_msg_text = f"📡 Network issue: {str(context.error)[:100]}. Check connection or try later."
    if isinstance(update, Update) and update.effective_message:
        try: await update.effective_message.reply_text(err_msg_text)
        except Exception as e_reply: logger.error(f"Failed to send error reply to user: {e_reply}")
    elif isinstance(update, Update) and update.callback_query:
        try: await update.callback_query.message.reply_text(err_msg_text)
        except Exception as e_reply_cb: logger.error(f"Failed to send error reply for callback: {e_reply_cb}")

def cleanup_downloads_atexit() -> None:
    logger.info("Cleaning up temp download files at exit...")
    count = 0
    try:
        if os.path.exists(DOWNLOAD_DIR):
            for item in os.listdir(DOWNLOAD_DIR):
                path = os.path.join(DOWNLOAD_DIR, item)
                try:
                    if os.path.isfile(path) and (any(item.endswith(ext) for ext in [".m4a", ".mp3", ".webm", ".ogg", ".opus"]) or "voice_" in item):
                        os.remove(path); count +=1
                except Exception as e: logger.error(f"Failed to remove {path}: {e}")
            logger.info(f"Cleaned {count} file(s) from '{DOWNLOAD_DIR}'." if count else f"No specific temp files to clean in '{DOWNLOAD_DIR}'.")
        else: logger.info(f"DL dir '{DOWNLOAD_DIR}' not found, no cleanup needed at exit.")
    except Exception as e: logger.error(f"Error during atexit cleanup of DL dir: {e}")

def signal_exit_handler(sig, frame) -> None:
    logger.info(f"Received signal {sig}, preparing for graceful exit...")
    if sig in [signal.SIGINT, signal.SIGTERM]: cleanup_downloads_atexit() 
    sys.exit(0)

def main() -> None:
    app_builder = Application.builder().token(TOKEN)
    app_builder.connect_timeout(20.0).read_timeout(30.0).write_timeout(45.0) # Longer timeouts
    app_builder.pool_timeout(180.0) # For large file uploads like audio
    app_builder.rate_limiter(AIORateLimiter(overall_max_rate=15, max_retries=2, group_max_rate=5)) # Adjust as needed
    application = app_builder.build()

    handlers = [
        CommandHandler("start", start), CommandHandler("help", help_command),
        CommandHandler("download", download_music), CommandHandler("search", search_command),
        CommandHandler("autodownload", auto_download_command), CommandHandler("lyrics", get_lyrics_command),
        CommandHandler("recommend", smart_recommend_music), CommandHandler("create_playlist", create_playlist),
        CommandHandler("clear", clear_history), CommandHandler("spotify_code", spotify_code_command),
        ConversationHandler(
            entry_points=[CommandHandler("link_spotify", link_spotify)],
            states={SPOTIFY_CODE: [MessageHandler(filters.TEXT & ~filters.COMMAND, spotify_code_handler),
                                   CommandHandler("spotify_code", spotify_code_handler),
                                   CallbackQueryHandler(cancel_spotify, pattern=f"^{CB_CANCEL_SPOTIFY}$")]},
            fallbacks=[CommandHandler("cancel", cancel)], conversation_timeout=timedelta(minutes=10).total_seconds(), per_message=True
        ),
        ConversationHandler(
            entry_points=[CommandHandler("mood", set_mood)],
            states={MOOD: [CallbackQueryHandler(enhanced_button_handler, pattern=f"^{CB_MOOD_PREFIX}")],
                    PREFERENCE: [CallbackQueryHandler(enhanced_button_handler, pattern=f"^{CB_PREFERENCE_PREFIX}")]},
            fallbacks=[CommandHandler("cancel", cancel)], conversation_timeout=timedelta(minutes=5).total_seconds(), per_message=True
        ),
        MessageHandler(filters.VOICE & ~filters.COMMAND, handle_voice),
        CallbackQueryHandler(enhanced_button_handler),
        MessageHandler(filters.TEXT & ~filters.COMMAND, enhanced_handle_message)
    ]
    for handler in handlers: application.add_handler(handler)
    application.add_error_handler(error_handler)

    signal.signal(signal.SIGINT, signal_exit_handler); signal.signal(signal.SIGTERM, signal_exit_handler)
    atexit.register(cleanup_downloads_atexit)
    logger.info("🚀 Starting MelodyMind Bot... Connecting to Telegram.")
    try: application.run_polling(allowed_updates=Update.ALL_TYPES, drop_pending_updates=True)
    except Exception as e: logger.critical(f"Bot polling CRASH: {e}", exc_info=True)
    finally: logger.info(" MelodyMind Bot has shut down.")

if __name__ == "__main__":
    if not TOKEN: logger.critical("TELEGRAM_TOKEN MISSING. Bot cannot start."); sys.exit(1)
    if not OPENAI_API_KEY: logger.warning("OPENAI_API_KEY not set. AI features degraded/disabled.")
    main()