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
from typing import Dict, List, Optional, Tuple, Any, Union
from dotenv import load_dotenv
from datetime import datetime, timedelta
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, RetryCallState
from telegram.error import TimedOut, NetworkError
import httpx
import asyncio
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes,
    filters, CallbackQueryHandler, ConversationHandler
)
from functools import lru_cache
import yt_dlp
from openai import AsyncOpenAI
import importlib
if importlib.util.find_spec("lyricsgenius") is not None:
    import lyricsgenius
else:
    lyricsgenius = None

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "https://your-callback-url.com")

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

aclient = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
genius_client = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=15, retries=2, verbose=False) if GENIUS_ACCESS_TOKEN and lyricsgenius else None

# ==================== UTILITY FUNCTIONS ====================

def sanitize_input(text: str, max_length: int = 250) -> str:
    if not text: return ""
    text = str(text)
    text = re.sub(r'[<>;]', '', text)
    text = text.strip()
    return text[:max_length]

def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return sanitized[:150]

def is_valid_youtube_url(url: str) -> bool:
    if not url: return False
    patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=',
        r'(https?://)?youtu\.be/',
        r'(https?://)?(www\.)?youtube\.com/shorts/'
    ]
    return any(re.search(pattern, url) for pattern in patterns)

# --- Spotify Link Check ---
def is_user_spotify_linked(user_id: int) -> bool:
    """Checks if the user has a potentially valid Spotify token."""
    context_data = user_contexts.get(user_id, {})
    spotify_auth = context_data.get("spotify", {})
    access_token = spotify_auth.get("access_token")
    expires_at = spotify_auth.get("expires_at", 0)
    
    # Consider a token valid if it exists and expires in more than, say, 5 minutes
    # This doesn't guarantee the token is still accepted by Spotify (could be revoked)
    # but is a good first-pass check.
    return bool(access_token and expires_at > (datetime.now(pytz.UTC).timestamp() + 300))

async def ensure_spotify_token_valid(user_id: int) -> bool:
    """Checks if Spotify token is present and tries to refresh if expired. Returns True if a valid token exists after check."""
    if is_user_spotify_linked(user_id):
        return True
    
    # If not linked or token seems expired, try to refresh (if refresh_token exists)
    context_data = user_contexts.get(user_id, {})
    if context_data.get("spotify", {}).get("refresh_token"):
        logger.info(f"Attempting to refresh Spotify token for user {user_id} during validity check.")
        refreshed_token = await refresh_spotify_token(user_id) # Uses the async wrapper
        if refreshed_token:
            logger.info(f"Spotify token successfully refreshed for user {user_id}.")
            return True
        else:
            logger.warning(f"Spotify token refresh failed for user {user_id}. User needs to re-link.")
            # Clear potentially stale/invalid spotify data if refresh failed
            if user_id in user_contexts and "spotify" in user_contexts[user_id]:
                 user_contexts[user_id]["spotify"] = {}
            return False
    return False # No access token and no refresh token


MOOD, PREFERENCE, SPOTIFY_CODE = range(3)
active_downloads = set()
user_contexts: Dict[int, Dict] = {}
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ==================== SPOTIFY HELPER FUNCTIONS (SYNC) ====================
# (These functions remain largely the same as your last provided version)
# ... (get_spotify_token_sync, search_spotify_track_sync (with type), Sould_retry_spotify_recs, 
#      get_spotify_recommendations_sync, get_user_spotify_token_sync, 
#      refresh_spotify_token_sync, get_user_spotify_data_sync, get_user_spotify_playlists_sync) ...
# For brevity, I'll skip pasting them again, assume they are correct from previous version.
# Ensure search_spotify_track_sync has the `type` parameter:
def search_spotify_track_sync(token: str, query: str, type: str = "track") -> Optional[Dict]: 
    if not token: return None
    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": query, "type": type, "limit": 1} # type can be "track", "artist", etc.
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        items_key = f"{type}s" 
        items = response.json().get(items_key, {}).get("items", [])
        return items[0] if items else None
    except (requests.exceptions.RequestException, IndexError) as e:
        logger.error(f"Error searching Spotify {type} for '{query}': {e}")
        return None

def Sould_retry_spotify_recs(retry_state: RetryCallState) -> bool:
    if retry_state.outcome.failed:
        exc = retry_state.outcome.exception()
        if isinstance(exc, requests.exceptions.HTTPError):
            if exc.response is not None and exc.response.status_code in [400, 404]:
                logger.warning(f"Spotify recommendations: Not retrying for HTTP status {exc.response.status_code} (seeds: {retry_state.args[1:] if len(retry_state.args) > 1 else 'N/A'})")
                return False
    return True

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=Sould_retry_spotify_recs
)
def get_spotify_recommendations_sync(token: str, seed_tracks: Optional[List[str]] = None, seed_artists: Optional[List[str]] = None, seed_genres: Optional[List[str]] = None, limit: int = 5) -> List[Dict]:
    if not token: logger.warning("No token for Spotify recommendations"); return []
    if not seed_tracks and not seed_artists and not seed_genres: logger.warning("No seeds for Spotify recommendations."); return []
    url = "https://api.spotify.com/v1/recommendations"; headers = {"Authorization": f"Bearer {token}"}; params = {"limit": limit}; total_seeds = 0
    if seed_tracks: tracks_to_seed = seed_tracks[:max(0, 5-total_seeds)];  (params.update({"seed_tracks": ",".join(tracks_to_seed)}), (total_seeds := total_seeds + len(tracks_to_seed))) if tracks_to_seed else None
    if seed_artists and total_seeds < 5: artists_to_seed = seed_artists[:max(0, 5-total_seeds)]; (params.update({"seed_artists": ",".join(artists_to_seed)}), (total_seeds := total_seeds + len(artists_to_seed))) if artists_to_seed else None
    if seed_genres and total_seeds < 5: genres_to_seed = seed_genres[:max(0, 5-total_seeds)]; (params.update({"seed_genres": ",".join(genres_to_seed)}), (total_seeds := total_seeds + len(genres_to_seed))) if genres_to_seed else None
    if total_seeds == 0: logger.warning("No valid seeds for Spotify recommendations."); return []
    logger.info(f"Requesting Spotify recommendations with params: {params}")
    try: response = requests.get(url, headers=headers, params=params, timeout=15); response.raise_for_status(); return response.json().get("tracks", [])
    except requests.exceptions.HTTPError as http_error: logger.error(f"Final HTTPError for Spotify recs (params: {params}): {http_error.response.status_code if http_error.response else 'N/A'} - {http_error.response.text if http_error.response else 'No text'}"); return []
    except requests.exceptions.RequestException as req_error: logger.error(f"Final RequestException for Spotify recs (params: {params}): {req_error}"); return []

def get_spotify_token_sync() -> Optional[str]:
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET: logger.warning("Spotify client credentials not configured"); return None
    auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"; auth_bytes = auth_string.encode("utf-8"); auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")
    url = "https://accounts.spotify.com/api/token"; headers = {"Authorization": f"Basic {auth_base64}", "Content-Type": "application/x-www-form-urlencoded"}; data = {"grant_type": "client_credentials"}
    try: response = requests.post(url, headers=headers, data=data, timeout=10); response.raise_for_status(); return response.json().get("access_token")
    except requests.exceptions.RequestException as e: logger.error(f"Error getting Spotify client token: {e}"); return None

def get_user_spotify_token_sync(code: str) -> Optional[Dict]:
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI: logger.warning("Spotify OAuth credentials for user token not configured"); return None
    url = "https://accounts.spotify.com/api/token"; headers = {"Authorization": f"Basic {base64.b64encode(f'{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}'.encode()).decode()}", "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "authorization_code", "code": code, "redirect_uri": SPOTIFY_REDIRECT_URI}
    try: response = requests.post(url, headers=headers, data=data, timeout=10); response.raise_for_status(); token_data = response.json(); token_data["expires_at"] = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600))).timestamp(); return token_data
    except requests.exceptions.RequestException as e: logger.error(f"Error getting user Spotify token with code: {e}"); return None

def refresh_spotify_token_sync(user_id: int) -> Optional[str]:
    context_data = user_contexts.get(user_id, {}); refresh_token = context_data.get("spotify", {}).get("refresh_token")
    if not refresh_token: logger.warning(f"No refresh token for user {user_id}."); return None
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET: logger.error("Spotify client credentials not for token refresh."); return None
    url = "https://accounts.spotify.com/api/token"; headers = {"Authorization": f"Basic {base64.b64encode(f'{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}'.encode()).decode()}", "Content-Type": "application/x-www-form-urlencoded"}; data = {"grant_type": "refresh_token", "refresh_token": refresh_token}
    try:
        response = requests.post(url, headers=headers, data=data, timeout=10); response.raise_for_status(); token_data = response.json(); expires_at = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600))).timestamp()
        if "spotify" not in user_contexts.setdefault(user_id, {}): user_contexts[user_id]["spotify"] = {}
        user_contexts[user_id]["spotify"].update({"access_token": token_data.get("access_token"), "refresh_token": token_data.get("refresh_token", refresh_token), "expires_at": expires_at})
        logger.info(f"Spotify token refreshed for user {user_id}"); return token_data.get("access_token")
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 400: logger.error(f"Invalid refresh grant for user {user_id}: {e.response.text}. Clearing token.")
        if user_id in user_contexts and "spotify" in user_contexts[user_id]: user_contexts[user_id]["spotify"] = {}
        else: logger.error(f"HTTP error refreshing Spotify token for user {user_id}: {e}")
        return None
    except requests.exceptions.RequestException as e: logger.error(f"Network error refreshing Spotify token for user {user_id}: {e}"); return None

def get_user_spotify_data_sync(user_id: int, endpoint: str) -> Optional[List[Dict]]:
    context_data = user_contexts.get(user_id, {}); spotify_data = context_data.get("spotify", {}); access_token, expires_at = spotify_data.get("access_token"), spotify_data.get("expires_at")
    if not access_token or (expires_at and datetime.now(pytz.UTC).timestamp() > expires_at):
        logger.info(f"Spotify token expired/missing for user {user_id}, refreshing for {endpoint}.")
        access_token = refresh_spotify_token_sync(user_id)
        if not access_token: logger.warning(f"Failed to refresh token for user {user_id} to fetch {endpoint}."); return None
    url = f"https://api.spotify.com/v1/me/{endpoint}"; headers = {"Authorization": f"Bearer {access_token}"}; params = {"limit": 10}
    try: response = requests.get(url, headers=headers, params=params, timeout=10); response.raise_for_status(); return response.json().get("items", [])
    except requests.exceptions.RequestException as e: logger.error(f"Error fetching Spotify user data ({endpoint}) for user {user_id}: {e}"); return None

def get_user_spotify_playlists_sync(user_id: int) -> Optional[List[Dict]]:
    context_data = user_contexts.get(user_id, {}); spotify_data = context_data.get("spotify", {}); access_token, expires_at = spotify_data.get("access_token"), spotify_data.get("expires_at")
    if not access_token or (expires_at and datetime.now(pytz.UTC).timestamp() > expires_at):
        logger.info(f"Spotify token expired/missing for user {user_id}, refreshing for playlists.")
        access_token = refresh_spotify_token_sync(user_id)
        if not access_token: logger.warning(f"Failed to refresh token for user {user_id} for playlists."); return None
    url = "https://api.spotify.com/v1/me/playlists"; headers = {"Authorization": f"Bearer {access_token}"}; params = {"limit": 10}
    try: response = requests.get(url, headers=headers, params=params, timeout=10); response.raise_for_status(); return response.json().get("items", [])
    except requests.exceptions.RequestException as e: logger.error(f"Error fetching Spotify playlists for user {user_id}: {e}"); return None

# ==================== ASYNC WRAPPERS FOR SYNC SPOTIFY HELPERS ====================
async def get_spotify_token() -> Optional[str]: return await asyncio.to_thread(get_spotify_token_sync)
async def search_spotify_track(token: str, query: str, type: str = "track") -> Optional[Dict]: return await asyncio.to_thread(search_spotify_track_sync, token, query, type)
async def get_spotify_recommendations(token: str, seed_tracks: Optional[List[str]] = None, seed_artists: Optional[List[str]] = None, seed_genres: Optional[List[str]] = None, limit: int = 5) -> List[Dict]:
    return await asyncio.to_thread(get_spotify_recommendations_sync, token, seed_tracks, seed_artists, seed_genres, limit)
async def get_user_spotify_token(code: str) -> Optional[Dict]: return await asyncio.to_thread(get_user_spotify_token_sync, code)
async def refresh_spotify_token(user_id: int) -> Optional[str]: return await asyncio.to_thread(refresh_spotify_token_sync, user_id)
async def get_user_spotify_data(user_id: int, endpoint: str) -> Optional[List[Dict]]: return await asyncio.to_thread(get_user_spotify_data_sync, user_id, endpoint)
async def get_user_spotify_playlists(user_id: int) -> Optional[List[Dict]]: return await asyncio.to_thread(get_user_spotify_playlists_sync, user_id)

# ==================== YOUTUBE HELPER FUNCTIONS (SYNC) ====================
def download_youtube_audio_sync(url: str) -> Dict[str, Any]:
    video_id_match = re.search(r'(?:v=|/|\.be/)([0-9A-Za-z_-]{11})', url)
    if not video_id_match: logger.error(f"Invalid YouTube URL/ID: {url}"); return {"success": False, "error": "Invalid YouTube URL or video ID."}
    ydl_opts = {'format': 'bestaudio[ext=m4a]/bestaudio[abr<=128]/bestaudio/best', 'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'), 'quiet': True, 'no_warnings': True, 'noplaylist': True, 'max_filesize': 50 * 1024 * 1024, 'prefer_ffmpeg': True, 'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'm4a', 'preferredquality': '128'}]}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Extracting info for URL: {url}")
            info = ydl.extract_info(url, download=False)
            if not info: return {"success": False, "error": "Could not extract video information."}
            title, artist = sanitize_filename(info.get('title', 'Unknown_Title')), info.get('artist', info.get('uploader', 'Unknown_Artist'))
            expected_path_no_ext = os.path.join(DOWNLOAD_DIR, title)
            logger.info(f"Downloading audio for: {title}")
            ydl.download([url])
            downloaded_file_path = f"{expected_path_no_ext}.m4a"
            if not os.path.exists(downloaded_file_path):
                logger.warning(f"Expected m4a file not found at {downloaded_file_path}, searching alternatives.")
                found_alternative = False
                for ext_candidate in ['webm', 'mp3', 'opus', 'ogg']:
                    alt_path = f"{expected_path_no_ext}.{ext_candidate}"
                    if os.path.exists(alt_path): downloaded_file_path = alt_path; logger.info(f"Found alternative: {downloaded_file_path}"); found_alternative = True; break
                if not found_alternative: logger.error(f"Downloaded file for '{title}' not found."); return {"success": False, "error": "Downloaded file not found. FFmpeg might be missing."}
            file_size_mb = os.path.getsize(downloaded_file_path) / (1024 * 1024)
            if file_size_mb > 50.5: os.remove(downloaded_file_path); logger.error(f"File '{title}' too large: {file_size_mb:.2f} MB."); return {"success": False, "error": "File is too large for Telegram (max 50 MB)."}
            return {"success": True, "title": title, "artist": artist, "thumbnail_url": info.get('thumbnail', ''), "duration": info.get('duration', 0), "audio_path": downloaded_file_path}
    except yt_dlp.utils.DownloadError as e:
        error_msg = f"Download failed: {str(e)[:100]}"
        if "Unsupported URL" in str(e): error_msg = "Unsupported URL."
        elif "Video unavailable" in str(e): error_msg = "Video unavailable."
        logger.error(f"YouTube download error for {url}: {e}")
        return {"success": False, "error": error_msg}
    except Exception as e:
        logger.error(f"Unexpected error downloading YouTube audio for {url}: {e}", exc_info=True)
        return {"success": False, "error": "Unexpected error during download."}

@lru_cache(maxsize=100)
def search_youtube_sync(query: str, max_results: int = 5) -> List[Dict]:
    query = sanitize_input(query)
    try:
        ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': 'discard_in_playlist', 'default_search': f'ytsearch{max_results}', 'noplaylist': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            if not info or 'entries' not in info: logger.info(f"No YouTube search results for: {query}"); return []
            results = []
            for entry in info['entries']:
                if entry and entry.get('id'): results.append({'title': entry.get('title', 'Unknown Title'), 'url': entry.get('webpage_url') or f"https://www.youtube.com/watch?v={entry['id']}", 'thumbnail': entry.get('thumbnail', ''), 'uploader': entry.get('uploader', 'Unknown Artist'), 'duration': entry.get('duration', 0), 'id': entry['id']})
            return results[:max_results]
    except Exception as e:
        logger.error(f"Error searching YouTube for '{query}': {e}", exc_info=True)
        return []

# ==================== ASYNC WRAPPERS FOR YOUTUBE HELPERS ====================
async def download_youtube_audio(url: str) -> Dict[str, Any]: return await asyncio.to_thread(download_youtube_audio_sync, url)
async def search_youtube(query: str, max_results: int = 5) -> List[Dict]: return await asyncio.to_thread(search_youtube_sync, query, max_results)

# ==================== LYRICS HELPER FUNCTIONS ====================
@lru_cache(maxsize=50)
def get_lyrics_sync(song_title: str, artist_name: Optional[str] = None) -> str:
    if not genius_client: return "Lyrics service not configured."
    song_title_s, artist_name_s = sanitize_input(song_title), sanitize_input(artist_name) if artist_name else None
    try:
        logger.info(f"Searching lyrics for: '{song_title_s}' by artist: '{artist_name_s}'")
        song = genius_client.search_song(song_title_s, artist_name_s) if artist_name_s else genius_client.search_song(song_title_s)
        if song and hasattr(song, 'lyrics') and song.lyrics:
            lyrics = song.lyrics
            lyrics = re.sub(r'^\d*ContributorsLyrics', '', lyrics, flags=re.IGNORECASE).strip()
            lyrics = re.sub(r'\[.*?\]', '', lyrics)
            lyrics = re.sub(r'\d*EmbedShare URLCopyEmbedCopy', '', lyrics, flags=re.IGNORECASE)
            lyrics = re.sub(r'\nYou might also like.*', '', lyrics, flags=re.DOTALL | re.IGNORECASE)
            lyrics = os.linesep.join([s for s in lyrics.splitlines() if s.strip()])
            if not lyrics: return f"Found '{song.title}' by {song.artist}, but lyrics empty/unclean."
            return f"🎶 Lyrics for **{song.title}** by **{song.artist}**:\n\n{lyrics}"
        else:
            search_term = f"'{song_title_s}'" + (f" by '{artist_name_s}'" if artist_name_s else "")
            return f"Sorry, couldn't find lyrics for {search_term}."
    except requests.exceptions.Timeout: logger.warning(f"Timeout searching lyrics for {song_title_s}"); return "Lyrics search timed out."
    except Exception as e: logger.error(f"Error getting lyrics for {song_title_s}: {e}", exc_info=True); return "Unexpected error fetching lyrics."

async def get_lyrics(song_title: str, artist_name: Optional[str] = None) -> str: return await asyncio.to_thread(get_lyrics_sync, song_title, artist_name)

# ==================== AI CONVERSATION FUNCTIONS ====================
async def generate_chat_response(user_id: int, message: str) -> str:
    if not aclient: return "AI service unavailable."
    message_s = sanitize_input(message)
    context_data = user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    system_prompt = "You are MelodyMind, a friendly, empathetic music companion bot. Converse naturally about music and feelings. Keep responses concise (1-3 sentences) and warm. Don't suggest commands. Acknowledge song requests gently but don't offer to download."
    messages = [{"role": "system", "content": system_prompt}]
    profile_info = []
    if context_data.get("mood"): profile_info.append(f"User's mood: {context_data['mood']}.")
    if context_data.get("preferences"): profile_info.append(f"Preferences: {', '.join(context_data['preferences'])}.")
    if context_data.get("spotify", {}).get("recently_played"):
        artists = list(set(item["track"]["artists"][0]["name"] for item in context_data["spotify"]["recently_played"][:3] if item.get("track")))
        if artists: profile_info.append(f"Recently listened to: {', '.join(artists)}.")
    if profile_info: messages.append({"role": "system", "content": "User context: " + " ".join(profile_info)})
    for hist_msg in context_data["conversation_history"][-20:]: messages.append(hist_msg)
    messages.append({"role": "user", "content": message_s})
    try:
        response = await aclient.chat.completions.create(model="gpt-3.5-turbo", messages=messages, max_tokens=150, temperature=0.75)
        reply = response.choices[0].message.content.strip()
        context_data["conversation_history"].extend([{"role": "user", "content": message_s}, {"role": "assistant", "content": reply}])
        context_data["conversation_history"] = context_data["conversation_history"][-20:]
        return reply
    except Exception as e: logger.error(f"Error in generate_chat_response for user {user_id}: {e}", exc_info=True); return "Trouble thinking. Favorite song?"

async def is_music_request(message: str) -> Dict:
    if not aclient: return {"is_music_request": False, "song_query": None}
    message_s = sanitize_input(message)
    try:
        response = await aclient.chat.completions.create(model="gpt-3.5-turbo-0125", messages=[{"role": "system", "content": "Is user requesting to play/download/find/get a specific song/music by artist? JSON: {'is_music_request': bool, 'song_query': str|null}. General music discussion is false."}, {"role": "user", "content": f"Analyze: '{message_s}'"}], max_tokens=80, temperature=0.1, response_format={"type": "json_object"})
        content = response.choices[0].message.content
        if not content: return {"is_music_request": False, "song_query": None}
        result = json.loads(content)
        is_req = result.get("is_music_request", False)
        if isinstance(is_req, str): is_req = is_req.lower() in ("true", "yes")
        song_q = result.get("song_query")
        if isinstance(song_q, str) and song_q.strip().lower() == "null": song_q = None
        return {"is_music_request": bool(is_req), "song_query": song_q.strip() if song_q else None}
    except Exception as e: logger.error(f"Error in is_music_request for '{message_s}': {e}", exc_info=True); return {"is_music_request": False, "song_query": None}

async def analyze_conversation(user_id: int) -> Dict:
    if not aclient: return {"genres": [], "artists": [], "mood": None}
    context_data = user_contexts.get(user_id, {})
    context_data.setdefault("mood", None); context_data.setdefault("preferences", []); context_data.setdefault("conversation_history", []); context_data.setdefault("spotify", {})
    if len(context_data["conversation_history"]) < 2 and not context_data["spotify"]: return {"genres": context_data["preferences"], "artists": [], "mood": context_data["mood"]}
    try:
        prompt_parts = []
        if context_data["conversation_history"]: prompt_parts.append(f"Recent conversation:\n" + "\n".join([f"{m['role']}: {m['content']}" for m in context_data["conversation_history"][-10:]]))
        spotify_summary = []
        if context_data["spotify"].get("recently_played"):
            tracks = [item['track'] for item in context_data["spotify"]["recently_played"][:5] if item.get('track')]
            track_info = [f"'{t['name']}' by {t['artists'][0]['name']}" for t in tracks if t.get('artists')]
            if track_info: spotify_summary.append(f"Recently played: {'; '.join(track_info)}.")
        if context_data["spotify"].get("top_tracks"):
            tracks = context_data["spotify"]["top_tracks"][:5]
            track_info = [f"'{t['name']}' by {t['artists'][0]['name']}" for t in tracks if t.get('artists')]
            if track_info: spotify_summary.append(f"Top tracks: {'; '.join(track_info)}.")
        if spotify_summary: prompt_parts.append("Spotify habits:\n" + "\n".join(spotify_summary))
        if not prompt_parts: return {"genres": context_data["preferences"], "artists": [], "mood": context_data["mood"]}
        user_content = "\n\n".join(prompt_parts)
        response = await aclient.chat.completions.create(model="gpt-3.5-turbo-0125", messages=[{"role": "system", "content": "Analyze conversation & Spotify. Infer mood (happy, sad, etc.), genres (pop, rock, etc.), artists. JSON: {'mood': str|null, 'genres': list[str], 'artists': list[str]}. Max 3 for lists. Concise."}, {"role": "user", "content": user_content}], max_tokens=150, temperature=0.2, response_format={"type": "json_object"})
        content = response.choices[0].message.content
        if not content: return {"genres": context_data["preferences"], "artists": [], "mood": context_data["mood"]}
        result = json.loads(content)
        in_mood = result.get("mood") if isinstance(result.get("mood"), str) else None
        in_genres = [str(g) for g in result.get("genres", []) if isinstance(g, str)][:3]
        in_artists = [str(a) for a in result.get("artists", []) if isinstance(a, str)][:3] # These are artist names
        if in_mood and (not context_data["mood"] or context_data["mood"] != in_mood) : context_data["mood"] = in_mood
        if in_genres and set(in_genres) != set(context_data["preferences"]): context_data["preferences"] = list(set(context_data["preferences"] + in_genres))[:3]
        return {"genres": in_genres or context_data["preferences"], "artists": in_artists, "mood": in_mood or context_data["mood"]}
    except Exception as e: logger.error(f"Error in analyze_conversation for user {user_id}: {e}", exc_info=True); return {"genres": context_data["preferences"], "artists": [], "mood": context_data["mood"]}

# ==================== MUSIC DETECTION FUNCTION (REGEX) ====================
def detect_music_in_message(text: str) -> Optional[str]:
    text_lower = text.lower()
    patterns = [r'(?:play|download|get|song)\s+(.+?)(?:\s+by\s+(.+))?$',]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            song_title = match.group(1).strip().rstrip(',.?!')
            artist = match.group(2).strip().rstrip(',.?!') if match.group(2) else None
            if song_title and song_title.lower() not in ["music", "a song", "something", "some music"]:
                return f"{song_title} {artist}" if artist else song_title
    music_keywords = ['music', 'song', 'track', 'tune', 'audio', 'listen to']
    if any(keyword in text_lower for keyword in music_keywords): return "AI_ANALYSIS_NEEDED"
    return None

# ==================== TELEGRAM BOT HANDLERS ====================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user; user_id = user.id
    welcome_msg = f"Hi {user.first_name}! 👋 I'm MelodyMind.\n\n"
    spotify_linked = await ensure_spotify_token_valid(user_id) # Check and refresh if needed

    if not spotify_linked:
        welcome_msg += "To unlock personalized music recommendations, please link your Spotify account using /link_spotify.\n\nYou can still use features like YouTube downloads and lyrics search.\n\n"
    else:
        welcome_msg += "Your Spotify account is linked! Let's find some great music.\n\n"
    welcome_msg += "🎵 Download music (send YouTube link or ask for a song!)\n📜 Find lyrics (`/lyrics Song Title`)\n💿 Recommend music (`/recommend` or `/mood`)\n💬 Chat about music!\n\nHow can I help?"
    await update.message.reply_text(welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("🎶 <b>MelodyMind Help</b> 🎶\n\n/start, /help, /download [URL], /search [song], /lyrics [song/artist-song], /recommend, /mood, /link_spotify, /clear.\nChat naturally too!", parse_mode=ParseMode.HTML)

async def download_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id; url = ""
    if context.args: url = " ".join(context.args)
    elif update.message and update.message.text:
        found_urls = [word for word in update.message.text.split() if is_valid_youtube_url(word)]
        if found_urls: url = found_urls[0]
        else: await update.message.reply_text("Provide YouTube URL for /download, or send link."); return
    else: await update.message.reply_text("No URL to download."); return
    if not is_valid_youtube_url(url): await update.message.reply_text("Invalid YouTube URL."); return
    if user_id in active_downloads: await update.message.reply_text("⚠️ Download in progress."); return
    active_downloads.add(user_id)
    status_msg, dl_info = None, {}
    try:
        status_msg = await update.message.reply_text("⏳ Starting download...")
        await status_msg.edit_text("🔍 Fetching info...")
        dl_info = await download_youtube_audio(url)
        if not dl_info["success"]: await status_msg.edit_text(f"❌ Download failed: {dl_info.get('error', 'Unknown')}"); return
        await status_msg.edit_text(f"✅ Downloaded: {dl_info['title']}\n⏳ Sending...")
        caption = f"🎵 {dl_info['title']}" + (f"\n🎤 Artist: {dl_info['artist']}" if dl_info.get("artist") != "Unknown_Artist" else "") + (f"\n⏱️ {int(divmod(dl_info['duration'],60)[0])}:{int(divmod(dl_info['duration'],60)[1]):02d}" if dl_info.get("duration") else "")
        logger.info(f"Sending audio: {dl_info['title']} for {user_id}")
        with open(dl_info["audio_path"], 'rb') as audio_f:
            await update.message.reply_audio(audio=audio_f, title=dl_info["title"][:64], performer=dl_info.get("artist", "Unknown")[:64], caption=caption, duration=dl_info.get('duration',0))
        await status_msg.delete()
        logger.info(f"Sent audio '{dl_info['title']}' & deleted status.")
    except (TimedOut, NetworkError) as net_err: logger.error(f"Net/Timeout error for {url}, user {user_id}: {net_err}"); msg = "⌛ Timeout/Network error."; await (status_msg.edit_text(msg) if status_msg else update.message.reply_text(msg))
    except Exception as e: logger.error(f"Unexpected error in download_music for {url}, user {user_id}: {e}", exc_info=True); err_msg = f"❌ Unexpected error: {str(e)[:100]}."; await (status_msg.edit_text(err_msg) if status_msg else update.message.reply_text(err_msg))
    finally:
        if user_id in active_downloads: active_downloads.remove(user_id)
        if dl_info.get("success") and dl_info.get("audio_path") and os.path.exists(dl_info["audio_path"]):
            try: os.remove(dl_info["audio_path"]); logger.info(f"Cleaned file: {dl_info['audio_path']}")
            except Exception as e_clean: logger.error(f"Error cleaning {dl_info['audio_path']}: {e_clean}")

async def link_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not all([SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI]) or SPOTIFY_REDIRECT_URI == "https://your-callback-url.com":
        await update.message.reply_text("Spotify linking not configured by admin."); return ConversationHandler.END
    user_id = update.effective_user.id
    auth_url = f"https://accounts.spotify.com/authorize?client_id={SPOTIFY_CLIENT_ID}&response_type=code&redirect_uri={SPOTIFY_REDIRECT_URI}&scope=user-read-recently-played%20user-top-read%20playlist-read-private%20playlist-read-collaborative&state={user_id}"
    kbd = [[InlineKeyboardButton("🔗 Link My Spotify", url=auth_url)], [InlineKeyboardButton("Cancel", callback_data="cancel_spotify_linking")]]
    await update.message.reply_text("Link Spotify: 1.Click below 2.Authorize 3.Copy `code` from URL 4.Paste code here.", reply_markup=InlineKeyboardMarkup(kbd), parse_mode=ParseMode.MARKDOWN)
    return SPOTIFY_CODE

async def spotify_code_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id; msg_text = update.message.text.strip()
    if not (msg_text and len(msg_text) > 30 and re.match(r'^[A-Za-z0-9_-]+$', msg_text)):
        reply = "Invalid Spotify code." + (" Sent command?" if msg_text.startswith('/') else "")
        await update.message.reply_text(reply + "\nPaste code or /cancel."); return SPOTIFY_CODE
    code = msg_text
    status_msg = await update.message.reply_text("⏳ Verifying Spotify code...")
    token_data = await get_user_spotify_token(code)
    if not token_data or not token_data.get("access_token"):
        await status_msg.edit_text("❌ Failed to link Spotify. Code invalid/expired? Try /link_spotify again."); return SPOTIFY_CODE
    user_contexts.setdefault(user_id, {})["spotify"] = {"access_token": token_data.get("access_token"), "refresh_token": token_data.get("refresh_token"), "expires_at": token_data.get("expires_at")}
    if await get_user_spotify_data(user_id, "player/recently-played") is not None: logger.info(f"Fetched recent Spotify for {user_id} post-link.")
    else: logger.warning(f"Could not fetch recent Spotify for {user_id} post-link.")
    await status_msg.edit_text("✅ Spotify linked! 🎉 Try /recommend."); return ConversationHandler.END

async def spotify_code_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None: # Fallback
    if not context.args: await update.message.reply_text("Use: `/spotify_code YOUR_CODE`", parse_mode=ParseMode.MARKDOWN); return
    user_id = update.effective_user.id; code = context.args[0].strip()
    if not (code and len(code) > 30 and re.match(r'^[A-Za-z0-9_-]+$', code)): await update.message.reply_text("Invalid Spotify code."); return
    status_msg = await update.message.reply_text("⏳ Verifying code (command)...")
    token_data = await get_user_spotify_token(code)
    if not token_data or not token_data.get("access_token"): await status_msg.edit_text("❌ Failed to link. Try /link_spotify."); return
    user_contexts.setdefault(user_id, {})["spotify"] = {"access_token": token_data.get("access_token"), "refresh_token": token_data.get("refresh_token"), "expires_at": token_data.get("expires_at")}
    await status_msg.edit_text("✅ Spotify linked via command! Try /recommend.")

async def cancel_spotify_linking(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query; await query.answer(); await query.edit_message_text("Spotify linking cancelled."); return ConversationHandler.END

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args: await update.message.reply_text("Search what? Ex: /search Song Artist"); return
    query = " ".join(context.args); status_msg = await update.message.reply_text(f"🔍 Searching YouTube: '{query}'...")
    results = await search_youtube(query, max_results=5); await status_msg.delete()
    await send_search_results_keyboard(update.message, query, results)

async def send_search_results_keyboard(reply_target, query: str, results: List[Dict]) -> None:
    if not results: await reply_target.reply_text(f"No YouTube songs for '{query}'."); return
    kbd, resp_text = [], f"🔎 YouTube results for '{query}':\n\n"
    for i, res in enumerate(results[:5]):
        if not res.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', res['id']): logger.warning(f"Invalid YT ID: {res.get('id')} for '{query}'"); continue
        dur_str = f" [{int(d//60)}:{int(d%60):02d}]" if (d:=res.get('duration')) else ""
        title = res['title']; resp_text += f"{i+1}. {title}{dur_str} (By: {res.get('uploader','N/A')})\n"
        btn_title = title[:40] + "..." if len(title)>40 else title
        kbd.append([InlineKeyboardButton(f"📥 {btn_title}", callback_data=f"download_{res['id']}")])
    if not kbd: await reply_target.reply_text(f"Found matches for '{query}', but no valid download options."); return
    kbd.append([InlineKeyboardButton("❌ Cancel Search", callback_data="cancel_search")])
    resp_text += "\nClick button to download audio:"
    await reply_target.reply_text(resp_text, reply_markup=InlineKeyboardMarkup(kbd))

async def auto_download_first_result(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str) -> None:
    user_id = update.effective_user.id
    if user_id in active_downloads: await update.message.reply_text("⚠️ Download in progress."); return
    active_downloads.add(user_id)
    status_msg, dl_info = await update.message.reply_text(f"🔍 Auto-searching '{query}'..."), {}
    try:
        results = await search_youtube(query, max_results=1)
        if not results or not results[0].get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', results[0]['id']):
            await status_msg.edit_text(f"❌ No valid result for '{query}' to auto-download."); active_downloads.remove(user_id); return
        res_info, vid_url = results[0], results[0]["url"]
        await status_msg.edit_text(f"✅ Found: {res_info['title']}\n⏳ Downloading...")
        dl_info = await download_youtube_audio(vid_url)
        if not dl_info["success"]: await status_msg.edit_text(f"❌ Auto-download failed: {dl_info['error']}"); return
        await status_msg.edit_text(f"✅ Downloaded: {dl_info['title']}\n⏳ Sending...")
        with open(dl_info["audio_path"], 'rb') as audio: await update.message.reply_audio(audio=audio, title=dl_info["title"][:64], performer=dl_info.get("artist","N/A")[:64], caption=f"🎵 {dl_info['title']} (Auto)")
        await status_msg.delete()
    except Exception as e: logger.error(f"Error in auto_download_first_result for '{query}': {e}", exc_info=True); err = f"❌ Error auto-download: {str(e)[:100]}"; await (status_msg.edit_text(err) if status_msg else update.message.reply_text(err))
    finally:
        if user_id in active_downloads: active_downloads.remove(user_id)
        if dl_info.get("success") and dl_info.get("audio_path") and os.path.exists(dl_info["audio_path"]):
            try: os.remove(dl_info["audio_path"])
            except Exception as e_c: logger.error(f"Error cleaning in auto_download: {e_c}")

async def auto_download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args: await update.message.reply_text("Auto-download what? Ex: /autodownload Song Artist"); return
    await auto_download_first_result(update, context, " ".join(context.args))

async def get_lyrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args: await update.message.reply_text("Lyrics for what song? Ex:\n`/lyrics Song Title`\n`/lyrics Artist - Song Title`", parse_mode=ParseMode.MARKDOWN); return
    query_full = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Searching lyrics: \"{query_full}\"...")
    try:
        artist, song_title = None, query_full
        m_as = re.match(r'^(.*?)\s*-\s*(.+)$', query_full)
        m_sba = re.match(r'^(.*?)\s+by\s+(.+)$', query_full, re.IGNORECASE)
        if m_as: p_art, p_song = m_as.groups(); artist, song_title = (p_art.strip(), p_song.strip()) if len(p_art) < len(p_song) or any(kw in p_art.lower() for kw in ["ft","feat"]) else (None, query_full)
        elif m_sba: song_title, artist = m_sba.group(1).strip(), m_sba.group(2).strip()
        lyrics_text = await get_lyrics(song_title, artist)
        await status_msg.edit_text(lyrics_text[:4090] + ("\n\n[Lyrics truncated]" if len(lyrics_text)>4090 else ""), parse_mode=ParseMode.MARKDOWN)
    except Exception as e: logger.error(f"Error in get_lyrics_command for '{query_full}': {e}", exc_info=True); await status_msg.edit_text("❌ Error fetching lyrics.")

async def provide_generic_recommendations(message_object, mood: str) -> None:
    mood = mood.lower() if mood else "general"
    searches = {"happy": ["upbeat pop playlist"], "sad": ["emotional acoustic songs"], "energetic": ["high energy workout music"], "relaxed": ["calming ambient music"], "focused": ["instrumental study music"], "nostalgic": ["classic hits playlist"]}
    sugg = searches.get(mood, searches["happy"])
    resp = f"🎵 For **{mood}** vibes, try YouTube searching for:\n\n" + "\n".join([f"{i+1}. <a href=\"https://www.youtube.com/results?search_query={requests.utils.quote(sq)}\">{sq.title()}</a>" for i,sq in enumerate(sugg)]) + "\n\n💡 Or /search [song/artist]!"
    await message_object.reply_text(resp, parse_mode=ParseMode.HTML, disable_web_page_preview=True)

async def set_mood(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    kbd = [[InlineKeyboardButton("😊 Happy", callback_data="mood_happy"), InlineKeyboardButton("😢 Sad", callback_data="mood_sad"), InlineKeyboardButton("💪 Energetic", callback_data="mood_energetic")], [InlineKeyboardButton("😌 Relaxed", callback_data="mood_relaxed"), InlineKeyboardButton("🧠 Focused", callback_data="mood_focused"), InlineKeyboardButton("🕰️ Nostalgic", callback_data="mood_nostalgic")], [InlineKeyboardButton("🚫 Skip/Other", callback_data="mood_skip")]]
    await update.message.reply_text("How are you feeling today?", reply_markup=InlineKeyboardMarkup(kbd))
    return MOOD

async def recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE, from_mood_setter: bool = False) -> Optional[int]:
    user_id = update.effective_user.id
    msg_obj = update.callback_query.message if from_mood_setter and update.callback_query else update.message
    
    # Soft mandatory Spotify linking check
    spotify_linked_and_valid = await ensure_spotify_token_valid(user_id)
    if not spotify_linked_and_valid:
        await msg_obj.reply_text(
            "To get the best personalized recommendations, please link your Spotify account first using /link_spotify. "
            "I can still give you some general YouTube suggestions based on mood!", 
            quote=False
        )
        # Proceed to give generic YouTube recommendations if Spotify is not linked.
        # Or, you could return here if you want /recommend to *only* work with Spotify.
        # For this example, we'll let it fall through to YouTube search if Spotify part fails.

    status_msg = await msg_obj.reply_text("🎧 Finding recommendations...", quote=False)
    try:
        user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
        
        if spotify_linked_and_valid: # Only fetch if token is likely valid
            logger.info(f"Updating Spotify data for user {user_id} for recommendation.")
            async def fetch_spotify_data():
                rp = await get_user_spotify_data(user_id, "player/recently-played")
                if rp is not None: user_contexts[user_id]["spotify"]["recently_played"] = rp
                tt = await get_user_spotify_data(user_id, "top/tracks?time_range=short_term")
                if tt is not None: user_contexts[user_id]["spotify"]["top_tracks"] = tt
            await fetch_spotify_data()

        ai_analysis = await analyze_conversation(user_id)
        cur_mood = ai_analysis.get("mood") or user_contexts[user_id].get("mood")
        cur_genres = list(set(ai_analysis.get("genres", []) + user_contexts[user_id].get("preferences", [])))[:3]
        cur_artists_names = ai_analysis.get("artists", []) # Names from AI

        if not cur_mood and not from_mood_setter: # If /recommend called and no mood known
            await status_msg.delete(); await msg_obj.reply_text("Need your mood for recs. Let's set it first!"); 
            return await set_mood(update, context) # Pass original update to set_mood

        s_tracks, s_artists_ids, s_genres_for_api = [], [], cur_genres 
        
        if spotify_linked_and_valid: # Only try to use Spotify seeds if linked and token likely valid
            spotify_user_data = user_contexts[user_id].get("spotify", {})
            if spotify_user_data.get("recently_played"): s_tracks.extend([t["track"]["id"] for t in spotify_user_data["recently_played"][:2] if t.get("track") and t["track"].get("id")])
            if len(s_tracks) < 2 and spotify_user_data.get("top_tracks"): s_tracks.extend([t["id"] for t in spotify_user_data["top_tracks"][:(2-len(s_tracks))] if t.get("id")])
            
            if cur_artists_names and (len(s_tracks) + len(s_artists_ids)) < 5 :
                sp_token_for_search = await get_spotify_token() # Client credential token for search
                if sp_token_for_search:
                    for name in cur_artists_names[:max(0, 5 - len(s_tracks) - len(s_artists_ids))]: # Fill up to 5 seeds
                        artist_data = await search_spotify_track(sp_token_for_search, f"artist:{name.strip()}", type="artist")
                        if artist_data and artist_data.get("id"): s_artists_ids.append(artist_data["id"])
        
        if spotify_linked_and_valid and (s_tracks or s_artists_ids or s_genres_for_api):
            logger.info(f"Spotify recs for {user_id} with seeds: tracks={s_tracks}, artists={s_artists_ids}, genres={s_genres_for_api}")
            sp_token_for_recs = await get_spotify_token() # Client credential token
            if sp_token_for_recs:
                sp_recs = await get_spotify_recommendations(sp_token_for_recs, seed_tracks=s_tracks, seed_artists=s_artists_ids, seed_genres=s_genres_for_api, limit=5)
                if sp_recs:
                    resp = f"🎵 Based on your vibe (mood: {cur_mood or 'general'}), Spotify recommendations:\n\n" + "\n".join([f"{i+1}. <a href=\"{t.get('external_urls',{}).get('spotify','#')}\"><b>{t['name']}</b></a> by {', '.join(a['name'] for a in t.get('artists',[]))}" + (f" (<i>{t.get('album',{}).get('name','')}</i>)" if t.get('album') else "") for i,t in enumerate(sp_recs)]) + "\n\n💡 Ask me to download these!"
                    await status_msg.edit_text(resp, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                    return ConversationHandler.END if from_mood_setter else None
        
        logger.info(f"Falling back to YouTube search for {user_id} recommendation (Spotify linked: {spotify_linked_and_valid}).")
        yt_query_parts = []
        if cur_mood: yt_query_parts.append(cur_mood)
        if cur_genres: yt_query_parts.append(cur_genres[0]) # Use primary genre
        yt_query_parts.append("music")
        if cur_artists_names: yt_query_parts.append(f"like {cur_artists_names[0]}")
        yt_query = " ".join(yt_query_parts) if yt_query_parts and " ".join(yt_query_parts).strip() != "music" else f"{cur_mood or 'popular'} music playlist"
        
        yt_res = await search_youtube(yt_query, max_results=5); await status_msg.delete()
        if yt_res: await send_search_results_keyboard(msg_obj, yt_query, yt_res)
        else: await provide_generic_recommendations(msg_obj, cur_mood or "happy")
        return ConversationHandler.END if from_mood_setter else None
    except Exception as e: logger.error(f"Error in recommend_music for {user_id}: {e}", exc_info=True); err="Recs error."; await (status_msg.edit_text(err) if status_msg else msg_obj.reply_text(err)); return ConversationHandler.END if from_mood_setter else None

async def enhanced_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    query = update.callback_query; await query.answer(); data = query.data; user_id = query.from_user.id
    logger.debug(f"Button: '{data}' from {user_id}"); user_contexts.setdefault(user_id, {})
    if data.startswith("mood_"):
        mood = data.split("_",1)[1]
        if mood == "skip": await query.edit_message_text("Okay, no mood set."); await recommend_music(update,context,True); return ConversationHandler.END
        user_contexts[user_id]["mood"] = mood; logger.info(f"User {user_id} mood: {mood}")
        kbd = [[InlineKeyboardButton("Pop",callback_data="pref_pop"),InlineKeyboardButton("Rock",callback_data="pref_rock"),InlineKeyboardButton("Hip-Hop",callback_data="pref_hiphop")],[InlineKeyboardButton("Electronic",callback_data="pref_electronic"),InlineKeyboardButton("Classical",callback_data="pref_classical"),InlineKeyboardButton("Indie/Alt",callback_data="pref_indie")],[InlineKeyboardButton("No Preference / Skip",callback_data="pref_skip")]]
        await query.edit_message_text(f"Feeling {mood}. Genre preference?", reply_markup=InlineKeyboardMarkup(kbd)); return PREFERENCE
    elif data.startswith("pref_"):
        pref = data.split("_",1)[1]
        if pref != "skip":
            prefs = user_contexts[user_id].setdefault("preferences", [])
            if pref not in prefs: prefs.append(pref)
            user_contexts[user_id]["preferences"] = prefs[:3]; logger.info(f"User {user_id} pref: {pref}")
            await query.edit_message_text(f"Noted {pref} preference!")
        else: await query.edit_message_text("No specific genre noted.")
        await recommend_music(update,context,True); return ConversationHandler.END
    elif data.startswith("download_") or data.startswith("auto_download_"):
        vid_id = data.split("_",2)[2] if data.startswith("auto_download_") else data.split("_",1)[1]
        if not re.match(r'^[0-9A-Za-z_-]{11}$', vid_id): await query.edit_message_text("❌ Invalid video ID."); return None
        url = f"https://www.youtube.com/watch?v={vid_id}"
        if user_id in active_downloads: await query.edit_message_text("⚠️ Download in progress."); return None
        active_downloads.add(user_id); await query.edit_message_text(f"⏳ Starting download for {vid_id}...")
        dl_info_cb = {}
        try:
            dl_info_cb = await download_youtube_audio(url)
            if not dl_info_cb["success"]: await query.edit_message_text(f"❌ Download failed: {dl_info_cb.get('error','Unknown')}"); return None
            await query.edit_message_text(f"✅ Downloaded: {dl_info_cb['title']}\n⏳ Sending...")
            with open(dl_info_cb["audio_path"],'rb') as audio_f: await context.bot.send_audio(chat_id=query.message.chat_id, audio=audio_f, title=dl_info_cb["title"][:64], performer=dl_info_cb.get("artist","N/A")[:64], caption=f"🎵 {dl_info_cb['title']}")
            await query.edit_message_text(f"✅ Sent: {dl_info_cb['title']}")
        except (TimedOut, NetworkError) as te: logger.error(f"Net/Timeout sending {vid_id}: {te}", exc_info=True); err="❌ Failed send (network)."; await (query.edit_message_text(err) if query.message else context.bot.send_message(chat_id=user_id, text=err)) # handle if original message was deleted
        except Exception as e: logger.error(f"Error in button dl {vid_id}: {e}", exc_info=True); err=f"❌ Error: {str(e)[:100]}"; await (query.edit_message_text(err) if query.message else context.bot.send_message(chat_id=user_id, text=err))
        finally:
            if user_id in active_downloads: active_downloads.remove(user_id)
            if dl_info_cb.get("success") and dl_info_cb.get("audio_path") and os.path.exists(dl_info_cb["audio_path"]):
                try: os.remove(dl_info_cb["audio_path"])
                except Exception as e_c: logger.error(f"Error cleaning in button_handler: {e_c}")
        return None
    elif data.startswith("show_options_"):
        s_query = data.split("show_options_",1)[1]; await query.edit_message_text(f"🔍 Fetching options for '{s_query}'...")
        results = await search_youtube(s_query, max_results=5)
        if not results: await query.edit_message_text(f"No other options for '{s_query}'."); return None
        await query.message.delete(); await send_search_results_keyboard(query.message, s_query, results); return None
    elif data == "cancel_search": await query.edit_message_text("❌ Search/download cancelled."); return None
    elif data == "cancel_spotify_linking": await query.edit_message_text("Spotify linking cancelled."); return ConversationHandler.END
    return None

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5), reraise=True)
async def reply_with_retry(msg_obj, text, **kwargs): return await msg_obj.reply_text(text, **kwargs)

async def enhanced_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text: return
    user_id = update.effective_user.id; text = sanitize_input(update.message.text)
    logger.debug(f"Msg from {user_id}: \"{text[:50]}...\"")
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    typing_task = asyncio.create_task(context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing'))
    try:
        if is_valid_youtube_url(text): await typing_task; context.args=[text]; await download_music(update,context); return
        
        # Try regex detection first
        detected_song_regex = detect_music_in_message(text)
        if detected_song_regex and detected_song_regex != "AI_ANALYSIS_NEEDED":
            await typing_task # Ensure typing is done before reply
            status_msg = await reply_with_retry(update.message, f"🔍 You asked for '{detected_song_regex}'. Searching YouTube...")
            results = await search_youtube(detected_song_regex, max_results=3)
            await status_msg.delete()
            if results: await send_search_results_keyboard(update.message, detected_song_regex, results)
            else: await reply_with_retry(update.message, f"Sorry, couldn't find '{detected_song_regex}' on YouTube.")
            return

        # If regex didn't trigger or needs AI, proceed to AI intent detection
        ai_intent = {"is_music_request": False, "song_query": None, "is_lyrics_request": False}
        if len(text.split()) > 1 or detected_song_regex == "AI_ANALYSIS_NEEDED":
            lyrics_kws = ["lyrics for", "lyrics to", "words to the song", "words for the song", "lyrics of", "lyrics"]
            txt_lower = text.lower(); actual_song_q_lyrics = None
            for kw in lyrics_kws: # Try to find the keyword and take text after it
                if kw in txt_lower:
                    parts = txt_lower.split(kw, 1)
                    if len(parts) > 1 and parts[1].strip("?.! "): # Ensure there's text after keyword
                        actual_song_q_lyrics = parts[1].strip("?.! ")
                        break 
            if actual_song_q_lyrics: 
                ai_intent["is_lyrics_request"], ai_intent["song_query"] = True, actual_song_q_lyrics
            elif not ai_intent["is_lyrics_request"]: # Only check for music download if not lyrics
                music_req_ai = await is_music_request(text)
                if music_req_ai["is_music_request"] and music_req_ai["song_query"]: 
                    ai_intent["is_music_request"], ai_intent["song_query"] = True, music_req_ai["song_query"]
        
        await typing_task # ensure awaited/cancelled
        if ai_intent["is_music_request"] and ai_intent["song_query"]:
            status_msg_ai = await reply_with_retry(update.message, f"🔍 Searching YouTube for '{ai_intent['song_query']}' (AI)...")
            results = await search_youtube(ai_intent['song_query'],max_results=3); await status_msg_ai.delete()
            if results: await send_search_results_keyboard(update.message,ai_intent['song_query'],results)
            else: await reply_with_retry(update.message, f"Sorry, couldn't find '{ai_intent['song_query']}' on YouTube (AI).")
            return
        if ai_intent["is_lyrics_request"] and ai_intent["song_query"]: 
            context.args=[ai_intent["song_query"]]; await get_lyrics_command(update,context); return
        
        txt_l = text.lower()
        if "i'm feeling" in txt_l or "i feel" in txt_l:
            try: mood_tok = txt_l.split("i'm feeling" if "i'm feeling" in txt_l else "i feel",1)[1].strip().split()[0].rstrip('.,?!')
            if mood_tok in ["happy","sad","energetic","relaxed","focused","nostalgic","anxious","stressed","calm","excited"]: user_contexts[user_id]["mood"]=mood_tok; logger.info(f"Mood '{mood_tok}' from text for {user_id}")
            except IndexError: pass # Mood phrase detected but no actual mood word followed
        chat_resp = await generate_chat_response(user_id,text); await reply_with_retry(update.message,chat_resp)
    except RetryError as re_err: logger.warning(f"Send/edit failed for {user_id}: {re_err}")
    except Exception as e: logger.error(f"Error in enhanced_handle_message for {user_id}, text \"{text[:50]}...\": {e}",exc_info=True); try:await reply_with_retry(update.message,"Pardon, hit a snag. Try /help?")
    finally:
        if not typing_task.done(): typing_task.cancel()

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id in user_contexts and "conversation_history" in user_contexts[user_id]: user_contexts[user_id]["conversation_history"]=[]; await update.message.reply_text("✅ Chat history cleared.")
    else: await update.message.reply_text("No chat history to clear.")

async def cancel_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    msg = "Okay, action cancelled."
    if update.callback_query: await update.callback_query.answer(); await (update.callback_query.edit_message_text(msg) if update.callback_query.message else context.bot.send_message(update.effective_chat.id, msg))
    elif update.message: await update.message.reply_text(msg)
    logger.info(f"Conversation cancelled for {update.effective_user.id}"); return ConversationHandler.END

async def handle_telegram_error(update: Optional[object], context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Update: {update} caused error: {context.error}", exc_info=context.error)
    err_msg = "Oops! Something went wrong.🛠️"
    if isinstance(context.error,(TimedOut,NetworkError)): err_msg = "Trouble connecting.🌐 Try again."
    chat_id, msg_id = (None, None)
    if update and hasattr(update,'effective_chat') and update.effective_chat: chat_id = update.effective_chat.id
    if update and hasattr(update,'effective_message') and update.effective_message: msg_id = update.effective_message.message_id; chat_id = chat_id or update.effective_message.chat_id
    if chat_id:
        try:
            if msg_id and hasattr(update,'effective_message'): await update.effective_message.reply_text(err_msg)
            else: await context.bot.send_message(chat_id=chat_id, text=err_msg)
        except Exception as e: logger.error(f"Failed to send error notification to {chat_id}: {e}")

def cleanup_downloads_sync() -> None:
    logger.info(f"Cleaning download dir: {DOWNLOAD_DIR}"); cleaned, errors = 0,0
    if os.path.exists(DOWNLOAD_DIR):
        for item in os.listdir(DOWNLOAD_DIR):
            item_path = os.path.join(DOWNLOAD_DIR,item)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path): os.unlink(item_path); cleaned+=1
                elif os.path.isdir(item_path): logger.warning(f"Unexpected subdir in downloads: {item_path}")
            except Exception as e: logger.error(f"Error removing {item_path}: {e}"); errors+=1
        logger.info(f"Cleaned {cleaned} file(s) from {DOWNLOAD_DIR}. {errors} errors.")
    else: logger.info(f"{DOWNLOAD_DIR} not found, no cleanup.")

async def cleanup_downloads(): await asyncio.to_thread(cleanup_downloads_sync)
def sig_handler(sig, frame): logger.info(f"Signal {sig}, shutting down..."); cleanup_downloads_sync(); logger.info("Cleanup done. Exit."); sys.exit(0)

def main() -> None:
    req_vars = ["TELEGRAM_TOKEN","OPENAI_API_KEY","SPOTIFY_CLIENT_ID","SPOTIFY_CLIENT_SECRET","SPOTIFY_REDIRECT_URI","GENIUS_ACCESS_TOKEN"]
    missing = [v for v in req_vars if not os.getenv(v)]
    if missing: logger.critical(f"FATAL: Missing env vars: {', '.join(missing)}. Bot cannot start."); sys.exit(1)
    if os.getenv("SPOTIFY_REDIRECT_URI") == "https://your-callback-url.com": logger.warning("SPOTIFY_REDIRECT_URI default. Spotify OAuth will NOT work.")
    app = (Application.builder().token(TOKEN).read_timeout(20).write_timeout(75).connect_timeout(15).pool_timeout(60).get_updates_read_timeout(40).build())
    app.add_handler(CommandHandler("start",start)); app.add_handler(CommandHandler("help",help_command)); app.add_handler(CommandHandler("download",download_music)); app.add_handler(CommandHandler("search",search_command)); app.add_handler(CommandHandler("autodownload",auto_download_command)); app.add_handler(CommandHandler("lyrics",get_lyrics_command)); app.add_handler(CommandHandler("recommend",recommend_music)); app.add_handler(CommandHandler("clear",clear_history)); app.add_handler(CommandHandler("spotify_code",spotify_code_command))
    spotify_conv = ConversationHandler(entry_points=[CommandHandler("link_spotify",link_spotify)], states={SPOTIFY_CODE:[MessageHandler(filters.TEXT & ~filters.COMMAND,spotify_code_handler),CallbackQueryHandler(cancel_spotify_linking,pattern="^cancel_spotify_linking$")]}, fallbacks=[CommandHandler("cancel",cancel_conversation),CallbackQueryHandler(cancel_spotify_linking,pattern="^cancel_spotify_linking$")], conversation_timeout=300, per_message=True)
    app.add_handler(spotify_conv)
    mood_conv = ConversationHandler(entry_points=[CommandHandler("mood",set_mood)], states={MOOD:[CallbackQueryHandler(enhanced_button_handler,pattern="^mood_")], PREFERENCE:[CallbackQueryHandler(enhanced_button_handler,pattern="^pref_")]}, fallbacks=[CommandHandler("cancel",cancel_conversation)], conversation_timeout=180, per_message=True)
    app.add_handler(mood_conv)
    app.add_handler(CallbackQueryHandler(enhanced_button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND,enhanced_handle_message))
    app.add_error_handler(handle_telegram_error)
    signal.signal(signal.SIGINT,sig_handler); signal.signal(signal.SIGTERM,sig_handler)
    atexit.register(cleanup_downloads_sync)
    logger.info("Initial cleanup..."); cleanup_downloads_sync()
    logger.info("Starting MelodyMind Bot...")
    try: app.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e: logger.critical(f"Critical error running bot: {e}", exc_info=True)
    finally: logger.info("Bot stopped. Final cleanup..."); cleanup_downloads_sync()

if __name__ == "__main__":
    main()