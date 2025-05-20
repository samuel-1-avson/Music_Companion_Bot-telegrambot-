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
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError
from telegram.error import TimedOut, NetworkError
import httpx # Already used by PTB, good for direct async calls if needed
import asyncio
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
# Telegram imports
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes,
    filters, CallbackQueryHandler, ConversationHandler
)
from functools import lru_cache
# API clients
import yt_dlp
# Use AsyncOpenAI for async operations
from openai import AsyncOpenAI
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
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "https://your-callback-url.com") # Keep default for warning

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize clients
aclient = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
genius_client = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=15, retries=2) if GENIUS_ACCESS_TOKEN and lyricsgenius else None

# ==================== UTILITY FUNCTIONS ====================

def sanitize_input(text: str, max_length: int = 250) -> str:
    """Sanitize user input to remove potentially harmful characters,
    control length, and trim whitespace.
    """
    if not text:
        return ""
    text = str(text) # Ensure it's a string
    # Remove characters that might be problematic in filenames, queries, or HTML contexts
    text = re.sub(r'[<>;]', '', text) # Remove some common injection/html chars
    text = text.strip() # Remove leading/trailing whitespace
    return text[:max_length] # Truncate to max_length

def sanitize_filename(filename: str) -> str:
    """Remove invalid characters from filenames."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return sanitized[:150] # Increased length slightly for very long titles

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


# Conversation states
MOOD, PREFERENCE, SPOTIFY_CODE = range(3)

# Track active downloads and user contexts
active_downloads = set() # Stores user_ids of users with active downloads
user_contexts: Dict[int, Dict] = {} # Stores user-specific data
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ==================== SPOTIFY HELPER FUNCTIONS (SYNC) ====================
# These will be called with asyncio.to_thread

def get_spotify_token_sync() -> Optional[str]:
    """Get Spotify access token using client credentials."""
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

def search_spotify_track_sync(token: str, query: str) -> Optional[Dict]:
    """Search for a track on Spotify."""
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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_spotify_recommendations_sync(token: str, seed_tracks: List[str], limit: int = 5) -> List[Dict]:
    """Get track recommendations from Spotify."""
    if not token or not seed_tracks:
        logger.warning("No token or seed tracks for Spotify recommendations")
        return []
    url = "https://api.spotify.com/v1/recommendations"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"seed_tracks": ",".join(seed_tracks[:5]), "limit": limit}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        return response.json().get("tracks", [])
    except requests.exceptions.HTTPError as http_error:
        logger.warning(f"Spotify recommendations HTTPError for seeds {seed_tracks}: {http_error.response.status_code} - {http_error.response.text if http_error.response else 'No response'}")
        if http_error.response and http_error.response.status_code == 400:
            return []
        raise
    except requests.exceptions.RequestException as req_error:
        logger.error(f"Error getting Spotify recommendations: {req_error}")
        return []

def get_user_spotify_token_sync(code: str) -> Optional[Dict]:
    """Exchange authorization code for Spotify access and refresh tokens."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI:
        logger.warning("Spotify OAuth credentials for user token not configured")
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
        response = requests.post(url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        token_data = response.json()
        token_data["expires_at"] = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600))).timestamp()
        return token_data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting user Spotify token with code: {e}")
        return None

def refresh_spotify_token_sync(user_id: int) -> Optional[str]:
    """Refresh Spotify access token using refresh token."""
    context = user_contexts.get(user_id, {})
    refresh_token = context.get("spotify", {}).get("refresh_token")
    if not refresh_token:
        logger.warning(f"No refresh token found for user {user_id} to refresh.")
        return None

    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        logger.error("Spotify client credentials not configured for token refresh.")
        return None

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}'.encode()).decode()}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token}

    try:
        response = requests.post(url, headers=headers, data=data, timeout=10)
        response.raise_for_status()
        token_data = response.json()
        expires_at = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600))).timestamp()
        
        if "spotify" not in user_contexts.setdefault(user_id, {}):
            user_contexts[user_id]["spotify"] = {}
            
        user_contexts[user_id]["spotify"].update({
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token", refresh_token),
            "expires_at": expires_at
        })
        logger.info(f"Spotify token refreshed for user {user_id}")
        return token_data.get("access_token")
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 400:
            logger.error(f"Invalid refresh token or grant for user {user_id}: {e.response.text}. Clearing stored token.")
            if user_id in user_contexts and "spotify" in user_contexts[user_id]:
                user_contexts[user_id]["spotify"] = {}
        else:
            logger.error(f"HTTP error refreshing Spotify token for user {user_id}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error refreshing Spotify token for user {user_id}: {e}")
        return None

def get_user_spotify_data_sync(user_id: int, endpoint: str) -> Optional[List[Dict]]:
    """Fetch user-specific Spotify data (recently played or top tracks)."""
    context = user_contexts.get(user_id, {})
    spotify_data = context.get("spotify", {})
    access_token = spotify_data.get("access_token")
    expires_at = spotify_data.get("expires_at")

    if not access_token or (expires_at and datetime.now(pytz.UTC).timestamp() > expires_at):
        logger.info(f"Spotify token expired or missing for user {user_id}, attempting refresh for endpoint {endpoint}.")
        access_token = refresh_spotify_token_sync(user_id)
        if not access_token:
            logger.warning(f"Failed to refresh Spotify token for user {user_id} to fetch {endpoint}.")
            return None

    url = f"https://api.spotify.com/v1/me/{endpoint}"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"limit": 10}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("items", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Spotify user data ({endpoint}) for user {user_id}: {e}")
        return None

def get_user_spotify_playlists_sync(user_id: int) -> Optional[List[Dict]]:
    """Fetch user's Spotify playlists."""
    context = user_contexts.get(user_id, {})
    spotify_data = context.get("spotify", {})
    access_token = spotify_data.get("access_token")
    expires_at = spotify_data.get("expires_at")

    if not access_token or (expires_at and datetime.now(pytz.UTC).timestamp() > expires_at):
        logger.info(f"Spotify token expired or missing for user {user_id}, attempting refresh for playlists.")
        access_token = refresh_spotify_token_sync(user_id)
        if not access_token:
            logger.warning(f"Failed to refresh Spotify token for user {user_id} to fetch playlists.")
            return None

    url = "https://api.spotify.com/v1/me/playlists"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"limit": 10}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("items", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Spotify playlists for user {user_id}: {e}")
        return None

# ==================== ASYNC WRAPPERS FOR SYNC SPOTIFY HELPERS ====================
async def get_spotify_token() -> Optional[str]:
    return await asyncio.to_thread(get_spotify_token_sync)

async def search_spotify_track(token: str, query: str) -> Optional[Dict]:
    return await asyncio.to_thread(search_spotify_track_sync, token, query)

async def get_spotify_recommendations(token: str, seed_tracks: List[str], limit: int = 5) -> List[Dict]:
    return await asyncio.to_thread(get_spotify_recommendations_sync, token, seed_tracks, limit)

async def get_user_spotify_token(code: str) -> Optional[Dict]:
    return await asyncio.to_thread(get_user_spotify_token_sync, code)

async def refresh_spotify_token(user_id: int) -> Optional[str]:
    return await asyncio.to_thread(refresh_spotify_token_sync, user_id)

async def get_user_spotify_data(user_id: int, endpoint: str) -> Optional[List[Dict]]:
    return await asyncio.to_thread(get_user_spotify_data_sync, user_id, endpoint)

async def get_user_spotify_playlists(user_id: int) -> Optional[List[Dict]]:
    return await asyncio.to_thread(get_user_spotify_playlists_sync, user_id)


# ==================== YOUTUBE HELPER FUNCTIONS (SYNC) ====================

def download_youtube_audio_sync(url: str) -> Dict[str, Any]:
    """Download audio from a YouTube video with improved error handling."""
    video_id_match = re.search(r'(?:v=|/|\.be/)([0-9A-Za-z_-]{11})', url)
    if not video_id_match:
        logger.error(f"Invalid YouTube URL or could not extract video ID: {url}")
        return {"success": False, "error": "Invalid YouTube URL or video ID."}

    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio[abr<=128]/bestaudio/best',
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'max_filesize': 50 * 1024 * 1024,
        'prefer_ffmpeg': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
            'preferredquality': '128',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Attempting to extract info for URL: {url}")
            info = ydl.extract_info(url, download=False)
            if not info:
                return {"success": False, "error": "Could not extract video information."}
            
            title = sanitize_filename(info.get('title', 'Unknown_Title'))
            artist = info.get('artist', info.get('uploader', 'Unknown_Artist'))
            expected_path_no_ext = os.path.join(DOWNLOAD_DIR, title)
            
            logger.info(f"Attempting to download audio for: {title}")
            ydl.download([url])

            downloaded_file_path = f"{expected_path_no_ext}.m4a"
            if not os.path.exists(downloaded_file_path):
                logger.warning(f"Expected m4a file not found at {downloaded_file_path}, searching for alternatives.")
                found_alternative = False
                for ext_candidate in ['webm', 'mp3', 'opus', 'ogg']: # Common alternatives
                    alt_path = f"{expected_path_no_ext}.{ext_candidate}"
                    if os.path.exists(alt_path):
                        downloaded_file_path = alt_path
                        logger.info(f"Found alternative audio file: {downloaded_file_path}")
                        found_alternative = True
                        break
                if not found_alternative:
                    logger.error(f"Downloaded file for '{title}' not found after download attempt. Searched for .m4a and alternatives.")
                    return {"success": False, "error": "Downloaded file not found. FFmpeg might be missing or an issue with postprocessing."}

            file_size_mb = os.path.getsize(downloaded_file_path) / (1024 * 1024)
            if file_size_mb > 50.5:
                os.remove(downloaded_file_path)
                logger.error(f"File '{title}' too large: {file_size_mb:.2f} MB (limit 50MB).")
                return {"success": False, "error": "File is too large for Telegram (max 50 MB)."}
            
            return {
                "success": True,
                "title": title,
                "artist": artist,
                "thumbnail_url": info.get('thumbnail', ''),
                "duration": info.get('duration', 0),
                "audio_path": downloaded_file_path
            }
    except yt_dlp.utils.DownloadError as e:
        error_msg = f"Download failed: {str(e)[:100]}"
        if "Unsupported URL" in str(e): error_msg = "Unsupported URL. Please provide a valid YouTube link."
        elif "Video unavailable" in str(e): error_msg = "This video is unavailable."
        # Add more specific messages as needed
        logger.error(f"YouTube download error for {url}: {e}")
        return {"success": False, "error": error_msg}
    except Exception as e:
        logger.error(f"Unexpected error downloading YouTube audio for {url}: {e}", exc_info=True)
        return {"success": False, "error": "An unexpected error occurred during download."}

@lru_cache(maxsize=100)
def search_youtube_sync(query: str, max_results: int = 5) -> List[Dict]:
    query = sanitize_input(query)
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'discard_in_playlist',
            'default_search': f'ytsearch{max_results}',
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            if not info or 'entries' not in info:
                logger.info(f"No YouTube search results for query: {query}")
                return []
            
            results = []
            for entry in info['entries']:
                if entry and entry.get('id'):
                    results.append({
                        'title': entry.get('title', 'Unknown Title'),
                        'url': entry.get('webpage_url') or f"https://www.youtube.com/watch?v={entry['id']}",
                        'thumbnail': entry.get('thumbnail', ''),
                        'uploader': entry.get('uploader', 'Unknown Artist'),
                        'duration': entry.get('duration', 0),
                        'id': entry['id']
                    })
            return results[:max_results]
    except Exception as e:
        logger.error(f"Error searching YouTube for '{query}': {e}", exc_info=True)
        return []

# ==================== ASYNC WRAPPERS FOR YOUTUBE HELPERS ====================
async def download_youtube_audio(url: str) -> Dict[str, Any]:
    return await asyncio.to_thread(download_youtube_audio_sync, url)

async def search_youtube(query: str, max_results: int = 5) -> List[Dict]:
    return await asyncio.to_thread(search_youtube_sync, query, max_results)

# ==================== LYRICS HELPER FUNCTIONS ====================
@lru_cache(maxsize=50)
def get_lyrics_sync(song_title: str, artist_name: Optional[str] = None) -> str:
    if not genius_client:
        return "Lyrics service is not configured or unavailable."
    
    song_title_s = sanitize_input(song_title)
    artist_name_s = sanitize_input(artist_name) if artist_name else None

    try:
        logger.info(f"Searching lyrics for song: '{song_title_s}' by artist: '{artist_name_s}'")
        if artist_name_s:
            song = genius_client.search_song(song_title_s, artist_name_s)
        else:
            song = genius_client.search_song(song_title_s)

        if song and hasattr(song, 'lyrics') and song.lyrics:
            lyrics = song.lyrics
            lyrics = re.sub(r'^\d*ContributorsLyrics', '', lyrics, flags=re.IGNORECASE).strip()
            lyrics = re.sub(r'\[.*?\]', '', lyrics)
            lyrics = re.sub(r'\d*EmbedShare URLCopyEmbedCopy', '', lyrics, flags=re.IGNORECASE)
            lyrics = re.sub(r'\nYou might also like.*', '', lyrics, flags=re.DOTALL | re.IGNORECASE) # DOTALL for multiline
            lyrics = os.linesep.join([s for s in lyrics.splitlines() if s.strip()])

            if not lyrics:
                return f"Found song '{song.title}' by {song.artist}, but lyrics seem empty or couldn't be cleaned."
            return f"🎶 Lyrics for **{song.title}** by **{song.artist}**:\n\n{lyrics}"
        else:
            search_term = f"'{song_title_s}'"
            if artist_name_s: search_term += f" by '{artist_name_s}'"
            return f"Sorry, I couldn't find lyrics for {search_term}."
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout searching lyrics for {song_title_s}")
        return "Sorry, the lyrics search timed out. Please try again."
    except Exception as e:
        logger.error(f"Error getting lyrics for {song_title_s}: {e}", exc_info=True)
        return "An unexpected error occurred while fetching lyrics."

async def get_lyrics(song_title: str, artist_name: Optional[str] = None) -> str:
    return await asyncio.to_thread(get_lyrics_sync, song_title, artist_name)


# ==================== AI CONVERSATION FUNCTIONS ====================

async def generate_chat_response(user_id: int, message: str) -> str:
    if not aclient:
        return "I'm having trouble connecting to my AI service. Please try again later."

    message_s = sanitize_input(message)
    context = user_contexts.setdefault(user_id, {
        "mood": None, "preferences": [], "conversation_history": [], "spotify": {}
    })

    system_prompt_content = (
        "You are a friendly, empathetic music companion bot named MelodyMind. "
        "Your role is to: "
        "1. Have natural conversations about music and feelings. "
        "2. Recommend songs based on mood and preferences (but don't list songs unless explicitly asked or very relevant). "
        "3. Provide emotional support through music-related conversation. "
        "4. Keep responses concise (1-3 sentences) and warm. "
        "5. Do not suggest commands like /recommend or /lyrics. Engage naturally. "
        "6. If you sense the user wants a specific song, you can gently acknowledge it but don't offer to play or download. "
    )
    messages = [{"role": "system", "content": system_prompt_content}]

    user_profile_info = []
    if context.get("mood"): user_profile_info.append(f"User's current mood: {context['mood']}.")
    if context.get("preferences"): user_profile_info.append(f"User's music preferences: {', '.join(context['preferences'])}.")
    if context.get("spotify", {}).get("recently_played"):
        artists = list(set(item["track"]["artists"][0]["name"] for item in context["spotify"].get("recently_played", []) if item.get("track")))
        if artists: user_profile_info.append(f"User recently listened to: {', '.join(artists[:3])}.")
    
    if user_profile_info: messages.append({"role": "system", "content": "Context about the user: " + " ".join(user_profile_info)})

    for hist_msg in context["conversation_history"][-20:]: messages.append(hist_msg)
    messages.append({"role": "user", "content": message_s})

    try:
        response = await aclient.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, max_tokens=150, temperature=0.75
        )
        reply = response.choices[0].message.content.strip()
        
        context["conversation_history"].append({"role": "user", "content": message_s})
        context["conversation_history"].append({"role": "assistant", "content": reply})
        context["conversation_history"] = context["conversation_history"][-20:]
        
        return reply
    except Exception as e:
        logger.error(f"Error generating chat response for user {user_id}: {e}", exc_info=True)
        return "I'm having a little trouble thinking right now. How about we talk about your favorite song?"


async def is_music_request(message: str) -> Dict:
    if not aclient:
        return {"is_music_request": False, "song_query": None}

    message_s = sanitize_input(message)
    try:
        response = await aclient.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": 
                    "You are an AI that analyzes user messages. Determine if the user is requesting to play, download, find, or get a specific song or music by an artist. "
                    "Respond in JSON format with two keys: 'is_music_request' (boolean) and 'song_query' (string, the song title and artist if identifiable, otherwise null). "
                    "If it's a general music discussion, 'is_music_request' should be false. Focus on explicit requests for a track."
                 },
                {"role": "user", "content": f"Analyze this message: '{message_s}'"}
            ],
            max_tokens=80, temperature=0.1, response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content: return {"is_music_request": False, "song_query": None}
        result = json.loads(content)
        
        is_request = result.get("is_music_request", False)
        if isinstance(is_request, str): is_request = is_request.lower() in ("true", "yes")
        song_query = result.get("song_query")
        if isinstance(song_query, str) and song_query.strip().lower() == "null": song_query = None
            
        return {"is_music_request": bool(is_request), "song_query": song_query.strip() if song_query else None}
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError in is_music_request for message '{message_s}': {e}. Response: {content if 'content' in locals() else 'N/A'}")
        return {"is_music_request": False, "song_query": None}
    except Exception as e:
        logger.error(f"Error in is_music_request for message '{message_s}': {e}", exc_info=True)
        return {"is_music_request": False, "song_query": None}


async def analyze_conversation(user_id: int) -> Dict:
    if not aclient:
        return {"genres": [], "artists": [], "mood": None}

    context = user_contexts.get(user_id, {})
    context.setdefault("mood", None); context.setdefault("preferences", []); context.setdefault("conversation_history", []); context.setdefault("spotify", {})

    if len(context["conversation_history"]) < 2 and not context["spotify"]:
        return {"genres": context["preferences"], "artists": [], "mood": context["mood"]}

    try:
        prompt_parts = []
        if context["conversation_history"]:
            history_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in context["conversation_history"][-10:]])
            prompt_parts.append(f"Recent conversation excerpts with user:\n{history_text}")

        spotify_summary = []
        if context["spotify"].get("recently_played"):
            tracks = [item['track'] for item in context["spotify"]["recently_played"][:5] if item.get('track')]
            track_info = [f"'{t['name']}' by {t['artists'][0]['name']}" for t in tracks if t.get('artists')]
            if track_info: spotify_summary.append(f"User recently played on Spotify: {'; '.join(track_info)}.")
        
        if context["spotify"].get("top_tracks"):
            tracks = context["spotify"]["top_tracks"][:5]
            track_info = [f"'{t['name']}' by {t['artists'][0]['name']}" for t in tracks if t.get('artists')]
            if track_info: spotify_summary.append(f"User's top tracks on Spotify: {'; '.join(track_info)}.")

        if spotify_summary: prompt_parts.append("Spotify listening habits:\n" + "\n".join(spotify_summary))
        if not prompt_parts: return {"genres": context["preferences"], "artists": [], "mood": context["mood"]}
        user_content = "\n\n".join(prompt_parts)

        response = await aclient.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": 
                    "You are a music preference analyzer. Based on the provided conversation and Spotify data, "
                    "infer the user's current mood (e.g., happy, sad, energetic, relaxed, focused, nostalgic), "
                    "preferred music genres (e.g., pop, rock, jazz, classical, electronic, hip-hop), "
                    "and liked artists. "
                    "Respond in JSON format with keys: 'mood' (string|null), 'genres' (list[string], up to 3), 'artists' (list[string], up to 3). "
                    "If a category cannot be confidently inferred, provide an empty list or null."
                },
                {"role": "user", "content": user_content}
            ],
            max_tokens=150, temperature=0.2, response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content: return {"genres": context["preferences"], "artists": [], "mood": context["mood"]}
        result = json.loads(content)

        inferred_mood = result.get("mood") if isinstance(result.get("mood"), str) else None
        inferred_genres = [str(g) for g in result.get("genres", []) if isinstance(g, str)][:3]
        inferred_artists = [str(a) for a in result.get("artists", []) if isinstance(a, str)][:3]

        if inferred_mood and (not context["mood"] or context["mood"] != inferred_mood) : context["mood"] = inferred_mood
        if inferred_genres and set(inferred_genres) != set(context["preferences"]):
            context["preferences"] = list(set(context["preferences"] + inferred_genres))[:3]
        
        return {
            "genres": inferred_genres or context["preferences"],
            "artists": inferred_artists,
            "mood": inferred_mood or context["mood"]
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError in analyze_conversation for user {user_id}: {e}. Response: {content if 'content' in locals() else 'N/A'}")
        return {"genres": context["preferences"], "artists": [], "mood": context["mood"]}
    except Exception as e:
        logger.error(f"Error in analyze_conversation for user {user_id}: {e}", exc_info=True)
        return {"genres": context["preferences"], "artists": [], "mood": context["mood"]}

# ==================== MUSIC DETECTION FUNCTION (REGEX) ====================

def detect_music_in_message(text: str) -> Optional[str]:
    text_lower = text.lower()
    patterns_with_artist = [
        r'(?:play|find|download|get|send me|i want to listen to|can you get|i need|find me|fetch|give me|send|song)\s+(.+?)\s+by\s+(.+)',
    ]
    patterns_song_only = [
        r'(?:play|find|download|get|send me|i want to listen to|can you get|i need|find me|fetch|give me|send|song)\s+(.+)',
    ]

    for pattern in patterns_with_artist:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            song_title = match.group(1).strip().rstrip(',.?!')
            artist = match.group(2).strip().rstrip(',.?!')
            if song_title and artist: return f"{song_title} {artist}"

    for pattern in patterns_song_only:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            song_title = match.group(1).strip().rstrip(',.?!')
            if song_title and song_title.lower() not in ["music", "a song", "something", "some music"]:
                return song_title
    
    music_keywords = ['music', 'song', 'track', 'tune', 'audio', 'listen to']
    if any(keyword in text_lower for keyword in music_keywords):
        if text_lower in ["music", "song", "audio", "play music", "play a song"]:
             return "AI_ANALYSIS_NEEDED"
        potential_title = text_lower
        for kw in music_keywords + ['play', 'find', 'download', 'get', 'send me', 'i want to', 'can you', 'i need', 'fetch', 'give me']:
            potential_title = potential_title.replace(kw, "").strip()
        if len(potential_title) > 3 : return "AI_ANALYSIS_NEEDED"
            
    return None

# ==================== TELEGRAM BOT HANDLERS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    welcome_msg = (
        f"Hi {user.first_name}! 👋 I'm MelodyMind, your Music Healing Companion.\n\n"
        "I can:\n"
        "🎵 Download music from YouTube (send link or ask for a song!)\n"
        "📜 Find lyrics (e.g., `/lyrics Bohemian Rhapsody`)\n"
        "💿 Recommend music based on your mood (`/recommend` or `/mood`)\n"
        "💬 Chat about music and feelings\n"
        "🔗 Link Spotify for personalized recommendations (`/link_spotify`)\n\n"
        "How can I help you today?"
    )
    await update.message.reply_text(welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "🎶 <b>MelodyMind - Your Music Companion</b> 🎶\n\n"
        "<b>Core Commands:</b>\n"
        "  /start - Welcome message\n"
        "  /help - This help message\n"
        "  /download [YouTube URL] - Download specific YouTube audio\n"
        "  /search [song name] - Search YouTube & get download options\n"
        "  /lyrics [song name] or [artist - song] - Get song lyrics\n"
        "  /recommend - Get personalized music recommendations\n"
        "  /mood - Set your current mood for better recommendations\n"
        "  /link_spotify - Connect your Spotify account\n"
        "  /clear - Clear our chat history (for AI context)\n\n"
        "<b>Smart Chat:</b> Try:\n"
        "  🔹 \"Download Shape of You by Ed Sheeran\"\n"
        "  🔹 \"What are the lyrics to Yesterday?\"\n"
        "  🔹 \"I'm feeling happy, suggest some tunes!\"\n"
        "  🔹 Send a YouTube link directly to download.\n"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)


async def download_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    url = ""

    if context.args:
        url = " ".join(context.args)
    elif update.message and update.message.text:
        words = update.message.text.split()
        found_urls = [word for word in words if is_valid_youtube_url(word)]
        if found_urls: url = found_urls[0]
        else:
            await update.message.reply_text("Please provide a valid YouTube URL after /download, or just send a YouTube link.")
            return
    else:
        await update.message.reply_text("Could not find a URL to download.")
        return

    if not is_valid_youtube_url(url):
        await update.message.reply_text("That doesn't look like a valid YouTube URL.")
        return

    if user_id in active_downloads:
        await update.message.reply_text("⚠️ You already have a download in progress. Please wait.")
        return
    active_downloads.add(user_id)

    status_msg = None
    download_info = {} # To store result for finally block
    try:
        status_msg = await update.message.reply_text("⏳ Starting download... this might take a moment.")
        await status_msg.edit_text("🔍 Fetching video information...")
        
        download_info = await download_youtube_audio(url) 

        if not download_info["success"]:
            error_message = download_info.get("error", "An unknown error occurred.")
            await status_msg.edit_text(f"❌ Download failed: {error_message}")
            return

        await status_msg.edit_text(f"✅ Downloaded: {download_info['title']}\n⏳ Preparing to send file...")
        
        audio_path = download_info["audio_path"]
        caption = f"🎵 {download_info['title']}"
        if download_info.get("artist") and download_info["artist"] != "Unknown_Artist": caption += f"\n🎤 Artist: {download_info['artist']}"
        if download_info.get("duration"):
            mins, secs = divmod(download_info['duration'], 60)
            caption += f"\n⏱️ Duration: {int(mins)}:{int(secs):02d}"

        logger.info(f"Sending audio: {download_info['title']} for user {user_id}")
        with open(audio_path, 'rb') as audio_file:
            await update.message.reply_audio(
                audio=audio_file, title=download_info["title"][:64],
                performer=download_info.get("artist", "Unknown Artist")[:64],
                caption=caption, duration=download_info.get('duration', 0)
            )
        await status_msg.delete()
        logger.info(f"Successfully sent audio '{download_info['title']}' and deleted status message.")

    except (TimedOut, NetworkError) as net_err:
        logger.error(f"Network/Timeout error during download for {url}, user {user_id}: {net_err}")
        msg_to_send = "⌛ The operation timed out or a network error occurred. Please try again."
        if status_msg: await status_msg.edit_text(msg_to_send)
        else: await update.message.reply_text(msg_to_send)
    except Exception as e:
        logger.error(f"Unexpected error in download_music for {url}, user {user_id}: {e}", exc_info=True)
        err_msg_user = f"❌ An unexpected error occurred: {str(e)[:100]}."
        if status_msg: await status_msg.edit_text(err_msg_user)
        else: await update.message.reply_text(err_msg_user)
    finally:
        if user_id in active_downloads: active_downloads.remove(user_id)
        if download_info.get("success") and download_info.get("audio_path"):
            if os.path.exists(download_info["audio_path"]):
                try:
                    os.remove(download_info["audio_path"])
                    logger.info(f"Cleaned up downloaded file: {download_info['audio_path']}")
                except Exception as e_clean:
                    logger.error(f"Error cleaning up file {download_info['audio_path']}: {e_clean}")

async def link_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI or SPOTIFY_REDIRECT_URI == "https://your-callback-url.com":
        await update.message.reply_text("Sorry, Spotify linking is not properly configured by the bot admin.")
        return ConversationHandler.END

    user_id = update.effective_user.id
    auth_url = (
        f"https://accounts.spotify.com/authorize?client_id={SPOTIFY_CLIENT_ID}"
        f"&response_type=code&redirect_uri={SPOTIFY_REDIRECT_URI}"
        f"&scope=user-read-recently-played%20user-top-read%20playlist-read-private%20playlist-read-collaborative"
        f"&state={user_id}"
    )
    keyboard = [
        [InlineKeyboardButton("🔗 Link My Spotify Account", url=auth_url)],
        [InlineKeyboardButton("Cancel", callback_data="cancel_spotify_linking")]
    ]
    await update.message.reply_text(
        "Let's link your Spotify to personalize your music experience! 🎵\n\n"
        "1. Click the button below to go to Spotify.\n"
        "2. Log in and authorize MelodyMind.\n"
        "3. Spotify will redirect you. **Copy the `code` parameter from the URL.**\n"
        "   (e.g., `your-redirect-uri/?code=AQC...&state=...`)\n"
        "4. Paste only that `code` back here.\n\nReady? Click below:",
        reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.MARKDOWN
    )
    return SPOTIFY_CODE

async def spotify_code_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    message_text = update.message.text.strip()
    
    if not (message_text and len(message_text) > 30 and re.match(r'^[A-Za-z0-9_-]+$', message_text)): # More robust code check
        reply_text = "That doesn't look like a valid Spotify code. Please paste the code from Spotify."
        if message_text.startswith('/'): reply_text = "It seems you sent a command. I'm expecting the Spotify code."
        await update.message.reply_text(reply_text + "\nOr type /cancel to stop.")
        return SPOTIFY_CODE

    code = message_text
    status_msg = await update.message.reply_text("⏳ Verifying your Spotify code...")

    token_data = await get_user_spotify_token(code)
    if not token_data or not token_data.get("access_token"):
        await status_msg.edit_text(
            "❌ Failed to link Spotify. The code might be invalid, expired, or already used. "
            "Try /link_spotify again to get a new link and code."
        )
        return SPOTIFY_CODE

    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    user_contexts[user_id]["spotify"] = {
        "access_token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "expires_at": token_data.get("expires_at")
    }

    recently_played = await get_user_spotify_data(user_id, "player/recently-played")
    if recently_played is not None:
        user_contexts[user_id]["spotify"]["recently_played"] = recently_played
        logger.info(f"Fetched recently played for user {user_id} after linking.")
    else:
        logger.warning(f"Could not fetch recently played for user {user_id} after linking, though token obtained.")

    await status_msg.edit_text(
        "✅ Spotify account linked successfully! 🎉\n"
        "I can now use your listening history for better recommendations. Try /recommend!"
    )
    return ConversationHandler.END


async def spotify_code_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Provide Spotify code: `/spotify_code YOUR_CODE`\n"
            "Better to paste code after /link_spotify.", parse_mode=ParseMode.MARKDOWN
        )
        return
    
    user_id = update.effective_user.id
    code = context.args[0].strip()

    if not (code and len(code) > 30 and re.match(r'^[A-Za-z0-9_-]+$', code)):
        await update.message.reply_text("That doesn't look like a valid Spotify code.")
        return

    status_msg = await update.message.reply_text("⏳ Verifying Spotify code (via command)...")
    token_data = await get_user_spotify_token(code)
    if not token_data or not token_data.get("access_token"):
        await status_msg.edit_text("❌ Failed to link Spotify with this code. Try /link_spotify process.")
        return

    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    user_contexts[user_id]["spotify"] = {
        "access_token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "expires_at": token_data.get("expires_at")
    }
    await status_msg.edit_text("✅ Spotify account linked via command! Try /recommend.")


async def cancel_spotify_linking(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Spotify linking cancelled. Try again anytime with /link_spotify.")
    return ConversationHandler.END

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("What song are you looking for? Example:\n/search Shape of You Ed Sheeran")
        return

    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Searching YouTube for: '{query}'...")
    results = await search_youtube(query, max_results=5)
    await status_msg.delete()
    await send_search_results_keyboard(update.message, query, results) # Pass update.message for reply context


async def send_search_results_keyboard(message_or_update, query: str, results: List[Dict]) -> None:
    """Sends YouTube search results with inline keyboard. Accepts message or update object."""
    reply_target = message_or_update.message if isinstance(message_or_update, Update) and message_or_update.message else message_or_update

    if not results:
        await reply_target.reply_text(f"Sorry, couldn't find songs on YouTube for '{query}'. Try different keywords?")
        return

    keyboard = []
    response_text = f"🔎 Here's what I found for '{query}' on YouTube:\n\n"
    
    for i, result in enumerate(results[:5]):
        if not result.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', result['id']):
            logger.warning(f"Skipping invalid YouTube search result ID: {result.get('id', 'N/A')} for query '{query}'")
            continue

        duration_str = ""
        if result.get('duration'):
            mins, secs = divmod(int(result['duration']), 60)
            duration_str = f" [{mins}:{secs:02d}]"
        
        title = result['title']
        response_text += f"{i+1}. {title}{duration_str} (Uploader: {result.get('uploader', 'N/A')})\n"
        button_title = title[:40] + "..." if len(title) > 40 else title
        keyboard.append([InlineKeyboardButton(f"📥 {button_title}", callback_data=f"download_{result['id']}")])

    if not keyboard:
        await reply_target.reply_text(f"Sorry, found matches for '{query}', but couldn't create download options.")
        return

    keyboard.append([InlineKeyboardButton("❌ Cancel Search", callback_data="cancel_search")])
    response_text += "\nClick a button to download the audio:"
    await reply_target.reply_text(response_text, reply_markup=InlineKeyboardMarkup(keyboard))


async def auto_download_first_result(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str) -> None:
    user_id = update.effective_user.id
    if user_id in active_downloads:
        await update.message.reply_text("⚠️ You have another download in progress. Please wait.")
        return
    active_downloads.add(user_id)
    
    status_msg = await update.message.reply_text(f"🔍 Searching for '{query}' to auto-download...")
    download_info = {}
    try:
        results = await search_youtube(query, max_results=1)
        if not results or not results[0].get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', results[0]['id']):
            await status_msg.edit_text(f"❌ Couldn't find a valid result for '{query}' to auto-download.")
            active_downloads.remove(user_id)
            return

        result_info = results[0]
        video_url = result_info["url"]
        await status_msg.edit_text(f"✅ Found: {result_info['title']}\n⏳ Downloading audio...")

        download_info = await download_youtube_audio(video_url)
        if not download_info["success"]:
            await status_msg.edit_text(f"❌ Auto-download failed: {download_info['error']}")
            return

        await status_msg.edit_text(f"✅ Downloaded: {download_info['title']}\n⏳ Sending file...")
        with open(download_info["audio_path"], 'rb') as audio:
            await update.message.reply_audio(
                audio=audio, title=download_info["title"][:64],
                performer=download_info.get("artist", "Unknown Artist")[:64],
                caption=f"🎵 {download_info['title']} (Auto-downloaded)"
            )
        await status_msg.delete()
    except Exception as e:
        logger.error(f"Error in auto_download_first_result for '{query}': {e}", exc_info=True)
        err_msg = f"❌ Error during auto-download: {str(e)[:100]}"
        if status_msg: await status_msg.edit_text(err_msg)
        else: await update.message.reply_text(err_msg)
    finally:
        if user_id in active_downloads: active_downloads.remove(user_id)
        if download_info.get("success") and download_info.get("audio_path"):
            if os.path.exists(download_info["audio_path"]):
                try: os.remove(download_info["audio_path"])
                except Exception as e_clean: logger.error(f"Error cleaning file in auto_download: {e_clean}")

async def auto_download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Specify song for auto-download. Example:\n/autodownload Shape of You Ed Sheeran")
        return
    query = " ".join(context.args)
    await auto_download_first_result(update, context, query)


async def get_lyrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Specify song title for lyrics. Examples:\n"
            "`/lyrics Bohemian Rhapsody`\n"
            "`/lyrics Queen - Bohemian Rhapsody` (artist first is better!)",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    query_full = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Searching lyrics for: \"{query_full}\"...")

    try:
        artist = None
        song_title = query_full
        match_artist_song = re.match(r'^(.*?)\s*-\s*(.+)$', query_full)
        match_song_by_artist = re.match(r'^(.*?)\s+by\s+(.+)$', query_full, re.IGNORECASE)

        if match_artist_song:
            potential_artist1, potential_song1 = match_artist_song.groups()
            if len(potential_artist1) < len(potential_song1) or any(kw in potential_artist1.lower() for kw in ["ft", "feat", "with"]):
                 artist = potential_artist1.strip(); song_title = potential_song1.strip()
            else: song_title = query_full
        elif match_song_by_artist:
            song_title = match_song_by_artist.group(1).strip(); artist = match_song_by_artist.group(2).strip()
        else: song_title = query_full 

        lyrics_text = await get_lyrics(song_title, artist)

        if len(lyrics_text) > 4090:
            await status_msg.edit_text(lyrics_text[:4090] + "\n\n[Message truncated. Full lyrics might be longer.]")
        else:
            await status_msg.edit_text(lyrics_text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.error(f"Error in get_lyrics_command for query '{query_full}': {e}", exc_info=True)
        await status_msg.edit_text("❌ Sorry, an unexpected error occurred while fetching lyrics.")


async def provide_generic_recommendations(message_object, mood: str) -> None:
    mood = mood.lower() if mood else "general"
    generic_searches = {
        "happy": ["upbeat pop songs playlist", "feel good indie music"],
        "sad": ["emotional acoustic songs", "sad songs for crying playlist"],
        "energetic": ["high energy workout music", "pump up rock anthems"],
        "relaxed": ["calming ambient music", "lofi hip hop radio"],
        "focused": ["instrumental study music", "alpha waves focus music"],
        "nostalgic": ["80s classic hits", "90s alternative rock"]
    }
    searches_to_suggest = generic_searches.get(mood, generic_searches["happy"]) 
    
    response_text = f"🎵 Since I couldn't get specific recommendations, here are YouTube search ideas for **{mood}** vibes:\n\n"
    for i, search_query in enumerate(searches_to_suggest, 1):
        yt_search_url = f"https://www.youtube.com/results?search_query={requests.utils.quote(search_query)}"
        response_text += f"{i}. <a href=\"{yt_search_url}\">{search_query.title()}</a>\n"
    response_text += "\n💡 Try `/search [song/artist]` or send a YouTube link to download!"
    await message_object.reply_text(response_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)


async def set_mood(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    keyboard = [
        [InlineKeyboardButton("😊 Happy", callback_data="mood_happy"), InlineKeyboardButton("😢 Sad", callback_data="mood_sad"), InlineKeyboardButton("💪 Energetic", callback_data="mood_energetic")],
        [InlineKeyboardButton("😌 Relaxed", callback_data="mood_relaxed"), InlineKeyboardButton("🧠 Focused", callback_data="mood_focused"), InlineKeyboardButton("🕰️ Nostalgic", callback_data="mood_nostalgic")],
        [InlineKeyboardButton("🚫 Skip/Other", callback_data="mood_skip")],
    ]
    await update.message.reply_text("How are you feeling today? This helps me tailor music suggestions.", reply_markup=InlineKeyboardMarkup(keyboard))
    return MOOD

async def recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE, from_mood_setter: bool = False) -> Optional[int]:
    user_id = update.effective_user.id
    message_object = update.callback_query.message if from_mood_setter and update.callback_query else update.message
    
    status_msg = await message_object.reply_text("🎧 Finding personalized music recommendations...", quote=False)

    try:
        user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})

        if user_contexts[user_id].get("spotify", {}).get("access_token"):
            logger.info(f"Updating Spotify data for user {user_id} before recommendation.")
            rp = await get_user_spotify_data(user_id, "player/recently-played")
            if rp is not None: user_contexts[user_id]["spotify"]["recently_played"] = rp
            tt = await get_user_spotify_data(user_id, "top/tracks?time_range=short_term")
            if tt is not None: user_contexts[user_id]["spotify"]["top_tracks"] = tt
            pl = await get_user_spotify_playlists(user_id)
            if pl is not None: user_contexts[user_id]["spotify"]["playlists"] = pl

        logger.info(f"Analyzing conversation for user {user_id} for recommendation.")
        ai_analysis = await analyze_conversation(user_id)
        
        current_mood = ai_analysis.get("mood") or user_contexts[user_id].get("mood")
        current_genres = list(set(ai_analysis.get("genres", []) + user_contexts[user_id].get("preferences", [])))[:3]
        current_artists = ai_analysis.get("artists", [])

        if not current_mood and not from_mood_setter:
            await status_msg.delete()
            await message_object.reply_text("To give the best recommendations, I need your mood. Let's set it first!")
            # This assumes `update` object for `set_mood` is the original one that triggered `/recommend`.
            # If `update` is from a callback, it might need careful handling if `set_mood` expects a message.
            # For simplicity, we ensure `set_mood` is called with appropriate `update` object.
            mood_update = update if update.message else update.callback_query # Try to get original message context
            return await set_mood(mood_update, context)


        seed_track_ids: List[str] = []
        spotify_user_data = user_contexts[user_id].get("spotify", {})
        
        if spotify_user_data.get("recently_played"):
            seed_track_ids.extend([t["track"]["id"] for t in spotify_user_data["recently_played"][:2] if t.get("track") and t["track"].get("id")])
        if len(seed_track_ids) < 2 and spotify_user_data.get("top_tracks"):
            seed_track_ids.extend([t["id"] for t in spotify_user_data["top_tracks"][:(2-len(seed_track_ids))] if t.get("id")])
        
        if not seed_track_ids and current_artists:
            client_token = await get_spotify_token()
            if client_token:
                for artist_name in current_artists[:2]:
                    artist_search_res = await search_spotify_track(client_token, f"artist:{artist_name}")
                    if artist_search_res and artist_search_res.get("id"): seed_track_ids.append(artist_search_res["id"])
                    if len(seed_track_ids) >=2: break

        if seed_track_ids:
            logger.info(f"Getting Spotify recommendations for user {user_id} with seeds: {seed_track_ids}")
            spotify_client_token = await get_spotify_token()
            if spotify_client_token:
                spotify_recs = await get_spotify_recommendations(spotify_client_token, seed_tracks=seed_track_ids, limit=5)
                if spotify_recs:
                    response = f"🎵 Based on your vibe (mood: {current_mood or 'general'}), here are Spotify recommendations:\n\n"
                    for i, track in enumerate(spotify_recs, 1):
                        artists_text = ", ".join(a["name"] for a in track.get("artists", []))
                        album = track.get("album", {}).get("name", "")
                        track_url = track.get("external_urls", {}).get("spotify", "#")
                        response += f"{i}. <a href=\"{track_url}\"><b>{track['name']}</b></a> by {artists_text}"
                        if album: response += f" (from <i>{album}</i>)"
                        response += "\n"
                    response += "\n💡 Ask me to download these by name or search for them!"
                    await status_msg.edit_text(response, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                    return ConversationHandler.END if from_mood_setter else None

        logger.info(f"Falling back to YouTube search for user {user_id} recommendation.")
        yt_search_query_parts = []
        if current_mood: yt_search_query_parts.append(current_mood)
        if current_genres: yt_search_query_parts.append(current_genres[0])
        yt_search_query_parts.append("music")
        if current_artists: yt_search_query_parts.append(f"like {current_artists[0]}")
        
        yt_search_query = " ".join(yt_search_query_parts) if yt_search_query_parts and " ".join(yt_search_query_parts).strip() != "music" else f"{current_mood or 'popular'} music playlist"

        yt_results = await search_youtube(yt_search_query, max_results=5)
        await status_msg.delete()
        if yt_results:
            await send_search_results_keyboard(message_object, yt_search_query, yt_results)
        else:
            await provide_generic_recommendations(message_object, current_mood or "happy")
        
        return ConversationHandler.END if from_mood_setter else None

    except Exception as e:
        logger.error(f"Error in recommend_music for user {user_id}: {e}", exc_info=True)
        err_msg = "I couldn't get recommendations. Please try again later."
        if status_msg: await status_msg.edit_text(err_msg)
        else: await message_object.reply_text(err_msg)
        return ConversationHandler.END if from_mood_setter else None


async def enhanced_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = query.from_user.id

    logger.debug(f"Button callback: '{data}' from user {user_id}")
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})

    if data.startswith("mood_"):
        mood = data.split("_", 1)[1]
        if mood == "skip":
            await query.edit_message_text("Okay, no mood set. You can use /mood later or tell me how you feel!")
            await recommend_music(update, context, from_mood_setter=True) # Try recommending without mood
            return ConversationHandler.END
            
        user_contexts[user_id]["mood"] = mood
        logger.info(f"User {user_id} set mood to: {mood}")

        keyboard = [
            [InlineKeyboardButton("Pop", callback_data="pref_pop"), InlineKeyboardButton("Rock", callback_data="pref_rock"), InlineKeyboardButton("Hip-Hop", callback_data="pref_hiphop")],
            [InlineKeyboardButton("Electronic", callback_data="pref_electronic"), InlineKeyboardButton("Classical", callback_data="pref_classical"), InlineKeyboardButton("Indie/Alt", callback_data="pref_indie")],
            [InlineKeyboardButton("No Preference / Skip Genre", callback_data="pref_skip")],
        ]
        await query.edit_message_text(f"Got it! You're feeling {mood}. Any specific music genre you're in the mood for?", reply_markup=InlineKeyboardMarkup(keyboard))
        return PREFERENCE

    elif data.startswith("pref_"):
        preference = data.split("_", 1)[1]
        if preference != "skip":
            if "preferences" not in user_contexts[user_id]: user_contexts[user_id]["preferences"] = []
            if preference not in user_contexts[user_id]["preferences"]: user_contexts[user_id]["preferences"].append(preference)
            user_contexts[user_id]["preferences"] = user_contexts[user_id]["preferences"][:3]
            logger.info(f"User {user_id} added preference: {preference}")
            await query.edit_message_text(f"Great, noted your preference for {preference} music!")
        else:
            await query.edit_message_text("Okay, no specific genre preference noted.")
        
        await recommend_music(update, context, from_mood_setter=True)
        return ConversationHandler.END

    elif data.startswith("download_") or data.startswith("auto_download_"):
        video_id_prefix = "auto_download_" if data.startswith("auto_download_") else "download_"
        video_id = data.split(video_id_prefix, 1)[1]

        if not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
            logger.error(f"Invalid YouTube video ID in callback: {video_id}")
            await query.edit_message_text("❌ Invalid video ID. Try searching again.")
            return None

        url = f"https://www.youtube.com/watch?v={video_id}"
        
        if user_id in active_downloads:
            await query.edit_message_text("⚠️ You already have a download in progress.")
            return None
        active_downloads.add(user_id)
        
        await query.edit_message_text(f"⏳ Starting download for video ID {video_id}...")
        download_info_cb = {} # For finally block
        try:
            download_info_cb = await download_youtube_audio(url)
            if not download_info_cb["success"]:
                await query.edit_message_text(f"❌ Download failed: {download_info_cb.get('error', 'Unknown reason')}")
                return None

            await query.edit_message_text(f"✅ Downloaded: {download_info_cb['title']}\n⏳ Sending file...")
            with open(download_info_cb["audio_path"], 'rb') as audio_file:
                await context.bot.send_audio(
                    chat_id=query.message.chat_id, audio=audio_file,
                    title=download_info_cb["title"][:64], performer=download_info_cb.get("artist", "Unknown Artist")[:64],
                    caption=f"🎵 {download_info_cb['title']}"
                )
            await query.edit_message_text(f"✅ Download complete & sent: {download_info_cb['title']}")
        
        except (TimedOut, NetworkError) as te:
            logger.error(f"Network/Timeout error sending audio for {video_id}: {te}", exc_info=True)
            err_msg = "❌ Failed to send audio due to network issue. Downloaded but couldn't send."
            try: await query.edit_message_text(err_msg)
            except Exception: await context.bot.send_message(query.message.chat_id, err_msg)
        except Exception as e:
            logger.error(f"Error in button download handler for {video_id}: {e}", exc_info=True)
            err_msg = f"❌ Error during download: {str(e)[:100]}"
            try: await query.edit_message_text(err_msg)
            except Exception: await context.bot.send_message(query.message.chat_id, err_msg)
        finally:
            if user_id in active_downloads: active_downloads.remove(user_id)
            if download_info_cb.get("success") and download_info_cb.get("audio_path"):
                 if os.path.exists(download_info_cb["audio_path"]):
                    try: os.remove(download_info_cb["audio_path"])
                    except Exception as e_clean: logger.error(f"Error cleaning file in button_handler: {e_clean}")
        return None

    elif data.startswith("show_options_"):
        search_query = data.split("show_options_", 1)[1]
        await query.edit_message_text(f"🔍 Fetching more options for '{search_query}'...")
        results = await search_youtube(search_query, max_results=5)
        if not results:
            await query.edit_message_text(f"Sorry, couldn't find other options for '{search_query}'.")
            return None
        await query.message.delete() # Delete old keyboard message
        await send_search_results_keyboard(query.message, search_query, results) # Pass query.message for context
        return None

    elif data == "cancel_search":
        await query.edit_message_text("❌ Search or download action cancelled.")
        return None
    
    elif data == "cancel_spotify_linking":
        await query.edit_message_text("Spotify linking process cancelled.")
        return ConversationHandler.END
    return None


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5), reraise=True)
async def reply_with_retry(message_obj, text, **kwargs):
    return await message_obj.reply_text(text, **kwargs)

async def enhanced_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not update.message or not update.message.text: return

    user_id = update.effective_user.id
    text = sanitize_input(update.message.text) # Uses the defined sanitize_input
    logger.debug(f"Handling message from user {user_id}: \"{text[:50]}...\"")
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    
    typing_task = asyncio.create_task(context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing'))

    try:
        if is_valid_youtube_url(text):
            await typing_task
            context.args = [text]
            await download_music(update, context)
            return

        detected_song_query_regex = detect_music_in_message(text)
        if detected_song_query_regex and detected_song_query_regex != "AI_ANALYSIS_NEEDED":
            await typing_task
            status_msg_regex = await reply_with_retry(update.message, f"🔍 I think you want '{detected_song_query_regex}'. Searching YouTube...")
            results = await search_youtube(detected_song_query_regex, max_results=3)
            await status_msg_regex.delete()
            if results: await send_search_results_keyboard(update.message, detected_song_query_regex, results)
            else: await reply_with_retry(update.message, f"Sorry, couldn't find '{detected_song_query_regex}' on YouTube.")
            return
        
        ai_intent = {"is_music_request": False, "song_query": None, "is_lyrics_request": False}
        if len(text.split()) > 2 or detected_song_query_regex == "AI_ANALYSIS_NEEDED":
            lyrics_keywords = ["lyrics", "words to the song", "song text for"]
            if any(keyword in text.lower() for keyword in lyrics_keywords):
                potential_song_for_lyrics = text.lower()
                for kw in lyrics_keywords: potential_song_for_lyrics = potential_song_for_lyrics.replace(kw, "")
                potential_song_for_lyrics = potential_song_for_lyrics.strip(".?! ")
                if potential_song_for_lyrics:
                    ai_intent["is_lyrics_request"] = True; ai_intent["song_query"] = potential_song_for_lyrics

            if not ai_intent["is_lyrics_request"]:
                music_request_analysis = await is_music_request(text)
                if music_request_analysis["is_music_request"] and music_request_analysis["song_query"]:
                    ai_intent["is_music_request"] = True; ai_intent["song_query"] = music_request_analysis["song_query"]
        
        await typing_task

        if ai_intent["is_music_request"] and ai_intent["song_query"]:
            status_msg_ai = await reply_with_retry(update.message, f"🔍 Got it! Searching YouTube for '{ai_intent['song_query']}'...")
            results = await search_youtube(ai_intent['song_query'], max_results=3)
            await status_msg_ai.delete()
            if results: await send_search_results_keyboard(update.message, ai_intent['song_query'], results)
            else: await reply_with_retry(update.message, f"Sorry, couldn't find '{ai_intent['song_query']}' on YouTube.")
            return

        if ai_intent["is_lyrics_request"] and ai_intent["song_query"]:
            context.args = [ai_intent["song_query"]]
            await get_lyrics_command(update, context)
            return

        lower_text = text.lower()
        if "i'm feeling" in lower_text or "i feel" in lower_text:
            try:
                split_phrase = "i'm feeling" if "i'm feeling" in lower_text else "i feel"
                mood_token = lower_text.split(split_phrase, 1)[1].strip().split()[0].rstrip('.,?!')
                valid_moods = ["happy", "sad", "energetic", "relaxed", "focused", "nostalgic", "anxious", "stressed", "calm", "excited"]
                if mood_token in valid_moods:
                    user_contexts[user_id]["mood"] = mood_token
                    logger.info(f"Mood '{mood_token}' detected in text for user {user_id}")
            except IndexError: pass

        chat_response_text = await generate_chat_response(user_id, text)
        await reply_with_retry(update.message, chat_response_text)

    except RetryError as re_err:
        logger.warning(f"Failed to send/edit message for user {user_id} after retries: {re_err}")
    except Exception as e:
        logger.error(f"Error in enhanced_handle_message for user {user_id}, text \"{text[:50]}...\": {e}", exc_info=True)
        try: await reply_with_retry(update.message, "Pardon me, I hit a snag. Try phrasing differently or use /help?")
        except Exception: logger.error(f"Failed to send error fallback to user {user_id}")
    finally:
        if not typing_task.done(): typing_task.cancel()


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id in user_contexts and "conversation_history" in user_contexts[user_id]:
        user_contexts[user_id]["conversation_history"] = []
        await update.message.reply_text("✅ Our chat history for AI context has been cleared.")
    else:
        await update.message.reply_text("Hmm, I don't seem to have a chat history with you to clear.")

async def cancel_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    message_text = "Okay, action cancelled."
    if update.callback_query:
        await update.callback_query.answer()
        try: await update.callback_query.edit_message_text(message_text)
        except Exception: await context.bot.send_message(update.effective_chat.id, message_text)
    elif update.message: await update.message.reply_text(message_text)
    
    logger.info(f"Conversation cancelled for user {update.effective_user.id}")
    return ConversationHandler.END

# ==================== ERROR HANDLING ====================
async def handle_telegram_error(update: Optional[object], context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error(f"Update: {update} caused error: {context.error}", exc_info=context.error)
    error_message = "Oops! Something went wrong. I've logged it. Please try again later. 🛠️"
    if isinstance(context.error, (TimedOut, NetworkError)):
        error_message = "I'm having trouble connecting. Please try again in a moment. 🌐"

    effective_chat_id = None
    reply_to_message_id = None

    if update and hasattr(update, 'effective_chat') and update.effective_chat:
        effective_chat_id = update.effective_chat.id
    if update and hasattr(update, 'effective_message') and update.effective_message:
        reply_to_message_id = update.effective_message.message_id
        if not effective_chat_id: # Should be redundant if effective_message exists
             effective_chat_id = update.effective_message.chat_id
    
    if effective_chat_id:
        try:
            # Prefer replying to the specific message if possible
            if reply_to_message_id and hasattr(update, 'effective_message'):
                 await update.effective_message.reply_text(error_message)
            else: # Send a new message to the chat
                await context.bot.send_message(chat_id=effective_chat_id, text=error_message)
        except Exception as e:
            logger.error(f"Failed to send error notification to user {effective_chat_id}: {e}")


# ==================== CLEANUP FUNCTIONS ====================
def cleanup_downloads_sync() -> None:
    logger.info(f"Attempting to clean up download directory: {DOWNLOAD_DIR}")
    cleaned_count, error_count = 0, 0
    if os.path.exists(DOWNLOAD_DIR):
        for item_name in os.listdir(DOWNLOAD_DIR):
            item_path = os.path.join(DOWNLOAD_DIR, item_name)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path); cleaned_count +=1
                elif os.path.isdir(item_path):
                    logger.warning(f"Found unexpected subdirectory in downloads: {item_path}")
            except Exception as e:
                logger.error(f"Error removing {item_path} during cleanup: {e}"); error_count +=1
        if cleaned_count > 0 or error_count == 0 : logger.info(f"Cleaned {cleaned_count} file(s) from {DOWNLOAD_DIR}. {error_count} errors.")
        elif error_count > 0: logger.error(f"Cleanup of {DOWNLOAD_DIR} encountered {error_count} errors.")
    else: logger.info(f"Download directory {DOWNLOAD_DIR} does not exist, no cleanup needed.")

async def cleanup_downloads():
    await asyncio.to_thread(cleanup_downloads_sync)

# ==================== SIGNAL HANDLERS ====================
def sig_handler(sig, frame):
    logger.info(f"Received signal {sig}, initiating shutdown and cleanup...")
    cleanup_downloads_sync()
    logger.info("Cleanup complete. Exiting.")
    sys.exit(0)

# ==================== MAIN FUNCTION ====================

def main() -> None:
    required_env_vars = ["TELEGRAM_TOKEN", "OPENAI_API_KEY", "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET", "SPOTIFY_REDIRECT_URI", "GENIUS_ACCESS_TOKEN"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.critical(f"FATAL: Missing required environment variables: {', '.join(missing_vars)}. Bot cannot start.")
        sys.exit(1)
    
    if os.getenv("SPOTIFY_REDIRECT_URI") == "https://your-callback-url.com":
        logger.warning("SPOTIFY_REDIRECT_URI is default. Spotify OAuth /link_spotify will NOT work.")

    application = (Application.builder().token(TOKEN)
        .read_timeout(20).write_timeout(75).connect_timeout(15).pool_timeout(60)
        .get_updates_read_timeout(40).build()
    )

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("download", download_music))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("autodownload", auto_download_command))
    application.add_handler(CommandHandler("lyrics", get_lyrics_command))
    application.add_handler(CommandHandler("recommend", recommend_music))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(CommandHandler("spotify_code", spotify_code_command))

    spotify_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("link_spotify", link_spotify)],
        states={SPOTIFY_CODE: [MessageHandler(filters.TEXT & ~filters.COMMAND, spotify_code_handler), CallbackQueryHandler(cancel_spotify_linking, pattern="^cancel_spotify_linking$")]},
        fallbacks=[CommandHandler("cancel", cancel_conversation), CallbackQueryHandler(cancel_spotify_linking, pattern="^cancel_spotify_linking$")],
        conversation_timeout=timedelta(minutes=5).total_seconds()
    )
    application.add_handler(spotify_conv_handler)

    mood_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("mood", set_mood)],
        states={
            MOOD: [CallbackQueryHandler(enhanced_button_handler, pattern="^mood_")],
            PREFERENCE: [CallbackQueryHandler(enhanced_button_handler, pattern="^pref_")]
        },
        fallbacks=[CommandHandler("cancel", cancel_conversation)],
        conversation_timeout=timedelta(minutes=3).total_seconds()
    )
    application.add_handler(mood_conv_handler)

    application.add_handler(CallbackQueryHandler(enhanced_button_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, enhanced_handle_message))
    application.add_error_handler(handle_telegram_error)

    signal.signal(signal.SIGINT, sig_handler); signal.signal(signal.SIGTERM, sig_handler)
    atexit.register(cleanup_downloads_sync)
    logger.info("Performing initial cleanup of download directory...")
    cleanup_downloads_sync()

    logger.info("Starting MelodyMind Bot...")
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.critical(f"Critical error running the bot application: {e}", exc_info=True)
    finally:
        logger.info("Bot application has stopped. Performing final cleanup...")
        cleanup_downloads_sync()

if __name__ == "__main__":
    main()