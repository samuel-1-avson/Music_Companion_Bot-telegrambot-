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
import httpx
import asyncio

# Telegram imports
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
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
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "https://your-callback-url.com") # User must set this in .env

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if GENIUS_ACCESS_TOKEN and lyricsgenius:
    genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=15, retries=2)
else:
    genius = None

# Conversation states
MOOD, PREFERENCE, ACTION, SPOTIFY_CODE = range(4)

# Track active downloads and user contexts
active_downloads = set() # Stores user_ids of users with an active download
download_lock = asyncio.Lock() # Global lock for serializing download operations

user_contexts: Dict[int, Dict] = {}
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ==================== UTILITY FUNCTIONS ====================
def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent injection and clean text."""
    if not text:
        return ""
    # Remove potentially dangerous characters and trim
    return re.sub(r'[<>;&]', '', text.strip())[:250] # Increased length slightly for queries

# ==================== SPOTIFY HELPER FUNCTIONS ====================

@lru_cache(maxsize=1) # Cache for a short period or 1 result
def get_spotify_token() -> Optional[str]:
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
        logger.error(f"Error getting Spotify client credentials token: {e}")
        return None

def search_spotify_track(token: str, query: str) -> Optional[Dict]:
    """Search for a track on Spotify."""
    if not token:
        return None

    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"q": sanitize_input(query), "type": "track", "limit": 1}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        items = response.json().get("tracks", {}).get("items", [])
        return items[0] if items else None
    except (requests.exceptions.RequestException, IndexError) as e:
        logger.error(f"Error searching Spotify track '{query}': {e}")
        return None

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_spotify_recommendations(token: str, seed_tracks: List[str], limit: int = 5) -> List[Dict]:
    """Get track recommendations from Spotify."""
    if not token or not seed_tracks:
        logger.warning("No token or seed tracks provided for Spotify recommendations")
        return []

    url = "https://api.spotify.com/v1/recommendations"
    headers = {"Authorization": f"Bearer {token}"}
    # Spotify API seed limit is 5. Take distinct tracks.
    params = {"seed_tracks": ",".join(list(set(seed_tracks))[:5]), "limit": limit}


    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("tracks", [])
    except requests.exceptions.HTTPError as http_error:
        logger.warning(f"Spotify recommendations HTTP error for seed tracks: {seed_tracks}, "
                       f"status: {http_error.response.status_code}, "
                       f"response: {http_error.response.text if http_error.response else 'No response'}")
        if http_error.response and http_error.response.status_code == 400: # Bad request, likely bad seed
             logger.warning("Bad seed track for Spotify recommendation.")
        return []
    except requests.exceptions.RequestException as req_error:
        logger.error(f"Error getting Spotify recommendations: {req_error}")
        return []

def get_user_spotify_token(user_id: int, code: str) -> Optional[Dict]:
    """Exchange authorization code for Spotify access and refresh tokens."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI:
        logger.warning("Spotify OAuth credentials not configured")
        return None
    if SPOTIFY_REDIRECT_URI == "https://your-callback-url.com":
        logger.error("SPOTIFY_REDIRECT_URI is not configured. Please set it in your .env file.")
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
        logger.error(f"Error getting user Spotify token: {e}. Response: {e.response.text if hasattr(e, 'response') and e.response else 'No response text'}")
        return None

def refresh_spotify_token(user_id: int) -> Optional[str]:
    """Refresh Spotify access token using refresh token."""
    if user_id not in user_contexts or "spotify" not in user_contexts[user_id]:
        logger.warning(f"No user context or Spotify data for user {user_id} to refresh token.")
        return None
        
    context = user_contexts.get(user_id, {}) # Should always exist if checked above
    refresh_token = context.get("spotify", {}).get("refresh_token")
    
    if not refresh_token:
        logger.warning(f"No refresh token found for user {user_id}. Needs re-authorization.")
        # Clear Spotify data to force re-auth if needed by other functions
        if "spotify" in user_contexts[user_id]:
            user_contexts[user_id]["spotify"] = {}
        return None

    url = "https://accounts.spotify.com/api/token"
    auth_header = base64.b64encode(f'{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}'.encode()).decode()
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
        
        # Update context carefully
        current_spotify_data = user_contexts[user_id].get("spotify", {})
        current_spotify_data.update({
            "access_token": token_data.get("access_token"),
            # Important: Spotify might return a new refresh token
            "refresh_token": token_data.get("refresh_token", refresh_token), 
            "expires_at": expires_at
        })
        user_contexts[user_id]["spotify"] = current_spotify_data
        
        logger.info(f"Successfully refreshed Spotify token for user {user_id}")
        return token_data.get("access_token")
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 400:
            logger.error(f"Invalid refresh token for user {user_id} (Spotify API 400 Error: {e.response.text}). Clearing Spotify context.")
            if "spotify" in user_contexts[user_id]:
                 user_contexts[user_id]["spotify"] = {} # Clear invalid token data
        else:
            logger.error(f"HTTP error refreshing Spotify token for user {user_id}: {e}. Response: {e.response.text if e.response else ''}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error refreshing Spotify token for user {user_id}: {e}")
        return None

def get_user_spotify_data(user_id: int, endpoint: str, params: Optional[Dict] = None) -> Optional[List[Dict]]:
    """Fetch user-specific Spotify data (e.g., 'player/recently-played', 'top/tracks')."""
    if user_id not in user_contexts or "spotify" not in user_contexts[user_id]:
        logger.info(f"User {user_id} has not linked their Spotify account or no context.")
        return None

    spotify_data = user_contexts[user_id].get("spotify", {})
    access_token = spotify_data.get("access_token")
    expires_at = spotify_data.get("expires_at")

    if not access_token or (expires_at and datetime.now(pytz.UTC).timestamp() > expires_at):
        logger.info(f"Spotify token expired or missing for user {user_id}. Refreshing...")
        access_token = refresh_spotify_token(user_id)
        if not access_token:
            logger.warning(f"Failed to refresh Spotify token for user {user_id}. Cannot fetch {endpoint}.")
            return None # Important: if refresh fails, bail out

    url = f"https://api.spotify.com/v1/me/{endpoint}"
    headers = {"Authorization": f"Bearer {access_token}"}
    
    default_params = {"limit": 10}
    if params:
        default_params.update(params)

    try:
        response = requests.get(url, headers=headers, params=default_params, timeout=10)
        response.raise_for_status()
        return response.json().get("items", [])
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 401: # Unauthorized
            logger.warning(f"Spotify token for user {user_id} became invalid (401). Attempting refresh again or user needs to re-link.")
            access_token = refresh_spotify_token(user_id) # try one more refresh
            if access_token: # retry call if token refreshed
                return get_user_spotify_data(user_id, endpoint, params) # one level recursion
            else: # if still no token, clear context
                user_contexts[user_id]["spotify"] = {}
        logger.error(f"HTTP error fetching Spotify user data ({endpoint}) for {user_id}: {e}. Response: {e.response.text if e.response else ''}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching Spotify user data ({endpoint}) for {user_id}: {e}")
        return None # Changed from `return` to `return None`

def get_user_spotify_playlists(user_id: int) -> Optional[List[Dict]]:
    """Fetch user's Spotify playlists."""
    return get_user_spotify_data(user_id, "playlists", params={"limit": 5}) # Limit to 5 for brevity

# ==================== YOUTUBE HELPER FUNCTIONS ====================

def is_valid_youtube_url(url: str) -> bool:
    """Check if the URL is a valid YouTube URL."""
    if not url:
        return False
    # Comprehensive regex for YouTube URLs (watch, shorts, shortlinks)
    patterns = [
        r'(https?://)?(www\.)?(youtube|music\.youtube)\.com/(watch\?v=|shorts/|embed/|v/|user/.+/watch\?v=)([0-9A-Za-z_-]{11})',
        r'(https?://)?(www\.)?youtu\.be/([0-9A-Za-z_-]{11})'
    ]
    return any(re.search(pattern, url) for pattern in patterns)

def sanitize_filename(filename: str) -> str:
    """Remove invalid characters from filenames and limit length."""
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return sanitized[:150] # Max filename length considerations

def download_youtube_audio(url: str) -> Dict[str, Any]:
    """Download audio from a YouTube video."""
    # Validate URL and extract ID
    video_id_match = re.search(r'(?:v=|/|embed/|shorts/)([0-9A-Za-z_-]{11})', url)
    if not video_id_match:
        logger.error(f"Invalid YouTube URL or video ID couldn't be extracted: {url}")
        return {"success": False, "error": "Invalid YouTube URL or video ID"}
    
    video_id = video_id_match.group(1)
    clean_url = f"https://www.youtube.com/watch?v={video_id}" # Use canonical URL for yt-dlp

    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio[abr<=128]/bestaudio/best', # Ensure m4a preferred
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a', # Telegram prefers m4a/aac
        }],
        'prefer_ffmpeg': True, # Necessary for postprocessing to specific codec
        'max_filesize': 50 * 1024 * 1024,  # 50 MB limit for Telegram
        'nocheckcertificate': True, # Can help in some environments
        'socket_timeout': 30, # Timeout for network operations
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(clean_url, download=False)
            if not info:
                return {"success": False, "error": "Could not extract video information."}

            file_title = sanitize_filename(info.get('title', f'audio_{video_id}'))
            artist = sanitize_filename(info.get('artist', info.get('uploader', 'Unknown Artist')))
            
            # Construct expected path more reliably
            # yt-dlp might change the extension based on actual format/postprocessing
            # For m4a, the extension will be m4a
            expected_path_base = os.path.join(DOWNLOAD_DIR, file_title)
            
            logger.info(f"Starting download for: {info.get('title', 'Unknown Title')}")
            ydl.download([clean_url])
            
            # Find the downloaded file, should be m4a if postprocessing worked
            audio_path = f"{expected_path_base}.m4a"
            if not os.path.exists(audio_path): # Fallback to other common extensions if m4a not found
                possible_extensions = ['mp3', 'webm', 'opus', 'ogg']
                for ext in possible_extensions:
                    temp_path = f"{expected_path_base}.{ext}"
                    if os.path.exists(temp_path):
                        audio_path = temp_path
                        logger.warning(f"Expected m4a, but found .{ext} for {file_title}")
                        break
                if not os.path.exists(audio_path):
                    logger.error(f"Downloaded file not found for title: {file_title}. Searched for .m4a and fallbacks.")
                    return {"success": False, "error": "Downloaded file not found after processing."}

            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            if file_size_mb > 50:
                logger.error(f"File '{audio_path}' too large: {file_size_mb:.2f} MB (limit 50MB)")
                if os.path.exists(audio_path): os.remove(audio_path)
                return {"success": False, "error": "File is too large for Telegram (max 50MB)."}
            
            logger.info(f"Successfully downloaded: {audio_path}")
            return {
                "success": True, "title": info.get('title', 'Unknown Title'),
                "artist": artist, "thumbnail_url": info.get('thumbnail'),
                "duration": info.get('duration', 0), "audio_path": audio_path
            }
    except yt_dlp.utils.MaxDownloadsReached:
        return {"success": False, "error": "Max downloads reached (should not happen for single URL)."}
    except yt_dlp.utils.SameFileError:
        # This can happen if file already exists. Assume it's fine if path logic is robust
        logger.warning(f"File already exists (SameFileError) for {url}. Re-checking path.")
        # Check if the expected file exists from a previous attempt
        file_title_temp = sanitize_filename(ydl.extract_info(clean_url, download=False).get('title', f'audio_{video_id}'))
        potential_path = os.path.join(DOWNLOAD_DIR, f"{file_title_temp}.m4a")
        if os.path.exists(potential_path) and os.path.getsize(potential_path) / (1024 * 1024) <= 50:
            return {"success": True, "title": file_title_temp, "artist": "Unknown", "audio_path": potential_path}
        return {"success": False, "error": "File existed but could not be reused or was invalid."}
    except yt_dlp.utils.DownloadError as e:
        # More specific error messages for common issues
        err_str = str(e).lower()
        if "video unavailable" in err_str:
            return {"success": False, "error": "Video is unavailable."}
        if "age restricted" in err_str:
            return {"success": False, "error": "Video is age-restricted and cannot be accessed."}
        if "private video" in err_str:
            return {"success": False, "error": "Video is private."}
        logger.error(f"YouTube download error for {url}: {e}")
        return {"success": False, "error": f"Download failed: {e}"}
    except Exception as e:
        logger.error(f"Unexpected error downloading YouTube audio for {url}: {e}", exc_info=True)
        return {"success": False, "error": "An unexpected error occurred during download."}

@lru_cache(maxsize=100)
def search_youtube(query: str, max_results: int = 5) -> List[Dict]:
    """Search YouTube for videos matching the query with caching."""
    query = sanitize_input(query)
    if not query: return []
    try:
        ydl_opts = {
            'quiet': True, 'no_warnings': True, 'extract_flat': 'discard_in_playlist',
            'default_search': 'ytsearch', 'noplaylist': True,
            'playlist_items': f'1-{max_results}'
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
            if not info or 'entries' not in info: return []
            
            results = []
            for entry in info.get('entries', []):
                if not entry or not entry.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', entry['id']):
                    logger.warning(f"Skipping search result with invalid/missing ID: {entry.get('title', 'N/A')}")
                    continue
                results.append({
                    'title': entry.get('title', 'Unknown Title'),
                    'url': entry.get('webpage_url') or f"https://www.youtube.com/watch?v={entry['id']}",
                    'thumbnail': entry.get('thumbnail'),
                    'uploader': entry.get('uploader', 'Unknown Artist'),
                    'duration': entry.get('duration', 0),
                    'id': entry['id']
                })
            return results
    except Exception as e:
        logger.error(f"Error searching YouTube for '{query}': {e}", exc_info=True)
        return []

# ==================== LYRICS FUNCTION ====================
@lru_cache(maxsize=50)
def get_lyrics_from_genius(song_title: str, artist_name: Optional[str] = None) -> str:
    """Fetch lyrics using LyricsGenius, with sanitization and cleaning."""
    if not genius:
        return "Lyrics service is not available (Genius client not initialized)."
    
    try:
        # Basic cleaning for search terms
        clean_song_title = re.sub(r"\(.*?\)|\[.*?\]", "", song_title).strip().lower()
        search_term = clean_song_title
        if artist_name:
            clean_artist_name = re.sub(r"\(.*?\)|\[.*?\]", "", artist_name).strip().lower()
            search_term = f"{clean_song_title} {clean_artist_name}"
        
        logger.info(f"Searching Genius lyrics for: '{search_term}'")
        song = genius.search_song(clean_song_title, artist_name if artist_name else None) # genius handles its own artist search logic

        if song and song.lyrics:
            lyrics_text = song.lyrics.strip()
            # Remove common Genius artifacts
            lyrics_text = re.sub(r'^.*Lyrics\n', '', lyrics_text, flags=re.IGNORECASE | re.MULTILINE) # Header like "Song Title Lyrics"
            lyrics_text = re.sub(r'\d*EmbedShare URLCopyEmbedCopy$', '', lyrics_text).strip() # Footer
            lyrics_text = re.sub(r'You might also like', '', lyrics_text, flags=re.IGNORECASE).strip() # "You might also like" section

            # Check for placeholder lyrics
            if "(missing lyrics)" in lyrics_text.lower() or \
               "lyrics for this song have yet to be released" in lyrics_text.lower() or \
               not lyrics_text or len(lyrics_text) < 20: # Arbitrary short length
                return f"Sorry, lyrics for '{song_title}' seem to be missing, incomplete, or not yet released on Genius."

            return f"📜 <b>{song.title}</b> by {song.artist}\n\n{lyrics_text}"
        else:
            logger.info(f"No lyrics found on Genius for: '{search_term}'")
            return f"Sorry, I couldn't find lyrics for '{song_title}'" + (f" by '{artist_name}'." if artist_name else ".")
            
    except requests.exceptions.Timeout:
        logger.warning(f"Genius API timeout for lyrics: '{song_title}'")
        return "The lyrics search timed out. Please try again."
    except Exception as e:
        logger.error(f"Error fetching lyrics from Genius for '{song_title}': {e}", exc_info=True)
        return "An error occurred while fetching lyrics from Genius."

# ==================== AI HELPER FUNCTIONS ====================

async def generate_chat_response(user_id: int, message: str) -> str:
    """Generate a conversational response using OpenAI."""
    if not client:
        return "I'm having trouble connecting to my AI service. Please try again later."

    message = sanitize_input(message) # Sanitize input first
    user_contexts.setdefault(user_id, {
        "mood": None, "preferences": [], "conversation_history": [], "spotify": {}
    })
    context = user_contexts[user_id]

    # Construct messages for OpenAI API
    messages = [{"role": "system", "content": (
        "You are MelodyMind, a friendly, empathetic music companion bot. Your primary goals are: "
        "1. Engage in natural, supportive conversations about music and feelings. "
        "2. Offer music recommendations tailored to the user's mood and preferences (use Spotify data if available). "
        "3. Provide lyrics or download links when asked or contextually appropriate. "
        "4. Keep responses concise (2-4 sentences), warm, and helpful. "
        "5. If unsure or if a request is vague, ask clarifying questions. "
        "6. Do not suggest commands like /recommend unless specifically relevant. Integrate actions smoothly. "
        "For example, if user says 'I'm sad', you could say 'I'm sorry to hear that. Music can sometimes help. "
        "Would you like me to find some comforting songs for you, or perhaps you'd like to talk about it?'"
    )}]

    # Add context about mood and Spotify
    mood_info = f"Current Mood: {context.get('mood', 'Unknown')}. "
    pref_info = f"Preferences: {', '.join(context.get('preferences', []))}. "
    spotify_info = ""
    if context.get("spotify", {}).get("recently_played"):
        artists = list(set(item["track"]["artists"][0]["name"] for item in context["spotify"]["recently_played"][:5] if item.get("track")))
        spotify_info = f"Recently listened to on Spotify: {', '.join(artists[:3])}. "
    elif context.get("spotify", {}).get("top_tracks"):
        artists = list(set(item["artists"][0]["name"] for item in context["spotify"]["top_tracks"][:5] if item.get("artists")))
        spotify_info = f"Top artists on Spotify: {', '.join(artists[:3])}. "
    
    if context.get("mood") or context.get("preferences") or spotify_info:
        messages.append({"role": "system", "content": f"User Info: {mood_info}{pref_info}{spotify_info}"})


    # Add recent conversation history (max 10 pairs = 20 messages)
    history_limit = 20 
    context["conversation_history"] = context["conversation_history"][-history_limit:]
    messages.extend(context["conversation_history"])
    messages.append({"role": "user", "content": message})

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo", messages=messages,
            max_tokens=150, temperature=0.75 # Slightly higher temp for more natural feel
        )
        reply = response.choices[0].message.content.strip()
        
        # Update conversation history
        context["conversation_history"].append({"role": "user", "content": message})
        context["conversation_history"].append({"role": "assistant", "content": reply})
        context["conversation_history"] = context["conversation_history"][-history_limit:] # Enforce limit
        user_contexts[user_id] = context
        return reply
    except Exception as e:
        logger.error(f"Error generating chat response for user {user_id}: {e}", exc_info=True)
        return "I'm having a little trouble thinking right now. How about we try finding some music for you instead?"

async def is_music_request(user_id: int, message: str) -> Dict:
    """Use AI to determine if a message is a music/song request and extract query."""
    if not client: return {"is_music_request": False, "song_query": None}

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo-0125", # Newer model, good with JSON
            messages=[
                {"role": "system", "content": 
                    "You are an AI that analyzes user messages. Determine if the message is a request for "
                    "a specific song, artist, or music download/search. If it is, extract a concise search query "
                    "(song title and optionally artist). Respond in JSON format with two keys: "
                    "'is_music_request' (boolean) and 'song_query' (string, or null if not a music request or no clear query)."
                },
                {"role": "user", "content": f"Analyze this message: '{sanitize_input(message)}'"}
            ],
            max_tokens=80, temperature=0.1, response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)

        is_request = result.get("is_music_request", False)
        if isinstance(is_request, str): # Handle if AI returns "true"/"false" as string
            is_request = is_request.lower() in ("true", "yes")
            
        song_query = result.get("song_query")
        if song_query and isinstance(song_query, str) and len(song_query.strip()) > 0:
             return {"is_music_request": bool(is_request), "song_query": song_query.strip()}
        else: # No valid query even if it's a request
             return {"is_music_request": bool(is_request), "song_query": None}

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError in is_music_request AI response: {content}, Error: {e}")
        return {"is_music_request": False, "song_query": None}
    except Exception as e:
        logger.error(f"Error in is_music_request for user {user_id}: {e}", exc_info=True)
        return {"is_music_request": False, "song_query": None}


async def analyze_conversation(user_id: int) -> Dict:
    """Analyze conversation history and Spotify data to extract preferences for recommendations."""
    if not client: return {"genres": [], "artists": [], "mood": None}

    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    context = user_contexts[user_id]

    # If very little data, rely on explicit context
    if len(context.get("conversation_history", [])) < 2 and not context.get("spotify") and not context.get("mood"):
        return {"genres": context.get("preferences", []), "artists": [], "mood": context.get("mood")}

    conversation_summary = ""
    if context.get("conversation_history"):
        # Select salient parts of history if it's long
        history_texts = [f"{msg['role']}: {msg['content']}" for msg in context["conversation_history"][-10:]] # Last 5 pairs
        conversation_summary = "\n".join(history_texts)

    spotify_summary = ""
    if context.get("spotify"):
        recently_played_tracks = context["spotify"].get("recently_played", [])
        top_tracks_info = context["spotify"].get("top_tracks", [])
        
        rp_summary = ", ".join([f"{item['track']['name']} by {item['track']['artists'][0]['name']}" 
                                for item in recently_played_tracks[:3] if item.get("track")])
        tt_summary = ", ".join([f"{item['name']} by {item['artists'][0]['name']}" 
                                for item in top_tracks_info[:3] if item.get("artists")])
        
        if rp_summary: spotify_summary += f"Recently played: {rp_summary}. "
        if tt_summary: spotify_summary += f"Top tracks: {tt_summary}."

    # Fallback if AI can't run or analyze
    fallback_result = {"genres": context.get("preferences", []), "artists": [], "mood": context.get("mood")}

    try:
        prompt_content = (
            "Analyze the following user interaction data to identify music preferences. "
            "Focus on explicit mentions of genres, artists, songs, and expressed mood. "
            "If Spotify data is available, prioritize it for artists and genres. "
            "Return a JSON object with keys: 'genres' (list of strings), 'artists' (list of strings), "
            "'mood' (string, e.g., 'happy', 'sad', 'energetic', 'reflective', or null if not clear). "
            f"User's current explicit mood setting: {context.get('mood', 'None')}.\n\n"
            f"Conversation Snippet:\n{conversation_summary if conversation_summary else 'No conversation yet.'}\n\n"
            f"Spotify Listening Data:\n{spotify_summary if spotify_summary else 'No Spotify data available.'}"
        )
        
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a music preference analyzer."},
                {"role": "user", "content": prompt_content}
            ],
            max_tokens=200, temperature=0.2, response_format={"type": "json_object"}
        )
        
        result_str = response.choices[0].message.content
        analysis = json.loads(result_str)

        # Validate and clean up AI output
        genres = analysis.get("genres", [])
        if isinstance(genres, str): genres = [g.strip() for g in genres.split(",") if g.strip()]
        if not isinstance(genres, list): genres = []
        
        artists = analysis.get("artists", [])
        if isinstance(artists, str): artists = [a.strip() for a in artists.split(",") if a.strip()]
        if not isinstance(artists, list): artists = []

        mood = analysis.get("mood")
        if not isinstance(mood, str) or mood.lower() == "null" or mood.lower() == "none":
            mood = context.get("mood") # Fallback to explicitly set mood

        # Update context with AI derived preferences if they are new
        if genres and not context.get("preferences"): context["preferences"] = list(set(genres))[:3]
        if mood and not context.get("mood"): context["mood"] = mood
        user_contexts[user_id] = context

        return {"genres": genres[:3], "artists": artists[:3], "mood": mood}

    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError in analyze_conversation AI response: {result_str}, Error: {e}")
        return fallback_result
    except Exception as e:
        logger.error(f"Error in analyze_conversation for user {user_id}: {e}", exc_info=True)
        return fallback_result

# ==================== MUSIC DETECTION (Non-AI) ====================

def detect_music_in_message(text: str) -> Optional[str]:
    """Detect if a message is asking for music (regex based, for quick checks)."""
    # More specific patterns, less likely for false positives compared to AI.
    # Focus on "play", "download", "find song" followed by something.
    # Uses non-capturing groups (?:) for "by" or "from".
    patterns = [
        r'(?:play|download|find song|get song|search for song)\s+(.+?)(?:\s+by\s+(.+))?$',
        r'i want to listen to\s+(.+?)(?:\s+by\s+(.+))?$',
        r'song\s+(.+?)(?:\s+by\s+(.+))?$',
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE)
        if match:
            song_title = match.group(1).strip()
            artist = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else None
            query = f"{song_title} {artist}" if artist else song_title
            if query and len(query) > 2 : # Basic check for meaningful query
                return query.strip()

    # If keywords like 'music', 'song' appear but not in specific patterns, might need AI.
    # No "AI_ANALYSIS_NEEDED" here; let main handler decide on AI call.
    return None

# ==================== TELEGRAM BOT HANDLERS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    user_contexts.setdefault(update.effective_user.id, { # Initialize context if not present
        "mood": None, "preferences": [], "conversation_history": [], "spotify": {}
    })
    welcome_msg = (
        f"Hi {user.first_name}! 👋 I'm MelodyMind, your Music Healing Companion.\n\n"
        "I can:\n"
        "🎵 Download music from YouTube (send a link or ask for a song)\n"
        "📜 Find lyrics for songs\n"
        "🎧 Recommend music based on your mood or Spotify listening\n"
        "💬 Chat about music and how you're feeling\n\n"
        "<b>Key Commands:</b>\n"
        "/help - Show all commands & tips\n"
        "/recommend - Get personalized music recommendations\n"
        "/mood - Tell me how you're feeling\n"
        "/link_spotify - Connect your Spotify for even better recommendations\n\n"
        "Just start chatting, ask for a song, or use a command!"
    )
    await update.message.reply_text(welcome_msg, parse_mode=ParseMode.HTML)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = (
        "🎶 <b>MelodyMind - Your Music Healing Companion</b> 🎶\n\n"
        "<b>Core Commands:</b>\n"
        "/start - Welcome message & basic info\n"
        "/help - This help guide\n"
        "/recommend - Get personalized music recommendations\n"
        "/mood - Set your current mood for tailored suggestions\n"
        "/link_spotify - Connect your Spotify account\n"
        "/lyrics <code>[song name]</code> or <code>[artist - song]</code> - Fetch lyrics\n"
        "/search <code>[song name]</code> - Search for a song on YouTube\n"
        "/download <code>[YouTube URL]</code> - Download audio from a YouTube link\n"
        "/autodownload <code>[song name]</code> - Search & download the top YouTube result\n"
        "/clear - Clear your chat history with me\n\n"
        "<b>Natural Language Interaction:</b>\n"
        "You can also just talk to me! For example:\n"
        "- \"<i>Play Shape of You by Ed Sheeran</i>\"\n"
        "- \"<i>I'm feeling a bit down today, any music suggestions?</i>\"\n"
        "- \"<i>What are the lyrics to Hotel California?</i>\"\n"
        "- \"<i>Download this: [paste YouTube link]</i>\"\n\n"
        "✨ The more you interact and if you link Spotify, the better I get at understanding your taste!"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

async def _execute_download_and_send(update_or_query: Union[Update, InlineKeyboardMarkup], 
                                     context: ContextTypes.DEFAULT_TYPE, 
                                     video_url: str, 
                                     status_message: Any,
                                     chat_id: int,
                                     user_id: int):
    """Helper to encapsulate download and send logic."""
    try:
        if hasattr(status_message, 'edit_text'):
            await status_message.edit_text("⏳ Fetching video information...")
        
        result = await asyncio.to_thread(download_youtube_audio, video_url)

        if not result or not result.get("success"):
            err_msg = result.get("error", "Unknown download error.")
            logger.error(f"Download failed for {video_url}: {err_msg}")
            if hasattr(status_message, 'edit_text'):
                await status_message.edit_text(f"❌ Download failed: {err_msg}")
            else: # if status_message was deleted or is not editable
                await context.bot.send_message(chat_id, f"❌ Download failed for {video_url}: {err_msg}")
            return

        file_path = result["audio_path"]
        title = result.get("title", "Unknown Title")
        artist = result.get("artist", "Unknown Artist")

        if hasattr(status_message, 'edit_text'):
            await status_message.edit_text(f"✅ Downloaded: {title}\n⏳ Preparing to send...")
        
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        logger.info(f"Sending audio: '{title}', size: {file_size_mb:.2f}MB from path {file_path}")

        audio_file = open(file_path, 'rb')
        try:
            # Use retry for sending audio as it can be flaky
            @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10),
                   retry=lambda retry_state: isinstance(retry_state.outcome.exception(), (TimedOut, NetworkError)))
            async def send_audio_with_retry():
                # Ensure file is seeked to start for retries
                audio_file.seek(0) 
                # Need to use the application's bot directly or pass it
                return await context.bot.send_audio(
                    chat_id=chat_id,
                    audio=audio_file, # Pass the file object
                    title=title[:60], # Telegram title limit
                    performer=artist[:60],
                    caption=f"🎵 {title}",
                    timeout=120 # Generous timeout for large files
                )
            await send_audio_with_retry()
            logger.info(f"Successfully sent: {title}")
            if hasattr(status_message, 'edit_text'):
                 await status_message.edit_text(f"✅ Sent: {title}") # Final status update if possible
            elif hasattr(status_message, 'delete'): # If it cannot be edited, maybe delete
                await status_message.delete()

        except RetryError as e: # Tenacity specific error after retries exhausted
            logger.error(f"Failed to send audio {title} after retries: {e.last_attempt.exception()}", exc_info=True)
            final_error_message = f"❌ Failed to send {title} after several attempts. Network issues?"
            if hasattr(status_message, 'edit_text'):
                 await status_message.edit_text(final_error_message)
            else:
                 await context.bot.send_message(chat_id, final_error_message)
        except (TimedOut, NetworkError) as e:
            logger.error(f"Telegram API error sending audio {title}: {e}", exc_info=True)
            error_message = f"❌ Telegram API error sending {title}. Please try again later."
            if hasattr(status_message, 'edit_text'):
                await status_message.edit_text(error_message)
            else:
                 await context.bot.send_message(chat_id, error_message)
        except Exception as e: # Catchall for other unexpected errors during send
            logger.error(f"Unexpected error sending audio {title}: {e}", exc_info=True)
            error_message = f"❌ An unexpected error occurred sending {title}."
            if hasattr(status_message, 'edit_text'):
                await status_message.edit_text(error_message)
            else:
                 await context.bot.send_message(chat_id, error_message)
        finally:
            audio_file.close() # Ensure file is closed
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    logger.info(f"Deleted temporary file: {file_path}")
                except OSError as e:
                    logger.error(f"Error deleting temporary file {file_path}: {e}")
                    
    except Exception as e: # Outer try-except for the whole download-send block
        logger.error(f"Outer error in _execute_download_and_send for {video_url}: {e}", exc_info=True)
        try:
            if hasattr(status_message, 'edit_text'):
                await status_message.edit_text(f"❌ An unexpected error occurred during the download process. {str(e)[:100]}")
            else:
                 await context.bot.send_message(chat_id, f"❌ An unexpected error occurred processing {video_url}. {str(e)[:100]}")
        except Exception: # If sending error message itself fails
            logger.error("Failed to even send the outer error message in _execute_download_and_send.")

async def download_music_handler_logic(update_or_query: Union[Update, InlineKeyboardMarkup], 
                                       context: ContextTypes.DEFAULT_TYPE, 
                                       video_url: Optional[str] = None):
    """Handles download logic, called by commands or callbacks."""
    is_callback = isinstance(update_or_query, Update) and update_or_query.callback_query is not None
    
    if is_callback:
        query_obj = update_or_query.callback_query # Renaming for clarity
        user_id = query_obj.from_user.id
        chat_id = query_obj.message.chat_id
        # video_url is passed directly for callbacks
        status_message = await query_obj.edit_message_text("🚀 Preparing download...")
    else: # Is an Update object from a command or message
        user_id = update_or_query.effective_user.id
        chat_id = update_or_query.effective_chat.id
        message_text = update_or_query.message.text
        
        if context.args: # From /download <url>
            video_url = " ".join(context.args)
        else: # From message text or /autodownload <query> needs video_url determined
            urls = [word for word in message_text.split() if is_valid_youtube_url(word)]
            video_url = urls[0] if urls else None

        if not video_url or not is_valid_youtube_url(video_url):
            await update_or_query.message.reply_text("❌ Please provide a valid YouTube URL or ask for a song by name.")
            return
        status_message = await update_or_query.message.reply_text("🚀 Preparing download...")
    
    if not video_url: # Should be caught above for commands, but double check for safety
        if hasattr(status_message, 'edit_text'): await status_message.edit_text("Could not determine video URL.")
        else: await context.bot.send_message(chat_id, "Could not determine video URL.")
        return

    # --- Lock and Active Download Management ---
    if user_id in active_downloads: # Quick check
        msg_text = "⚠️ You already have a download in progress. Please wait."
        if is_callback: await status_message.edit_text(msg_text) # Edit previous status if callback
        else: await status_message.edit_text(msg_text) # Edit or reply if command
        return

    async with download_lock: # Acquire global download lock
        if user_id in active_downloads: # Re-check inside lock (critical section)
            msg_text = "⚠️ Another of your download requests just started. Please wait."
            if is_callback: await status_message.edit_text(msg_text)
            else: await status_message.edit_text(msg_text)
            return
        active_downloads.add(user_id)

        try:
            await _execute_download_and_send(update_or_query, context, video_url, status_message, chat_id, user_id)
        finally:
            # Ensure user_id is removed from active_downloads set *within the lock's scope*
            if user_id in active_downloads:
                active_downloads.remove(user_id)
            # Status message is handled within _execute_download_and_send,
            # but if it was a callback message that was NOT updated, we might want to delete it.
            # However, _execute_download_and_send tries to update it to a final state.
            # If _execute_download_and_send crashes hard, status_message might be left hanging.
            # Deleting it here might be too aggressive if _execute_download_and_send successfully sent audio.
            # For now, rely on _execute_download_and_send to manage its status_message lifecycle.
            pass
            

async def download_music_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Called by /download command."""
    await download_music_handler_logic(update, context)


async def auto_download_first_result(update: Update, context: ContextTypes.DEFAULT_TYPE, song_query: str, chat_id: int) -> None:
    """Search for a song and initiate download of the first result."""
    user_id = update.effective_user.id

    status_msg = await context.bot.send_message(chat_id, f"🔍 Searching for '{song_query}'...")
    
    results = search_youtube(song_query, max_results=1) # Already cached and sanitized
    if not results or not results[0].get("id"):
        await status_msg.edit_text(f"❌ Couldn't find any valid results for '{song_query}'.")
        return

    video_url = results[0]["url"]
    title = results[0]['title']
    
    # Now, manage locks and active_downloads for this specific user before calling _execute_download_and_send
    if user_id in active_downloads:
        await status_msg.edit_text(f"⚠️ You already have a download for '{title}' (or other) in progress. Please wait.")
        return

    async with download_lock:
        if user_id in active_downloads:
            await status_msg.edit_text(f"⚠️ Another of your download requests for '{title}' (or other) just started. Please wait.")
            return
        active_downloads.add(user_id)
        try:
            await status_msg.edit_text(f"✅ Found: {title}\n⏳ Download starting...")
            await _execute_download_and_send(update, context, video_url, status_msg, chat_id, user_id)
        finally:
            if user_id in active_downloads:
                active_downloads.remove(user_id)

async def link_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Initiate Spotify OAuth flow."""
    if not all([SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET, SPOTIFY_REDIRECT_URI]):
        await update.message.reply_text("Sorry, Spotify linking is not fully configured by the bot admin.")
        return ConversationHandler.END
    if SPOTIFY_REDIRECT_URI == "https://your-callback-url.com":
        await update.message.reply_text(
            "⚠️ Spotify linking requires the bot admin to set a `SPOTIFY_REDIRECT_URI` in the configuration. "
            "This feature might not work correctly until that is done."
        )
        # Proceed anyway, user might still be able to get the code from the URL if they know how.

    user_id = update.effective_user.id
    # Required scopes for functionalities used
    scopes = "user-read-recently-played user-top-read playlist-read-private playlist-read-collaborative"
    auth_url = (
        f"https://accounts.spotify.com/authorize"
        f"?client_id={SPOTIFY_CLIENT_ID}"
        f"&response_type=code"
        f"&redirect_uri={SPOTIFY_REDIRECT_URI}"
        f"&scope={requests.utils.quote(scopes)}" # URL-encode scopes
        f"&state={user_id}" # Using user_id as state for potential validation
    )
    keyboard = [
        [InlineKeyboardButton("🔗 Link My Spotify", url=auth_url)],
        [InlineKeyboardButton("Cancel", callback_data="cancel_spotify_link")]
    ]
    instructions = (
        "Let's link your Spotify account to get personalized music recommendations! 🎵\n\n"
        "1. Click the 'Link My Spotify' button below. This will take you to Spotify's login page.\n"
        "2. Log in and authorize MelodyMind.\n"
        "3. After authorization, Spotify will redirect you. *This redirect page might show an error* like 'Site can't be reached', especially if the redirect URL is a placeholder. This is usually OK.\n"
        "4. **Copy the ENTIRE URL** from your browser's address bar on that redirect page.\n"
        "5. Paste the full URL back here in the chat.\n\n"
        "Ready? Click below to start:"
    )
    await update.message.reply_text(instructions, reply_markup=InlineKeyboardMarkup(keyboard))
    return SPOTIFY_CODE

async def spotify_code_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle Spotify authorization code from conversation or /spotify_code command."""
    user_id = update.effective_user.id
    message_text = update.message.text.strip()
    code = None

    # Try to extract code from a full URL first
    code_match = re.search(r"[?&]code=([^&]+)", message_text)
    if code_match:
        code = code_match.group(1)
        logger.info(f"Extracted Spotify code from URL for user {user_id}")
    elif not message_text.startswith('/'): # Assume raw code if not a command and not a URL
        code = message_text
        logger.info(f"Received raw Spotify code string for user {user_id}")
    elif message_text.startswith('/spotify_code ') and context.args: # Explicit command
        code = context.args[0]
        logger.info(f"Received Spotify code via /spotify_code command for user {user_id}")
    else:
        await update.message.reply_text(
            "Hmm, that doesn't look like a Spotify authorization code or the redirect URL. "
            "Please paste the *full URL* from your browser after authorizing on Spotify, or just the code itself. "
            "Or, you can use /link_spotify to try again."
        )
        return SPOTIFY_CODE # Remain in the same state

    if not code: # Should be caught by else above, but as safety
        await update.message.reply_text("No code found. Please try again.")
        return SPOTIFY_CODE

    status_msg = await update.message.reply_text("🔄 Verifying your Spotify authorization...")
    token_data = get_user_spotify_token(user_id, code) # This is a synchronous call

    if not token_data or not token_data.get("access_token"):
        await status_msg.edit_text(
            "❌ Failed to link Spotify account. The code might be invalid, expired, or the redirect URI might be misconfigured. "
            "Please try /link_spotify again. Ensure you copy the *full URL* from the page Spotify redirects you to."
        )
        return SPOTIFY_CODE # Stay in SPOTIFY_CODE state to allow another attempt

    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    user_contexts[user_id]["spotify"] = {
        "access_token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "expires_at": token_data.get("expires_at") # Already calculated in get_user_spotify_token
    }
    
    # Fetch initial data to confirm linking works
    await status_msg.edit_text("✅ Spotify account linked! Fetching initial listening data...")
    recently_played = get_user_spotify_data(user_id, "player/recently-played", params={"limit": 5})
    if recently_played:
        user_contexts[user_id]["spotify"]["recently_played"] = recently_played
        await status_msg.edit_text(
            "🎉 Spotify account linked successfully! I can now see your recent listening history. "
            "Try /recommend to get music suggestions tailored to you!"
        )
    else:
        await status_msg.edit_text(
            "✅ Spotify account linked! However, I couldn't fetch your recent listening data right away. "
            "Don't worry, I'll try again when you ask for recommendations. Try /recommend!"
        )
    return ConversationHandler.END

async def spotify_code_command_entry(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Entry point for /spotify_code command when not in conversation. Effectively calls spotify_code_handler."""
    if not context.args:
        await update.message.reply_text(
            "Please provide the Spotify authorization code or the full URL from Spotify. Example:\n"
            "/spotify_code <your_code_here>\nOr paste the full URL like: https://your-redirect-uri.com/?code=XXXXX&state=YYYYY"
        )
        return ConversationHandler.END # Or a specific state if you want to transition
    # Mimic calling the spotify_code_handler logic. It expects to be in SPOTIFY_CODE state.
    # For simplicity, directly call it if it's designed to handle being called outside conv.
    # The current spotify_code_handler doesn't depend on conversation state for its logic other than returning it.
    return await spotify_code_handler(update, context)


async def cancel_spotify_link(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Spotify linking cancelled. You can try again anytime with /link_spotify.")
    return ConversationHandler.END


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Please specify what you're looking for. Example:\n/search Shape of You"
        )
        return

    query = sanitize_input(" ".join(context.args))
    status_msg = await update.message.reply_text(f"🔍 Searching YouTube for: '{query}'...")
    results = search_youtube(query) # Already cached

    if not results:
        await status_msg.edit_text(f"Sorry, I couldn't find any YouTube results for '{query}'.")
        return

    keyboard = []
    response_text = f"🔎 YouTube Search Results for '{query}':\n\n"
    for i, result in enumerate(results[:5], 1): # Show top 5
        duration_str = ""
        if result.get('duration') and isinstance(result['duration'], (int, float)) and result['duration'] > 0:
            minutes, seconds = divmod(int(result['duration']), 60)
            duration_str = f" [{minutes}:{seconds:02d}]"
        
        title = result['title'][:50] + "..." if len(result['title']) > 50 else result['title']
        response_text += f"{i}. <b>{result['title']}</b> by {result.get('uploader', 'N/A')}{duration_str}\n"
        # Video ID already validated in search_youtube
        keyboard.append([InlineKeyboardButton(f"⬇️ {title}", callback_data=f"download_{result['id']}")])

    keyboard.append([InlineKeyboardButton("Cancel Search", callback_data="cancel_search_results")])
    await status_msg.edit_text(response_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)

async def auto_download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Please specify what song you want. Example:\n/autodownload Shape of You"
        )
        return
    song_query = sanitize_input(" ".join(context.args))
    await auto_download_first_result(update, context, song_query, update.effective_chat.id)

async def get_lyrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text(
            "Please specify a song. Examples:\n"
            "/lyrics Bohemian Rhapsody\n"
            "/lyrics Queen - Bohemian Rhapsody"
        )
        return

    query = sanitize_input(" ".join(context.args))
    status_msg = await update.message.reply_text(f"🔍 Searching for lyrics: \"{query}\"...")

    artist = None
    song_title_query = query
    # Simple parsing for "artist - song" or "song by artist"
    if " - " in query:
        parts = query.split(" - ", 1)
        artist, song_title_query = parts[0].strip(), parts[1].strip()
    elif " by " in query.lower(): # "song by artist"
        parts = re.split(r'\s+by\s+', query, maxsplit=1, flags=re.IGNORECASE)
        song_title_query, artist = parts[0].strip(), parts[1].strip()

    lyrics_result = await asyncio.to_thread(get_lyrics_from_genius, song_title_query, artist)
    
    if len(lyrics_result) > 4096: # Telegram message limit
        await status_msg.edit_text("Lyrics found, but they are too long! Sending in parts...")
        parts = [lyrics_result[i:i + 4000] for i in range(0, len(lyrics_result), 4000)]
        for i, part_text in enumerate(parts):
            msg_suffix = f"\n\n--- Part {i+1}/{len(parts)} ---" if len(parts) > 1 else ""
            await update.message.reply_text(part_text + msg_suffix, parse_mode=ParseMode.HTML)
    else:
        await status_msg.edit_text(lyrics_result, parse_mode=ParseMode.HTML)

async def recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    status_msg = await context.bot.send_message(chat_id, "🎧 Thinking about some music for you...")

    try:
        user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
        
        # Attempt to update Spotify data if linked
        if user_contexts[user_id].get("spotify", {}).get("access_token"):
            logger.info(f"Updating Spotify data for user {user_id} before recommendation.")
            # These are synchronous, consider asyncio.to_thread if they become slow
            # Limiting fetches to reduce wait time.
            recently_played = get_user_spotify_data(user_id, "player/recently-played", {"limit": 5})
            if recently_played: user_contexts[user_id]["spotify"]["recently_played"] = recently_played
            
            top_tracks = get_user_spotify_data(user_id, "top/tracks", {"limit": 5})
            if top_tracks: user_contexts[user_id]["spotify"]["top_tracks"] = top_tracks

            # Only fetch playlists if other seeds are sparse
            if not recently_played and not top_tracks:
                 playlists = get_user_spotify_playlists(user_id) # Fetches default limit (e.g. 5)
                 if playlists: user_contexts[user_id]["spotify"]["playlists"] = playlists

        # Analyze conversation for mood, genres, artists
        await status_msg.edit_text("🔍 Analyzing your preferences and listening history...")
        analysis = await analyze_conversation(user_id)
        
        mood = analysis.get("mood")
        genres = analysis.get("genres", [])
        artists = analysis.get("artists", [])

        if not mood: # If AI couldn't determine mood and not set, prompt user
            await status_msg.delete()
            keyboard = [
                [InlineKeyboardButton("Happy 😊", callback_data="mood_happy"), InlineKeyboardButton("Sad 😢", callback_data="mood_sad")],
                [InlineKeyboardButton("Energetic 💪", callback_data="mood_energetic"), InlineKeyboardButton("Relaxed 😌", callback_data="mood_relaxed")],
                [InlineKeyboardButton("Focused 🧠", callback_data="mood_focused"), InlineKeyboardButton("Nostalgic 🕰️", callback_data="mood_nostalgic")],
                [InlineKeyboardButton("No specific mood", callback_data="mood_any")]
            ]
            await context.bot.send_message(chat_id, 
                "I'd love to recommend some music! First, how are you feeling today, or what kind of vibe are you looking for?",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return MOOD # Transition to mood state for ConversationHandler if /mood command leads here.
                       # If called directly, this return is just informational.

        await status_msg.edit_text(f"Okay, looking for {mood} music" + 
                                   (f" with genres like {', '.join(genres)}" if genres else "") +
                                   (f" similar to artists like {', '.join(artists)}" if artists else "") + "...")
        
        # Spotify Recommendations
        spotify_access_token = get_spotify_token() # Client credentials token
        seed_track_ids = []
        seed_artist_ids = [] # Future: Use seed_artists
        seed_genres = genres[:2] # Future: Use seed_genres, limit to 2 as per Spotify max 5 seeds total

        # Prioritize user's Spotify data for seeds
        user_spotify = user_contexts[user_id].get("spotify", {})
        if user_spotify.get("recently_played"):
            seed_track_ids.extend([t["track"]["id"] for t in user_spotify["recently_played"][:2] if t.get("track", {}).get("id")])
        if not seed_track_ids and user_spotify.get("top_tracks"):
            seed_track_ids.extend([t["id"] for t in user_spotify["top_tracks"][:2] if t.get("id")])
        
        # If specific artists mentioned in analysis, try to find their Spotify IDs
        if not seed_track_ids and artists and spotify_access_token:
            for artist_name in artists[:2]: # Max 2 artist seeds for now
                track_info = search_spotify_track(spotify_access_token, f"artist:{artist_name}")
                if track_info and track_info.get("id"):
                    seed_track_ids.append(track_info["id"])
        
        # Use a general search if mood and genre are available but no track seeds
        if not seed_track_ids and mood and genres and spotify_access_token:
            query = f"{mood} {genres[0]}"
            track_info = search_spotify_track(spotify_access_token, query)
            if track_info and track_info.get("id"):
                seed_track_ids.append(track_info["id"])

        spotify_recs = []
        if spotify_access_token and (seed_track_ids or seed_artist_ids or seed_genres):
            # Note: get_spotify_recommendations takes seed_tracks, need to adapt if using artist/genre seeds
            spotify_recs = get_spotify_recommendations(spotify_access_token, seed_track_ids, limit=5) # Synchronous call

        if spotify_recs:
            response = f"🎵 <b>Spotify Recommendations for a {mood} mood:</b>\n\n"
            for i, track in enumerate(spotify_recs, 1):
                artists_text = ", ".join(a["name"] for a in track.get("artists", []))
                album = track.get("album", {}).get("name", "")
                spotify_url = track.get("external_urls", {}).get("spotify")
                response += f"{i}. <b>{track['name']}</b> by {artists_text}"
                if album: response += f" (<i>{album}</i>)"
                if spotify_url: response += f" - <a href='{spotify_url}'>Listen on Spotify</a>"
                response += "\n"
            response += "\n💡 <i>Like any of these? Ask me to download them by name!</i>"
            await status_msg.edit_text(response, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            return

        # Fallback to YouTube Search if Spotify recs fail or not enough seeds
        await status_msg.edit_text("Couldn't get specific Spotify recommendations, searching YouTube instead...")
        yt_query = sanitize_input(f"{mood} music {' '.join(genres[:1])} {'inspired by ' + artists[0] if artists else ''}".strip())
        yt_results = search_youtube(yt_query, max_results=5) # Cached

        if yt_results:
            response_text = f"🎵 <b>YouTube Music Suggestions for a {mood} mood:</b>\n\n"
            keyboard_buttons = []
            for i, result in enumerate(yt_results[:5], 1):
                duration_str = ""
                if result.get('duration') and isinstance(result['duration'], (int,float)) and result['duration'] > 0:
                    mins, secs = divmod(int(result['duration']), 60)
                    duration_str = f" [{mins}:{secs:02d}]"
                
                title_short = result['title'][:40] + "..." if len(result['title']) > 40 else result['title']
                response_text += f"{i}. <b>{result['title']}</b> by {result.get('uploader', 'N/A')}{duration_str}\n"
                # Video ID already validated in search_youtube
                keyboard_buttons.append([InlineKeyboardButton(f"⬇️ {title_short}", callback_data=f"download_{result['id']}")])
            
            await status_msg.edit_text(response_text, parse_mode=ParseMode.HTML, reply_markup=InlineKeyboardMarkup(keyboard_buttons))
        else:
            await status_msg.edit_text(f"Sorry, I couldn't find specific recommendations for '{mood}' right now. Try a broader search or a different mood!")
            await provide_generic_recommendations(update, mood if mood else "happy") # Fallback generic
    
    except Exception as e:
        logger.error(f"Error in recommend_music for user {user_id}: {e}", exc_info=True)
        await status_msg.edit_text("I encountered an issue getting recommendations. Please try again in a bit.")

async def provide_generic_recommendations(update: Update, mood: str) -> None:
    """Provide generic recommendations when other methods fail."""
    # Simplified list
    mood_recommendations = {
        "happy": ["Happy - Pharrell Williams", "Walking on Sunshine - Katrina & The Waves"],
        "sad": ["Someone Like You - Adele", "Hallelujah - Leonard Cohen (any version)"],
        "energetic": ["Don't Stop Me Now - Queen", "Can't Stop the Feeling - Justin Timberlake"],
        "relaxed": ["Weightless - Marconi Union", "Clair de Lune - Debussy"],
        "focused": ["The Four Seasons - Vivaldi", "Time - Hans Zimmer (Inception OST)"],
        "nostalgic": ["Bohemian Rhapsody - Queen", "Yesterday - The Beatles"],
        "any": ["Three Little Birds - Bob Marley", "Here Comes The Sun - The Beatles"]
    }
    recommendations = mood_recommendations.get(mood.lower(), mood_recommendations["any"])
    response = f"🎵 <b>Some general {mood} music ideas:</b>\n\n"
    for i, track in enumerate(recommendations, 1):
        response += f"{i}. {track}\n"
    response += "\n💡 <i>You can ask me to search for these songs or provide a YouTube link to download them!</i>"
    await update.message.reply_text(response, parse_mode=ParseMode.HTML)


async def set_mood(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start conversation to set user's mood. Part of MOOD_CONV_HANDLER."""
    keyboard = [
        [InlineKeyboardButton("Happy 😊", callback_data="mood_happy"), InlineKeyboardButton("Sad 😢", callback_data="mood_sad")],
        [InlineKeyboardButton("Energetic 💪", callback_data="mood_energetic"), InlineKeyboardButton("Relaxed 😌", callback_data="mood_relaxed")],
        [InlineKeyboardButton("Focused 🧠", callback_data="mood_focused"), InlineKeyboardButton("Nostalgic 🕰️", callback_data="mood_nostalgic")],
        [InlineKeyboardButton("Other/Unsure", callback_data="mood_any"), InlineKeyboardButton("Cancel", callback_data="cancel_mood_set")]
    ]
    await update.message.reply_text(
        "How are you feeling today? This helps me tailor music suggestions.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return MOOD

async def mood_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    """Handles button presses for mood selection, part of MOOD_CONV_HANDLER or general callbacks."""
    query = update.callback_query
    await query.answer() # Acknowledge callback
    data = query.data
    user_id = query.from_user.id

    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})

    if data.startswith("mood_"):
        mood = data.split("_", 1)[1]
        user_contexts[user_id]["mood"] = mood
        logger.info(f"User {user_id} set mood to: {mood}")
        
        # After setting mood, ask for genre preference or offer recommendations
        keyboard_pref = [
            [InlineKeyboardButton("Pop", callback_data="pref_pop"), InlineKeyboardButton("Rock", callback_data="pref_rock")],
            [InlineKeyboardButton("Hip-Hop", callback_data="pref_hiphop"), InlineKeyboardButton("Electronic", callback_data="pref_electronic")],
            [InlineKeyboardButton("Classical", callback_data="pref_classical"), InlineKeyboardButton("Jazz", callback_data="pref_jazz")],
            [InlineKeyboardButton("No Preference / Skip", callback_data="pref_skip")],
            [InlineKeyboardButton("Get Recs Now 👍", callback_data="action_recommend_now")]
        ]
        await query.edit_message_text(
            f"Got it! You're feeling {mood}. 🎶\n\nAny specific music genre you're in the mood for, or shall I surprise you?",
            reply_markup=InlineKeyboardMarkup(keyboard_pref)
        )
        return PREFERENCE # Next state in mood conversation

    elif data.startswith("pref_"):
        preference = data.split("_", 1)[1]
        if preference != "skip":
            current_prefs = user_contexts[user_id].get("preferences", [])
            if preference not in current_prefs:
                current_prefs.append(preference)
            user_contexts[user_id]["preferences"] = list(set(current_prefs))[:3] # Max 3 prefs
            logger.info(f"User {user_id} added preference: {preference}")
            await query.edit_message_text(f"Preference '{preference}' noted! Ready for recommendations?")
        else: # Skipped preference
            await query.edit_message_text("No problem! I'll use your general profile for recommendations.")
        
        # Automatically trigger recommendation after preference or skip
        await query.message.reply_text("Let me find some music for you...") # New message
        await recommend_music(update, context) # Call recommend_music directly
        return ConversationHandler.END # End mood conversation

    elif data == "action_recommend_now":
        await query.edit_message_text("Alright, let's find some music based on your current mood!")
        await recommend_music(update, context) # Call recommend_music directly
        return ConversationHandler.END # End mood conversation

    elif data == "cancel_mood_set":
        await query.edit_message_text("Mood setting cancelled. Feel free to tell me anytime!")
        return ConversationHandler.END

    # Fallback if data not handled by mood_conv states but is a mood prefix.
    # This should not be reached if ConversationHandler states are correct.
    logger.warning(f"Unhandled mood-related callback: {data} for user {user_id}")
    await query.edit_message_text("Something unexpected happened. Please try again.")
    return ConversationHandler.END


async def general_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles general button callbacks not part of a specific conversation (like search results downloads)."""
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = query.from_user.id

    logger.debug(f"General button handler: Received '{data}' from user {user_id}")

    if data.startswith("download_"):
        video_id = data.split("_", 1)[1]
        if not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
            logger.error(f"Invalid YouTube video ID in callback: {video_id}")
            await query.edit_message_text("❌ Error: Invalid video ID. Cannot download.")
            return
        
        video_url = f"https://www.youtube.com/watch?v={video_id}"
        # Call the unified download logic.
        # Pass 'update' because download_music_handler_logic expects an Update object where it can access callback_query
        await download_music_handler_logic(update, context, video_url=video_url) 
        # The original message with buttons will be edited by download_music_handler_logic
        return

    elif data.startswith("show_options_"): # From enhanced_handle_message "Show me options"
        search_query_ai = data.split("show_options_", 1)[1]
        status_msg = await query.edit_message_text(f"🔍 Fetching search options for '{search_query_ai}'...")
        results = search_youtube(search_query_ai) # Sanitized and cached

        if not results:
            await status_msg.edit_text(f"Sorry, I couldn't find any YouTube results for '{search_query_ai}'.")
            return

        keyboard = []
        response_text = f"🔎 YouTube Search Results for '{search_query_ai}':\n\n"
        for i, res_item in enumerate(results[:5], 1): # Top 5
            title_short = res_item['title'][:40] + "..." if len(res_item['title']) > 40 else res_item['title']
            duration_str = ""
            if res_item.get('duration') and isinstance(res_item['duration'], (int, float)) and res_item['duration'] > 0:
                 mins, secs = divmod(int(res_item['duration']), 60)
                 duration_str = f" [{mins}:{secs:02d}]"
            response_text += f"{i}. <b>{res_item['title']}</b> by {res_item.get('uploader', 'N/A')}{duration_str}\n"
            # Video ID already validated in search_youtube
            keyboard.append([InlineKeyboardButton(f"⬇️ {title_short}", callback_data=f"download_{res_item['id']}")])
        
        keyboard.append([InlineKeyboardButton("Cancel", callback_data="cancel_search_results")])
        await status_msg.edit_text(response_text, reply_markup=InlineKeyboardMarkup(keyboard), parse_mode=ParseMode.HTML)
        return

    elif data == "cancel_search_results" or data == "cancel_search": # General cancel for search results displays
        await query.edit_message_text("❌ Search or action cancelled.")
        return
        
    # If the callback is part of the spotify_conv_handler or mood_conv_handler, it should be handled there.
    # This check helps route stray callbacks or identify issues.
    # E.g., "cancel_spotify_link" and mood related buttons should be caught by their respective ConversationHandlers.
    if data == "cancel_spotify_link" or data.startswith("mood_") or data.startswith("pref_") or data.startswith("action_"):
        logger.warning(f"Callback '{data}' was handled by general_button_handler but should be part of a ConversationHandler. Check setup.")
        # It might mean the conversation timed out, and this is a fallback.
        # Try to gracefully end, or inform the user.
        await query.edit_message_text("This action seems to have expired. Please try starting the command again (e.g., /mood or /link_spotify).")
        return

    logger.warning(f"Unhandled general callback: {data}")
    # await query.edit_message_text("Sorry, I'm not sure what to do with that button press anymore.")


async def enhanced_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enhanced message handler for text messages (not commands)."""
    if not update.message or not update.message.text: return # Ignore empty messages

    user_id = update.effective_user.id
    chat_id = update.effective_chat.id
    text = update.message.text # Original text for some checks
    sanitized_text = sanitize_input(text) # Sanitized for AI and general use

    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})

    # 1. Check for YouTube URL for direct download
    if is_valid_youtube_url(text): # Check original text for URL
        logger.info(f"YouTube URL detected in message from user {user_id}. Initiating download.")
        # Pass `update` to use its context (like `args`) and message object.
        await download_music_handler_logic(update, context, video_url=text.strip())
        return

    # 2. Quick regex check for simple download/play commands
    detected_song_query_regex = detect_music_in_message(text)
    if detected_song_query_regex:
        logger.info(f"Regex detected music query: '{detected_song_query_regex}' from user {user_id}.")
        # Offer to download or show options
        status_msg = await update.message.reply_text(f"You might be looking for '{detected_song_query_regex}'. Let me check YouTube...")
        results = search_youtube(detected_song_query_regex, max_results=1)
        if results and results[0].get("id"):
            video_id = results[0]["id"] # Validated in search_youtube
            title = results[0]['title']
            uploader = results[0]['uploader']
            keyboard = [
                [InlineKeyboardButton(f"✅ Yes, download '{title[:30]}...'", callback_data=f"download_{video_id}")],
                [InlineKeyboardButton("👀 Show more options", callback_data=f"show_options_{detected_song_query_regex}")],
                [InlineKeyboardButton("❌ No, that's not it", callback_data="cancel_search")]
            ]
            await status_msg.edit_text(
                f"I found '{title}' by {uploader} on YouTube. Would you like to download this one?",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        else:
            await status_msg.edit_text(f"Sorry, I couldn't find '{detected_song_query_regex}' on YouTube right away. You can try /search or rephrase.")
        return

    # 3. If not a URL or simple regex match, use AI for deeper understanding (music request, lyrics, mood)
    # Set a typing action to indicate processing
    await context.bot.send_chat_action(chat_id=chat_id, action="typing")

    # AI Check: Is it a music request?
    ai_music_analysis = await is_music_request(user_id, sanitized_text)
    if ai_music_analysis.get("is_music_request") and ai_music_analysis.get("song_query"):
        song_query_ai = ai_music_analysis["song_query"]
        logger.info(f"AI detected music query: '{song_query_ai}' from user {user_id}.")
        status_msg = await update.message.reply_text(f"AI thinks you're looking for '{song_query_ai}'. Checking YouTube...")
        results = search_youtube(song_query_ai, max_results=1) # search_youtube sanitizes its input
        if results and results[0].get("id"):
            video_id = results[0]["id"]
            title = results[0]['title']
            uploader = results[0]['uploader']
            keyboard = [
                [InlineKeyboardButton(f"✅ Yes, download '{title[:30]}...'", callback_data=f"download_{video_id}")],
                [InlineKeyboardButton("👀 Show more options", callback_data=f"show_options_{song_query_ai}")],
                [InlineKeyboardButton("❌ No, that's not it", callback_data="cancel_search")]
            ]
            await status_msg.edit_text(
                f"AI found '{title}' by {uploader} on YouTube. Should I download this for you?",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
        else:
            await status_msg.edit_text(f"AI thought you wanted '{song_query_ai}', but I couldn't find it on YouTube. Try /search or rephrasing.")
        return
        
    # AI Check: Is it a lyrics request? (Simplified, main /lyrics is more robust)
    # This is a very basic check, complex lyric requests might need more nuance.
    if any(keyword in sanitized_text.lower() for keyword in ["lyrics of", "words to", "what are the lyrics for"]):
        # Try to extract song title for lyrics
        # This is a naive extraction, /lyrics command is better
        potential_song_for_lyrics = re.sub(r"(lyrics of|words to|what are the lyrics for)\s*", "", sanitized_text, flags=re.I).strip()
        if len(potential_song_for_lyrics) > 3 : # If something was extracted
             logger.info(f"Potential lyrics request detected in chat: '{potential_song_for_lyrics}' from user {user_id}.")
             context.args = [potential_song_for_lyrics] # Hack to use get_lyrics_command structure
             await get_lyrics_command(update, context)
             return

    # 4. If none of the above, it's general conversation
    logger.info(f"Passing message to general AI chat for user {user_id}: '{sanitized_text[:50]}...'")
    chat_response = await generate_chat_response(user_id, sanitized_text)
    await update.message.reply_text(chat_response)


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    if user_id in user_contexts:
        user_contexts[user_id]["conversation_history"] = []
        user_contexts[user_id]["mood"] = None # Also clear mood/prefs if desired
        user_contexts[user_id]["preferences"] = []
        await update.message.reply_text("✅ Your conversation history, mood, and preferences with me have been cleared.")
    else:
        await update.message.reply_text("You don't have any saved conversation history with me yet.")

async def cancel_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Generic cancel handler for ConversationHandlers."""
    # Check if called from a callback query or a command/message
    if update.callback_query:
        await update.callback_query.answer()
        await update.callback_query.edit_message_text("Okay, action cancelled. What's next?")
    else:
        await update.message.reply_text("Okay, action cancelled. Feel free to start over or try something else!")
    return ConversationHandler.END

# ==================== ERROR HANDLING ====================
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates."""
    logger.error(f"Update {update} caused error {context.error}", exc_info=context.error)
    
    # Try to inform the user, if possible
    if isinstance(update, Update) and update.effective_message:
        try:
            await update.effective_message.reply_text("😔 Oops! Something went wrong on my end. I've logged the issue. Please try again later.")
        except (TimedOut, NetworkError):
            logger.error("Failed to send error message to user due to network issue.")
        except Exception as e:
            logger.error(f"Failed to send error message to user: {e}")
    elif update and hasattr(update, 'effective_chat') and update.effective_chat: # For callback queries without a message
         try:
            await context.bot.send_message(chat_id=update.effective_chat.id, text="😔 Oops! Something went wrong. Please try again.")
         except Exception as e:
            logger.error(f"Failed to send fallback error message via chat_id: {e}")


# ==================== CLEANUP FUNCTIONS ====================
def cleanup_downloads_atexit() -> None:
    """Clean up temporary download files on exit."""
    logger.info("Performing cleanup of download directory...")
    deleted_count = 0
    if os.path.exists(DOWNLOAD_DIR):
        for filename in os.listdir(DOWNLOAD_DIR):
            file_path = os.path.join(DOWNLOAD_DIR, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                    deleted_count +=1
            except Exception as e:
                logger.error(f'Failed to delete {file_path}. Reason: {e}')
    logger.info(f"Cleanup finished. Deleted {deleted_count} files.")

def signal_exit_handler(sig, frame) -> None:
    """Handle termination signals for graceful shutdown."""
    logger.info(f"Received signal {sig}, initiating graceful shutdown...")
    cleanup_downloads_atexit() # Ensure cleanup runs
    sys.exit(0)

# ==================== MAIN FUNCTION ====================
def main() -> None:
    """Start the bot."""
    # Environment variable validation
    required_env = ["TELEGRAM_TOKEN", "OPENAI_API_KEY", "SPOTIFY_CLIENT_ID", 
                    "SPOTIFY_CLIENT_SECRET", "SPOTIFY_REDIRECT_URI", "GENIUS_ACCESS_TOKEN"]
    missing_vars = [var for var in required_env if not os.getenv(var)]
    if missing_vars:
        logger.critical(f"FATAL: Missing critical environment variables: {', '.join(missing_vars)}. Bot cannot start.")
        sys.exit(1)
    
    if SPOTIFY_REDIRECT_URI == "https://your-callback-url.com":
        logger.warning("SPOTIFY_REDIRECT_URI is set to its default placeholder. "
                       "Spotify OAuth /link_spotify may not work as expected until this is properly configured in your .env file.")

    # Application setup with custom http client settings for potentially longer timeouts
    # ptb_client = httpx.AsyncClient(timeout=httpx.Timeout(60.0)) # General longer timeout
    application = (
        Application.builder()
        .token(TOKEN)
        # .read_timeout(30) # Read timeout for bot.get_updates
        # .write_timeout(30) # Write timeout for bot.send_message etc.
        # .connect_timeout(30)
        # .pool_timeout(30) # For a pool of connections if used by httpx
        # .http_version("1.1") # Default, but can be "2" if server supports and beneficial
        # .request_client(ptb_client) # Inject custom client - careful with this for advanced use.
        .build()
    )

    # Command Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("download", download_music_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("autodownload", auto_download_command))
    application.add_handler(CommandHandler("lyrics", get_lyrics_command))
    application.add_handler(CommandHandler("recommend", recommend_music))
    application.add_handler(CommandHandler("clear", clear_history))

    # Spotify Conversation Handler
    spotify_conv_handler = ConversationHandler(
        entry_points=[
            CommandHandler("link_spotify", link_spotify),
            CommandHandler("spotify_code", spotify_code_command_entry) # Allows /spotify_code <token> directly
        ],
        states={
            SPOTIFY_CODE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, spotify_code_handler),
                CommandHandler("spotify_code", spotify_code_handler), # Handles /spotify_code if user types it while in state
            ]
        },
        fallbacks=[
            CallbackQueryHandler(cancel_spotify_link, pattern="^cancel_spotify_link$"),
            CommandHandler("cancel", cancel_conversation)
        ],
        conversation_timeout=timedelta(minutes=5).total_seconds() # Timeout for conversation
    )
    application.add_handler(spotify_conv_handler)

    # Mood Conversation Handler
    mood_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("mood", set_mood)],
        states={
            MOOD: [CallbackQueryHandler(mood_button_handler, pattern="^mood_")],
            PREFERENCE: [CallbackQueryHandler(mood_button_handler, pattern="^(pref_|action_recommend_now)")]
            # ACTION state not explicitly used here as PREFERENCE handler leads to recommendation or end.
        },
        fallbacks=[
            CallbackQueryHandler(mood_button_handler, pattern="^cancel_mood_set$"), # Specific cancel for mood
            CommandHandler("cancel", cancel_conversation) # Generic cancel
        ],
        conversation_timeout=timedelta(minutes=3).total_seconds()
    )
    application.add_handler(mood_conv_handler)

    # General CallbackQueryHandler (must be after ConversationHandlers to give them priority for their patterns)
    # This handles callbacks like download buttons from search results.
    application.add_handler(CallbackQueryHandler(general_button_handler))
    
    # Message Handler for general text (must be one of the last handlers)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, enhanced_handle_message))
    
    # Error Handler (must be registered last)
    application.add_error_handler(error_handler)

    # Register cleanup on exit
    atexit.register(cleanup_downloads_atexit)
    signal.signal(signal.SIGINT, signal_exit_handler)
    signal.signal(signal.SIGTERM, signal_exit_handler)
    
    logger.info("MelodyMind Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()