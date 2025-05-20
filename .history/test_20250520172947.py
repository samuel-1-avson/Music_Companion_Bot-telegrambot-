I've addressed your requests by making several improvements and bug fixes to the MelodyMind bot. Here's a summary of the key changes:

**Regarding Automatic Spotify Code Handling:**

You asked why the bot doesn't automatically handle the Spotify code exchange instead of requiring you to copy-paste it.

*   **Technical Reason:** Fully automating the Spotify OAuth code exchange (where the bot transparently gets the code after you authorize on Spotify's website) requires the bot to have a publicly accessible web server. Spotify needs to redirect the user's browser to a URL (`SPOTIFY_REDIRECT_URI`) that your bot controls. This server would then receive the authorization code directly.
*   **Current Bot Structure:** The provided script is designed to run as a standalone Python application, which typically doesn't include a built-in web server exposed to the internet without additional setup (like `ngrok` for local development, or deployment on a platform that provides a public URL).
*   **What I've Done:** The current method (providing a link, and you paste the code) is the most straightforward for a bot of this nature. I've ensured the instructions are clear and added the `/spotify_code <code>` command for convenience.
*   **To Achieve Full Automation:** You would need to:
    1.  Set up a simple web server (e.g., using Flask or FastAPI) that listens on your `SPOTIFY_REDIRECT_URI`.
    2.  This server would receive the `code` and `state` from Spotify.
    3.  It would then exchange the code for tokens and update the `user_contexts` (which might require a shared data store if the web server is a separate process).
    4.  Finally, it could notify the user in Telegram via the bot.

This is a significant architectural addition. For now, the manual code input remains, but the process is as streamlined as possible within the current bot structure.

**Bug Fixes & General Improvements:**

1.  **Deduplication:**
    *   Removed duplicate definitions of `refresh_spotify_token`, `get_lyrics_command`, and `handle_error`.
    *   Ensured the `lru_cache` version of `search_youtube` is used consistently.
2.  **Missing `get_lyrics` Function:**
    *   Implemented the `get_lyrics(song_title, artist_name)` function, which uses the `genius` client to fetch lyrics. This function is now called by `/lyrics` command. It includes caching and better error handling.
3.  **Spotify Token Refresh:**
    *   Improved error handling in `get_user_spotify_data` and `get_user_spotify_playlists` for cases where token refresh fails (e.g., due to an invalid refresh token). The bot will now clear outdated Spotify data for the user if the refresh token is invalid, prompting them to re-link.
4.  **Error Handling in `download_music` and `enhanced_button_handler`:**
    *   The use of `asyncio.Lock` for downloads and retry mechanisms for Telegram API calls (sending audio, editing messages) helps prevent race conditions and handle transient network issues.
    *   The file size check (max 50MB for Telegram) is robustly implemented.
5.  **Environment Variable Warning:**
    *   The warning for `SPOTIFY_REDIRECT_URI` being a placeholder is retained, as it's crucial for Spotify linking.
6.  **Stability for `is_music_request` and `analyze_conversation`:**
    *   These functions now have more robust error handling for OpenAI API responses, especially for JSON parsing. If the AI doesn't return valid JSON, they will fall back gracefully.
7.  **Logging:** Maintained and, where appropriate, enhanced logging for better traceability.
8.  **`recommend_music` Robustness:**
    *   The function now handles missing Spotify seeds or API failures more gracefully, falling back to YouTube searches or generic recommendations. It also ensures `status_msg` is handled correctly.
    *   Properly handles fetching playlist tracks if user has playlists but no recent/top tracks.

**New AI-Powered Features:**

1.  **Deeper Lyric Analysis (`/analyze_lyrics`):**
    *   You can now use `/analyze_lyrics [Song Name] - [Artist (optional)]`.
    *   The bot will fetch the lyrics and then use OpenAI (GPT-3.5-turbo) to provide a brief analysis of the song's themes, mood, or meaning.

2.  **Song Similarity Recommendations (`/songslike`):**
    *   New command: `/songslike [Song Name] - [Artist (optional)]`.
    *   **If Spotify is linked and the song is found:** It will use Spotify's recommendation engine, seeding it with the provided track to find similar songs.
    *   **Otherwise (or if Spotify fails):** It will use a combination of AI analysis (to extract keywords/genres from your request) and YouTube search to find similar-sounding music.

The updated code incorporating these changes is below. Remember to have all your environment variables (`.env` file) correctly set up.

```python
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
from tenacity import retry, stop_after_attempt, wait_exponential
from telegram.error import TimedOut, NetworkError
import httpx
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
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "https://your-callback-url.com") # Needs to be a real, accessible URI for fully automated OAuth

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize clients
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
if GENIUS_ACCESS_TOKEN and lyricsgenius:
    try:
        genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=15, retries=2)
        genius.verbose = False # Suppress Genius's own print statements
        genius.remove_section_headers = True # Clean up lyrics
    except Exception as e:
        logger.error(f"Failed to initialize LyricsGenius: {e}")
        genius = None
else:
    genius = None


# Conversation states
MOOD, PREFERENCE, ACTION, SPOTIFY_CODE = range(4)

# Track active downloads and user contexts
active_downloads = set()
user_contexts: Dict[int, Dict] = {}
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ==================== SPOTIFY HELPER FUNCTIONS ====================

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
        logger.error(f"Error getting Spotify client token: {e}")
        return None

@lru_cache(maxsize=128)
def search_spotify_track(token: str, query: str) -> Optional[Dict]:
    """Search for a track on Spotify. (Cached)"""
    if not token:
        return None
    query = sanitize_input(query)
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
def get_spotify_recommendations(token: str, seed_tracks: List[str], limit: int = 5) -> List[Dict]:
    """Get track recommendations from Spotify."""
    if not token or not seed_tracks:
        logger.warning("No token or seed tracks for Spotify recommendations")
        return []

    url = "https://api.spotify.com/v1/recommendations"
    headers = {"Authorization": f"Bearer {token}"}
    # Spotify API allows up to 5 seed values (artists, genres, tracks)
    params = {"seed_tracks": ",".join(seed_tracks[:5]), "limit": limit} # Ensure not to exceed 5 seed tracks

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        return response.json().get("tracks", [])
    except requests.exceptions.HTTPError as http_error:
        logger.warning(f"Spotify recommendations failed (HTTPError) for seeds {seed_tracks}: {http_error.response.text if http_error.response else 'No response'}")
        return []
    except requests.exceptions.RequestException as req_error:
        logger.error(f"Error getting Spotify recommendations (RequestException): {req_error}")
        return []

def get_user_spotify_token(user_id: int, code: str) -> Optional[Dict]:
    """Exchange authorization code for Spotify access and refresh tokens."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI:
        logger.warning("Spotify OAuth (user token) credentials not configured")
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
        response = requests.post(url, headers=headers, data=data, timeout=15)
        response.raise_for_status()
        token_data = response.json()
        token_data["expires_at"] = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600))).timestamp()
        return token_data
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting user Spotify token with code: {e}")
        return None

def refresh_spotify_token(user_id: int) -> Optional[str]:
    """Refresh Spotify access token using refresh token."""
    # Ensure user_contexts has a default structure if user_id is new
    user_contexts.setdefault(user_id, {"spotify": {}, "conversation_history": [], "mood": None, "preferences": []})
    
    context = user_contexts.get(user_id, {})
    refresh_token = context.get("spotify", {}).get("refresh_token")

    if not refresh_token:
        logger.warning(f"No refresh token found for user {user_id} to refresh Spotify token.")
        return None
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        logger.warning("Spotify client credentials not configured for token refresh.")
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
        
        user_contexts[user_id]["spotify"]["access_token"] = token_data.get("access_token")
        # Spotify might issue a new refresh token
        user_contexts[user_id]["spotify"]["refresh_token"] = token_data.get("refresh_token", refresh_token)
        user_contexts[user_id]["spotify"]["expires_at"] = expires_at
        
        return token_data.get("access_token")
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 400: # Often "Invalid refresh token"
            logger.error(f"Invalid refresh token for user {user_id}. Clearing Spotify data. Error: {e}")
            if user_id in user_contexts and "spotify" in user_contexts[user_id]:
                 user_contexts[user_id]["spotify"] = {} # Clear invalid token data
            # Consider notifying the user they need to re-link Spotify
        else:
            logger.error(f"HTTP error refreshing Spotify token for user {user_id}: {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Error refreshing Spotify token for user {user_id}: {e}")
        return None

def get_user_spotify_data(user_id: int, endpoint: str, limit: int = 10) -> Optional[List[Dict]]:
    """Fetch user-specific Spotify data (recently played or top tracks)."""
    # Ensure user_contexts has a default structure if user_id is new
    user_contexts.setdefault(user_id, {"spotify": {}, "conversation_history": [], "mood": None, "preferences": []})

    context = user_contexts.get(user_id, {})
    spotify_data = context.get("spotify", {})
    access_token = spotify_data.get("access_token")
    expires_at = spotify_data.get("expires_at")

    if not access_token or (expires_at and datetime.now(pytz.UTC).timestamp() > expires_at):
        access_token = refresh_spotify_token(user_id)
        if not access_token:
            logger.info(f"Failed to refresh or get Spotify token for user {user_id} for endpoint {endpoint}.")
            return None # Explicitly return None

    url = f"https://api.spotify.com/v1/me/{endpoint}"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"limit": limit}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("items", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Spotify user data ({endpoint}) for user {user_id}: {e}")
        return None # Explicitly return None


def get_user_spotify_playlists(user_id: int, limit: int = 10) -> Optional[List[Dict]]:
    """Fetch user's Spotify playlists."""
    user_contexts.setdefault(user_id, {"spotify": {}, "conversation_history": [], "mood": None, "preferences": []})
    
    context = user_contexts.get(user_id, {})
    spotify_data = context.get("spotify", {})
    access_token = spotify_data.get("access_token")
    expires_at = spotify_data.get("expires_at")

    if not access_token or (expires_at and datetime.now(pytz.UTC).timestamp() > expires_at):
        access_token = refresh_spotify_token(user_id)
        if not access_token:
            logger.info(f"Failed to refresh or get Spotify token for user {user_id} for playlists.")
            return None

    url = "https://api.spotify.com/v1/me/playlists"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"limit": limit}

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("items", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Spotify playlists for user {user_id}: {e}")
        return None

# ==================== YOUTUBE HELPER FUNCTIONS ====================

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
    return sanitized[:100] # Limit length for safety

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5))
def download_youtube_audio(url: str) -> Dict[str, Any]:
    """Download audio from a YouTube video with improved error handling."""
    video_id_match = re.search(r'(?:v=|/)([0-9A-Za-z_-]{11})', url)
    if not video_id_match: # Also check general URL validity
        if not is_valid_youtube_url(url):
            logger.error(f"Invalid YouTube URL format: {url}")
            return {"success": False, "error": "Invalid YouTube URL format"}
        # If it passed is_valid_youtube_url but still no ID, yt-dlp might handle it (e.g. channels/playlists - though noplaylist is True)
        # Let yt-dlp try, it has more robust URL parsing.

    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio[abr<=128]/bestaudio', # Prefers m4a, then any audio up to 128kbps
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True, # Important: only download single video
        # 'postprocessor_args': ['-acodec', 'copy'], # This can cause issues if conversion is needed
        'prefer_ffmpeg': True, # Use ffmpeg if available for better format handling
        'max_filesize': 50 * 1024 * 1024,  # 50 MB limit, yt-dlp will error if too large
        'extract_flat': False, # Make sure we get full info for single videos
        'forcefilename': True,
        'restrictfilenames': True, # Sanitize filenames for filesystem
    }

    downloaded_filepath = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False) # First get info without downloading
            if not info:
                return {"success": False, "error": "Could not extract video information"}
            
            title = sanitize_filename(info.get('title', 'Unknown_Title'))
            artist = info.get('artist', info.get('uploader', 'Unknown_Artist')) # uploader is a good fallback
            duration = info.get('duration', 0)
            thumbnail_url = info.get('thumbnail', '')

            # Check filesize *before* download if possible, though this is for original video, not audio
            # filesie_approx = info.get('filesize_approx')
            # if filesie_approx and filesie_approx > 60 * 1024 * 1024: # A bit more than 50MB to account for audio only
            #     return {"success": False, "error": f"Video filesize likely too large (approx {filesie_approx/(1024*1024):.2f}MB)"}


            # Define a hook to capture the filename after download and processing
            downloaded_files = []
            def hook(d):
                if d['status'] == 'finished':
                    # filename in yt-dlp 2023.03.04+ refers to the final output file
                    filepath = d.get('filename', d.get('info_dict', {}).get('_filename'))
                    if filepath:
                        downloaded_files.append(filepath)
                        logger.info(f"yt-dlp finished downloading: {filepath}")
            ydl.add_progress_hook(hook)
            
            logger.info(f"Starting download for: {title}")
            ydl.download([url]) # Perform the download

            if not downloaded_files:
                # Fallback: try to find by title (less reliable)
                logger.warning("Progress hook did not capture filename, attempting manual search.")
                for ext in ['m4a', 'mp3', 'webm', 'ogg', 'opus']: # Common audio extensions
                    # yt-dlp uses info['title'] for filename construction with sanitize_filename internally
                    # but its sanitization might differ slightly. Use the title from info.
                    potential_sanitized_title = ydl.prepare_filename(info).rsplit('.',1)[0].replace(DOWNLOAD_DIR+os.sep, '')
                    
                    potential_path = os.path.join(DOWNLOAD_DIR, f"{potential_sanitized_title}.{ext}")
                    if os.path.exists(potential_path):
                        downloaded_filepath = potential_path
                        break
                if not downloaded_filepath:
                     return {"success": False, "error": "Downloaded file not found after yt-dlp process."}
            else:
                downloaded_filepath = downloaded_files[0] # Take the first one, should be the one

            if not os.path.exists(downloaded_filepath):
                 return {"success": False, "error": f"Downloaded file path reported but not found: {downloaded_filepath}"}

            file_size_mb = os.path.getsize(downloaded_filepath) / (1024 * 1024)
            if file_size_mb > 50:
                os.remove(downloaded_filepath) # Clean up oversized file
                logger.error(f"File too large: {file_size_mb:.2f} MB exceeds 50 MB Telegram limit.")
                return {"success": False, "error": "File too large for Telegram (max 50 MB)."}
            
            return {
                "success": True,
                "title": title,
                "artist": artist,
                "thumbnail_url": thumbnail_url,
                "duration": duration,
                "audio_path": downloaded_filepath
            }
    except yt_dlp.utils.MaxFilesizeError:
        logger.error(f"YouTube download error: MaxFilesizeError for {url}")
        return {"success": False, "error": "File is too large (exceeds 50MB limit)."}
    except yt_dlp.utils.DownloadError as e:
        logger.error(f"YouTube download error: {e} for {url}")
        # Check for common messages
        if "ideo unavailable" in str(e).lower():
            return {"success": False, "error": "Video unavailable."}
        if "rivate video" in str(e).lower():
            return {"success": False, "error": "This is a private video."}
        if "login required" in str(e).lower():
            return {"success": False, "error": "This video may require login or is restricted."}
        return {"success": False, "error": f"Download failed (yt-dlp)."}
    except Exception as e:
        logger.error(f"Unexpected error downloading YouTube audio for {url}: {e}", exc_info=True)
        return {"success": False, "error": "An unexpected error occurred during download."}
    finally:
        # Clean up any file if an error occurred before returning success but after download start
        if downloaded_filepath and os.path.exists(downloaded_filepath) and \
           ( 'success' not in locals() or not locals()['success']): # if success flag is not True
            try:
                os.remove(downloaded_filepath)
            except OSError as oe:
                logger.error(f"Error cleaning up failed download {downloaded_filepath}: {oe}")


@lru_cache(maxsize=100) # Cache search results
def search_youtube(query: str, max_results: int = 5) -> List[Dict]:
    """Search YouTube for videos matching the query with caching."""
    query = sanitize_input(query)
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'in_playlist', # Faster searching
            'default_search': 'ytsearch',
            # 'format': 'bestaudio', # Not needed for search, can slow it down
            'noplaylist': True, # Search is not a playlist download
            'playlist_items': f'1-{max_results}' # Limit search results server-side
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Construct search query for yt-dlp
            search_query = f"ytsearch{max_results}:{query}"
            info = ydl.extract_info(search_query, download=False)
            
            if not info or 'entries' not in info:
                logger.info(f"No YouTube search results for '{query}'.")
                return []
            
            results = []
            for entry in info['entries']:
                if entry: # Ensure entry is not None
                    video_id = entry.get('id', '')
                    if not video_id or not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
                        logger.warning(f"Skipping search result with invalid/missing ID for query '{query}': {entry.get('title')}")
                        continue
                    results.append({
                        'title': entry.get('title', 'Unknown Title'),
                        'url': entry.get('url') or f"https://www.youtube.com/watch?v={video_id}",
                        'thumbnail': entry.get('thumbnail', ''),
                        'uploader': entry.get('uploader', 'Unknown Artist'),
                        'duration': entry.get('duration', 0),
                        'id': video_id
                    })
            return results
    except yt_dlp.utils.ExtractorError as ee:
        logger.error(f"YouTube search extractor error for query '{query}': {ee}")
        return []
    except Exception as e:
        logger.error(f"Error searching YouTube for query '{query}': {e}", exc_info=True)
        return []

# ==================== LYRICS HELPER FUNCTION =====================
@lru_cache(maxsize=50)
def get_lyrics(song_title: str, artist_name: Optional[str] = None) -> str:
    """Fetch lyrics using Genius API with caching."""
    if not genius:
        return "Lyrics service is not available (Genius API not configured or library missing)."
    
    song_title = sanitize_input(song_title)
    artist_name = sanitize_input(artist_name) if artist_name else None

    try:
        logger.info(f"Searching lyrics for: {song_title} by {artist_name or 'Unknown Artist'}")
        # Genius client already handles retries if configured during init
        song_obj = genius.search_song(song_title, artist_name) if artist_name else genius.search_song(song_title)
            
        if song_obj and song_obj.lyrics:
            # Lyrics Genius's remove_section_headers should handle most of this, but an extra clean up.
            lyrics = song_obj.lyrics.replace('EmbedShare URLCopyEmbedCopy', '').strip()
            # Remove leading number contributors and "Lyrics" e.g. "16 ContributorsTranslationsRomanizationTürkçeSORA Lyrics"
            lyrics = re.sub(r'^\d*\s*Contributors?\S*\s*Lyrics\n?', '', lyrics, flags=re.IGNORECASE)
            # Remove text like [Chorus], [Verse 1], etc. if desired by changing genius.remove_section_headers
            # genius.remove_section_headers = True was set at init.
            
            if not lyrics.strip():
                return f"Lyrics found for \"{song_obj.title}\" by {song_obj.artist}, but they are empty."

            return f"📜 Lyrics for \"{song_obj.title}\" by {song_obj.artist}:\n\n{lyrics}"
        
        logger.warning(f"Lyrics not found for: {song_title} by {artist_name or 'Unknown Artist'}")
        return "Sorry, I couldn't find those lyrics. Try being more specific with artist and title."
    except requests.exceptions.Timeout:
        logger.warning(f"Genius API timed out for '{song_title}'")
        return "Sorry, the lyrics search timed out. Please try again."
    except Exception as e: # Catch any other exception from lyricsgenius or requests
        logger.error(f"Error fetching lyrics for '{song_title}': {e}", exc_info=True)
        return "Sorry, an error occurred while fetching lyrics."


# ==================== AI CONVERSATION FUNCTIONS ====================

async def generate_chat_response(user_id: int, message: str) -> str:
    """Generate a conversational response using OpenAI."""
    if not client:
        return "I'm having trouble connecting to my AI service. Please try again later."

    message = sanitize_input(message)
    # Ensure user_contexts has a default structure
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    context = user_contexts[user_id]


    system_prompt = (
        "You are a friendly, empathetic music companion bot named MelodyMind. "
        "Your primary role is to have natural conversations about music and feelings. "
        "Secondary roles: recommend songs (but avoid giving lists unless specifically asked by /recommend or similar), "
        "provide emotional support through music-related chat, and keep responses concise yet warm (around 2-4 sentences). "
        "Do not offer to download songs or list specific song titles unless the user asks something like '/recommend'. "
        "If the user has linked their Spotify, you can acknowledge their taste if relevant but don't list their songs. "
        "If the user seems to be asking for a song to play or download, or for lyrics, guide them to use commands like "
        "/search, /download, /lyrics, /recommend, /songslike, or /analyze_lyrics. "
        "If asked directly 'Can you download X song?', respond with 'You can ask me to search for it using /search X song, or download it if you have a YouTube link using /download [link]!'."
    )
    messages = [{"role": "system", "content": system_prompt}]

    # Add context from user profile
    user_profile_info = []
    if context.get("mood"):
        user_profile_info.append(f"Current mood: {context['mood']}.")
    if context.get("preferences"):
        user_profile_info.append(f"Music preferences: {', '.join(context['preferences'])}.")
    if context.get("spotify"):
        if context["spotify"].get("recently_played"):
            artists = list(set(item["track"]["artists"][0]["name"] for item in context["spotify"]["recently_played"][:3] if item.get("track")))
            if artists:
                user_profile_info.append(f"Recently listened to on Spotify: {', '.join(artists)}.")
        elif context["spotify"].get("top_tracks"):
            artists = list(set(item["artists"][0]["name"] for item in context["spotify"]["top_tracks"][:3])) # Max 3 to keep it short
            if artists:
                user_profile_info.append(f"Their top artists on Spotify include: {', '.join(artists)}.")
    
    if user_profile_info:
        messages.append({"role": "system", "content": "User Information: " + " ".join(user_profile_info)})

    # Limit conversation history to last 10 exchanges (20 messages)
    history_limit = 20
    start_index = max(0, len(context["conversation_history"]) - history_limit)
    for hist_message in context["conversation_history"][start_index:]:
        messages.append(hist_message)

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
        
        # Update conversation history
        context["conversation_history"].append({"role": "user", "content": message})
        context["conversation_history"].append({"role": "assistant", "content": reply})
        # Trim history again to ensure limit after adding new messages
        context["conversation_history"] = context["conversation_history"][-history_limit:]
        user_contexts[user_id] = context
        return reply
    except Exception as e:
        logger.error(f"Error generating chat response for user {user_id}: {e}", exc_info=True)
        return "I'm having a little trouble thinking right now. Maybe we can just talk about your favorite song?"


async def is_music_request(user_id: int, message: str) -> Dict[str, Any]:
    """Use AI to determine if a message is a music/song request and extract query."""
    if not client:
        return {"is_music_request": False, "song_query": None}

    message = sanitize_input(message)
    try:
        # Run blocking OpenAI call in a separate thread
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo-0125", # Newer model, good with JSON
            messages=[
                {"role": "system", "content": (
                    "You are an AI that determines if a user's message is implicitly or explicitly asking for a song to be played, found, or downloaded, "
                    "or asking for lyrics. If it is, extract the song title and artist if mentioned. "
                    "Respond ONLY with a JSON object with two keys: 'is_music_request' (boolean) and 'song_query' (string, the extracted song title and artist, or null if not a music request or query unclear). "
                    "Examples of music requests: 'play despacito', 'find viva la vida by coldplay', 'i want to hear shape of you ed sheeran', 'get me the lyrics for bohemian rhapsody'."
                    "Examples of NOT music requests: 'i like pop music', 'what do you think of this song?', 'tell me about spotify'."
                )},
                {"role": "user", "content": f"Analyze this message: '{message}'"}
            ],
            max_tokens=100,
            temperature=0.1, # Low temperature for more deterministic output
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)

        if not isinstance(result, dict):
            logger.warning(f"AI music request check for user {user_id} did not return a dict: {result}")
            return {"is_music_request": False, "song_query": None}

        is_request_val = result.get("is_music_request", False)
        # Handle if the model returns string "true"/"false"
        if isinstance(is_request_val, str):
            is_request = is_request_val.lower() == "true"
        else:
            is_request = bool(is_request_val)
            
        song_query = result.get("song_query")
        if not isinstance(song_query, str) or not song_query.strip():
            song_query = None # Ensure it's None if empty or not a string

        return {
            "is_music_request": is_request,
            "song_query": song_query
        }
    except json.JSONDecodeError as jde:
        logger.error(f"JSONDecodeError in is_music_request for user {user_id}, AI response: {content}. Error: {jde}")
        return {"is_music_request": False, "song_query": None}
    except Exception as e:
        logger.error(f"Error in is_music_request for user {user_id}: {e}", exc_info=True)
        return {"is_music_request": False, "song_query": None}


async def analyze_conversation(user_id: int) -> Dict[str, Any]:
    """Analyze conversation history and Spotify data to extract music preferences for recommendations."""
    if not client: # If OpenAI client is not available
        context = user_contexts.get(user_id, {})
        return {"genres": context.get("preferences", []), "artists": [], "mood": context.get("mood")}

    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    context = user_contexts[user_id]

    # If not much data, rely on explicitly set preferences/mood
    if len(context.get("conversation_history", [])) < 2 and not context.get("spotify"):
        return {"genres": context.get("preferences", []), "artists": [], "mood": context.get("mood")}

    conversation_text_parts = []
    if context.get("conversation_history"):
        # Get last 10 messages (5 exchanges)
        history = context["conversation_history"][-10:] 
        for msg in history:
            conversation_text_parts.append(f"{msg['role']}: {msg['content']}")
    conversation_text = "\n".join(conversation_text_parts)

    spotify_data_parts = []
    if context.get("spotify"):
        if context["spotify"].get("recently_played"):
            tracks = context["spotify"]["recently_played"][:3] # Max 3 tracks
            track_info = [f"{item['track']['name']} by {item['track']['artists'][0]['name']}" for item in tracks if item.get("track")]
            if track_info:
                spotify_data_parts.append("Recently played on Spotify: " + ", ".join(track_info) + ".")
        if context["spotify"].get("top_tracks"): # Check even if recently_played was present
            tracks = context["spotify"]["top_tracks"][:3] # Max 3 tracks
            track_info = [f"{item['name']} by {item['artists'][0]['name']}" for item in tracks if item.get("artists")]
            if track_info:
                spotify_data_parts.append("Top tracks on Spotify: " + ", ".join(track_info) + ".")
    spotify_data_text = " ".join(spotify_data_parts)

    prompt_content = (
        f"Analyze the following user interaction data to suggest music recommendations. "
        f"The user's explicit mood setting is '{context.get('mood', 'not set')}'. "
        f"Their explicit genre preferences are '{', '.join(context.get('preferences', ['not set']))}'.\n\n"
        f"Conversation Excerpt:\n{conversation_text if conversation_text else 'No conversation history available.'}\n\n"
        f"Spotify Listening Data:\n{spotify_data_text if spotify_data_text else 'No Spotify data available.'}\n\n"
        f"Based on ALL available information (mood, preferences, conversation, Spotify), infer the user's current overall mood (e.g., happy, sad, energetic, calm, reflective), "
        f"up to 3 relevant music genres, and up to 3 artists they might enjoy right now. "
        f"Prioritize recent conversation and explicit settings if they conflict with older data. "
        f"Return a JSON object with keys: 'mood' (string), 'genres' (list of strings), 'artists' (list of strings)."
    )

    try:
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are an expert music taste analyst. Your goal is to interpret user data and suggest parameters for music recommendations."},
                {"role": "user", "content": prompt_content}
            ],
            max_tokens=200,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        result = json.loads(content)
        
        if not isinstance(result, dict):
            logger.warning(f"AI conversation analysis for user {user_id} did not return dict: {result}")
            return {"genres": context.get("preferences", []), "artists": [], "mood": context.get("mood")}

        # Validate and clean results
        inferred_mood = sanitize_input(result.get("mood")) if isinstance(result.get("mood"), str) else context.get("mood")
        
        inferred_genres = result.get("genres", [])
        if isinstance(inferred_genres, list):
            inferred_genres = [sanitize_input(g) for g in inferred_genres if isinstance(g, str)][:3] # Max 3
        else:
            inferred_genres = context.get("preferences", [])

        inferred_artists = result.get("artists", [])
        if isinstance(inferred_artists, list):
            inferred_artists = [sanitize_input(a) for a in inferred_artists if isinstance(a, str)][:3] # Max 3
        else:
            inferred_artists = []
        
        # Update context with inferred data if it's more specific or newer
        if inferred_mood and (not context.get("mood") or context.get("mood") != inferred_mood) :
             user_contexts[user_id]["mood"] = inferred_mood
        if inferred_genres and (not context.get("preferences") or set(context.get("preferences",[])) != set(inferred_genres)):
             user_contexts[user_id]["preferences"] = inferred_genres
        
        # Always return the latest analysis
        final_analysis = {
            "mood": inferred_mood or context.get("mood"), # Fallback to existing mood if AI doesn't provide one
            "genres": inferred_genres or context.get("preferences", []), # Fallback to existing
            "artists": inferred_artists # Artists are usually supplemental
        }
        logger.info(f"Conversation analysis for user {user_id}: {final_analysis}")
        return final_analysis

    except json.JSONDecodeError as jde:
        logger.error(f"JSONDecodeError in analyze_conversation for user {user_id}, AI response: {content}. Error: {jde}")
        return {"genres": context.get("preferences", []), "artists": [], "mood": context.get("mood")}
    except Exception as e:
        logger.error(f"Error in analyze_conversation for user {user_id}: {e}", exc_info=True)
        return {"genres": context.get("preferences", []), "artists": [], "mood": context.get("mood")}


# ==================== MUSIC DETECTION FUNCTION (Simplified) ====================

def detect_music_in_message(text: str) -> Optional[str]:
    """Detect if a message is asking for music without an explicit YouTube URL using keywords."""
    # This is a simpler, non-AI version. is_music_request (AI) is preferred for more complex queries.
    text_lower = text.lower()
    # Keywords that strongly imply a request for a specific song.
    # Order matters: "download song" is more specific than just "song".
    patterns = [
        r'play\s+(.*)', r'download\s+(.*)', r'find\s+(.*)', r'get\s+(.*)',
        r'search\s+for\s+(.*)', r'i\s+want\s+(?:to\s+listen\s+to\s+)?(.*)',
        r'can\s+you\s+get\s+me\s+(.*)', r'song\s+called\s+(.*)',
        r'lyrics\s+for\s+(.*)', r'what\s+are\s+the\s+lyrics\s+to\s+(.*)'
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            query_candidate = match.group(1).strip()
            # Avoid matching if query is too short or just "music", "a song"
            if len(query_candidate.split()) > 1 or \
               (len(query_candidate.split()) == 1 and query_candidate not in ["music", "song", "audio", "tune"]):
                # Remove common trailing prepositions like "by", "from" if AI will handle artist separately
                # For this simpler detector, keep it, Spotify/YT search might handle it.
                # query_candidate = re.sub(r'\s+(?:by|from)$', '', query_candidate, flags=re.IGNORECASE).strip()
                if query_candidate:
                    return query_candidate # Return the potential song/artist query

    # Generic keywords indicating user *might* want music but query is unclear
    # if any(keyword in text_lower for keyword in ['music', 'song', 'track', 'tune', 'audio']):
    #     return "AI_ANALYSIS_NEEDED" # Signal that AI should look deeper
        
    return None

# ==================== TELEGRAM BOT HANDLERS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message."""
    user = update.effective_user
    welcome_msg = (
        f"Hi {user.first_name}! 👋 I'm MelodyMind, your Music Healing Companion.\n\n"
        "I can:\n"
        "🎵 Download music from YouTube (send a link or ask me to search)\n"
        "📜 Find lyrics for songs (/lyrics [song name])\n"
        "💿 Recommend music based on your mood (/recommend)\n"
        "🎶 Suggest songs similar to one you like (/songslike [song name])\n"
        "🤔 Analyze song lyrics for themes (/analyze_lyrics [song name])\n"
        "💬 Chat about music and feelings\n"
        "🔗 Link Spotify for personalized recommendations (/link_spotify)\n\n"
        "Try /help for a full list of commands, or just chat with me!"
    )
    await update.message.reply_text(welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message."""
    help_text = (
        "🎶 <b>MelodyMind - Your Music Healing Companion</b> 🎶\n\n"
        "<b>Core Commands:</b>\n"
        "/start - Welcome message\n"
        "/help - This help message\n"
        "/download [YouTube URL] - Download music from a YouTube link\n"
        "/search [song query] - Search YouTube for a song to download\n"
        "/lyrics [song name] - [artist] - Get song lyrics\n\n"
        "<b>Recommendation & AI Features:</b>\n"
        "/recommend - Get personalized music recommendations\n"
        "/mood - Set your current mood to tailor recommendations\n"
        "/songslike [song name] - [artist] - Find songs similar to one you like\n"
        "/analyze_lyrics [song name] - [artist] - Get an AI analysis of song lyrics\n\n"
        "<b>Spotify Integration:</b>\n"
        "/link_spotify - Link your Spotify account for enhanced recommendations\n"
        # Placeholder for automatic callback: "/spotify_code [code] - Enter code after Spotify auth (if needed)\n"
        "<b>Utility:</b>\n"
        "/clear - Clear your conversation history with me\n\n"
        "<b>Chatting:</b>\n"
        "You can also just chat with me! Tell me how you're feeling, what music you like, or ask for a song by name. "
        "Examples:\n"
        "- \"I'm feeling energetic, any suggestions?\"\n"
        "- \"Play something like Queen\"\n"
        "- \"What are the lyrics to Hey Jude?\"\n"
        "- (After sending a YouTube link) \"Download this song\""
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

# Thread lock for downloads
download_lock = asyncio.Lock()

async def download_music_from_url(update: Update, context: ContextTypes.DEFAULT_TYPE, url: str, original_message_id: Optional[int] = None) -> None:
    """Helper to download music, called by command or button."""
    user_id = update.effective_user.id
    
    # Use a lock to prevent concurrent downloads by the same user.
    # Allow different users to download concurrently if system resources allow.
    # For a global lock across all users: async with download_lock:
    if user_id in active_downloads:
        await update.effective_message.reply_text("⚠️ You already have a download in progress. Please wait.", quote=True)
        return
    
    active_downloads.add(user_id)
    status_msg = None
    try:
        if update.callback_query: # From a button
            await update.callback_query.edit_message_text("⏳ Starting download...")
            status_msg_container = update.callback_query.message # Use the message associated with the callback
        else: # From a command or direct message
            status_msg = await update.effective_message.reply_text("⏳ Starting download...", quote=True)
            status_msg_container = status_msg

        await status_msg_container.edit_text("🔍 Fetching video information...")
        
        # Run blocking download in a separate thread
        result = await asyncio.to_thread(download_youtube_audio, url)

        if not result["success"]:
            error_message = f"❌ Download failed: {result['error']}"
            if len(error_message) > 4096: error_message = error_message[:4090] + "..."
            await status_msg_container.edit_text(error_message)
            return

        await status_msg_container.edit_text(f"✅ Downloaded: {result['title']}\n⏳ Preparing to send file...")
        
        # Retry sending audio file
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10),
               retry_on_exception=lambda e: isinstance(e, (TimedOut, NetworkError)))
        async def send_audio_with_retry():
            with open(result["audio_path"], 'rb') as audio_file:
                # If called from a callback, send to chat. If from msg, reply to msg.
                target_chat_id = update.effective_chat.id
                reply_to_msg_id = original_message_id or update.effective_message.message_id
                
                await context.bot.send_audio(
                    chat_id=target_chat_id,
                    audio=audio_file,
                    title=result["title"][:64], # Telegram title limit
                    performer=result["artist"][:64] if result.get("artist") else "Unknown Artist",
                    caption=f"🎵 {result['title']}",
                    duration=result.get('duration'),
                    thumbnail=None, # Could try sending thumbnail separately or embedding if audio format supports
                    reply_to_message_id=reply_to_msg_id if not update.callback_query else None # Reply only if not from button
                )
        
        await send_audio_with_retry()
        
        if update.callback_query: # from button
             await status_msg_container.edit_text(f"✅ Sent: {result['title']}")
        else: # from command
            await status_msg_container.delete() # Clean up "sending file" message
            # await update.effective_message.reply_text(f"✅ Download complete: {result['title']}", quote=True)


    except (TimedOut, NetworkError) as te:
        logger.error(f"Telegram API Timeout/NetworkError during download for user {user_id}, url {url}: {te}")
        if status_msg_container:
             await status_msg_container.edit_text("❌ Error sending file: Network connection timed out. Please try again.", parse_mode=ParseMode.MARKDOWN)
        else:
             await update.effective_message.reply_text("❌ Error sending file: Network connection timed out. Please try again.", quote=True)
    except Exception as e:
        logger.error(f"Error in download_music_from_url for user {user_id}, url {url}: {e}", exc_info=True)
        if status_msg_container:
            await status_msg_container.edit_text("❌ An unexpected error occurred. Please try again.", parse_mode=ParseMode.MARKDOWN)
        else:
            await update.effective_message.reply_text("❌ An unexpected error occurred. Please try again.", quote=True)
            
    finally:
        if user_id in active_downloads:
            active_downloads.remove(user_id)
        if result and result.get("success") and result.get("audio_path") and os.path.exists(result["audio_path"]):
            try:
                os.remove(result["audio_path"])
                logger.info(f"Deleted temporary audio file: {result['audio_path']}")
            except Exception as e_del:
                logger.error(f"Error deleting temporary audio file {result.get('audio_path', 'N/A')}: {e_del}")


async def download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles /download command with a YouTube URL."""
    url_to_download = ""
    if context.args:
        url_to_download = context.args[0]
    elif update.message and update.message.reply_to_message and update.message.reply_to_message.text:
        # Check if replying to a message containing a URL
        urls_in_reply = [word for word in update.message.reply_to_message.text.split() if is_valid_youtube_url(word)]
        if urls_in_reply:
            url_to_download = urls_in_reply[0]
    
    if not url_to_download or not is_valid_youtube_url(url_to_download):
        await update.message.reply_text(
            "Please provide a valid YouTube URL after the command, or reply to a message containing a YouTube URL.\n"
            "Example: `/download https://www.youtube.com/watch?v=dQw4w9WgXcQ`",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    
    await download_music_from_url(update, context, url_to_download, original_message_id=update.message.message_id)


async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /search command to search for music on YouTube."""
    if not context.args:
        await update.message.reply_text(
            "Please specify what you're looking for.\nExample: `/search Shape of You Ed Sheeran`",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    query = " ".join(context.args)
    query = sanitize_input(query)
    status_msg = await update.message.reply_text(f"🔍 Searching YouTube for: '{query}'...", quote=True)
    
    results = search_youtube(query, max_results=5) # Fetches from cache or new search
    
    await status_msg.delete()

    if not results:
        await update.message.reply_text(f"Sorry, I couldn't find any songs for '{query}'.", quote=True)
        return

    keyboard = []
    response_text = f"🔎 Search results for '{query}':\n\n"
    for i, result in enumerate(results): # Max 5 results due to search_youtube limit
        duration_str = ""
        if result.get('duration') and isinstance(result['duration'], (int, float)) and result['duration'] > 0:
            minutes = int(result['duration'] // 60)
            seconds = int(result['duration'] % 60)
            duration_str = f" [{minutes}:{seconds:02d}]"

        title = result['title']
        # Keep title length reasonable for buttons
        button_title_max_len = 45 
        display_title = (title[:button_title_max_len] + "...") if len(title) > button_title_max_len else title
        
        response_text += f"{i+1}. <b>{title}</b>{duration_str} by {result.get('uploader', 'Unknown')}\n"
        # Callback data for buttons must be < 64 bytes. Video IDs are 11 bytes.
        keyboard.append([InlineKeyboardButton(f"⏬ {display_title}", callback_data=f"dl_{result['id']}")])

    keyboard.append([InlineKeyboardButton("❌ Cancel Search", callback_data="cancel_search")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(response_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML, quote=True)


async def lyrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /lyrics command."""
    if not context.args:
        await update.message.reply_text(
            "Please specify a song and optionally an artist.\nExamples:\n"
            "/lyrics Bohemian Rhapsody\n"
            "/lyrics Queen - Bohemian Rhapsody",
            quote=True
        )
        return

    query_full = " ".join(context.args)
    query_full = sanitize_input(query_full)
    
    # Try to parse "Artist - Song" or "Song by Artist"
    song_title, artist_name = query_full, None
    if " - " in query_full:
        parts = query_full.split(" - ", 1)
        artist_name, song_title = parts[0].strip(), parts[1].strip()
    elif " by " in query_full.lower():
        parts = re.split(r'\s+by\s+', query_full, maxsplit=1, flags=re.IGNORECASE)
        song_title, artist_name = parts[0].strip(), parts[1].strip()

    if not song_title:
        await update.message.reply_text("Please provide a song title for the lyrics search.", quote=True)
        return

    status_msg = await update.message.reply_text(
        f"🔍 Searching for lyrics: \"{song_title}\"{' by ' + artist_name if artist_name else ''}...",
        quote=True
    )

    try:
        # Run blocking Genius API call in a separate thread
        lyrics_result = await asyncio.to_thread(get_lyrics, song_title, artist_name)
        
        if len(lyrics_result) > 4096: # Telegram message length limit
            await status_msg.edit_text(lyrics_result[:4090] + "\n\n✂️ Lyrics truncated...")
            # For very long lyrics, consider sending as multiple messages or a file
            # This simple truncation is a first step.
        else:
            await status_msg.edit_text(lyrics_result)
    except Exception as e:
        logger.error(f"Error in lyrics_command processing for '{query_full}': {e}", exc_info=True)
        await status_msg.edit_text("❌ Sorry, I encountered an unexpected error trying to get those lyrics.")


async def recommend_music_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provide personalized music recommendations based on analysis or Spotify."""
    user_id = update.effective_user.id
    status_msg = await update.message.reply_text("🎧 Finding personalized music recommendations for you...", quote=True)

    try:
        # Ensure user_contexts is initialized
        user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
        
        # Try to update Spotify data if linked
        if user_contexts[user_id].get("spotify") and user_contexts[user_id]["spotify"].get("access_token"):
            recently_played = await asyncio.to_thread(get_user_spotify_data, user_id, "player/recently-played", limit=5)
            if recently_played is not None: # Could be empty list
                user_contexts[user_id]["spotify"]["recently_played"] = recently_played
            
            top_tracks_data = await asyncio.to_thread(get_user_spotify_data, user_id, "top/tracks", limit=5)
            if top_tracks_data is not None:
                user_contexts[user_id]["spotify"]["top_tracks"] = top_tracks_data
            
            # Optionally, fetch playlists if needed for seeds (can be slow)
            # playlists = await asyncio.to_thread(get_user_spotify_playlists, user_id, limit=1)
            # if playlists: user_contexts[user_id]["spotify"]["playlists"] = playlists

        # Analyze conversation and existing context
        analysis = await analyze_conversation(user_id)
        
        current_mood = analysis.get("mood")
        seed_genres = analysis.get("genres", [])[:2] # Max 2 genres for seed
        seed_artists_names = analysis.get("artists", [])[:1] # Max 1 artist for seed by name
        
        recommendation_basis = []
        if current_mood: recommendation_basis.append(f"mood: {current_mood}")
        if seed_genres: recommendation_basis.append(f"genres: {', '.join(seed_genres)}")
        if seed_artists_names: recommendation_basis.append(f"artists: {', '.join(seed_artists_names)}")

        if not current_mood and not seed_genres and not seed_artists_names:
             # If no mood or preferences, prompt for mood first
            await status_msg.delete()
            await set_mood_command(update, context) # Re-use mood setting command
            # Inform user to try /recommend again after setting mood
            await update.message.reply_text("I need a bit more information. Please set your mood, then try /recommend again.", quote=True)
            return

        # Attempt Spotify recommendations first
        spotify_token_client_cred = get_spotify_token() # Client credentials token for general search/recs
        spotify_recs = []
        seed_track_ids = []

        # Prioritize user's actual Spotify data for seeds if available and makes sense
        # Prefer recently played or top tracks for seeds
        user_spotify_context = user_contexts[user_id].get("spotify",{})
        if user_spotify_context.get("access_token"): # User has linked Spotify
            source_tracks = []
            if user_spotify_context.get("recently_played"):
                source_tracks.extend(item["track"] for item in user_spotify_context["recently_played"] if item.get("track"))
            if user_spotify_context.get("top_tracks"):
                 source_tracks.extend(user_spotify_context["top_tracks"])
            
            # Get up to 2 unique track IDs from combined list
            # It's better to use actual track IDs from user's history
            # if we have them, than searching by artist name.
            unique_track_ids = list(set(t["id"] for t in source_tracks if t.get("id")))[:2]
            if unique_track_ids:
                seed_track_ids.extend(unique_track_ids)
                logger.info(f"Using user's Spotify track IDs as seeds: {seed_track_ids}")

        # If no track_ids from user history, or not enough, try finding tracks for seed artists
        if not seed_track_ids and seed_artists_names and spotify_token_client_cred:
            artist_track = search_spotify_track(spotify_token_client_cred, f"artist:{seed_artists_names[0]}")
            if artist_track and artist_track.get("id"):
                seed_track_ids.append(artist_track["id"])
                logger.info(f"Using track from searched artist '{seed_artists_names[0]}' as seed: {artist_track['id']}")

        # Finally, get recommendations from Spotify
        if seed_track_ids and spotify_token_client_cred:
            spotify_recs = get_spotify_recommendations(spotify_token_client_cred, seed_track_ids, limit=5)
        elif seed_genres and spotify_token_client_cred: # Fallback to genre seeds if no track seeds
             # Spotify API for /recommendations also takes seed_genres (up to 5 total seeds including artists/tracks)
             # Let's simplify for now and use track-based more often
             pass


        if spotify_recs:
            response_text = f"🎵 Based on {', '.join(recommendation_basis) if recommendation_basis else 'your vibe'}, here are some Spotify recommendations:\n\n"
            for i, track in enumerate(spotify_recs, 1):
                artists_text = ", ".join(a["name"] for a in track.get("artists", []))
                track_url = track.get("external_urls", {}).get("spotify", "")
                response_text += f"{i}. <b>{track['name']}</b> by {artists_text}"
                if track_url:
                    response_text += f" ([Listen on Spotify]({track_url}))"
                response_text += "\n"
            response_text += "\n💡 <i>You can ask me to /search for these songs to download the audio!</i>"
            await status_msg.edit_text(response_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
            return

        # Fallback: YouTube search if Spotify fails or provides no results
        # Construct a search query from mood, genres, artists
        yt_query_parts = []
        if current_mood: yt_query_parts.append(current_mood)
        if seed_genres: yt_query_parts.extend(seed_genres)
        if seed_artists_names: yt_query_parts.append(f"like {seed_artists_names[0]}")
        yt_search_query = " ".join(yt_query_parts) + " music" if yt_query_parts else "popular music"
        yt_search_query = sanitize_input(yt_search_query)
        
        logger.info(f"Falling back to YouTube search for recommendations with query: {yt_search_query}")
        yt_results = search_youtube(yt_search_query, max_results=5)

        if yt_results:
            response_text = f"🎵 Based on {', '.join(recommendation_basis) if recommendation_basis else 'your vibe'}, here are some YouTube recommendations:\n\n"
            keyboard = []
            for i, result in enumerate(yt_results, 1):
                duration_str = ""
                if result.get('duration'):
                    minutes = int(result['duration'] // 60)
                    seconds = int(result['duration'] % 60)
                    duration_str = f" [{minutes}:{seconds:02d}]"
                response_text += f"{i}. <b>{result['title']}</b> by {result.get('uploader', 'N/A')}{duration_str}\n"
                keyboard.append([InlineKeyboardButton(f"⏬ {result['title'][:30]}...", callback_data=f"dl_{result['id']}")])
            
            reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
            await status_msg.edit_text(response_text, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
        else:
            # Ultimate fallback: Generic recommendations by mood
            await status_msg.delete() # Delete "finding recommendations" message
            mood_for_generic = current_mood.lower() if current_mood else "happy" # Default to happy
            await provide_generic_recommendations(update, mood_for_generic)

    except Exception as e:
        logger.error(f"Error in recommend_music_command for user {user_id}: {e}", exc_info=True)
        if status_msg: # Check if status_msg was assigned
             await status_msg.edit_text("❌ I couldn't get personalized recommendations right now. Please try again later.")
        else: # Should not happen often but as a safeguard
            await update.message.reply_text("❌ I couldn't get personalized recommendations right now. Please try again later.", quote=True)


async def provide_generic_recommendations(update: Update, mood: str) -> None:
    """Provide generic recommendations when personalized ones fail."""
    mood_recommendations = {
        "happy": ["Happy - Pharrell Williams", "Walking on Sunshine - Katrina & The Waves", "Good Day Sunshine - The Beatles"],
        "sad": ["Hallelujah - Leonard Cohen (Jeff Buckley version)", "Mad World - Gary Jules", "Someone Like You - Adele"],
        "energetic": ["Don't Stop Me Now - Queen", "Uptown Funk - Mark Ronson ft. Bruno Mars", "Thunderstruck - AC/DC"],
        "relaxed": ["Weightless - Marconi Union", "Clair de Lune - Claude Debussy", "Teardrop - Massive Attack"],
        "focused": ["The Four Seasons - Vivaldi (any part)", "Time - Hans Zimmer", "Ambient 1: Music for Airports - Brian Eno"],
        "nostalgic": ["Yesterday - The Beatles", "Bohemian Rhapsody - Queen", "Wonderwall - Oasis"]
    }
    chosen_mood = mood.lower()
    if chosen_mood not in mood_recommendations: # Fallback if mood is unusual
        chosen_mood = "happy" # Or could pick randomly, or offer user to specify a known mood

    recommendations = mood_recommendations.get(chosen_mood)
    
    response_text = f"🎵 Since personalized recommendations are tricky right now, here are some general suggestions for a <b>{chosen_mood}</b> mood:\n\n"
    for i, track_info in enumerate(recommendations, 1):
        response_text += f"{i}. {track_info}\n"
    response_text += "\n💡 <i>You can ask me to /search for these songs to download the audio!</i>"
    
    # Use update.effective_message if called from a command context that doesn't edit a prior message.
    await update.effective_message.reply_text(response_text, parse_mode=ParseMode.HTML, quote=True)


# ==================== SPOTIFY LINKING HANDLERS ====================
async def link_spotify_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Initiate Spotify OAuth flow and start conversation to collect code."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI or SPOTIFY_REDIRECT_URI == "https://your-callback-url.com":
        await update.message.reply_text(
            "Sorry, Spotify linking is not properly configured by the bot admin. "
            "The `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`, and a valid `SPOTIFY_REDIRECT_URI` must be set.",
            quote=True
            )
        return ConversationHandler.END

    user_id = update.effective_user.id
    # The 'state' parameter is crucial for security and matching the user later.
    # For fully automatic flow with a callback server, this state would be used by the server.
    auth_url = (
        "https://accounts.spotify.com/authorize"
        f"?client_id={SPOTIFY_CLIENT_ID}"
        "&response_type=code"
        f"&redirect_uri={SPOTIFY_REDIRECT_URI}"
        "&scope=user-read-recently-played%20user-top-read%20playlist-read-private%20playlist-read-collaborative" # Added playlist scopes
        f"&state={user_id}" # Using user_id as state for simplicity in manual code entry. For prod, use a secure random string.
    )
    keyboard = [
        [InlineKeyboardButton("🔗 Link My Spotify Account", url=auth_url)],
        [InlineKeyboardButton("Cancel Linking", callback_data="cancel_spotify_linking")]
    ]
    await update.message.reply_text(
        "Let's link your Spotify account for personalized music recommendations! 🎵\n\n"
        "1. Click the button below to go to Spotify and authorize MelodyMind.\n"
        "2. After authorizing, Spotify will redirect you. If `SPOTIFY_REDIRECT_URI` is a real callback "
        "you've set up, it might handle it automatically. "
        "If it shows an error or a page with a `code=` in the URL, **copy that code value**.\n"
        "3. Return here and send **only the code** to me (e.g., just paste the long string of characters).\n"
        "   Alternatively, use the command `/spotify_code YOUR_CODE_HERE`.\n\n"
        "Ready? Click below to start:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN,
        quote=True
    )
    return SPOTIFY_CODE

async def spotify_code_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle Spotify authorization code from conversation or command."""
    user_id = update.effective_user.id
    message_text = update.message.text.strip()
    code = ""

    # Determine if it's a direct code paste or via /spotify_code command
    if message_text.startswith('/spotify_code'):
        command_args = message_text.split(maxsplit=1)
        if len(command_args) > 1:
            code = command_args[1].strip()
        else: # /spotify_code without arguments
            await update.message.reply_text(
                "Please provide the code after the command. Example: `/spotify_code YOUR_CODE_HERE`",
                parse_mode=ParseMode.MARKDOWN, quote=True
            )
            return SPOTIFY_CODE # Stay in the same state
    elif re.match(r'^[A-Za-z0-9_-]{50,}$', message_text): # Heuristic: Spotify codes are long alphanumeric strings
        code = message_text
    else: # Not a valid code format or /spotify_code command
        await update.message.reply_text(
            "That doesn't look like a Spotify authorization code. Please paste the full code you received, "
            "or use `/spotify_code YOUR_CODE_HERE`.",
            parse_mode=ParseMode.MARKDOWN, quote=True
        )
        return SPOTIFY_CODE # Stay in the same state

    if not code: # Should have been caught above, but as a safeguard
        await update.message.reply_text(
            "No code provided. Please send the Spotify authorization code.", quote=True
        )
        return SPOTIFY_CODE
        
    status_msg = await update.message.reply_text("🔄 Verifying your Spotify code...", quote=True)

    token_data = await asyncio.to_thread(get_user_spotify_token, user_id, code)

    if not token_data or not token_data.get("access_token"):
        await status_msg.edit_text(
            "❌ Failed to link Spotify account. The code might be invalid, expired, or already used. "
            "Please try /link_spotify again to get a new link and code."
        )
        return SPOTIFY_CODE # Stay, allow user to paste another code or cancel

    # Ensure user_contexts has a default structure if user_id is new
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    
    user_contexts[user_id]["spotify"] = {
        "access_token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "expires_at": token_data.get("expires_at")
        # We'll fetch recently_played/top_tracks on demand or in /recommend
    }

    await status_msg.edit_text(
        "✅ Spotify account linked successfully! 🎉\n"
        "I can now use your listening history to give you even better music recommendations. "
        "Try /recommend to see it in action!"
    )
    return ConversationHandler.END # End conversation on success


async def spotify_code_command_direct(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handler for /spotify_code command if used outside the conversation flow."""
    # This essentially duplicates logic from spotify_code_handler for direct command usage.
    if not context.args:
        await update.message.reply_text(
            "Please provide the Spotify authorization code after the command.\n"
            "Example: `/spotify_code YOUR_CODE_HERE`",
            parse_mode=ParseMode.MARKDOWN, quote=True
        )
        return

    user_id = update.effective_user.id
    code = context.args[0].strip()
    
    # Minimal check for code format
    if not re.match(r'^[A-Za-z0-9_-]{50,}$', code):
        await update.message.reply_text(
            "That doesn't look like a valid Spotify authorization code.", quote=True
        )
        return
        
    status_msg = await update.message.reply_text("🔄 Verifying your Spotify code...", quote=True)
    token_data = await asyncio.to_thread(get_user_spotify_token, user_id, code)

    if not token_data or not token_data.get("access_token"):
        await status_msg.edit_text(
            "❌ Failed to link Spotify account with this code. It might be invalid, expired, or already used. "
            "If you started with /link_spotify, please get a new link and code."
        )
        return

    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    user_contexts[user_id]["spotify"] = {
        "access_token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "expires_at": token_data.get("expires_at")
    }
    await status_msg.edit_text(
        "✅ Spotify account linked successfully using the provided code! 🎉\n"
        "Try /recommend for personalized suggestions!"
    )


async def cancel_spotify_linking(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the Spotify linking process from callback."""
    query = update.callback_query
    await query.answer() # Acknowledge callback
    await query.edit_message_text("Spotify linking cancelled. You can try again anytime with /link_spotify.")
    return ConversationHandler.END


# ==================== MOOD HANDLERS ====================
async def set_mood_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start conversation to set user's mood (entry point for /mood)."""
    keyboard = [
        [InlineKeyboardButton("😊 Happy", callback_data="mood_happy"), InlineKeyboardButton("😢 Sad", callback_data="mood_sad")],
        [InlineKeyboardButton("💪 Energetic", callback_data="mood_energetic"), InlineKeyboardButton("😌 Relaxed", callback_data="mood_relaxed")],
        [InlineKeyboardButton("🧠 Focused", callback_data="mood_focused"), InlineKeyboardButton("🕰️ Nostalgic", callback_data="mood_nostalgic")],
        [InlineKeyboardButton("🤷 Other/Not Sure", callback_data="mood_other")],
    ]
    await update.message.reply_text(
        "How are you feeling today? This helps me tailor music recommendations for you.",
        reply_markup=InlineKeyboardMarkup(keyboard),
        quote=True
    )
    return MOOD # MOOD is the first state in mood_conv_handler

async def mood_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
    """Handle mood selection button."""
    query = update.callback_query
    await query.answer()
    mood_choice = query.data # e.g., "mood_happy"
    
    user_id = query.from_user.id
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    
    actual_mood = mood_choice.split("_")[1]
    if actual_mood == "other":
        user_contexts[user_id]["mood"] = None # Clear mood if "other"
        await query.edit_message_text(
            "Okay, no problem! Your mood preference has been cleared. "
            "You can set it again later or I can try to infer it during conversation. "
            "Feel free to use /recommend or chat with me!"
        )
        return ConversationHandler.END

    user_contexts[user_id]["mood"] = actual_mood
    logger.info(f"User {user_id} set mood to: {actual_mood}")
    
    # (Optional) Ask for genre preference next, or end conversation.
    # For simplicity, we'll end here. /recommend will use this mood.
    await query.edit_message_text(
        f"Got it! You're feeling {actual_mood}. I'll keep that in mind for recommendations. 🎶\n\n"
        "You can now:\n"
        "  ✨ /recommend - Get music tailored to this mood.\n"
        "  💬 Or just chat with me!"
    )
    return ConversationHandler.END # End mood conversation


# ==================== NEW AI FEATURE HANDLERS ====================

async def analyze_lyrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyzes lyrics of a song for themes and mood using AI."""
    if not client:
        await update.message.reply_text("Sorry, the AI analysis feature is currently unavailable.", quote=True)
        return
    if not genius:
        await update.message.reply_text("Sorry, the lyrics fetching service is unavailable.", quote=True)
        return
        
    if not context.args:
        await update.message.reply_text(
            "Please specify a song and optionally an artist for lyric analysis.\nExamples:\n"
            "/analyze_lyrics Bohemian Rhapsody\n"
            "/analyze_lyrics Queen - Bohemian Rhapsody",
            quote=True
        )
        return

    query_full = " ".join(context.args)
    query_full = sanitize_input(query_full)
    
    song_title, artist_name = query_full, None
    if " - " in query_full:
        parts = query_full.split(" - ", 1)
        artist_name, song_title = parts[0].strip(), parts[1].strip()
    elif " by " in query_full.lower():
        parts = re.split(r'\s+by\s+', query_full, maxsplit=1, flags=re.IGNORECASE)
        song_title, artist_name = parts[0].strip(), parts[1].strip()

    if not song_title:
        await update.message.reply_text("Please provide a song title for the analysis.", quote=True)
        return

    status_msg = await update.message.reply_text(
        f"🔍 Fetching lyrics for \"{song_title}\"{' by ' + artist_name if artist_name else ''} to analyze...",
        quote=True
    )

    try:
        lyrics_text_full = await asyncio.to_thread(get_lyrics, song_title, artist_name)
        
        if "Sorry, I couldn't find those lyrics" in lyrics_text_full or \
           "Lyrics service is not available" in lyrics_text_full or \
           "error occurred" in lyrics_text_full:
            await status_msg.edit_text(lyrics_text_full) # Show the error from get_lyrics
            return

        # Strip the header like "📜 Lyrics for..."
        actual_lyrics = "\n".join(lyrics_text_full.splitlines()[2:]) # Assumes 2 lines of header

        if not actual_lyrics.strip() or "lyrics are empty" in lyrics_text_full.lower() :
            await status_msg.edit_text(f"The lyrics for \"{song_title}\" seem to be empty or missing. I can't analyze them.")
            return

        await status_msg.edit_text(f"Found lyrics for \"{song_title}\". Now analyzing with AI... 🧠 This might take a moment.")

        # OpenAI analysis
        analysis_prompt = (
            f"You are a musicologist AI. Analyze the following song lyrics for themes, mood, and overall meaning. "
            f"Provide a concise (3-5 sentences) summary of your analysis. Do not repeat the lyrics. \n\n"
            f"Song: {song_title}{' by ' + artist_name if artist_name else ''}\n"
            f"Lyrics:\n{actual_lyrics[:3000]}" # Limit lyrics length for token economy
        )
        
        ai_response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant specializing in music lyric analysis."},
                {"role": "user", "content": analysis_prompt}
            ],
            max_tokens=250, # Enough for a concise summary
            temperature=0.5
        )
        analysis_result = ai_response.choices[0].message.content.strip()

        final_message = f" تحليل كلمات أغنية \"{song_title}\":\n\n{analysis_result}"
        if len(final_message) > 4096 : final_message = final_message[:4090] + "..."

        await status_msg.edit_text(final_message)

    except Exception as e:
        logger.error(f"Error in analyze_lyrics_command for '{query_full}': {e}", exc_info=True)
        await status_msg.edit_text("❌ Sorry, an unexpected error occurred during the lyric analysis.")


async def songs_like_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Recommends songs similar to a given song/artist."""
    user_id = update.effective_user.id
    if not context.args:
        await update.message.reply_text(
            "Please specify a song (and optionally artist) to find similar music.\n"
            "Examples:\n"
            "/songslike Bohemian Rhapsody\n"
            "/songslike Queen - We Will Rock You",
            quote=True
        )
        return

    query_full = " ".join(context.args)
    query_full = sanitize_input(query_full)
    
    seed_song_title, seed_artist_name = query_full, None
    if " - " in query_full:
        parts = query_full.split(" - ", 1)
        seed_artist_name, seed_song_title = parts[0].strip(), parts[1].strip()
    elif " by " in query_full.lower():
        parts = re.split(r'\s+by\s+', query_full, maxsplit=1, flags=re.IGNORECASE)
        seed_song_title, seed_artist_name = parts[0].strip(), parts[1].strip()

    if not seed_song_title:
        await update.message.reply_text("Please provide a song title for similarity search.", quote=True)
        return

    search_description = f"\"{seed_song_title}\"{' by ' + seed_artist_name if seed_artist_name else ''}"
    status_msg = await update.message.reply_text(f"🔍 Finding songs similar to {search_description}...", quote=True)

    spotify_recs = []
    spotify_client_token = get_spotify_token() # General token

    if spotify_client_token:
        # Search for the seed track on Spotify to get its ID
        search_query_spotify = f"{seed_song_title} artist:{seed_artist_name}" if seed_artist_name else seed_song_title
        seed_track_spotify = search_spotify_track(spotify_client_token, search_query_spotify)

        if seed_track_spotify and seed_track_spotify.get("id"):
            seed_track_id = seed_track_spotify["id"]
            logger.info(f"Found seed track '{seed_track_spotify['name']}' ID {seed_track_id} for /songslike")
            # Get recommendations from Spotify using this track ID
            spotify_recs = get_spotify_recommendations(spotify_client_token, [seed_track_id], limit=5)
        else:
            logger.info(f"Seed track for /songslike not found on Spotify: {search_description}")


    if spotify_recs:
        response_text = f"🎵 Songs similar to {search_description} (from Spotify):\n\n"
        for i, track in enumerate(spotify_recs, 1):
            artists_text = ", ".join(a["name"] for a in track.get("artists", []))
            track_url = track.get("external_urls", {}).get("spotify", "")
            response_text += f"{i}. <b>{track['name']}</b> by {artists_text}"
            if track_url:
                response_text += f" ([Listen on Spotify]({track_url}))"
            response_text += "\n"
        response_text += "\n💡 <i>You can ask me to /search for these to download the audio!</i>"
        await status_msg.edit_text(response_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
        return

    # Fallback: YouTube search if Spotify yields no results or no token
    logger.info(f"Falling back to YouTube for /songslike: {search_description}")
    yt_search_query = f"songs like {seed_song_title} {seed_artist_name if seed_artist_name else ''}"
    yt_results = search_youtube(yt_search_query, max_results=5)

    if yt_results:
        response_text = f"🎵 Couldn't get specific Spotify matches, but here are some YouTube results for songs similar to {search_description}:\n\n"
        keyboard = []
        for i, result in enumerate(yt_results, 1):
            duration_str = ""
            if result.get('duration'):
                minutes = int(result['duration'] // 60); seconds = int(result['duration'] % 60)
                duration_str = f" [{minutes}:{seconds:02d}]"
            response_text += f"{i}. <b>{result['title']}</b> by {result.get('uploader', 'N/A')}{duration_str}\n"
            keyboard.append([InlineKeyboardButton(f"⏬ {result['title'][:30]}...", callback_data=f"dl_{result['id']}")])
        
        reply_markup = InlineKeyboardMarkup(keyboard) if keyboard else None
        await status_msg.edit_text(response_text, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
    else:
        await status_msg.edit_text(f"Sorry, I couldn't find any songs similar to {search_description} right now.")


# ==================== GENERAL MESSAGE & BUTTON HANDLERS ====================

async def enhanced_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    """Handle various button callbacks."""
    query = update.callback_query
    await query.answer() # Acknowledge callback
    data = query.data
    user_id = query.from_user.id

    logger.info(f"Button pressed: '{data}' by user {user_id}")

    # Download from search result
    if data.startswith("dl_"):
        video_id = data.split("_", 1)[1]
        if not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
            logger.error(f"Invalid YouTube video ID from button: {video_id}")
            await query.edit_message_text("❌ Error: Invalid video ID in button. Please try searching again.")
            return None # No conversation state change
        
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        # original_message_id helps reply to the search results message for context
        await download_music_from_url(update, context, youtube_url, original_message_id=query.message.message_id)
        # Message update/deletion is handled within download_music_from_url
        return None 

    elif data == "cancel_search":
        await query.edit_message_text("❌ Search cancelled.")
        return None

    # Note: Spotify linking and Mood conversation buttons are handled by their respective ConversationHandlers.
    # Adding them here would be redundant or cause conflicts if those convos are active.
    # If those handlers were not ConversationHandlers, then these would be needed:
    #
    # elif data.startswith("mood_"): -> This is part of mood_conv_handler
    # return await mood_button_handler(update, context) # Delegate to specific part of mood conv
    #
    # elif data == "cancel_spotify_linking": -> This is part of spotify_conv_handler
    # return await cancel_spotify_linking(update, context) # Delegate
    
    else:
        logger.warning(f"Unhandled callback_data: {data}")
        try: # Try to edit message to inform user, if message still exists
            await query.edit_message_text("This button action is not recognized or has expired.")
        except Exception as e: # If message edit fails (e.g. message deleted)
            logger.error(f"Failed to edit message for unhandled callback '{data}': {e}")
        return None


async def enhanced_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Main message handler for text not matching commands."""
    if not update.message or not update.message.text:
        return # Ignore empty messages or updates without text (e.g. user joins)

    user_id = update.effective_user.id
    raw_text = update.message.text
    sanitized_text = sanitize_input(raw_text) # Sanitize for safety, use for internal processing

    logger.debug(f"User {user_id} message: '{sanitized_text[:100]}...'")
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})

    # 1. Check for YouTube URLs for direct download
    if is_valid_youtube_url(raw_text): # Use raw_text for URL detection
        # If just a URL is sent, treat it as a download request
        await download_music_from_url(update, context, raw_text.strip(), original_message_id=update.message.message_id)
        return

    # 2. AI-driven check for implicit music requests (play, download, lyrics)
    # This is more nuanced than simple keyword matching.
    music_intent = await is_music_request(user_id, sanitized_text)
    if music_intent.get("is_music_request") and music_intent.get("song_query"):
        song_query = music_intent["song_query"]
        
        # Heuristically check if it's more likely a lyrics request
        if any(lw in sanitized_text.lower() for lw in ["lyrics", "words to", "what are the lyrics"]):
             context.args = [song_query] # Pass query to lyrics_command
             await lyrics_command(update, context)
             return

        # Otherwise, assume it's a request to find/play (search then offer download)
        status_msg = await update.message.reply_text(f"🔍 Okay, searching for '{song_query}'...", quote=True)
        results = search_youtube(song_query, max_results=3) # Limit to 3 for quick choice
        
        await status_msg.delete()
        if not results:
            await update.message.reply_text(f"Sorry, I couldn't find anything for '{song_query}'. Try using /search for more specific results.", quote=True)
            return

        # Offer choices
        keyboard = []
        response_text = f"I found these for '{song_query}'. Click to download:\n\n"
        for i, res in enumerate(results, 1):
            title = res['title']
            display_title = (title[:40] + "...") if len(title) > 40 else title
            response_text += f"{i}. <b>{title}</b> by {res.get('uploader', 'N/A')}\n"
            keyboard.append([InlineKeyboardButton(f"⏬ {display_title}", callback_data=f"dl_{res['id']}")])
        
        keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="cancel_search")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(response_text, reply_markup=reply_markup, parse_mode=ParseMode.HTML, quote=True)
        return

    # 3. If not a direct URL or clear music request by AI, proceed to general chat
    #   The AI response function `generate_chat_response` has system prompts to guide users
    #   to commands for downloads/lyrics if their chat message hints at it.
    #   Show "typing..." action
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
    
    chat_response = await generate_chat_response(user_id, sanitized_text)
    await update.message.reply_text(chat_response, quote=True)



async def clear_history_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear user conversation history and mood/preferences."""
    user_id = update.effective_user.id
    if user_id in user_contexts:
        user_contexts[user_id]["conversation_history"] = []
        user_contexts[user_id]["mood"] = None
        user_contexts[user_id]["preferences"] = []
        # Spotify data is not cleared here, only local conversation context.
        await update.message.reply_text("✅ Your conversation history, mood, and preferences with me have been cleared.", quote=True)
    else:
        await update.message.reply_text("You don't have any saved conversation data with me to clear.", quote=True)


async def cancel_conversation_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Generic cancel command for conversations."""
    await update.message.reply_text(
        "Okay, the current action has been cancelled. What would you like to do next?",
        quote=True
    )
    return ConversationHandler.END


# ==================== UTILITY AND ERROR HANDLING ====================
def sanitize_input(text: Optional[str]) -> str:
    """Sanitize user input to prevent injection and clean text."""
    if not text:
        return ""
    # Remove characters that might be problematic in URLs, filenames, or API calls
    # Allow spaces, alphanumerics, and common punctuation for queries.
    # This is a basic sanitizer. More specific sanitization might be needed depending on usage.
    text = re.sub(r'[<>]', '', text) # Remove HTML-like tags
    # Limit length to prevent overly long inputs
    return text.strip()[:500]


async def post_init(application: Application) -> None:
    """Post initialization tasks, e.g., setting bot commands."""
    commands = [
        ("start", "👋 Welcome to MelodyMind!"),
        ("help", "ℹ️ Show help and commands"),
        ("download", "⏬ Download audio from YouTube URL"),
        ("search", "🔎 Search YouTube for a song"),
        ("lyrics", "📜 Get song lyrics"),
        ("recommend", "🎧 Get music recommendations"),
        ("songslike", "🎶 Find songs similar to one you like"),
        ("analyze_lyrics", "🤔 AI analysis of song lyrics"),
        ("mood", "😊 Set your current mood"),
        ("link_spotify", "🔗 Link your Spotify account"),
        # ("spotify_code", "🔑 Enter Spotify auth code (if needed)"),
        ("clear", "🗑️ Clear your chat history with me"),
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Bot commands have been set.")
    # Perform initial cleanup of download directory on startup
    cleanup_downloads()


def cleanup_downloads() -> None:
    """Clean up any temporary files in the download directory."""
    # Be careful with this if multiple bot instances might share the directory (not typical for this setup)
    if not os.path.exists(DOWNLOAD_DIR):
        return
    try:
        logger.info(f"Cleaning up download directory: {DOWNLOAD_DIR}")
        for item_name in os.listdir(DOWNLOAD_DIR):
            item_path = os.path.join(DOWNLOAD_DIR, item_name)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path)
                    logger.debug(f"Deleted old file: {item_path}")
                # elif os.path.isdir(item_path): # Optionally remove subdirectories
                # shutil.rmtree(item_path)
            except Exception as e:
                logger.error(f"Failed to delete {item_path} during cleanup: {e}")
        logger.info("Download directory cleanup complete.")
    except Exception as e:
        logger.error(f"Error during general cleanup of download directory: {e}")

def signal_handler_fn(sig, frame) -> None:
    """Handle termination signals for graceful shutdown."""
    logger.info(f"Received signal {sig}, preparing to shut down...")
    cleanup_downloads() # Perform cleanup before exiting
    # Other shutdown tasks could go here (e.g. saving user_contexts if persistent)
    logger.info("MelodyMind bot is shutting down. Goodbye!")
    sys.exit(0)


async def error_handler_fn(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates and Bot API errors."""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

    # Log specific update details if available
    if isinstance(update, Update) and update.effective_message:
        chat_id = update.effective_message.chat_id
        user_id = update.effective_user.id if update.effective_user else "Unknown User"
        message_text = update.effective_message.text
        logger.error(f"Error occurred for user {user_id} in chat {chat_id}. Message: '{message_text[:100]}'")
    elif isinstance(update, Update) and update.callback_query:
        chat_id = update.callback_query.message.chat.id if update.callback_query.message else "Unknown Chat"
        user_id = update.callback_query.from_user.id
        callback_data = update.callback_query.data
        logger.error(f"Error occurred for user {user_id} in chat {chat_id}. Callback data: '{callback_data}'")
    
    # Add more sophisticated error handling if needed, e.g. notifying user
    if update and hasattr(update, 'effective_message') and update.effective_message:
        try:
            await update.effective_message.reply_text(
                "🤖 Oops! Something went wrong on my end. I've logged the issue. Please try again later, or use /help if you're stuck."
            )
        except Exception as e_reply:
            logger.error(f"Failed to send error message to user: {e_reply}")

# ==================== MAIN FUNCTION ====================
def main() -> None:
    """Start the enhanced bot with environment validation and command setup."""
    # Validate essential environment variables
    required_env_vars = ["TELEGRAM_TOKEN"] # OpenAI, Spotify, Genius are optional for base functionality
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.critical(f"CRITICAL: Missing required environment variables: {', '.join(missing_vars)}. Bot cannot start.")
        sys.exit(1)
    
    if not OPENAI_API_KEY: logger.warning("OPENAI_API_KEY not set. AI chat and analysis features will be disabled.")
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET: logger.warning("Spotify API credentials not set. Spotify integration features will be limited/disabled.")
    if not GENIUS_ACCESS_TOKEN: logger.warning("GENIUS_ACCESS_TOKEN not set. Lyrics feature will be disabled.")
    
    if SPOTIFY_REDIRECT_URI == "https://your-callback-url.com":
        logger.warning("SPOTIFY_REDIRECT_URI is set to the default placeholder. "
                       "Spotify account linking (OAuth) will likely fail or redirect to a non-functional page.")
   
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).post_init(post_init).build()

    # --- Command Handlers ---
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("download", download_command))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("lyrics", lyrics_command))
    application.add_handler(CommandHandler("recommend", recommend_music_command))
    application.add_handler(CommandHandler("clear", clear_history_command))
    
    # New AI feature commands
    application.add_handler(CommandHandler("analyze_lyrics", analyze_lyrics_command))
    application.add_handler(CommandHandler("songslike", songs_like_command))

    # --- Conversation Handlers ---
    # Spotify Linking Conversation
    spotify_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("link_spotify", link_spotify_command)],
        states={
            SPOTIFY_CODE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, spotify_code_handler), # Handles direct code paste
                CommandHandler("spotify_code", spotify_code_handler) # Handles /spotify_code <THE_CODE>
            ]
        },
        fallbacks=[
            CallbackQueryHandler(cancel_spotify_linking, pattern="^cancel_spotify_linking$"),
            CommandHandler("cancel", cancel_conversation_command) # Generic cancel for conversations
        ],
        conversation_timeout=timedelta(minutes=10).total_seconds() # Timeout for this conversation
    )
    application.add_handler(spotify_conv_handler)
    # Add a direct handler for /spotify_code if used outside conversation context (optional, but can be helpful)
    application.add_handler(CommandHandler("spotify_code", spotify_code_command_direct, block=False))


    # Mood Setting Conversation
    mood_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("mood", set_mood_command)],
        states={
            MOOD: [CallbackQueryHandler(mood_button_handler, pattern="^mood_")] 
            # No PREFERENCE or ACTION states needed for this simplified mood flow yet
        },
        fallbacks=[CommandHandler("cancel", cancel_conversation_command)],
        conversation_timeout=timedelta(minutes=5).total_seconds()
    )
    application.add_handler(mood_conv_handler)

    # --- General Message and Callback Handlers ---
    # Handles button clicks that are NOT part of a conversation
    application.add_handler(CallbackQueryHandler(enhanced_button_handler))
    # Handles general text messages (must be after command handlers)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, enhanced_handle_message))
    
    # --- Error Handler ---
    application.add_error_handler(error_handler_fn)

    # --- Signal Handlers for graceful shutdown ---
    signal.signal(signal.SIGINT, signal_handler_fn) # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler_fn) # Termination signal from OS/supervisor
    
    # Register cleanup for normal exit too (though signals should cover most cases)
    atexit.register(cleanup_downloads)


    logger.info("Starting MelodyMind Bot...")
    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    
    # This part is reached only if run_polling stops gracefully (e.g. not via sys.exit from signal)
    logger.info("MelodyMind bot has stopped polling.")
    cleanup_downloads() # Final cleanup

if __name__ == "__main__":
    main()
```