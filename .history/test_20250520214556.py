
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
from openai import AsyncOpenAI # CHANGED
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
aclient = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None # CHANGED to AsyncOpenAI
genius_client = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=15, retries=2) if GENIUS_ACCESS_TOKEN and lyricsgenius else None # Renamed for clarity

# Conversation states
MOOD, PREFERENCE, SPOTIFY_CODE = range(3) # REMOVED ACTION as it's unused

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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10)) # Adjusted tenacity params
def get_spotify_recommendations_sync(token: str, seed_tracks: List[str], limit: int = 5) -> List[Dict]:
    """Get track recommendations from Spotify."""
    if not token or not seed_tracks:
        logger.warning("No token or seed tracks for Spotify recommendations")
        return []
    url = "https://api.spotify.com/v1/recommendations"
    headers = {"Authorization": f"Bearer {token}"}
    # Spotify API allows up to 5 seed entities in total (tracks, artists, genres)
    params = {"seed_tracks": ",".join(seed_tracks[:5]), "limit": limit} # Using up to 5 seed tracks
    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        response.raise_for_status()
        return response.json().get("tracks", [])
    except requests.exceptions.HTTPError as http_error:
        logger.warning(f"Spotify recommendations HTTPError for seeds {seed_tracks}: {http_error.response.status_code} - {http_error.response.text if http_error.response else 'No response'}")
        if http_error.response and http_error.response.status_code == 400: # Bad request, likely invalid seed
            return [] # Don't retry bad requests aggressively
        raise # Reraise to allow tenacity to retry for other HTTP errors
    except requests.exceptions.RequestException as req_error:
        logger.error(f"Error getting Spotify recommendations: {req_error}")
        return [] # Return empty on other request errors after retries

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

# Kept the more robust version of refresh_spotify_token
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
        
        # Ensure spotify context exists
        if "spotify" not in user_contexts.setdefault(user_id, {}):
            user_contexts[user_id]["spotify"] = {}
            
        user_contexts[user_id]["spotify"].update({
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token", refresh_token), # Spotify might return a new refresh token
            "expires_at": expires_at
        })
        logger.info(f"Spotify token refreshed for user {user_id}")
        return token_data.get("access_token")
    except requests.exceptions.HTTPError as e:
        if e.response and e.response.status_code == 400: # Often "invalid_grant" for expired/revoked refresh_token
            logger.error(f"Invalid refresh token or grant for user {user_id}: {e.response.text}. Clearing stored token.")
            if user_id in user_contexts and "spotify" in user_contexts[user_id]:
                user_contexts[user_id]["spotify"] = {} # Clear invalid token data
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
        access_token = refresh_spotify_token_sync(user_id) # Already sync
        if not access_token:
            logger.warning(f"Failed to refresh Spotify token for user {user_id} to fetch {endpoint}.")
            return None

    url = f"https://api.spotify.com/v1/me/{endpoint}"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"limit": 10} # Consider making limit configurable or dynamic

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("items", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Spotify user data ({endpoint}) for user {user_id}: {e}")
        return None # Changed from `return` to `return None` for consistency

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
    params = {"limit": 10} # Consider making limit configurable

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

async def get_user_spotify_token(code: str) -> Optional[Dict]: # Removed user_id, not used in sync version
    return await asyncio.to_thread(get_user_spotify_token_sync, code)

async def refresh_spotify_token(user_id: int) -> Optional[str]:
    return await asyncio.to_thread(refresh_spotify_token_sync, user_id)

async def get_user_spotify_data(user_id: int, endpoint: str) -> Optional[List[Dict]]:
    return await asyncio.to_thread(get_user_spotify_data_sync, user_id, endpoint)

async def get_user_spotify_playlists(user_id: int) -> Optional[List[Dict]]:
    return await asyncio.to_thread(get_user_spotify_playlists_sync, user_id)


# ==================== YOUTUBE HELPER FUNCTIONS (SYNC) ====================

def is_valid_youtube_url(url: str) -> bool:
    if not url:
        return False
    patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=',
        r'(https?://)?youtu\.be/',
        r'(https?://)?(www\.)?youtube\.com/shorts/'
    ]
    return any(re.search(pattern, url) for pattern in patterns)

def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return sanitized[:150] # Increased length slightly for very long titles

def download_youtube_audio_sync(url: str) -> Dict[str, Any]: # Renamed to _sync
    """Download audio from a YouTube video with improved error handling."""
    video_id_match = re.search(r'(?:v=|/|\.be/)([0-9A-Za-z_-]{11})', url) # Improved regex
    if not video_id_match:
        logger.error(f"Invalid YouTube URL or could not extract video ID: {url}")
        return {"success": False, "error": "Invalid YouTube URL or video ID."}

    # Simplified ydl_opts, relying more on yt-dlp defaults with ffmpeg
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio[abr<=128]/bestaudio/best', # Ensures some audio is grabbed
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'), # Use os.path.join
        'quiet': True,
        'no_warnings': True,
        'noplaylist': True,
        'max_filesize': 50 * 1024 * 1024,  # 50 MB limit
        'prefer_ffmpeg': True, # Generally better to allow ffmpeg
        'postprocessors': [{ # Ensure m4a output if possible, or convert
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a', # yt-dlp will use aac for m4a typically
            'preferredquality': '128', # Corresponds to abr<=128
        }],
        # Remove 'postprocessor_args': ['-acodec', 'copy'] as it can be problematic
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Attempting to extract info for URL: {url}")
            info = ydl.extract_info(url, download=False)
            if not info:
                return {"success": False, "error": "Could not extract video information."}
            
            title = sanitize_filename(info.get('title', 'Unknown_Title'))
            artist = info.get('artist', info.get('uploader', 'Unknown_Artist')) # Better default
            # Construct expected filename, yt-dlp may change extension
            # The actual downloaded filename will be determined after download.
            # Forcing extension to m4a due to postprocessor
            expected_path_no_ext = os.path.join(DOWNLOAD_DIR, title)
            
            logger.info(f"Attempting to download audio for: {title}")
            ydl.download([url])

            # Find the downloaded file (yt-dlp might choose .m4a or other if conversion fails)
            # The postprocessor requests m4a, so that should be the primary target.
            downloaded_file_path = f"{expected_path_no_ext}.m4a"
            if not os.path.exists(downloaded_file_path):
                # Fallback: check for other common audio extensions if .m4a wasn't created
                # This might happen if ffmpeg is missing or postprocessing fails for some reason
                logger.warning(f"Expected m4a file not found at {downloaded_file_path}, searching for alternatives.")
                found_alternative = False
                for ext_candidate in ['webm', 'mp3', 'opus', 'ogg']:
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
            if file_size_mb > 50.5:  # Add a small buffer for calculation precision
                os.remove(downloaded_file_path) # Clean up oversized file
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
        # More specific error messages for common issues
        if "Unsupported URL" in str(e):
            error_msg = "Unsupported URL. Please provide a valid YouTube link."
        elif "Video unavailable" in str(e):
            error_msg = "This video is unavailable."
        elif "Private video" in str(e):
            error_msg = "This video is private."
        elif "Premiere" in str(e):
            error_msg = "This video is a premiere and not available for download yet."
        elif "members-only" in str(e):
            error_msg = "This video is for members only."
        else:
            error_msg = f"Download failed: {str(e)[:100]}" # Keep it concise for user
        logger.error(f"YouTube download error for {url}: {e}")
        return {"success": False, "error": error_msg}
    except Exception as e:
        logger.error(f"Unexpected error downloading YouTube audio for {url}: {e}", exc_info=True)
        return {"success": False, "error": "An unexpected error occurred during download."}

# Keep the cached version of search_youtube, make it _sync
@lru_cache(maxsize=100)
def search_youtube_sync(query: str, max_results: int = 5) -> List[Dict]: # Renamed
    """Search YouTube for videos matching the query with caching."""
    query = sanitize_input(query) # Sanitize query before search
    try:
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': 'discard_in_playlist', # Changed from True for better results
            'default_search': f'ytsearch{max_results}', # More direct way to specify search
            # 'format': 'bestaudio', # Not strictly needed for search info
            'noplaylist': True, # Handled by ytsearch not ytdl internals
            # 'playlist_items': f'1-{max_results}' # Not needed with ytsearch{max_results}
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # search_query = f"ytsearch{max_results}:{query}" # Now in default_search
            info = ydl.extract_info(query, download=False) # Pass query directly
            if not info or 'entries' not in info:
                logger.info(f"No YouTube search results for query: {query}")
                return []
            
            results = []
            for entry in info['entries']:
                if entry and entry.get('id'): # Ensure entry and ID exist
                    results.append({
                        'title': entry.get('title', 'Unknown Title'),
                        'url': entry.get('webpage_url') or f"https://www.youtube.com/watch?v={entry['id']}", # Prefer webpage_url
                        'thumbnail': entry.get('thumbnail', ''),
                        'uploader': entry.get('uploader', 'Unknown Artist'),
                        'duration': entry.get('duration', 0),
                        'id': entry['id']
                    })
            return results[:max_results] # Ensure max_results respected if API returns more
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
    """Fetch lyrics using LyricsGenius (synchronous)."""
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
            # Clean up common Genius annotations and headers
            lyrics = re.sub(r'^\d*ContributorsLyrics', '', lyrics, flags=re.IGNORECASE).strip()
            lyrics = re.sub(r'\[.*?\]', '', lyrics)  # Remove [Chorus], [Verse], etc.
            lyrics = re.sub(r'\d*EmbedShare URLCopyEmbedCopy', '', lyrics, flags=re.IGNORECASE) # Remove embed junk
            lyrics = re.sub(r'\nYou might also like', '', lyrics, flags=re.IGNORECASE) # Remove "You might also like"
            lyrics = os.linesep.join([s for s in lyrics.splitlines() if s.strip()]) # Remove empty lines

            if not lyrics:
                return f"Found song '{song.title}' by {song.artist}, but lyrics seem empty or couldn't be cleaned."
            
            # Truncate if extremely long, though Telegram splitting is preferred
            # max_genius_len = 15000 # Arbitrary internal limit before splitting for Telegram
            # if len(lyrics) > max_genius_len:
            #     lyrics = lyrics[:max_genius_len] + "\n\n[Lyrics truncated due to excessive length]"

            return f"🎶 Lyrics for **{song.title}** by **{song.artist}**:\n\n{lyrics}"
        else:
            search_term = f"'{song_title_s}'"
            if artist_name_s:
                search_term += f" by '{artist_name_s}'"
            return f"Sorry, I couldn't find lyrics for {search_term}."
    except requests.exceptions.Timeout:
        logger.warning(f"Timeout searching lyrics for {song_title_s}")
        return "Sorry, the lyrics search timed out. Please try again."
    except Exception as e:
        logger.error(f"Error getting lyrics for {song_title_s}: {e}", exc_info=True)
        return "An unexpected error occurred while fetching lyrics. The service might be down."

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

    # System prompt
    system_prompt_content = (
        "You are a friendly, empathetic music companion bot named MelodyMind. "
        "Your role is to: "
        "1. Have natural conversations about music and feelings. "
        "2. Recommend songs based on mood and preferences (but don't list songs unless explicitly asked or very relevant). "
        "3. Provide emotional support through music-related conversation. "
        "4. Keep responses concise (1-3 sentences) and warm. "
        "5. Do not suggest commands like /recommend or /lyrics. Engage naturally. "
        "6. If you sense the user wants a specific song, you can gently acknowledge it but don't offer to play or download. "
        "Example: User: 'I want to hear Blinding Lights'. You: 'Blinding Lights is a great track! What do you like about it?'"
    )
    messages = [{"role": "system", "content": system_prompt_content}]

    # Add context from bot's knowledge
    user_profile_info = []
    if context.get("mood"):
        user_profile_info.append(f"User's current mood: {context['mood']}.")
    if context.get("preferences"):
        user_profile_info.append(f"User's music preferences: {', '.join(context['preferences'])}.")
    if context.get("spotify", {}).get("recently_played"):
        artists = list(set(item["track"]["artists"][0]["name"] for item in context["spotify"].get("recently_played", []) if item.get("track"))) # Unique artists
        if artists:
            user_profile_info.append(f"User recently listened to: {', '.join(artists[:3])}.")
    
    if user_profile_info:
        messages.append({"role": "system", "content": "Context about the user: " + " ".join(user_profile_info)})

    # Conversation history (limit to last 10 exchanges / 20 messages)
    for hist_msg in context["conversation_history"][-20:]:
        messages.append(hist_msg)
    
    messages.append({"role": "user", "content": message_s})

    try:
        response = await aclient.chat.completions.create( # Use aclient
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.75 # Slightly higher for more natural convo
        )
        reply = response.choices[0].message.content.strip()
        
        # Update conversation history
        context["conversation_history"].append({"role": "user", "content": message_s})
        context["conversation_history"].append({"role": "assistant", "content": reply})
        context["conversation_history"] = context["conversation_history"][-20:] # Ensure limit
        
        return reply
    except Exception as e:
        logger.error(f"Error generating chat response for user {user_id}: {e}", exc_info=True)
        return "I'm having a little trouble thinking right now. How about we talk about your favorite song?"


async def is_music_request(message: str) -> Dict: # Removed user_id, not used
    """Use AI to determine if a message is a music/song request and extract query."""
    if not aclient:
        return {"is_music_request": False, "song_query": None}

    message_s = sanitize_input(message)
    try:
        response = await aclient.chat.completions.create( # Use aclient
            model="gpt-3.5-turbo-0125", # Specify model version, 0125 supports JSON mode well
            messages=[
                {"role": "system", "content": 
                    "You are an AI that analyzes user messages. Determine if the user is requesting to play, download, find, or get a specific song or music by an artist. "
                    "Respond in JSON format with two keys: 'is_music_request' (boolean) and 'song_query' (string, the song title and artist if identifiable, otherwise null). "
                    "If it's a general music discussion, 'is_music_request' should be false. Focus on explicit requests for a track."
                    "Examples: "
                    "'Play Bohemian Rhapsody by Queen' -> {\"is_music_request\": true, \"song_query\": \"Bohemian Rhapsody Queen\"}. "
                    "'Can you find me some sad songs?' -> {\"is_music_request\": false, \"song_query\": null}. "
                    "'I love listening to music.' -> {\"is_music_request\": false, \"song_query\": null}. "
                    "'Get me the song Africa by Toto' -> {\"is_music_request\": true, \"song_query\": \"Africa Toto\"}."
                 },
                {"role": "user", "content": f"Analyze this message: '{message_s}'"}
            ],
            max_tokens=80,
            temperature=0.1, # Low temp for structured output
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content:
            return {"is_music_request": False, "song_query": None}

        result = json.loads(content)
        
        is_request = result.get("is_music_request", False)
        # Ensure boolean interpretation
        if isinstance(is_request, str):
            is_request = is_request.lower() in ("true", "yes")
            
        song_query = result.get("song_query")
        if isinstance(song_query, str) and song_query.strip().lower() == "null": # Handle "null" string
            song_query = None
            
        return {
            "is_music_request": bool(is_request),
            "song_query": song_query.strip() if song_query else None
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError in is_music_request for message '{message_s}': {e}. Response: {content if 'content' in locals() else 'N/A'}")
        return {"is_music_request": False, "song_query": None}
    except Exception as e:
        logger.error(f"Error in is_music_request for message '{message_s}': {e}", exc_info=True)
        return {"is_music_request": False, "song_query": None}


async def analyze_conversation(user_id: int) -> Dict:
    """Analyze conversation history and Spotify data to extract music preferences using AI."""
    if not aclient: # Use aclient
        return {"genres": [], "artists": [], "mood": None}

    context = user_contexts.get(user_id, {})
    # Initialize default structure if context is empty or missing keys
    context.setdefault("mood", None)
    context.setdefault("preferences", [])
    context.setdefault("conversation_history", [])
    context.setdefault("spotify", {})


    # Only run analysis if there's enough data or explicit Spotify link
    if len(context["conversation_history"]) < 2 and not context["spotify"]:
        return {
            "genres": context["preferences"],
            "artists": [],
            "mood": context["mood"]
        }

    try:
        prompt_parts = []
        if context["conversation_history"]:
            # Select recent, relevant parts of conversation
            history_text = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in context["conversation_history"][-10:] # Last 5 exchanges
            ])
            prompt_parts.append(f"Recent conversation excerpts with user:\n{history_text}")

        spotify_summary = []
        if context["spotify"].get("recently_played"):
            tracks = [item['track'] for item in context["spotify"]["recently_played"][:5] if item.get('track')]
            track_info = [f"'{t['name']}' by {t['artists'][0]['name']}" for t in tracks if t.get('artists')]
            if track_info:
                spotify_summary.append(f"User recently played on Spotify: {'; '.join(track_info)}.")
        
        if context["spotify"].get("top_tracks"):
            tracks = context["spotify"]["top_tracks"][:5]
            track_info = [f"'{t['name']}' by {t['artists'][0]['name']}" for t in tracks if t.get('artists')]
            if track_info:
                spotify_summary.append(f"User's top tracks on Spotify: {'; '.join(track_info)}.")

        if spotify_summary:
            prompt_parts.append("Spotify listening habits:\n" + "\n".join(spotify_summary))
        
        if not prompt_parts: # Not enough data to analyze
             return {"genres": context["preferences"], "artists": [], "mood": context["mood"]}

        user_content = "\n\n".join(prompt_parts)

        response = await aclient.chat.completions.create( # Use aclient
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": 
                    "You are a music preference analyzer. Based on the provided conversation and Spotify data, "
                    "infer the user's current mood (e.g., happy, sad, energetic, relaxed, focused, nostalgic, reflective, adventurous), "
                    "preferred music genres (e.g., pop, rock, jazz, classical, electronic, hip-hop, folk, indie), "
                    "and liked artists. "
                    "Respond in JSON format with three keys: 'mood' (string or null), 'genres' (list of strings, up to 3), 'artists' (list of strings, up to 3). "
                    "If a category cannot be confidently inferred, provide an empty list or null. Be concise."
                },
                {"role": "user", "content": user_content}
            ],
            max_tokens=150,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        content = response.choices[0].message.content
        if not content:
            return {"genres": context["preferences"], "artists": [], "mood": context["mood"]}
            
        result = json.loads(content)

        # Validate and normalize results
        inferred_mood = result.get("mood") if isinstance(result.get("mood"), str) else None
        inferred_genres = [str(g) for g in result.get("genres", []) if isinstance(g, str)][:3]
        inferred_artists = [str(a) for a in result.get("artists", []) if isinstance(a, str)][:3]

        # Update context if new information is inferred
        if inferred_mood and (not context["mood"] or context["mood"] != inferred_mood) :
            context["mood"] = inferred_mood
        if inferred_genres and set(inferred_genres) != set(context["preferences"]):
            context["preferences"] = list(set(context["preferences"] + inferred_genres))[:3] # Merge and limit
        
        # user_contexts[user_id] = context # context is a reference, already updated

        return {
            "genres": inferred_genres or context["preferences"], # Fallback to existing
            "artists": inferred_artists,
            "mood": inferred_mood or context["mood"] # Fallback to existing
        }
    except json.JSONDecodeError as e:
        logger.error(f"JSONDecodeError in analyze_conversation for user {user_id}: {e}. Response: {content if 'content' in locals() else 'N/A'}")
        return {"genres": context["preferences"], "artists": [], "mood": context["mood"]}
    except Exception as e:
        logger.error(f"Error in analyze_conversation for user {user_id}: {e}", exc_info=True)
        return {"genres": context["preferences"], "artists": [], "mood": context["mood"]}


# ==================== MUSIC DETECTION FUNCTION ====================

def detect_music_in_message(text: str) -> Optional[str]:
    """Detect if a message is asking for music without an explicit YouTube URL, using regex patterns."""
    text_lower = text.lower()
    # Patterns look for "action verb" + "song title" + optional "by artist"
    # Prioritize patterns that include an artist
    patterns_with_artist = [
        r'(?:play|find|download|get|send me|i want to listen to|can you get|i need|find me|fetch|give me|send|song)\s+(.+?)\s+by\s+(.+)',
    ]
    patterns_song_only = [
        r'(?:play|find|download|get|send me|i want tolisten to|can you get|i need|find me|fetch|give me|send|song)\s+(.+)',
    ]

    for pattern in patterns_with_artist:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            song_title = match.group(1).strip().rstrip(',.?!')
            artist = match.group(2).strip().rstrip(',.?!')
            if song_title and artist:
                return f"{song_title} {artist}"

    for pattern in patterns_song_only:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            song_title = match.group(1).strip().rstrip(',.?!')
            if song_title:
                 # Avoid matching overly generic phrases like "play music"
                if song_title.lower() in ["music", "a song", "something", "some music"]:
                    continue
                return song_title

    # If no direct action verb + title, check for keywords that might indicate a music request
    # to be confirmed by AI.
    music_keywords = ['music', 'song', 'track', 'tune', 'audio', 'listen to']
    if any(keyword in text_lower for keyword in music_keywords):
        # If it's just "music" or "song" very generically, might not be specific enough
        if text_lower in ["music", "song", "audio", "play music", "play a song"]:
             return "AI_ANALYSIS_NEEDED" # Let AI decide if this is a request for specific music
        # Check if the text minus keywords is substantial enough to be a title
        potential_title = text_lower
        for kw in music_keywords + ['play', 'find', 'download', 'get', 'send me', 'i want to', 'can you', 'i need', 'fetch', 'give me']:
            potential_title = potential_title.replace(kw, "").strip()
        if len(potential_title) > 3 : # Arbitrary length to avoid matching just keywords
            return "AI_ANALYSIS_NEEDED" # Could be a title, let AI parse
            
    return None

# ==================== TELEGRAM BOT HANDLERS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued."""
    user = update.effective_user
    welcome_msg = (
        f"Hi {user.first_name}! 👋 I'm MelodyMind, your Music Healing Companion.\n\n"
        "I can:\n"
        "🎵 Download music from YouTube (just send a link or ask for a song!)\n"
        "📜 Find lyrics for any song (e.g., `/lyrics Bohemian Rhapsody`)\n"
        "💿 Recommend music based on your mood (try `/recommend` or `/mood`)\n"
        "💬 Chat about music and feelings\n"
        "🔗 Link your Spotify for personalized recommendations (`/link_spotify`)\n\n"
        "How can I help you today?"
    )
    await update.message.reply_text(welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message when the command /help is issued."""
    help_text = (
        "🎶 <b>MelodyMind - Your Music Companion</b> 🎶\n\n"
        "<b>Core Commands:</b>\n"
        "  /start - Welcome message & features overview\n"
        "  /help - This help message\n"
        "  /download [YouTube URL] - Download specific YouTube audio\n"
        "  /search [song name] - Search YouTube for a song & get download options\n"
        "  /lyrics [song name] or [artist - song] - Get song lyrics\n"
        "  /recommend - Get personalized music recommendations\n"
        "  /mood - Set your current mood for better recommendations\n"
        "  /link_spotify - Connect your Spotify account\n"
        "  /clear - Clear our chat history (for AI context)\n\n"
        "<b>Smart Chat:</b>\n"
        "You can also just talk to me! Try:\n"
        "  🔹 \"Download Shape of You by Ed Sheeran\"\n"
        "  🔹 \"What are the lyrics to Yesterday?\"\n"
        "  🔹 \"I'm feeling happy, suggest some tunes!\"\n"
        "  🔹 Send a YouTube link directly to download.\n\n"
        "I'll do my best to understand and help you find the music you need!"
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

# Global lock for modifying the active_downloads set if needed,
# though per-user checks often suffice.
# For this structure, the per-user check `if user_id in active_downloads:` is primary.
# download_access_lock = asyncio.Lock()

async def download_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download music from YouTube URL. Handles /download command and direct URL messages."""
    user_id = update.effective_user.id
    url = ""

    if context.args: # From /download <url>
        url = " ".join(context.args)
    elif update.message and update.message.text: # From direct message
        # Extract URL from message text (e.g., if user just pastes a link)
        # This part is also handled by enhanced_handle_message, but good to have in /download explicitly
        words = update.message.text.split()
        found_urls = [word for word in words if is_valid_youtube_url(word)]
        if found_urls:
            url = found_urls[0]
        else:
            await update.message.reply_text(
                "Please provide a valid YouTube URL after the /download command, or just send me a YouTube link directly."
            )
            return
    else: # Should not happen if filters are set correctly
        await update.message.reply_text("Could not find a URL to download.")
        return

    if not is_valid_youtube_url(url):
        await update.message.reply_text("That doesn't look like a valid YouTube URL. Please try again with a valid link (e.g., youtube.com/watch?v=...).")
        return

    # async with download_access_lock: # If a global lock for the set is desired
    if user_id in active_downloads:
        await update.message.reply_text("⚠️ You already have a download in progress. Please wait for it to complete before starting a new one.")
        return
    active_downloads.add(user_id)

    status_msg = None
    try:
        status_msg = await update.message.reply_text("⏳ Starting download... this might take a moment.")
        
        await status_msg.edit_text("🔍 Fetching video information...")
        # Call the async version of download_youtube_audio
        result = await download_youtube_audio(url) 

        if not result["success"]:
            error_message = result.get("error", "An unknown error occurred during download.")
            await status_msg.edit_text(f"❌ Download failed: {error_message}")
            return # active_downloads removed in finally

        await status_msg.edit_text(f"✅ Downloaded: {result['title']}\n⏳ Preparing to send file...")
        
        audio_path = result["audio_path"]
        with open(audio_path, 'rb') as audio_file:
            caption = f"🎵 {result['title']}"
            if result.get("artist") and result["artist"] != "Unknown_Artist":
                caption += f"\n🎤 Artist: {result['artist']}"
            if result.get("duration"):
                mins, secs = divmod(result['duration'], 60)
                caption += f"\n⏱️ Duration: {int(mins)}:{int(secs):02d}"

            logger.info(f"Sending audio: {result['title']} for user {user_id}")
            # PTB's default timeouts should be configured via ApplicationBuilder
            await update.message.reply_audio(
                audio=audio_file,
                title=result["title"][:64], # Telegram limit
                performer=result.get("artist", "Unknown Artist")[:64],
                caption=caption,
                thumbnail=None, # Could try sending thumbnail if available: result.get('thumbnail_url')
                duration=result.get('duration', 0)
            )
        
        # Delete original status message after successful send
        await status_msg.delete()
        logger.info(f"Successfully sent audio '{result['title']}' and deleted status message.")

    except TimedOut:
        logger.error(f"Timeout error during download process for {url}, user {user_id}.")
        if status_msg:
            await status_msg.edit_text("⌛ The operation timed out. Please try again. If the file is large, it might take longer.")
        else:
            await update.message.reply_text("⌛ The operation timed out. Please try again.")
    except NetworkError as ne:
        logger.error(f"Network error during download process for {url}, user {user_id}: {ne}")
        if status_msg:
            await status_msg.edit_text("🌐 A network error occurred. Please check your connection and try again.")
        else:
            await update.message.reply_text("🌐 A network error occurred. Please check your connection and try again.")
    except Exception as e:
        logger.error(f"Unexpected error in download_music for {url}, user {user_id}: {e}", exc_info=True)
        if status_msg:
            await status_msg.edit_text(f"❌ An unexpected error occurred: {str(e)[:100]}. Please try again.")
        else:
            await update.message.reply_text(f"❌ An unexpected error occurred: {str(e)[:100]}. Please try again.")
    finally:
        # async with download_access_lock: # If using global lock
        if user_id in active_downloads:
            active_downloads.remove(user_id)
        # Clean up the downloaded file
        if 'result' in locals() and result.get("success") and result.get("audio_path"):
            if os.path.exists(result["audio_path"]):
                try:
                    os.remove(result["audio_path"])
                    logger.info(f"Cleaned up downloaded file: {result['audio_path']}")
                except Exception as e_clean:
                    logger.error(f"Error cleaning up file {result['audio_path']}: {e_clean}")
        elif status_msg and not ('result' in locals() and result.get("success")): # If download failed and status_msg exists
             try:
                # If we are in finally and download didn't succeed, but status_msg was sent,
                # ensure it doesn't stay on "downloading..." if an error wasn't sent.
                # This is a fallback, usually specific errors are handled.
                pass # Error messages are usually sent in except blocks.
             except Exception:
                pass # Avoid errors in finally


async def link_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Initiate Spotify OAuth flow."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI or SPOTIFY_REDIRECT_URI == "https://your-callback-url.com":
        await update.message.reply_text("Sorry, Spotify linking is not properly configured by the bot admin.")
        return ConversationHandler.END

    user_id = update.effective_user.id
    # Note: 'state' should be securely generated and verified to prevent CSRF if used in a web context.
    # For a Telegram bot, user_id is often sufficient if the redirect URI is specific to the bot.
    auth_url = (
        "https://accounts.spotify.com/authorize"
        f"?client_id={SPOTIFY_CLIENT_ID}"
        "&response_type=code"
        f"&redirect_uri={SPOTIFY_REDIRECT_URI}"
        "&scope=user-read-recently-played%20user-top-read%20playlist-read-private%20playlist-read-collaborative" # Added playlist scopes
        f"&state={user_id}" # Using user_id as state for simple verification
    )
    keyboard = [
        [InlineKeyboardButton("🔗 Link My Spotify Account", url=auth_url)],
        [InlineKeyboardButton("Cancel", callback_data="cancel_spotify_linking")] # Changed callback_data
    ]
    await update.message.reply_text(
        "Let's link your Spotify to personalize your music experience! 🎵\n\n"
        "1. Click the button below to go to Spotify.\n"
        "2. Log in and authorize MelodyMind.\n"
        "3. Spotify will redirect you. **Copy the `code` parameter from the URL in your browser's address bar.**\n"
        "   (It will look something like `your-redirect-uri/?code=AQC...&state=...`)\n"
        "4. Paste only that `code` back here in the chat.\n\n"
        "Ready? Click below:",
        reply_markup=InlineKeyboardMarkup(keyboard),
        parse_mode=ParseMode.MARKDOWN # Already default, but good to be explicit
    )
    return SPOTIFY_CODE

async def spotify_code_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle Spotify authorization code from user message."""
    user_id = update.effective_user.id
    message_text = update.message.text.strip()
    
    # Basic validation for code format (Spotify codes are long and alphanumeric)
    if not (message_text and len(message_text) > 30 and message_text.isalnum()): # Basic check
        # More specific regex could be: r'^[A-Za-z0-9_-]{100,}$' (length varies)
        if message_text.startswith('/'): # User might be trying a command
             await update.message.reply_text(
                "It seems you sent a command. I'm expecting the Spotify authorization code. "
                "Please paste the code you got from Spotify."
            )
             return SPOTIFY_CODE # Remain in this state

        await update.message.reply_text(
            "That doesn't look like a valid Spotify code. It's usually a long string of letters and numbers. "
            "Please copy the code from the URL after authorizing on Spotify and paste it here.\n"
            "Or type /cancel to stop."
        )
        return SPOTIFY_CODE # Remain in this state

    code = message_text
    status_msg = await update.message.reply_text("⏳ Verifying your Spotify code...")

    token_data = await get_user_spotify_token(code) # Async wrapper
    if not token_data or not token_data.get("access_token"):
        await status_msg.edit_text(
            "❌ Failed to link Spotify. The code might be invalid, expired, or already used. "
            "Please try /link_spotify again to get a new link and code.\n"
            "Make sure you copy the new code correctly."
        )
        return SPOTIFY_CODE # Let user try again or cancel

    # Initialize context if not present
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    user_contexts[user_id]["spotify"] = {
        "access_token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "expires_at": token_data.get("expires_at")
    }

    # Fetch initial data to confirm working token (optional, but good UX)
    recently_played = await get_user_spotify_data(user_id, "player/recently-played")
    if recently_played is not None: # Check for None, as empty list is valid
        user_contexts[user_id]["spotify"]["recently_played"] = recently_played
        logger.info(f"Fetched recently played for user {user_id} after linking.")
    else: # Token might be valid but scopes insufficient or other issue
        logger.warning(f"Could not fetch recently played for user {user_id} immediately after linking, though token obtained.")


    await status_msg.edit_text(
        "✅ Spotify account linked successfully! 🎉\n"
        "I can now use your listening history for even better music recommendations. "
        "Try /recommend to see!"
    )
    return ConversationHandler.END


async def spotify_code_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /spotify_code command outside conversation (less ideal, but can be a fallback)."""
    if not context.args:
        await update.message.reply_text(
            "Please provide the Spotify authorization code after the command. Example:\n`/spotify_code YOUR_CODE_HERE`\n\n"
            "It's usually better to just paste the code after using /link_spotify.",
            parse_mode=ParseMode.MARKDOWN
        )
        return
    # Simulate a message to reuse spotify_code_handler if it were designed for direct calls
    # For now, this command isn't part of the ConversationHandler, so it acts independently.
    # This is a simplified path, assumes user is not in SPOTIFY_CODE state.
    
    user_id = update.effective_user.id
    code = context.args[0].strip()

    if not (code and len(code) > 30 and code.isalnum()):
        await update.message.reply_text("That doesn't look like a valid Spotify code.")
        return

    status_msg = await update.message.reply_text("⏳ Verifying your Spotify code (via command)...")
    token_data = await get_user_spotify_token(code)
    if not token_data or not token_data.get("access_token"):
        await status_msg.edit_text(
            "❌ Failed to link Spotify using this code (it might be invalid or expired). "
            "Please try the full /link_spotify process."
        )
        return

    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    user_contexts[user_id]["spotify"] = {
        "access_token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "expires_at": token_data.get("expires_at")
    }
    await status_msg.edit_text("✅ Spotify account linked successfully via command! Try /recommend.")


async def cancel_spotify_linking(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int: # Matched callback_data
    """Cancel the Spotify linking process from callback."""
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Spotify linking cancelled. You can try again anytime with /link_spotify.")
    return ConversationHandler.END

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /search command to search for music on YouTube."""
    if not context.args:
        await update.message.reply_text(
            "What song are you looking for? Example:\n"
            "/search Shape of You Ed Sheeran"
        )
        return

    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Searching YouTube for: '{query}'...")
    results = await search_youtube(query, max_results=5) # Use async version
    await status_msg.delete()
    
    await send_search_results_keyboard(update, query, results) # Use helper for consistency

async def send_search_results_keyboard(update: Update, query: str, results: List[Dict]) -> None:
    """Send YouTube search results with inline keyboard buttons for download."""
    if not results:
        await update.message.reply_text(f"Sorry, I couldn't find any songs on YouTube for '{query}'. Try different keywords?")
        return

    keyboard = []
    response_text = f"🔎 Here's what I found for '{query}' on YouTube:\n\n"
    
    for i, result in enumerate(results[:5]): # Limit to 5 options
        if not result.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', result['id']):
            logger.warning(f"Skipping invalid YouTube search result ID: {result.get('id', 'N/A')} for query '{query}'")
            continue

        duration_str = ""
        if result.get('duration'):
            minutes = int(result['duration'] // 60)
            seconds = int(result['duration'] % 60)
            duration_str = f" [{minutes}:{seconds:02d}]"
        
        title = result['title']
        # Button text needs to be short. Title in message can be longer.
        response_text += f"{i+1}. {title}{duration_str} (Uploader: {result.get('uploader', 'N/A')})\n"
        
        button_title = title[:40] + "..." if len(title) > 40 else title # Max button text length
        keyboard.append([InlineKeyboardButton(f"📥 {button_title}", callback_data=f"download_{result['id']}")])

    if not keyboard: # All results had invalid IDs
        await update.message.reply_text(f"Sorry, I found some matches for '{query}', but couldn't create download options. Please try again.")
        return

    keyboard.append([InlineKeyboardButton("❌ Cancel Search", callback_data="cancel_search")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    response_text += "\nClick a button to download the audio:"
    await update.message.reply_text(response_text, reply_markup=reply_markup)


async def auto_download_first_result(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str) -> None:
    """Helper to search and attempt download of the first YouTube result."""
    user_id = update.effective_user.id

    if user_id in active_downloads:
        await update.message.reply_text("⚠️ You have another download in progress. Please wait.")
        return
    active_downloads.add(user_id)
    
    status_msg = await update.message.reply_text(f"🔍 Searching for '{query}' to auto-download...")
    try:
        results = await search_youtube(query, max_results=1)
        if not results or not results[0].get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', results[0]['id']):
            await status_msg.edit_text(f"❌ Couldn't find a valid result for '{query}' to auto-download.")
            active_downloads.remove(user_id) # Remove here as it's an early exit
            return

        result_info = results[0]
        video_url = result_info["url"]
        await status_msg.edit_text(f"✅ Found: {result_info['title']}\n⏳ Downloading audio...")

        download_result = await download_youtube_audio(video_url)
        if not download_result["success"]:
            await status_msg.edit_text(f"❌ Auto-download failed: {download_result['error']}")
            return # active_downloads removed in finally

        await status_msg.edit_text(f"✅ Downloaded: {download_result['title']}\n⏳ Sending file...")
        with open(download_result["audio_path"], 'rb') as audio:
            await update.message.reply_audio(
                audio=audio,
                title=download_result["title"][:64],
                performer=download_result.get("artist", "Unknown Artist")[:64],
                caption=f"🎵 {download_result['title']} (Auto-downloaded)"
            )
        await status_msg.delete()
    except Exception as e:
        logger.error(f"Error in auto_download_first_result for '{query}': {e}", exc_info=True)
        if status_msg: # Check if status_msg was assigned
            try:
                await status_msg.edit_text(f"❌ Error during auto-download: {str(e)[:100]}")
            except Exception: # If editing fails (e.g. message deleted)
                 await update.message.reply_text(f"❌ Error during auto-download: {str(e)[:100]}")
        else:
            await update.message.reply_text(f"❌ Error during auto-download: {str(e)[:100]}")
    finally:
        if user_id in active_downloads:
            active_downloads.remove(user_id)
        if 'download_result' in locals() and download_result.get("success") and download_result.get("audio_path"):
            if os.path.exists(download_result["audio_path"]):
                try:
                    os.remove(download_result["audio_path"])
                except Exception as e_clean:
                    logger.error(f"Error cleaning up file in auto_download: {e_clean}")

async def auto_download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /autodownload command."""
    if not context.args:
        await update.message.reply_text(
            "Please specify what song you want to auto-download. Example:\n"
            "/autodownload Shape of You Ed Sheeran"
        )
        return
    query = " ".join(context.args)
    await auto_download_first_result(update, context, query)


async def get_lyrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /lyrics command."""
    if not context.args:
        await update.message.reply_text(
            "Please specify a song title for lyrics. Examples:\n"
            "`/lyrics Bohemian Rhapsody`\n"
            "`/lyrics Queen - Bohemian Rhapsody` (artist first is usually better!)",
            parse_mode=ParseMode.MARKDOWN
        )
        return

    query_full = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Searching for lyrics for: \"{query_full}\"...")

    try:
        # Try to parse artist and song
        # Common patterns: "Artist - Song", "Song by Artist"
        artist = None
        song_title = query_full

        # Regex for "Artist - Song Title" or "Song Title - Artist" (less common for input)
        # Prioritize "Artist - Song Title"
        match_artist_song = re.match(r'^(.*?)\s*-\s*(.+)$', query_full)
        # Regex for "Song Title by Artist"
        match_song_by_artist = re.match(r'^(.*?)\s+by\s+(.+)$', query_full, re.IGNORECASE)

        if match_artist_song:
            # Ambiguity: is it "Artist - Song" or "Song with hyphen - Artist"?
            # Let's assume "Artist - Song" is more common for Genius search.
            # Or, try searching with full query first, then parsed.
            # For simplicity here, we'll try one parsing.
            # A more advanced approach could try multiple interpretations or let Genius figure it out.
            potential_artist1, potential_song1 = match_artist_song.groups()
            # Heuristic: if the part before " - " is shorter, it's more likely the artist.
            # This is not foolproof.
            if len(potential_artist1) < len(potential_song1) or any(kw in potential_artist1.lower() for kw in ["ft", "feat", "with"]):
                 artist = potential_artist1.strip()
                 song_title = potential_song1.strip()
            else: # Assume it might be "Song with hyphen - Artist" or just a song with a hyphen
                 song_title = query_full # Keep original and let Genius try
                 # artist = potential_song1.strip() # Alternative parsing
                 # song_title = potential_artist1.strip()

        elif match_song_by_artist:
            song_title = match_song_by_artist.group(1).strip()
            artist = match_song_by_artist.group(2).strip()
        else:
            # No clear delimiter, Genius will try to parse `query_full` as song title
            song_title = query_full 

        lyrics_text = await get_lyrics(song_title, artist) # Use async version

        # Telegram message length limit is 4096
        if len(lyrics_text) > 4090: # Leave a little margin
            await status_msg.edit_text(lyrics_text[:4090] + "\n\n[Message truncated due to length. Full lyrics might be longer.]")
            # For very long lyrics, splitting into multiple messages is an option:
            # parts = [lyrics_text[i:i + 4090] for i in range(0, len(lyrics_text), 4090)]
            # await status_msg.edit_text(parts[0] + ("\n\n(Lyrics continue in next message...)" if len(parts) > 1 else ""))
            # for part in parts[1:]:
            #     await update.message.reply_text(part + ("\n\n(Lyrics continue in next message...)" if len(parts) > parts.index(part) + 1 else ""))
        else:
            await status_msg.edit_text(lyrics_text, parse_mode=ParseMode.MARKDOWN) # Genius can return markdown

    except Exception as e:
        logger.error(f"Error in get_lyrics_command for query '{query_full}': {e}", exc_info=True)
        await status_msg.edit_text("❌ Sorry, an unexpected error occurred while fetching lyrics.")


async def provide_generic_recommendations(update: Update, mood: str) -> None:
    """Provide generic YouTube search queries as recommendations when Spotify/AI fails."""
    mood = mood.lower() if mood else "general"
    generic_searches = {
        "happy": ["upbeat pop songs playlist", "feel good indie music", "happy dance music"],
        "sad": ["emotional acoustic songs", "sad songs for crying playlist", "reflective instrumental music"],
        "energetic": ["high energy workout music", "pump up rock anthems", "electronic dance music mix"],
        "relaxed": ["calming ambient music for sleep", "lofi hip hop radio beats to relax", "chillout instrumental playlist"],
        "focused": ["instrumental study music concentration", "alpha waves focus music", "classical music for reading"],
        "nostalgic": ["80s classic hits", "90s alternative rock playlist", "old school R&B Jams"]
    }
    # Fallback for unknown mood
    searches_to_suggest = generic_searches.get(mood, generic_searches["happy"]) 
    
    response_text = f"🎵 Since I couldn't get specific recommendations right now, here are some general YouTube search ideas for **{mood}** vibes:\n\n"
    for i, search_query in enumerate(searches_to_suggest, 1):
        # Create a clickable link to search on YouTube (opens in browser)
        yt_search_url = f"https://www.youtube.com/results?search_query={requests.utils.quote(search_query)}"
        response_text += f"{i}. <a href=\"{yt_search_url}\">{search_query.title()}</a>\n"
        # Or, offer to search via bot:
        # response_text += f"{i}. {search_query.title()} (Try: `/search {search_query}`)\n"

    response_text += "\n💡 You can also try `/search [song/artist]` or just send me a YouTube link to download!"
    await update.message.reply_text(response_text, parse_mode=ParseMode.HTML, disable_web_page_preview=True)


async def set_mood(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start conversation to set user's mood."""
    keyboard = [
        [
            InlineKeyboardButton("😊 Happy", callback_data="mood_happy"),
            InlineKeyboardButton("😢 Sad", callback_data="mood_sad"),
            InlineKeyboardButton("💪 Energetic", callback_data="mood_energetic"),
        ],
        [
            InlineKeyboardButton("😌 Relaxed", callback_data="mood_relaxed"),
            InlineKeyboardButton("🧠 Focused", callback_data="mood_focused"),
            InlineKeyboardButton("🕰️ Nostalgic", callback_data="mood_nostalgic"),
        ],
         [InlineKeyboardButton("🚫 Skip/Other", callback_data="mood_skip")],
    ]
    await update.message.reply_text(
        "How are you feeling today? This helps me tailor music suggestions for you.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return MOOD # Transition to MOOD state

# Combined recommend_music and mood button handler part
async def recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE, from_mood_setter: bool = False) -> Optional[int]:
    """Provide music recommendations. Can be called by /recommend or after mood setting."""
    user_id = update.effective_user.id
    message_object = update.callback_query.message if from_mood_setter and update.callback_query else update.message
    
    status_msg = await message_object.reply_text("🎧 Finding personalized music recommendations...", quote=False)

    try:
        # Ensure user context exists
        user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})

        # 1. Update Spotify data if linked (recently_played, top_tracks, playlists)
        #    This data is used by analyze_conversation and for direct seed tracks.
        if user_contexts[user_id].get("spotify", {}).get("access_token"):
            logger.info(f"Updating Spotify data for user {user_id} before recommendation.")
            # Run these in parallel if desired, but sequentially is fine too.
            rp = await get_user_spotify_data(user_id, "player/recently-played")
            if rp is not None: user_contexts[user_id]["spotify"]["recently_played"] = rp
            
            tt = await get_user_spotify_data(user_id, "top/tracks?time_range=short_term") # short_term for current vibe
            if tt is not None: user_contexts[user_id]["spotify"]["top_tracks"] = tt
            
            pl = await get_user_spotify_playlists(user_id)
            if pl is not None: user_contexts[user_id]["spotify"]["playlists"] = pl

        # 2. Analyze conversation and existing context (AI call)
        #    This can infer mood, genres, artists.
        logger.info(f"Analyzing conversation for user {user_id} for recommendation.")
        ai_analysis = await analyze_conversation(user_id)
        
        current_mood = ai_analysis.get("mood") or user_contexts[user_id].get("mood")
        current_genres = list(set(ai_analysis.get("genres", []) + user_contexts[user_id].get("preferences", [])))[:3] # Merge and limit
        current_artists = ai_analysis.get("artists", [])

        if not current_mood and not from_mood_setter: # If /recommend called and no mood known
            await status_msg.delete()
            # Delegate to /mood command flow
            await message_object.reply_text("To give you the best recommendations, I need to know your mood. Let's set it first!")
            return await set_mood(update, context) # This will start the mood conversation

        # 3. Prepare seed data for Spotify recommendations
        seed_track_ids: List[str] = []
        seed_artist_ids: List[str] = []
        # Using Spotify data primarily for seeds if available
        spotify_user_data = user_contexts[user_id].get("spotify", {})
        
        # Priority for seeds: Recently played > Top Tracks > Playlist tracks > AI inferred artists
        if spotify_user_data.get("recently_played"):
            seed_track_ids.extend([
                track["track"]["id"] for track in spotify_user_data["recently_played"][:2] 
                if track.get("track") and track["track"].get("id")
            ])
        if len(seed_track_ids) < 2 and spotify_user_data.get("top_tracks"):
            seed_track_ids.extend([
                track["id"] for track in spotify_user_data["top_tracks"][:(2-len(seed_track_ids))] 
                if track.get("id")
            ])
        if len(seed_track_ids) < 1 and spotify_user_data.get("playlists"): # Try one track from a playlist
            first_playlist_id = spotify_user_data["playlists"][0].get("id") if spotify_user_data["playlists"] else None
            if first_playlist_id:
                # Need to fetch playlist tracks (this is another API call)
                # For simplicity, this is omitted here but could be added.
                # For now, let's assume AI artists or genre search is fallback.
                pass
        
        # Use AI inferred artists for seeds if no track seeds from Spotify
        if not seed_track_ids and current_artists:
            client_token = await get_spotify_token()
            if client_token:
                for artist_name in current_artists[:2]: # Max 2 artist seeds
                    artist_search_res = await search_spotify_track(client_token, f"artist:{artist_name}")
                    if artist_search_res and artist_search_res.get("artists") and artist_search_res["artists"][0].get("id"):
                        # Actually Spotify recs want artist ID, not track ID for artist seed
                        # For simplicity, using track from this artist as seed.
                        seed_track_ids.append(artist_search_res["id"]) 
                        # Or, if API supports seed_artists: seed_artist_ids.append(artist_search_res["artists"][0]["id"])
                    if len(seed_track_ids) >=2: break # Max 2 seeds from here

        # 4. Attempt Spotify API recommendations (uses client credential token)
        if seed_track_ids: # Requires at least one seed track, artist, or genre
            logger.info(f"Getting Spotify recommendations for user {user_id} with seed_tracks: {seed_track_ids}")
            spotify_client_token = await get_spotify_token()
            if spotify_client_token:
                # Pass seed_genres (target attributes also possible: target_valence for mood)
                spotify_recs = await get_spotify_recommendations(spotify_client_token, seed_tracks=seed_track_ids, limit=5)
                if spotify_recs:
                    response = f"🎵 Based on your vibe (mood: {current_mood or 'general'}), here are some Spotify recommendations:\n\n"
                    for i, track in enumerate(spotify_recs, 1):
                        artists_text = ", ".join(a["name"] for a in track.get("artists", []))
                        album = track.get("album", {}).get("name", "")
                        track_url = track.get("external_urls", {}).get("spotify", "#")
                        response += f"{i}. <a href=\"{track_url}\"><b>{track['name']}</b></a> by {artists_text}"
                        if album: response += f" (from <i>{album}</i>)"
                        response += "\n"
                    response += "\n💡 You can ask me to download these by name or search for them!"
                    await status_msg.edit_text(response, parse_mode=ParseMode.HTML, disable_web_page_preview=True)
                    return ConversationHandler.END if from_mood_setter else None # End conv if from mood setter

        # 5. Fallback to YouTube search if Spotify recs fail or not enough seeds
        logger.info(f"Falling back to YouTube search for user {user_id} recommendation.")
        # Construct a YouTube search query
        yt_search_query_parts = []
        if current_mood: yt_search_query_parts.append(current_mood)
        if current_genres: yt_search_query_parts.append(current_genres[0]) # Use primary genre
        yt_search_query_parts.append("music")
        if current_artists: yt_search_query_parts.append(f"like {current_artists[0]}")
        
        yt_search_query = " ".join(yt_search_query_parts)
        if not yt_search_query.strip() or yt_search_query.strip() == "music": # Too generic
            yt_search_query = f"{current_mood or 'popular'} music playlist" # Default fallback

        yt_results = await search_youtube(yt_search_query, max_results=5)
        await status_msg.delete() # Delete "finding..." message before sending results
        if yt_results:
            await send_search_results_keyboard(message_object, yt_search_query, yt_results) # Use helper
        else:
            # Ultimate fallback: generic suggestions based on mood
            await provide_generic_recommendations(message_object, current_mood or "happy")
        
        return ConversationHandler.END if from_mood_setter else None

    except Exception as e:
        logger.error(f"Error in recommend_music for user {user_id}: {e}", exc_info=True)
        if status_msg:
            try:
                await status_msg.edit_text("I couldn't get personalized recommendations right now. Please try again later or try a more general search.")
            except Exception: # If status_msg cannot be edited
                await message_object.reply_text("I couldn't get personalized recommendations. Please try again.")
        else:
             await message_object.reply_text("I couldn't get personalized recommendations. Please try again.")
        return ConversationHandler.END if from_mood_setter else None


async def enhanced_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Optional[int]:
    """Handle button callbacks for mood, preferences, downloads, and search options."""
    query = update.callback_query
    await query.answer() # Acknowledge callback
    data = query.data
    user_id = query.from_user.id

    logger.debug(f"Button callback: '{data}' from user {user_id}")
    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})

    if data.startswith("mood_"):
        mood = data.split("_", 1)[1]
        if mood == "skip":
            await query.edit_message_text("Okay, no mood set for now. You can always use /mood later or just tell me how you feel!")
            # Proceed to recommend without mood, or end here.
            # Let's try recommending directly.
            await recommend_music(update, context, from_mood_setter=True)
            return ConversationHandler.END
            
        user_contexts[user_id]["mood"] = mood
        logger.info(f"User {user_id} set mood to: {mood}")

        # Ask for genre preference (optional step)
        keyboard = [
            [
                InlineKeyboardButton("Pop", callback_data="pref_pop"),
                InlineKeyboardButton("Rock", callback_data="pref_rock"),
                InlineKeyboardButton("Hip-Hop", callback_data="pref_hiphop"),
            ],
            [
                InlineKeyboardButton("Electronic", callback_data="pref_electronic"),
                InlineKeyboardButton("Classical", callback_data="pref_classical"),
                InlineKeyboardButton("Indie/Alt", callback_data="pref_indie"),
            ],
            [InlineKeyboardButton("No Preference / Skip Genre", callback_data="pref_skip")],
        ]
        await query.edit_message_text(
            f"Got it! You're feeling {mood}. Any specific music genre you're in the mood for?",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return PREFERENCE # Transition to preference state

    elif data.startswith("pref_"):
        preference = data.split("_", 1)[1]
        if preference != "skip":
            # Add to preferences, ensure no duplicates, limit total
            if "preferences" not in user_contexts[user_id]: user_contexts[user_id]["preferences"] = []
            if preference not in user_contexts[user_id]["preferences"]:
                user_contexts[user_id]["preferences"].append(preference)
            user_contexts[user_id]["preferences"] = user_contexts[user_id]["preferences"][:3] # Limit
            logger.info(f"User {user_id} added preference: {preference}")
            await query.edit_message_text(f"Great, noted your preference for {preference} music!")
        else:
            await query.edit_message_text("Okay, no specific genre preference noted.")
        
        # Now, automatically trigger recommendation flow
        await recommend_music(update, context, from_mood_setter=True)
        return ConversationHandler.END # End mood conversation

    elif data.startswith("download_") or data.startswith("auto_download_"): # auto_download_ is from enhanced_handle_message prompt
        video_id_prefix = "auto_download_" if data.startswith("auto_download_") else "download_"
        video_id = data.split(video_id_prefix, 1)[1]

        if not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
            logger.error(f"Invalid YouTube video ID in callback: {video_id}")
            await query.edit_message_text("❌ Invalid video ID. Please try searching again.")
            return None # Stay in current state or end if no conv

        url = f"https://www.youtube.com/watch?v={video_id}"
        
        if user_id in active_downloads:
            await query.edit_message_text("⚠️ You already have a download in progress. Please wait.")
            return None
        active_downloads.add(user_id)
        
        original_message_text = query.message.text # Save for potential restoration or context
        await query.edit_message_text(f"⏳ Starting download for video ID {video_id}...")
        
        try:
            result = await download_youtube_audio(url) # Use async version
            if not result["success"]:
                await query.edit_message_text(f"❌ Download failed: {result.get('error', 'Unknown reason')}")
                return None # active_downloads removed in finally

            file_path = result["audio_path"]
            # File size already checked in download_youtube_audio_sync

            await query.edit_message_text(f"✅ Downloaded: {result['title']}\n⏳ Sending file to you...")
            
            with open(file_path, 'rb') as audio_file:
                await context.bot.send_audio(
                    chat_id=query.message.chat_id,
                    audio=audio_file,
                    title=result["title"][:64],
                    performer=result.get("artist", "Unknown Artist")[:64],
                    caption=f"🎵 {result['title']}"
                )
            # Edit the original inline keyboard message to confirm completion or remove it
            await query.edit_message_text(f"✅ Download complete & sent: {result['title']}")
        
        except (TimedOut, NetworkError) as te:
            logger.error(f"Network/Timeout error sending audio for {video_id}: {te}", exc_info=True)
            try:
                await query.edit_message_text("❌ Failed to send the audio due to a network issue. The file was downloaded but couldn't be sent. Try again later?")
            except Exception: # If edit fails
                 await context.bot.send_message(query.message.chat_id, "❌ Failed to send the audio due to a network issue after download.")

        except Exception as e:
            logger.error(f"Error in button download handler for {video_id}: {e}", exc_info=True)
            try:
                await query.edit_message_text(f"❌ An error occurred during download: {str(e)[:100]}")
            except Exception:
                await context.bot.send_message(query.message.chat_id, f"❌ An error occurred: {str(e)[:100]}")
        finally:
            if user_id in active_downloads:
                active_downloads.remove(user_id)
            if 'result' in locals() and result.get("success") and result.get("audio_path"):
                 if os.path.exists(result["audio_path"]):
                    try:
                        os.remove(result["audio_path"])
                    except Exception as e_clean:
                         logger.error(f"Error cleaning file in button_handler: {e_clean}")
        return None # Not part of a conversation usually, or ends here

    elif data.startswith("show_options_"):
        search_query = data.split("show_options_", 1)[1]
        await query.edit_message_text(f"🔍 Fetching more options for '{search_query}'...")
        results = await search_youtube(search_query, max_results=5) # Use async
        if not results:
            await query.edit_message_text(f"Sorry, I couldn't find other options for '{search_query}'.")
            return None
        # Replace the current message with new search results
        # This reuses the search result display logic.
        # We need an update-like object for send_search_results_keyboard.
        # Simplest is to delete current and send new.
        await query.message.delete()
        await send_search_results_keyboard(query.message, search_query, results) # Pass query.message as "update"
        return None

    elif data == "cancel_search":
        await query.edit_message_text("❌ Search or download action cancelled.")
        return None
    
    elif data == "cancel_spotify_linking": # From Spotify link initiation
        await query.edit_message_text("Spotify linking process cancelled.")
        return ConversationHandler.END # End Spotify conversation if active

    return None # Fallback, should not happen with defined callbacks


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5), reraise=True)
async def reply_with_retry(message_obj, text, **kwargs):
    """Helper to reply with retry for common network issues."""
    return await message_obj.reply_text(text, **kwargs)

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=5), reraise=True)
async def edit_message_with_retry(message_obj, new_text, **kwargs):
    """Helper to edit message with retry."""
    return await message_obj.edit_text(new_text, **kwargs)

async def enhanced_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Main message handler for non-command text."""
    if not update.message or not update.message.text:
        return # Ignore non-text messages here

    user_id = update.effective_user.id
    text = sanitize_input(update.message.text) # Sanitize and limit length
    logger.debug(f"Handling message from user {user_id}: \"{text[:50]}...\"")

    user_contexts.setdefault(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    
    typing_action_task = asyncio.create_task(context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing'))


    try:
        # 1. Check for YouTube URL for direct download
        if is_valid_youtube_url(text):
            await typing_action_task # Ensure it's awaited or cancelled
            # Delegate to /download command logic
            context.args = [text] # Simulate arguments for download_music
            await download_music(update, context)
            return

        # 2. Detect explicit song request patterns (regex based)
        detected_song_query_regex = detect_music_in_message(text)

        if detected_song_query_regex and detected_song_query_regex != "AI_ANALYSIS_NEEDED":
            await typing_action_task
            status_msg_regex = await reply_with_retry(update.message, f"🔍 I think you want '{detected_song_query_regex}'. Searching YouTube...")
            results = await search_youtube(detected_song_query_regex, max_results=3)
            await status_msg_regex.delete()
            if results:
                await send_search_results_keyboard(update, detected_song_query_regex, results)
            else:
                await reply_with_retry(update.message, f"Sorry, I couldn't find '{detected_song_query_regex}' on YouTube.")
            return
        
        # 3. If regex didn't find a clear song, or marked for AI, use AI for intent
        #    (is_music_request / lyrics request / general chat)
        #    Only use AI if it's not a simple greeting or short phrase.
        ai_intent = {"is_music_request": False, "song_query": None, "is_lyrics_request": False}
        if len(text.split()) > 2 or detected_song_query_regex == "AI_ANALYSIS_NEEDED": # Avoid AI for very short texts unless regex suggested it
            # This could be a more complex AI call to determine intent: download, lyrics, chat
            # For now, using is_music_request and specific keywords for lyrics
            
            # Check for lyrics intent based on keywords before general music request AI
            lyrics_keywords = ["lyrics", "words to the song", "song text for"] # Simplified
            if any(keyword in text.lower() for keyword in lyrics_keywords):
                # Extract potential song title for lyrics
                potential_song_for_lyrics = text.lower()
                for kw in lyrics_keywords: potential_song_for_lyrics = potential_song_for_lyrics.replace(kw, "")
                potential_song_for_lyrics = potential_song_for_lyrics.strip(".?! ")
                if potential_song_for_lyrics:
                    ai_intent["is_lyrics_request"] = True
                    ai_intent["song_query"] = potential_song_for_lyrics # This will be passed to lyrics command

            if not ai_intent["is_lyrics_request"]: # If not lyrics, check for music download/search request
                music_request_analysis = await is_music_request(text)
                if music_request_analysis["is_music_request"] and music_request_analysis["song_query"]:
                    ai_intent["is_music_request"] = True
                    ai_intent["song_query"] = music_request_analysis["song_query"]
        
        await typing_action_task # Ensure it's handled before next reply

        if ai_intent["is_music_request"] and ai_intent["song_query"]:
            status_msg_ai = await reply_with_retry(update.message, f"🔍 Got it! Searching YouTube for '{ai_intent['song_query']}' based on your message...")
            results = await search_youtube(ai_intent['song_query'], max_results=3)
            await status_msg_ai.delete()
            if results:
                await send_search_results_keyboard(update, ai_intent['song_query'], results)
            else:
                await reply_with_retry(update.message, f"Sorry, I couldn't find '{ai_intent['song_query']}' on YouTube after checking.")
            return

        if ai_intent["is_lyrics_request"] and ai_intent["song_query"]:
            context.args = [ai_intent["song_query"]] # Simulate args for command
            await get_lyrics_command(update, context)
            return

        # 4. If none of the above, treat as general conversation
        #    (Update mood from text if applicable, then generate chat response)
        lower_text = text.lower()
        mood_detected_in_text = None
        if "i'm feeling" in lower_text or "i feel" in lower_text:
            try:
                split_phrase = "i'm feeling" if "i'm feeling" in lower_text else "i feel"
                mood_token = lower_text.split(split_phrase, 1)[1].strip().split()[0].rstrip('.,?!')
                # Very basic mood mapping, can be expanded
                valid_moods = ["happy", "sad", "energetic", "relaxed", "focused", "nostalgic", "anxious", "stressed", "calm", "excited"]
                if mood_token in valid_moods:
                    mood_detected_in_text = mood_token
                    user_contexts[user_id]["mood"] = mood_detected_in_text
                    logger.info(f"Mood '{mood_detected_in_text}' detected in text for user {user_id}")
            except IndexError:
                pass # Splitting failed, no mood token found

        # Typing action for AI response will be handled by send_chat_action in generate_chat_response if needed
        chat_response_text = await generate_chat_response(user_id, text)
        await reply_with_retry(update.message, chat_response_text)

    except RetryError as re: # From reply_with_retry or edit_message_with_retry
        logger.warning(f"Failed to send/edit message for user {user_id} after retries: {re}")
        # Don't try to send another message if the core send failed.
    except Exception as e:
        logger.error(f"Error in enhanced_handle_message for user {user_id}, text \"{text[:50]}...\": {e}", exc_info=True)
        try:
            await reply_with_retry(update.message, "Pardon me, I seem to have hit a snag. Could you try phrasing that differently or try a command like /help?")
        except Exception: # If even error reporting fails
            logger.error(f"Failed to send error fallback message to user {user_id}")
    finally:
        if not typing_action_task.done():
            typing_action_task.cancel() # Cancel if not already done


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear user conversation history for AI context."""
    user_id = update.effective_user.id
    if user_id in user_contexts and "conversation_history" in user_contexts[user_id]:
        user_contexts[user_id]["conversation_history"] = []
        await update.message.reply_text("✅ Our chat history for AI context has been cleared.")
    else:
        await update.message.reply_text("Hmm, I don't seem to have a chat history with you to clear.")

async def cancel_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int: # Renamed for clarity
    """Generic cancel handler for conversations."""
    message_text = "Okay, action cancelled."
    if update.callback_query:
        await update.callback_query.answer()
        try: # Try editing the message with the inline keyboard
            await update.callback_query.edit_message_text(message_text)
        except Exception: # If it fails (e.g. no text to edit, or message too old)
            await context.bot.send_message(update.effective_chat.id, message_text)
    elif update.message:
        await update.message.reply_text(message_text)
    
    logger.info(f"Conversation cancelled for user {update.effective_user.id}")
    return ConversationHandler.END

# ==================== ERROR HANDLING ====================
# Use the async version of handle_error
async def handle_telegram_error(update: Optional[object], context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log Errors caused by Updates and Tries to Notify User."""
    logger.error(f"Update: {update} caused error: {context.error}", exc_info=context.error)

    # Try to notify the user, if an update object is available
    if isinstance(context.error, (TimedOut, NetworkError)):
        error_message = "I'm having trouble connecting. Please try again in a moment. 🌐"
    else:
        error_message = "Oops! Something went wrong on my end. I've logged the issue. Please try again later. 🛠️"

    if update and hasattr(update, 'effective_message') and update.effective_message:
        try:
            await update.effective_message.reply_text(error_message)
        except Exception as e:
            logger.error(f"Failed to send error notification to user after an error: {e}")
            # Fallback if replying to the message fails (e.g. message deleted)
            if hasattr(update, 'effective_chat') and update.effective_chat:
                try:
                    await context.bot.send_message(chat_id=update.effective_chat.id, text=error_message)
                except Exception as e_fallback:
                    logger.error(f"Failed to send fallback error notification: {e_fallback}")
    elif update and hasattr(update, 'effective_chat') and update.effective_chat: # For callback query errors without message
         try:
            await context.bot.send_message(chat_id=update.effective_chat.id, text=error_message)
         except Exception as e_cb_fallback:
            logger.error(f"Failed to send error notification for callback query error: {e_cb_fallback}")


# ==================== CLEANUP FUNCTIONS ====================
def cleanup_downloads_sync() -> None: # Renamed for clarity
    """Clean up any temporary files in the download directory."""
    logger.info(f"Attempting to clean up download directory: {DOWNLOAD_DIR}")
    cleaned_count = 0
    error_count = 0
    if os.path.exists(DOWNLOAD_DIR):
        for item_name in os.listdir(DOWNLOAD_DIR):
            item_path = os.path.join(DOWNLOAD_DIR, item_name)
            try:
                if os.path.isfile(item_path) or os.path.islink(item_path):
                    os.unlink(item_path) # Use unlink for files/symlinks
                    cleaned_count +=1
                elif os.path.isdir(item_path): # Should not have subdirs, but good to check
                    # import shutil; shutil.rmtree(item_path) # If recursive delete needed
                    logger.warning(f"Found unexpected subdirectory in downloads: {item_path}")
            except Exception as e:
                logger.error(f"Error removing {item_path} during cleanup: {e}")
                error_count +=1
        if cleaned_count > 0 or error_count == 0 :
            logger.info(f"Cleaned up {cleaned_count} file(s) from {DOWNLOAD_DIR}. {error_count} errors.")
        elif error_count > 0:
             logger.error(f"Cleanup of {DOWNLOAD_DIR} encountered {error_count} errors.")
    else:
        logger.info(f"Download directory {DOWNLOAD_DIR} does not exist, no cleanup needed.")

async def cleanup_downloads(): # Async wrapper if needed, though usually run at exit
    await asyncio.to_thread(cleanup_downloads_sync)

# ==================== SIGNAL HANDLERS ====================
def sig_handler(sig, frame): # Must be synchronous for signal module
    logger.info(f"Received signal {sig}, initiating shutdown and cleanup...")
    cleanup_downloads_sync() # Run synchronous version here
    # Other cleanup tasks can be added here
    logger.info("Cleanup complete. Exiting.")
    sys.exit(0)

# ==================== MAIN FUNCTION ====================

def main() -> None:
    """Start the bot."""
    # SPOTIFY_REDIRECT_URI is crucial for /link_spotify
    required_env_vars = [
        "TELEGRAM_TOKEN", "OPENAI_API_KEY", 
        "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET", "SPOTIFY_REDIRECT_URI",
        "GENIUS_ACCESS_TOKEN"
    ]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]
    if missing_vars:
        logger.critical(f"FATAL: Missing required environment variables: {', '.join(missing_vars)}. Bot cannot start.")
        sys.exit(1)
    
    if os.getenv("SPOTIFY_REDIRECT_URI") == "https://your-callback-url.com":
        logger.warning("SPOTIFY_REDIRECT_URI is set to the default placeholder. Spotify OAuth /link_spotify will NOT work correctly.")

    # Configure custom timeouts for the bot's HTTPX client
    # These values are examples, adjust based on needs (especially write_timeout for large files)
    application = (
        Application.builder()
        .token(TOKEN)
        .read_timeout(20)      # For bot API calls (not getUpdates)
        .write_timeout(75)     # For sending messages/files (esp. audio)
        .connect_timeout(15)   # For establishing connections
        .pool_timeout(60)      # For connections from the pool
        .get_updates_read_timeout(40) # For long polling getUpdates
        .build()
    )

    # Command Handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("download", download_music)) # Handles /download <url>
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("autodownload", auto_download_command))
    application.add_handler(CommandHandler("lyrics", get_lyrics_command))
    application.add_handler(CommandHandler("recommend", recommend_music)) # Direct recommend
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(CommandHandler("spotify_code", spotify_code_command)) # Fallback if not in conv

    # Spotify Linking Conversation Handler
    spotify_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("link_spotify", link_spotify)],
        states={
            SPOTIFY_CODE: [
                MessageHandler(filters.TEXT & ~filters.COMMAND, spotify_code_handler),
                # Callback for "Cancel" button within the link_spotify message
                CallbackQueryHandler(cancel_spotify_linking, pattern="^cancel_spotify_linking$")
            ]
        },
        fallbacks=[
            CommandHandler("cancel", cancel_conversation),
            CallbackQueryHandler(cancel_spotify_linking, pattern="^cancel_spotify_linking$") # Ensure cancel button works
        ],
        conversation_timeout=timedelta(minutes=5).total_seconds() # Timeout for conversation
    )
    application.add_handler(spotify_conv_handler)

    # Mood Setting and Recommendation Conversation Handler
    mood_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("mood", set_mood)],
        states={
            MOOD: [CallbackQueryHandler(enhanced_button_handler, pattern="^mood_")],
            PREFERENCE: [CallbackQueryHandler(enhanced_button_handler, pattern="^pref_")]
            # ACTION state was removed as it wasn't used. Recommendation happens after PREFERENCE.
        },
        fallbacks=[CommandHandler("cancel", cancel_conversation)],
        conversation_timeout=timedelta(minutes=3).total_seconds()
    )
    application.add_handler(mood_conv_handler)

    # General CallbackQueryHandler for buttons not part of specific conversations (e.g., download from search)
    application.add_handler(CallbackQueryHandler(enhanced_button_handler))

    # MessageHandler for general text (must be last among message handlers)
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, enhanced_handle_message))
    
    # Error Handler
    application.add_error_handler(handle_telegram_error)

    # Signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)
    # Register cleanup for normal exit too
    atexit.register(cleanup_downloads_sync) # Use sync version for atexit

    # Perform initial cleanup of download dir on startup
    logger.info("Performing initial cleanup of download directory...")
    cleanup_downloads_sync()

    logger.info("Starting MelodyMind Bot...")
    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.critical(f"Critical error running the bot application: {e}", exc_info=True)
    finally:
        logger.info("Bot application has stopped. Performing final cleanup...")
        cleanup_downloads_sync() # Final cleanup attempt

if __name__ == "__main__":
    main()
