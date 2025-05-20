import os
import logging
import requests # Still used for some sync Spotify calls, and by lyricsgenius
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
from tenacity import retry, stop_after_attempt, wait_exponential, AsyncRetrying
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
SPOTIFY_REDIRECT_URI = os.getenv("SPOTIFY_REDIRECT_URI", "https://your-callback-url.com")

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO # Consider logging.DEBUG for development
)
logger = logging.getLogger(__name__)

# Initialize clients
client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN, timeout=20, retries=3, sleep_time=0.5) if GENIUS_ACCESS_TOKEN and lyricsgenius else None

# Conversation states
MOOD, PREFERENCE, ACTION, SPOTIFY_CODE = range(4) # ACTION state currently unused in handlers

# Track active downloads and user contexts
active_downloads = set()
user_contexts: Dict[int, Dict] = {}
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def sanitize_input(text: str) -> str:
    if not text: return ""
    return re.sub(r'[<>;&]', '', text.strip())[:200]

# ==================== SPOTIFY HELPER FUNCTIONS ====================
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=6))
async def get_spotify_token_async() -> Optional[str]:
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        logger.warning("Spotify credentials not configured for client token")
        return None
    auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    auth_base64 = str(base64.b64encode(auth_string.encode("utf-8")), "utf-8")
    url = "https://accounts.spotify.com/api/token"
    headers = {"Authorization": f"Basic {auth_base64}", "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "client_credentials"}
    try:
        async with httpx.AsyncClient(timeout=10.0) as http_client: # Standard timeout for this
            response = await http_client.post(url, headers=headers, data=data)
            response.raise_for_status()
        return response.json().get("access_token")
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error getting Spotify client token: {e.response.status_code} - {e.response.text if e.response else 'No text'}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Request Error getting Spotify client token: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error getting Spotify client token: {e}", exc_info=True)
        return None

# SYNC version - run in thread if called from async
def get_user_spotify_token(user_id: int, code: str) -> Optional[Dict]:
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI:
        logger.warning(f"Spotify OAuth credentials not configured for user {user_id}")
        return None
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}'.encode()).decode()}",
        "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "authorization_code", "code": code, "redirect_uri": SPOTIFY_REDIRECT_URI}
    try:
        logger.info(f"Attempting to get user Spotify token for user {user_id}.")
        response = requests.post(url, headers=headers, data=data, timeout=15) # Sync request timeout
        response.raise_for_status()
        token_data = response.json()
        token_data["expires_at"] = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600))).timestamp()
        logger.info(f"Successfully obtained user Spotify token for user {user_id}.")
        return token_data
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP error ({e.response.status_code}) getting user Spotify token for {user_id}: {e.response.text if e.response else 'No body'}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"RequestException getting user Spotify token for {user_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error in get_user_spotify_token for {user_id}: {e}", exc_info=True)
        return None

async def refresh_spotify_token_async(user_id: int) -> Optional[str]:
    # ... (refresh_spotify_token_async - assumed okay from previous iterations) ...
    context = user_contexts.get(user_id, {})
    refresh_token_val = context.get("spotify", {}).get("refresh_token")
    if not refresh_token_val:
        logger.warning(f"No refresh token found for user {user_id} to refresh.")
        return None
    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {base64.b64encode(f'{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}'.encode()).decode()}",
        "Content-Type": "application/x-www-form-urlencoded"}
    data = {"grant_type": "refresh_token", "refresh_token": refresh_token_val}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(url, headers=headers, data=data)
            response.raise_for_status()
        token_data = response.json()
        expires_at = (datetime.now(pytz.UTC) + timedelta(seconds=token_data.get("expires_in", 3600))).timestamp()
        user_contexts.setdefault(user_id, {}).setdefault("spotify", {}).update({
            "access_token": token_data.get("access_token"),
            "refresh_token": token_data.get("refresh_token", refresh_token_val), 
            "expires_at": expires_at})
        logger.info(f"Successfully refreshed Spotify token for user {user_id}.")
        return token_data.get("access_token")
    except httpx.HTTPStatusError as e:
        if e.response and e.response.status_code == 400:
            logger.error(f"Invalid refresh token for user {user_id}: {e.response.text if e.response else ''}. Clearing Spotify context.")
            if user_id in user_contexts and "spotify" in user_contexts[user_id]: user_contexts[user_id]["spotify"] = {}
            return None
        logger.error(f"HTTP error refreshing Spotify token for {user_id}: {e.response.status_code if e.response else 'Unknown status'} - {e.response.text if e.response else ''}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Request error refreshing Spotify token for {user_id}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error refreshing Spotify token for {user_id}: {e}", exc_info=True)
        return None

async def get_user_spotify_data_async(user_id: int, endpoint: str) -> List[Dict]: # Return List, not Optional[List]
    context = user_contexts.get(user_id, {})
    spotify_data = context.get("spotify", {})
    access_token = spotify_data.get("access_token")
    expires_at = spotify_data.get("expires_at")
    if not access_token or (expires_at and datetime.now(pytz.UTC).timestamp() > expires_at):
        access_token = await refresh_spotify_token_async(user_id)
        if not access_token: return [] 
    url = f"https://api.spotify.com/v1/me/{endpoint}"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"limit": 10} # Can be tuned
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
        return response.json().get("items", [])
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error ({e.response.status_code if e.response else 'N/A'}) fetching Spotify user data ({endpoint}) for {user_id}: {e.response.text if e.response else ''}")
        return []
    except httpx.RequestError as e:
        logger.error(f"Request Error fetching Spotify user data ({endpoint}) for {user_id}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching Spotify user data for {user_id} ({endpoint}): {e}", exc_info=True)
        return []

async def get_user_spotify_playlists_async(user_id: int) -> List[Dict]:
    # ... (Similar to get_user_spotify_data_async) ...
    context = user_contexts.get(user_id, {})
    spotify_data = context.get("spotify", {})
    access_token = spotify_data.get("access_token")
    expires_at = spotify_data.get("expires_at")
    if not access_token or (expires_at and datetime.now(pytz.UTC).timestamp() > expires_at):
        access_token = await refresh_spotify_token_async(user_id)
        if not access_token: return []
    url = "https://api.spotify.com/v1/me/playlists"
    headers = {"Authorization": f"Bearer {access_token}"}
    params = {"limit": 10}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
        return response.json().get("items", [])
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP Error ({e.response.status_code if e.response else 'N/A'}) fetching Spotify playlists for {user_id}: {e.response.text if e.response else ''}")
        return []
    except httpx.RequestError as e:
        logger.error(f"Request Error fetching Spotify playlists for {user_id}: {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching Spotify playlists for {user_id}: {e}", exc_info=True)
        return []

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=2, max=10))
async def get_spotify_recommendations_async(token: str, seed_tracks: List[str], limit: int = 5) -> List[Dict]:
    if not token or not seed_tracks:
        logger.warning("No token or seed tracks for Spotify recommendations.")
        return []
    # Ensure seed_tracks are unique and take up to 5 (Spotify API limit for seeds)
    unique_seed_tracks = list(set(seed_tracks))[:5]
    if not unique_seed_tracks:
        logger.warning("Empty unique_seed_tracks list for recommendations.")
        return []
    url = "https://api.spotify.com/v1/recommendations"
    headers = {"Authorization": f"Bearer {token}"}
    params = {"seed_tracks": ",".join(unique_seed_tracks), "limit": limit}
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()
        return response.json().get("tracks", [])
    except httpx.HTTPStatusError as http_error:
        logger.warning(f"Spotify recommendations failed for seeds: {unique_seed_tracks}, status: {http_error.response.status_code if http_error.response else 'N/A'}, response: {http_error.response.text if http_error.response else ''}")
        return []
    except httpx.RequestError as req_error:
        logger.error(f"Request Error getting Spotify recommendations: {req_error}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in get_spotify_recommendations_async: {e}", exc_info=True)
        return []


@lru_cache(maxsize=100) # Increased cache size
def get_lyrics(song_title: str, artist_name: Optional[str] = None) -> str:
    # ... (get_lyrics as previously provided, seems fine)
    if not genius: return "Lyrics service (Genius) is not configured or available."
    try:
        song_title = sanitize_input(song_title)
        artist_name = sanitize_input(artist_name) if artist_name else None
        logger.info(f"Searching lyrics for: '{song_title}' by '{artist_name or 'Unknown Artist'}'")
        if artist_name: song = genius.search_song(song_title, artist_name)
        else: song = genius.search_song(song_title)
        if song and song.lyrics:
            lyrics = song.lyrics
            lyrics = re.sub(r'\[.*?\]\s*\n?', '', lyrics) # Remove [Chorus] etc. more cleanly
            lyrics = re.sub(r'\d*EmbedShare URLCopyEmbedCopy', '', lyrics, flags=re.IGNORECASE)
            lyrics = re.sub(r'.*?Lyrics(\[.*?\])?', '', lyrics, count=1).strip() # Remove "SongX Lyrics" prefix
            lyrics = re.sub(r'\d+\s*Contributors\S*Lyrics', '', lyrics, flags=re.IGNORECASE).strip() # Remove "X ContributorsSong Lyrics"
            lyrics = re.sub(r'You might also like', '', lyrics, flags=re.IGNORECASE).strip() # Remove "You might also like"
            lyrics = re.sub(r'\n{3,}', '\n\n', lyrics).strip() # Normalize newlines
            return lyrics if lyrics else "Lyrics found but are empty after cleaning."
        return f"Lyrics not found for '{song_title}' by '{artist_name or 'any artist'}'."
    except requests.exceptions.Timeout:
        logger.error(f"Timeout fetching lyrics from Genius for '{song_title}'.")
        return "Sorry, the lyrics search timed out. Please try again."
    except Exception as e:
        logger.error(f"Error fetching lyrics from Genius for '{song_title}': {e}", exc_info=True)
        return "Sorry, an error occurred while fetching lyrics."

# ... (is_valid_youtube_url, sanitize_filename, download_youtube_audio, search_youtube (cached) as previously) ...
def is_valid_youtube_url(url: str) -> bool:
    if not url: return False
    patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=',
        r'(https?://)?youtu\.be/',
        r'(https?://)?(www\.)?youtube\.com/shorts/'
    ]
    return any(re.search(pattern, url) for pattern in patterns)

def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    return sanitized[:100] # Limit length for safety

def download_youtube_audio(url: str) -> Dict[str, Any]:
    video_id_match = re.search(r'(?:v=|/|\.be/)([0-9A-Za-z_-]{11})', url)
    if not video_id_match:
        logger.warning(f"Could not extract video ID from YouTube URL: {url}")
        return {"success": False, "error": "Invalid YouTube URL or video ID"}
    
    title_placeholder = f"youtube_{video_id_match.group(1)}" # Placeholder if title fetch fails

    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio[abr<=128]/bestaudio/best', # Added /best as a fallback
        'outtmpl': os.path.join(DOWNLOAD_DIR, '%(title)s.%(ext)s'), # Use os.path.join
        'quiet': True, 'no_warnings': True, 'noplaylist': True,
        'postprocessor_args': ['-acodec', 'copy'], # May not always work if re-encoding needed
        'prefer_ffmpeg': False, 
        'max_filesize': 50 * 1024 * 1024, # 50MB
        'retries': 3, # yt-dlp internal retries
        'fragment_retries': 3,
        'skip_unavailable_fragments': True,
        # 'verbose': True, # Uncomment for detailed yt-dlp logging during development
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            logger.info(f"Attempting to extract info for YouTube URL: {url}")
            info = ydl.extract_info(url, download=False)
            if not info: 
                logger.error(f"Could not extract video information for {url}")
                return {"success": False, "error": "Could not extract video information"}
            
            title = sanitize_filename(info.get('title', title_placeholder))
            artist = sanitize_filename(info.get('artist', info.get('uploader', 'Unknown Artist')))
            
            # Update outtmpl with the sanitized title to prevent issues if title had problematic chars
            # that ydl_opts placeholder replacement handles differently.
            # This requires careful handling as ydl creates filenames.
            # Simpler: let ydl create it, then find it.

            logger.info(f"Starting download for: {title} from {url}")
            ydl.download([url]) # Actual download
            
            # Find the downloaded file (more robustly)
            audio_path = None
            # yt-dlp often names files like "Title-VIDEOID.ext" if title has special chars
            # or directly uses the title. We need to search for it.
            # This logic might need refinement based on yt-dlp's exact naming.
            # A common way is to let ydl return the filename or find the newest file matching expected extensions.
            
            # For now, try matching the title we expect
            for ext in ['m4a', 'webm', 'mp3', 'opus', 'ogg']: # Common audio extensions
                # yt-dlp might append the video ID for uniqueness if title is generic or problematic
                potential_path_simple_title = os.path.join(DOWNLOAD_DIR, f"{title}.{ext}")
                # Construct a pattern that might include the video ID yt-dlp might add
                # This part is tricky, as yt-dlp's sanitization can be complex.
                # For now, prioritize the simple title match.

                if os.path.exists(potential_path_simple_title):
                    audio_path = potential_path_simple_title
                    logger.info(f"Found downloaded file at: {audio_path}")
                    break
            
            if not audio_path: # Fallback: search directory for newest file of expected type
                logger.warning(f"Could not find downloaded file with simple title '{title}'. Searching dir...")
                files_in_dir = [os.path.join(DOWNLOAD_DIR, f) for f in os.listdir(DOWNLOAD_DIR)]
                audio_files = [f for f in files_in_dir if os.path.isfile(f) and f.lower().endswith(('.m4a', '.mp3', '.webm', '.opus', '.ogg'))]
                if audio_files:
                    audio_files.sort(key=os.path.getmtime, reverse=True) # Sort by modification time
                    # Heuristic: assume the newest audio file is the one we just downloaded,
                    # if its title (or part of it) matches.
                    # This could be fragile if multiple downloads happen nearly simultaneously by the system.
                    # A better approach might involve using yt-dlp's hook to get the exact filename.
                    audio_path = audio_files[0] 
                    logger.info(f"Found newest audio file as likely download: {audio_path}")


            if not audio_path or not os.path.exists(audio_path):
                logger.error(f"Downloaded file not found for '{title}' in {DOWNLOAD_DIR}. URL: {url}")
                return {"success": False, "error": "Downloaded file not found after download."}
            
            file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
            if file_size_mb > 50:
                logger.error(f"File '{audio_path}' too large: {file_size_mb:.2f} MB. Max 50 MB.")
                if os.path.exists(audio_path): os.remove(audio_path)
                return {"success": False, "error": "File too large for Telegram (max 50 MB)"}
            
            logger.info(f"Successfully downloaded and verified: {audio_path}")
            return {"success": True, "title": title, "artist": artist, 
                    "thumbnail_url": info.get('thumbnail', ''), 
                    "duration": info.get('duration', 0), "audio_path": audio_path}

    except yt_dlp.utils.MaxFilesizeError:
        logger.error(f"Max filesize exceeded during download for {url}.")
        return {"success": False, "error": "File is too large (exceeded 50MB during download)."}
    except yt_dlp.utils.DownloadError as e:
        # More specific DownloadError messages
        err_str = str(e).lower()
        if "video unavailable" in err_str:
            logger.warning(f"YouTube video unavailable: {url}. Error: {e}")
            return {"success": False, "error": "Video is unavailable."}
        if "private video" in err_str:
            logger.warning(f"YouTube video is private: {url}. Error: {e}")
            return {"success": False, "error": "Video is private."}
        if "age restricted" in err_str: # yt-dlp might sometimes struggle with these
            logger.warning(f"YouTube video is age-restricted: {url}. Error: {e}")
            return {"success": False, "error": "Video is age-restricted and could not be accessed."}
        logger.error(f"YouTube download error for {url}: {e}")
        return {"success": False, "error": "Failed to download video content."} # Generic yt-dlp download error
    except Exception as e:
        logger.error(f"Unexpected error downloading YouTube audio for {url}: {e}", exc_info=True)
        return {"success": False, "error": "An unexpected error occurred during download."}


@lru_cache(maxsize=150) # Increased cache
def search_youtube(query: str, max_results: int = 5) -> List[Dict]:
    # ... (search_youtube as previously) ...
    query = sanitize_input(query)
    logger.info(f"Searching YouTube for: '{query}' (max_results={max_results})")
    try:
        ydl_opts = {
            'quiet': True, 'no_warnings': True, 'extract_flat': 'discard_in_playlist', # More efficient for search
            'default_search': f'ytsearch{max_results}', # Search N results directly
            'format': 'bestaudio/best', # Request audio format info if available
            'noplaylist': True,
            # 'dump_single_json': True, # Alternative to extract_info with direct JSON, but extract_info is fine
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # info = ydl.extract_info(query, download=False) # Using default_search now
            info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)


            if not info or 'entries' not in info or not info['entries']:
                logger.warning(f"No YouTube search results found for query: '{query}'")
                return []
            
            results = []
            for entry in info['entries']:
                if entry and entry.get('id'): # Basic validation: entry and ID must exist
                    results.append({
                        'title': entry.get('title', 'Unknown Title'),
                        'url': entry.get('webpage_url') or entry.get('url') or f"https://www.youtube.com/watch?v={entry['id']}",
                        'thumbnail': entry.get('thumbnail', ''),
                        'uploader': entry.get('uploader', entry.get('channel', 'Unknown Artist')),
                        'duration': entry.get('duration', 0), # Duration in seconds
                        'id': entry['id']
                    })
            logger.info(f"Found {len(results)} YouTube results for '{query}'")
            return results
    except Exception as e:
        logger.error(f"Error searching YouTube for '{query}': {e}", exc_info=True)
        return []

# ... (AI Conversation functions: generate_chat_response, detect_music_in_message, is_music_request, analyze_conversation - as previously provided, seem fine) ...
async def generate_chat_response(user_id: int, message: str) -> str:
    if not client: return "I'm having trouble connecting to my AI service. Please try again later."
    message = sanitize_input(message)
    context_data = user_contexts.get(user_id, {"mood": None, "preferences": [], "conversation_history": [], "spotify": {}})
    messages_payload = [{"role": "system", "content": (
        "You are a friendly, empathetic music companion bot named MelodyMind. Your role is to: "
        "1. Have natural conversations about music and feelings. 2. Recommend songs based on mood and preferences. "
        "3. Provide emotional support through music. 4. Keep responses concise but warm (around 2-3 sentences). "
        "If the user has linked their Spotify account, use their listening history to personalize responses.")}]
    system_prompt_additions = []
    if context_data.get("mood"): system_prompt_additions.append(f"The user's current mood is: {context_data['mood']}.")
    if context_data.get("preferences"): system_prompt_additions.append(f"Their music preferences include: {', '.join(context_data['preferences'])}.")
    r_p_tracks = context_data.get("spotify", {}).get("recently_played", [])
    if r_p_tracks:
        valid_artists = [item["track"]["artists"][0]["name"] for item in r_p_tracks if item and isinstance(item.get("track"), dict) and isinstance(item["track"].get("artists"), list) and item["track"]["artists"] and isinstance(item["track"]["artists"][0], dict) and item["track"]["artists"][0].get("name")]
        if valid_artists: system_prompt_additions.append(f"They recently listened to artists: {', '.join(list(set(valid_artists))[:3])}.")
    if system_prompt_additions: messages_payload.append({"role": "system", "content": " ".join(system_prompt_additions)})
    context_data["conversation_history"] = context_data.get("conversation_history", [])[-20:]
    for hist in context_data["conversation_history"][-10:]: messages_payload.append(hist)
    messages_payload.append({"role": "user", "content": message})
    try:
        response = await client.chat.completions.create(model="gpt-3.5-turbo", messages=messages_payload, max_tokens=150, temperature=0.7)
        reply = response.choices[0].message.content.strip() if response.choices and response.choices[0].message else "I'm not sure how to respond to that right now."
        context_data["conversation_history"].extend([{"role": "user", "content": message}, {"role": "assistant", "content": reply}])
        context_data["conversation_history"] = context_data["conversation_history"][-20:]
        user_contexts[user_id] = context_data
        return reply
    except Exception as e:
        logger.error(f"Error generating chat response for user {user_id}: {e}", exc_info=True)
        return "I'm having trouble thinking right now. Let's talk about music instead!"

def detect_music_in_message(text: str) -> Optional[str]:
    patterns = [r'play (.*?)(?:by|from|$)', r'find (.*?)(?:by|from|$)',r'download (.*?)(?:by|from|$)', r'get (.*?)(?:by|from|$)',r'send me (.*?)(?:by|from|$)', r'i want to listen to (.*?)(?:by|from|$)',r'can you get (.*?)(?:by|from|$)', r'i need (.*?)(?:by|from|$)',r'find me (.*?)(?:by|from|$)', r'fetch (.*?)(?:by|from|$)',r'give me (.*?)(?:by|from|$)', r'send (.*?)(?:by|from|$)',r'song (.*?)(?:by|from|$)']
    keywords = ['music', 'song', 'track', 'tune', 'audio']
    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            song_title = match.group(1).strip()
            if not song_title: continue # Skip if query is empty after pattern match
            artist_match = re.search(r'by (.*?)(?:from|$)', text, re.IGNORECASE)
            if artist_match and artist_match.group(1).strip(): return f"{song_title} {artist_match.group(1).strip()}"
            return song_title
    if any(keyword in text_lower for keyword in keywords): return "AI_ANALYSIS_NEEDED"
    return None

async def is_music_request(user_id: int, message: str) -> Dict:
    if not client: return {"is_music_request": False, "song_query": None}
    try:
        response = await client.chat.completions.create(model="gpt-3.5-turbo",messages=[{"role": "system", "content": "You are an AI that determines if a message is requesting a song or music. If it is, extract the song/artist as 'song_query'. Respond in JSON with 'is_music_request' (boolean) and 'song_query' (string|null)."},{"role": "user", "content": f"Is this message asking for a song or music? If yes, what song/artist? Message: '{message}'"}],max_tokens=100, temperature=0.2, response_format={"type": "json_object"})
        content_str = response.choices[0].message.content
        if not content_str: logger.warning("AI (is_music_request) response empty"); return {"is_music_request": False, "song_query": None}
        try:
            result = json.loads(content_str)
            if not isinstance(result, dict): logger.warning(f"AI (is_music_request) response not dict: {result}"); return {"is_music_request": False, "song_query": None}
        except json.JSONDecodeError as jde: logger.error(f"Failed to decode JSON (is_music_request): {jde}. Response: {content_str[:200]}"); return {"is_music_request": False, "song_query": None}
        is_req = result.get("is_music_request", False)
        if isinstance(is_req, str): is_req = is_req.lower() in ("yes", "true", "1")
        s_query = result.get("song_query") or result.get("song", "") or result.get("artist", "") or result.get("query", "")
        return {"is_music_request": bool(is_req), "song_query": s_query.strip() if s_query else None}
    except Exception as e:
        logger.error(f"Error in is_music_request for user {user_id}: {e}", exc_info=True)
        return {"is_music_request": False, "song_query": None}

async def analyze_conversation(user_id: int) -> Dict:
    default_return = {"genres": [], "artists": [], "mood": None}
    if not client: return default_return
    user_context_data = user_contexts.get(user_id, {})
    current_mood = user_context_data.get("mood")
    current_prefs = user_context_data.get("preferences", [])
    default_return.update({"mood": current_mood, "genres": current_prefs})
    conv_history = user_context_data.get("conversation_history", [])
    spotify_context = user_context_data.get("spotify", {})
    if len(conv_history) < 2 and not spotify_context.get("recently_played") and not spotify_context.get("top_tracks"): return default_return
    conv_text_short = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conv_history[-10:]]) 
    spotify_summary_parts = []
    if spotify_context.get("recently_played"):
        rp_tracks = [f"{item['track']['name']} by {item['track']['artists'][0]['name']}" for item in spotify_context.get("recently_played", [])[:3] if item and item.get("track") and item["track"].get("artists") and item["track"]["artists"][0].get("name")]
        if rp_tracks: spotify_summary_parts.append("Recently played: " + ", ".join(rp_tracks))
    if spotify_context.get("top_tracks"):
        tt_tracks = [f"{item['name']} by {item['artists'][0]['name']}" for item in spotify_context.get("top_tracks", [])[:3] if item and item.get("artists") and item["artists"][0].get("name")]
        if tt_tracks: spotify_summary_parts.append("Top tracks: " + ", ".join(tt_tracks))
    spotify_data_summary = ". ".join(spotify_summary_parts)
    prompt_content = f"User's current mood: {current_mood if current_mood else 'Not set'}. User's stated preferences: {', '.join(current_prefs) if current_prefs else 'None'}.\nConversation history:\n{conv_text_short}\n\nSpotify listening data summary:\n{spotify_data_summary if spotify_data_summary else 'None'}"
    try:
        response = await client.chat.completions.create(model="gpt-3.5-turbo",messages=[{"role": "system", "content": "Analyze conversation and Spotify data to infer user's current musical mood, preferred genres, and liked artists. Respond in JSON with 'mood' (string), 'genres' (list of strings), 'artists' (list of strings). Prioritize recent signals."},{"role": "user", "content": prompt_content}],max_tokens=150, temperature=0.3, response_format={"type": "json_object"})
        content_str = response.choices[0].message.content
        if not content_str: return default_return
        try: result = json.loads(content_str); assert isinstance(result, dict)
        except (json.JSONDecodeError, AssertionError): logger.warning(f"AI (analyze_conversation) bad JSON: {content_str[:100]}"); return default_return
        inferred_mood = result.get("mood") or current_mood
        inferred_genres_raw = result.get("genres", []); inferred_genres = [g.strip().lower() for g in inferred_genres_raw if isinstance(g, str)] if isinstance(inferred_genres_raw, list) else []
        final_genres = list(dict.fromkeys(inferred_genres + current_prefs))[:3] 
        inferred_artists_raw = result.get("artists", []); final_artists = [a.strip() for a in inferred_artists_raw if isinstance(a, str)] if isinstance(inferred_artists_raw, list) else []
        final_artists = list(dict.fromkeys(final_artists))[:3] 
        if inferred_mood and inferred_mood != current_mood: user_contexts.setdefault(user_id,{})["mood"] = inferred_mood
        if final_genres and set(final_genres) != set(current_prefs): user_contexts.setdefault(user_id,{})["preferences"] = final_genres
        return {"genres": final_genres, "artists": final_artists, "mood": inferred_mood}
    except Exception as e:
        logger.error(f"Error in analyze_conversation for user {user_id}: {e}", exc_info=True)
        return default_return

# ==================== TELEGRAM HANDLERS ====================
# ... (start, help_command as previously) ...
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_text(f"Hi {user.first_name}! 👋 I'm MelodyMind, your Music Healing Companion.\n\n"
        "I can: 🎵 Download music 📜 Find lyrics 💿 Recommend music "
        "💬 Chat about music & feelings 🔗 Link Spotify for personalized experience.\n\n"
        "Try /help for commands, or just chat!")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = ("🎶 <b>MelodyMind Commands</b> 🎶\n"
        "/start - Welcome\n/help - This message\n"
        "/download [YouTube URL] or just send a link\n"
        "/autodownload [song name] - Search & DL 1st result\n"
        "/search [song name] - Show YouTube search options\n"
        "/lyrics [song name / artist - song]\n"
        "/recommend - Personalized music recommendations\n"
        "/mood - Set your current mood\n"
        "/link_spotify - Connect your Spotify\n"
        "/clear - Clear my memory of our chat\n\n"
        "<b>Or just chat naturally!</b> Examples:\n"
        "- \"I'm feeling sad, suggest some songs.\"\n"
        "- \"Play Shape of You by Ed Sheeran\"\n"
        "- \"What are the lyrics to Bohemian Rhapsody?\"")
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

download_lock = asyncio.Lock()
async def download_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    url = None
    if context.args: url = " ".join(context.args)
    elif update.message and update.message.text:
        valid_urls = [word for word in update.message.text.split() if is_valid_youtube_url(word)]
        if valid_urls: url = valid_urls[0]
    if not url or not is_valid_youtube_url(url):
        await update.message.reply_text("❌ Please provide a valid YouTube URL."); return
    user_id = update.effective_user.id
    async with download_lock:
        if user_id in active_downloads: await update.message.reply_text("⚠️ Download in progress. Please wait."); return
        active_downloads.add(user_id)
    status_msg = await update.message.reply_text("⏳ Starting download...")
    try:
        await status_msg.edit_text("🔍 Fetching video info...")
        dl_result = await asyncio.to_thread(download_youtube_audio, url)
        if not dl_result["success"]: await status_msg.edit_text(f"❌ Download failed: {dl_result['error']}"); raise Exception("Download failed")
        await status_msg.edit_text(f"✅ Downloaded: {dl_result['title']}\n⏳ Sending file...")
        
        @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1.5, min=5, max=45))
        async def send_audio_with_timeout_and_retry_dm():
            logger.info(f"Attempting to send audio (download_music): {dl_result['title']}")
            with open(dl_result["audio_path"], 'rb') as audio_f:
                await update.message.reply_audio(
                    audio=audio_f, title=dl_result["title"][:64],
                    performer=dl_result.get("artist", "Unknown Artist")[:64], caption=f"🎵 {dl_result['title']}",
                    read_timeout=90.0, write_timeout=90.0)
        
        await send_audio_with_timeout_and_retry_dm()
        message_sent_successfully = True
        
    except Exception as e:
        logger.error(f"Error in download_music for {url}: {e}", exc_info=True)
        if status_msg and not str(e) == "Download failed": # Only edit if not already edited for dl failure
             try: await status_msg.edit_text(f"❌ An error occurred during download/send: {str(e)[:100]}")
             except: pass # Ignore if status_msg cannot be edited
        message_sent_successfully = False # Mark as failed for cleanup
    finally:
        if 'dl_result' in locals() and dl_result.get("success") and os.path.exists(dl_result["audio_path"]):
            try: os.remove(dl_result["audio_path"]); logger.info(f"Deleted temp file: {dl_result['audio_path']}")
            except Exception as e_del: logger.error(f"Error deleting file {dl_result['audio_path']}: {e_del}")
        
        if 'message_sent_successfully' in locals() and message_sent_successfully and status_msg:
            try: await status_msg.delete()
            except Exception: pass # If status message deletion fails, it's not critical

        async with download_lock: # Always release lock
            if user_id in active_downloads: active_downloads.remove(user_id)

async def link_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (link_spotify - no changes needed from previous version) ...
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET or not SPOTIFY_REDIRECT_URI:
        await update.message.reply_text("Sorry, Spotify linking is not available (misconfigured).")
        return ConversationHandler.END
    user_id = update.effective_user.id
    auth_url = (f"https://accounts.spotify.com/authorize?client_id={SPOTIFY_CLIENT_ID}"
                f"&response_type=code&redirect_uri={SPOTIFY_REDIRECT_URI}"
                f"&scope=user-read-recently-played%20user-top-read%20playlist-read-private" 
                f"&state={user_id}") 
    keyboard = [[InlineKeyboardButton("🔗 Link Spotify", url=auth_url)],
                [InlineKeyboardButton("Cancel", callback_data="cancel_spotify")]]
    await update.message.reply_text(
        "🔗 Let's link your Spotify! \n1. Click below. \n2. Log in & authorize. \n3. Copy the code from the page. \n4. Send the code back to me here.",
        reply_markup=InlineKeyboardMarkup(keyboard))
    return SPOTIFY_CODE

# THIS IS THE FIXED spotify_code_handler
async def spotify_code_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    user_id = update.effective_user.id
    code = None
    if update.message and update.message.text:
        message_text = update.message.text.strip()
        if not message_text.startswith('/'): # Regular text could be a code
            code = message_text
        # Allow /spotify_code <CODE> even when in conversation state
        elif message_text.startswith('/spotify_code') and context.args: 
            code = context.args[0]
    
    if not code:
        await update.message.reply_text(
            "Please send the Spotify authorization code you received. "
            "You can just paste the code directly, or use /spotify_code <code>."
        )
        return SPOTIFY_CODE # Stay in this state to re-prompt for code

    logger.info(f"Received Spotify code from user {user_id}: {code[:10]}...") # Log partial code
    
    # Run the synchronous get_user_spotify_token in a separate thread
    token_data = await asyncio.to_thread(get_user_spotify_token, user_id, code) 

    if not token_data or not token_data.get("access_token"):
        await update.message.reply_text(
            "❌ Failed to link Spotify. The code might be invalid or expired. "
            "Please try /link_spotify again to get a new link and code."
        )
        return ConversationHandler.END # MODIFIED: End the conversation cleanly on failure.

    # If successful:
    user_contexts.setdefault(user_id, {}).setdefault("spotify", {}).update({
        "access_token": token_data.get("access_token"),
        "refresh_token": token_data.get("refresh_token"),
        "expires_at": token_data.get("expires_at")
    })
    
    # Fetch initial data
    logger.info(f"Spotify successfully linked for user {user_id}. Fetching initial data.")
    recently_played = await get_user_spotify_data_async(user_id, "player/recently-played")
    if recently_played is not None: 
        user_contexts.setdefault(user_id, {}).setdefault("spotify", {})["recently_played"] = recently_played

    await update.message.reply_text(
        "✅ Spotify account linked successfully! 🎉\n"
        "I can now use your listening history to recommend music. Try /recommend to get started!"
    )
    return ConversationHandler.END # Correct: End the conversation on success.

async def spotify_code_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (spotify_code_command - no changes needed) ...
    if not context.args:
        await update.message.reply_text("Please provide the code: /spotify_code <code>")
        return
    # This will effectively call the spotify_code_handler logic if the conversation isn't active,
    # or be handled by the ConversationHandler if it is.
    # To be absolutely sure it doesn't interfere if a convo IS active, we might make it
    # explicitly NOT enter a convo if it's just this command.
    # However, current setup reuses it if ConversationHandler picks it up.
    # If outside a convo, it needs its own logic or a way to pass to a generic code handler.
    # For simplicity now, this assumes spotify_code_handler can be called stand-alone
    # or PTB will route it correctly based on ConversationHandler states.
    
    # To ensure it uses the same logic when no conversation is active:
    temp_convo_active = bool(context.user_data.get(ConversationHandler.CONVERSATION_KEY)) # Check if in any convo state from this handler

    if temp_convo_active: # If in SPOTIFY_CODE state, it will be handled by the convo's MessageHandler or CommandHandler
        # Let the ConversationHandler deal with it
        # Or we could force a state transition here if complex logic is needed
        pass # Should be handled by the ConversationHandler
    else: # No active Spotify conversation, handle as a standalone command
        mock_message = update.message # Create a pseudo-update to pass to spotify_code_handler
        mock_message.text = f"/spotify_code {context.args[0]}" # Ensure context.args[0] is the code
        
        # We are calling it as if it's a normal handler.
        # Since spotify_code_handler expects to return a state for ConversationHandler,
        # we might need a wrapper or a separate non-convo handler if used standalone often.
        # For now, let's assume this use case is less common and just log it.
        logger.info(f"/spotify_code command used outside conversation by {update.effective_user.id}")
        # The original `spotify_code_handler` will reply and handle.
        # It does not require a return value here as it's not driving a ConversationHandler state transition.
        await spotify_code_handler(update, context) # Call it, its return value will be ignored here


async def cancel_spotify(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (cancel_spotify - no changes needed) ...
    query = update.callback_query
    await query.answer()
    await query.edit_message_text("Spotify linking cancelled. /link_spotify to try again.")
    return ConversationHandler.END

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (search_command - no changes needed) ...
    if not context.args:
        await update.message.reply_text("What to search? /search Shape of You Ed Sheeran")
        return
    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Searching YouTube for: '{query}'...")
    results = search_youtube(query) 
    if status_msg: await status_msg.delete()
    await send_search_results(update, query, results)

async def auto_download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (auto_download_command - no changes needed) ...
    if not context.args:
        await update.message.reply_text("What song? /autodownload Shape of You Ed Sheeran")
        return
    query = " ".join(context.args)
    await auto_download_first_result(update, context, query)

async def provide_generic_recommendations(update: Update, mood: str) -> None:
    # ... (provide_generic_recommendations - no changes needed) ...
    mood_recommendations = {
        "happy": ["Walking on Sunshine - Katrina & The Waves", "Happy - Pharrell Williams"],
        "sad": ["Someone Like You - Adele", "Fix You - Coldplay"],
        "energetic": ["Eye of the Tiger - Survivor", "Don't Stop Me Now - Queen"],
        "relaxed": ["Weightless - Marconi Union", "Clair de Lune - Claude Debussy"],
        "focused": ["The Four Seasons - Vivaldi", "Time - Hans Zimmer"],
        "nostalgic": ["Yesterday - The Beatles", "Vienna - Billy Joel"]
    } 
    recommendations = mood_recommendations.get(mood.lower(), mood_recommendations["happy"])
    response = f"🎵 <b>Some {mood.capitalize()} music ideas:</b>\n" + "\n".join([f"{i+1}. {track}" for i, track in enumerate(recommendations)])
    response += "\n\n💡 <i>Send a YouTube link to download!</i>"
    if update.callback_query and update.callback_query.message: 
        try: await update.callback_query.message.edit_text(response, parse_mode=ParseMode.HTML)
        except Exception as e: # If edit fails (e.g. message too old, or not changed)
            logger.warning(f"Failed to edit message for generic recommendations: {e}")
            await update.callback_query.message.reply_text(response, parse_mode=ParseMode.HTML) # Send new
    elif update.message:
        await update.message.reply_text(response, parse_mode=ParseMode.HTML)


async def set_mood(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (set_mood - no changes needed) ...
    keyboard = [
        [InlineKeyboardButton("Happy 😊", callback_data="mood_happy"), InlineKeyboardButton("Sad 😢", callback_data="mood_sad")],
        [InlineKeyboardButton("Energetic 💪", callback_data="mood_energetic"), InlineKeyboardButton("Relaxed 😌", callback_data="mood_relaxed")],
        [InlineKeyboardButton("Focused 🧠", callback_data="mood_focused"), InlineKeyboardButton("Nostalgic 🕰️", callback_data="mood_nostalgic")],
    ]
    await update.message.reply_text("How are you feeling today?", reply_markup=InlineKeyboardMarkup(keyboard))
    return MOOD

async def enhanced_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> Union[int, None]:
    # ... (enhanced_button_handler - minor async wrap for download_youtube_audio) ...
    query = update.callback_query
    await query.answer()
    data = query.data
    user_id = query.from_user.id
    logger.debug(f"Callback: {data} for user {user_id}")

    if data.startswith("mood_"):
        mood = data.split("_")[1]
        user_contexts.setdefault(user_id, {}).update({"mood": mood, "preferences": user_contexts.get(user_id, {}).get("preferences", [])})
        keyboard_genres = [
            [InlineKeyboardButton("Pop", callback_data="pref_pop"), InlineKeyboardButton("Rock", callback_data="pref_rock"), InlineKeyboardButton("Hip-Hop", callback_data="pref_hiphop")],
            [InlineKeyboardButton("Classical", callback_data="pref_classical"), InlineKeyboardButton("Electronic", callback_data="pref_electronic"), InlineKeyboardButton("Jazz", callback_data="pref_jazz")],
            [InlineKeyboardButton("Skip", callback_data="pref_skip")],]
        await query.edit_message_text(f"Got it! You're feeling {mood}. 🎶\n\nAny specific music genre preference?", reply_markup=InlineKeyboardMarkup(keyboard_genres))
        return PREFERENCE
    elif data.startswith("pref_"):
        preference = data.split("_")[1]
        current_prefs = user_contexts.setdefault(user_id, {}).setdefault("preferences", [])
        if preference != "skip" and preference not in current_prefs : current_prefs.append(preference) # Append unique
        user_contexts[user_id]["preferences"] = current_prefs[:3] # Keep max 3
        await query.edit_message_text("Great! Try /recommend, /download, /lyrics, or just chat!")
        return ConversationHandler.END
    elif data.startswith("download_") or data.startswith("auto_download_"):
        video_id = data.split("_")[-1] 
        if not re.match(r'^[0-9A-Za-z_-]{11}$', video_id):
            logger.error(f"Invalid YouTube video ID from callback: {video_id}")
            await query.edit_message_text("❌ Invalid video ID. Try another song.")
            return None
        
        url = f"https://www.youtube.com/watch?v={video_id}"
        await query.edit_message_text(f"⏳ Starting download for yt.be/{video_id}...")
        
        async with download_lock:
            if user_id in active_downloads:
                await query.edit_message_text("⚠️ You already have a download in progress. Please wait.")
                return None
            active_downloads.add(user_id)
        try:
            result = await asyncio.to_thread(download_youtube_audio, url) 
            if not result["success"]:
                await query.edit_message_text(f"❌ Download failed: {result['error']}")
                return None # Handled in finally
            await query.edit_message_text(f"✅ Downloaded: {result['title']}\n⏳ Sending file...")
            
            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
            async def send_audio_with_retry_cb():
                async with httpx.AsyncClient(timeout=60.0) as temp_client:
                    original_bot_client = context.bot.request._client
                    context.bot.request._client = temp_client
                    try:
                        with open(result["audio_path"], 'rb') as audio_file_cb:
                            return await context.bot.send_audio(
                                chat_id=query.message.chat_id, audio=audio_file_cb,
                                title=result["title"][:64],
                                performer=result["artist"][:64] if result.get("artist") else "Unknown Artist",
                                caption=f"🎵 {result['title']}")
                    finally: context.bot.request._client = original_bot_client
            
            await send_audio_with_retry_cb()
            
            @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=5))
            async def edit_final_message():
                 await query.edit_message_text(f"✅ Download complete: {result['title']}")

            if os.path.exists(result["audio_path"]):
                try: os.remove(result["audio_path"]); logger.info(f"Deleted file: {result['audio_path']}")
                except Exception as e_del: logger.error(f"Error deleting file {result['audio_path']}: {e_del}")
            await edit_final_message()
        except Exception as e_btn:
            logger.error(f"Error in button download for video {video_id}: {e_btn}", exc_info=True)
            try: await query.edit_message_text(f"❌ Error during download: {str(e_btn)[:150]}") # Shorter error
            except Exception: pass 
        finally:
            async with download_lock:
                if user_id in active_downloads: active_downloads.remove(user_id)
        return None
    elif data.startswith("show_options_"):
        search_query = data.split("show_options_")[1]
        results = search_youtube(search_query) 
        if not results: await query.edit_message_text(f"Sorry, no YouTube songs for '{search_query}'."); return None
        keyboard_show = []
        for i, res_item in enumerate(results[:5]):
            if not res_item.get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', res_item.get('id','')): continue # Check ID
            if not res_item.get('title'): continue # Skip if no title
            mins, secs = divmod(int(res_item.get('duration', 0)), 60)
            dur_str = f" [{mins}:{secs:02d}]" if res_item.get('duration', 0) > 0 else ""
            title_disp = res_item['title'][:37] + "..." if len(res_item['title']) > 40 else res_item['title']
            keyboard_show.append([InlineKeyboardButton(f"{i+1}. {title_disp}{dur_str}", callback_data=f"download_{res_item['id']}")])
        if not keyboard_show: await query.edit_message_text(f"No valid YouTube results for '{search_query}'."); return None
        keyboard_show.append([InlineKeyboardButton("Cancel", callback_data="cancel_search")])
        await query.edit_message_text(f"🔎 Results for '{search_query}':\nClick to download:", reply_markup=InlineKeyboardMarkup(keyboard_show))
        return None
    elif data == "cancel_search": await query.edit_message_text("❌ Search cancelled."); return None
    elif data == "cancel_spotify": await query.edit_message_text("❌ Spotify linking cancelled."); return ConversationHandler.END
    return None 

async def handle_error(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (handle_error - no changes needed) ...
    logger.error(f"Update {update} caused error {context.error}", exc_info=True)
    effective_chat = update.effective_chat
    if not effective_chat: # Should not happen with most updates handled
        logger.error("handle_error called with an update without an effective_chat")
        return

    error_message = "Sorry, something went wrong. Please try again later."
    try:
        if update.effective_message: # Prefer replying to the original message if possible
            await update.effective_message.reply_text(error_message)
        else: # Fallback to sending a new message to the chat
            await context.bot.send_message(chat_id=effective_chat.id, text=error_message)
    except (TimedOut, NetworkError) as e_reply:
        logger.error(f"Failed to send error message to user {effective_chat.id}: {e_reply}")
        # Further fallback or just log if even this fails
    except Exception as e_send:
        logger.error(f"Unexpected error when trying to send error message to {effective_chat.id}: {e_send}")

async def enhanced_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (enhanced_handle_message - minor async/sync handling improvements) ...
    if not update.message or not update.message.text : return 
    user_id = update.effective_user.id
    text = sanitize_input(update.message.text)
    logger.debug(f"Sanitized message from {user_id}: {text[:50]}...")

    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
    async def reply_with_retry(msg_text, reply_markup=None):
        async with httpx.AsyncClient(timeout=20.0) as temp_http_client:
            original_bot_client = context.bot.request._client
            context.bot.request._client = temp_http_client
            try: return await update.message.reply_text(msg_text, reply_markup=reply_markup)
            finally: context.bot.request._client = original_bot_client
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
    async def delete_message_with_retry(message_to_delete):
        if not message_to_delete: return # Guard against None
        async with httpx.AsyncClient(timeout=10.0) as temp_http_client_del: 
            original_bot_client_del = context.bot.request._client
            context.bot.request._client = temp_http_client_del
            try: return await message_to_delete.delete()
            finally: context.bot.request._client = original_bot_client_del
    try:
        if is_valid_youtube_url(text):
            await download_music(update, context) 
            return

        detected_song_query = detect_music_in_message(text)
        if detected_song_query:
            if detected_song_query == "AI_ANALYSIS_NEEDED":
                ai_song_detection = await is_music_request(user_id, text)
                detected_song_query = ai_song_detection.get("song_query") if ai_song_detection.get("is_music_request") else None
            if detected_song_query:
                status_msg_search = await reply_with_retry(f"🔍 Searching YouTube for: '{detected_song_query}'...")
                yt_results = await asyncio.to_thread(search_youtube, detected_song_query) # Run sync search in thread
                if status_msg_search: await delete_message_with_retry(status_msg_search)
                
                if not yt_results or not yt_results[0].get('id') or not re.match(r'^[0-9A-Za-z_-]{11}$', yt_results[0].get('id','')):
                    await reply_with_retry(f"Sorry, no valid YouTube songs found for '{detected_song_query}'.")
                    return
                first_yt_result = yt_results[0]
                keyboard_confirm = [
                    [InlineKeyboardButton("✅ Yes, download it", callback_data=f"auto_download_{first_yt_result['id']}")],
                    [InlineKeyboardButton("👀 Show me options", callback_data=f"show_options_{detected_song_query}")],
                    [InlineKeyboardButton("❌ No, cancel", callback_data="cancel_search")]]
                uploader_name = first_yt_result.get('uploader', 'Unknown Artist') # Default for uploader
                await reply_with_retry(f"Found '{first_yt_result['title']}' by {uploader_name}.\nDownload this?", reply_markup=InlineKeyboardMarkup(keyboard_confirm))
                return
        
        lower_text = text.lower()
        if any(p in lower_text for p in ["lyrics", "words to", "song that goes"]):
            song_query_lyrics = text
            # More robustly remove phrases only if they appear as distinct words/phrases
            phrases_to_remove = ["lyrics of", "lyrics for", "lyrics to", "words to", "what are the lyrics to", "what's the song that goes", "lyrics"]
            for p_lyric in phrases_to_remove:
                # Use regex to ensure it's a whole word/phrase match and case-insensitive
                song_query_lyrics = re.sub(rf'(?i)\b{re.escape(p_lyric)}\b\s*', '', song_query_lyrics, count=1).strip()

            if not song_query_lyrics.strip(): # If removing phrases left it empty
                 await reply_with_retry("Please specify which song's lyrics you want after the phrase (e.g., 'lyrics Bohemian Rhapsody').")
                 return

            context.args = [song_query_lyrics] 
            await get_lyrics_command(update, context)
            return
        
        ai_chat_response = await generate_chat_response(user_id, text)
        await reply_with_retry(ai_chat_response)

    except (TimedOut, NetworkError) as net_err:
        logger.error(f"Network error in handle_message: {net_err}", exc_info=True)
        try: await update.message.reply_text("Sorry, network hiccup. Please try again.")
        except: pass
    except Exception as main_handler_err:
        logger.error(f"Error in enhanced_handle_message: {main_handler_err}", exc_info=True)
        try: await update.message.reply_text("Oops, something went sideways. Try again?")
        except: pass


async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # ... (clear_history - no changes needed) ...
    user_id = update.effective_user.id
    if user_id in user_contexts and "conversation_history" in user_contexts[user_id]:
        user_contexts[user_id]["conversation_history"] = []
        await update.message.reply_text("✅ My memory of our chat has been cleared.")
    else:
        await update.message.reply_text("You don't have any saved conversation history with me yet.")

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    # ... (cancel - no changes needed) ...
    if update.message:
        await update.message.reply_text("No problem! Feel free to chat or use commands anytime.")
    elif update.callback_query and update.callback_query.message: # Check if message exists
        try:
            await update.callback_query.edit_message_text("Action cancelled.")
        except Exception as e: # Handle cases where edit might fail (e.g. message too old)
            logger.warning(f"Failed to edit message on cancel callback: {e}")
            await update.callback_query.message.reply_text("Action cancelled.")

    return ConversationHandler.END
# ==================== CLEANUP & SIGNAL HANDLING ====================
# ... (cleanup_downloads, signal_handler - no changes) ...
def cleanup_downloads() -> None:
    try:
        if os.path.exists(DOWNLOAD_DIR):
            for file_name in os.listdir(DOWNLOAD_DIR):
                file_path = os.path.join(DOWNLOAD_DIR, file_name)
                if os.path.isfile(file_path):
                    try: os.remove(file_path)
                    except Exception as e_del_file: logger.error(f"Failed to delete {file_path}: {e_del_file}")
            logger.info("Cleaned up download directory.")
    except Exception as e: logger.error(f"Error cleaning up downloads: {e}")

def signal_handler(sig, frame) -> None:
    logger.info(f"Received signal {sig}, cleaning up and exiting...")
    cleanup_downloads() 
    # Perform any other necessary cleanup before exiting
    # For asyncio applications, ensure tasks are cancelled or awaited if needed
    # However, for a hard signal like SIGINT/SIGTERM, immediate exit after cleanup is often the goal.
    sys.exit(0)


# ==================== MAIN FUNCTION ====================
def main() -> None:
    # ... (main function - handler registration - ensure pattern for cancel_spotify is specific)
    required_env_vars = ["TELEGRAM_TOKEN", "OPENAI_API_KEY", "SPOTIFY_CLIENT_ID", "SPOTIFY_CLIENT_SECRET", "GENIUS_ACCESS_TOKEN"]
    if any(not os.getenv(var) for var in required_env_vars):
        missing = [var for var in required_env_vars if not os.getenv(var)]
        logger.error(f"FATAL: Missing environment variables: {', '.join(missing)}. Exiting.")
        sys.exit(1)
    if SPOTIFY_REDIRECT_URI == "https://your-callback-url.com":
        logger.warning("SPOTIFY_REDIRECT_URI is default. Spotify OAuth might fail.")
   
    application = Application.builder().token(TOKEN).build()

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
        states={SPOTIFY_CODE: [
            MessageHandler(filters.TEXT & ~filters.COMMAND, spotify_code_handler),
            CommandHandler("spotify_code", spotify_code_handler), 
            CallbackQueryHandler(cancel_spotify, pattern="^cancel_spotify$") # Specific pattern
        ]},
        fallbacks=[CommandHandler("cancel", cancel), CallbackQueryHandler(cancel, pattern="^cancel$")])
    application.add_handler(spotify_conv_handler)

    mood_conv_handler = ConversationHandler(
        entry_points=[CommandHandler("mood", set_mood)],
        states={
            MOOD: [CallbackQueryHandler(enhanced_button_handler, pattern="^mood_")],
            PREFERENCE: [CallbackQueryHandler(enhanced_button_handler, pattern="^pref_")]
        },
        fallbacks=[CommandHandler("cancel", cancel), CallbackQueryHandler(cancel, pattern="^cancel$")])
    application.add_handler(mood_conv_handler)
    
    # More specific pattern for general callbacks to avoid clashes
    application.add_handler(CallbackQueryHandler(enhanced_button_handler, pattern="^(download_|auto_download_|show_options_|cancel_search$)")) 
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, enhanced_handle_message))
    
    application.add_error_handler(handle_error)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_downloads) 

    logger.info("🚀 Enhanced MelodyMind Bot is starting polling... 🎶")
    application.run_polling()

if __name__ == "__main__":
    main()