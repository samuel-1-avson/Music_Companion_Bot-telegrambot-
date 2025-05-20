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

# Telegram imports
from telegram import Update, InputFile, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes,
    filters, CallbackContext, ConversationHandler, CallbackQueryHandler
)

# API clients
import yt_dlp
from openai import OpenAI
import lyricsgenius

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Initialize Genius API client
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN) if GENIUS_ACCESS_TOKEN else None

# Conversation states
MOOD, PREFERENCE, ACTION = range(3)

# Track active downloads and user contexts
active_downloads = set()
user_contexts: Dict[int, Dict] = {}  # Stores user preferences and conversation context

# Directory for downloads
DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# ==================== SPOTIFY HELPER FUNCTIONS ====================

def get_spotify_token() -> Optional[str]:
    """Get Spotify access token using client credentials."""
    if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
        logger.warning("Spotify credentials not configured")
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
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        return response.json().get("access_token")
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting Spotify token: {e}")
        return None

def search_spotify_track(token: str, query: str) -> Optional[Dict]:
    """Search for a track on Spotify."""
    if not token:
        return None
        
    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "q": query,
        "type": "track",
        "limit": 1
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        items = response.json().get("tracks", {}).get("items", [])
        return items[0] if items else None
    except (requests.exceptions.RequestException, IndexError) as e:
        logger.error(f"Error searching Spotify track: {e}")
        return None

def get_spotify_recommendations(token: str, seed_tracks: List[str], limit: int = 5) -> List[Dict]:
    """Get track recommendations from Spotify."""
    if not token or not seed_tracks:
        return []
        
    url = "https://api.spotify.com/v1/recommendations"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "seed_tracks": ",".join(seed_tracks),
        "limit": limit
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        return response.json().get("tracks", [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error getting Spotify recommendations: {e}")
        return []


# ==================== YOUTUBE SEARCH FUNCTION ====================

def search_youtube(query: str, max_results: int = 5) -> List[Dict]:
    """Search YouTube for videos matching the query."""
    try:
        # Build a list of video options using yt-dlp
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,  # Don't download, just get info
            'default_search': 'ytsearch',  # Use YouTube search
            'format': 'bestaudio',
            'noplaylist': True,
            'playlist_items': f'1-{max_results}'  # Limit results
        }
        
        # Perform search
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            search_query = f"ytsearch{max_results}:{query}"
            info = ydl.extract_info(search_query, download=False)
            
            if not info or 'entries' not in info:
                return []
                
            # Format the results
            results = []
            for entry in info['entries']:
                if entry:
                    results.append({
                        'title': entry.get('title', 'Unknown Title'),
                        'url': entry.get('url') or f"https://www.youtube.com/watch?v={entry.get('id')}",
                        'thumbnail': entry.get('thumbnail', ''),
                        'uploader': entry.get('uploader', 'Unknown Artist'),
                        'duration': entry.get('duration', 0),
                        'id': entry.get('id', '')
                    })
            return results
    except Exception as e:
        logger.error(f"Error searching YouTube: {e}")
        return []

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
    # First replace any problematic characters with underscores
    sanitized = re.sub(r'[\\/*?:"<>|]', "_", filename)
    # Limit filename length to avoid path issues
    return sanitized[:100]  # Reasonable length limit

def download_youtube_audio(url: str) -> Dict[str, Any]:
    """Download audio from a YouTube video."""
    try:
        # Setup download options
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio/best',  # Prefer m4a format
            'outtmpl': f'{DOWNLOAD_DIR}/%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            'postprocessor_args': ['-acodec', 'copy'],  # No re-encoding
            'prefer_ffmpeg': False,  # Avoid ffmpeg if possible
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First extract info without downloading
            info = ydl.extract_info(url, download=False)
            if not info:
                return {"success": False, "error": "Could not extract video information"}
            
            # Sanitize the filename
            title = sanitize_filename(info.get('title', 'Unknown Title'))
            artist = info.get('artist', info.get('uploader', 'Unknown Artist'))
            
            # Download the audio
            ydl.download([url])
            
            # Find the downloaded file
            audio_path = None
            for ext in ['m4a', 'webm', 'mp3', 'opus']:
                potential_path = os.path.join(DOWNLOAD_DIR, f"{title}.{ext}")
                if os.path.exists(potential_path):
                    audio_path = potential_path
                    break
            
            if not audio_path:
                # Try finding a file that starts with the title
                for file in os.listdir(DOWNLOAD_DIR):
                    if file.startswith(title[:20]):
                        audio_path = os.path.join(DOWNLOAD_DIR, file)
                        break
            
            if not audio_path:
                return {"success": False, "error": "Downloaded file not found"}
            
            return {
                "success": True,
                "title": title,
                "artist": artist,
                "thumbnail_url": info.get('thumbnail', ''),
                "duration": info.get('duration', 0),
                "audio_path": audio_path
            }
                
    except Exception as e:
        logger.error(f"Error downloading YouTube audio: {e}")
        return {"success": False, "error": str(e)}

# ==================== AI CONVERSATION FUNCTIONS ====================

async def generate_chat_response(user_id: int, message: str) -> str:
    """Generate a conversational response using OpenAI."""
    if not client:
        return "I'm having trouble connecting to my AI service. Please try again later."
    
    # Get or initialize user context
    context = user_contexts.get(user_id, {
        "mood": None,
        "preferences": [],
        "conversation_history": []
    })
    
    # Prepare conversation history
    messages = [
        {"role": "system", "content": (
            "You are a friendly, empathetic music companion bot named MelodyMind. "
            "Your role is to: "
            "1. Have natural conversations about music and feelings "
            "2. Recommend songs based on mood and preferences "
            "3. Provide emotional support through music "
            "4. Keep responses concise but warm (around 2-3 sentences) "
            "You can suggest songs, discuss music therapy, or just chat about feelings. "
            "If the user sends a YouTube link, offer to download it for them."
        )}
    ]
    
    # Add user context if available
    if context.get("mood"):
        messages.append({
            "role": "system", 
            "content": f"The user's current mood is: {context['mood']}. "
                       f"Their music preferences include: {', '.join(context.get('preferences', ['various genres']))}"
        })
    
    # Add conversation history - keep last 5 exchanges for context
    for hist in context["conversation_history"][-10:]:  
        messages.append(hist)
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.7
        )
        reply = response.choices[0].message.content
        
        # Update conversation history
        context["conversation_history"].extend([
            {"role": "user", "content": message},
            {"role": "assistant", "content": reply}
        ])
        user_contexts[user_id] = context
        
        return reply
    except Exception as e:
        logger.error(f"Error generating chat response: {e}")
        return "I'm having trouble thinking right now. Let's talk about music instead!"

def get_lyrics(song_title: str, artist: Optional[str] = None) -> str:
    """Get lyrics for a song using Genius API."""
    if not genius:
        return "Sorry, lyrics service is not available right now."
        
    try:
        if artist:
            song = genius.search_song(song_title, artist)
        else:
            song = genius.search_song(song_title)
            
        if not song:
            return f"Couldn't find lyrics for '{song_title}'" + (f" by {artist}" if artist else "")
            
        # Clean up the lyrics to remove Genius annotations
        lyrics = song.lyrics
        lyrics = re.sub(r'\[.*?\]', '', lyrics)  # Remove [Verse], [Chorus], etc.
        lyrics = re.sub(r'\d+Embed$', '', lyrics)  # Remove Genius embed numbers
        lyrics = re.sub(r'Embed$', '', lyrics)  # Remove "Embed" text
        
        # Add song info header
        header = f"🎵 {song.title} by {song.artist} 🎵\n\n"
        
        return header + lyrics.strip()
    except Exception as e:
        logger.error(f"Error fetching lyrics: {e}")
        return "Sorry, I couldn't retrieve the lyrics right now."
    
    
# ==================== MUSIC DETECTION FUNCTION ====================

def detect_music_in_message(text: str) -> Optional[str]:
    """Detect if a message is asking for music without an explicit YouTube URL."""
    # Common patterns for song requests
    patterns = [
        r'play (.*?)(?:by|from|$)',  # "play Shape of You by Ed Sheeran"
        r'find (.*?)(?:by|from|$)',   # "find Shape of You by Ed Sheeran"
        r'download (.*?)(?:by|from|$)', # "download Shape of You by Ed Sheeran"
        r'get (.*?)(?:by|from|$)',    # "get Shape of You by Ed Sheeran"
        r'send me (.*?)(?:by|from|$)', # "send me Shape of You by Ed Sheeran"
        r'I want to listen to (.*?)(?:by|from|$)', # "I want to listen to Shape of You by Ed Sheeran"
        r'can you get (.*?)(?:by|from|$)', # "can you get Shape of You by Ed Sheeran"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            song_title = match.group(1).strip()
            
            # Extract artist if present
            artist_match = re.search(r'by (.*?)(?:from|$)', text, re.IGNORECASE)
            if artist_match:
                artist = artist_match.group(1).strip()
                return f"{song_title} {artist}"
            return song_title
            
    return None  


# ==================== INLINE KEYBOARD FOR SEARCH RESULTS ====================

async def send_search_results(update: Update, query: str, results: List[Dict]) -> None:
    """Send search results with inline keyboard buttons."""
    if not results:
        await update.message.reply_text(f"Sorry, I couldn't find any songs for '{query}'.")
        return
        
    # Create keyboard with search results
    keyboard = []
    for i, result in enumerate(results):
        # Format duration to minutes:seconds
        duration_str = ""
        if result.get('duration'):
            minutes = result['duration'] // 60
            seconds = result['duration'] % 60
            duration_str = f" [{minutes}:{seconds:02d}]"
            
        # Truncate title if too long
        title = result['title']
        if len(title) > 40:
            title = title[:37] + "..."
            
        # Button text: "1. Title by Uploader [duration]"
        button_text = f"{i+1}. {title}{duration_str}"
        
        # Callback data: download_video_id
        keyboard.append([InlineKeyboardButton(button_text, callback_data=f"download_{result['id']}")])
    
    # Add cancel button
    keyboard.append([InlineKeyboardButton("Cancel", callback_data="cancel_search")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        f"🔎 Search results for '{query}':\n\nClick on a song to download:",
        reply_markup=reply_markup
    )
    
    
    
    
    # Improved music detection function
def detect_music_in_message(text: str) -> Optional[str]:
    """Detect if a message is asking for music without an explicit YouTube URL."""
    # Enhanced patterns for song requests
    patterns = [
        r'play (.*?)(?:by|from|$)',  # "play Shape of You by Ed Sheeran"
        r'find (.*?)(?:by|from|$)',   # "find Shape of You by Ed Sheeran"
        r'download (.*?)(?:by|from|$)', # "download Shape of You by Ed Sheeran"
        r'get (.*?)(?:by|from|$)',    # "get Shape of You by Ed Sheeran"
        r'send me (.*?)(?:by|from|$)', # "send me Shape of You by Ed Sheeran"
        r'i want to listen to (.*?)(?:by|from|$)', # "I want to listen to Shape of You by Ed Sheeran"
        r'can you get (.*?)(?:by|from|$)', # "can you get Shape of You by Ed Sheeran"
        r'i need (.*?)(?:by|from|$)', # "I need Shape of You"
        r'find me (.*?)(?:by|from|$)', # "find me Shape of You"
        r'fetch (.*?)(?:by|from|$)', # "fetch Shape of You"
        r'give me (.*?)(?:by|from|$)', # "give me Shape of You" 
        r'send (.*?)(?:by|from|$)', # "send Shape of You"
        r'song (.*?)(?:by|from|$)', # "song Shape of You"
    ]
    
    # Additional keywords that might indicate a song request
    keywords = ['music', 'song', 'track', 'tune', 'audio']
    
    # First try to match explicit patterns
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            song_title = match.group(1).strip()
            
            # Extract artist if present
            artist_match = re.search(r'by (.*?)(?:from|$)', text, re.IGNORECASE)
            if artist_match:
                artist = artist_match.group(1).strip()
                return f"{song_title} {artist}"
            return song_title
    
    # If no pattern matched but contains music keywords, use AI to determine if it's a song request
    if any(keyword in text.lower() for keyword in keywords):
        return "AI_ANALYSIS_NEEDED"
            
    return None


# New function for auto-downloading the first result
async def auto_download_first_result(update: Update, context: ContextTypes.DEFAULT_TYPE, query: str) -> None:
    """Automatically download the first song result for a query."""
    user_id = update.effective_user.id
    
    # Check if another download is in progress for this user
    if user_id in active_downloads:
        await update.message.reply_text("⚠️ You already have a download in progress. Please wait.")
        return
    
    # Add user to active downloads set
    active_downloads.add(user_id)
    
    # Send initial status message
    status_msg = await update.message.reply_text(f"🔍 Searching for '{query}'...")
    
    try:
        # Search YouTube
        results = search_youtube(query, max_results=1)
        
        if not results:
            await status_msg.edit_text(f"❌ Couldn't find any results for '{query}'.")
            active_downloads.remove(user_id)
            return
        
        # Get the first result
        result = results[0]
        video_url = result["url"]
        
        # Update status message
        await status_msg.edit_text(f"✅ Found: {result['title']}\n⏳ Downloading...")
        
        # Download the audio
        download_result = download_youtube_audio(video_url)
        
        if not download_result["success"]:
            await status_msg.edit_text(f"❌ Download failed: {download_result['error']}")
            active_downloads.remove(user_id)
            return
        
        # Update status message
        await status_msg.edit_text(f"✅ Downloaded: {download_result['title']}\n⏳ Sending file...")
        
        # Send the audio file
        with open(download_result["audio_path"], 'rb') as audio:
            await update.message.reply_audio(
                audio=audio,
                title=download_result["title"][:64],  # Telegram title limit
                performer=download_result["artist"][:64] if download_result.get("artist") else "Unknown Artist",
                caption=f"🎵 {download_result['title']}"
            )
        
        # Clean up
        if os.path.exists(download_result["audio_path"]):
            try:
                os.remove(download_result["audio_path"])
                logger.info(f"Deleted file: {download_result['audio_path']}")
            except Exception as e:
                logger.error(f"Error deleting file: {e}")
        
        # Delete status message
        await status_msg.delete()
        
    except Exception as e:
        logger.error(f"Error in auto_download_first_result: {e}")
        await status_msg.edit_text(f"❌ Error: {str(e)[:200]}")
    finally:
        # Remove user from active downloads
        if user_id in active_downloads:
            active_downloads.remove(user_id)


# New auto download command for one-click downloading
async def auto_download_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /autodownload command to search and download the first result."""
    if not context.args:
        await update.message.reply_text(
            "Please specify what song you want. Example:\n"
            "/autodownload Shape of You Ed Sheeran"
        )
        return
    
    query = " ".join(context.args)
    await auto_download_first_result(update, context, query)


# Function to directly add buttons for auto-download
async def send_auto_download_options(update: Update, query: str, results: List[Dict]) -> None:
    """Send search results with buttons for auto-downloading."""
    if not results:
        await update.message.reply_text(f"Sorry, I couldn't find any songs for '{query}'.")
        return
        
    # Create keyboard with search results
    keyboard = []
    for i, result in enumerate(results[:3]):  # Limit to top 3 results
        # Format duration to minutes:seconds
        duration_str = ""
        if result.get('duration'):
            minutes = result['duration'] // 60
            seconds = result['duration'] % 60
            duration_str = f" [{minutes}:{seconds:02d}]"
            
        # Truncate title if too long
        title = result['title']
        if len(title) > 40:
            title = title[:37] + "..."
            
        # Button text: "Download: Title by Uploader [duration]"
        button_text = f"Download: {title}{duration_str}"
        
        # Callback data: download_video_id
        keyboard.append([InlineKeyboardButton(button_text, callback_data=f"download_{result['id']}")])
    
    # Add auto-download first result button
    keyboard.append([InlineKeyboardButton("🚀 Auto-download first result", callback_data=f"auto_download_{results[0]['id']}")])
    
    # Add cancel button
    keyboard.append([InlineKeyboardButton("Cancel", callback_data="cancel_search")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text(
        f"🔎 Found results for '{query}':\n\nClick to download:",
        reply_markup=reply_markup
    )
    
    
    
    
    
    

# ==================== ENHANCED BUTTON HANDLER ====================
async def enhanced_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks including download and auto-download buttons."""
    query = update.callback_query
    await query.answer()
    
    # Get the callback data
    data = query.data
    user_id = query.from_user.id
    
    # Handle mood and preference buttons
    if data.startswith("mood_") or data.startswith("pref_"):
        # Your existing button handler code here
        await button_handler(update, context)
        return
    
    # Handle auto-download buttons
    if data.startswith("auto_download_"):
        video_id = data.split("_")[2]  # Extract video ID
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Update message to show download is starting
        await query.edit_message_text(f"⏳ Starting automatic download...")
        
        # Check if another download is in progress for this user
        if user_id in active_downloads:
            await query.edit_message_text("⚠️ You already have a download in progress. Please wait.")
            return
        
        # Add user to active downloads set
        active_downloads.add(user_id)
        
        try:
            # Download the audio
            result = download_youtube_audio(url)
            
            if not result["success"]:
                await query.edit_message_text(f"❌ Download failed: {result['error']}")
                return
            
            # Update message
            await query.edit_message_text(f"✅ Downloaded: {result['title']}\n⏳ Sending file...")
            
            # Send the audio file
            with open(result["audio_path"], 'rb') as audio:
                await context.bot.send_audio(
                    chat_id=query.message.chat_id,
                    audio=audio,
                    title=result["title"][:64],  # Telegram title limit
                    performer=result["artist"][:64] if result.get("artist") else "Unknown Artist",
                    caption=f"🎵 {result['title']}"
                )
            
            # Clean up
            if os.path.exists(result["audio_path"]):
                try:
                    os.remove(result["audio_path"])
                    logger.info(f"Deleted file: {result['audio_path']}")
                except Exception as e:
                    logger.error(f"Error deleting file: {e}")
            
            # Update final message
            await query.edit_message_text(f"✅ Download complete: {result['title']}")
            
        except Exception as e:
            logger.error(f"Error in auto-download button handler: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)[:200]}")
        finally:
            # Remove user from active downloads
            if user_id in active_downloads:
                active_downloads.remove(user_id)
    
    # Handle regular download buttons
    elif data.startswith("download_"):
        video_id = data.split("_")[1]
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Update message to show download is starting
        await query.edit_message_text(f"⏳ Starting download for video ID: {video_id}...")
        
        # Check if another download is in progress for this user
        if user_id in active_downloads:
            await query.edit_message_text("⚠️ You already have a download in progress. Please wait.")
            return
        
        # Add user to active downloads set
        active_downloads.add(user_id)
        
        try:
            # Download the audio
            result = download_youtube_audio(url)
            
            if not result["success"]:
                await query.edit_message_text(f"❌ Download failed: {result['error']}")
                return
            
            # Update message
            await query.edit_message_text(f"✅ Downloaded: {result['title']}\n⏳ Sending file...")
            
            # Send the audio file
            with open(result["audio_path"], 'rb') as audio:
                await context.bot.send_audio(
                    chat_id=query.message.chat_id,
                    audio=audio,
                    title=result["title"][:64],  # Telegram title limit
                    performer=result["artist"][:64] if result.get("artist") else "Unknown Artist",
                    caption=f"🎵 {result['title']}"
                )
            
            # Clean up
            if os.path.exists(result["audio_path"]):
                try:
                    os.remove(result["audio_path"])
                    logger.info(f"Deleted file: {result['audio_path']}")
                except Exception as e:
                    logger.error(f"Error deleting file: {e}")
            
            # Update final message
            await query.edit_message_text(f"✅ Download complete: {result['title']}")
            
        except Exception as e:
            logger.error(f"Error in download button handler: {e}")
            await query.edit_message_text(f"❌ Error: {str(e)[:200]}")
        finally:
            # Remove user from active downloads
            if user_id in active_downloads:
                active_downloads.remove(user_id)
    
    # Handle show options button
    elif data.startswith("show_options_"):
        # Extract the search query
        search_query = data.split("show_options_")[1]
        
        # Search YouTube again
        results = search_youtube(search_query)
        
        if not results:
            await query.edit_message_text(f"Sorry, I couldn't find any songs for '{search_query}'.")
            return
        
        # Create keyboard with search results
        keyboard = []
        for i, result in enumerate(results[:5]):  # Limit to top 5 results
            # Format duration to minutes:seconds
            duration_str = ""
            if result.get('duration'):
                minutes = result['duration'] // 60
                seconds = result['duration'] % 60
                duration_str = f" [{minutes}:{seconds:02d}]"
                
            # Truncate title if too long
            title = result['title']
            if len(title) > 40:
                title = title[:37] + "..."
                
            # Button text: "1. Title by Uploader [duration]"
            button_text = f"{i+1}. {title}{duration_str}"
            
            # Callback data: download_video_id
            keyboard.append([InlineKeyboardButton(button_text, callback_data=f"download_{result['id']}")])
        
        # Add cancel button
        keyboard.append([InlineKeyboardButton("Cancel", callback_data="cancel_search")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text(
            f"🔎 Search results for '{search_query}':\n\nClick on a song to download:",
            reply_markup=reply_markup
        )
                
    # Handle cancel button
    elif data == "cancel_search":
        await query.edit_message_text("❌ Search cancelled.")
        
        
# ==================== ENHANCED MESSAGE HANDLER ====================

async def enhanced_handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Enhanced message handler with music detection and auto-download capabilities."""
    user_id = update.effective_user.id
    text = update.message.text
    
    # Check if it's a YouTube URL (existing functionality)
    if is_valid_youtube_url(text):
        # User sent a YouTube link - process it for download
        context.args = []  # Reset args
        await download_music(update, context)
        return
    
    # Check for music detection in message
    detected_song = detect_music_in_message(text)
    
    if detected_song:
        if detected_song == "AI_ANALYSIS_NEEDED":
            # Use AI to determine if this is a song request and extract the query
            ai_response = await is_music_request(user_id, text)
            if ai_response.get("is_music_request") and ai_response.get("song_query"):
                detected_song = ai_response.get("song_query")
            else:
                # Not a song request or couldn't determine the song
                detected_song = None
        
        if detected_song:
            # Let user know we're searching
            status_msg = await update.message.reply_text(f"🔍 Searching for: '{detected_song}'...")
            
            # Search YouTube
            results = search_youtube(detected_song)
            
            # Delete status message
            await status_msg.delete()
            
            if not results:
                await update.message.reply_text(f"Sorry, I couldn't find any songs for '{detected_song}'.")
                return
            
            # Ask user if they want to auto-download or choose from results
            keyboard = [
                [InlineKeyboardButton("✅ Yes, download it", callback_data=f"auto_download_{results[0]['id']}")],
                [InlineKeyboardButton("👀 Show me options", callback_data=f"show_options_{detected_song}")],
                [InlineKeyboardButton("❌ No, cancel", callback_data="cancel_search")]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            message = (
                f"I found '{results[0]['title']}' by {results[0]['uploader']}.\n\n"
                f"Would you like me to download this song for you?"
            )
            await update.message.reply_text(message, reply_markup=reply_markup)
            return
    
    # Handle the rest of the message processing (existing functionality)
    # Determine intent from message
    lower_text = text.lower()
    
    # Check for download requests with URL in text
    if ("download" in lower_text or "get this song" in lower_text) and any(domain in lower_text for domain in ['youtube.com', 'youtu.be']):
        urls = [word for word in text.split() if is_valid_youtube_url(word)]
        if urls:
            context.args = [urls[0]]
            await download_music(update, context)
            return
    
    # Check for lyrics requests
    if any(phrase in lower_text for phrase in ["lyrics", "words to", "what's the song that goes"]):
        # Try to extract song info
        song_query = text
        for phrase in ["lyrics", "words to", "what's the song that goes"]:
            song_query = song_query.replace(phrase, "")
        context.args = [song_query.strip()]
        await get_lyrics_command(update, context)
        return
    
    # Check for song search requests without specific pattern
    if any(word in lower_text for word in ["song", "music", "track", "listen", "audio"]):
        # Use AI to determine if this is a song request
        response = await is_music_request(user_id, text)
        if response.get("is_music_request") and response.get("song_query"):
            # Let user know we're searching
            status_msg = await update.message.reply_text(f"🔍 Searching for: '{response['song_query']}'...")
            
            # Search YouTube
            results = search_youtube(response['song_query'])
            
            # Delete status message
            await status_msg.delete()
            
            if not results:
                await update.message.reply_text(f"Sorry, I couldn't find any songs for '{response['song_query']}'.")
                return
                
            # Ask user if they want to auto-download the first result
            keyboard = [
                [InlineKeyboardButton("✅ Yes, download it", callback_data=f"auto_download_{results[0]['id']}")],
                [InlineKeyboardButton("👀 Show me options", callback_data=f"show_options_{response['song_query']}")],
                [InlineKeyboardButton("❌ No, cancel", callback_data="cancel_search")]
            ]
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            message = (
                f"I found '{results[0]['title']}' by {results[0]['uploader']}.\n\n"
                f"Would you like me to download this song for you?"
            )
            await update.message.reply_text(message, reply_markup=reply_markup)
            return
    
    # Check for mood setting (existing functionality)
    if "i'm feeling" in lower_text or "i feel" in lower_text:
        # Extract potential mood
        text_after_feeling = ""
        if "i'm feeling" in lower_text:
            text_after_feeling = lower_text.split("i'm feeling")[1].strip()
        elif "i feel" in lower_text:
            text_after_feeling = lower_text.split("i feel")[1].strip()
        
        # Extract first word as potential mood
        if text_after_feeling:
            mood = text_after_feeling.split()[0].strip('.,!?')
            
            # Initialize or update user context
            if user_id not in user_contexts:
                user_contexts[user_id] = {"mood": mood, "preferences": [], "conversation_history": []}
            else:
                user_contexts[user_id]["mood"] = mood
    
    # Send "thinking" indicator
    typing_msg = await update.message.reply_text("🎵 Thinking about music...")
    
    try:
        # Generate AI response
        response = await generate_chat_response(user_id, text)
        await update.message.reply_text(response)
    except Exception as e:
        logger.error(f"Error in handle_message: {e}")
        await update.message.reply_text("I'm having trouble responding right now. Try again later?")
    finally:
        # Delete the typing message
        await typing_msg.delete()        
        
# ==================== AI MUSIC REQUEST ANALYZER ====================

async def is_music_request(user_id: int, message: str) -> Dict:
    """Use AI to determine if a message is a music/song request."""
    if not client:
        return {"is_music_request": False}
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                    "You are an AI that determines if a message is requesting a song or music. "
                    "If it is, extract the song/artist the user wants to hear."
                },
                {"role": "user", "content": 
                    f"Is this message asking for a song or music? If yes, what song/artist? Message: '{message}'"
                }
            ],
            max_tokens=100,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        
        # Make sure the response has the expected structure
        if not isinstance(result, dict):
            return {"is_music_request": False}
            
        # Extract information
        is_request = result.get("is_music_request", False)
        if isinstance(is_request, str):
            is_request = is_request.lower() == "yes" or is_request.lower() == "true"
            
        song_query = result.get("song", "") or result.get("artist", "") or result.get("query", "")
        
        return {
            "is_music_request": bool(is_request),
            "song_query": song_query if song_query else None
        }
        
    except Exception as e:
        logger.error(f"Error in is_music_request: {e}")
        return {"is_music_request": False}
   
# ==================== SEARCH COMMAND ====================

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /search command to search for music."""
    if not context.args:
        await update.message.reply_text(
            "Please specify what you're looking for. Example:\n"
            "/search Shape of You Ed Sheeran"
        )
        return
    
    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Searching for: '{query}'...")
    
    # Search YouTube
    results = search_youtube(query)
    
    # Delete status message
    await status_msg.delete()
    
    # Send results with buttons
    await send_search_results(update, query, results)                 

# ==================== TELEGRAM BOT HANDLERS ====================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued."""
    user = update.effective_user
    welcome_msg = (
        f"Hi {user.first_name}! 👋 I'm MelodyMind, your Music Healing Companion.\n\n"
        "I can:\n"
        "🎵 Download music from YouTube links\n"
        "📜 Find lyrics for any song\n"
        "💿 Recommend music based on your mood\n"
        "💬 Chat about music and feelings\n\n"
        "Just send me a YouTube link or start chatting! Try /help for more details."
    )
    await update.message.reply_text(welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message when the command /help is issued."""
    help_text = (
        "🎶 <b>MelodyMind - Music Healing Companion</b> 🎶\n\n"
        "<b>Commands:</b>\n"
        "/start - Welcome message\n"
        "/help - This help message\n"
        "/download [YouTube URL] - Download music from YouTube\n"
        "/autodownload [song name] - Automatically search and download a song\n"
        "/search [song name] - Search for a song\n"
        "/lyrics [song name] or [artist - song] - Get lyrics\n"
        "/recommend - Get music recommendations\n"
        "/mood - Set your current mood for better recommendations\n"
        "/clear - Clear your conversation history\n\n"
        "<b>Or just chat with me!</b> I can understand natural language requests like:\n"
        "- \"I'm feeling sad, what songs might help?\"\n"
        "- \"Play Shape of You by Ed Sheeran\"\n"
        "- \"Get me the new Taylor Swift song\"\n"
        "- \"Download this song: [YouTube link]\"\n"
        "- \"What are the lyrics to Bohemian Rhapsody?\""
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

async def download_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download music from YouTube URL."""
    message_text = update.message.text
    
    # Extract URL from command or message
    if context.args:
        url = " ".join(context.args)
    else:
        # Try to extract URL from message text
        words = message_text.split()
        urls = [word for word in words if is_valid_youtube_url(word)]
        if urls:
            url = urls[0]
        else:
            await update.message.reply_text(
                "❌ Please provide a valid YouTube URL. Example:\n"
                "/download https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            )
            return
    
    if not is_valid_youtube_url(url):
        await update.message.reply_text("❌ Invalid YouTube URL. Please send a valid YouTube link.")
        return
    
    # Check if another download is in progress for this user
    user_id = update.effective_user.id
    if user_id in active_downloads:
        await update.message.reply_text("⚠️ You already have a download in progress. Please wait.")
        return
    
    # Add user to active downloads set
    active_downloads.add(user_id)
    
    # Send initial status message
    status_msg = await update.message.reply_text("⏳ Starting download...")
    
    try:
        # Update status message
        await status_msg.edit_text("🔍 Fetching video information...")
        
        # Download the audio
        result = download_youtube_audio(url)
        
        if not result["success"]:
            await status_msg.edit_text(f"❌ Download failed: {result['error']}")
            return
        
        # Update status message
        await status_msg.edit_text(f"✅ Downloaded: {result['title']}\n⏳ Sending file...")
        
        # Send the audio file
        with open(result["audio_path"], 'rb') as audio:
            await update.message.reply_audio(
                audio=audio,
                title=result["title"][:64],  # Telegram title limit
                performer=result["artist"][:64] if result.get("artist") else "Unknown Artist",
                caption=f"🎵 {result['title']}"
            )
        
        # Clean up
        if os.path.exists(result["audio_path"]):
            try:
                os.remove(result["audio_path"])
                logger.info(f"Deleted file: {result['audio_path']}")
            except Exception as e:
                logger.error(f"Error deleting file: {e}")
        
        # Delete status message
        await status_msg.delete()
        
    except Exception as e:
        logger.error(f"Error in download_music: {e}")
        await status_msg.edit_text(f"❌ Error: {str(e)[:200]}")
    finally:
        # Remove user from active downloads
        if user_id in active_downloads:
            active_downloads.remove(user_id)

async def get_lyrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle lyrics requests."""
    if not context.args:
        await update.message.reply_text(
            "Please specify a song. Examples:\n"
            "/lyrics Bohemian Rhapsody\n"
            "/lyrics Queen - Bohemian Rhapsody"
        )
        return
    
    query = " ".join(context.args)
    status_msg = await update.message.reply_text(f"🔍 Searching for lyrics: {query}")
    
    try:
        # Parse the query to extract artist and song
        artist = None
        song = query
        
        # Try different formats: "artist - song" or "song by artist"
        if " - " in query:
            parts = query.split(" - ", 1)
            artist, song = parts[0].strip(), parts[1].strip()
        elif " by " in query.lower():
            parts = query.lower().split(" by ", 1)
            song, artist = parts[0].strip(), parts[1].strip()
        
        # Get lyrics
        lyrics = get_lyrics(song, artist)
        
        # If lyrics are too long, split into multiple messages
        if len(lyrics) > 4000:
            # Send first part
            await status_msg.edit_text(lyrics[:4000] + "\n\n(Message continues in next part...)")
            
            # Send remaining parts
            remaining = lyrics[4000:]
            while remaining:
                part = remaining[:4000]
                remaining = remaining[4000:]
                await update.message.reply_text(
                    part + ("\n\n(Continued in next part...)" if remaining else "")
                )
        else:
            await status_msg.edit_text(lyrics)
            
    except Exception as e:
        logger.error(f"Error in get_lyrics_command: {e}")
        await status_msg.edit_text("Sorry, I couldn't find those lyrics.")

async def recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provide music recommendations based on user context."""
    user_id = update.effective_user.id
    context_data = user_contexts.get(user_id, {})
    
    if not context_data.get("mood"):
        # No mood set, ask user to set it
        keyboard = [
            [
                InlineKeyboardButton("Happy 😊", callback_data="mood_happy"),
                InlineKeyboardButton("Sad 😢", callback_data="mood_sad"),
            ],
            [
                InlineKeyboardButton("Energetic 💪", callback_data="mood_energetic"),
                InlineKeyboardButton("Relaxed 😌", callback_data="mood_relaxed"),
            ],
            [
                InlineKeyboardButton("Focused 🧠", callback_data="mood_focused"),
                InlineKeyboardButton("Nostalgic 🕰️", callback_data="mood_nostalgic"),
            ],
        ]
        
        await update.message.reply_text(
            "I'd love to recommend some music! First, how are you feeling today?",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
        return
    
    mood = context_data["mood"]
    await update.message.reply_text(f"🎧 Finding {mood} music recommendations for you...")
    
    try:
        # If Spotify API is not configured, provide generic recommendations
        if not SPOTIFY_CLIENT_ID or not SPOTIFY_CLIENT_SECRET:
            await provide_generic_recommendations(update, mood)
            return
            
        # Get Spotify token
        token = get_spotify_token()
        if not token:
            await provide_generic_recommendations(update, mood)
            return
        
        # Search for a seed track based on mood
        seed_query = f"{mood} music"
        if context_data.get("preferences"):
            # Include user preferences in search
            preferences = context_data["preferences"]
            if preferences:
                seed_query = f"{mood} {preferences[0]} music"
        
        seed_track = search_spotify_track(token, seed_query)
        if not seed_track:
            await provide_generic_recommendations(update, mood)
            return
        
        # Get recommendations
        recommendations = get_spotify_recommendations(token, [seed_track["id"]])
        if not recommendations:
            await provide_generic_recommendations(update, mood)
            return
        
        # Format recommendations
        response = f"🎵 <b>Recommended {mood} music:</b>\n\n"
        for i, track in enumerate(recommendations[:5], 1):
            artists = ", ".join(a["name"] for a in track["artists"])
            album = track.get("album", {}).get("name", "")
            
            response += f"{i}. <b>{track['name']}</b> by {artists}"
            if album:
                response += f" (from {album})"
            response += "\n"
        
        # Add YouTube search links
        response += "\n💡 <i>Send me a YouTube link of any song to download it!</i>"
        
        await update.message.reply_text(response, parse_mode=ParseMode.HTML)
        
    except Exception as e:
        logger.error(f"Error in recommend_music: {e}")
        await update.message.reply_text("I couldn't get recommendations right now. Maybe try again later?")

async def provide_generic_recommendations(update: Update, mood: str) -> None:
    """Provide generic recommendations when Spotify API is not available."""
    # Dictionary of mood-based recommendations
    mood_recommendations = {
        "happy": [
            "Walking on Sunshine - Katrina & The Waves",
            "Happy - Pharrell Williams",
            "Can't Stop the Feeling - Justin Timberlake",
            "Uptown Funk - Mark Ronson ft. Bruno Mars",
            "Good Vibrations - The Beach Boys"
        ],
        "sad": [
            "Someone Like You - Adele",
            "Fix You - Coldplay",
            "Everybody Hurts - R.E.M.",
            "Nothing Compares 2 U - Sinéad O'Connor",
            "Tears in Heaven - Eric Clapton"
        ],
        "energetic": [
            "Eye of the Tiger - Survivor",
            "Don't Stop Me Now - Queen", 
            "Thunderstruck - AC/DC",
            "Stronger - Kanye West",
            "Shake It Off - Taylor Swift"
        ],
        "relaxed": [
            "Weightless - Marconi Union",
            "Clair de Lune - Claude Debussy",
            "Watermark - Enya",
            "Breathe - Pink Floyd",
            "Gymnopedie No.1 - Erik Satie"
        ],
        "focused": [
            "The Four Seasons - Vivaldi",
            "Time - Hans Zimmer",
            "Intro - The xx",
            "Brain Waves - Alpha Waves",
            "Experience - Ludovico Einaudi"
        ],
        "nostalgic": [
            "Yesterday - The Beatles",
            "Viva la Vida - Coldplay",
            "Landslide - Fleetwood Mac",
            "Vienna - Billy Joel",
            "Time After Time - Cyndi Lauper"
        ]
    }
    
    # Default to happy if mood not found
    recommendations = mood_recommendations.get(mood.lower(), mood_recommendations["happy"])
    
    response = f"🎵 <b>Recommended {mood} music:</b>\n\n"
    for i, track in enumerate(recommendations, 1):
        response += f"{i}. {track}\n"
    
    response += "\n💡 <i>Send me a YouTube link of any song to download it!</i>"
    
    await update.message.reply_text(response, parse_mode=ParseMode.HTML)

async def set_mood(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start conversation to set user's mood."""
    keyboard = [
        [
            InlineKeyboardButton("Happy 😊", callback_data="mood_happy"),
            InlineKeyboardButton("Sad 😢", callback_data="mood_sad"),
        ],
        [
            InlineKeyboardButton("Energetic 💪", callback_data="mood_energetic"),
            InlineKeyboardButton("Relaxed 😌", callback_data="mood_relaxed"),
        ],
        [
            InlineKeyboardButton("Focused 🧠", callback_data="mood_focused"),
            InlineKeyboardButton("Nostalgic 🕰️", callback_data="mood_nostalgic"),
        ],
    ]
    
    await update.message.reply_text(
        "How are you feeling today? This will help me recommend better music.",
        reply_markup=InlineKeyboardMarkup(keyboard)
    )
    return MOOD

# async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Handle button callbacks."""
#     query = update.callback_query
#     await query.answer()
    
#     # Get the callback data
#     data = query.data
#     user_id = query.from_user.id
    
#     if data.startswith("mood_"):
#         # Set mood from button selection
#         mood = data.split("_")[1]
        
#         # Initialize or update user context
#         if user_id not in user_contexts:
#             user_contexts[user_id] = {"mood": mood, "preferences": [], "conversation_history": []}
#         else:
#             user_contexts[user_id]["mood"] = mood
        
#         # Ask for preferences
#         keyboard = [
#             [
#                 InlineKeyboardButton("Pop", callback_data="pref_pop"),
#                 InlineKeyboardButton("Rock", callback_data="pref_rock"),
#                 InlineKeyboardButton("Hip-Hop", callback_data="pref_hiphop"),
#             ],
#             [
#                 InlineKeyboardButton("Classical", callback_data="pref_classical"),
#                 InlineKeyboardButton("Electronic", callback_data="pref_electronic"),
#                 InlineKeyboardButton("Jazz", callback_data="pref_jazz"),
#             ],
#             [
#                 InlineKeyboardButton("Skip", callback_data="pref_skip"),
#             ],
#         ]
        
#         await query.edit_message_text(
#             f"Got it! You're feeling {mood}. 🎶\n\nAny specific music genre preference?",
#             reply_markup=InlineKeyboardMarkup(keyboard)
#         )
    
#     elif data.startswith("pref_"):
#         # Set preference from button selection
#         preference = data.split("_")[1]
        
#         if user_id in user_contexts:
#             if preference != "skip":
#                 user_contexts[user_id]["preferences"] = [preference]
        
#         # Show recommendation options
#         await query.edit_message_text(
#             "Great! Now you can:\n"
#             "/recommend - Get music recommendations\n"
#             "/download - Download specific songs\n"
#             "/lyrics - Find song lyrics\n\n"
#             "Or just chat with me about music!"
#         )

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear user conversation history."""
    user_id = update.effective_user.id
    
    if user_id in user_contexts:
        # Keep mood and preferences but clear conversation history
        user_contexts[user_id]["conversation_history"] = []
        await update.message.reply_text("✅ Your conversation history has been cleared.")
    else:
        await update.message.reply_text("You don't have any saved conversation history.")

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancel the conversation."""
    await update.message.reply_text("No problem! Feel free to chat or use commands anytime.")
    return ConversationHandler.END

# async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
#     """Handle all text messages that are not commands."""
#     user_id = update.effective_user.id
#     text = update.message.text
    
#     # Check if it's a YouTube URL
#     if is_valid_youtube_url(text):
#         # User sent a YouTube link - process it for download
#         context.args = []  # Reset args
#         await download_music(update, context)
#         return
    
#     # Determine intent from message
#     lower_text = text.lower()
    
#     # Check for download requests with URL in text
#     if ("download" in lower_text or "get this song" in lower_text) and any(domain in lower_text for domain in ['youtube.com', 'youtu.be']):
#         urls = [word for word in text.split() if is_valid_youtube_url(word)]
#         if urls:
#             context.args = [urls[0]]
#             await download_music(update, context)
#             return
    
#     # Check for lyrics requests
#     if any(phrase in lower_text for phrase in ["lyrics", "words to", "what's the song that goes"]):
#         # Try to extract song info
#         song_query = text
#         for phrase in ["lyrics", "words to", "what's the song that goes"]:
#             song_query = song_query.replace(phrase, "")
#         context.args = [song_query.strip()]
#         await get_lyrics_command(update, context)
#         return
    
#     # Check for mood setting
#     if "i'm feeling" in lower_text or "i feel" in lower_text:
#         # Extract potential mood
#         text_after_feeling = ""
#         if "i'm feeling" in lower_text:
#             text_after_feeling = lower_text.split("i'm feeling")[1].strip()
#         elif "i feel" in lower_text:
#             text_after_feeling = lower_text.split("i feel")[1].strip()
        
#         # Extract first word as potential mood
#         if text_after_feeling:
#             mood = text_after_feeling.split()[0].strip('.,!?')
            
#             # Initialize or update user context
#             if user_id not in user_contexts:
#                 user_contexts[user_id] = {"mood": mood, "preferences": [], "conversation_history": []}
#             else:
#                 user_contexts[user_id]["mood"] = mood
    
#     # Send "thinking" indicator
#     typing_msg = await update.message.reply_text("🎵 Thinking about music...")
    
#     try:
#         # Generate AI response
#         response = await generate_chat_response(user_id, text)
#         await update.message.reply_text(response)
#     except Exception as e:
#         logger.error(f"Error in handle_message: {e}")
#         await update.message.reply_text("I'm having trouble responding right now. Try again later?")
#     finally:
#         # Delete the typing message
#         await typing_msg.delete()
        
# The existing code is complete up to line 692
# Here's what needs to be added:

# ==================== ERROR HANDLING ====================

def handle_error(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors caused by updates."""
    logger.error(f"Update {update} caused error {context.error}")
    
    # Notify user
    if update and update.effective_message:
        update.effective_message.reply_text(
            "Sorry, something went wrong. Please try again later."
        )

# ==================== CLEANUP FUNCTIONS ====================

def cleanup_downloads() -> None:
    """Clean up any temporary files in the download directory."""
    try:
        if os.path.exists(DOWNLOAD_DIR):
            for file in os.listdir(DOWNLOAD_DIR):
                file_path = os.path.join(DOWNLOAD_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            logger.info("Cleaned up download directory")
    except Exception as e:
        logger.error(f"Error cleaning up downloads: {e}")

# ==================== SIGNAL HANDLERS ====================

def signal_handler(sig, frame) -> None:
    """Handle termination signals."""
    logger.info("Received termination signal, cleaning up...")
    cleanup_downloads()
    sys.exit(0)
    
    
# ==================== CONVERSATION ANALYSIS ====================

async def analyze_conversation(user_id: int) -> Dict:
    """Analyze the conversation history to improve recommendations."""
    if not client:
        return {"genres": [], "artists": [], "mood": None}
        
    # Get user context
    context = user_contexts.get(user_id, {
        "mood": None,
        "preferences": [],
        "conversation_history": []
    })
    
    # If not enough conversation history, return basic info
    if len(context.get("conversation_history", [])) < 4:  # Need at least 2 exchanges
        return {
            "genres": context.get("preferences", []),
            "artists": [],
            "mood": context.get("mood")
        }
    
    try:
        # Get the last 10 exchanges (or fewer if not available)
        history = context["conversation_history"][-20:]
        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": 
                    "You are an AI that analyzes conversations about music to extract preferences. "
                    "Extract: 1) Music genres mentioned, 2) Artists mentioned, and 3) User's mood."
                },
                {"role": "user", "content": 
                    f"Analyze this conversation and extract music preferences:\n\n{conversation_text}"
                }
            ],
            max_tokens=150,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        
        # Make sure we have the expected structure
        if not isinstance(result, dict):
            return {"genres": context.get("preferences", []), "artists": [], "mood": context.get("mood")}
            
        # Extract and format the results
        genres = result.get("genres", [])
        if isinstance(genres, str):
            genres = [g.strip() for g in genres.split(",")]
            
        artists = result.get("artists", [])
        if isinstance(artists, str):
            artists = [a.strip() for a in artists.split(",")]
            
        mood = result.get("mood")
        
        # Update user context with this analysis
        if genres:
            context["preferences"] = genres[:3]  # Keep top 3 genres
        if mood and not context.get("mood"):
            context["mood"] = mood
            
        # Save updated context
        user_contexts[user_id] = context
        
        return {
            "genres": genres,
            "artists": artists,
            "mood": mood
        }
        
    except Exception as e:
        logger.error(f"Error in analyze_conversation: {e}")
        return {"genres": context.get("preferences", []), "artists": [], "mood": context.get("mood")}

# ==================== SMART RECOMMENDATIONS ====================

async def smart_recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provide smarter music recommendations based on conversation analysis."""
    user_id = update.effective_user.id
    
    # Send initial message
    status_msg = await update.message.reply_text("🎧 Finding personalized music recommendations...")
    
    try:
        # Analyze conversation to get better context
        analysis = await analyze_conversation(user_id)
        
        # If no mood is set, ask user to set it
        if not analysis.get("mood"):
            await status_msg.delete()
            keyboard = [
                [
                    InlineKeyboardButton("Happy 😊", callback_data="mood_happy"),
                    InlineKeyboardButton("Sad 😢", callback_data="mood_sad"),
                ],
                [
                    InlineKeyboardButton("Energetic 💪", callback_data="mood_energetic"),
                    InlineKeyboardButton("Relaxed 😌", callback_data="mood_relaxed"),
                ],
                [
                    InlineKeyboardButton("Focused 🧠", callback_data="mood_focused"),
                    InlineKeyboardButton("Nostalgic 🕰️", callback_data="mood_nostalgic"),
                ],
            ]
            
            await update.message.reply_text(
                "I'd love to recommend some music! First, how are you feeling today?",
                reply_markup=InlineKeyboardMarkup(keyboard)
            )
            return
        
        mood = analysis.get("mood")
        genres = analysis.get("genres", [])
        artists = analysis.get("artists", [])
        
        # Build search query based on analysis
        search_query = mood if mood else "music"
        
        # Add genre if available
        if genres:
            search_query += f" {genres[0]} music"
        
        # Include artist if available
        if artists:
            search_query += f" like {artists[0]}"
            
        # Update status message
        await status_msg.edit_text(f"🔍 Searching for {search_query}...")
        
        # Get recommendations
        if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
            # Use Spotify if available
            token = get_spotify_token()
            if token:
                seed_track = search_spotify_track(token, search_query)
                if seed_track:
                    recommendations = get_spotify_recommendations(token, [seed_track["id"]])
                    if recommendations:
                        # Format recommendations
                        response = f"🎵 <b>Recommended music for you:</b>\n\n"
                        for i, track in enumerate(recommendations[:5], 1):
                            artists_text = ", ".join(a["name"] for a in track["artists"])
                            album = track.get("album", {}).get("name", "")
                            
                            response += f"{i}. <b>{track['name']}</b> by {artists_text}"
                            if album:
                                response += f" (from {album})"
                            response += "\n"
                        
                        # Add download option hint
                        response += "\n💡 <i>Send me the song name to download it!</i>"
                        
                        await status_msg.edit_text(response, parse_mode=ParseMode.HTML)
                        return
        
        # If Spotify fails or isn't available, use YouTube search
        results = search_youtube(search_query, max_results=5)
        
        if results:
            # Create response with search results
            response = f"🎵 <b>Recommended music for you:</b>\n\n"
            for i, result in enumerate(results, 1):
                duration_min = result.get('duration', 0) // 60
                duration_sec = result.get('duration', 0) % 60
                duration_str = f"[{duration_min}:{duration_sec:02d}]"
                
                response += f"{i}. <b>{result['title']}</b> - {result['uploader']} {duration_str}\n"
            
            # Create inline keyboard for direct download
            keyboard = []
            for result in results:
                button_text = f"Download: {result['title'][:30]}..." if len(result['title']) > 30 else f"Download: {result['title']}"
                keyboard.append([InlineKeyboardButton(button_text, callback_data=f"download_{result['id']}")])
            
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await status_msg.edit_text(response, parse_mode=ParseMode.HTML, reply_markup=reply_markup)
        else:
            # Fall back to generic recommendations
            await status_msg.delete()
            await provide_generic_recommendations(update, mood if mood else "happy")
            
    except Exception as e:
        logger.error(f"Error in smart_recommend_music: {e}")
        await status_msg.edit_text("I couldn't get personalized recommendations right now. Try again later?")

# ==================== UPDATED MAIN FUNCTION ====================    

# ==================== MAIN FUNCTION ====================

def main() -> None:
    """Start the enhanced bot."""
    # Create the Application and pass it your bot's token
    application = Application.builder().token(TOKEN).build()
    
    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("download", download_music))
    application.add_handler(CommandHandler("search", search_command))
    application.add_handler(CommandHandler("autodownload", auto_download_command))  # New auto-download command
    application.add_handler(CommandHandler("lyrics", get_lyrics_command))
    application.add_handler(CommandHandler("recommend", smart_recommend_music))
    application.add_handler(CommandHandler("clear", clear_history))
    
    # Set up conversation handler for mood and preference setting
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("mood", set_mood)],
        states={
            MOOD: [CallbackQueryHandler(enhanced_button_handler)],
            PREFERENCE: [CallbackQueryHandler(enhanced_button_handler)],
            ACTION: [CallbackQueryHandler(enhanced_button_handler)]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )
    application.add_handler(conv_handler)
    
    # Handle callback queries with enhanced handler
    application.add_handler(CallbackQueryHandler(enhanced_button_handler))
    
    # General message handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, enhanced_handle_message))
    
    # Error handler
    application.add_error_handler(handle_error)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup function
    atexit.register(cleanup_downloads)
    
    # Start the Bot
    logger.info("Starting Enhanced MelodyMind Bot with Auto-Download capabilities")
    application.run_polling()
if __name__ == "__main__":
    main()