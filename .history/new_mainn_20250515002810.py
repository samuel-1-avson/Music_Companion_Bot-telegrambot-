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

# ==================== ENHANCED BUTTON HANDLER ====================

async def enhanced_button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks including download buttons."""
    query = update.callback_query
    await query.answer()
    
    # Get the callback data
    data = query.data
    user_id = query.from_user.id
    
    # Handle mood and preference buttons (existing code)
    if data.startswith("mood_") or data.startswith("pref_"):
        # Your existing button handler code here
        await button_handler(update, context)
        return
        
    # Handle download buttons
    if data.startswith("download_"):
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
                
    # Handle cancel button
    elif data == "cancel_search":
        await query.edit_message_text("❌ Search cancelled.")
        
        
          

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
        "/lyrics [song name] or [artist - song] - Get lyrics\n"
        "/recommend - Get music recommendations\n"
        "/mood - Set your current mood for better recommendations\n"
        "/clear - Clear your conversation history\n\n"
        "<b>Or just chat with me!</b> I can understand natural language requests like:\n"
        "- \"I'm feeling sad, what songs might help?\"\n"
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

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks."""
    query = update.callback_query
    await query.answer()
    
    # Get the callback data
    data = query.data
    user_id = query.from_user.id
    
    if data.startswith("mood_"):
        # Set mood from button selection
        mood = data.split("_")[1]
        
        # Initialize or update user context
        if user_id not in user_contexts:
            user_contexts[user_id] = {"mood": mood, "preferences": [], "conversation_history": []}
        else:
            user_contexts[user_id]["mood"] = mood
        
        # Ask for preferences
        keyboard = [
            [
                InlineKeyboardButton("Pop", callback_data="pref_pop"),
                InlineKeyboardButton("Rock", callback_data="pref_rock"),
                InlineKeyboardButton("Hip-Hop", callback_data="pref_hiphop"),
            ],
            [
                InlineKeyboardButton("Classical", callback_data="pref_classical"),
                InlineKeyboardButton("Electronic", callback_data="pref_electronic"),
                InlineKeyboardButton("Jazz", callback_data="pref_jazz"),
            ],
            [
                InlineKeyboardButton("Skip", callback_data="pref_skip"),
            ],
        ]
        
        await query.edit_message_text(
            f"Got it! You're feeling {mood}. 🎶\n\nAny specific music genre preference?",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    
    elif data.startswith("pref_"):
        # Set preference from button selection
        preference = data.split("_")[1]
        
        if user_id in user_contexts:
            if preference != "skip":
                user_contexts[user_id]["preferences"] = [preference]
        
        # Show recommendation options
        await query.edit_message_text(
            "Great! Now you can:\n"
            "/recommend - Get music recommendations\n"
            "/download - Download specific songs\n"
            "/lyrics - Find song lyrics\n\n"
            "Or just chat with me about music!"
        )

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

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all text messages that are not commands."""
    user_id = update.effective_user.id
    text = update.message.text
    
    # Check if it's a YouTube URL
    if is_valid_youtube_url(text):
        # User sent a YouTube link - process it for download
        context.args = []  # Reset args
        await download_music(update, context)
        return
    
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
    
    # Check for mood setting
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

# ==================== MAIN FUNCTION ====================

def main() -> None:
    """Start the bot."""
    # Create the Application and pass it your bot's token
    application = Application.builder().token(TOKEN).build()
    
    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("download", download_music))
    application.add_handler(CommandHandler("lyrics", get_lyrics_command))
    application.add_handler(CommandHandler("recommend", recommend_music))
    application.add_handler(CommandHandler("clear", clear_history))
    
    # Set up conversation handler for mood and preference setting
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("mood", set_mood)],
        states={
            MOOD: [CallbackQueryHandler(button_handler)],
            PREFERENCE: [CallbackQueryHandler(button_handler)],
            ACTION: [CallbackQueryHandler(button_handler)]
        },
        fallbacks=[CommandHandler("cancel", cancel)]
    )
    application.add_handler(conv_handler)
    
    # Handle callback queries
    application.add_handler(CallbackQueryHandler(button_handler))
    
    # General message handler
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Error handler
    application.add_error_handler(handle_error)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup function
    atexit.register(cleanup_downloads)
    
    # Start the Bot
    logger.info("Starting MelodyMind Bot")
    application.run_polling()

if __name__ == "__main__":
    main()