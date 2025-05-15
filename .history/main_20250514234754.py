import os
import logging
import sys
import requests
import yt_dlp
from dotenv import load_dotenv
from telegram import Update, InputFile
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes,
    filters, CallbackContext, ConversationHandler
)
import pytz
import signal
import atexit
import openai
import lyricsgenius
import json
import base64
from typing import Dict, List, Optional
from openai import OpenAI
import os
import re
from typing import Optional
import yt_dlp
from telegram import Update
from telegram.ext import ContextTypes
# Load environment variables
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SPOTIFY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIFY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
GENIUS_ACCESS_TOKEN = os.getenv("GENIUS_ACCESS_TOKEN")


# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None



# Initialize APIs
openai.api_key = OPENAI_API_KEY
genius = lyricsgenius.Genius(GENIUS_ACCESS_TOKEN)

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', 
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Conversation states
MOOD, PREFERENCE, ACTION = range(3)

# Track active downloads and user contexts
active_downloads = set()
user_contexts: Dict[int, Dict] = {}  # Stores user preferences and conversation context

# Spotify Helper Functions
def get_spotify_token():
    """Get Spotify access token using client credentials."""
    auth_string = f"{SPOTIFY_CLIENT_ID}:{SPOTIFY_CLIENT_SECRET}"
    auth_bytes = auth_string.encode("utf-8")
    auth_base64 = str(base64.b64encode(auth_bytes), "utf-8")

    url = "https://accounts.spotify.com/api/token"
    headers = {
        "Authorization": f"Basic {auth_base64}",
        "Content-Type": "application/x-www-form-urlencoded"
    }
    data = {"grant_type": "client_credentials"}
    response = requests.post(url, headers=headers, data=data)
    return response.json().get("access_token")

def search_spotify_track(token: str, query: str):
    """Search for a track on Spotify."""
    url = "https://api.spotify.com/v1/search"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "q": query,
        "type": "track",
        "limit": 1
    }
    response = requests.get(url, headers=headers, params=params)
    return response.json().get("tracks", {}).get("items", [])[0] if response.status_code == 200 else None

def get_spotify_recommendations(token: str, seed_tracks: List[str], limit: int = 5):
    """Get track recommendations from Spotify."""
    url = "https://api.spotify.com/v1/recommendations"
    headers = {"Authorization": f"Bearer {token}"}
    params = {
        "seed_tracks": ",".join(seed_tracks),
        "limit": limit
    }
    response = requests.get(url, headers=headers, params=params)
    return response.json().get("tracks", []) if response.status_code == 200 else []

# YouTube Helper Functions
def is_valid_youtube_url(url: str) -> bool:
    """Check if the URL is a valid YouTube URL."""
    if not url:
        return False
    
    valid_domains = ['youtube.com', 'youtu.be', 'www.youtube.com']
    try:
        return any(domain in url for domain in valid_domains)
    except:
        return False

def download_youtube_audio(url: str, download_dir: str = "downloads") -> Dict:
    """Download audio from a YouTube video."""
    os.makedirs(download_dir, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio',
        'outtmpl': os.path.join(download_dir, '%(title)s.%(ext)s'),
        'quiet': True,
        'noplaylist': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if not info:
                return {"success": False, "error": "Failed to extract video info"}
            
            title = info.get('title', 'Unknown Title')
            artist = info.get('artist', info.get('uploader', 'Unknown Artist'))
            thumbnail_url = info.get('thumbnail', '')
            
            # Download audio
            ydl.download([url])
            
            # Find the downloaded file
            audio_path = os.path.join(download_dir, f"{title}.m4a")
            if not os.path.exists(audio_path):
                for file in os.listdir(download_dir):
                    if file.startswith(title[:20]) and file.endswith('.m4a'):
                        audio_path = os.path.join(download_dir, file)
                        break
            
            return {
                "success": True,
                "title": title,
                "artist": artist,
                "thumbnail_url": thumbnail_url,
                "audio_path": audio_path
            }
    except Exception as e:
        logger.error(f"Error downloading from YouTube: {e}")
        return {"success": False, "error": str(e)}

# AI Conversation Functions
from openai import OpenAI  # Add this import at the top

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Then update your generate_chat_response function:
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
            "You are a friendly, empathetic music companion bot. Your role is to: "
            "1. Have natural conversations about music and feelings "
            "2. Recommend songs based on mood and preferences "
            "3. Provide emotional support through music "
            "4. Keep responses concise but warm "
            "You can suggest songs, discuss lyrics, or just chat about music therapy."
        )}
    ]
    
    # Add conversation history
    for hist in context["conversation_history"][-5:]:
        messages.append(hist)
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150
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
        return "I'm having trouble thinking right now. Maybe we could talk about music instead?"
    
    # Prepare conversation history
    messages = [
        {"role": "system", "content": (
            "You are a friendly, empathetic music companion bot. Your role is to: "
            "1. Have natural conversations about music and feelings "
            "2. Recommend songs based on mood and preferences "
            "3. Provide emotional support through music "
            "4. Keep responses concise but warm "
            "You can suggest songs, discuss lyrics, or just chat about music therapy."
        )}
    ]
    
    # Add conversation history
    for hist in context["conversation_history"][-5:]:  # Keep last 5 messages for context
        messages.append(hist)
    
    # Add current message
    messages.append({"role": "user", "content": message})
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150
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
        return "I'm having trouble thinking right now. Maybe we could talk about music instead?"

def get_lyrics(song_title: str, artist: str) -> str:
    """Get lyrics for a song using Genius API."""
    try:
        song = genius.search_song(song_title, artist)
        return song.lyrics if song else "Couldn't find the lyrics for this song."
    except Exception as e:
        logger.error(f"Error fetching lyrics: {e}")
        return "Sorry, I couldn't retrieve the lyrics right now."

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued."""
    user = update.effective_user
    welcome_msg = (
        f"Hi {user.first_name}! 👋 I'm your Music Healing Companion.\n\n"
        "I can:\n"
        "🎵 Download music from YouTube links\n"
        "📜 Find lyrics for any song\n"
        "💿 Recommend music based on your mood\n"
        "💬 Chat about music and feelings\n\n"
        "Just send me a YouTube link or start chatting!"
    )
    await update.message.reply_text(welcome_msg)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a help message when the command /help is issued."""
    help_text = (
        "🎶 <b>Music Healing Companion Bot</b> 🎶\n\n"
        "<b>Commands:</b>\n"
        "/start - Welcome message\n"
        "/help - This help message\n"
        "/download - Download music from YouTube\n"
        "/lyrics - Get lyrics for a song\n"
        "/recommend - Get music recommendations\n"
        "/mood - Set your current mood for better recommendations\n\n"
        "<b>Or just chat with me!</b> I can understand natural language requests like:\n"
        "- \"I'm feeling sad, what songs might help?\"\n"
        "- \"Download this song: [YouTube link]\"\n"
        "- \"What are the lyrics to Bohemian Rhapsody?\""
    )
    await update.message.reply_text(help_text, parse_mode=ParseMode.HTML)

async def download_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Download music without FFmpeg conversion."""
    user_id = update.effective_user.id
    
    # Get URL from message
    url = update.message.text if not context.args else " ".join(context.args)
    
    if not is_valid_youtube_url(url):
        await update.message.reply_text("❌ Invalid YouTube URL. Please send a valid YouTube link.")
        return

    status_msg = await update.message.reply_text("⏳ Starting download...")

    try:
        # Setup download options (no FFmpeg needed)
        ydl_opts = {
            'format': 'bestaudio[ext=m4a]/bestaudio',  # Prefer m4a format
            'outtmpl': 'downloads/%(title)s.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'noplaylist': True,
            'postprocessor_args': ['-acodec', 'copy'],  # No re-encoding
            'prefer_ffmpeg': False,  # Explicitly disable ffmpeg
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # First extract info
            info = ydl.extract_info(url, download=False)
            if not info:
                raise Exception("Couldn't extract video information")
            
            title = sanitize_filename(info.get('title', 'audio'))
            await status_msg.edit_text(f"⬇️ Downloading: {title[:50]}...")
            
            # Perform download
            result = ydl.download([url])
            if result != 0:
                raise Exception("Download failed")

        # Find the downloaded file
        audio_file = find_downloaded_file(title)
        if not audio_file:
            raise Exception("Downloaded file not found")

        # Send to user
        await send_audio_file(update, audio_file, title, info.get('uploader', 'Unknown Artist'))

    except Exception as e:
        await handle_download_error(update, e)
    finally:
        await cleanup_download(status_msg, audio_file if 'audio_file' in locals() else None)
        
        
def is_valid_youtube_url(url: str) -> bool:
    """Check if URL is a valid YouTube URL."""
    patterns = [
        r'(https?://)?(www\.)?youtube\.com/watch\?v=',
        r'(https?://)?youtu\.be/',
        r'(https?://)?(www\.)?youtube\.com/shorts/'
    ]
    return any(re.search(pattern, url) for pattern in patterns)

def sanitize_filename(filename: str) -> str:
    """Remove invalid characters from filenames."""
    return re.sub(r'[\\/*?:"<>|]', "", filename)

def find_downloaded_file(title: str) -> Optional[str]:
    """Find the downloaded audio file."""
    for ext in ['m4a', 'webm', 'mp3']:  # Check possible extensions
        path = f"downloads/{title}.{ext}"
        if os.path.exists(path):
            return path
    return None        
        
        
async def get_lyrics_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle lyrics requests."""
    if not context.args:
        await update.message.reply_text("Please specify a song. Example: /lyrics Bohemian Rhapsody")
        return
    
    query = " ".join(context.args)
    await update.message.reply_text(f"🔍 Searching for lyrics: {query}")
    
    try:
        # Try to split into artist and song (format: "artist - song" or "song by artist")
        if " by " in query:
            parts = query.split(" by ")
            song, artist = parts[0], parts[1]
        elif " - " in query:
            parts = query.split(" - ")
            artist, song = parts[0], parts[1]
        else:
            song, artist = query, None
        
        lyrics = get_lyrics(song, artist)
        await update.message.reply_text(lyrics[:4000])  # Telegram message limit
    except Exception as e:
        logger.error(f"Error in get_lyrics_command: {e}")
        await update.message.reply_text("Sorry, I couldn't find those lyrics.")

async def recommend_music(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Provide music recommendations based on user context."""
    user_id = update.effective_user.id
    context_data = user_contexts.get(user_id, {})
    
    if not context_data.get("mood"):
        await update.message.reply_text(
            "I'd love to recommend some music! First, tell me about your mood with /mood "
            "or just say something like \"I'm feeling relaxed\" or \"Recommend some workout music\"."
        )
        return
    
    mood = context_data["mood"]
    await update.message.reply_text(f"🎧 Finding {mood} music recommendations for you...")
    
    try:
        # Get Spotify token
        token = get_spotify_token()
        if not token:
            raise Exception("Couldn't connect to Spotify")
        
        # Search for a seed track based on mood
        seed_query = f"{mood} music"
        seed_track = search_spotify_track(token, seed_query)
        if not seed_track:
            raise Exception("Couldn't find seed track")
        
        # Get recommendations
        recommendations = get_spotify_recommendations(token, [seed_track["id"]])
        if not recommendations:
            raise Exception("No recommendations found")
        
        # Format recommendations
        response = f"🎵 <b>Recommended {mood} music:</b>\n\n"
        for i, track in enumerate(recommendations[:5], 1):
            artists = ", ".join(a["name"] for a in track["artists"])
            response += f"{i}. <b>{track['name']}</b> by {artists}\n"
        
        response += "\nSend me a YouTube link of any song to download it!"
        await update.message.reply_text(response, parse_mode=ParseMode.HTML)
        
    except Exception as e:
        logger.error(f"Error in recommend_music: {e}")
        await update.message.reply_text("I couldn't get recommendations right now. Maybe try again later?")

async def set_mood(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start conversation to set user's mood."""
    await update.message.reply_text(
        "How are you feeling today? This will help me recommend better music.\n\n"
        "Examples: happy, sad, relaxed, energetic, nostalgic, focused"
    )
    return MOOD

async def mood_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store user's mood and ask for preferences."""
    user_id = update.effective_user.id
    mood = update.message.text.lower()
    
    # Initialize or update user context
    if user_id not in user_contexts:
        user_contexts[user_id] = {"mood": mood, "preferences": [], "conversation_history": []}
    else:
        user_contexts[user_id]["mood"] = mood
    
    await update.message.reply_text(
        f"Got it! You're feeling {mood}. 🎶\n\n"
        "Any specific music preferences? (genres, artists, etc.)\n"
        "Or just say 'none' if you don't have any."
    )
    return PREFERENCE

async def preference_received(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Store user preferences and suggest actions."""
    user_id = update.effective_user.id
    preference = update.message.text
    
    if user_id in user_contexts:
        if preference.lower() != "none":
            user_contexts[user_id]["preferences"] = [p.strip() for p in preference.split(",")]
    
    await update.message.reply_text(
        "Great! Now you can:\n"
        "/recommend - Get music recommendations\n"
        "/download - Download specific songs\n"
        "/lyrics - Find song lyrics\n\n"
        "Or just chat with me about music!"
    )
    return ConversationHandler.END

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
        await download_music(update, context)
        return
    
    # Otherwise treat as conversational message
    typing_msg = await update.message.reply_text("🎵 Thinking about music...")
    
    try:
        # Generate AI response
        response = await generate_chat_response(user_id, text)
        
        # Check if the response contains a song request pattern
        if "download" in text.lower() and "youtube.com" in text.lower():
            # Extract URL and download
            urls = [word for word in text.split() if is_valid_youtube_url(word)]
            if urls:
                context.args = [urls[0]]
                await download_music(update, context)
                return
        
        # Check if asking for lyrics
        if "lyric" in text.lower() or "words to" in text.lower():
            # Try to extract song info
            song_query = text.replace("lyrics", "").replace("lyric", "").replace("words to", "")
            context.args = [song_query.strip()]
            await get_lyrics_command(update, context)
            return
        
        # Otherwise send the AI response
        await update.message.reply_text(response)
        
    except Exception as e:
        logger.error(f"Error in handle_message: {e}")
        await update.message.reply_text("I'm having trouble responding right now. Try again later?")
    finally:
        await typing_msg.delete()

async def error_handler(update: Update, context: CallbackContext) -> None:
    """Log errors and send a user-friendly message."""
    logger.error(f"Update {update} caused error {context.error}")
    if update.effective_message:
        await update.effective_message.reply_text(
            "Oops! Something went wrong. Please try again later."
        )

def main() -> None:
    """Start the bot."""
    if not TOKEN:
        logger.error("TELEGRAM_TOKEN not found in environment variables.")
        return
    
    # Create the Application
    application = Application.builder().token(TOKEN).build()
    
    # Add conversation handler for mood setting
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("mood", set_mood)],
        states={
            MOOD: [MessageHandler(filters.TEXT & ~filters.COMMAND, mood_received)],
            PREFERENCE: [MessageHandler(filters.TEXT & ~filters.COMMAND, preference_received)],
        },
        fallbacks=[CommandHandler("cancel", cancel)],
    )
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("download", download_music))
    application.add_handler(CommandHandler("lyrics", get_lyrics_command))
    application.add_handler(CommandHandler("recommend", recommend_music))
    application.add_handler(conv_handler)
    
    # Add message handler for all text messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Add error handler
    application.add_error_handler(error_handler)
    
    # Start the bot
    logger.info("Bot starting...")
    application.run_polling()

if __name__ == "__main__":
    main()