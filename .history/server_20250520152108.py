# server.py
from flask import Flask, request
import redis
import logging

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
redis_client = redis.Redis(host='localhost', port=6379, db=0)

@app.route('/callback')
def spotify_callback():
    code = request.args.get('code')
    state = request.args.get('state')  # user_id passed via state
    if code and state:
        # Store code in Redis with user_id as key, expires in 5 minutes
        redis_client.setex(f"spotify_code:{state}", 300, code)
        logger.info(f"Stored Spotify code for user {state}")
        return """
        <h1>Success!</h1>
        <p>You can now return to Telegram. Your Spotify account is being linked.</p>
        """
    logger.error(f"Invalid Spotify callback: code={code}, state={state}")
    return "Error: Invalid code or state", 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)