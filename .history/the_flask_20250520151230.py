# Add to main.py or a separate server script
from flask import Flask, request
import redis  # For temporary storage

app = Flask(__name__)
redis_client = redis.Redis(host='localhost', port=6379, db=0)

@app.route('/callback')
def spotify_callback():
    code = request.args.get('code')
    state = request.args.get('state')  # user_id passed via state
    if code and state:
        # Store code in Redis with user_id as key
        redis_client.setex(f"spotify_code:{state}", 300, code)  # Expires in 5 minutes
        return "Success! You can return to Telegram and continue using the bot."
    return "Error: Invalid code or state", 400

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)