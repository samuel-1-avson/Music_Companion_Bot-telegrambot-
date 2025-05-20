from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
import logging
import os
from html import escape

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            # Send response headers
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()

            # Parse the query string from the URL
            full_url = f"{self.requestline} {self.path}"
            logger.info(f"Received request: {full_url}")

            query = urllib.parse.urlparse(self.path).query
            params = urllib.parse.parse_qs(query)

            # Extract code and state, default to empty string if not present
            code = params.get('code', [''])[0]
            state = params.get('state', [''])[0]

            # Sanitize inputs to prevent HTML injection
            code = escape(code) if code else ""
            state = escape(state) if state else ""

            # Log the extracted parameters
            logger.info(f"Callback parameters - code: {code}, state: {state}")

            # Prepare the HTML response
            if code and state:
                html = f"""
                <html>
                    <body>
                        <h2>Spotify Authorization Successful</h2>
                        <p><b>Authorization Code:</b> {code}</p>
                        <p><b>State (User ID):</b> {state}</p>
                        <p>Please return to Telegram and send: <code>/spotify_code {code}</code></p>
                        <p>Tip: Copy the code above and paste it into the bot.</p>
                    </body>
                </html>
                """
            else:
                html = """
                <html>
                    <body>
                        <h2>Spotify Authorization Error</h2>
                        <p><b>Error:</b> Authorization code or state not received.</p>
                        <p>Please try again by sending /link_spotify in Telegram.</p>
                        <p>If the issue persists, ensure the redirect URI matches in Spotify Dashboard.</p>
                    </body>
                </html>
                """
                logger.error("Missing code or state in callback request")

            # Write the response
            self.wfile.write(html.encode())
        except Exception as e:
            logger.error(f"Error handling request: {str(e)}")
            self.send_response(500)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            error_html = f"""
            <html>
                <body>
                    <h2>Server Error</h2>
                    <p>An unexpected error occurred: {escape(str(e))}</p>
                    <p>Please try again later.</p>
                </body>
            </html>
            """
            self.wfile.write(error_html.encode())

def run_server():
    # Get port from environment variable, default to 8000
    port = int(os.getenv("CALLBACK_PORT", 8000))
    server_address = ('', port)
    try:
        httpd = HTTPServer(server_address, CallbackHandler)
        logger.info(f'Starting callback server on http://localhost:{port}...')
        httpd.serve_forever()
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

if __name__ == '__main__':
    run_server()