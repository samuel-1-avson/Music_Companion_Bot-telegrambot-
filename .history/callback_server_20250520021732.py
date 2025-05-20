from http.server import BaseHTTPRequestHandler, HTTPServer
import urllib.parse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CallbackHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        query = urllib.parse.urlparse(self.path).query
        params = urllib.parse.parse_qs(query)
        code = params.get('code', [''])[0]
        state = params.get('state', [''])[0]
        logger.info(f"Received callback: path={self.path}, code={code}, state={state}")
        html = f"""
        <html>
            <body>
                <h2>Spotify Authorization</h2>
                <p><b>Authorization Code:</b> {code}</p>
                <p><b>State (User ID):</b> {state}</p>
                <p>Please return to Telegram and send: <code>/spotify_code {code}</code></p>
                <p>Tip: Copy the code above and paste it into the bot.</p>
            </body>
        </html>
        """
        self.wfile.write(html.encode())

def run_server():
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, CallbackHandler)
    logger.info('Starting callback server on http://localhost:8000...')
    httpd.serve_forever()

if __name__ == '__main__':
    run_server()