import os
import sys

# Ensure root directory is in sys.path for importing app modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from werkzeug.middleware.proxy_fix import ProxyFix
from app import app

class VercelPathMiddleware:
    """
    WSGI middleware to ensure Flask receives the exact client-requested path
    when deployed behind Vercel Edge proxy, route matches, and rewrites.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        # Vercel supplies the actual client path in one of these headers
        matched_path = (
            environ.get('HTTP_X_MATCHED_PATH') or
            environ.get('HTTP_X_FORWARDED_URI') or
            environ.get('HTTP_X_FORWARDED_PATH')
        )
        if matched_path:
            # Strip query parameters if present
            if '?' in matched_path:
                matched_path = matched_path.split('?', 1)[0]
            environ['PATH_INFO'] = matched_path
        elif environ.get('PATH_INFO') in ('/api/index', '/api/index.py', '/api', '/api/'):
            environ['PATH_INFO'] = '/'

        return self.wsgi_app(environ, start_response)

# Apply ProxyFix for SSL/host forwarding and VercelPathMiddleware for route resolution
app.wsgi_app = VercelPathMiddleware(
    ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
)

# Export app for WSGI / Vercel serverless function
# Do not call app.run() here as Vercel manages the execution cycle.
