import os
import sys

# Ensure root directory is in sys.path for importing app modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app

# Vercel serverless function entrypoint
if __name__ == "__main__":
    app.run()
