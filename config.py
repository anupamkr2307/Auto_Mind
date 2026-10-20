import os

# Auto_Mind Configuration
# AI API Key for Chatbot functionality (prefer environment variables)
AK = os.getenv('GEMINI_API_KEY') or os.getenv('AK') or "AIzaSyDi_pLm4wkcwg9XTPB6Di9PPITZc36E95E"

# Flask Configuration
FLASK_ENV = os.getenv('FLASK_ENV', 'development')
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1')

# Application Settings
APP_HOST = os.getenv('APP_HOST', '0.0.0.0')
APP_PORT = int(os.getenv('PORT', os.getenv('APP_PORT', '5000')))

