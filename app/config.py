import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
ALLOWED_EXTENSIONS = [".pdf",".txt"]
MAX_FILE_SIZE_MB = 10
UPLOAD_DIR = "uploads"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")