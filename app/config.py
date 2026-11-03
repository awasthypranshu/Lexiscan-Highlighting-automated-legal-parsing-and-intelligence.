import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI")
ALLOWED_EXTENSIONS = [".pdf", ".txt"]
MAX_FILE_SIZE_MB = 10
UPLOAD_DIR = "uploads"
GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")

# Gemini model — must match the model registered in your Google AI Studio account.
# The existing codebase uses gemini-4-argon; kept as-is so existing analysis continues to work.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-4-argon")

# RAG / Vector-search configuration
# QDRANT_PATH: local filesystem path for Qdrant's embedded storage.
# NOTE: Embedded local mode (path=) is intended for single-process development.
# It must NOT be used with multiple uvicorn workers sharing the same directory.
# For production, switch to a standalone Qdrant server or Qdrant Cloud and set
# QDRANT_URL + QDRANT_API_KEY instead.
QDRANT_PATH = os.getenv("QDRANT_PATH", "./qdrant_data")

# Sentence-Transformers model used to generate chunk and query embeddings.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
