import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

UPLOAD_FOLDER = "uploads"
VECTOR_DB = "vectorstore"
TEMP_FOLDER = "temp"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

LLM_MODEL = "llama-3.3-70b-versatile"