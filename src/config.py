# This file manages environment variables and configuration

import os
from dotenv import load_dotenv

load_dotenv() # it reads the .env file and loads secrets like API keys that aren't hardcoded in the source code.

# API keys
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Model configuration
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gemini-2.5-flash"

# Chunk settings
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Vector DB settings
CHROMA_COLLECTION_NAME = 'rag_collection'
DATA_DIR = 'data'

