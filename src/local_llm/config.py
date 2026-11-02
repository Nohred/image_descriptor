from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[1]

OLLAMA_ENDPOINT = "http://127.0.0.1:11434"

CHAT_MODEL = "qwen2.5:3b"
EMBED_MODEL = "nomic-embed-text-v2-moe"

EMBED_DIM = 768

CHAT_CONTEXT = 4096
TEMPERATURE = 0.0
TIMEOUT = 120.0

EMBED_BATCH_SIZE = 16

DATABASE_PATH = (
    SRC_DIR
    / "database"
    / "generated_features"
    / "data"
    / "florence.sqlite3"
)