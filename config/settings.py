import os
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
ENABLE_SEMANTIC_SEARCH = os.getenv("ENABLE_SEMANTIC_SEARCH", "false")
SEMANTIC_BACKEND = os.getenv("SEMANTIC_BACKEND", "local")
SEMANTIC_MODEL_NAME = os.getenv(
    "SEMANTIC_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2"
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "travel_destinations")
QDRANT_TOUR_COLLECTION = os.getenv("QDRANT_TOUR_COLLECTION", "travel_tours")

MINIO_URL = os.getenv("MINIO_URL", "http://localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "chatbot-models")
MINIO_MODEL_FILE = os.getenv("MINIO_MODEL_FILE", "latest_model.tar.gz")
MINIO_MODEL_URL = os.getenv(
    "MINIO_MODEL_URL",
    f"{MINIO_URL}/{MINIO_BUCKET}/{MINIO_MODEL_FILE}"
)

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "Travel_Chatbot_Rasa")

RASA_API_URL = os.getenv("RASA_API_URL", "http://localhost:5005")
RASA_ACTION_URL = os.getenv("RASA_ACTION_URL", "http://localhost:5055/webhook")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = os.getenv("RABBITMQ_PORT", "5672")
RABBITMQ_MGM_PORT = os.getenv("RABBITMQ_MGM_PORT", "15672")

DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5001"))
