from pathlib import Path
import os

from dotenv import load_dotenv


# ============================================================
# Base Paths
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

ENV_PATH = (
    BASE_DIR
    / ".env"
)

load_dotenv(
    dotenv_path=ENV_PATH
)


# ============================================================
# Model
# ============================================================

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_model.keras"
)

# ============================================================
# Domain Validation Model
# ============================================================

DOMAIN_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "domain_validator.keras"
)


DOMAIN_THRESHOLD_PATH = (
    BASE_DIR
    / "models"
    / "domain_threshold.json"
)

IMAGE_SIZE = (
    128,
    128
)

DEFAULT_THRESHOLD = (
    0.5
)

F1_THRESHOLD = (
    0.6690409183502197
)

HIGH_RECALL_THRESHOLD = (
    0.4456476867198944
)


# ============================================================
# Uploads
# ============================================================

UPLOAD_FOLDER = (
    BASE_DIR
    / "static"
    / "uploads"
)

GRADCAM_FOLDER = (
    BASE_DIR
    / "static"
    / "gradcam"
)

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg"
}

MAX_CONTENT_LENGTH = (
    10
    * 1024
    * 1024
)


# ============================================================
# Flask
# ============================================================

FLASK_SECRET_KEY = os.getenv(
    "FLASK_SECRET_KEY",
    "local-development-key"
)


# ============================================================
# Gemini
# ============================================================

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.6-flash"
)