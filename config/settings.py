"""
Global configuration for the Satellite Imagery Change Detection system.
Loads environment variables from .env and defines all hyperparameters.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ===== PATHS =====
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OSCD_DIR = DATA_DIR / "oscd"
ZONES_DIR = DATA_DIR / "zones"
REGULATIONS_DIR = DATA_DIR / "regulations"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
CHANGE_MAPS_DIR = OUTPUTS_DIR / "change_maps"
REPORTS_DIR = OUTPUTS_DIR / "reports"
NOTICES_DIR = OUTPUTS_DIR / "notices"
AUDIT_CHAIN_DIR = OUTPUTS_DIR / "audit_chain"
TRT_ENGINES_DIR = OUTPUTS_DIR / "trt_engines"
GEOJSON_DIR = OUTPUTS_DIR / "geojson"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"
FEEDBACK_DIR = DATA_DIR / "feedback_loop"
MODELS_DIR = OUTPUTS_DIR / "models"

# ===== SENTINEL HUB =====
SH_CLIENT_ID = os.getenv("SH_CLIENT_ID", "")
SH_CLIENT_SECRET = os.getenv("SH_CLIENT_SECRET", "")
SH_INSTANCE_ID = os.getenv("SH_INSTANCE_ID", "")

# ===== PostGIS =====
POSTGIS_HOST = os.getenv("POSTGIS_HOST", "localhost")
POSTGIS_PORT = int(os.getenv("POSTGIS_PORT", "5432"))
POSTGIS_DB = os.getenv("POSTGIS_DB", "satellite_cd")
POSTGIS_USER = os.getenv("POSTGIS_USER", "postgres")
POSTGIS_PASSWORD = os.getenv("POSTGIS_PASSWORD", "postgres")
POSTGIS_URL = f"postgresql://{POSTGIS_USER}:{POSTGIS_PASSWORD}@{POSTGIS_HOST}:{POSTGIS_PORT}/{POSTGIS_DB}"

# ===== MODEL HYPERPARAMETERS =====
# Siamese-SNN
BETA = 0.9                    # snn.Leaky beta (membrane decay)
NUM_STEPS = 20                # Temporal time-steps for SNN
PATCH_SIZE = 512              # Input patch size
RESOLUTION = 10               # Sentinel-2 resolution in meters
NUM_BANDS = 13                # Sentinel-2 band count
ENCODER_CHANNELS = [64, 128, 256, 512]  # U-Net encoder progression

# Training
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 4
GRADIENT_CLIP = 1.0
LOSS_ALPHA = 0.7              # Weight for ce_rate_loss vs weighted_bce
CLASS_WEIGHT_CHANGE = 5.0     # Weight for change pixels (rare)

# ===== LLM CONFIG =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ===== RAG CONFIG =====
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
RAG_TOP_K = 5

# ===== ALERT CONFIG =====
SCAN_INTERVAL_MINUTES = 60
SEVERITY_THRESHOLDS = {
    "LOW": 0.5,
    "MEDIUM": 0.7,
    "HIGH": 0.85,
    "CRITICAL": 0.95,
}

# ===== MAE CONFIG =====
MAE_MASK_RATIO = 0.75
MAE_ENCODER_DEPTH = 12
MAE_ENCODER_HEADS = 12
MAE_EMBED_DIM = 768
MAE_DECODER_DEPTH = 4
MAE_DECODER_DIM = 384
MAE_DECODER_HEADS = 6
MAE_PATCH_SIZE = 16
MAE_EPOCHS = 300
MAE_LR = 1.5e-4
MAE_WEIGHT_DECAY = 0.05
MAE_WARMUP_EPOCHS = 40

# ===== BLOCKCHAIN CONFIG =====
MERKLE_BATCH_SIZE = 100
ANCHOR_INTERVAL_HOURS = 24
ANCHOR_SERVICE = "opentimestamps"

# ===== TENSORRT / EDGE-AI CONFIG =====
TRT_FP16 = True
TRT_BATCH_SIZE = 8
TRT_WORKSPACE_GB = 1
PRUNING_AMOUNT = 0.3

# ===== CITY SCANNER CONFIG =====
SURAT_BBOX = [72.72, 21.08, 72.92, 21.28]  # [lon_min, lat_min, lon_max, lat_max]
TILE_SIZE = 512
SCAN_BATCH_SIZE = 8

# ===== TWILIO CONFIG =====
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_NUMBER = os.getenv("TWILIO_WHATSAPP_NUMBER", "whatsapp:+14155238886")

# ===== APP CONFIG =====
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", "8000"))
DEBUG = os.getenv("DEBUG", "true").lower() == "true"

# ===== CLOUD MASKING =====
CLOUD_THRESHOLD = 0.4
SCL_CLOUD_CLASSES = [8, 9, 10]  # Cloud medium, Cloud high, Cirrus
SCL_SHADOW_CLASSES = [3]        # Cloud shadow
