from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "dataset"
GLIOMA_DIR = DATA_DIR / "Glioma"
MENINGIOMA_DIR = DATA_DIR / "Meningioma"
PITUITARY_DIR = DATA_DIR / "Pituitary tumor"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
MODEL_DIR = OUTPUT_DIR / "models"

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
SEED = 42