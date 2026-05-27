from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "dataset"
GLIOMA_DIR = DATA_DIR / "Glioma"
MENINGIOMA_DIR = DATA_DIR / "Meningioma"
PITUITARY_DIR = DATA_DIR / "Pituitary"

IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 1e-4
SEED = 42