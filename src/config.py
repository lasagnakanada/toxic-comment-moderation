from pathlib import Path
from pydantic import BaseModel

SRC_DIR = Path(__file__).parent

class ModelSettings(BaseModel):
    MODEL_PATH: Path = SRC_DIR / "toxic_comment_classifier.pth"
    THRESHOLDS_PATH: Path = SRC_DIR / "best_thresholds.json"
    BERT_MODEL_NAME: str = 'distilbert-base-uncased'
    MAX_LENGTH: int = 128
    INPUT_SIZE: int = 768
    HIDDEN_SIZE: int = 384
    OUTPUT_SIZE: int = 6
    LABELS: list[str] = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    DATA_DIR: Path = SRC_DIR.parent / "data"
    EMBEDDINGS_PATH: Path = DATA_DIR / "embeddings.pt"
    LABELS_PATH: Path = DATA_DIR / "labels.npy"

settings = ModelSettings()
