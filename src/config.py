"""Project-wide paths and constants. Import from here instead of hard-coding paths."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Data, embeddings and predictions live on the shared Google Drive, not in Git.
# Set DATA_ROOT to your copy of that folder; without it, ./data is used.
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))
RAW_CSV = DATA_ROOT / "raw" / "stock_market_crash_2022.csv"
CLEAN_DIR = DATA_ROOT / "clean"
EMB_DIR = DATA_ROOT / "emb"
PRED_DIR = DATA_ROOT / "preds"

# Small, versioned outputs stay in the repo.
SPLITS_CSV = PROJECT_ROOT / "splits" / "folds.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

SEED = 42
N_FOLDS = 5
TEST_SIZE = 0.2

# Label encoding defined by the dataset description.
LABELS = {0: "Neutral", 1: "Positive", 2: "Negative"}
