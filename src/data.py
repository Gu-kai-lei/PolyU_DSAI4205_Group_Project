"""Loading helpers shared by all notebooks."""
import pandas as pd

from .config import CLEAN_DIR, RAW_CSV, SPLITS_CSV


def load_raw() -> pd.DataFrame:
    """Original CSV with a stable `id` (row order of the raw file) and a `label` column."""
    df = pd.read_csv(RAW_CSV, encoding="utf-8")
    df = df.rename(columns={"text_sentiment": "label"})
    df.insert(0, "id", range(len(df)))
    return df


def load_clean(version: int) -> pd.DataFrame:
    """Cleaned data written by notebooks/01_cleaning.ipynb."""
    return pd.read_csv(CLEAN_DIR / f"clean_v{version}.csv", encoding="utf-8")


def load_splits() -> pd.DataFrame:
    """Fixed split with columns `id`, `split` ("train"/"test") and `fold` (0-4 for train, -1 for test)."""
    return pd.read_csv(SPLITS_CSV)
