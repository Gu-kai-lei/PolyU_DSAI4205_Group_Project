"""Shared scoring helpers so every model is evaluated the same way.

Maintained by member 4. Agree on interface changes in the group chat first.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
)

from .config import LABELS, RESULTS_DIR

LABEL_IDS = list(LABELS)
LABEL_NAMES = [LABELS[i] for i in LABEL_IDS]


def evaluate(y_true, y_pred, *, model: str, embedding: str, split: str = "test",
             train_seconds: float | None = None, notes: str = "") -> dict:
    """Score predictions and return one result row."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0
    )
    per_class_f1 = f1_score(y_true, y_pred, labels=LABEL_IDS, average=None, zero_division=0)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "embedding": embedding,
        "split": split,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        **{f"f1_{name.lower()}": score for name, score in zip(LABEL_NAMES, per_class_f1)},
        "train_seconds": train_seconds,
        "notes": notes,
    }


def save_result(row: dict, file_stem: str) -> Path:
    """Append a row to results/<file_stem>.csv; one file per owner avoids merge conflicts."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{file_stem}.csv"
    new = pd.DataFrame([row])
    table = pd.concat([pd.read_csv(path), new], ignore_index=True) if path.exists() else new
    table.to_csv(path, index=False)
    return path


def collect_results() -> pd.DataFrame:
    """Combine every results/*.csv into one comparison table."""
    files = sorted(RESULTS_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f).assign(source=f.stem) for f in files], ignore_index=True)


def plot_confusion(y_true, y_pred, title: str, save_as: Path | None = None):
    """Confusion matrix labelled with class names instead of 0/1/2."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=LABEL_IDS, display_labels=LABEL_NAMES,
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    if save_as is not None:
        fig.savefig(save_as, dpi=200)
    return fig
