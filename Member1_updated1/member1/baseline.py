"""Member-1 baseline. Never evaluates the reserved holdout.

Readiness: python member1/baseline.py --check
Temporary experiment: python member1/baseline.py --mode provisional
Team splits: python member1/baseline.py --mode team
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import hashlib
from importlib.metadata import version
from time import perf_counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.config import (LABELS as TEAM_LABELS, SEED, N_FOLDS, CLEAN_DIR,
                        CLEAN_VERSION, SPLITS_DIR, SPLIT_VERSION, FIGURES_DIR, VERSION_TAGS)
from src.data import load_clean, load_splits, check_clean
from src.evaluate import evaluate, new_run_id, save_result, plot_confusion
from src.cache import save_preds

LABELS = list(TEAM_LABELS)
NAMES = list(TEAM_LABELS.values())


def readiness() -> dict:
    """Read-only preflight; never creates a split or runs training."""
    clean_path = CLEAN_DIR / f"clean_v{CLEAN_VERSION}.csv"
    split_path = SPLITS_DIR / f"folds_v{SPLIT_VERSION}.csv"
    errors = []
    count = None
    try:
        clean = load_clean()
        count = len(clean)
        splits = load_splits(clean=clean)
        validate_splits(clean, team_to_local(splits))
    except (FileNotFoundError, ValueError) as exc:
        errors.append(str(exc))
    # Only the baseline runtime subset, not unrelated neural-model dependencies.
    expected = {"numpy": "1.26.4", "scipy": "1.13.1", "pandas": "2.2.3",
                "scikit-learn": "1.5.2", "matplotlib": "3.9.2"}
    actual = {package: version(package) for package in expected}
    mismatches = {p: {"expected": v, "actual": actual[p]}
                  for p, v in expected.items() if actual[p] != v}
    if sys.version_info[:2] != (3, 11):
        mismatches["python"] = {"expected": "3.11", "actual": platform.python_version()}
    return {"clean_path": str(clean_path), "split_path": str(split_path),
            "clean_rows": count, "input_errors": errors, "environment_mismatches": mismatches,
            "ready_for_team_run": not errors and not mismatches}


def team_to_local(splits: pd.DataFrame) -> pd.DataFrame:
    return splits.rename(columns={"split": "partition"}).assign(
        partition=lambda x: x.partition.map({"train": "development", "test": "holdout"}))


def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    if "keep" in df:
        df = df.loc[pd.to_numeric(df["keep"]) == 1].copy()
    required = {"id", "group_id", "label", "text_norm"}
    if not required <= set(df):
        raise ValueError(f"Missing columns: {required - set(df)}")
    for col in ["id", "group_id", "label"]:
        values = pd.to_numeric(df[col], errors="raise")
        if not np.isfinite(values).all() or not (values == np.floor(values)).all():
            raise ValueError(f"Invalid integer column: {col}")
        df[col] = values.astype(int)
    if df.empty or df.id.duplicated().any():
        raise ValueError("Empty dataset or duplicate IDs")
    if set(df.label) != set(LABELS) or df.text_norm.str.strip().eq("").any():
        raise ValueError("Expected three labels and nonempty normalized text")
    return df.sort_values("id").reset_index(drop=True)


def provisional_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Temporary 20% holdout and five development folds, all group-disjoint."""
    outer = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    dev_idx, holdout_idx = next(outer.split(df.text_norm, df.label, df.group_id))
    splits = df[["id", "group_id"]].copy()
    splits["partition"] = "development"
    splits["fold"] = -1
    splits.loc[holdout_idx, "partition"] = "holdout"
    dev = df.iloc[dev_idx]
    inner = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    for fold, (_, valid_idx) in enumerate(inner.split(dev.text_norm, dev.label, dev.group_id)):
        splits.loc[dev_idx[valid_idx], "fold"] = fold
    validate_splits(df, splits)
    return splits


def validate_splits(df: pd.DataFrame, splits: pd.DataFrame) -> None:
    if splits.id.duplicated().any() or set(splits.id) != set(df.id):
        raise ValueError("Split IDs must match the retained dataset exactly")
    joined = df[["id", "group_id", "label"]].merge(
        splits[["id", "partition", "fold"]], on="id", validate="one_to_one")
    if set(joined.partition) != {"development", "holdout"}:
        raise ValueError("Expected development and holdout partitions")
    if (joined.groupby("group_id").partition.nunique() > 1).any():
        raise ValueError("Group crosses holdout boundary")
    dev = joined[joined.partition == "development"]
    if set(dev.fold) != set(range(N_FOLDS)) or (dev.groupby("group_id").fold.nunique() > 1).any():
        raise ValueError("Invalid folds or group crosses validation boundary")
    if not joined.loc[joined.partition == "holdout", "fold"].eq(-1).all():
        raise ValueError("Holdout must have fold -1")
    for fold in range(N_FOLDS):
        for labels in [dev.loc[dev.fold == fold, "label"], dev.loc[dev.fold != fold, "label"]]:
            if set(labels) != set(LABELS):
                raise ValueError("Every training and validation fold needs all three classes")


def make_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 1), min_df=1)),
        ("lr", LogisticRegression(C=1.0, max_iter=2000, random_state=SEED)),
    ])


def score(y_true, y_pred) -> dict:
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f)}


def run(data_path: Path | None = None, output_root: Path | None = None,
        *, mode: str = "provisional") -> Path:
    if mode not in {"provisional", "team"}:
        raise ValueError("mode must be provisional or team")
    output_root = output_root or ROOT / "member1" / "outputs"
    data_path = data_path or CLEAN_DIR / f"clean_v{CLEAN_VERSION}.csv"
    if mode == "team":
        if data_path.resolve() != (CLEAN_DIR / f"clean_v{CLEAN_VERSION}.csv").resolve():
            raise ValueError("Team mode must use configured clean data")
        status = readiness()
        if not status["ready_for_team_run"]:
            raise RuntimeError("Team run prerequisites are incomplete:\n" + json.dumps(status, indent=2))
        df = load_clean().sort_values("id").reset_index(drop=True)
        splits = team_to_local(load_splits(clean=df))
        validate_splits(df, splits)
        run_id = new_run_id("baseline", "tfidf")
    else:
        df = load_data(data_path)
        check_clean(df)
        splits = provisional_splits(df)
        run_id = "provisional_baseline_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    title_prefix = "PROVISIONAL" if mode == "provisional" else "Team splits"
    out = output_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    splits.to_csv(out / ("provisional_splits.csv" if mode == "provisional" else "split_snapshot.csv"), index=False)
    dev = df.merge(splits[["id", "partition", "fold"]], on="id", validate="one_to_one")
    dev = dev.loc[dev.partition == "development"].reset_index(drop=True)
    probabilities = np.full((len(dev), 3), np.nan)
    fold_scores = []
    train_seconds = 0.0
    for fold in range(N_FOLDS):
        train, valid = dev.fold != fold, dev.fold == fold
        model = make_model()
        started = perf_counter()
        model.fit(dev.loc[train, "text_norm"], dev.loc[train, "label"])
        train_seconds += perf_counter() - started
        assert list(model.classes_) == LABELS
        probabilities[valid] = model.predict_proba(dev.loc[valid, "text_norm"])
        pred = np.asarray(LABELS)[probabilities[valid].argmax(axis=1)]
        fold_scores.append({"fold": fold, "n_train": int(train.sum()),
                            "n_valid": int(valid.sum()), **score(dev.loc[valid, "label"], pred)})
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Incomplete or invalid OOF probabilities")
    pred = np.asarray(LABELS)[probabilities.argmax(axis=1)]
    oof = dev[["id", "group_id", "fold", "label"]].copy()
    oof["prediction"] = pred
    for i, name in enumerate(NAMES):
        oof[f"prob_{name.lower()}"] = probabilities[:, i]
    oof.to_csv(out / "oof_predictions.csv", index=False)
    folds = pd.DataFrame(fold_scores)
    folds.to_csv(out / "fold_metrics.csv", index=False)
    report = classification_report(dev.label, pred, labels=LABELS,
                                   target_names=NAMES, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(out / "classification_report.csv")
    confusion = plot_confusion(dev.label, pred, f"{title_prefix} baseline: OOF",
                               save_as=out / "confusion_matrix.png")
    if mode == "team":
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        confusion.savefig(FIGURES_DIR / f"baseline_{run_id}_confusion.png", dpi=160)
    plt.close(confusion)
    # Refit only on development data for interpretable coefficients, not evaluation.
    model = make_model().fit(dev.text_norm, dev.label)
    vocab = model.named_steps["tfidf"].get_feature_names_out()
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    words = []
    for label, name, ax, coef in zip(LABELS, NAMES, axes, model.named_steps["lr"].coef_):
        indices = np.argsort(coef)[-12:]
        ax.barh(vocab[indices], coef[indices])
        ax.set_title(name)
        ax.set_xlabel("Logistic regression coefficient")
        words.extend({"label": label, "class": name, "word": vocab[i], "coefficient": float(coef[i])}
                     for i in indices[::-1])
    fig.suptitle(f"{title_prefix}: top positive coefficients (development refit)")
    fig.tight_layout()
    fig.savefig(out / "top_words.png", dpi=160)
    if mode == "team":
        fig.savefig(FIGURES_DIR / f"baseline_{run_id}_top_words.png", dpi=160)
    plt.close(fig)
    pd.DataFrame(words).to_csv(out / "top_words.csv", index=False)
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    params = {"tfidf_ngram_range": [1, 1], "tfidf_min_df": 1,
              "lr_C": 1.0, "lr_max_iter": 2000, "class_weight": None}
    row = evaluate(dev.label, pred, run_id=run_id, model="baseline", embedding="tfidf",
                   split="cv", params=params, train_seconds=train_seconds,
                   notes=f"{title_prefix}; pooled OOF; fit time sums five training folds; test predictions deferred")
    if mode == "team":
        # Empty test arrays explicitly mean deferred, not test predictions of zero.
        save_preds(run_id, oof_ids=dev.id.to_numpy(), oof_proba=probabilities,
                   test_ids=np.array([], dtype=int), test_proba=np.empty((0, len(LABELS))), classes=LABELS)
        save_result(row, "baseline")
    else:
        # Never tag provisional predictions as the official folds_v1.
        row["split_version"] = "provisional_local"
    (out / "team_metric_record.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    summary = {"run_id": run_id, "status": title_prefix,
               "data_version": VERSION_TAGS["data_version"],
               "split_version": VERSION_TAGS["split_version"] if mode == "team" else "provisional_local",
               "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
               "split_snapshot_sha256": hashlib.sha256(splits.to_csv(index=False).encode()).hexdigest(),
               "seed": SEED, "data_file": str(data_path.resolve()),
               "retained_rows": len(df), "development_rows": len(dev),
               "holdout_rows": len(df) - len(dev), "holdout_evaluated": False,
               "cv_mean": folds[metrics].mean().to_dict(),
               "cv_std_sample": folds[metrics].std(ddof=1).to_dict(),
               "pooled_oof": score(dev.label, pred),
               "parameters": params,
               "cv_fit_seconds": train_seconds,
               "test_predictions_status": "deferred; no test inference or evaluation",
               "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                            "numpy": np.__version__, "pandas": pd.__version__,
                            "scipy": version("scipy"), "matplotlib": matplotlib.__version__},
               "platform": platform.platform(),
               "split_file_sha256": hashlib.sha256(
                   (SPLITS_DIR / f"folds_v{SPLIT_VERSION}.csv").read_bytes()).hexdigest()
                   if mode == "team" and (SPLITS_DIR / f"folds_v{SPLIT_VERSION}.csv").exists() else None}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Outputs: {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check data, team splits and baseline environment")
    parser.add_argument("--mode", choices=["provisional", "team"], default="team")
    parser.add_argument("--data", type=Path, default=CLEAN_DIR / f"clean_v{CLEAN_VERSION}.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "member1" / "outputs")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(readiness(), indent=2, ensure_ascii=False))
    else:
        run(args.data, args.output, mode=args.mode)
