"""Member-1 SVM experiments on sentence vectors.

Two embeddings are supported:

* ``fasttext_mean`` - scratch-trained FastText, trained only on each outer
  training fold.  The reserved test texts are encoded only by the ``full``
  embedding trained on the development set.
* ``sbert_multi`` - frozen pre-trained multilingual SBERT
  (paraphrase-multilingual-mpnet-base-v2, from member 5).  It is not trained on
  our data, so one shared matrix is used for every fold and for the test set.

The reserved test set is never used for model selection.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import load_embedding, save_embedding, save_preds
from src.config import DATA_ROOT, LABELS, N_FOLDS, RESULTS_DIR, SEED
from src.data import load_clean, load_splits
from src.evaluate import evaluate, new_run_id, plot_confusion, save_result


EMBEDDING = "fasttext_mean"  # default, kept for backward compatibility
EMBEDDINGS = {
    "fasttext_mean": "scratch-trained FastText mean pooling",
    "sbert_multi": "frozen paraphrase-multilingual-mpnet-base-v2 (L2-normalised)",
}
LABEL_IDS = list(LABELS)


def emb_name(embedding: str, tag: str) -> str:
    """Cache name for one fold.  Frozen SBERT has a single shared matrix."""
    return embedding if embedding == "sbert_multi" else f"{embedding}_{tag}"


def _load_frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    clean = load_clean()
    splits = load_splits(clean=clean)
    frame = clean[["id", "group_id", "label", "text_norm"]].merge(
        splits[["id", "split", "fold"]], on="id", validate="one_to_one"
    )
    return frame, splits


def _embedding_files_exist(name: str) -> bool:
    return all((DATA_ROOT / "emb" / f"{name}{suffix}.npy").exists()
               for suffix in ("", "_ids"))


def ensure_fasttext_embeddings(frame: pd.DataFrame) -> None:
    required = [f"{EMBEDDING}_fold{k}" for k in range(N_FOLDS)] + [f"{EMBEDDING}_full"]
    if all(_embedding_files_exist(name) for name in required):
        return
    try:
        from gensim.models import FastText
    except ImportError as exc:
        raise RuntimeError(
            "FastText embeddings are missing and gensim is not installed. "
            "Install the team environment or gensim before running SVM."
        ) from exc

    params = {
        "vector_size": 100, "window": 5, "min_count": 2, "sg": 1,
        "min_n": 3, "max_n": 5, "epochs": 50, "bucket": 200_000,
    }

    def stable_hash(text: str) -> int:
        return zlib.crc32(str(text).encode("utf-8"))

    def encode(model, texts: pd.Series) -> np.ndarray:
        vectors = np.zeros((len(texts), model.wv.vector_size), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = str(text).split()
            if tokens:
                vectors[row] = np.stack([model.wv[token] for token in tokens]).mean(axis=0)
        return vectors

    dev = frame[frame.split == "train"].sort_values("id")
    all_ids = frame.sort_values("id").id.to_numpy()
    text_by_id = frame.set_index("id").text_norm
    for tag, train_ids, encode_ids in [
        (f"fold{k}", dev.loc[dev.fold != k, "id"].to_numpy(), dev.id.to_numpy())
        for k in range(N_FOLDS)
    ] + [("full", dev.id.to_numpy(), all_ids)]:
        train_texts = [str(text_by_id.loc[int(i)]).split() for i in train_ids]
        model = FastText(
            sentences=train_texts, seed=SEED, workers=1, hashfxn=stable_hash, **params
        )
        ids = np.asarray(encode_ids)
        save_embedding(f"{EMBEDDING}_{tag}", ids, encode(model, text_by_id.loc[ids]))


def make_model(kernel: str, C: float, gamma: str | float,
               class_weight: str | None = None) -> Pipeline:
    params = {"C": C, "kernel": kernel, "probability": True,
              "random_state": SEED, "class_weight": class_weight}
    if kernel == "rbf":
        params["gamma"] = gamma
    return Pipeline([
        ("scale", StandardScaler()),
        ("svc", SVC(**params)),
    ])


def run_config(frame: pd.DataFrame, kernel: str, C: float, gamma: str | float,
               embedding: str = EMBEDDING, class_weight: str | None = None,
               ) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    dev = frame[frame.split == "train"].sort_values("id").reset_index(drop=True)
    test = frame[frame.split == "test"].sort_values("id").reset_index(drop=True)
    oof = np.zeros((len(dev), len(LABEL_IDS)), dtype=float)
    started = perf_counter()

    for fold in range(N_FOLDS):
        train_mask = dev.fold != fold
        valid_mask = dev.fold == fold
        train_ids = dev.loc[train_mask, "id"].to_numpy()
        valid_ids = dev.loc[valid_mask, "id"].to_numpy()
        X_train = load_embedding(emb_name(embedding, f"fold{fold}"), ids=train_ids)
        X_valid = load_embedding(emb_name(embedding, f"fold{fold}"), ids=valid_ids)
        model = make_model(kernel, C, gamma, class_weight)
        model.fit(X_train, dev.loc[train_mask, "label"].to_numpy())
        oof[valid_mask] = model.predict_proba(X_valid)

    y = dev.label.to_numpy()
    pred = np.asarray(LABEL_IDS)[oof.argmax(axis=1)]
    run_id = new_run_id("svm", embedding)
    params = {"kernel": kernel, "C": C, "gamma": gamma,
              "class_weight": class_weight,
              "scaler": "training-fold StandardScaler",
              "embedding": EMBEDDINGS[embedding]}
    row = evaluate(y, pred, run_id=run_id, model="svm", embedding=embedding,
                   split="cv", params=params, train_seconds=perf_counter() - started,
                   notes="OOF pooled over official folds; test labels not used")

    X_dev = load_embedding(emb_name(embedding, "full"), ids=dev.id.to_numpy())
    X_test = load_embedding(emb_name(embedding, "full"), ids=test.id.to_numpy())
    final_model = make_model(kernel, C, gamma, class_weight)
    final_model.fit(X_dev, y)
    test_proba = final_model.predict_proba(X_test)
    save_preds(run_id, oof_ids=dev.id.to_numpy(), oof_proba=oof,
               test_ids=test.id.to_numpy(), test_proba=test_proba,
               classes=final_model.named_steps["svc"].classes_)
    return row, oof, test_proba, pred


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--kernel", choices=("linear", "rbf"), default="linear")
    parser.add_argument("--C", type=float, nargs="+", default=[0.1, 1.0, 10.0])
    parser.add_argument("--gamma", nargs="+", default=["scale"])
    parser.add_argument("--embedding", choices=tuple(EMBEDDINGS), default=EMBEDDING)
    parser.add_argument("--class-weight", nargs="+", choices=("none", "balanced"),
                        default=["none"], dest="class_weight")
    args = parser.parse_args()

    frame, _ = _load_frame()
    if args.embedding == "fasttext_mean":
        ensure_fasttext_embeddings(frame)
    elif not _embedding_files_exist(args.embedding):
        raise FileNotFoundError(
            f"{DATA_ROOT / 'emb' / args.embedding}.npy is missing: copy member 5's "
            "sbert_multi.npy and sbert_multi_ids.npy into data/emb/ first.")
    gamma_values = [
        gamma if gamma in {"scale", "auto"} else float(gamma)
        for gamma in args.gamma
    ]
    weights = [None if w == "none" else w for w in args.class_weight]
    configs = [(args.kernel, C, gamma, weight)
               for weight in weights
               for C in args.C
               for gamma in (gamma_values if args.kernel == "rbf" else ["scale"])]
    if args.check:
        print(json.dumps({"rows": len(frame), "development": int((frame.split == "train").sum()),
                          "holdout": int((frame.split == "test").sum()),
                          "embedding": args.embedding, "configs": configs}, indent=2))
        return

    rows = []
    best = None
    for kernel, C, gamma, weight in configs:
        row, oof, test_proba, pred = run_config(frame, kernel, C, gamma,
                                                args.embedding, weight)
        rows.append(row)
        print(f"{args.embedding} {kernel} C={C} gamma={gamma} class_weight={weight}: "
              f"accuracy={row['accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}", flush=True)
        if best is None or row["macro_f1"] > best["row"]["macro_f1"]:
            best = {"row": row, "oof": oof, "test_proba": test_proba, "pred": pred}

    RESULT_FILE = RESULTS_DIR / "m1_svm.csv"
    new_results = pd.DataFrame(rows)
    if RESULT_FILE.exists():
        old_results = pd.read_csv(RESULT_FILE)
        new_results = pd.concat([old_results, new_results], ignore_index=True)
    new_results.to_csv(RESULT_FILE, index=False)
    plot_confusion(
        frame.loc[frame.split == "train"].sort_values("id").label,
        best["pred"],
        f"SVM ({args.embedding}) OOF confusion matrix",
        save_as=ROOT / "figures" / f"{best['row']['run_id']}_confusion.png",
    )
    print(f"Best Macro-F1: {best['row']['macro_f1']:.4f}")
    print(f"Results: {RESULT_FILE}")


if __name__ == "__main__":
    main()
