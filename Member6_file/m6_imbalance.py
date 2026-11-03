"""不平衡处理对比。

必做：不处理、class_weight、随机过采样。选做里的 SMOTE 用同一套分类器一起比。
三种（加 SMOTE）都用「按折 FastText 平均向量 + 逻辑回归」，这样差别只来自不平衡处理，
也不会在 baseline 之外使用 TF-IDF。

过采样和 SMOTE 只作用在当前折的训练部分，验证折和测试集保持原样。
选策略只看 5 折 OOF 的 macro-F1；测试集只给最终推荐的策略打一次分。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, LABELS, N_FOLDS, SEED
from src.evaluate import evaluate, new_run_id, save_result
from src.cache import save_preds

from Member6_file.m6_data import ensure_fasttext, labels_for, load_frames, mean_matrix

STRATEGIES = ("none", "class_weight", "random_oversample", "smote")
STRATEGY_LABELS = {
    "none": "No treatment",
    "class_weight": "class_weight",
    "random_oversample": "Random oversample",
    "smote": "SMOTE",
}
# macro-F1 差距小于这个值时，选更简单的策略。数据最大类/最小类只有 1.40。
TIE_MARGIN = 0.005
SIMPLICITY = {"none": 0, "class_weight": 1, "random_oversample": 2, "smote": 3}
RECALL_NAMES = [f"recall_{LABELS[i].lower()}" for i in LABELS]


def _fit_resample(X, y, strategy: str, seed: int):
    if strategy in ("none", "class_weight"):
        return X, y
    if strategy == "random_oversample":
        return RandomOverSampler(random_state=seed).fit_resample(X, y)
    if strategy == "smote":
        return SMOTE(random_state=seed, k_neighbors=5).fit_resample(X, y)
    raise ValueError(f"未知策略：{strategy}")


def _classifier(strategy: str) -> LogisticRegression:
    return LogisticRegression(
        C=1.0,
        max_iter=2000,
        solver="lbfgs",
        class_weight="balanced" if strategy == "class_weight" else None,
        random_state=SEED,
    )


def _align_proba(clf: LogisticRegression, X) -> np.ndarray:
    raw = clf.predict_proba(X)
    proba = np.zeros((len(X), len(LABELS)), dtype=float)
    for column, label in enumerate(clf.classes_):
        proba[:, int(label)] = raw[:, column]
    return proba


def _recall_columns(y_true, y_pred) -> dict:
    _, recall, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(LABELS), average=None, zero_division=0
    )
    return {name: float(score) for name, score in zip(RECALL_NAMES, recall)}


def _oof_and_test(dev, test, y_dev, y_test, strategy: str):
    """返回 OOF 概率、测试概率，以及五折训练时间之和。测试概率只用 full 向量，不参与选策略。"""
    oof = np.zeros((len(dev), len(LABELS)), dtype=float)
    seconds = 0.0
    dev_ids = dev["id"].to_numpy()
    folds = dev["fold"].to_numpy()
    for fold in range(N_FOLDS):
        matrix = mean_matrix(f"fold{fold}", dev_ids)
        train = folds != fold
        valid = ~train
        started = time.perf_counter()
        X_fit, y_fit = _fit_resample(matrix[train], y_dev[train], strategy, SEED + fold)
        clf = _classifier(strategy)
        clf.fit(X_fit, y_fit)
        seconds += time.perf_counter() - started
        oof[valid] = _align_proba(clf, matrix[valid])

    full_ids = np.concatenate([dev_ids, test["id"].to_numpy()])
    full = mean_matrix("full", full_ids)
    X_dev, X_test = full[: len(dev)], full[len(dev) :]
    X_fit, y_fit = _fit_resample(X_dev, y_dev, strategy, SEED)
    clf = _classifier(strategy)
    clf.fit(X_fit, y_fit)
    return oof, _align_proba(clf, X_test), seconds


def choose_strategy(cv_rows: pd.DataFrame) -> str:
    """macro-F1 最高；差距在 TIE_MARGIN 以内时选更简单的做法。"""
    ranked = cv_rows.sort_values(["macro_f1", "strategy"], ascending=[False, True])
    best_score = float(ranked.iloc[0]["macro_f1"])
    close = ranked[ranked["macro_f1"] >= best_score - TIE_MARGIN]
    close = close.assign(simplicity=close["strategy"].map(SIMPLICITY))
    return str(close.sort_values(["simplicity", "macro_f1"], ascending=[True, False]).iloc[0]["strategy"])


def plot_comparison(cv_rows: pd.DataFrame, recommended: str, path: Path) -> None:
    metrics = RECALL_NAMES + ["macro_f1"]
    tick_labels = ["Neutral recall", "Positive recall", "Negative recall", "Macro-F1"]
    x = np.arange(len(metrics))
    width = 0.18
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for index, strategy in enumerate(STRATEGIES):
        row = cv_rows.loc[cv_rows["strategy"] == strategy].iloc[0]
        offset = (index - (len(STRATEGIES) - 1) / 2) * width
        bars = ax.bar(x + offset, [row[metric] for metric in metrics], width=width, label=STRATEGY_LABELS[strategy])
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                ha="center",
                va="bottom",
                fontsize=7,
            )
    ax.set_xticks(x, tick_labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title(f"Imbalance strategies on FastText + logistic regression (CV)\nRecommended: {STRATEGY_LABELS[recommended]}")
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_imbalance() -> pd.DataFrame:
    clean, splits, dev, test, label_of, _ = load_frames()
    ensure_fasttext(clean, splits)
    y_dev = labels_for(dev["id"], label_of)
    y_test = labels_for(test["id"], label_of)
    rows = []
    stored = {}
    for strategy in STRATEGIES:
        print(f"imbalance strategy: {strategy}", flush=True)
        oof, test_proba, seconds = _oof_and_test(dev, test, y_dev, y_test, strategy)
        stored[strategy] = (oof, test_proba, seconds)
        oof_pred = oof.argmax(axis=1)
        run_id = new_run_id("logreg", "fasttext_mean")
        row = evaluate(
            y_dev,
            oof_pred,
            run_id=run_id,
            model="logreg",
            embedding="fasttext_mean",
            split="cv",
            params={"strategy": strategy, "C": 1.0, "solver": "lbfgs", "max_iter": 2000},
            train_seconds=seconds,
            notes="Fold-wise FastText mean vectors. Resampling stays inside the training fold. Strategy chosen on this CV score.",
        )
        row.update(_recall_columns(y_dev, oof_pred))
        row["strategy"] = strategy
        rows.append(row)

    cv = pd.DataFrame(rows)
    recommended = choose_strategy(cv)
    print(f"recommended strategy: {recommended}", flush=True)

    saved = list(rows)
    for row in saved:
        strategy = row["strategy"]
        oof, test_proba, _ = stored[strategy]
        save_preds(
            row["run_id"],
            oof_ids=dev["id"].to_numpy(),
            oof_proba=oof,
            test_ids=test["id"].to_numpy(),
            test_proba=test_proba,
            classes=list(LABELS),
        )
        save_result(row, "imbalance")
        if strategy != recommended:
            continue
        test_pred = test_proba.argmax(axis=1)
        test_row = evaluate(
            y_test,
            test_pred,
            run_id=row["run_id"],
            model="logreg",
            embedding="fasttext_mean",
            split="test",
            params={"strategy": strategy, "C": 1.0, "solver": "lbfgs", "max_iter": 2000},
            notes="Scored once, after the strategy was chosen on CV macro-F1.",
        )
        test_row.update(_recall_columns(y_test, test_pred))
        test_row["strategy"] = strategy
        save_result(test_row, "imbalance")
        rows.append(test_row)

    figure = FIGURES_DIR / "analysis_imbalance.png"
    plot_comparison(cv, recommended, figure)
    print(f"wrote {figure}", flush=True)
    table = pd.DataFrame(rows)
    show = table.loc[table["split"] == "cv", ["strategy", "macro_f1", *RECALL_NAMES, "run_id"]]
    print(show.to_string(index=False), flush=True)
    return table


if __name__ == "__main__":
    run_imbalance()
