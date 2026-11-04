"""4 号的实验：自训 FastText 平均向量 + MLP（三组参数），以及一个 TF-IDF + 逻辑回归参照。只用开发集，不给测试集评分。

流程（CAO Taihe 的设计）：
1. 读取清洗数据和固定划分，按折训练 FastText（src.fasttext_emb，缓存到 DATA_ROOT/emb）
2. 每组参数、每个外层折：在外层训练部分里再按组分出一份内层验证集选早停轮数，
   再用完整的外层训练部分按这个轮数重新训练，预测外层验证折，得到 OOF 概率
3. 标准化只在对应的训练行上拟合，类别权重只用训练行计算
4. 按 OOF Macro-F1 选出最好的一组；用全部开发集按各折轮数的中位数重新训练，只保存测试集概率，不评分
5. 成绩写入 results/m4_mlp.csv，概率按 run_id 保存，证据表写入 Member4_file/evidence/，图写入 figures/

三组参数在同一份外层交叉验证上比较，选出来的分数偏乐观；真正的泛化表现要等实验冻结后看测试集。

入口：run() 返回概况字典（与 Member4_file/summary.json 的结构一致）。

维护人：4 号。
"""
from __future__ import annotations

import json
import platform
import time
from importlib.metadata import version

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .cache import load_embedding
from .config import FIGURES_DIR, LABELS, N_FOLDS, PROJECT_ROOT, SEED
from .data import load_clean, load_splits
from .evaluate import evaluate, new_run_id, plot_confusion, save_result
from .fasttext_emb import build_fold_embeddings, tokenize
from .training import MLP, TrainConfig, fit_model, predict_proba, save_predictions, seed_everything

EVIDENCE_DIR = PROJECT_ROOT / "Member4_file" / "evidence"
EMBEDDING = "fasttext_mean"
SHARED = {"hidden": (512, 256), "batch_size": 64, "max_epochs": 50, "patience": 7}
CANDIDATES = [
    {"name": "initial", "dropout": 0.5, "lr": 1e-3, "weight_decay": 0.01, "class_weight": False},
    {"name": "regularized", "dropout": 0.3, "lr": 5e-4, "weight_decay": 0.05, "class_weight": False},
    {"name": "weighted", "dropout": 0.3, "lr": 5e-4, "weight_decay": 0.05, "class_weight": True},
]


def _prepare_embeddings(clean: pd.DataFrame, splits: pd.DataFrame) -> float:
    """按当前划分重新训练按折的 FastText（约半分钟），保证嵌入和划分一致。返回耗时（秒）。"""
    lengths = clean["text_norm"].map(lambda s: len(tokenize(s)))
    started = time.perf_counter()
    build_fold_embeddings(clean, splits, int(np.ceil(np.percentile(lengths, 99))))
    return time.perf_counter() - started


def _train(x_fit, y_fit, x_stop, y_stop, candidate: dict, epochs: int):
    """标准化只用 x_fit 拟合；给了 x_stop 时早停，否则固定训练 epochs 轮。"""
    scaler = StandardScaler().fit(x_fit)
    seed_everything(SEED)
    model = MLP(x_fit.shape[1], hidden=SHARED["hidden"], dropout=candidate["dropout"])
    config = TrainConfig(epochs=epochs, patience=SHARED["patience"], batch_size=SHARED["batch_size"],
                         lr=candidate["lr"], weight_decay=candidate["weight_decay"],
                         class_weight=candidate["class_weight"])
    result = fit_model(model, scaler.transform(x_fit), y_fit,
                       None if x_stop is None else scaler.transform(x_stop), y_stop, config=config)
    return model, scaler, result


def _cross_validate(candidate: dict, dev: pd.DataFrame):
    """外层 5 折 OOF；每折内再按组分出内层验证集选早停轮数，然后在完整外层训练部分上重训。"""
    y = dev["label"].to_numpy()
    oof = np.zeros((len(dev), len(LABELS)))
    epochs_by_fold, folds, history = [], [], []
    started = time.perf_counter()
    for k in range(N_FOLDS):
        in_valid = (dev["fold"] == k).to_numpy()
        train, valid = dev[~in_valid], dev[in_valid]
        x_train = load_embedding(f"{EMBEDDING}_fold{k}", ids=train["id"])
        y_train = y[~in_valid]
        inner = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        fit_idx, stop_idx = next(inner.split(x_train, y_train, train["group_id"]))
        _, _, early = _train(x_train[fit_idx], y_train[fit_idx], x_train[stop_idx], y_train[stop_idx],
                             candidate, SHARED["max_epochs"])
        model, scaler, _ = _train(x_train, y_train, None, None, candidate, early["best_epoch"])
        oof[in_valid] = predict_proba(model, scaler.transform(load_embedding(f"{EMBEDDING}_fold{k}", ids=valid["id"])))
        epochs_by_fold.append(early["best_epoch"])
        pred = oof[in_valid].argmax(1)
        folds.append({"candidate": candidate["name"], "fold": k, "n_valid": int(in_valid.sum()),
                      "best_epoch": early["best_epoch"], "accuracy": float((pred == y[in_valid]).mean()),
                      "macro_f1": f1_score(y[in_valid], pred, labels=list(LABELS), average="macro", zero_division=0)})
        history += [{"candidate": candidate["name"], "fold": k, **row} for row in early["history"]]
    return oof, epochs_by_fold, folds, history, time.perf_counter() - started


def _baseline_reference(dev: pd.DataFrame) -> dict:
    """TF-IDF + 逻辑回归（与 1 号 baseline 设置相同），只作为同折参照，不写入 results/。"""
    oof = np.zeros(len(dev), dtype=int)
    started = time.perf_counter()
    for k in range(N_FOLDS):
        in_valid = (dev["fold"] == k).to_numpy()
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 1), min_df=1),
                              LogisticRegression(C=1.0, max_iter=2000, random_state=SEED))
        model.fit(dev.loc[~in_valid, "text_norm"], dev.loc[~in_valid, "label"])
        oof[in_valid] = model.predict(dev.loc[in_valid, "text_norm"])
    return evaluate(dev["label"], oof, run_id=new_run_id("baseline_reference_m4", "tfidf"), model="baseline_reference_m4",
                    embedding="tfidf", split="cv", params={"C": 1.0, "max_iter": 2000, "ngram_range": [1, 1]},
                    train_seconds=time.perf_counter() - started,
                    notes="independent same-fold reference; does not replace member 1's results; no test evaluation")


def _figures(candidates: pd.DataFrame, reference: dict, dev: pd.DataFrame, oof: np.ndarray, history: pd.DataFrame,
             selected: str) -> None:
    """三张图：候选和参照的 Macro-F1、所选模型的 OOF 混淆矩阵、内层验证的学习曲线。"""
    surface, ink, ink_2, axis = "#fcfcfb", "#0b0b0b", "#52514e", "#c3c2b7"
    names = list(candidates["name"]) + ["TF-IDF + LR reference"]
    scores = list(candidates["macro_f1"]) + [reference["macro_f1"]]
    fig, ax = plt.subplots(figsize=(6.8, 2.4), facecolor=surface)
    ax.set_facecolor(surface)
    ax.barh(names, scores, height=0.55, color=["#2a78d6"] * len(candidates) + ["#52514e"])
    for y, s in enumerate(scores):
        ax.text(s + 0.005, y, f"{s:.3f}", va="center", fontsize=8, color=ink_2)
    ax.invert_yaxis()
    ax.set_xlim(0, max(scores) * 1.15)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(axis)
    ax.tick_params(colors=ink_2, labelsize=8, length=0)
    ax.set_xlabel("development OOF macro-F1", fontsize=8, color=ink_2)
    ax.set_title("FastText + MLP candidates and the same-fold reference", loc="left", fontsize=10, color=ink)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "m4_comparison.png", dpi=200, facecolor=surface)
    plt.close(fig)

    fig = plot_confusion(dev["label"], oof.argmax(1), f"MLP ({selected}), out-of-fold",
                         save_as=FIGURES_DIR / "m4_oof_confusion.png")
    plt.close(fig)

    curves = history[history["candidate"] == selected]
    fig, ax = plt.subplots(figsize=(6.8, 2.8), facecolor=surface)
    ax.set_facecolor(surface)
    for k, part in curves.groupby("fold"):
        ax.plot(part["epoch"], part["valid_macro_f1"], linewidth=1.5, label=f"fold {k}")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=ink_2, labelsize=8)
    ax.set_xlabel("epoch", fontsize=8, color=ink_2)
    ax.set_ylabel("inner validation macro-F1", fontsize=8, color=ink_2)
    ax.set_title(f"Early stopping on the inner split ({selected})", loc="left", fontsize=10, color=ink)
    ax.legend(frameon=False, fontsize=7, ncol=5, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "m4_learning_curves.png", dpi=200, facecolor=surface)
    plt.close(fig)


def run() -> dict:
    """完整跑一遍 4 号的实验，返回概况字典。"""
    total_started = time.perf_counter()
    clean = load_clean()
    splits = load_splits(clean=clean)
    data = splits.merge(clean[["id", "label", "text_norm"]], on="id", validate="one_to_one")
    dev = data[data["split"] == "train"].reset_index(drop=True)
    test = data[data["split"] == "test"].reset_index(drop=True)
    embedding_seconds = _prepare_embeddings(clean, splits)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    rows, oofs, epochs, fold_rows, history = [], {}, {}, [], []
    for candidate in CANDIDATES:
        oof, epochs_by_fold, folds, hist, seconds = _cross_validate(candidate, dev)
        params = {**candidate, "hidden": list(SHARED["hidden"]), "batch_size": SHARED["batch_size"],
                  "max_epochs": SHARED["max_epochs"], "early_stopping": "grouped inner split, patience 7, macro_f1",
                  "epochs_by_fold": epochs_by_fold, "optimizer": "AdamW", "scaler": "training-only StandardScaler",
                  "seed": SEED}
        row = evaluate(dev["label"], oof.argmax(1), run_id=new_run_id("mlp", EMBEDDING), model="mlp",
                       embedding=EMBEDDING, split="cv", params=params, train_seconds=seconds,
                       notes="pooled outer OOF; inner early stopping; selected on the same outer CV, so the best score is optimistic")
        save_result(row, "m4_mlp")
        rows.append({"name": candidate["name"], **row})
        oofs[candidate["name"]], epochs[candidate["name"]] = oof, epochs_by_fold
        fold_rows += folds
        history += hist

    candidates = pd.DataFrame(rows)
    best = candidates.loc[candidates["macro_f1"].idxmax()]
    selected = next(c for c in CANDIDATES if c["name"] == best["name"])

    # Final refit on the whole development set; test probabilities are saved for stacking but not scored
    refit_epochs = int(np.median(epochs[selected["name"]]))
    x_dev = load_embedding(f"{EMBEDDING}_full", ids=dev["id"])
    model, scaler, _ = _train(x_dev, dev["label"].to_numpy(), None, None, selected, refit_epochs)
    test_proba = predict_proba(model, scaler.transform(load_embedding(f"{EMBEDDING}_full", ids=test["id"])))
    for name, oof in oofs.items():
        run_id = candidates.loc[candidates["name"] == name, "run_id"].iloc[0]
        if name == selected["name"]:
            save_predictions(run_id, dev["id"], oof, test["id"], test_proba)
        else:
            save_predictions(run_id, dev["id"], oof, [], np.empty((0, len(LABELS))))

    reference = _baseline_reference(dev)
    folds = pd.DataFrame(fold_rows)
    history = pd.DataFrame(history)
    selected_oof = oofs[selected["name"]]
    oof_table = dev[["id", "group_id", "fold", "label"]].assign(prediction=selected_oof.argmax(1))
    for i, name in LABELS.items():
        oof_table[f"prob_{name.lower()}"] = selected_oof[:, i]
    report = classification_report(dev["label"], selected_oof.argmax(1), labels=list(LABELS),
                                   target_names=list(LABELS.values()), output_dict=True, zero_division=0)

    candidates.drop(columns="name").to_csv(EVIDENCE_DIR / "candidate_metrics.csv", index=False)
    folds.to_csv(EVIDENCE_DIR / "fold_metrics.csv", index=False)
    folds.groupby("candidate")["macro_f1"].agg(["mean", "std", "min", "max"]).reset_index().to_csv(
        EVIDENCE_DIR / "fold_statistics.csv", index=False)
    pd.DataFrame([reference]).to_csv(EVIDENCE_DIR / "baseline_reference.csv", index=False)
    history.to_csv(EVIDENCE_DIR / "training_history.csv", index=False)
    oof_table.to_csv(EVIDENCE_DIR / "selected_oof_predictions.csv", index=False)
    pd.DataFrame(report).T.to_csv(EVIDENCE_DIR / "selected_classification_report.csv")
    _figures(candidates, reference, dev, selected_oof, history, selected["name"])

    packages = ["numpy", "pandas", "scikit-learn", "torch", "gensim", "matplotlib"]
    summary = {
        "embedding": EMBEDDING, "development_rows": len(dev), "test_rows": len(test), "holdout_evaluated": False,
        "selected_source_run": best["run_id"], "selected_parameters": {**selected, **SHARED,
                                                                        "epochs_by_fold": epochs[selected["name"]]},
        "full_refit_epochs": refit_epochs,
        "selected_cv": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.drop(labels="name").items()},
        "baseline_reference": reference,
        "embedding_seconds": embedding_seconds, "total_wall_seconds": time.perf_counter() - total_started,
        "environment": {"python": platform.python_version(), "packages": {p: version(p) for p in packages}},
        "caution": "Three configurations compared on the same outer CV: the selected score is optimistic; the test split is not scored.",
    }
    (EVIDENCE_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
