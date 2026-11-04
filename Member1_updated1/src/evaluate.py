"""统一评分：所有模型都用这里的函数计算指标、保存结果，保证结果可以直接比较。

每次实验先用 new_run_id() 生成一个 run_id，评分记录和预测文件（src.cache.save_preds）都带上它。
这样同一个模型重跑多次，也能准确找到每次成绩对应的那一份预测。

函数一览：
- new_run_id("svm", "sbert_multi")           生成本次实验的 run_id，例如 svm__sbert_multi__20261027-153012-a1b2
- evaluate(y_true, y_pred, run_id=..., ...)  计算 Accuracy、宏平均 P/R/F1、各类别 F1，返回一行结果
- save_result(row, "m1_svm")                 把这行结果追加到 results/m1_svm.csv
- collect_results()                          把 results/ 下所有 csv 合并成一张对比表
- best_runs(metric="macro_f1", split="cv")   每个 (model, embedding) 组合取成绩最好的一次实验
- plot_confusion(y_true, y_pred, title)      画混淆矩阵，坐标轴显示类别名

一个模型 notebook 的典型写法：
    run_id = new_run_id("svm", "sbert_multi")
    row = evaluate(y_train, oof_pred, run_id=run_id, model="svm", embedding="sbert_multi",
                   split="cv", params={"C": 10, "kernel": "rbf"})
    save_result(row, "m1_svm")
    save_preds(run_id, oof_ids=train_ids, oof_proba=oof_proba,
               test_ids=test_ids, test_proba=test_proba, classes=clf.classes_)

维护人：4 号。改函数参数或返回值前先在群里说。
"""
from __future__ import annotations

import json
import uuid
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

from .config import LABELS, RESULTS_DIR, VERSION_TAGS

LABEL_IDS = list(LABELS)
LABEL_NAMES = [LABELS[i] for i in LABEL_IDS]
SPLITS = ("cv", "test")


def new_run_id(model: str, embedding: str) -> str:
    """生成本次实验的唯一编号：模型__嵌入__时间-随机后缀。"""
    return f"{model}__{embedding}__{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"


def evaluate(y_true, y_pred, *, run_id: str, model: str, embedding: str, split: str,
             params: dict | None = None, train_seconds: float | None = None, notes: str = "") -> dict:
    """计算各项指标，返回一行结果。split 填 "cv"（OOF 预测）或 "test"；params 记录本次实验的超参数。"""
    if split not in SPLITS:
        raise ValueError(f"split 只能是 {SPLITS}，收到的是 {split!r}")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABEL_IDS, average="macro", zero_division=0
    )
    per_class_f1 = f1_score(y_true, y_pred, labels=LABEL_IDS, average=None, zero_division=0)
    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "embedding": embedding,
        "split": split,
        **VERSION_TAGS,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        **{f"f1_{name.lower()}": score for name, score in zip(LABEL_NAMES, per_class_f1)},
        "params": json.dumps(params or {}, sort_keys=True, default=str),
        "train_seconds": train_seconds,
        "notes": notes,
    }


def save_result(row: dict, file_stem: str) -> Path:
    """把一行结果追加到 results/<file_stem>.csv。每人用自己的文件，避免合并冲突。"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{file_stem}.csv"
    new = pd.DataFrame([row])
    if path.exists():
        old = pd.read_csv(path)
        if ((old["run_id"] == row["run_id"]) & (old["split"] == row["split"])).any():
            raise ValueError(f"{path.name} 里已经有 run_id={row['run_id']}、split={row['split']} 的记录")
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)
    return path


def collect_results() -> pd.DataFrame:
    """合并 results/ 下所有 csv，source 列记录来自哪个文件。"""
    files = sorted(RESULTS_DIR.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f).assign(source=f.stem) for f in files], ignore_index=True)


def best_runs(metric: str = "macro_f1", split: str = "cv") -> pd.DataFrame:
    """每个 (model, embedding) 组合取 split 上 metric 最高的一次实验，结果里的 run_id 可直接传给 load_preds()。"""
    table = collect_results()
    if table.empty:
        return table
    table = table[table["split"] == split]
    best = table.loc[table.groupby(["model", "embedding"])[metric].idxmax()]
    return best.sort_values(metric, ascending=False).reset_index(drop=True)


def plot_confusion(y_true, y_pred, title: str, save_as: Path | None = None):
    """画混淆矩阵，坐标轴显示 Neutral / Positive / Negative；传入 save_as 时同时保存图片。"""
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
