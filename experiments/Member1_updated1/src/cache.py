"""嵌入和预测结果的读写。文件存在共享网盘的 DATA_ROOT 下，不进 Git。

每个文件都连同数据的 id 一起保存，读取时按 id 对齐，避免不同人的文件行顺序错位。

函数一览：
- save_embedding(name, ids, matrix)      保存嵌入，例如 SBERT 句向量、FastText 向量、token 序列
- load_embedding(name, ids=None)         读取嵌入；传入 ids 时按这些 id 的顺序返回
- save_preds(run_id, oof_ids=..., ...)   保存一次实验的 OOF 概率和测试集概率，文件名就是 run_id，不会覆盖
- load_preds(run_id)                     读取某次实验的预测，附带数据版本、划分版本、随机种子和类别顺序
- list_preds()                           列出已保存预测的所有 run_id

嵌入命名约定：
- 外部预训练且冻结的模型（如 SBERT、token 序列）：不用训练，全组共用一份，例如 sbert_multi
- 自己训练的嵌入（FastText 等）：交叉验证时第 k 折只能用该折的训练部分训练，按折保存为
  <name>_fold0 … <name>_fold4；预测测试集用的版本用全部训练集训练，保存为 <name>_full。
  测试集的文本任何时候都不参与训练。

维护人：4 号。
"""
from __future__ import annotations

import numpy as np

from .config import EMB_DIR, LABELS, PRED_DIR, VERSION_TAGS

LABEL_IDS = list(LABELS)
N_CLASSES = len(LABEL_IDS)


def save_embedding(name: str, ids, matrix) -> None:
    """保存嵌入矩阵和对应的 id，文件为 EMB_DIR/<name>.npy 和 EMB_DIR/<name>_ids.npy。"""
    ids, matrix = np.asarray(ids), np.asarray(matrix)
    if len(ids) != len(matrix):
        raise ValueError(f"ids 有 {len(ids)} 行，matrix 有 {len(matrix)} 行，行数必须一致")
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMB_DIR / f"{name}.npy", matrix)
    np.save(EMB_DIR / f"{name}_ids.npy", ids)


def load_embedding(name: str, ids=None):
    """不传 ids 时返回 (ids, matrix)；传入 ids 时只返回按这些 id 顺序排好的 matrix。"""
    matrix = np.load(EMB_DIR / f"{name}.npy")
    cached_ids = np.load(EMB_DIR / f"{name}_ids.npy")
    if ids is None:
        return cached_ids, matrix
    position = {int(i): k for k, i in enumerate(cached_ids)}
    missing = [int(i) for i in ids if int(i) not in position]
    if missing:
        raise KeyError(f"{name} 里缺少 {len(missing)} 个 id，例如 {missing[:5]}")
    return matrix[[position[int(i)] for i in ids]]


def save_preds(run_id: str, *, oof_ids, oof_proba, test_ids, test_proba, classes) -> None:
    """保存一次实验的 OOF 概率和测试集概率，文件为 PRED_DIR/<run_id>.npz。

    classes 是概率各列对应的标签，sklearn 模型直接传 clf.classes_，PyTorch 模型传 [0, 1, 2]；
    保存时会统一把列排成 LABELS 的顺序（0 Neutral, 1 Positive, 2 Negative）。
    """
    classes = [int(c) for c in classes]
    if sorted(classes) != LABEL_IDS:
        raise ValueError(f"classes 应该恰好包含 {LABEL_IDS}，收到的是 {classes}")
    order = [classes.index(label) for label in LABEL_IDS]

    arrays = {}
    for part, ids, proba in (("oof", oof_ids, oof_proba), ("test", test_ids, test_proba)):
        proba = np.asarray(proba, dtype=float)
        if proba.shape != (len(ids), N_CLASSES):
            raise ValueError(f"{part} 概率的形状应为 ({len(ids)}, {N_CLASSES})，实际是 {proba.shape}")
        arrays[f"{part}_ids"] = np.asarray(ids)
        arrays[f"{part}_proba"] = proba[:, order]

    path = PRED_DIR / f"{run_id}.npz"
    if path.exists():
        raise FileExistsError(f"{path.name} 已存在：每次实验请用 new_run_id() 生成新的 run_id")
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays, classes=np.asarray(LABEL_IDS), run_id=run_id, **VERSION_TAGS)


def load_preds(run_id: str) -> dict:
    """读取某次实验的预测：oof_ids、oof_proba、test_ids、test_proba、classes，以及 run_id 和版本信息。"""
    with np.load(PRED_DIR / f"{run_id}.npz") as f:
        return {key: f[key].item() if f[key].ndim == 0 else f[key] for key in f.files}


def list_preds() -> list[str]:
    """列出 PRED_DIR 里已保存预测的所有 run_id。"""
    return sorted(p.stem for p in PRED_DIR.glob("*.npz"))
