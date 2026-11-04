"""数据划分：先按组分出测试集，再把开发集分成 N_FOLDS 折。全组所有模型都用同一份划分。

方法（4 号 CAO Taihe 的设计）：按 id 排序后，用 StratifiedGroupKFold(1 / TEST_SIZE 份，打乱，种子 SEED)
取第 1 份作测试集（约 20%）；剩下的开发集再用 StratifiedGroupKFold(N_FOLDS 折) 分折。两步都按 group_id 分组，
同组推文不会跨训练/测试或分到不同折；分层让每份的类别比例接近总体。

函数一览：
- make_splits(clean)             生成划分，列为 id, group_id, split, fold（测试集 fold = -1）；结果与输入行顺序无关
- validate_splits(splits, clean) 在 check_splits 之外，再检查每个训练、验证和测试部分都有三个类别
- split_summary(clean, splits)   每个部分的条数、占比和类别数量，报告表 5 用
- write_splits(clean)            生成并保存 splits/folds_v<SPLIT_VERSION>.csv；文件已存在且内容不同时报错，不覆盖

维护人：4 号。
"""
from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .config import LABELS, N_FOLDS, SEED, SPLIT_VERSION, SPLITS_DIR, TEST_SIZE
from .data import SPLIT_COLUMNS, check_splits


def make_splits(clean: pd.DataFrame) -> pd.DataFrame:
    """按组分层：先取约 TEST_SIZE 的测试集，再把开发集分成 N_FOLDS 折。"""
    clean = clean.sort_values("id").reset_index(drop=True)
    outer = StratifiedGroupKFold(n_splits=round(1 / TEST_SIZE), shuffle=True, random_state=SEED)
    dev_idx, test_idx = next(outer.split(clean, clean["label"], clean["group_id"]))

    splits = clean[["id", "group_id"]].copy()
    splits["split"] = "train"
    splits["fold"] = -1
    splits.loc[test_idx, "split"] = "test"

    dev = clean.iloc[dev_idx]
    inner = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    for fold, (_, valid_idx) in enumerate(inner.split(dev, dev["label"], dev["group_id"])):
        splits.loc[dev.index[valid_idx], "fold"] = fold

    splits = splits[SPLIT_COLUMNS]
    validate_splits(splits, clean)
    return splits


def validate_splits(splits: pd.DataFrame, clean: pd.DataFrame) -> None:
    """check_splits 的全部检查，加上：每个训练、验证和测试部分都必须包含三个类别。"""
    check_splits(splits, clean)
    labels = splits.merge(clean[["id", "label"]], on="id", validate="one_to_one")
    expected = set(LABELS)
    dev = labels[labels["split"] == "train"]
    parts = {"test": labels[labels["split"] == "test"]}
    for fold in range(N_FOLDS):
        parts[f"fold {fold} validation"] = dev[dev["fold"] == fold]
        parts[f"fold {fold} training"] = dev[dev["fold"] != fold]
    for name, part in parts.items():
        if set(part["label"]) != expected:
            raise ValueError(f"{name} 缺少类别：{sorted(expected - set(part['label']))}")


def split_summary(clean: pd.DataFrame, splits: pd.DataFrame) -> pd.DataFrame:
    """每个部分的条数、占全部数据的比例和各类别数量。"""
    labels = splits.merge(clean[["id", "label"]], on="id", validate="one_to_one")
    part = labels["split"].where(labels["split"] == "test", "fold " + labels["fold"].astype(str))
    table = pd.crosstab(part, labels["label"].map(LABELS))[list(LABELS.values())]
    table.loc["development"] = table.drop(index="test").sum()
    table["records"] = table.sum(axis=1)
    table["share"] = (table["records"] / len(labels)).round(4)
    return table


def write_splits(clean: pd.DataFrame) -> pd.DataFrame:
    """生成划分并保存；同名文件已存在时只允许内容完全相同（划分发布后不能修改，要改就升级 SPLIT_VERSION）。"""
    splits = make_splits(clean)
    path = SPLITS_DIR / f"folds_v{SPLIT_VERSION}.csv"
    if path.exists():
        existing = pd.read_csv(path)
        if not existing.equals(splits):
            raise FileExistsError(f"{path.name} 已存在且内容不同：划分发布后不能修改，请升级 SPLIT_VERSION")
        return splits
    path.parent.mkdir(parents=True, exist_ok=True)
    splits.to_csv(path, index=False, lineterminator="\n")
    return splits
