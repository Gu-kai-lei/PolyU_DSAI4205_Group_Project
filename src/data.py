"""读取数据，并检查清洗数据和数据划分是否符合全组约定的格式。

函数一览：
- load_raw()                      读取原始 CSV，并加上稳定的 id 列
- load_clean(version)             读取清洗数据 clean_v<version>.csv，并检查格式
- load_splits(version, clean)     读取数据划分 folds_v<version>.csv；传入 clean 时检查两者是否对得上
- check_clean(df)                 检查清洗数据：必需列、id 唯一、标签合法、文本非空
- check_splits(splits, clean)     检查数据划分：必需列、折号合法、同一 group 不跨折、和清洗数据一致

接口约定（2 号和 4 号共同维护）：
- 清洗数据的列：id, group_id, text_raw, text_full, text_light, label
  - id 是原始文件的行号，永远指向同一条推文，任何版本都不重新编号
  - group_id 标记近似重复的推文，同一组必须落在同一折；没有近似重复的推文自成一组
- 划分文件的列：id, group_id, split（train / test）, fold（训练集 0–4，测试集 -1）
"""
from __future__ import annotations

import pandas as pd

from .config import DATA_ROOT, CLEAN_VERSION, LABELS, N_FOLDS, RAW_CSV, SPLIT_VERSION, SPLITS_DIR

CLEAN_COLUMNS = ["id", "group_id", "text_raw", "text_full", "text_light", "label"]
SPLIT_COLUMNS = ["id", "group_id", "split", "fold"]


def load_raw() -> pd.DataFrame:
    """读取原始 CSV：text_sentiment 改名为 label，并加上 id（原始文件的行号，之后所有文件都按它对齐）。"""
    df = pd.read_csv(RAW_CSV, encoding="utf-8")
    df = df.rename(columns={"text_sentiment": "label"})
    df.insert(0, "id", range(len(df)))
    return df


def load_clean(version: int = CLEAN_VERSION) -> pd.DataFrame:
    """读取 clean_v<version>/clean_v<version>.csv，并检查格式。"""
    df = pd.read_csv(DATA_ROOT / f"clean_v{version}" / f"clean_v{version}.csv", encoding="utf-8")
    check_clean(df)
    return df


def load_splits(version: int = SPLIT_VERSION, clean: pd.DataFrame | None = None) -> pd.DataFrame:
    """读取 02_splits.ipynb 产出的 folds_v<version>.csv；传入 clean 时同时检查 id 和 group_id 是否对得上。"""
    splits = pd.read_csv(SPLITS_DIR / f"folds_v{version}.csv")
    check_splits(splits, clean)
    return splits


def _require_columns(df: pd.DataFrame, columns: list[str], what: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{what}缺少列：{missing}")


def check_clean(df: pd.DataFrame) -> None:
    """检查清洗数据是否符合约定，不符合时直接报错并说明原因。"""
    _require_columns(df, CLEAN_COLUMNS, "清洗数据")
    if not df["id"].is_unique:
        raise ValueError("清洗数据的 id 有重复")
    unknown = set(df["label"]) - set(LABELS)
    if unknown:
        raise ValueError(f"清洗数据出现未知标签：{sorted(unknown)}")
    for col in ("text_full", "text_light"):
        empty = df[col].isna() | (df[col].astype(str).str.strip() == "")
        if empty.any():
            raise ValueError(f"{col} 有 {int(empty.sum())} 行为空，清洗时应删除或处理")


def check_splits(splits: pd.DataFrame, clean: pd.DataFrame | None = None) -> None:
    """检查数据划分是否符合约定；传入 clean 时还检查 id 和 group_id 是否与清洗数据一致。"""
    _require_columns(splits, SPLIT_COLUMNS, "数据划分")
    if not splits["id"].is_unique:
        raise ValueError("数据划分的 id 有重复")
    if not splits["split"].isin(["train", "test"]).all():
        raise ValueError("split 列只能是 train 或 test")
    is_test = splits["split"] == "test"
    if not (splits.loc[is_test, "fold"] == -1).all():
        raise ValueError("测试集的 fold 必须是 -1")
    if not splits.loc[~is_test, "fold"].isin(range(N_FOLDS)).all():
        raise ValueError(f"训练集的 fold 必须在 0–{N_FOLDS - 1} 之间")
    crossing = splits.groupby("group_id")["fold"].nunique() > 1
    if crossing.any():
        raise ValueError(f"有 {int(crossing.sum())} 个 group_id 分散在不同的折或训练/测试集，近似重复的推文会泄露")
    if clean is None:
        return
    merged = clean[["id", "group_id"]].merge(
        splits[["id", "group_id"]], on="id", how="outer", suffixes=("_clean", "_split"), indicator=True
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("划分和清洗数据的 id 不一致：清洗数据增删了行，必须重新生成划分并升级 SPLIT_VERSION")
    if (merged["group_id_clean"] != merged["group_id_split"]).any():
        raise ValueError("划分和清洗数据的 group_id 不一致：近似重复的分组改了，必须重新生成划分并升级 SPLIT_VERSION")
