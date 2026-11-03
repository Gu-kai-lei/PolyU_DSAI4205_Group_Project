"""读取 6 号两个实验共用的数据和按折 FastText。

嵌入文件写在 data/emb/，不进 Git。生成过程和 2 号的 build_fold_embeddings 相同：
第 k 折的 FastText 只用其余四折的文本训练，测试集文本不参与任何训练。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import load_embedding
from src.config import EMB_DIR, N_FOLDS
from src.data import load_clean, load_splits
from src.fasttext_emb import build_fold_embeddings, tokenize


def load_frames():
    """返回 clean、splits，以及按行对齐的训练集和测试集。"""
    clean = load_clean()
    splits = load_splits(clean=clean)
    dev = splits[splits["split"] == "train"].reset_index(drop=True)
    test = splits[splits["split"] == "test"].reset_index(drop=True)
    label_of = dict(zip(clean["id"].astype(int), clean["label"].astype(int)))
    text_of = dict(zip(clean["id"].astype(int), clean["text_norm"].astype(str)))
    return clean, splits, dev, test, label_of, text_of


def labels_for(ids, label_of: dict) -> np.ndarray:
    return np.array([label_of[int(i)] for i in ids], dtype=int)


def token_max_len(clean) -> int:
    """和 11_emb_fasttext 相同：text_norm 词数的 99 分位，向上取整。"""
    lengths = clean["text_norm"].map(lambda text: len(tokenize(str(text))))
    return int(np.ceil(np.percentile(lengths, 99)))


def _required_names() -> list[str]:
    names = [f"fasttext_mean_fold{k}.npy" for k in range(N_FOLDS)]
    names.append("fasttext_mean_full.npy")
    names += [f"fasttext_tok_fold{k}.npy" for k in range(N_FOLDS)]
    names += [f"fasttext_tok_fold{k}_mask.npy" for k in range(N_FOLDS)]
    names += ["fasttext_tok_full.npy", "fasttext_tok_full_mask.npy"]
    return names


def ensure_fasttext(clean, splits) -> int:
    """本地没有按折 FastText 时才训练。返回 token 序列的 max_len。"""
    max_len = token_max_len(clean)
    paths = [EMB_DIR / name for name in _required_names()]
    if all(path.exists() for path in paths):
        cached = int(np.load(EMB_DIR / "fasttext_tok_full.npy", mmap_mode="r").shape[1])
        if cached != max_len:
            raise ValueError(
                f"已有 token 序列的 max_len={cached}，按当前文本算出来是 {max_len}。"
                "请删掉 data/emb/fasttext_* 后重跑，避免和 2 号的设定不一致。"
            )
        print(f"reusing FastText cache in {EMB_DIR}, max_len={cached}")
        return cached
    print(f"training fold-wise FastText, max_len={max_len}")
    summary, _ = build_fold_embeddings(clean, splits, max_len)
    print(summary.to_string(index=False))
    return max_len


def mean_matrix(tag: str, ids) -> np.ndarray:
    return np.asarray(load_embedding(f"fasttext_mean_{tag}", ids), dtype=np.float32)


def token_batch(tag: str, ids) -> tuple[np.ndarray, np.ndarray]:
    """返回 float32 的 (N, max_len, dim) 和 bool mask。mask 为 True 的位置是真实 token。"""
    sequences = np.asarray(load_embedding(f"fasttext_tok_{tag}", ids), dtype=np.float32)
    mask = np.asarray(load_embedding(f"fasttext_tok_{tag}_mask", ids), dtype=bool)
    if not mask.any(axis=1).all():
        raise ValueError(f"fasttext_tok_{tag} 里有整行都是 padding 的推文")
    return sequences, mask
