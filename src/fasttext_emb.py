"""自训 FastText 嵌入：按折训练，生成句向量（平均、SIF）和给序列模型用的 token 向量序列。

为什么按折训练：第 k 折的验证集文本不能参与训练第 k 折用的词向量，否则交叉验证分数会偏高；
测试集文本任何时候都不参与训练。

保存的嵌入（都通过 src.cache.save_embedding，按 id 对齐）：
- fasttext_mean_fold{k} / fasttext_sif_fold{k}    第 k 折的句向量，覆盖训练集全部推文（第 k 折模型只用其余 4 折训练）
- fasttext_tok_fold{k} 和 fasttext_tok_fold{k}_mask   第 k 折的 token 向量序列 (N, max_len, dim) 和 mask，给 5、6 号
- 以上每种的 _full 版本：用全部训练集训练，覆盖训练集和测试集，用来预测测试集

函数一览：
- tokenize(text_norm)                            text_norm 按空格切词
- train_fasttext(token_lists)                    在给定文本上训练 FastText（skip-gram，结果可复现）
- mean_vectors(model, token_lists)               每条推文的词向量平均
- SIFEncoder(model, train_tokens).encode(...)    SIF 加权平均（词频只用训练部分统计），再去掉第一主成分
- token_sequences(model, token_lists, max_len)   token 向量序列和 mask
- build_fold_embeddings(clean, splits, max_len)  按折生成上面所有嵌入并保存，返回 (每个版本的概况, full 版本的模型)

维护人：2 号。
"""
from __future__ import annotations

import zlib
from collections import Counter

import numpy as np
import pandas as pd

from .cache import save_embedding
from .config import N_FOLDS, SEED

FASTTEXT_PARAMS = {
    "vector_size": 100,
    "window": 5,
    "min_count": 2,   # 只出现一次的词不单独建向量，靠子词 n-gram 拼出来
    "sg": 1,          # skip-gram：小语料上通常比 CBOW 好
    "min_n": 3,
    "max_n": 5,
    "epochs": 50,     # 语料只有一千多条推文，多训练几轮
    "bucket": 200_000,  # 子词哈希桶数；gensim 默认 200 万个桶，光这张表就有 800 MB，对小语料没有必要
}


def _stable_hash(text) -> int:
    """gensim 默认用 Python 自带的 hash()，每次启动结果都不同；换成固定的 crc32 保证可复现。"""
    return zlib.crc32(str(text).encode("utf-8"))


def tokenize(text_norm: str) -> list[str]:
    """text_norm 已经是小写、去标点、词形还原过的文本，按空格切开即可。"""
    return text_norm.split()


def train_fasttext(token_lists: list[list[str]], params: dict | None = None):
    """在给定文本上训练 FastText。单线程 + 固定种子 + 固定哈希，同样的输入总是得到同样的向量。"""
    from gensim.models import FastText

    return FastText(sentences=token_lists, seed=SEED, workers=1, hashfxn=_stable_hash,
                    **(params or FASTTEXT_PARAMS))


def _token_matrix(model, tokens: list[str]) -> np.ndarray:
    # FastText 能给训练时没见过的词拼出向量（子词 n-gram），所以不会有缺失
    return np.stack([model.wv[t] for t in tokens])


def mean_vectors(model, token_lists: list[list[str]]) -> np.ndarray:
    """每条推文所有词向量的平均，形状 (N, dim)。"""
    out = np.zeros((len(token_lists), model.wv.vector_size), dtype=np.float32)
    for i, tokens in enumerate(token_lists):
        if tokens:
            out[i] = _token_matrix(model, tokens).mean(axis=0)
    return out


class SIFEncoder:
    """SIF 句向量（Arora et al., 2017）：词向量按 a / (a + p(w)) 加权平均，再去掉所有句向量共有的第一主成分。

    p(w) 和第一主成分都只用训练部分计算，再套用到验证集和测试集。
    """

    def __init__(self, model, train_tokens: list[list[str]], a: float = 1e-3):
        from sklearn.decomposition import TruncatedSVD

        self.model = model
        counts = Counter(t for tokens in train_tokens for t in tokens)
        total = sum(counts.values())
        self.weights = {w: a / (a + c / total) for w, c in counts.items()}
        train_vectors = self._weighted_mean(train_tokens)
        self.component = TruncatedSVD(n_components=1, random_state=SEED).fit(train_vectors).components_[0]

    def _weighted_mean(self, token_lists: list[list[str]]) -> np.ndarray:
        out = np.zeros((len(token_lists), self.model.wv.vector_size), dtype=np.float32)
        for i, tokens in enumerate(token_lists):
            if tokens:
                w = np.array([self.weights.get(t, 1.0) for t in tokens], dtype=np.float32)  # 训练部分没出现过的词权重为 1
                out[i] = (w[:, None] * _token_matrix(self.model, tokens)).mean(axis=0)
        return out

    def encode(self, token_lists: list[list[str]]) -> np.ndarray:
        vectors = self._weighted_mean(token_lists)
        return vectors - np.outer(vectors @ self.component, self.component)


def token_sequences(model, token_lists: list[list[str]], max_len: int) -> tuple[np.ndarray, np.ndarray]:
    """每条推文的 token 向量序列 (N, max_len, dim)，float16；超出 max_len 的部分截断，不足的补 0。

    mask (N, max_len) 为 True 的位置是真实 token，padding 为 False。
    """
    dim = model.wv.vector_size
    sequences = np.zeros((len(token_lists), max_len, dim), dtype=np.float16)
    mask = np.zeros((len(token_lists), max_len), dtype=bool)
    for i, tokens in enumerate(token_lists):
        tokens = tokens[:max_len]
        if tokens:
            sequences[i, :len(tokens)] = _token_matrix(model, tokens)
            mask[i, :len(tokens)] = True
    return sequences, mask


def build_fold_embeddings(clean: pd.DataFrame, splits: pd.DataFrame, max_len: int):
    """按折训练 FastText 并保存全部嵌入，返回 (每个版本的概况, full 版本的模型)。

    clean 是 load_clean() 的输出，splits 是 load_splits() 的输出。概况包括训练文本数、词表大小和编码的推文数；
    full 版本的模型只返回、不保存（子词表很大），用来查看近邻词。
    """
    tokens = dict(zip(clean["id"], clean["text_norm"].map(tokenize)))
    dev = splits[splits["split"] == "train"]
    dev_ids = dev["id"].to_numpy()
    all_ids = splits["id"].to_numpy()

    versions = [(f"fold{k}", dev.loc[dev["fold"] != k, "id"].to_numpy(), dev_ids) for k in range(N_FOLDS)]
    versions.append(("full", dev_ids, all_ids))

    rows = []
    for tag, train_ids, encode_ids in versions:
        train_tokens = [tokens[i] for i in train_ids]
        encode_tokens = [tokens[i] for i in encode_ids]
        model = train_fasttext(train_tokens)

        save_embedding(f"fasttext_mean_{tag}", encode_ids, mean_vectors(model, encode_tokens))
        save_embedding(f"fasttext_sif_{tag}", encode_ids, SIFEncoder(model, train_tokens).encode(encode_tokens))
        sequences, mask = token_sequences(model, encode_tokens, max_len)
        save_embedding(f"fasttext_tok_{tag}", encode_ids, sequences)
        save_embedding(f"fasttext_tok_{tag}_mask", encode_ids, mask)

        rows.append({"version": tag, "trained on": len(train_ids), "vocabulary": len(model.wv.key_to_index),
                     "tweets encoded": len(encode_ids)})
    return pd.DataFrame(rows), model  # 循环最后一个版本就是 full
