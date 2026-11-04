"""PyTorch 训练模板：给 MLP（4 号）、TextCNN（5 号）和 BiLSTM（6 号）共用。用法见 Member4_file/TRAINING_GUIDE.md。

约定：
- 输入是句向量 (N, D) 或 token 序列 (N, L, D)；序列的 mask 是 (N, L) 的 bool，True 表示真实 token
- 模型实现 forward(x, mask=None)，返回 (N, 3) 的原始 logits，不做 softmax（训练用 CrossEntropyLoss，预测时再 softmax）
- 先调用 seed_everything(42) 再构造模型，同一台机器、同一套包版本下结果可复现

函数一览：
- seed_everything(seed)                       固定 random、NumPy 和 PyTorch 的随机种子
- TrainConfig(...)                            训练参数：轮数、早停耐心、批大小、学习率、权重衰减、类别权重等
- MLP(input_dim, hidden, dropout)             两层（或多层）全连接 + BatchNorm + ReLU + Dropout
- masked_mean(x, mask) / masked_max(x, mask)  忽略 padding 的平均和最大池化；全是 padding 的样本返回 0
- fit_model(model, x_fit, y_fit, x_stop, y_stop, ...)
                                              AdamW 训练；给了 x_stop 就按其 Macro-F1 早停并恢复最好的权重，
                                              不给就固定训练 config.epochs 轮（用于按选好的轮数重新训练）
- predict_proba(model, x, mask=None)          预测概率，列顺序为 [0, 1, 2]
- save_predictions(run_id, ...)               按团队约定保存 OOF 和测试集概率（调用 src.cache.save_preds）

维护人：4 号。
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .cache import save_preds
from .config import LABELS, SEED

N_CLASSES = len(LABELS)


def seed_everything(seed: int = SEED) -> None:
    """固定所有随机源。必须在构造模型之前调用。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


@dataclass
class TrainConfig:
    epochs: int = 50            # 最多训练轮数；不早停时就是实际轮数
    patience: int = 7           # 验证 Macro-F1 连续这么多轮不提升就停
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.01
    class_weight: bool = False  # True 时按训练部分的类别频率加权（只用训练数据计算）
    clip_norm: float = 1.0      # 梯度裁剪；0 表示不裁剪
    device: str = "cpu"
    seed: int = SEED            # 控制每轮打乱顺序


class MLP(nn.Module):
    """全连接分类器：每个隐藏层是 Linear → BatchNorm → ReLU → Dropout。"""

    def __init__(self, input_dim: int, hidden=(512, 256), dropout: float = 0.5, n_classes: int = N_CLASSES):
        super().__init__()
        layers, previous = [], input_dim
        for width in hidden:
            layers += [nn.Linear(previous, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout)]
            previous = width
        layers.append(nn.Linear(previous, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x, mask=None):
        return self.net(x)


def masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(N, L, D) 按 mask 求平均，得到 (N, D)；没有真实 token 的样本返回 0。"""
    weight = mask.unsqueeze(-1).to(x.dtype)
    return (x * weight).sum(1) / weight.sum(1).clamp(min=1)


def masked_max(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """(N, L, D) 按 mask 求最大值，得到 (N, D)；没有真实 token 的样本返回 0。"""
    pooled = x.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
    return pooled.masked_fill(~mask.any(1).unsqueeze(-1), 0.0)


def _to_tensor(array, dtype):
    return None if array is None else torch.as_tensor(np.asarray(array), dtype=dtype)


def _batches(n: int, batch_size: int, generator: torch.Generator) -> list[list[int]]:
    """打乱后切成批；最后一批只有 1 条时并入前一批，避免 BatchNorm 在单个样本上报错。"""
    order = torch.randperm(n, generator=generator).tolist()
    batches = [order[i:i + batch_size] for i in range(0, n, batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2] += batches.pop()
    return batches


def predict_proba(model: nn.Module, x, mask=None, batch_size: int = 512, device: str = "cpu") -> np.ndarray:
    """返回 (N, 3) 概率，列顺序 [0 Neutral, 1 Positive, 2 Negative]。"""
    model.eval()
    x, mask = _to_tensor(x, torch.float32), _to_tensor(mask, torch.bool)
    out = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = x[start:start + batch_size].to(device)
            mb = None if mask is None else mask[start:start + batch_size].to(device)
            out.append(torch.softmax(model(xb, mb), dim=1).cpu())
    return torch.cat(out).numpy().astype(np.float64)


def fit_model(model: nn.Module, x_fit, y_fit, x_stop=None, y_stop=None, *, train_mask=None, valid_mask=None,
              config: TrainConfig | None = None, checkpoint=None) -> dict:
    """训练模型。给了 (x_stop, y_stop) 时按其 Macro-F1 早停并恢复最好的权重；否则固定训练 config.epochs 轮。

    返回 history（每轮的训练损失和验证 Macro-F1）、best_epoch、best_macro_f1 和 fit_seconds。
    """
    config = config or TrainConfig()
    device = torch.device(config.device)
    model.to(device)
    x, y = _to_tensor(x_fit, torch.float32), _to_tensor(y_fit, torch.long)
    mask = _to_tensor(train_mask, torch.bool)

    weight = None
    if config.class_weight:
        counts = torch.bincount(y, minlength=N_CLASSES).float()
        weight = (counts.sum() / (N_CLASSES * counts.clamp(min=1))).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    generator = torch.Generator().manual_seed(config.seed)

    validate = x_stop is not None
    history, best_f1, best_epoch, best_state, stale = [], -1.0, 0, None, 0
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for idx in _batches(len(y), config.batch_size, generator):
            xb, yb = x[idx].to(device), y[idx].to(device)
            mb = None if mask is None else mask[idx].to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb, mb), yb)
            loss.backward()
            if config.clip_norm:
                nn.utils.clip_grad_norm_(model.parameters(), config.clip_norm)
            optimizer.step()
            total += loss.item() * len(idx)
        row = {"epoch": epoch, "train_loss": total / len(y)}
        if validate:
            pred = predict_proba(model, x_stop, valid_mask, device=config.device).argmax(1)
            row["valid_macro_f1"] = f1_score(y_stop, pred, labels=list(LABELS), average="macro", zero_division=0)
            if row["valid_macro_f1"] > best_f1:
                best_f1, best_epoch, stale = row["valid_macro_f1"], epoch, 0
                best_state = copy.deepcopy(model.state_dict())
                if checkpoint is not None:
                    torch.save(best_state, checkpoint)
            else:
                stale += 1
        history.append(row)
        if validate and stale >= config.patience:
            break

    if validate:
        model.load_state_dict(best_state)
    else:
        best_epoch = len(history)
        if checkpoint is not None:
            torch.save(model.state_dict(), checkpoint)
    return {"history": history, "best_epoch": best_epoch, "best_macro_f1": best_f1 if validate else None,
            "fit_seconds": time.perf_counter() - started}


def save_predictions(run_id: str, oof_ids, oof_proba, test_ids, test_proba) -> None:
    """按团队约定保存概率（列为 [0, 1, 2]）。测试集预测还没做时传空数组，不要用 0 概率代替。"""
    save_preds(run_id, oof_ids=oof_ids, oof_proba=oof_proba, test_ids=test_ids, test_proba=test_proba,
               classes=list(LABELS))
