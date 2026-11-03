"""BiLSTM + 加性注意力。

输入是已经算好的 token 向量（本版用按折 FastText；5 号的冻结 token 序列到了可以直接换，
网络吃的是 (batch, length, dim) 加 mask，不依赖词表）。

1 层和 2 层都试，hidden 每方向 128，双向。padding 用 pack_padded_sequence 跳过。
早停看的是训练折内部再划出的 15%（按 group_id 整组划出），外层验证折不参与早停，
所以 OOF 的 macro-F1 没有用验证标签来挑 epoch。测试集只对选中的层数打一次分，
epoch 数用五折最佳 epoch 的中位数，不再看测试标签。
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cache import save_preds
from src.config import FIGURES_DIR, LABELS, N_FOLDS, SEED
from src.evaluate import evaluate, new_run_id, plot_confusion, save_result
from src.fasttext_emb import tokenize

from Member6_file.m6_data import ensure_fasttext, labels_for, load_frames, token_batch

HIDDEN = 128
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 32
MAX_EPOCHS = 20
PATIENCE = 5
LAYER_CHOICES = (1, 2)
INNER_VALID = 0.15


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class AdditiveAttention(nn.Module):
    """Bahdanau 加性注意力。padding 位置在 softmax 之前被掩掉。"""

    def __init__(self, hidden: int):
        super().__init__()
        self.score = nn.Linear(hidden, hidden)
        self.vector = nn.Linear(hidden, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        weights = self.vector(torch.tanh(self.score(hidden_states))).squeeze(-1)
        weights = weights.masked_fill(~mask, -1e9)
        alpha = torch.softmax(weights, dim=1)
        context = torch.sum(alpha.unsqueeze(-1) * hidden_states, dim=1)
        return context, alpha


class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim: int, hidden: int = HIDDEN, layers: int = 1, dropout: float = DROPOUT, n_classes: int = 3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.attention = AdditiveAttention(hidden * 2)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden * 2, n_classes)

    def forward(self, sequences: torch.Tensor, mask: torch.Tensor):
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(sequences, lengths, batch_first=True, enforce_sorted=False)
        encoded, _ = self.lstm(packed)
        hidden_states, _ = nn.utils.rnn.pad_packed_sequence(encoded, batch_first=True, total_length=sequences.size(1))
        context, alpha = self.attention(hidden_states, mask)
        return self.output(self.dropout(context)), alpha


class FocalLoss(nn.Module):
    """选做。gamma=2 的多分类 focal loss，用来和普通交叉熵对比，不参与最终模型选择。"""

    def __init__(self, gamma: float = 2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_prob = torch.log_softmax(logits, dim=1)
        gathered = log_prob.gather(1, target.unsqueeze(1)).squeeze(1)
        probability = gathered.exp()
        return (-((1 - probability) ** self.gamma) * gathered).mean()


def _loss_function(kind: str) -> nn.Module:
    if kind == "cross_entropy":
        return nn.CrossEntropyLoss()
    if kind == "focal":
        return FocalLoss()
    raise ValueError(kind)


def _batches(count: int, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    for start in range(0, count, batch_size):
        yield order[start : start + batch_size]


def _macro_f1(y_true, proba) -> float:
    return float(f1_score(y_true, proba.argmax(axis=1), labels=list(LABELS), average="macro", zero_division=0))


def predict(model: BiLSTMAttention, sequences: np.ndarray, mask: np.ndarray, batch_size: int = 64):
    model.eval()
    probabilities = []
    attention = []
    with torch.inference_mode():
        for start in range(0, len(sequences), batch_size):
            stop = start + batch_size
            logits, alpha = model(
                torch.from_numpy(sequences[start:stop]),
                torch.from_numpy(mask[start:stop]),
            )
            probabilities.append(torch.softmax(logits, dim=1).numpy())
            attention.append(alpha.numpy())
    return np.concatenate(probabilities), np.concatenate(attention)


def train_once(X_train, mask_train, y_train, X_stop, mask_stop, y_stop, *, layers: int, loss_name: str, epochs: int | None, seed: int):
    """epochs 为 None 时按验证 macro-F1 早停，并返回最佳 epoch；否则正好训练 epochs 轮。"""
    set_seed(seed)
    model = BiLSTMAttention(X_train.shape[-1], layers=layers)
    criterion = _loss_function(loss_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_state = copy.deepcopy(model.state_dict())
    best_f1 = -1.0
    best_epoch = 0
    waiting = 0
    limit = MAX_EPOCHS if epochs is None else epochs
    started = time.perf_counter()
    for epoch in range(1, limit + 1):
        model.train()
        for index in _batches(len(X_train), BATCH_SIZE, seed + epoch):
            optimizer.zero_grad()
            logits, _ = model(torch.from_numpy(X_train[index]), torch.from_numpy(mask_train[index]))
            loss = criterion(logits, torch.from_numpy(y_train[index].astype(np.int64)))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epochs is not None:
            best_epoch = epoch
            continue
        proba, _ = predict(model, X_stop, mask_stop)
        score = _macro_f1(y_stop, proba)
        if score > best_f1 + 1e-4:
            best_f1 = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            waiting = 0
        else:
            waiting += 1
            if waiting >= PATIENCE:
                break
    if epochs is None:
        model.load_state_dict(best_state)
    return model, best_epoch, time.perf_counter() - started


def _split_inner(X, mask, y, groups, seed: int):
    inner_train, inner_stop = next(
        GroupShuffleSplit(n_splits=1, test_size=INNER_VALID, random_state=seed).split(X, y, groups)
    )
    return (
        X[inner_train], mask[inner_train], y[inner_train],
        X[inner_stop], mask[inner_stop], y[inner_stop],
    )


def _param(row: dict, key: str):
    params = row["params"]
    if isinstance(params, str):
        params = json.loads(params)
    return params[key]


def cross_validate(dev, y_dev, *, layers: int, loss_name: str) -> tuple[np.ndarray, list[int], float]:
    oof = np.zeros((len(dev), len(LABELS)), dtype=float)
    epochs = []
    seconds = 0.0
    folds = dev["fold"].to_numpy()
    groups = dev["group_id"].to_numpy()
    for fold in range(N_FOLDS):
        sequences, mask = token_batch(f"fold{fold}", dev["id"])
        outer = folds != fold
        held = ~outer
        X_tr, M_tr, y_tr, X_stop, M_stop, y_stop = _split_inner(
            sequences[outer], mask[outer], y_dev[outer], groups[outer], SEED + fold
        )
        model, best_epoch, elapsed = train_once(
            X_tr, M_tr, y_tr, X_stop, M_stop, y_stop,
            layers=layers, loss_name=loss_name, epochs=None, seed=SEED + fold,
        )
        proba, _ = predict(model, sequences[held], mask[held])
        oof[held] = proba
        epochs.append(best_epoch)
        seconds += elapsed
        print(f"layers={layers} loss={loss_name} fold={fold} best_epoch={best_epoch} val_f1={_macro_f1(y_dev[held], proba):.3f}", flush=True)
    return oof, epochs, seconds


def fit_full(dev, test, y_dev, *, layers: int, loss_name: str, epochs: int, seed: int):
    ids = np.concatenate([dev["id"].to_numpy(), test["id"].to_numpy()])
    sequences, mask = token_batch("full", ids)
    X_dev, M_dev = sequences[: len(dev)], mask[: len(dev)]
    X_test, M_test = sequences[len(dev) :], mask[len(dev) :]
    model, _, elapsed = train_once(
        X_dev, M_dev, y_dev, X_dev[:1], M_dev[:1], y_dev[:1],
        layers=layers, loss_name=loss_name, epochs=epochs, seed=seed,
    )
    test_proba, test_alpha = predict(model, X_test, M_test)
    return test_proba, test_alpha, M_test, elapsed


def _median_epochs(fold_epochs: list[int]) -> int:
    return max(1, int(round(float(np.median(fold_epochs)))))


def _params(layers: int, loss_name: str, fold_epochs: list[int], final_epochs: int, max_len: int) -> dict:
    return {
        "layers": layers,
        "hidden": HIDDEN,
        "bidirectional": True,
        "dropout": DROPOUT,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "patience": PATIENCE,
        "loss": loss_name,
        "max_len": max_len,
        "fold_best_epochs": fold_epochs,
        "final_epochs": final_epochs,
    }


def _save_cv_row(y_dev, oof, *, layers: int, loss_name: str, fold_epochs: list[int], seconds: float, max_len: int, notes: str):
    run_id = new_run_id("bilstm", "fasttext_tok")
    row = evaluate(
        y_dev,
        oof.argmax(axis=1),
        run_id=run_id,
        model="bilstm",
        embedding="fasttext_tok",
        split="cv",
        params=_params(layers, loss_name, fold_epochs, _median_epochs(fold_epochs), max_len),
        train_seconds=seconds,
        notes=notes,
    )
    _, recall, _, _ = precision_recall_fscore_support(
        y_dev, oof.argmax(axis=1), labels=list(LABELS), average=None, zero_division=0
    )
    for label, score in zip(LABELS, recall):
        row[f"recall_{LABELS[label].lower()}"] = float(score)
    return row


def _plot_attention(tokens: list[str], weights: np.ndarray, title: str, path: Path) -> None:
    order = np.arange(len(tokens))[::-1]
    fig, ax = plt.subplots(figsize=(8, max(2.4, 0.32 * len(tokens))))
    ax.barh(order, weights, color="#4C72B0")
    ax.set_yticks(order, tokens)
    ax.set_xlabel("Attention weight")
    ax.set_xlim(0, max(0.05, float(weights.max()) * 1.15))
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _attention_figures(test, y_test, proba, alpha, mask, text_of: dict) -> None:
    pred = proba.argmax(axis=1)
    ids = test["id"].to_numpy()
    chosen = []
    for label in LABELS:
        correct = np.where((y_test == label) & (pred == label))[0]
        pool = correct if len(correct) else np.where(y_test == label)[0]
        chosen.append((int(pool[np.argmax(proba[pool, label])]), LABELS[label].lower()))
    wrong = np.where(pred != y_test)[0]
    if len(wrong):
        confidence = proba[wrong, pred[wrong]]
        chosen.append((int(wrong[np.argmax(confidence)]), "error"))
    for index, name in chosen:
        valid = int(mask[index].sum())
        tokens = tokenize(text_of[int(ids[index])])[:valid]
        title = f"id {int(ids[index])}  true {LABELS[int(y_test[index])]}  predicted {LABELS[int(pred[index])]}"
        _plot_attention(tokens, alpha[index, :valid], title, FIGURES_DIR / f"m6_attention_{name}.png")
        print(f"wrote figures/m6_attention_{name}.png", flush=True)


def self_check() -> None:
    """不读数据，只确认 padding 不参与注意力。"""
    model = BiLSTMAttention(input_dim=8, hidden=16, layers=2, dropout=0.1)
    sequences = torch.randn(4, 7, 8)
    mask = torch.tensor(
        [
            [1, 1, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.bool,
    )
    logits, alpha = model(sequences, mask)
    if logits.shape != (4, 3):
        raise AssertionError(logits.shape)
    if not torch.allclose(alpha.sum(dim=1), torch.ones(4), atol=1e-5):
        raise AssertionError(alpha.sum(dim=1))
    if not torch.all(alpha[~mask] < 1e-5):
        raise AssertionError(alpha[~mask].max())
    print("attention self-check passed")


def run_bilstm() -> pd.DataFrame:
    self_check()
    clean, splits, dev, test, label_of, text_of = load_frames()
    max_len = ensure_fasttext(clean, splits)
    y_dev = labels_for(dev["id"], label_of)
    y_test = labels_for(test["id"], label_of)
    rows = []
    oof_by_layers = {}
    plain_note = "FastText token sequences. Early stopping uses an inner group holdout, not the reported fold."
    for layers in LAYER_CHOICES:
        oof, fold_epochs, seconds = cross_validate(dev, y_dev, layers=layers, loss_name="cross_entropy")
        oof_by_layers[layers] = (oof, fold_epochs, seconds)
        rows.append(_save_cv_row(
            y_dev, oof, layers=layers, loss_name="cross_entropy", fold_epochs=fold_epochs,
            seconds=seconds, max_len=max_len, notes=plain_note,
        ))

    chosen = max(LAYER_CHOICES, key=lambda layers: _macro_f1(y_dev, oof_by_layers[layers][0]))
    print(f"selected layers={chosen}", flush=True)
    oof, fold_epochs, seconds = oof_by_layers[chosen]
    final_epochs = _median_epochs(fold_epochs)
    test_proba, test_alpha, test_mask, test_seconds = fit_full(
        dev, test, y_dev, layers=chosen, loss_name="cross_entropy", epochs=final_epochs, seed=SEED,
    )
    chosen_row = next(
        row for row in rows
        if _param(row, "layers") == chosen and _param(row, "loss") == "cross_entropy"
    )
    save_preds(
        chosen_row["run_id"],
        oof_ids=dev["id"].to_numpy(),
        oof_proba=oof,
        test_ids=test["id"].to_numpy(),
        test_proba=test_proba,
        classes=list(LABELS),
    )
    save_result(chosen_row, "m6_bilstm")
    test_row = evaluate(
        y_test,
        test_proba.argmax(axis=1),
        run_id=chosen_row["run_id"],
        model="bilstm",
        embedding="fasttext_tok",
        split="test",
        params=_params(chosen, "cross_entropy", fold_epochs, final_epochs, max_len),
        train_seconds=test_seconds,
        notes="Single test evaluation. Epoch count is the median best epoch from CV, not tuned on the test labels.",
    )
    _, test_recall, _, _ = precision_recall_fscore_support(
        y_test, test_proba.argmax(axis=1), labels=list(LABELS), average=None, zero_division=0
    )
    for label, score in zip(LABELS, test_recall):
        test_row[f"recall_{LABELS[label].lower()}"] = float(score)
    save_result(test_row, "m6_bilstm")
    rows.append(test_row)

    # 没被选中的层数也留下 CV 记录，但不看测试集。
    for row in rows:
        if row["split"] == "cv" and row["run_id"] != chosen_row["run_id"]:
            save_result(row, "m6_bilstm")

    # 选做：focal loss 只在选中的层数上做 CV，不参与选层，也不打测试分。
    focal_oof, focal_epochs, focal_seconds = cross_validate(dev, y_dev, layers=chosen, loss_name="focal")
    focal_row = _save_cv_row(
        y_dev, focal_oof, layers=chosen, loss_name="focal", fold_epochs=focal_epochs,
        seconds=focal_seconds, max_len=max_len,
        notes="Optional focal loss (gamma=2) on the selected depth. Not used to choose the submitted model.",
    )
    save_result(focal_row, "m6_bilstm")
    rows.append(focal_row)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plot_confusion(
        y_test,
        test_proba.argmax(axis=1),
        title=f"BiLSTM test ({chosen} layer{'s' if chosen > 1 else ''})",
        save_as=FIGURES_DIR / "m6_bilstm_confusion.png",
    )
    plt.close("all")
    _attention_figures(test, y_test, test_proba, test_alpha, test_mask, text_of)
    table = pd.DataFrame([{key: row[key] for key in ("run_id", "split", "macro_f1", "accuracy", "params")} for row in rows])
    print(table.to_string(index=False), flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    run_bilstm()
