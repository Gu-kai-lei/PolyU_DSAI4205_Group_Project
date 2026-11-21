"""Grouped development-only evaluation for BiLSTM and class imbalance."""

from dataclasses import replace

import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .cache import load_embedding
from .config import N_FOLDS, SEED
from .fasttext_emb import tokenize, token_sequences, train_fasttext
from .project_review import development_data, score
from .training import TrainConfig, fit_model, predict_proba, seed_everything


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, states, mask):
        scores = self.v(torch.tanh(self.W(states))).squeeze(-1)
        empty = ~mask.any(dim=1, keepdim=True)
        scores = scores.masked_fill(~mask, float("-inf")).masked_fill(empty, 0.0)
        weights = torch.softmax(scores, dim=1).masked_fill(empty, 0.0)
        return torch.bmm(weights.unsqueeze(1), states).squeeze(1)


class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim=100, hidden=128, n_layers=2, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, num_layers=n_layers, bidirectional=True,
                            batch_first=True, dropout=dropout if n_layers > 1 else 0.0)
        self.attn = AdditiveAttention(hidden * 2)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 2, 3))

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        lengths = mask.sum(dim=1).clamp(min=1).cpu()
        sequence = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output, _ = self.lstm(sequence)
        states, _ = pad_packed_sequence(output, batch_first=True)
        return self.fc(self.attn(states, mask[:, :states.shape[1]]))


def bilstm_oof(dev=None):
    """Refit on each outer train set after selecting epochs with untouched inner validation."""
    if dev is None:
        dev = development_data()
    probability = np.full((len(dev), 3), np.nan)
    epochs = []
    config = TrainConfig(epochs=20, patience=5, batch_size=32, lr=1e-3)
    for fold in sorted(dev["fold"].unique()):
        valid = dev["fold"].eq(fold)
        train, validation = dev.loc[~valid], dev.loc[valid]
        key = f"fasttext_tok_fold{fold}"
        x = load_embedding(key, train.index)
        x_valid = load_embedding(key, validation.index)
        train_mask = load_embedding(f"{key}_mask", train.index)
        outer_mask = load_embedding(f"{key}_mask", validation.index)

        splitter = StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
        fit, stop = next(splitter.split(train, train["label"], train["group_id"]))
        fitting, stopping = train.iloc[fit], train.iloc[stop]
        train_tokens = fitting["text_norm"].map(tokenize).tolist()
        stop_tokens = stopping["text_norm"].map(tokenize).tolist()
        embedding = train_fasttext(train_tokens)
        x_fit, mask_fit = token_sequences(embedding, train_tokens, x.shape[1])
        x_stop, mask_stop = token_sequences(embedding, stop_tokens, x.shape[1])
        seed_everything(SEED)
        selection = fit_model(BiLSTMAttention(input_dim=x.shape[-1]), x_fit, fitting["label"].to_numpy(),
                              x_stop, stopping["label"].to_numpy(), train_mask=mask_fit,
                              valid_mask=mask_stop, config=config)
        count = selection["best_epoch"]
        epochs.append({"fold": int(fold), "best_epoch": count})

        seed_everything(SEED)
        model = BiLSTMAttention(input_dim=x.shape[-1])
        fit_model(model, x, train["label"].to_numpy(), train_mask=train_mask,
                  config=replace(config, epochs=count))
        probability[valid] = predict_proba(model, x_valid, outer_mask)
    return probability, pd.DataFrame(epochs)


def imbalance_oof(dev=None):
    """Apply weighting or resampling on training rows of each fold."""
    if dev is None:
        dev = development_data()
    probabilities = {name: np.full((len(dev), 3), np.nan)
                     for name in ("none", "class_weight", "oversample")}
    for fold in sorted(dev["fold"].unique()):
        valid = dev["fold"].eq(fold)
        key = f"fasttext_mean_fold{fold}"
        x, y = load_embedding(key, dev.index[~valid]), dev.loc[~valid, "label"]
        x_valid = load_embedding(key, dev.index[valid])
        for name, probability in probabilities.items():
            weight = "balanced" if name == "class_weight" else None
            x_fit, y_fit = x, y
            if name == "oversample":
                x_fit, y_fit = RandomOverSampler(random_state=SEED).fit_resample(x, y)
            model = LogisticRegression(max_iter=1000, class_weight=weight, random_state=SEED)
            probability[valid] = model.fit(x_fit, y_fit).predict_proba(x_valid)
    return pd.DataFrame([score(dev, values, name) for name, values in probabilities.items()])
