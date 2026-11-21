"""Train the fixed C3 model or reproduce the four-candidate selection workflow."""
from collections import defaultdict
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from safetensors.torch import save_file
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from experiments.Member5_file.mvp import TextCNN
from experiments.Member5_file.sbert_cache import prepare_embeddings
from src.cache import load_embedding
from src.config import DATA_ROOT, LABELS, N_FOLDS, VERSION_TAGS
from src.data import load_clean, load_splits
from src.training import TrainConfig, fit_model, predict_proba, seed_everything

SEEDS = (42, 17, 2026)
CANDIDATES = {
    "C0": (100, 0.001, False),
    "C1": (32, 0.001, False),
    "C3": (64, 0.0003, False),
    "C7": (64, 0.0003, True),
}


def macro_f1(labels, probabilities):
    return f1_score(labels, probabilities.argmax(axis=1), labels=list(LABELS),
                    average="macro", zero_division=0)


def fit_candidate(x, y, mask, candidate, seed, device, epochs=20, validation=None):
    filters, lr, class_weight = CANDIDATES[candidate]
    seed_everything(seed)
    model = TextCNN(x.shape[-1], filters=filters, dropout=0.5)
    config = TrainConfig(
        epochs=epochs, patience=4, batch_size=32, lr=lr, weight_decay=0.01,
        class_weight=class_weight, device=device, seed=seed,
    )
    x_valid, y_valid, valid_mask = validation if validation is not None else (None, None, None)
    history = fit_model(
        model, x, y, x_valid, y_valid,
        train_mask=mask, valid_mask=valid_mask, config=config,
    )
    return model, history


def _select(x, y, mask, groups, indices, device):
    """Compare each candidate within the provided training partition only."""
    splitter = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=42)
    scores = defaultdict(list)
    epochs = defaultdict(list)
    parameters = {}
    rows = []
    for fold, (a, b) in enumerate(splitter.split(indices, y[indices], groups[indices])):
        train, valid = indices[a], indices[b]
        if set(groups[train]) & set(groups[valid]):
            raise ValueError("Inner fold groups overlap")
        if set(y[train]) != set(LABELS) or set(y[valid]) != set(LABELS):
            raise ValueError("Inner folds must contain all three classes")
        for candidate in CANDIDATES:
            for seed in SEEDS:
                model, training = fit_candidate(
                    x[train], y[train], mask[train], candidate, seed, device,
                    validation=(x[valid], y[valid], mask[valid]),
                )
                probability = predict_proba(model, x[valid], mask[valid], batch_size=64, device=device)
                score = macro_f1(y[valid], probability)
                scores[candidate].append(score)
                epochs[candidate].append(training["best_epoch"])
                parameters[candidate] = sum(p.numel() for p in model.parameters())
                rows.append((candidate, fold, seed, score, training["best_epoch"], parameters[candidate]))
    best = min(CANDIDATES, key=lambda c: (-np.mean(scores[c]), parameters[c], c))
    return best, {c: int(np.median(epochs[c])) for c in CANDIDATES}, pd.DataFrame(
        rows, columns=["candidate", "inner_fold", "seed", "macro_f1", "best_epoch", "parameters"],
    )


def run_c3(mode="fixed", device="cpu"):
    """Write each experiment to its own local directory. Holdout labels are never scored."""
    if mode not in {"fixed", "nested"}:
        raise ValueError("mode must be 'fixed' or 'nested'")
    if prepare_embeddings(representations=("tokens_multi", "tokens_mask")) is None:
        raise FileNotFoundError("Install the token and mask cache or run 12_emb_tokens.ipynb first.")

    clean = load_clean()
    splits = load_splits(clean=clean)
    dev = splits.loc[splits["split"] == "train"].sort_values("id")
    test_ids = splits.loc[splits["split"] == "test", "id"].sort_values().to_numpy()
    dev_ids = dev["id"].to_numpy()
    y = clean.set_index("id").loc[dev_ids, "label"].to_numpy()
    groups = dev["group_id"].to_numpy()
    x = load_embedding("tokens_multi", ids=dev_ids)
    mask = load_embedding("tokens_mask", ids=dev_ids)

    output_root = DATA_ROOT / "local_artifacts" / "m5_c3"
    output_root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix=f"{mode}-", dir=output_root))
    final, final_epochs = "C3", 9
    payload = {}
    if mode == "nested":
        oof = {(name, seed): np.full((len(dev), 3), np.nan)
               for name in ("selected", "C0") for seed in SEEDS}
        for fold in range(N_FOLDS):
            fit = np.flatnonzero(dev["fold"].to_numpy() != fold)
            valid = np.flatnonzero(dev["fold"].to_numpy() == fold)
            winner, epochs, inner = _select(x, y, mask, groups, fit, device)
            inner.to_csv(output / f"outer{fold}_inner_scores.csv", index=False)
            for name, candidate in (("selected", winner), ("C0", "C0")):
                for seed in SEEDS:
                    model, _ = fit_candidate(
                        x[fit], y[fit], mask[fit], candidate, seed, device, epochs[candidate],
                    )
                    oof[name, seed][valid] = predict_proba(
                        model, x[valid], mask[valid], batch_size=64, device=device,
                    )
            print(f"Outer fold {fold}: {winner}, {epochs[winner]} epochs", flush=True)

        rows = []
        for (name, seed), probability in oof.items():
            if not np.isfinite(probability).all():
                raise ValueError("Incomplete outer predictions")
            rows.append((name, seed, accuracy_score(y, probability.argmax(1)), macro_f1(y, probability)))
        pd.DataFrame(rows, columns=["workflow", "seed", "accuracy", "macro_f1"]).to_csv(
            output / "oof_metrics.csv", index=False,
        )
        payload = {"oof_ids": dev_ids, "oof_proba": oof["selected", 42]}
        final, epochs, inner = _select(x, y, mask, groups, np.arange(len(dev)), device)
        final_epochs = epochs[final]
        inner.to_csv(output / "final_inner_scores.csv", index=False)

    model, _ = fit_candidate(x, y, mask, final, 42, device, final_epochs)
    probability = predict_proba(
        model, load_embedding("tokens_multi", ids=test_ids),
        load_embedding("tokens_mask", ids=test_ids), batch_size=64, device=device,
    )
    save_file({name: value.cpu().contiguous() for name, value in model.state_dict().items()},
              str(output / "weights.safetensors"))
    np.savez(output / "predictions.npz", **payload, test_ids=test_ids, test_proba=probability,
             classes=list(LABELS), candidate=final, epochs=final_epochs,
             run_id=output.name, **{**VERSION_TAGS, "seed": 42})
    return output
