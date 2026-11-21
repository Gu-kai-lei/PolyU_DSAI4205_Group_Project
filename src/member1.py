"""An outer grouped OOF estimate of the frozen SBERT SVM selection procedure."""

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .cache import load_embedding
from .config import LABELS, SEED
from .project_review import development_data

CANDIDATES = (
    ("linear", 0.1, "scale", None),
    ("linear", 1, "scale", "balanced"),
    ("rbf", 1, "scale", None),
    ("rbf", 3, "scale", "balanced"),
    ("rbf", 10, "scale", "balanced"),
)


def nested_svm_oof(vectors, dev=None):
    """Select the classifier with the remaining official folds, predict the excluded fold."""
    if dev is None:
        dev = development_data()
    labels, folds = dev["label"].to_numpy(), dev["fold"].to_numpy()
    if len(vectors) != len(dev) or not np.isfinite(vectors).all():
        raise ValueError("SBERT embeddings must be finite and aligned to development IDs")

    def fit_predict(candidate, train, predict):
        kernel, c, gamma, weight = candidate
        model = make_pipeline(StandardScaler(), SVC(
            kernel=kernel, C=c, gamma=gamma, class_weight=weight,
            random_state=SEED, probability=True,
        )).fit(vectors[train], labels[train])
        if list(model.classes_) != list(LABELS):
            raise ValueError("Unexpected SVM classes")
        return model.predict_proba(vectors[predict])

    probabilities = np.full((len(dev), len(LABELS)), np.nan)
    selected = []
    for fold in sorted(set(folds)):
        outer_train = folds != fold

        def inner_score(candidate):
            prediction = np.full(len(dev), -1)
            for inner in sorted(set(folds[outer_train])):
                stop = folds == inner
                prediction[stop] = fit_predict(candidate, outer_train & ~stop, stop).argmax(1)
            return f1_score(labels[outer_train], prediction[outer_train], labels=list(LABELS),
                            average="macro", zero_division=0)

        best = max(CANDIDATES, key=inner_score)
        probabilities[~outer_train] = fit_predict(best, outer_train, ~outer_train)
        selected.append((fold, *best))
    return probabilities, pd.DataFrame(selected, columns=["fold", "kernel", "C", "gamma", "class_weight"])


def sbert_svm_oof():
    from experiments.Member5_file.sbert_cache import prepare_embeddings

    dev = development_data()
    if prepare_embeddings(representations=("sbert_multi",)) is None:
        raise FileNotFoundError("Verified SBERT sentence cache is missing. Run 10_emb_sbert.ipynb first.")
    return nested_svm_oof(load_embedding("sbert_multi", dev.index), dev)
