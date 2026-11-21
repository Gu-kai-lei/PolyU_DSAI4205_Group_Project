"""Reproduce the fixed and tuned baselines and validate the archived OOF handoffs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from threadpoolctl import threadpool_limits

from .config import CLEAN_DIR, CLEAN_VERSION, LABELS, PROJECT_ROOT, SEED, SPLITS_DIR, SPLIT_VERSION, VERSION_TAGS
from .data import load_clean, load_splits
from .evaluate import evaluate
from .splitting import validate_splits

CLASSES = list(LABELS)


def development_data():
    clean = load_clean()
    splits = load_splits(clean=clean)
    validate_splits(splits, clean)
    return (splits.loc[splits["split"] == "train"]
            .merge(clean.drop(columns="group_id"), on="id", validate="one_to_one")
            .sort_values("id").set_index("id"))


def baseline_model(ngram=(1, 1), c=1, weight=None, analyzer="word"):
    character = analyzer == "char"
    return make_pipeline(
        TfidfVectorizer(ngram_range=ngram, analyzer=analyzer, min_df=2 if character else 1,
                        sublinear_tf=character),
        LogisticRegression(C=c, class_weight=weight, max_iter=2000, random_state=SEED),
    )


def baseline_oof(dev, text="text_norm", estimator=None):
    if estimator is None:
        estimator = baseline_model()
    probability = np.full((len(dev), len(CLASSES)), np.nan)
    for fold in sorted(dev["fold"].unique()):
        valid = dev["fold"].eq(fold)
        fitted = clone(estimator).fit(dev.loc[~valid, text], dev.loc[~valid, "label"])
        if list(fitted.classes_) != CLASSES:
            raise ValueError(f"Unexpected baseline classes in fold {fold}")
        probability[valid] = fitted.predict_proba(dev.loc[valid, text])
    return probability


@threadpool_limits.wrap(limits=1, user_api="blas")
def tuned_baseline_oof(dev):
    """Select features and regularization only inside each outer training partition."""
    representations = (
        ("text_norm", (1, 1), "word"),
        ("text_norm", (1, 2), "word"),
        ("text_clean", (1, 2), "word"),
        ("text_clean", (3, 5), "char"),
    )
    candidates = [
        (text, ngram, analyzer, c, weight)
        for text, ngram, analyzer in representations
        for c in (0.25, 1, 4, 16)
        for weight in (None, "balanced")
    ]
    probability = np.full((len(dev), len(CLASSES)), np.nan)
    selected = []
    for fold in sorted(dev["fold"].unique()):
        valid = dev["fold"].eq(fold)
        training = dev.loc[~valid]

        matrices = {}
        for text, ngram, analyzer in representations:
            vectorizer = baseline_model(ngram, analyzer=analyzer).steps[0][1]
            folds = []
            for inner_fold in sorted(training["fold"].unique()):
                inner_valid = training["fold"].eq(inner_fold)
                train_features = vectorizer.fit_transform(training.loc[~inner_valid, text])
                valid_features = vectorizer.transform(training.loc[inner_valid, text])
                folds.append((inner_valid, train_features, valid_features))
            matrices[text, ngram, analyzer] = folds

        def inner_macro_f1(candidate):
            text, ngram, analyzer, c, weight = candidate
            classifier = baseline_model(ngram, c, weight, analyzer).steps[-1][1]
            prediction = np.full((len(training), len(CLASSES)), np.nan)
            for inner_valid, train_features, valid_features in matrices[text, ngram, analyzer]:
                classifier.fit(train_features, training.loc[~inner_valid, "label"])
                if list(classifier.classes_) != CLASSES:
                    raise ValueError("Unexpected inner baseline classes")
                prediction[inner_valid] = classifier.predict_proba(valid_features)
            return score(training, prediction, "inner_baseline")["macro_f1"]

        choice = max(candidates, key=inner_macro_f1)
        text, ngram, analyzer, c, weight = choice
        model = baseline_model(ngram, c, weight, analyzer).fit(training[text], training["label"])
        probability[valid] = model.predict_proba(dev.loc[valid, text])
        selected.append({"fold": int(fold), "text": text, "ngram": ngram, "analyzer": analyzer,
                         "C": c, "class_weight": weight})
    return probability, pd.DataFrame(selected)


def archived_oof(dev, path, columns, expected_macro_f1):
    """Load ID-aligned probabilities and reject evidence that disagrees with its score."""
    table = pd.read_csv(path).set_index("id", verify_integrity=True).sort_index()
    pd.testing.assert_frame_equal(table[["group_id", "fold", "label"]],
                                  dev[["group_id", "fold", "label"]])
    values = table[list(columns)].to_numpy()
    check_probability(dev, values)
    np.testing.assert_array_equal(table["prediction"], values.argmax(axis=1))
    actual = score(dev, values, "archived_oof")["macro_f1"]
    if not np.isclose(actual, expected_macro_f1, rtol=0, atol=1e-10):
        raise ValueError(f"{path}: OOF Macro-F1 {actual} differs from run score {expected_macro_f1}")
    return values


def check_probability(dev, values):
    if values.shape != (len(dev), len(CLASSES)) or not np.isfinite(values).all():
        raise ValueError("Expected one finite probability vector per development row")
    if (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Each probability vector must be in [0, 1] and sum to one")


def score(dev, values, run_id):
    check_probability(dev, values)
    return evaluate(dev["label"], values.argmax(axis=1), run_id=run_id,
                    model=run_id, embedding="", split="cv")


def read_run(path, run_id):
    rows = pd.read_csv(path)
    match = rows.loc[(rows["run_id"] == run_id) & (rows["split"] == "cv")]
    if len(match) != 1:
        raise ValueError(f"Expected one CV row for {run_id} in {path}")
    row = match.iloc[0]
    if any(row[key] != value for key, value in VERSION_TAGS.items()):
        raise ValueError(f"Version mismatch for {run_id} in {path}")
    return row


# Run IDs are explicit so new experimental rows cannot silently change the comparison.
RUNS = (
    ("SVM", "experiments/Member1_updated1/results/m1_svm.csv", "svm__sbert_multi__20261121-145728-06de",
     None, (), "同一外折选优，缺逐条 OOF"),
    ("kNN", "results/m2_knn.csv", "knn__sbert_multi__20261118-133810-8033",
     "experiments/Member2_file/predictions/knn_oof.csv", ("prob_0", "prob_1", "prob_2"), "同一外折选优"),
    ("XGBoost", "results/m3_xgboost.csv", "xgboost__sbert_multi__20261121-143943-a632",
     None, (), "分组嵌套，缺逐条 OOF"),
    ("MLP", "results/m4_mlp.csv", "mlp__fasttext_mean__20261104-103233-2f07",
     "experiments/Member4_file/evidence/selected_oof_predictions.csv",
     ("prob_neutral", "prob_positive", "prob_negative"), "同一外折选配置"),
    ("TextCNN", "results/m5_textcnn_adopted.csv", "textcnn_refinement_colab_latest_20261112",
     "experiments/Member5_file/evidence/refinement_v1_20261112/selected_seed42_oof.csv",
     ("prob_0", "prob_1", "prob_2"), "分组嵌套选择流程，seed 42"),
    ("BiLSTM（归档）", "results/m6_bilstm.csv", "bilstm__fasttext_tok__20261104-232636-341a",
     None, (), "外折早停，仅作历史记录"),
    ("BiLSTM（分组内层早停）", "results/m6_bilstm_grouped.csv", "bilstm__fasttext_tok__m2_grouped_inner_v1",
     "experiments/Member6_file/evidence/bilstm__fasttext_tok__m2_grouped_inner_v1.csv",
     ("prob_0", "prob_1", "prob_2"), "内层分组选轮数并重训 FastText"),
)


def team_evidence(dev, root: Path = PROJECT_ROOT):
    reference = json.loads((root / "experiments/Member5_file" / "embedding_manifest.json").read_text())
    inputs = ((CLEAN_DIR / f"clean_v{CLEAN_VERSION}.csv", "data_sha256"),
              (SPLITS_DIR / f"folds_v{SPLIT_VERSION}.csv", "split_sha256"))
    for path, key in inputs:
        if hashlib.sha256(path.read_bytes()).hexdigest() != reference[key]:
            raise ValueError(f"Archived model evidence was produced from a different snapshot than {path}")
    rows, probabilities = [], {}
    for name, metrics, run_id, handoff, cols, protocol in RUNS:
        row = read_run(root / metrics, run_id)
        if handoff:
            probabilities[name] = archived_oof(dev, root / handoff, cols, row["macro_f1"])
        rows.append({"model": name, "run_id": run_id, "macro_f1": row["macro_f1"],
                     "protocol": protocol, "oof_verified": handoff is not None})
    return pd.DataFrame(rows), probabilities


def comparison(tuned=False):
    """Compare development evidence without scoring the historical holdout."""
    dev = development_data()
    archived, probabilities = team_evidence(dev)
    fixed = baseline_oof(dev)
    probabilities["Baseline"] = fixed
    if tuned:
        probabilities["Tuned baseline"], parameters = tuned_baseline_oof(dev)
    else:
        parameters = pd.DataFrame()
    name = "Tuned baseline" if tuned else "Baseline"
    probabilities["Equal-weight vote"] = (probabilities[name] + probabilities["TextCNN"]) / 2

    rows = []
    for label in (name, "Equal-weight vote"):
        row = score(dev, probabilities[label], label)
        rows.append({"model": label, "macro_f1": row["macro_f1"], "oof_verified": True,
                     "run_id": {"Baseline": "baseline_fixed", "Tuned baseline": "baseline_nested",
                                "Equal-weight vote": "vote_equal_weight"}[label],
                     "protocol": {"Baseline": "固定公式复算", "Tuned baseline": "分组嵌套",
                                  "Equal-weight vote": "探索性 OOF 概率平均"}[label]})
    if tuned:
        rows.insert(0, {"model": "Baseline", "macro_f1": score(dev, fixed, "baseline")["macro_f1"],
                        "oof_verified": True, "run_id": "baseline_fixed", "protocol": "固定公式复算"})
    return pd.concat([archived, pd.DataFrame(rows)], ignore_index=True), parameters, probabilities


def verify_cleaning():
    """Rebuild the tracked clean_v2 artifacts in memory and compare their bytes."""
    import importlib.util

    folder = PROJECT_ROOT / "data" / "clean_v2"
    source = importlib.util.spec_from_file_location("datavinci_clean_v2", folder / "clean_v2.py")
    module = importlib.util.module_from_spec(source)
    source.loader.exec_module(module)
    raw = pd.read_csv(module.RAW_CSV)
    records, log, evidence = module.build(raw)
    full = records.assign(keep=records["keep"].astype(int))[["keep", "drop_reason", *module.COLUMNS]]
    kept = full.loc[full["keep"].eq(1), module.COLUMNS]
    rebuilt = {
        "clean_v2.csv": kept.to_csv(index=False, lineterminator="\n"),
        "clean_v2_all.csv": full.to_csv(index=False, lineterminator="\n"),
        "clean_v2_log.md": module.render_log(records, log, evidence),
    }
    for name, contents in rebuilt.items():
        if (folder / name).read_bytes() != contents.encode("utf-8"):
            raise ValueError(f"Cleaning output differs from the tracked {name}")
    return pd.DataFrame(log)[["step", "operation", "rows_affected", "rows_remaining"]]


def diagnostic_agreement(dev):
    """Compare the supplied labels against VADER independently of model fitting."""
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    rows = []
    for text in ("text_raw", "text_clean", "text_norm"):
        polarity = dev[text].map(lambda value: analyzer.polarity_scores(value)["compound"])
        pred = np.select([polarity.ge(0.05), polarity.le(-0.05)], [1, 2], default=0)
        rows.append({"text": text, "agreement_with_given_labels": (pred == dev["label"]).mean()})
    return pd.DataFrame(rows)


def paired_group_interval(dev, candidate, reference, draws=4000):
    """Bootstrap paired OOF differences conditional on the trained predictions."""
    for probability in (candidate, reference):
        check_probability(dev, probability)
    groups = pd.factorize(dev["group_id"])[0]
    count = int(groups.max()) + 1
    weights = np.random.default_rng(SEED).multinomial(count, np.full(count, 1 / count), size=draws)
    labels = dev["label"].to_numpy()

    def sampled_f1(probability):
        confusion = np.zeros((count, 9))
        np.add.at(confusion, (groups, 3 * labels + probability.argmax(1)), 1)
        sampled = (weights @ confusion).reshape(-1, 3, 3)
        correct = np.diagonal(sampled, axis1=1, axis2=2)
        totals = sampled.sum(1) + sampled.sum(2)
        return np.divide(2 * correct, totals, out=np.zeros_like(correct), where=totals > 0).mean(1)

    return np.quantile(sampled_f1(candidate) - sampled_f1(reference), [0.025, 0.975])
