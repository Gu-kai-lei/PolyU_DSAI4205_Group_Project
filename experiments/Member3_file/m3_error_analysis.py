"""Compare official baseline and Member 3 development-set OOF predictions."""
import argparse
from pathlib import Path
import re
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline

CLASSES = [0, 1, 2]
NAMES = ['Neutral', 'Positive', 'Negative']
RUN = 'xgboost__sbert_multi__20261121-143943-a632'
BASELINE_RUN = 'baseline__tfidf__20261102-181839-3628'


def compare(reference, prediction):
    return confusion_matrix(reference, prediction, labels=CLASSES)


def macro_f1(matrix):
    return (2 * np.diagonal(matrix, axis1=-2, axis2=-1) /
            (matrix.sum(axis=-2) + matrix.sum(axis=-1))).mean(axis=-1)


def bootstrap(frame, base, candidate):
    """Paired group bootstrap of fixed OOF predictions, stratified by outer fold."""
    rng = np.random.default_rng(42)
    bootstrap_matrices = np.zeros((5000, 2, 3, 3), dtype=int)
    y = frame.label.to_numpy()
    for _, fold_rows in frame.groupby('fold'):
        group_codes, groups = pd.factorize(fold_rows.group_id)
        idx = fold_rows.index.to_numpy()
        per_group = np.zeros((len(groups), 2, 3, 3), dtype=int)
        np.add.at(per_group[:, 0], (group_codes, y[idx], base[idx]), 1)
        np.add.at(per_group[:, 1], (group_codes, y[idx], candidate[idx]), 1)
        for start in range(0, 5000, 100):
            draws = rng.integers(len(groups), size=(100, len(groups)))
            bootstrap_matrices[start:start + 100] += per_group[draws].sum(axis=1)
    scores = macro_f1(bootstrap_matrices)
    return np.quantile(scores[:, 1] - scores[:, 0], [.025, .975])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--pred', type=Path, help='Team NPZ containing the Member 3 OOF probabilities')
    parser.add_argument('--run-id', default=RUN, help='XGBoost run ID in the development score ledger')
    parser.add_argument('--output', type=Path, help='Directory for the new analysis')
    args = parser.parse_args()
    repo = args.repo.resolve()
    pred_path = args.pred or repo / 'data' / 'preds' / (args.run_id + '.npz')
    analysis_root = repo / 'local_artifacts' / 'm3_error_analysis'
    if args.output is not None:
        out = args.output.resolve()
        if out == Path(__file__).resolve().parent:
            parser.error('Use a separate output directory to preserve the published analysis')
    if not pred_path.is_file():
        parser.error(f'Missing predictions for run {args.run_id}: {pred_path}')
    clean = pd.read_csv(repo / 'data/clean_v2/clean_v2.csv')
    split = pd.read_csv(repo / 'splits/folds_v1.csv')
    frame = (split[split.split.eq('train')]
             .merge(clean, on=['id', 'group_id'], validate='one_to_one')
             .sort_values('id').reset_index(drop=True))
    assert len(frame) == 1222 and set(frame.fold) == set(range(5))
    assert frame.groupby('group_id').fold.nunique().max() == 1
    y = frame.label.to_numpy()
    ledger = pd.read_csv(repo/'results/m3_xgboost.csv')
    matching = ledger.loc[ledger['run_id'].eq(args.run_id) & ledger['split'].eq('cv')]
    if len(matching) != 1:
        raise ValueError(f'Expected one development score row for {args.run_id}')
    xgb_row = matching.iloc[0]
    baseline_row = pd.read_csv(repo/'results/baseline.csv').set_index('run_id').loc[BASELINE_RUN]
    base = np.full(len(frame), -1)
    for fold in range(5):
        valid = frame.fold.eq(fold).to_numpy()
        model = make_pipeline(TfidfVectorizer(ngram_range=(1, 1), min_df=1),
                              LogisticRegression(C=1.0, max_iter=2000, random_state=42))
        model.fit(frame.loc[~valid, 'text_norm'], y[~valid])
        base[valid] = model.predict(frame.loc[valid, 'text_norm'])
    with np.load(pred_path, allow_pickle=False) as data:
        assert data['run_id'].item() == args.run_id and data['classes'].tolist() == CLASSES
        ids = data['oof_ids']
        assert len(ids) == len(set(ids)) == len(frame) and set(ids) == set(frame.id)
        indices = pd.Index(ids).get_indexer(frame.id)
        probability = data['oof_proba'][indices]
        if probability.shape != (len(frame), len(CLASSES)) or not np.isfinite(probability).all():
            raise ValueError('Invalid OOF probability shape or values')
        if (probability < 0).any() or (probability > 1).any() or not np.allclose(probability.sum(1), 1):
            raise ValueError('Expected normalized class probabilities')
        candidate = probability.argmax(axis=1)
    matrices = [compare(y, pred) for pred in [base, candidate]]
    f1 = np.array([macro_f1(m) for m in matrices])
    np.testing.assert_allclose(f1, [baseline_row.macro_f1, xgb_row.macro_f1], rtol=0, atol=1e-10)
    np.testing.assert_allclose([np.trace(m)/len(y) for m in matrices],
                               [baseline_row.accuracy, xgb_row.accuracy], rtol=0, atol=1e-10)
    if args.output is None:
        analysis_root.mkdir(parents=True, exist_ok=True)
        out = Path(tempfile.mkdtemp(prefix='run-', dir=analysis_root))
    else:
        out.mkdir(parents=True, exist_ok=True)
    print(f'Writing analysis of {args.run_id} to {out}')
    class_rows = []
    for name, pred in [('TF-IDF + LR', base), ('XGBoost', candidate)]:
        precision, recall, by_class, support = precision_recall_fscore_support(y, pred, labels=CLASSES)
        class_rows.extend({'model': name, 'class': label, 'support': size, 'precision': p, 'recall': r, 'f1': f}
                          for label, size, p, r, f in zip(NAMES, support, precision, recall, by_class))
    pd.DataFrame(class_rows).to_csv(out/'m3_baseline_class_comparison.csv', index=False)

    right_base, right_xgb = base == y, candidate == y
    finance = frame.text_norm.str.contains(r'\b(?:red|green|bearish|bullish|fed|nifty|put|crash|dip)\b',
                                         flags=re.IGNORECASE, regex=True).to_numpy()
    subsets = [('All', np.ones(len(y), dtype=bool))]
    subsets += [(label, y == idx) for idx, label in enumerate(NAMES)]
    subsets += [('Length ≤80', frame.char_len.le(80).to_numpy()),
                ('Length 81–140', frame.char_len.between(81, 140).to_numpy()),
                ('Length 141–220', frame.char_len.between(141, 220).to_numpy()),
                ('Length >220', frame.char_len.gt(220).to_numpy()),
                ('Finance term present', finance), ('Finance term absent', ~finance)]
    paired = []
    for label, rows in subsets:
        counts = [int((rows & a & b).sum()) for a,b in
                  [(right_base, right_xgb), (right_base, ~right_xgb), (~right_base, right_xgb), (~right_base, ~right_xgb)]]
        paired.append(dict(subset=label, n=int(rows.sum()), both_correct=counts[0], baseline_only=counts[1],
                           xgboost_only=counts[2], both_wrong=counts[3]))
    pd.DataFrame(paired).to_csv(out/'m3_baseline_paired_errors.csv', index=False)
    interval = bootstrap(frame, base, candidate)

    style = {'axes.titleweight': 'bold', 'font.size': 11, 'axes.labelcolor': '#284254', 'text.color': '#142a3b'}
    with plt.rc_context(style):
        fig, axes = plt.subplots(1, 3, figsize=(12.7, 4.8), layout='constrained', gridspec_kw={'width_ratios':[1,1,1.15]})
        fig.suptitle('XGBoost catches more Neutral tweets and misses more directional sentiment', fontsize=16, weight='bold')
        for matrix, ax, title, score in zip(matrices, axes[:2], ['TF-IDF baseline', 'SBERT + XGBoost'], f1):
            display = ConfusionMatrixDisplay(matrix, display_labels=NAMES)
            display.plot(ax=ax, colorbar=False, cmap='Blues', im_kw={'vmin':0, 'vmax':340})
            ax.set_title(f'{title}\nMacro-F1 {score:.4f}', pad=10, fontsize=12)
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('Given label')
            ax.tick_params(labelsize=8)
        differences = np.array([row['f1'] for row in class_rows[3:]]) - np.array([row['f1'] for row in class_rows[:3]])
        ax = axes[2]
        ax.barh(NAMES, differences*100, color=['#008c7a' if d>0 else '#c75b42' for d in differences], height=.56)
        ax.set_xlim(-5.2,6.8)
        ax.axvline(0,color='#637989',lw=1)
        for i,d in enumerate(differences*100):
            ax.text(d + (.25 if d>=0 else -.25),i,f'{d:+.2f}',va='center',ha='left' if d>=0 else 'right',fontsize=10)
        ax.set_title('Change in class F1', fontsize=12)
        ax.set_xlabel('XGBoost minus baseline (percentage points)')
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0)
        ax.invert_yaxis()
        fig.savefig(out/'m3_baseline_error_comparison.png',dpi=180,facecolor='white')
        plt.close(fig)

    print('Baseline / XGBoost matrices:', [m.tolist() for m in matrices])
    print('Paired group bootstrap interval for XGBoost minus baseline:', interval.tolist())
    print(pd.DataFrame(paired).to_string(index=False))


if __name__ == '__main__':
    main()
