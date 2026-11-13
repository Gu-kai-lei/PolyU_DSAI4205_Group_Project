"""Bounded, resumable nested-group TextCNN study. Never scores test labels.

Run from repository root. `audit`, `benchmark`, `run`, `report` are explicit stages.
Large fit caches/weights stay below DATA_ROOT/local_artifacts, not in Git.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from src.config import DATA_ROOT, CLEAN_DIR, EMB_DIR, VERSION_TAGS
from src.data import load_clean, load_splits
from src.training import TrainConfig, fit_model, predict_proba, seed_everything

SEEDS = (42, 17, 2026)
BASE = dict(filters=100, lr=.001, dropout=.5, weight_decay=.01, class_weight=False,
            max_epochs=20, patience=4)
CANDIDATES = {
    'C0': {**BASE},
    'C1': {**BASE, 'filters': 32},
    'C2': {**BASE, 'filters': 64},
    'C3': {**BASE, 'filters': 64, 'lr': .0003},
    'C4': {**BASE, 'filters': 64, 'lr': .0003, 'dropout': .3},
    'C5': {**BASE, 'filters': 64, 'lr': .0003, 'dropout': .6},
    'C6': {**BASE, 'filters': 64, 'lr': .0003, 'weight_decay': .001},
    'C7': {**BASE, 'filters': 64, 'lr': .0003, 'class_weight': True},
}
EVIDENCE = ROOT / 'Member5_file' / 'evidence' / 'refinement'


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False,
                                    default=lambda x: x.tolist() if isinstance(x, np.ndarray) else str(x)) + '\n')
    temporary.replace(path)


def protocol_signature(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(x):
    if x is None:
        return None
    x = np.ascontiguousarray(x)
    h = hashlib.sha256(str((x.shape, x.dtype)).encode())
    h.update(memoryview(x).cast('B'))
    return h.hexdigest()


def parameter_count(cfg, dim=768):
    return dim * cfg['filters'] * 14 + cfg['filters'] * 4 + cfg['filters'] * 4 * 3 + 3


def inner_partitions(frame):
    splitter = StratifiedGroupKFold(3, shuffle=True, random_state=42)
    pairs = list(splitter.split(frame, frame.label, frame.group_id))
    for fit, valid in pairs:
        if not set(frame.iloc[fit].group_id).isdisjoint(frame.iloc[valid].group_id):
            raise ValueError('Inner group overlap')
        if set(frame.iloc[fit].label) != {0, 1, 2} or set(frame.iloc[valid].label) != {0, 1, 2}:
            raise ValueError('Every inner partition needs three labels')
    return pairs


def choose_candidate(rows, names):
    table = pd.DataFrame(rows)
    expected = {(fold, seed) for fold in range(3) for seed in SEEDS}
    means = []
    for name in names:
        group = table.loc[table.candidate == name]
        if len(group) != 9 or set(zip(group.inner_fold, group.seed)) != expected:
            raise ValueError('Candidate selection requires complete nine-fit results')
        if not np.isfinite(group.score).all():
            raise ValueError('Nonfinite selection score')
        means.append(dict(candidate=name, score=float(group.score.mean()),
                          epochs=max(1, int(np.median(group.best_epoch))),
                          parameters=int(group.parameters.iloc[0])))
    return sorted(means, key=lambda r: (-r['score'], r['parameters'], r['candidate']))[0]


def load_token_cache(directory, requested_ids):
    directory = Path(directory)
    x = np.load(directory / 'tokens_multi.npy', mmap_mode='r')
    mask = np.load(directory / 'tokens_mask.npy', mmap_mode='r')
    ids = np.load(directory / 'tokens_multi_ids.npy')
    mask_ids = np.load(directory / 'tokens_mask_ids.npy')
    if len(np.unique(ids)) != len(ids) or not np.array_equal(ids, mask_ids):
        raise ValueError('Duplicate or misaligned token cache IDs')
    if x.ndim != 3 or mask.shape != x.shape[:2] or mask.dtype != bool or len(ids) != len(x):
        raise ValueError('Malformed sequence or mask dimensions')
    if not np.isfinite(x).all() or np.any(x[~mask] != 0) or np.any(np.diff(mask.astype(int), axis=1) > 0):
        raise ValueError('Nonfinite token, nonzero padding, or noncompact mask')
    positions = {int(i): j for j, i in enumerate(ids)}
    if len(set(requested_ids)) != len(requested_ids) or not set(requested_ids) <= set(positions):
        raise ValueError('Requested IDs are duplicated or absent')
    order = [positions[int(i)] for i in requested_ids]
    return np.asarray(x[order]), np.asarray(mask[order])


def validate_proba(p, n):
    if p.shape != (n, 3) or not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Invalid probabilities')
    if not np.allclose(p.sum(1), 1, atol=1e-5):
        raise ValueError('Probabilities must sum to one')


def fit_cached(directory, key, x, y, mask, xv, yv, mv, cfg, seed, device, provenance,
               *, save_weights=False):
    """Resume only a fit with identical data, protocol, source and verified predictions."""
    torch.set_num_threads(2)
    folder = Path(directory) / key
    signature = protocol_signature(dict(cfg=cfg, seed=seed, device=device,
        provenance=provenance, source=file_hash(__file__),
        arrays=[array_hash(a) for a in (x, y, mask, xv, yv, mv)], save_weights=save_weights))
    meta = folder / 'fit.json'
    if meta.exists():
        old = json.loads(meta.read_text())
        if old['signature'] != signature:
            raise ValueError('Cached fit signature differs; use a new run directory')
        if old.get('status') == 'complete':
            if file_hash(folder / 'predictions.npy') != old['prediction_sha256']:
                raise ValueError('Cached predictions checksum mismatch')
            if save_weights and file_hash(folder / 'weights.safetensors') != old['weight_sha256']:
                raise ValueError('Cached weights checksum mismatch')
            p = np.load(folder / 'predictions.npy')
            validate_proba(p, len(xv))
            return {**old, 'probabilities': p, 'resumed': True}
    folder.mkdir(parents=True, exist_ok=True)
    write_json(meta, dict(signature=signature, status='running', provenance=provenance,
                         cfg=cfg, seed=seed))
    start = time.perf_counter()
    try:
        from mvp import TextCNN
        seed_everything(seed)
        model = TextCNN(x.shape[-1], filters=cfg['filters'], dropout=cfg['dropout'])
        config = TrainConfig(epochs=cfg['max_epochs'], patience=cfg['patience'],
            batch_size=32, lr=cfg['lr'], weight_decay=cfg['weight_decay'],
            class_weight=cfg['class_weight'], device=device, seed=seed)
        history = fit_model(model, x, y, xv if yv is not None else None, yv,
                            train_mask=mask, valid_mask=mv, config=config)
        if any(not np.isfinite(r['train_loss']) for r in history['history']):
            raise ValueError('Nonfinite training loss')
        p = predict_proba(model, xv, mv, batch_size=64, device=device)
        validate_proba(p, len(xv))
        np.save(folder / 'predictions.npy', p)
        weights = {}
        if save_weights:
            from safetensors.torch import save_file
            save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()},
                      str(folder / 'weights.safetensors'))
            weights['weight_sha256'] = file_hash(folder / 'weights.safetensors')
        result = dict(signature=signature, status='complete', cfg=cfg, seed=seed,
            provenance=provenance, history=history, seconds=time.perf_counter()-start,
            parameters=sum(v.numel() for v in model.parameters() if v.requires_grad),
            prediction_sha256=file_hash(folder / 'predictions.npy'), **weights)
        write_json(meta, result)
        print(f'{key}: {result["seconds"]:.2f}s epoch={history["best_epoch"]}', flush=True)
        return {**result, 'probabilities': p, 'resumed': False}
    except Exception as exc:
        write_json(meta, dict(signature=signature, status='failed', error=repr(exc),
                             seconds=time.perf_counter()-start, cfg=cfg, seed=seed,
                             provenance=provenance))
        raise


def metrics(y, p):
    validate_proba(p, len(y))
    pred = p.argmax(1)
    report = classification_report(y, pred, labels=[0,1,2], output_dict=True, zero_division=0)
    return dict(accuracy=float(accuracy_score(y, pred)),
                macro_precision=float(report['macro avg']['precision']),
                macro_recall=float(report['macro avg']['recall']),
                macro_f1=float(report['macro avg']['f1-score']),
                **{f'f1_{c}': float(report[str(c)]['f1-score']) for c in range(3)},
                classification_report=report, confusion_matrix=confusion_matrix(y, pred, labels=[0,1,2]).tolist())


def paired_group_bootstrap(y, groups, a, b, n=2000):
    # One paired draw per group, preserving all rows and both models within that draw.
    y, groups = np.asarray(y), np.asarray(groups)
    pa, pb = np.asarray(a).argmax(1), np.asarray(b).argmax(1)
    score = lambda actual, pred: f1_score(actual, pred, labels=[0,1,2], average='macro', zero_division=0)
    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    rng = np.random.default_rng(42)
    deltas = []
    for _ in range(n):
        ix = np.concatenate([index[g] for g in rng.choice(unique, len(unique), replace=True)])
        deltas.append(score(y[ix], pa[ix])-score(y[ix], pb[ix]))
    lo, hi = np.quantile(deltas, [.025, .975])
    return dict(delta=float(score(y, pa)-score(y, pb)), lower=float(lo), upper=float(hi),
                resamples=n, unit='group_id', interpretation='descriptive, conditional on fitted OOF predictions')


def data():
    clean = load_clean().sort_values('id').reset_index(drop=True)
    split = load_splits(clean=clean)
    joined = split[['id','split','fold']].merge(clean, on='id', validate='one_to_one')
    dev = joined.loc[joined.split == 'train'].sort_values('id').reset_index(drop=True)
    test = joined.loc[joined.split == 'test'].sort_values('id').reset_index(drop=True)
    if set(dev.group_id) & set(test.group_id) or dev.groupby('group_id').fold.nunique().max() > 1:
        raise ValueError('Official group isolation failed')
    return clean, dev, test


def audit(out):
    clean, dev, test = data()
    raw = pd.read_csv(DATA_ROOT / 'raw' / 'stock_market_crash_2022.csv', keep_default_na=False)
    checks = []
    for directory in sorted((ROOT/'Member5_file'/'evidence').glob('textcnn__*/')):
        if not (directory/'summary.json').exists():
            continue
        summary = json.loads((directory/'summary.json').read_text())
        csv = pd.read_csv(directory/'oof_predictions.csv').set_index('id').loc[dev.id]
        p = csv[[f'prob_{c}' for c in range(3)]].to_numpy()
        if not np.array_equal(csv.label.to_numpy(), dev.label.to_numpy()):
            raise ValueError('Historical OOF labels misaligned')
        value = metrics(dev.label, p)
        if not np.isclose(value['macro_f1'], summary['metrics']['macro_f1'], atol=1e-12):
            raise ValueError('Historical metrics do not reconcile')
        checks.append(dict(run_id=directory.name, **{k:v for k,v in value.items() if k not in ['classification_report','confusion_matrix']}))
    baseline = np.full((len(dev),3), np.nan)
    start = time.perf_counter()
    for fold in range(5):
        fit, valid = dev.fold.to_numpy() != fold, dev.fold.to_numpy() == fold
        model = make_pipeline(TfidfVectorizer(ngram_range=(1,1), min_df=1),
                              LogisticRegression(C=1, max_iter=2000, random_state=42))
        model.fit(dev.loc[fit,'text_norm'], dev.loc[fit,'label'])
        baseline[valid] = model.predict_proba(dev.loc[valid,'text_norm'])
    baseline_seconds = time.perf_counter()-start
    official = pd.read_csv(ROOT/'results'/'baseline.csv')
    official = official.loc[official.run_id == 'baseline__tfidf__20261102-181839-3628'].iloc[0]
    bm = metrics(dev.label, baseline)
    if not np.isclose(bm['macro_f1'], official.macro_f1, atol=1e-12):
        raise ValueError('Official baseline reproduction differs')
    prediction_frame(dev, baseline).to_csv(out/'baseline_oof.csv', index=False)
    sample = dev.groupby('label', group_keys=False).sample(n=20, random_state=42)
    sample[['id','group_id','label','text_raw','text_clean','text_norm']].to_csv(out/'data_review_60.csv', index=False)
    result = dict(raw_rows=len(raw), clean_rows=len(clean), development_rows=len(dev),
        reserved_rows=len(test), data_sha256=file_hash(CLEAN_DIR/'clean_v2.csv'),
        split_sha256=file_hash(ROOT/'splits'/'folds_v1.csv'),
        historical_reconciled=checks, baseline=bm, baseline_fit_inference_seconds=baseline_seconds,
        test_labels_scored=False, test_history='Previously evaluated by other members; not a pristine team holdout',
        manual_label_review='Not independently human relabelled; diagnostic sample only',
        limitations=['historical outer OOF already inspected', 'possible lexicon-generated labels',
                    'no evidence of financial-return prediction'])
    write_json(out/'audit.json', result)
    return result


def prediction_frame(dev, p):
    result = dev[['id','group_id','fold','label']].copy()
    result['prediction'] = p.argmax(1)
    for c in range(3):
        result[f'prob_{c}'] = p[:,c]
    return result


class Study:
    def __init__(self, out, device, hours):
        self.out, self.device = out, device
        self.cache = DATA_ROOT/'local_artifacts'/'m5_refinement'/out.name
        self.cache.mkdir(parents=True, exist_ok=True)
        self.limit = hours * 3600
        self.clean, self.dev, self.test = data()
        self.x, self.mask = load_token_cache(EMB_DIR, self.dev.id.tolist())
        self.y = self.dev.label.to_numpy()
        manifest = json.loads((ROOT/'Member5_file'/'evidence'/'embedding_manifest.json').read_text())
        for filename, checksum in manifest['files'].items():
            if file_hash(EMB_DIR/filename) != checksum:
                raise ValueError(f'Embedding checksum differs: {filename}')
        self.protocol = dict(candidates=CANDIDATES, seeds=SEEDS, outer_folds=5, inner_folds=3,
            device=device, budget_seconds=self.limit, num_threads=2,
            data_sha256=file_hash(CLEAN_DIR/'clean_v2.csv'),
            split_sha256=file_hash(ROOT/'splits'/'folds_v1.csv'),
            encoder_revision=manifest['revision'], max_len=manifest['max_len'],
            code_sha256=file_hash(__file__), training_sha256=file_hash(ROOT/'src'/'training.py'),
            model_sha256=file_hash(ROOT/'Member5_file'/'mvp.py'),
            versions={p:importlib.metadata.version(p) for p in ['torch','numpy','pandas','scikit-learn']},
            python=platform.python_version())
        path = out/'protocol.json'
        if path.exists() and protocol_signature(json.loads(path.read_text())) != protocol_signature(self.protocol):
            raise ValueError('Study protocol changed; choose a fresh run ID')
        write_json(path, self.protocol)

    def spent(self):
        rows = [json.loads(p.read_text()) for p in self.cache.glob('*/fit.json')]
        return sum(r.get('seconds',0) for r in rows), max([r.get('seconds',0) for r in rows] or [1])

    def fit(self, key, fit, valid, cfg, seed, *, selection=True, weights=False):
        used, longest = self.spent()
        if not (self.cache/key/'fit.json').exists() and used + max(2*longest, 30) > self.limit:
            raise TimeoutError('Study budget exhausted; completed fits retained for resume')
        provenance = dict(fit_ids=self.dev.iloc[fit].id.tolist(), valid_ids=self.dev.iloc[valid].id.tolist(),
                          protocol_signature=protocol_signature(self.protocol), purpose='inner' if selection else 'outer_refit')
        return fit_cached(self.cache, key, self.x[fit], self.y[fit], self.mask[fit],
            self.x[valid], self.y[valid] if selection else None, self.mask[valid], cfg,
            seed, self.device, provenance, save_weights=weights)

    def select(self, positions, names, tag):
        frame = self.dev.iloc[positions].reset_index(drop=True)
        rows = []
        for name in names:
            for k, (fit, valid) in enumerate(inner_partitions(frame)):
                for seed in SEEDS:
                    result = self.fit(f'{tag}_{name}_i{k}_s{seed}', positions[fit], positions[valid],
                                      CANDIDATES[name], seed)
                    rows.append(dict(candidate=name, inner_fold=k, seed=seed,
                        score=metrics(self.y[positions[valid]], result['probabilities'])['macro_f1'],
                        best_epoch=result['history']['best_epoch'], parameters=result['parameters']))
            pd.DataFrame(rows).to_csv(self.out/f'{tag}_inner_scores.csv', index=False)
        return choose_candidate(rows, names), rows

    def benchmark(self):
        fit, valid = inner_partitions(self.dev.loc[self.dev.fold != 0].reset_index(drop=True))[0]
        outer = np.flatnonzero(self.dev.fold.to_numpy() != 0)
        times = []
        for name in ['C0','C1']:
            r = self.fit(f'pilot_{name}', outer[fit], outer[valid], CANDIDATES[name], 42)
            times.append(r['seconds'])
        # Conservative: selection, refits and final selection all cost the slower pilot fit.
        full = max(times)*463
        reduced = max(times)*247
        names = list(CANDIDATES) if full <= 3*3600 else ['C0','C1','C3','C7']
        affordable = full <= 3*3600 or reduced <= 3*3600
        value = dict(pilot_seconds=times, full_projected_seconds=full,
            reduced_projected_seconds=reduced, candidate_ids=names, feasible=affordable,
            budget_selection_uses_scores=False, actual_device=self.device)
        write_json(self.out/'benchmark.json', value)
        return value

    def run(self):
        plan = json.loads((self.out/'benchmark.json').read_text()) if (self.out/'benchmark.json').exists() else self.benchmark()
        if not plan['feasible']:
            write_json(self.out/'status.json', {'status':'budget_infeasible','complete':False})
            return
        names = plan['candidate_ids']
        oof = {kind:{seed:np.full((len(self.dev),3),np.nan) for seed in SEEDS} for kind in ['selected','C0']}
        selections = []
        for fold in range(5):
            outer = np.flatnonzero(self.dev.fold.to_numpy() != fold)
            valid = np.flatnonzero(self.dev.fold.to_numpy() == fold)
            winner, rows = self.select(outer, names, f'outer{fold}')
            control = choose_candidate(rows, ['C0'])
            selections.append(dict(fold=fold, **winner))
            for kind, choice in [('selected',winner),('C0',control)]:
                cfg = {**CANDIDATES[choice['candidate']], 'max_epochs':choice['epochs']}
                for seed in SEEDS:
                    r = self.fit(f'outer{fold}_{kind}_s{seed}', outer, valid, cfg, seed,
                                 selection=False, weights=True)
                    oof[kind][seed][valid] = r['probabilities']
            write_json(self.out/'status.json', dict(status='running', completed_outer_folds=fold+1, complete=False))
            write_json(self.out/'outer_selections.json', selections)
        for kind in oof:
            for seed, p in oof[kind].items():
                validate_proba(p,len(self.dev))
                prediction_frame(self.dev,p).to_csv(self.out/f'{kind}_seed{seed}_oof.csv',index=False)
        winner, _ = self.select(np.arange(len(self.dev)), names, 'final')
        write_json(self.out/'final_selection.json',winner)
        xt, mt = load_token_cache(EMB_DIR,self.test.id.tolist())
        cfg = {**CANDIDATES[winner['candidate']], 'max_epochs':winner['epochs']}
        used,longest = self.spent()
        if used+2*longest > self.limit:
            raise TimeoutError('Insufficient budget for final refit')
        final = fit_cached(self.cache, 'final_seed42', self.x,self.y,self.mask,xt,None,mt,
            cfg,42,self.device,dict(fit_ids=self.dev.id.tolist(),prediction_ids=self.test.id.tolist(),
                protocol_signature=protocol_signature(self.protocol),purpose='final_no_test_scoring'),save_weights=True)
        # Team format, explicit actual seed. Do not call shared writer with its fixed seed tag.
        pred_dir = self.cache/'preds'
        pred_dir.mkdir(exist_ok=True)
        destination = pred_dir/f'{self.out.name}.npz'
        payload=dict(oof_ids=self.dev.id.to_numpy(),oof_proba=oof['selected'][42],
                     test_ids=self.test.id.to_numpy(),test_proba=final['probabilities'],
                     classes=np.array([0,1,2]),run_id=self.out.name,**{**VERSION_TAGS,'seed':42})
        if not destination.exists():
            np.savez(destination,**payload)
        else:
            with np.load(destination) as old:
                if any(not np.array_equal(old[k],v) for k,v in payload.items()):
                    raise ValueError('Refusing to overwrite different final predictions')
        self.export_fits()
        report(self.out)
        write_json(self.out/'status.json',dict(status='complete',complete=True,
                   training_seconds=self.spent()[0],test_labels_scored=False,
                   prediction_archive=str(destination),weight_file=str(self.cache/'final_seed42'/'weights.safetensors')))

    def export_fits(self):
        records = [json.loads(p.read_text()) | {'key':p.parent.name} for p in sorted(self.cache.glob('*/fit.json'))]
        write_json(self.out/'training_audit.json',records)


def report(out):
    baseline=pd.read_csv(out/'baseline_oof.csv')
    y,groups=baseline.label.to_numpy(),baseline.group_id.to_numpy()
    bp=baseline[[f'prob_{c}' for c in range(3)]].to_numpy()
    records,intervals=[],[]
    for seed in SEEDS:
        frames={kind:pd.read_csv(out/f'{kind}_seed{seed}_oof.csv').set_index('id').loc[baseline.id]
                for kind in ['selected','C0']}
        ps={kind:frame[[f'prob_{c}' for c in range(3)]].to_numpy() for kind,frame in frames.items()}
        for kind,p in ps.items():
            value=metrics(y,p)
            write_json(out/f'{kind}_seed{seed}_metrics.json',value)
            records.append(dict(model=kind,seed=seed,**{k:v for k,v in value.items() if not isinstance(v,(dict,list))}))
        for control,p in [('C0',ps['C0']),('baseline',bp)]:
            intervals.append(dict(seed=seed,comparison=f'selected minus {control}',
                                  **paired_group_bootstrap(y,groups,ps['selected'],p)))
    table=pd.DataFrame(records)
    table.to_csv(out/'comparison.csv',index=False)
    write_json(out/'bootstrap.json',intervals)
    selected=table.loc[table.model=='selected'].sort_values('seed')
    control=table.loc[table.model=='C0'].sort_values('seed')
    delta=selected.macro_f1.to_numpy()-control.macro_f1.to_numpy()
    regressions=[float(selected[f'f1_{c}'].mean()-control[f'f1_{c}'].mean()) for c in range(3)]
    result=dict(selected_mean=float(selected.macro_f1.mean()),selected_sample_sd=float(selected.macro_f1.std(ddof=1)),
        control_mean=float(control.macro_f1.mean()),baseline_macro_f1=metrics(y,bp)['macro_f1'],
        mean_delta=float(delta.mean()),positive_seeds=int((delta>0).sum()),class_mean_deltas=regressions,
        practical_target_met=bool(delta.mean()>=.01 and (delta>0).sum()>=2 and min(regressions)>=-.02),
        statistical_advantage_confirmed=all(r['lower']>0 for r in intervals if r['comparison']=='selected minus C0'),
        limitations=['Conditional OOF bootstrap does not include model-selection/training uncertainty',
                     'Historical outer OOF was inspected before this preregistered study',
                     'Team test set was previously inspected; no fresh independent test claim',
                     'Baseline not retuned in this study', 'No independent human ground-truth labels'])
    write_json(out/'conclusion.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    p=pd.read_csv(out/'selected_seed42_oof.csv')
    ConfusionMatrixDisplay.from_predictions(p.label,p.prediction,labels=[0,1,2],
        display_labels=['Neutral','Positive','Negative'],cmap='Blues')
    plt.title('Selected TextCNN pipeline: outer OOF, seed 42')
    plt.tight_layout(); plt.savefig(out/'confusion_selected.png',dpi=160); plt.close()
    clean,dev,_=data()
    errors=dev.merge(p[['id','prediction','prob_0','prob_1','prob_2']],on='id',validate='one_to_one')
    errors['word_count']=errors.text_clean.str.split().str.len()
    errors['contains_negation']=errors.text_clean.str.contains(r"\b(?:not|never|no|n't)\b",case=False,regex=True)
    errors['contains_market_term']=errors.text_clean.str.contains(r'\b(?:bear|bull|short|dip|crash|rally)\b',case=False,regex=True)
    errors.loc[errors.label!=errors.prediction].to_csv(out/'error_cases.csv',index=False)
    plt.figure(figsize=(7,4))
    for kind in ['selected','C0']:
        part=table.loc[table.model==kind]
        plt.scatter(part.seed.astype(str),part.macro_f1,label=kind)
    plt.axhline(metrics(y,bp)['macro_f1'],linestyle='--',color='grey',label='Official baseline')
    plt.ylabel('Pooled outer OOF Macro-F1');plt.xlabel('Training seed');plt.legend();plt.tight_layout()
    plt.savefig(out/'seed_comparison.png',dpi=160);plt.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['audit','benchmark','run','report'])
    parser.add_argument('--run-id',default='textcnn_refinement_20261112')
    parser.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu')
    parser.add_argument('--hours',type=float,default=4)
    args=parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',args.run_id) or not 0<args.hours<=4:
        parser.error('Use a simple run ID and a positive budget no greater than four hours')
    out=EVIDENCE/args.run_id
    out.mkdir(parents=True,exist_ok=True)
    if args.stage=='audit':
        print(json.dumps(audit(out),indent=2),flush=True)
    elif args.stage=='report':
        report(out)
    else:
        if not (out/'audit.json').exists():
            audit(out)
        study=Study(out,args.device,args.hours)
        try:
            if args.stage=='benchmark':
                print(json.dumps(study.benchmark(),indent=2),flush=True)
            else:
                study.run()
        except Exception as exc:
            study.export_fits()
            write_json(out/'status.json',dict(status='interrupted',complete=False,error=repr(exc),
                                              training_seconds=study.spent()[0]))
            raise


if __name__=='__main__':
    main()
