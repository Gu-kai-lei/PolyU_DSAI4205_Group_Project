"""Member 5 MVP. Run from a clone; caches stay in DATA_ROOT and never enter Git."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Save report figures without requiring a desktop GUI.
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.model_selection import StratifiedGroupKFold
from src.cache import load_embedding, save_embedding, save_preds
from src.config import CLEAN_DIR, DATA_ROOT, EMB_DIR, SEED, VERSION_TAGS
from src.data import load_clean, load_splits
from src.evaluate import evaluate, new_run_id, plot_confusion, save_result
from src.fasttext_emb import build_fold_embeddings
from src.training import TrainConfig, fit_model, predict_proba, seed_everything

OUT = ROOT / 'Member5_file' / 'evidence'
MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + '\n')


def inputs():
    clean = load_clean().sort_values('id').reset_index(drop=True)
    splits = load_splits(clean=clean)
    return clean, splits


def signature():
    return {**VERSION_TAGS, 'data_sha256': digest(CLEAN_DIR / 'clean_v2.csv'),
            'split_sha256': digest(ROOT / 'splits/folds_v1.csv')}


class TextCNN(nn.Module):
    """Full valid windows; one zero-extended window for nonempty texts shorter than k."""
    def __init__(self, input_dim, filters=100, dropout=0.5):
        super().__init__()
        self.kernels = (2, 3, 4, 5)
        self.convs = nn.ModuleList(nn.Conv1d(input_dim, filters, k) for k in self.kernels)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(filters * len(self.kernels), 3)

    def forward(self, x, mask=None):
        if mask is None:
            mask = torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        x = x.masked_fill(~mask.unsqueeze(-1), 0)
        if x.shape[1] < 5:
            extra = 5 - x.shape[1]
            x = F.pad(x, (0, 0, 0, extra))
            mask = F.pad(mask, (0, extra), value=False)
        lengths = mask.sum(1)
        features = []
        for k, conv in zip(self.kernels, self.convs):
            h = F.relu(conv(x.transpose(1, 2)))
            valid = mask.unfold(1, k, 1).all(-1)
            short = (lengths > 0) & (lengths < k)
            valid[:, 0] = valid[:, 0] | short
            pooled = h.masked_fill(~valid.unsqueeze(1), float('-inf')).amax(-1)
            pooled = torch.where(valid.any(1).unsqueeze(1), pooled, torch.zeros_like(pooled))
            features.append(pooled)
        return self.output(self.dropout(torch.cat(features, dim=1)))


def validate_probabilities(ids, proba):
    if len(set(ids)) != len(ids) or proba.shape != (len(ids), 3):
        raise ValueError('Duplicate IDs or wrong probability dimensions')
    if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any():
        raise ValueError('Nonfinite or out-of-range probabilities')
    if not np.allclose(proba.sum(1), 1, atol=1e-5):
        raise ValueError('Probability rows do not sum to one')


def encode(args):
    from huggingface_hub import model_info
    from transformers import AutoModel, AutoTokenizer
    clean, splits = inputs()
    revision = args.revision or model_info(MODEL).sha
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=revision, trust_remote_code=False)
    dev_ids = set(splits.loc[splits.split == 'train', 'id'])
    texts = clean.text_clean.tolist()
    lengths = [len(tokenizer(t, truncation=False)['input_ids']) for t, i in zip(texts, clean.id) if i in dev_ids]
    max_len = min(128, max(5, int(np.ceil(np.percentile(lengths, 99)))))
    manifest = {**signature(), 'model': MODEL, 'revision': revision, 'max_len': max_len,
                'sentence_pooling': 'attention-masked mean including special tokens, then L2',
                'sequence_policy': 'remove special tokens; compact real tokens; zero right padding',
                'length_source': 'development texts only, 99th percentile capped at model default 128',
                'precision': 'float32 inference; float16 token storage', 'frozen': True}
    manifest_path = EMB_DIR / 'm5_transformer_manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if all(old.get(k) == v for k, v in manifest.items()):
            print('Matching frozen embedding cache exists', flush=True)
            verify_embeddings(clean)
            return
        raise ValueError('Embedding cache has different provenance; use a new DATA_ROOT')
    model = AutoModel.from_pretrained(MODEL, revision=revision, use_safetensors=True,
                                      trust_remote_code=False).to(args.device).eval()
    model.requires_grad_(False)
    sentences, sequences, masks = [], [], []
    start = time.perf_counter()
    for begin in range(0, len(texts), args.encode_batch):
        batch = tokenizer(texts[begin:begin + args.encode_batch], padding='max_length',
                          truncation=True, max_length=max_len, return_tensors='pt',
                          return_special_tokens_mask=True)
        specials = batch.pop('special_tokens_mask').bool()
        attention = batch['attention_mask'].bool()
        with torch.inference_mode():
            h = model(**{k: v.to(args.device) for k, v in batch.items()}).last_hidden_state.cpu()
        mean = (h * attention.unsqueeze(-1)).sum(1) / attention.sum(1).unsqueeze(-1)
        sentences.append(F.normalize(mean, dim=1).numpy())
        seq = np.zeros((len(h), max_len, h.shape[-1]), dtype=np.float16)
        mask = np.zeros((len(h), max_len), dtype=bool)
        for j in range(len(h)):
            real = h[j][attention[j] & ~specials[j]].numpy()
            seq[j, :len(real)] = real
            mask[j, :len(real)] = True
        sequences.append(seq)
        masks.append(mask)
        print(f'Encoded {min(begin + args.encode_batch, len(texts))}/{len(texts)}', flush=True)
    for name, matrix in [('sbert_multi', np.concatenate(sentences)),
                         ('tokens_multi', np.concatenate(sequences)),
                         ('tokens_mask', np.concatenate(masks))]:
        save_embedding(name, clean.id, matrix)
    manifest.update(seconds=time.perf_counter() - start,
                    files={p.name: digest(p) for name in ['sbert_multi', 'tokens_multi', 'tokens_mask']
                           for p in [EMB_DIR / f'{name}.npy', EMB_DIR / f'{name}_ids.npy']})
    dump(manifest_path, manifest)
    dump(OUT / 'embedding_manifest.json', manifest)
    verify_embeddings(clean)


def verify_embeddings(clean):
    manifest = json.loads((EMB_DIR / 'm5_transformer_manifest.json').read_text())
    for key, value in signature().items():
        if manifest.get(key) != value:
            raise ValueError(f'Stale cache: {key}')
    for file, expected in manifest['files'].items():
        if digest(EMB_DIR / file) != expected:
            raise ValueError(f'Cache checksum mismatch: {file}')
    sentence = load_embedding('sbert_multi', ids=clean.id)
    tokens = load_embedding('tokens_multi', ids=clean.id)
    mask = load_embedding('tokens_mask', ids=clean.id)
    assert tokens.shape[:2] == mask.shape and tokens.shape[0] == len(clean)
    assert np.isfinite(tokens).all() and np.isfinite(sentence).all()
    assert np.allclose(np.linalg.norm(sentence, axis=1), 1, atol=1e-5)
    assert np.all(tokens[~mask] == 0)
    print('Embedding integrity checks passed', flush=True)


def train(args):
    clean, splits = inputs()
    dev = splits[splits.split == 'train'].merge(clean[['id', 'label']], on='id', validate='one_to_one')
    test_ids = splits.loc[splits.split == 'test', 'id'].to_numpy()
    ids, y = dev.id.to_numpy(), dev.label.to_numpy()
    embedding = 'transformer_tokens' if args.embedding == 'transformer' else 'fasttext_tok'
    if args.embedding == 'transformer':
        verify_embeddings(clean)
    else:
        dev_text = clean[clean.id.isin(ids)].text_norm
        max_len = max(5, int(np.ceil(np.percentile(dev_text.str.split().str.len(), 99))))
        # Regenerate rather than silently trust an unversioned shared cache.
        started = time.perf_counter()
        summary, _ = build_fold_embeddings(clean, splits, max_len)
        dump(OUT / 'fasttext_manifest.json', {**signature(), 'max_len': max_len,
             'length_source': 'development texts only', 'audits': summary.to_dict('records'),
             'seconds': time.perf_counter() - started})
    run_id = new_run_id('textcnn', embedding)
    run_dir = OUT / run_id
    run_dir.mkdir(parents=True)
    oof = np.full((len(dev), 3), np.nan)
    fold_rows, histories, epochs = [], [], []
    started = time.perf_counter()
    def arrays(tag, requested_ids):
        name = 'tokens_multi' if args.embedding == 'transformer' else f'fasttext_tok_{tag}'
        mask_name = 'tokens_mask' if args.embedding == 'transformer' else name + '_mask'
        return load_embedding(name, ids=requested_ids), load_embedding(mask_name, ids=requested_ids)
    for k in range(5):
        fit_positions = np.flatnonzero(dev.fold.to_numpy() != k)
        valid_positions = np.flatnonzero(dev.fold.to_numpy() == k)
        outer = dev.iloc[fit_positions]
        inner = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
        fi, si = next(inner.split(outer, outer.label, outer.group_id))
        assert set(outer.iloc[fi].group_id).isdisjoint(set(outer.iloc[si].group_id))
        x, mask = arrays(f'fold{k}', ids[fit_positions])
        xv, mv = arrays(f'fold{k}', ids[valid_positions])
        config = TrainConfig(epochs=args.epochs, patience=3, batch_size=32, device=args.device)
        seed_everything(SEED)
        selector = TextCNN(x.shape[-1])
        selection = fit_model(selector, x[fi], y[fit_positions][fi], x[si], y[fit_positions][si],
                              train_mask=mask[fi], valid_mask=mask[si], config=config)
        epoch = selection['best_epoch']
        epochs.append(epoch)
        seed_everything(SEED)
        model = TextCNN(x.shape[-1])
        config.epochs = epoch
        fitted = fit_model(model, x, y[fit_positions], train_mask=mask, config=config)
        oof[valid_positions] = predict_proba(model, xv, mv, batch_size=64, device=args.device)
        row = evaluate(y[valid_positions], oof[valid_positions].argmax(1), run_id=run_id,
                       model='textcnn', embedding=embedding, split='cv', params={'fold': k, 'epochs': epoch})
        fold_rows.append(row)
        histories.append({'fold': k, 'selection': selection, 'refit': fitted,
                          'fit_ids': ids[fit_positions][fi].tolist(), 'stop_ids': ids[fit_positions][si].tolist(),
                          'outer_validation_ids': ids[valid_positions].tolist()})
        dump(run_dir / 'training_audit.json', histories)
        dump(run_dir / 'progress.json', {'completed_folds': k + 1, 'epochs': epochs})
        print(f'{embedding} fold {k}: epochs={epoch}, macro_f1={row["macro_f1"]:.4f}', flush=True)
    validate_probabilities(ids, oof)
    config.epochs = max(1, int(np.median(epochs)))
    x, mask = arrays('full', ids)
    xt, mt = arrays('full', test_ids)
    seed_everything(SEED)
    model = TextCNN(x.shape[-1])
    fit_model(model, x, y, train_mask=mask, config=config)
    from safetensors.torch import save_file
    weight_path = DATA_ROOT / 'models' / f'{run_id}.safetensors'
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(weight_path))
    test_proba = predict_proba(model, xt, mt, batch_size=64, device=args.device)
    validate_probabilities(test_ids, test_proba)
    assert set(ids).isdisjoint(set(test_ids))
    save_preds(run_id, oof_ids=ids, oof_proba=oof, test_ids=test_ids,
               test_proba=test_proba, classes=[0, 1, 2])
    params = {'input_dim': int(x.shape[-1]), 'kernels': [2, 3, 4, 5], 'filters_per_kernel': 100, 'dropout': .5,
              'outer_epochs': epochs, 'full_epochs': config.epochs, 'config': asdict(config)}
    row = evaluate(y, oof.argmax(1), run_id=run_id, model='textcnn', embedding=embedding,
                   split='cv', params=params, train_seconds=time.perf_counter() - started,
                   notes='MVP fixed configuration; grouped inner early stopping then outer refit; test unscored')
    save_result(row, 'm5_textcnn')
    pd.DataFrame([row]).to_csv(run_dir / 'metrics.csv', index=False)
    pd.DataFrame(fold_rows).to_csv(run_dir / 'fold_metrics.csv', index=False)
    pred = pd.DataFrame({'id': ids, 'fold': dev.fold, 'label': y, 'prediction': oof.argmax(1)})
    for c in range(3):
        pred[f'prob_{c}'] = oof[:, c]
    pred.to_csv(run_dir / 'oof_predictions.csv', index=False)
    title = 'Transformer' if args.embedding == 'transformer' else 'FastText'
    plot_confusion(y, oof.argmax(1), f'TextCNN + {title}\nOuter OOF', run_dir / 'confusion.png')
    dump(run_dir / 'training_audit.json', histories)
    baseline = pd.read_csv(ROOT / 'results/baseline.csv').iloc[0]
    dump(run_dir / 'summary.json', {**signature(), 'run_id': run_id, 'metrics': row,
         'fold_macro_f1_mean': float(np.mean([r['macro_f1'] for r in fold_rows])),
         'fold_macro_f1_sample_sd': float(np.std([r['macro_f1'] for r in fold_rows], ddof=1)),
         'baseline_macro_f1': float(baseline.macro_f1),
         'difference_from_baseline': row['macro_f1'] - float(baseline.macro_f1),
         'test_evaluated': False, 'test_prediction_rows': len(test_ids), 'development_rows': len(ids),
         'environment': {p: importlib.metadata.version(p) for p in ['torch', 'transformers',
             'sentence-transformers', 'numpy', 'scipy', 'pandas', 'scikit-learn', 'gensim']},
         'python': platform.python_version(), 'device': args.device,
         'source_base': '0498603aa095f27af50b0e08c9ebec6ffdd1d0cc',
         'limitations': ['single seed and bounded MVP training budget',
                         'given labels may be lexicon generated', 'no independent human-label or financial-return validation',
                         'test partition had already been evaluated by other members; not a pristine team holdout']})
    print(json.dumps(row, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['encode', 'train'])
    parser.add_argument('--embedding', choices=['transformer', 'fasttext'], default='transformer')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cpu')
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--encode-batch', type=int, default=16)
    parser.add_argument('--revision', default='4328cf26390c98c5e3c738b4460a05b95f4911f5')
    args = parser.parse_args()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    encode(args) if args.stage == 'encode' else train(args)
