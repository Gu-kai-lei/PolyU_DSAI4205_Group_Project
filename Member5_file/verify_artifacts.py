"""Independently reconcile saved predictions, split membership and reported metrics."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file
from sklearn.metrics import accuracy_score, f1_score

from mvp import ROOT, OUT, TextCNN, inputs, signature, validate_probabilities, verify_embeddings
from src.cache import load_preds, load_embedding
from src.config import DATA_ROOT
from src.training import predict_proba


def verify():
    torch.set_num_threads(2)
    clean, splits = inputs()
    verify_embeddings(clean)
    dev = splits[splits.split == 'train'].merge(clean[['id', 'label']], on='id', validate='one_to_one')
    test_ids = set(splits.loc[splits.split == 'test', 'id'])
    labels = clean.set_index('id').label
    groups = clean.set_index('id').group_id
    checks = []
    for path in sorted(OUT.glob('textcnn__*/summary.json')):
        summary = json.loads(path.read_text())
        for key, expected in signature().items():
            assert summary[key] == expected, key
        saved = load_preds(summary['run_id'])
        for part in ['oof', 'test']:
            validate_probabilities(saved[part + '_ids'], saved[part + '_proba'])
        assert set(saved['oof_ids']) == set(dev.id)
        assert set(saved['test_ids']) == test_ids
        expected_y = labels.loc[saved['oof_ids']].to_numpy()
        predicted = saved['oof_proba'].argmax(1)
        score = f1_score(expected_y, predicted, average='macro', labels=[0, 1, 2])
        assert np.isclose(score, summary['metrics']['macro_f1'])
        assert np.isclose(accuracy_score(expected_y, predicted), summary['metrics']['accuracy'])
        csv = pd.read_csv(path.parent / 'oof_predictions.csv').set_index('id').loc[saved['oof_ids']]
        np.testing.assert_allclose(csv[[f'prob_{c}' for c in range(3)]], saved['oof_proba'])
        np.testing.assert_array_equal(csv.label, expected_y)
        audit = json.loads((path.parent / 'training_audit.json').read_text())
        seen = []
        for fold in audit:
            fit, stop, valid = [set(fold[k]) for k in ['fit_ids', 'stop_ids', 'outer_validation_ids']]
            expected_valid = set(dev.loc[dev.fold == fold['fold'], 'id'])
            assert valid == expected_valid
            assert fit | stop == set(dev.id) - valid
            assert fit.isdisjoint(stop) and (fit | stop).isdisjoint(valid)
            group_sets = [set(groups.loc[list(ids)]) for ids in [fit, stop, valid]]
            assert all(group_sets[a].isdisjoint(group_sets[b]) for a, b in [(0, 1), (0, 2), (1, 2)])
            seen.extend(valid)
        assert len(seen) == len(set(seen)) == len(dev)
        assert summary['test_evaluated'] is False
        params = json.loads(summary['metrics']['params'])
        model = TextCNN(params['input_dim'])
        model.load_state_dict(load_file(str(DATA_ROOT / 'models' / f'{summary["run_id"]}.safetensors')))
        prefix = 'tokens_multi' if summary['metrics']['embedding'] == 'transformer_tokens' else 'fasttext_tok_full'
        mask_name = 'tokens_mask' if prefix == 'tokens_multi' else prefix + '_mask'
        sample_ids = saved['test_ids'][:8]
        reloaded = predict_proba(model, load_embedding(prefix, ids=sample_ids),
                                load_embedding(mask_name, ids=sample_ids), batch_size=8)
        np.testing.assert_allclose(reloaded, saved['test_proba'][:8], atol=1e-5, rtol=1e-4)
        checks.append({'run_id': summary['run_id'], 'macro_f1_recomputed': score,
                       'oof_rows': len(expected_y), 'test_rows': len(test_ids),
                       'safetensors_reload_predictions_match': True,
                       'all_split_and_probability_checks_passed': True})
    assert {json.loads(p.read_text())['metrics']['embedding'] for p in OUT.glob('textcnn__*/summary.json')} == {'fasttext_tok', 'transformer_tokens'}
    result = {'source': signature(), 'verified_runs': checks, 'test_labels_scored': False}
    (OUT / 'verification.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify()
