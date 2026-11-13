"""Regression checks for selection isolation, resumability and actual model behavior."""
import numpy as np
import pandas as pd
import pytest
import torch

from refine import (CANDIDATES, choose_candidate, inner_partitions,
                    paired_group_bootstrap, fit_cached, protocol_signature,
                    load_token_cache, write_json)


def test_grouped_inner_splits_exclude_outer_validation():
    rows = pd.DataFrame({'id': range(72), 'group_id': np.repeat(range(36), 2),
                         'label': np.repeat(np.arange(36) % 3, 2),
                         'fold': np.repeat(np.arange(36) % 5, 2)})
    outer = rows.loc[rows.fold != 0].reset_index(drop=True)
    pairs = inner_partitions(outer)
    seen = []
    for fit, valid in pairs:
        assert set(outer.iloc[fit].group_id).isdisjoint(outer.iloc[valid].group_id)
        assert set(outer.iloc[fit].id).isdisjoint(rows.loc[rows.fold == 0, 'id'])
        seen.extend(valid)
    assert sorted(seen) == list(range(len(outer)))


def test_selection_rejects_incomplete_candidate_and_breaks_ties_by_size():
    scores = [{'candidate': name, 'inner_fold': fold, 'seed': seed, 'score': .6,
               'best_epoch': 3, 'parameters': count}
              for name, count in [('C0', 100), ('C1', 32)]
              for fold in range(3) for seed in [42, 17, 2026]]
    assert choose_candidate(scores, ['C0', 'C1'])['candidate'] == 'C1'
    with pytest.raises(ValueError, match='complete'):
        choose_candidate(scores[:-1], ['C0', 'C1'])


def test_bootstrap_preserves_pairing_and_identical_predictions_have_zero_delta():
    y = np.array([0, 1, 2, 0, 1, 2])
    p = np.eye(3)[[0, 1, 1, 0, 2, 2]]
    result = paired_group_bootstrap(y, np.array([1, 1, 2, 3, 3, 4]), p, p, n=30)
    assert result['lower'] == result['upper'] == result['delta'] == 0


def test_token_cache_rejects_wrong_alignment_and_padding(tmp_path):
    np.save(tmp_path / 'tokens_multi.npy', np.ones((2, 5, 4), dtype=np.float16))
    np.save(tmp_path / 'tokens_multi_ids.npy', [3, 3])
    np.save(tmp_path / 'tokens_mask.npy', np.ones((2, 5), dtype=bool))
    np.save(tmp_path / 'tokens_mask_ids.npy', [3, 4])
    with pytest.raises(ValueError):
        load_token_cache(tmp_path, [3, 4])


def test_resume_uses_verified_completed_fit_and_rejects_changed_inputs(tmp_path):
    x = np.random.default_rng(2).normal(size=(12, 5, 4)).astype('float32')
    mask = np.ones((12, 5), dtype=bool)
    y = np.arange(12) % 3
    cfg = {**CANDIDATES['C1'], 'max_epochs': 1, 'patience': 1}
    first = fit_cached(tmp_path, 'one', x[:9], y[:9], mask[:9], x[9:], y[9:], mask[9:],
                       cfg, 42, 'cpu', {'fit_ids': list(range(9)), 'valid_ids': [9,10,11]})
    second = fit_cached(tmp_path, 'one', x[:9], y[:9], mask[:9], x[9:], y[9:], mask[9:],
                        cfg, 42, 'cpu', {'fit_ids': list(range(9)), 'valid_ids': [9,10,11]})
    assert second['resumed'] is True
    np.testing.assert_array_equal(first['probabilities'], second['probabilities'])
    with pytest.raises(ValueError, match='signature'):
        fit_cached(tmp_path, 'one', x[:9], (y[:9]+1)%3, mask[:9], x[9:], y[9:], mask[9:],
                   cfg, 42, 'cpu', {'fit_ids': list(range(9)), 'valid_ids': [9,10,11]})


def test_atomic_json_survives_replacement(tmp_path):
    p = tmp_path / 'state.json'
    write_json(p, {'status': 'running'})
    write_json(p, {'status': 'complete'})
    import json
    assert json.loads(p.read_text()) == {'status': 'complete'}


def test_signature_changes_with_protocol_not_dictionary_order():
    assert protocol_signature({'a': 1, 'b': 2}) == protocol_signature({'b': 2, 'a': 1})
    assert protocol_signature({'a': 1}) != protocol_signature({'a': 2})
