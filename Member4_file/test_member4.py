"""Regression checks for leakage, masks, reproducibility, and probability alignment."""
import unittest
import tempfile
import uuid
import shutil
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from src.data import load_clean, load_splits
from src.splitting import make_splits, validate_splits
from src.training import MLP, TrainConfig, seed_everything, fit_model, predict_proba, masked_mean, masked_max
from src.cache import load_embedding, save_preds, load_preds
import src.cache as cache


@contextmanager
def scratch():
    # Windows managed sandboxes cannot reopen mkdir(mode=0700) from tempfile.
    path = Path.cwd() / 'local_artifacts' / ('test_' + uuid.uuid4().hex)
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path)


class Member4Tests(unittest.TestCase):
    def test_split_is_deterministic_and_group_disjoint(self):
        clean = load_clean()
        a, b = make_splits(clean), make_splits(clean.sample(frac=1, random_state=9))
        pd.testing.assert_frame_equal(a, b)
        validate_splits(load_splits(clean=clean), clean)
        broken = a.copy()
        group = clean.group_id.value_counts().loc[lambda s:s > 1].index[0]
        indices = broken.index[broken.group_id == group]
        broken.loc[indices[0], ['split', 'fold']] = ['train', 3]
        broken.loc[indices[1], ['split', 'fold']] = ['test', -1]
        with self.assertRaises(ValueError):
            validate_splits(broken, clean)

    def test_padding_invariance_and_empty_sequences(self):
        x = torch.tensor([[[1., 2.], [3., 4.], [999., -999.]], [[3., 1.], [8., 6.], [1., 2.]]])
        mask = torch.tensor([[True, True, False], [False, False, False]])
        torch.testing.assert_close(masked_mean(x, mask), torch.tensor([[2., 3.], [0., 0.]]))
        torch.testing.assert_close(masked_max(x, mask), torch.tensor([[3., 4.], [0., 0.]]))
        x[:, 2] = -10000
        torch.testing.assert_close(masked_mean(x, mask), torch.tensor([[2., 3.], [0., 0.]]))

    def test_training_determinism_singleton_batch_and_checkpoint(self):
        torch.set_num_threads(2)
        x = np.random.default_rng(7).normal(size=(17, 4)).astype('float32')
        y = np.arange(17) % 3
        cfg = TrainConfig(epochs=3, patience=2, batch_size=8)
        results = []
        with scratch() as d:
            for _ in range(2):
                seed_everything(42)
                m = MLP(4, hidden=(12, 6))
                r = fit_model(m, x, y, x[:6], y[:6], config=cfg, checkpoint=Path(d)/'model.pt')
                self.assertIn(r['best_epoch'], range(1, 4))
                results.append(predict_proba(m, x))
            np.testing.assert_array_equal(*results)

    def test_cache_reorders_ids_and_probability_columns(self):
        ids, x = load_embedding('fasttext_mean_fold0')
        np.testing.assert_array_equal(load_embedding('fasttext_mean_fold0', ids[:5][::-1]), x[:5][::-1])
        with self.assertRaises(KeyError):
            load_embedding('fasttext_mean_fold0', [-999])
        old = cache.PRED_DIR
        with scratch() as d:
            cache.PRED_DIR = Path(d)
            try:
                p = np.array([[.2, .3, .5]])
                save_preds('test', oof_ids=[5], oof_proba=p, test_ids=[], test_proba=np.empty((0, 3)), classes=[2, 0, 1])
                np.testing.assert_allclose(load_preds('test')['oof_proba'], [[.3, .5, .2]])
            finally:
                cache.PRED_DIR = old


if __name__ == '__main__':
    unittest.main()
