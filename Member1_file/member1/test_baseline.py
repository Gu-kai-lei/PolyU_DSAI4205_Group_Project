"""Run from project root: python -m unittest member1.test_baseline -v"""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from member1 import baseline as b
from src import cache, evaluate as evaluation
from src.data import check_splits


class BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = b.load_clean().sort_values("id").reset_index(drop=True)
        local = b.provisional_splits(cls.clean)
        cls.splits = local.rename(columns={"partition": "split"}).assign(
            split=lambda x: x.split.map({"development": "train", "holdout": "test"}))

    def test_group_and_id_corruption_rejected(self):
        bad = self.splits.copy()
        bad.loc[0, "group_id"] = -12345
        with self.assertRaises(ValueError):
            check_splits(bad, self.clean)
        with self.assertRaises(ValueError):
            check_splits(self.splits.iloc[1:], self.clean)
        group = self.clean.group_id.value_counts().loc[lambda x: x > 1].index[0]
        bad = self.splits.copy()
        idx = bad.index[bad.group_id == group][0]
        bad.loc[idx, ["split", "fold"]] = ["train", 4] if bad.loc[idx, "fold"] != 4 else ["test", -1]
        with self.assertRaises(ValueError):
            check_splits(bad, self.clean)

    def test_validation_does_not_expand_vocabulary(self):
        model = b.make_model().fit(
            ["good nice", "great gain", "bad loss", "awful fall", "flat market", "steady price"],
            [1, 1, 2, 2, 0, 0])
        model.predict(["validationonlytoken"])
        self.assertNotIn("validationonlytoken", model.named_steps["tfidf"].vocabulary_)

    def test_missing_prerequisites_do_not_write(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(b, "readiness", return_value={"ready_for_team_run": False}):
            with self.assertRaises(RuntimeError):
                b.run(output_root=Path(temp), mode="team")
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_team_integration_in_temporary_directory(self):
        # Test fixture only: never create an actual folds_v1.csv or project Results.
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            with (patch.object(b, "readiness", return_value={"ready_for_team_run": True}),
                  patch.object(b, "load_splits", return_value=self.splits),
                  patch.object(b, "FIGURES_DIR", root / "figures"),
                  patch.object(evaluation, "RESULTS_DIR", root / "results"),
                  patch.object(cache, "PRED_DIR", root / "preds")):
                out = b.run(output_root=root / "outputs", mode="team")
                table = pd.read_csv(root / "results/baseline.csv")
                self.assertEqual(len(table), 1)
                preds = cache.load_preds(table.iloc[0].run_id)
                expected = set(self.splits.loc[self.splits.split == "train", "id"])
                self.assertEqual(set(preds["oof_ids"]), expected)
                self.assertEqual(len(preds["oof_ids"]), len(expected))
                self.assertEqual(preds["test_proba"].shape, (0, 3))
                self.assertEqual(len(preds["test_ids"]), 0)
                np.testing.assert_allclose(preds["oof_proba"].sum(axis=1), 1)
                self.assertTrue((out / "confusion_matrix.png").exists())
                oof = pd.read_csv(out / "oof_predictions.csv")
                self.assertAlmostEqual(table.iloc[0].macro_f1, b.score(oof.label, oof.prediction)["f1_macro"])


if __name__ == "__main__":
    unittest.main()
