import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT
from src.project_review import archived_oof, baseline_oof, development_data, score, team_evidence
from src.member1 import nested_svm_oof


class ProjectReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = development_data()
        cls.table, cls.probs = team_evidence(cls.dev)

    def test_fixed_baseline_matches_historical_development_score(self):
        score_row = score(self.dev, baseline_oof(self.dev), "baseline")
        self.assertAlmostEqual(score_row["macro_f1"], 0.5819958867503482)

    def test_mlp_must_match_run_and_aligned_official_folds(self):
        columns = ["prob_neutral", "prob_positive", "prob_negative"]
        path = PROJECT_ROOT / "experiments/Member4_file" / "evidence" / "selected_oof_predictions.csv"
        metric = self.table.loc[self.table["model"].eq("MLP"), "macro_f1"].item()
        with tempfile.TemporaryDirectory() as folder:
            handoff = Path(folder) / "oof.csv"
            data = pd.read_csv(path).sample(frac=1, random_state=2)
            data.to_csv(handoff, index=False)
            np.testing.assert_allclose(archived_oof(self.dev, handoff, columns, metric), self.probs["MLP"])
            with self.assertRaisesRegex(ValueError, "differs from run score"):
                archived_oof(self.dev, handoff, columns, 0.5402844194289936)
            data.iloc[0, data.columns.get_loc("fold")] = (int(data.iloc[0]["fold"]) + 1) % 5
            data.to_csv(handoff, index=False)
            with self.assertRaises(AssertionError):
                archived_oof(self.dev, handoff, columns, metric)

    def test_corrected_bilstm_oof_matches_artifact(self):
        row = score(self.dev, self.probs["BiLSTM（分组内层早停）"], "bilstm")
        self.assertAlmostEqual(row["macro_f1"], 0.5211176496902978)

    def test_svm_selects_and_scores_each_outer_fold(self):
        subset = self.dev.groupby(["fold", "label"]).head(8)
        vectors = subset[["char_len", "word_len", "n_hashtags"]].to_numpy()
        probability, selected = nested_svm_oof(vectors, subset)
        self.assertEqual(selected["fold"].tolist(), [0, 1, 2, 3, 4])
        np.testing.assert_allclose(probability.sum(1), 1)


if __name__ == "__main__":
    unittest.main()
