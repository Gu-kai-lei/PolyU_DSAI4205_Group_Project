"""Check delivered evidence recovery and fail closed on altered evidence."""
import json
import unittest
from pathlib import Path

import pandas as pd

from member1.recover_member4_splits import recover, sha256

ROOT = Path(__file__).resolve().parents[1]


class SplitRecoveryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clean = pd.read_csv(ROOT / "data/clean_v2/clean_v2.csv")
        source = ROOT / "Member4_file"
        if not source.exists():
            source = ROOT / "member1/evidence/member4"
        cls.summary = json.loads((source / "summary.json").read_text())
        cls.oof = pd.read_csv(source / "selected_oof_predictions.csv")

    def test_exact_member4_hash(self):
        payload = recover(self.clean, self.oof, self.summary)
        self.assertEqual(sha256(payload), self.summary["embedding_details"]["signature"]["split_hash"])

    def test_missing_oof_rejected(self):
        with self.assertRaises(ValueError):
            recover(self.clean, self.oof.iloc[:-1], self.summary)

    def test_changed_fold_rejected(self):
        changed = self.oof.copy()
        changed.loc[0, "fold"] = (changed.loc[0, "fold"] + 1) % 5
        with self.assertRaises(ValueError):
            recover(self.clean, changed, self.summary)

    def test_changed_text_rejected(self):
        changed = self.clean.copy()
        changed.loc[0, "text_norm"] += " changed"
        with self.assertRaises(ValueError):
            recover(changed, self.oof, self.summary)


if __name__ == "__main__":
    unittest.main()
