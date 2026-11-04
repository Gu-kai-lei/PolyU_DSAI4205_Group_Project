"""Recover the exact Member-4 split file from delivered OOF evidence.

No new split is generated: the recovered bytes must match Member 4's SHA-256.
Run from the repository root: python member1/recover_member4_splits.py
"""
from pathlib import Path
import hashlib
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.data import check_clean, check_splits


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def recover(clean, oof, summary):
    check_clean(clean)
    signature = summary["embedding_details"]["signature"]
    # Member 4 exported pandas CSVs on Windows. Normalize serialization only.
    clean_hash = sha256(clean.to_csv(index=False, lineterminator="\r\n").encode())
    if clean_hash != signature["data_hash"]:
        raise ValueError("Clean data does not match Member 4's recorded data hash")
    if not oof.id.is_unique or not set(oof.id) <= set(clean.id):
        raise ValueError("OOF IDs must be unique and belong to clean_v2")
    if len(oof) != summary["development_rows"]:
        raise ValueError("Incomplete Member-4 OOF evidence")
    expected = clean.set_index("id").loc[oof.id, ["group_id", "label"]]
    if not (expected.to_numpy() == oof[["group_id", "label"]].to_numpy()).all():
        raise ValueError("OOF groups or labels differ from clean_v2")
    if set(oof.fold) != set(range(5)):
        raise ValueError("Expected all five development folds")
    splits = clean[["id", "group_id"]].merge(oof[["id", "fold"]], on="id", how="left")
    splits["split"] = splits.fold.notna().map({True: "train", False: "test"})
    splits["fold"] = splits.fold.fillna(-1).astype(int)
    splits = splits[["id", "group_id", "split", "fold"]].sort_values("id")
    check_splits(splits, clean)
    if int((splits.split == "test").sum()) != summary["test_rows"]:
        raise ValueError("Holdout row count differs from Member 4")
    payload = splits.to_csv(index=False, lineterminator="\r\n").encode()
    if sha256(payload) != signature["split_hash"]:
        raise ValueError("Recovered split bytes do not match Member 4's SHA-256")
    return payload


def main():
    source = ROOT / "Member4_file"
    if not source.exists():
        source = ROOT / "member1/evidence/member4"
    clean_path = ROOT / "data/clean_v2/clean_v2.csv"
    oof_path = source / "selected_oof_predictions.csv"
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text())
    payload = recover(pd.read_csv(clean_path), pd.read_csv(oof_path), summary)
    target = ROOT / "splits/folds_v1.csv"
    if target.exists() and target.read_bytes() != payload:
        raise ValueError("Refusing to overwrite a different existing folds_v1.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)
    evidence = {
        "method": "Recovered development id/fold assignments from Member-4 OOF; complement is holdout. No random repartitioning.",
        "split_sha256": sha256(payload),
        "matches_member4_split_hash": True,
        "clean_matches_member4_after_windows_csv_serialization": True,
        "development_rows": summary["development_rows"],
        "test_rows": summary["test_rows"],
        "sources": {str(p.relative_to(ROOT)): sha256(p.read_bytes())
                    for p in [clean_path, oof_path, summary_path]},
    }
    out = ROOT / "member1/split_provenance.json"
    out.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
