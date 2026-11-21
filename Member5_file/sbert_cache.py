"""Prepare or check the two SBERT representations used by the team."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from src.config import CLEAN_DIR, EMB_DIR, SPLITS_DIR
from src.data import load_clean, load_splits

REFERENCE = Path(__file__).with_name("embedding_manifest.json")


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare_embeddings(generate_if_missing=False, representations=("sbert_multi", "tokens_multi", "tokens_mask")):
    """Use the shared cache if present, or optionally generate it with mvp.py."""
    reference = json.loads(REFERENCE.read_text())
    clean_path = CLEAN_DIR / "clean_v2.csv"
    split_path = SPLITS_DIR / "folds_v1.csv"
    for path, key in [(clean_path, "data_sha256"), (split_path, "split_sha256")]:
        if sha256(path) != reference[key]:
            raise ValueError(f"Unexpected input version: {path}")

    metadata = EMB_DIR / "m5_transformer_manifest.json"
    files = {name: EMB_DIR / name for name in reference["files"]}
    required = {f"{name}{suffix}.npy" for name in representations for suffix in ("", "_ids")}
    if not required or required - files.keys():
        raise ValueError("Unknown SBERT representation")
    if "tokens_multi" in representations or "tokens_mask" in representations:
        required.update({"tokens_multi.npy", "tokens_multi_ids.npy", "tokens_mask.npy", "tokens_mask_ids.npy"})
    missing = [name for name, path in files.items() if not path.is_file()]
    if required.intersection(missing):
        if not generate_if_missing:
            return None
        if len(missing) != len(files) or metadata.exists():
            raise ValueError(f"Incomplete SBERT cache in {EMB_DIR}. Restore the complete cache or use a fresh DATA_ROOT.")
        subprocess.run(
            [sys.executable, str(REFERENCE.with_name("mvp.py")), "encode",
             "--revision", reference["revision"], "--device", "cpu"],
            check=True,
        )

    manifest = json.loads(metadata.read_text()) if metadata.is_file() else reference
    for key in ("model", "revision", "max_len", "data_sha256", "split_sha256", "frozen",
                "sentence_pooling", "sequence_policy", "precision"):
        if manifest.get(key) != reference[key]:
            raise ValueError(f"Unexpected SBERT cache metadata: {key}")
    if set(manifest["files"]) != set(files):
        raise ValueError("Unexpected SBERT cache file list")

    for name in sorted(required):
        path = files[name]
        if sha256(path) != manifest["files"][name]:
            raise ValueError(f"SBERT cache checksum mismatch: {path}")

    clean = load_clean()
    load_splits(clean=clean)
    arrays = {
        name: np.load(files[name], mmap_mode="r", allow_pickle=False)
        for name in required
    }
    row_count = len(clean)
    expected_shapes = {
        "sbert_multi": ((row_count, 768), np.float32),
        "tokens_multi": ((row_count, reference["max_len"], 768), np.float16),
        "tokens_mask": ((row_count, reference["max_len"]), np.bool_),
    }
    expected_ids = set(clean["id"])
    expected_shapes = {name: spec for name, spec in expected_shapes.items() if f"{name}.npy" in required}
    for name, (shape, dtype) in expected_shapes.items():
        matrix = arrays[f"{name}.npy"]
        ids = arrays[f"{name}_ids.npy"]
        if matrix.shape != shape or matrix.dtype != dtype:
            raise ValueError(f"Unexpected shape or type for {name}: {matrix.shape}, {matrix.dtype}")
        if ids.shape != (row_count,) or not np.issubdtype(ids.dtype, np.integer) or set(ids) != expected_ids:
            raise ValueError(f"IDs in {name} do not match clean_v2")
    if "tokens_multi" in expected_shapes and not np.array_equal(arrays["tokens_multi_ids.npy"], arrays["tokens_mask_ids.npy"]):
        raise ValueError("Token and mask IDs are not aligned")

    return expected_shapes
