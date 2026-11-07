# Member 5 MVP — YU Guanlin

This contribution extends main commit `0498603aa095f27af50b0e08c9ebec6ffdd1d0cc` on branch `m5-textcnn-mvp-20261107`. Shared code and other members' deliverables are unchanged.

## What is delivered

- `mvp.py`: frozen multilingual SBERT extraction and independent TextCNN experiments with Transformer or fold-trained FastText token inputs.
- `Member5_MVP.ipynb`: entry point, actual result review, and optional full reproduction.
- `evidence/`: cache manifests, run configurations, fold metrics, OOF predictions, training/early-stopping audits and confusion matrices.
- `report_draft.md`: English methodology and actual first results.
- `verify_artifacts.py`: independent metric reconciliation, fold/group isolation, cache integrity and Safetensors reload checks.
- `test_mvp.py` and `test_results.txt`: short-sequence/padding invariance and malformed-probability checks.
- `requirements.txt`: direct dependencies for Python 3.12; `requirements-macos-lock.txt` records the complete locally tested environment. Use direct requirements on Linux/Colab; the Mac lock includes platform-specific packages.

## Run

Clone the private repository using your own authorized GitHub access, select this branch, and run from the repository root. Never paste access tokens into notebook source, clone URLs, or saved output.

```bash
python -m pip install -r Member5_file/requirements.txt
python -m pytest Member5_file/test_mvp.py -q
python Member5_file/mvp.py encode --device cpu
python Member5_file/mvp.py train --embedding fasttext --device cpu --epochs 12
python Member5_file/mvp.py train --embedding transformer --device cpu --epochs 12
```

For Colab, select a Python 3.12-compatible runtime before installing these dependencies. Clone the repository first, install requirements, restart the kernel if needed, and change into the clone. Use `--device cuda` when a GPU is available. This package was measured on a local CPU; a live Colab session has not been verified. In the notebook, set `RUN_EXPERIMENTS=True` to retrain; default mode displays committed evidence without requiring downloaded caches. Set `CHECK_LOCAL_CACHES=True` to validate local/shared artifacts.

## Shared artifact contract

The default data directory is the clone's `data/`. If using a different `DATA_ROOT`, copy the versioned `raw/` and `clean_v2/` input folders there first, and set the environment variable before importing `src`. The official split remains `splits/folds_v1.csv` in the clone.

Frozen outputs:

- `emb/sbert_multi.npy` and `emb/sbert_multi_ids.npy`: L2-normalized sentence vectors, all 1,525 IDs, 768 dimensions.
- `emb/tokens_multi.npy` and matching IDs: float16 compact real-token sequences with zero right padding.
- `emb/tokens_mask.npy` and matching IDs: boolean masks aligned to sequences.
- `emb/m5_transformer_manifest.json`: encoder revision, lengths, source hashes, cache checksums and pooling policy.

Load through `src.cache.load_embedding(name, ids=requested_ids)`, including masks through `load_embedding('tokens_mask', ids=...)`. Sentence pooling includes attended special tokens, matching the pretrained model's mean-pooling convention; sequence inputs exclude special tokens. Maximum token length is selected on development text only and capped at 128.

FastText outputs follow the existing `fasttext_tok_fold{k}`, corresponding `_mask`, and `_full` conventions. Each fold's FastText is fitted without that outer validation fold. The script regenerates these artifacts in its local DATA_ROOT rather than trusting an unversioned external cache; use an isolated DATA_ROOT when preserving another experiment's cache.

Predictions are `preds/<run_id>.npz`, with OOF IDs and 303 test IDs, probability column order `[0 Neutral, 1 Positive, 2 Negative]`. Test labels are not scored. Final model weights, when saved, are `models/<run_id>.safetensors`. Large caches and weights are intentionally excluded from Git. `local_artifacts.zip` is a local handoff archive, also excluded from Git; unpack its contents under DATA_ROOT.

## Evaluation interpretation

Five official outer grouped folds; one grouped inner split per outer fold selects epochs with patience 3 and a 12-epoch ceiling. The classifier is reinitialized and refitted on all outer training rows before producing outer validation probabilities. The full refit uses the median selected epoch count. AdamW, learning rate 0.001, weight decay 0.01, batch size 32, kernels 2/3/4/5 with 100 filters each, dropout 0.5. Nonempty inputs shorter than a kernel use a single zero-extended window; empty inputs have zero pooled features.

These are first experiments with a fixed architecture and one seed, not an exhaustive tuning study. Choosing a representation after seeing these CV results can introduce selection optimism. The supplied labels may reflect a lexicon labeller; model accuracy is not evidence of trading profitability. The team's test partition had already been inspected in another member's experiments, so it cannot be called an untouched team holdout.

`train_seconds` measures the classifier workflow after embedding preparation: inner selection, outer refits and inference, final refit, test inference and weight serialization. It is not directly comparable with the baseline's fit-only timing. Encoder extraction time and FastText preparation time are recorded separately in their manifests. Model download time is excluded from encoder extraction time.

## Team integration

Members 1–4 can consume `sbert_multi`; member 6 can consume `tokens_multi` and `tokens_mask`. Compare against the official pooled OOF baseline, not provisional fold means or results from different runs. Keep this module and its notebook together. For the final single-notebook submission, member 4 must inline the required functions and shared modules, or explicitly assemble their source cells. This contribution is not the team's final combined notebook.
