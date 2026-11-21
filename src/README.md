# Shared code

- `config.py`, `data.py`, and `splitting.py` load and check the frozen experiment inputs.
- `cache.py` aligns embeddings and predictions by tweet ID. `evaluate.py` computes the shared metrics.
- `fasttext_emb.py` fits the fold-specific embeddings, and `training.py` runs the PyTorch training loop.
- `member1.py`, `member4.py`, and `member6.py` contain the SVM, MLP, and BiLSTM paths.
- `project_review.py` reproduces the baseline and checks the evidence used in the final comparison.

The [submission](../submission/DataVinci.ipynb) includes the code and data it needs.
