# Shared PyTorch training interface

Author: Cao Taihe, 24101734d. For Members 5 and 6.

`src.training` accepts frozen vector features `(N,D)` or token features `(N,L,D)`.
Sequence masks have shape `(N,L)` and dtype bool: **True = real token**.
Implement `forward(x, mask=None)` returning `(N,3)` raw logits. Never return
softmax from the model: training uses CrossEntropyLoss; prediction applies softmax.

```python
from src.training import seed_everything, TrainConfig, fit_model, predict_proba
seed_everything(42)  # BEFORE constructing your model
model = YourModel(...)
result = fit_model(model, x_fit, y_fit, x_stop, y_stop,
                   train_mask=mask_fit, valid_mask=mask_stop,
                   config=TrainConfig(epochs=50, patience=7, device="cpu"),
                   checkpoint="local_artifacts/best_model.pt")
probabilities = predict_proba(model, x_valid, mask_valid)
```

Use a grouped **inner split of outer training** for x_stop. Then reinitialize
and refit on the complete outer training data for `result['best_epoch']` epochs,
calling fit_model without validation data. This keeps the outer fold independent
of epoch selection. Full development refit uses the median selected fold epoch
count. Fit all transforms and class weights on applicable training data only.
Never use holdout labels for early stopping, selection, thresholds or ensembles.

TextCNN: compute a valid-window mask for each convolution width with
`mask.unfold(1, kernel, 1).all(-1)`. Exclude invalid windows before max pooling,
and replace all-invalid pooled outputs by zero. Simply zeroing padded tokens
is insufficient because convolution bias can still win the max operation.
Pad short examples to the largest kernel width before convolution.

BiLSTM: use `pack_padded_sequence` with lengths from `mask.sum(1)`,
`batch_first=True`, `enforce_sorted=False`. Clamp zero lengths to one for packing,
then explicitly return zero representations for originally empty sequences.
Mask padding before attention normalization and handle empty sequences to avoid
softmax over only negative infinity. `masked_mean` and `masked_max` provide safe
pooling examples and are covered by regression checks.

`fit_model` returns history, best epoch, best inner Macro-F1 and fit wall time,
restores best validation weights, supports training-only class weights, AdamW,
gradient clipping and optional checkpoints. The final singleton training batch
is merged with the previous batch to avoid BatchNorm failure.

Collect one OOF row per development id. After development-only selection,
use `save_predictions(run_id, oof_ids, oof_proba, test_ids, test_proba)` or the
team `src.cache.save_preds`. Columns must map to `[0,1,2]`, with probabilities
finite, nonnegative and summing to one. An empty test array means deferred
predictions; never substitute zero probabilities for missing predictions.

CPU results were reproduced exactly in a fresh kernel with two PyTorch threads.
Seeds do not guarantee identical results across different hardware or package
versions. Keep the tested environment record with each experiment.
