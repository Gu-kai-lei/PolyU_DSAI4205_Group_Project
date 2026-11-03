# Member 1 Baseline methodology draft

Status (27 Oct): methodology draft only; official results were pending the shared data split and team environment.

**Status (2 Nov): official run completed.** Member 2 ran `03_baseline` in `MODE = "team"` on `folds_v1` in the `dsai4205` environment for the interim report; the results are in the table below and in `results/baseline.csv`.

## Section 3.4 Baseline

The baseline combines TF-IDF with Logistic Regression for three-class sentiment classification: Neutral (0), Positive (1), and Negative (2). It uses `text_norm` from clean_v2, retaining the preprocessing decisions made by the data-cleaning pipeline. TF-IDF uses unigram features with `min_df=1`, default smoothed inverse document frequency and L2 normalization. Logistic Regression uses L2 regularization, `C=1.0`, the `lbfgs` solver, `max_iter=2000`, and no class weighting. The random seed is 42. These parameters define the initial baseline before refinement.

The pipeline fits both TF-IDF and Logistic Regression separately within each Training Fold. Neither the vocabulary nor the inverse-document-frequency weights are learned from the Validation Fold. The shared split file is checked against the clean data by ID and group ID before training. Related tweets must remain in the same partition and fold.

Performance is measured using Accuracy, Macro-Precision, Macro-Recall and Macro-F1. The team comparison record uses pooled Out-of-Fold Predictions, while the mean and sample standard deviation across the five Validation Folds are reported separately. The Confusion Matrix and per-class metrics also use Out-of-Fold Predictions. Macro-F1 is the primary Model Selection metric. The Test Set is not used during baseline development or Hyperparameter Tuning.

For interpretation, the pipeline is refitted on the full Development Set, and the highest positive Logistic Regression Coefficients are displayed for each class. This refit is used only for interpretation, not to calculate the reported Cross-Validation metrics. Coefficients indicate model associations rather than word frequency or causality.

## Results of the official run (2 Nov)

| Item | Status |
|---|---|
| Official run_id and Data Split version | `baseline__tfidf__20261102-181839-3628`, `folds_v1` |
| Pooled OOF Accuracy / Macro-Precision / Macro-Recall / Macro-F1 | 0.592 / 0.605 / 0.577 / 0.582 |
| Five-fold mean and standard deviation | Accuracy 0.593 ± 0.042; Macro-F1 0.582 ± 0.042 (folds 0.512–0.621) |
| Per-class metrics and Confusion Matrix | F1 Neutral 0.556, Positive 0.639, Negative 0.550; `figures/baseline_baseline__tfidf__20261102-181839-3628_confusion.png` |
| Interpretation of Top Words and typical errors | All 12 top Positive and 11 of 12 top Negative words are VADER lexicon entries; 129 Neutral and 145 Negative tweets predicted as Positive (interim report, Section 4.2) |

Do not copy provisional Results into the official table. The notebook and script preserve run_id, parameters, environment information, and data/split hashes for traceability.

## Planned Refinement

After the initial official run, compare a predefined set of `ngram_range`, `min_df`, `C`, and `class_weight` settings on the shared Development Folds. Save every candidate and identify selection scores as such: the best score from a tuning search is not an unbiased estimate of generalization. Do not inspect Test Set performance to select settings. For Stacking, tuning within outer Training Folds is needed if tuned base-model OOF probabilities are to remain independent of the outer Validation Labels. This refinement and ensemble stage has not yet been implemented or executed.
