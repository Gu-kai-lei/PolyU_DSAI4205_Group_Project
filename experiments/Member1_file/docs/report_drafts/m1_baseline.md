# Member 1 Baseline methodology draft

Status: Methodology draft only. Official Results are pending the shared Data Splits and team Environment. The following describes the implemented procedure; it does not claim an official experiment has been completed.

## Section 3.4 Baseline

The baseline combines TF-IDF with Logistic Regression for three-class sentiment classification: Neutral (0), Positive (1), and Negative (2). It uses `text_norm` from clean_v2, retaining the preprocessing decisions made by the data-cleaning pipeline. TF-IDF uses unigram features with `min_df=1`, default smoothed inverse document frequency and L2 normalization. Logistic Regression uses L2 regularization, `C=1.0`, the `lbfgs` solver, `max_iter=2000`, and no class weighting. The random seed is 42. These parameters define the initial baseline before refinement.

The pipeline fits both TF-IDF and Logistic Regression separately within each Training Fold. Neither the vocabulary nor the inverse-document-frequency weights are learned from the Validation Fold. The shared split file is checked against the clean data by ID and group ID before training. Related tweets must remain in the same partition and fold.

Performance is measured using Accuracy, Macro-Precision, Macro-Recall and Macro-F1. The team comparison record uses pooled Out-of-Fold Predictions, while the mean and sample standard deviation across the five Validation Folds are reported separately. The Confusion Matrix and per-class metrics also use Out-of-Fold Predictions. Macro-F1 is the primary Model Selection metric. The Test Set is not used during baseline development or Hyperparameter Tuning.

For interpretation, the pipeline is refitted on the full Development Set, and the highest positive Logistic Regression Coefficients are displayed for each class. This refit is used only for interpretation, not to calculate the reported Cross-Validation metrics. Coefficients indicate model associations rather than word frequency or causality.

## Results to complete after official execution

| Item | Status |
|---|---|
| Official run_id and Data Split version | Pending |
| Pooled OOF Accuracy / Macro-Precision / Macro-Recall / Macro-F1 | Pending |
| Five-fold mean and standard deviation | Pending |
| Per-class metrics and Confusion Matrix | Pending |
| Interpretation of Top Words and typical errors | Pending |

Do not copy provisional Results into the official table. The notebook and script preserve run_id, parameters, environment information, and data/split hashes for traceability.

## Planned Refinement

After the initial official run, compare a predefined set of `ngram_range`, `min_df`, `C`, and `class_weight` settings on the shared Development Folds. Save every candidate and identify selection scores as such: the best score from a tuning search is not an unbiased estimate of generalization. Do not inspect Test Set performance to select settings. For Stacking, tuning within outer Training Folds is needed if tuned base-model OOF probabilities are to remain independent of the outer Validation Labels. This refinement and ensemble stage has not yet been implemented or executed.
