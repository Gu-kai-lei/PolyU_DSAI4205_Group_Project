# Member 3 error analysis against the official baseline

We compared both models on the same 1,222 development tweets in `folds_v1`. We reproduced the official TF-IDF and logistic regression baseline from `text_norm` within each training fold. Its pooled out-of-fold scores matched `results/baseline.csv`. We then aligned the reproduced predictions with the submitted SBERT plus XGBoost predictions by tweet ID. The comparison uses only development labels.

| Given label | Tweets | Baseline correct | XGBoost correct | Baseline F1 | XGBoost F1 |
|---|---:|---:|---:|---:|---:|
| Neutral | 348 | 168 | 207 | 0.5563 | 0.6026 |
| Positive | 486 | 357 | 317 | 0.6392 | 0.6096 |
| Negative | 388 | 199 | 188 | 0.5505 | 0.5244 |

XGBoost gained 39 correct Neutral predictions, but lost 40 correct Positive predictions and 11 correct Negative predictions. It labelled 67 Positive tweets and 65 Negative tweets as Neutral, compared with 44 in each group for the baseline. It also confused more Positive tweets with Negative (102 versus 85), while confusing fewer Negative tweets with Positive (135 versus 145). The gains in Neutral F1 (+4.63 percentage points) were offset by the drops in Positive (−2.96) and Negative (−2.61). With the three classes weighted equally, pooled Macro-F1 was 0.5789 versus 0.5820.

The models got 511 tweets right together and 297 wrong together. The baseline alone got 213 right, while XGBoost alone got 201 right. This shows that they make different errors even though their overall scores are close. A paired bootstrap resampled tweet groups within each fold 5,000 times and gave a descriptive 95% interval of **−3.56 to +3.08 percentage points** for XGBoost minus baseline Macro-F1. It includes zero. The interval holds the trained models and their out-of-fold predictions fixed, so it does not capture the variation from retraining or quantify performance on new data.

Character count, word count and hashtag ratio were the strongest named SHAP features for XGBoost and had their largest average contributions for Neutral. That fits the EDA observation that Neutral tweets were shorter, and is consistent with increased Neutral recall (48.3% to 59.5%) alongside lower Neutral precision (65.6% to 61.1%). SHAP alone cannot establish that length caused the mistakes on Positive or Negative tweets. Another plausible explanation is that unigram TF-IDF preserves exact word cues associated with the supplied labels, while the SBERT representation pools the text and PCA compresses the result. The comparison changes representation, input features and classifier together, so it does not isolate that explanation.

We also checked a simple keyword slice based on the terms `red`, `green`, `bearish`, `bullish`, `fed`, `nifty`, `put`, `crash` and `dip` in `text_norm`. XGBoost was correct on 175 of 285 matching tweets and the baseline on 156. On the 937 other tweets, the counts were 537 and 568, respectively. This exploratory accuracy comparison offers no evidence that these finance keywords explain XGBoost’s overall deficit. The keyword list does not capture all financial language.

The business implication is that XGBoost identifies more truly Neutral items, while missing more labelled Positive and Negative items overall. In a review queue where missed directional sentiment matters, report the class-level recall alongside Macro-F1 and examine messages moved into Neutral before applying a review threshold.

- [Comparison figure](m3_baseline_error_comparison.png)
- [Class metrics](m3_baseline_class_comparison.csv)
- [Paired counts and descriptive slices](m3_baseline_paired_errors.csv)
- Reproduce from the repository root with `python Member3_file/m3_error_analysis.py --pred <path-to-combined-run-npz>` in the team environment. The Member 3 probability cache is available to the team through the existing handoff.
