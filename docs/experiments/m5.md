# Member 5 experiment summary

| Run | Development Macro-F1 | Interpretation |
|---|---:|---|
| Official TF-IDF + logistic regression | 0.5820 | Baseline on the fixed team folds |
| FastText + TextCNN MVP | 0.5361 | Earlier single-seed protocol |
| Frozen SBERT + TextCNN MVP | 0.5796 | Earlier single-seed protocol |
| C3 v1 study, matched C0 | 0.6070 | Mean across three seeded OOF runs |
| C3 v1 study, selected pipeline | **0.6167** | Mean across three seeded OOF runs, adopted workflow |
| Later six-configuration local study | 0.6059 | Below its matched C0 at 0.6155, archived separately |

The adopted workflow used five outer grouped folds, three inner grouped folds, candidates C0/C1/C3/C7 and seeds 42/17/2026. The improvement over its matched C0 averaged 0.0097. All three paired group-bootstrap intervals included zero, and the estimated gain fell short of the 0.010 target.

The fitted final model is C3 with 689,155 trainable parameters and nine epochs at seed 42. The team results CSV uses the seed-42 pipeline OOF value of 0.617926. The three-seed mean estimates the adaptive workflow. There are 303 reserved predictions in the handoff archive, with no scoring against their labels in this experiment. The original audit logs and weights are retained by Member 5.
