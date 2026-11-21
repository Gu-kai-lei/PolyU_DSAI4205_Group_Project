# Member 3 experiment log

| Stage | Evidence |
|---|---|
| EDA handoff | `00_eda.ipynb`, six `eda_*.png`, source tables and English report |
| 21 Nov, input alignment | `clean_v2` and `folds_v1` SHA256 match the Member 5 manifest. Frozen SBERT at `4328cf26390c98c5e3c738b4460a05b95f4911f5` regenerated from float32 ONNX. 1,525 rows, 768 dimensions, aligned by ID. |
| Nested exploration | Three predefined settings, selected by inner grouped OOF Macro-F1. PCA and early stopping inside inner partitions. Outer folds use the official split. `m3_inner_search.csv` contains all candidates and chosen rounds. |
| Feature comparison | Embedding-only pooled OOF Macro-F1 0.5275. Combined pooled OOF Macro-F1 0.5789. Official baseline 0.5820. Outer fold evidence is in `m3_fold_metrics.csv`. |
| Team prediction handoff | Combined run `xgboost__sbert_multi__20261121-143943-a632`, 1,222 development OOF and 303 test probability rows in label order 0/1/2. Test labels reserved for project-level evaluation. |
| Interpretation | Exact TreeSHAP margins checked against XGBoost outputs on every held-out fold. Named-feature ranking, class metrics, confusion matrix and business implications in this folder. |
