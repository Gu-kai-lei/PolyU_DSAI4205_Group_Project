# Interim report outline

The interim report (20% of the project, due **6 Nov 23:55**) documents the work up to week 9: a summary of the EDA, the cleaning steps and the baseline results, to discuss at the consultation. Member 2 assembles it. Each owner sends their part, in English, to member 2 by **4 Nov**; everyone checks the assembled draft on 5 Nov.

Section numbers follow the final report template, so the same text can be reused in the final report.

**Status (2 Nov):** the assembled report is `reports/DSAI4205_DataVinci_Interim_Report.docx`. Members 3 and 4 should review the EDA and data-split sections, which member 2 completed for them.

| Part | Owner | Content | Source | Status (27 Oct) |
|---|---|---|---|---|
| 1 Dataset and task | 2 | Decision problem in two sentences; Table 1 overview and Table 2 data dictionary | `m2_dataset.md` | In the report |
| 2 EDA | 3 | Data quality table (Table 3), six figures each answering one question, class imbalance | `00_eda` | In the report (done by member 2; member 3 to review) |
| 2.1 Label origin | 2 | VADER reproduces 93.0% of the labels; consequences for the project | `m2_cleaning.md` | In the report |
| 3.1 Data cleaning | 2 | Table 4 of cleaning steps, before/after examples, class balance after cleaning | `m2_cleaning.md` | In the report |
| 3.2 Data split | 4 | Grouped and stratified split, Table 5, how leakage is prevented | `02_splits` | In the report (written by member 2; member 4 to review) |
| 3.4 Baseline | 1 | TF-IDF + logistic regression settings; Accuracy, Precision, Recall, F1, confusion matrix, top words | `m1_baseline.md`, `03_baseline` | Official run done (macro-F1 0.582); in the report |
| Next steps and questions | 2 | Plan to 22 Nov and the questions below | this file | In the report |

## Next steps (draft)

Six embedding and classifier pairs will be compared with the baseline on the same five folds: SVM, kNN, XGBoost and an MLP on sentence vectors, and TextCNN and a BiLSTM with attention on token sequences. The embeddings are a frozen Sentence-BERT model and our own FastText trained per fold. After tuning and imbalance handling, the models are combined by stacking, and the test split is used once, after the experiments are frozen on 18 Nov.

## Questions for the consultation (draft)

1. The labels match VADER on 93% of the tweets, so they were most likely generated automatically. May we report results against these labels and discuss label noise, or should we also relabel a sample by hand?
2. We use Sentence-BERT only to extract vectors, without fine-tuning, and train separate classifiers on them. Does this meet the requirement that the embedding and the classifier are separate models?
3. Three of our six models (MLP, TextCNN, BiLSTM) are neural networks. Is that acceptable, or does it count as too little variety between models?
4. We removed 102 non-English tweets, whose labels are mostly Neutral because VADER cannot read them. Is removing them acceptable?
