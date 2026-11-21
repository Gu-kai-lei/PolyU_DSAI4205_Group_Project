# Report assembly notes

Use the score table in [the technical submission](../submission/DataVinci.ipynb) for the main Word results table. The [report source](DataVinci_Final_Report.md) follows the same selection of runs. Keep these notes near the draft while merging the individual sections.

| Member | What to carry into the report |
|---|---|
| FENG Dingran | Fixed TF-IDF + LR reaches 0.5820 Macro-F1. Nested tuning raises it to 0.6135. Report the nested SBERT SVM rerun at 0.5922. Label the 0.6639 ensemble as exploratory equal-weight fusion. |
| GU Kailei | SBERT kNN reaches 0.5395 on the folds used for parameter selection and about 0.522 in the nested check. Keep the six original Table 8 cases and the four archived prediction sets used to label them. The VADER checks use different denominators. |
| YANG Qitian | Add the paired XGBoost analysis against the fixed baseline. Use the published XGBoost run for the paired counts and the rerun for the comparison table. Both round to 0.5789 Macro-F1. |
| CAO Taihe | Use 0.5387 for the FastText MLP rerun in the main table. Keep the archived 0.5345 run tied to its predictions in the four-model error analysis. |
| YU Guanlin | Report 0.6179 for the archived seed-42 TextCNN selection workflow. The new fixed C3 training verified inference and checkpoint reload. Its reserve predictions were generated without scoring the labels. |
| CHOI CHUN MAN | Use 0.5211 for the corrected BiLSTM. In the imbalance comparison, label the archived and regenerated-cache results separately. Check Sections 4.2 and 5.5 against the group's final decision on how to present the historical reserve scores. |

The technical submission leaves the 303 reserve labels unscored. Earlier BiLSTM and imbalance experiments reported reserve scores. An independent estimate for the selected ensemble is still unavailable.

Sources: [kNN and report sections](../docs/report_drafts/m2_final_sections.md), [XGBoost paired errors](../experiments/Member3_file/m3_baseline_error_analysis.md), [TextCNN experiments](../docs/experiments/m5.md), [corrected BiLSTM score](../results/m6_bilstm_grouped.csv), [original imbalance ledger](../results/imbalance.csv).
