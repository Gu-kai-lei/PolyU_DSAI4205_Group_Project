# Member 3 final report sections

## 2.1 Data quality and descriptive statistics

One record represents one tweet. We checked the raw text and labels, matched retained IDs to the original rows, and reconciled exclusions with the cleaning log. Table 3 summarises the checks. The raw file contained no missing or blank text, no missing labels and no invalid label codes. The retained set contains 1,513 related-text groups. Twenty records belong to eight groups with multiple members, which the shared split keeps together.

| Measure | Raw | Retained `clean_v2` |
|---|---:|---:|
| Tweets | 1,698 | 1,525 |
| Blank or missing text, or missing labels | 0 | 0 |
| Invalid sentiment codes | 0 | 0 |
| Extra copies with identical original text | 9 | 0 |
| Median text length in characters | 139 | 135 |

The character medians describe different text versions and different populations. Member 2’s cleaning excluded 173 records, including 102 non-English tweets, 39 without lexical content, 21 with unreadable encoding and 11 duplicate copies. Retained-text statistics apply to this cleaned sample. Member 2’s label-origin paragraph can follow this section.

## 2.2 Class distribution

In the raw sample, 575 tweets are Neutral, 627 Positive and 496 Negative. After cleaning, the counts are 433 Neutral (28.4%), 607 Positive (39.8%) and 485 Negative (31.8%). Neutral falls from 33.9% to 28.4%, as many excluded non-English and wordless examples carried a Neutral label. The largest retained class is 1.40 times the smallest class. We report Macro-F1 and class-level recall alongside accuracy to make performance on each class visible.

Insert Figure 1 from [`eda_01_class_distribution.png`](../../experiments/Member3_file/eda_01_class_distribution.png).

## 2.3 Text characteristics

Neutral has a median of 86 characters and 13 whitespace tokens in `text_clean`. Positive has medians of 156 characters and 25 tokens, while Negative has 161 characters and 26 tokens. The distributions overlap. In the shortest bin (at most 80 characters), 191 of 353 tweets (54.1%) are Neutral. In the longest bin (more than 220), 13 of 325 (4.0%) are Neutral. This makes text length a plausible candidate feature, with its value tested in Section 3.4.

The collection tags dominate document frequency. `#stockmarketcrash` appears in 920 retained tweets (60.3%) and `#bearmarket` in 599 (39.3%). For vocabulary, we rank normalized words by the share of tweets in each class containing them, counting each term once per tweet and excluding the collection terms and technical remnants for display. Common terms such as *market*, *stock* and *not* occur across classes, so a single word provides limited context. The tagged crypto, US equity and India subsets show similar Positive shares (40.3%, 37.8% and 40.3%). Forty-five tweets appear in more than one subset and 774 have none of the three market flags. Each share describes its own set of mentions.

Insert Figure 2 from [`eda_02_text_lengths.png`](../../experiments/Member3_file/eda_02_text_lengths.png). Suggested appendix figures are [`hashtags`](../../experiments/Member3_file/eda_03_hashtags.png), [`vocabulary`](../../experiments/Member3_file/eda_04_vocabulary.png), [`market subsets`](../../experiments/Member3_file/eda_05_market_sentiment.png) and [`feature bins`](../../experiments/Member3_file/eda_06_feature_relationships.png). Underlying CSVs are in the same folder.

## 3.4 XGBoost

We use frozen multilingual Sentence-BERT sentence vectors from `text_clean`, at Member 5’s pinned revision. We recomputed them with the published float32 ONNX export using a maximum of 82 tokens, attention-masked mean pooling and L2 normalization. Within each training partition, PCA reduces the 768 dimensions to 64 or 96. The combined model appends 13 measured text descriptors from `clean_v2` and the `emoji_lost` flag. The descriptors cover length, punctuation, capitalization, hashtag use and market mentions. The XGBoost classifier returns probabilities for Neutral, Positive and Negative.

On the shared `folds_v1` split (1,222 development and 303 test tweets), we compared three configurations using three stratified group inner folds for each of the five outer folds. We varied tree depth (2, 3 or 4), learning rate (0.08, 0.05 or 0.04), subsampling (0.85 or 0.75), L2 regularization (4 or 8) and PCA width (64 or 96). Each inner fit used multiclass log-loss early stopping with patience 12 and a 160-round cap. We ranked configurations by inner pooled Macro-F1 and used the median chosen round count to refit on the outer training fold. The combined model’s full-development refit used 64 PCs, depth 2, learning rate 0.08, subsample 0.85, L2 4 and 118 rounds. Test probabilities were exported for the team’s integration without calculating test scores here.

| Candidate | OOF Accuracy | Macro-Precision | Macro-Recall | Macro-F1 | Neutral F1 | Positive F1 | Negative F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| TF-IDF + logistic regression | 0.5925 | 0.6053 | 0.5767 | 0.5820 | 0.5563 | 0.6392 | 0.5505 |
| XGBoost, SBERT PCA alone | 0.5426 | 0.5399 | 0.5265 | 0.5275 | 0.4551 | 0.6084 | 0.5189 |
| XGBoost, SBERT PCA + descriptors | 0.5827 | 0.5848 | 0.5772 | 0.5789 | 0.6026 | 0.6096 | 0.5244 |

Descriptors improved XGBoost by 0.0514 Macro-F1 overall and 0.1475 F1 for Neutral under the paired outer-fold comparison. The combined model stayed 0.0031 below the baseline in pooled Macro-F1. Its Neutral F1 was higher, while Positive and Negative F1 were lower. A possible explanation is that TF-IDF preserves individual words more directly when labels may reflect lexicon rules. This comparison does not isolate the mechanism behind the difference. The CPU time for the combined model’s nested search and outer fits was 22.4 seconds, excluding vector generation, SHAP and the final refit.

## 5.3 Business impact and interpretation

The combined model’s development recall was 59.5% for Neutral, 65.2% for Positive and 48.5% for Negative. The lower Negative recall is consequential for a triage dashboard. An analyst who checks only items flagged Negative could miss approximately half of the tweets with the dataset’s Negative label under this validation protocol. Conversely, a false alarm can send an ordinary market update into the review queue. Analysts should inspect uncertain cases and examples that contain market-specific vocabulary or little text. Coverage and review thresholds should be agreed with users before deployment.

On outer-held-out tweets, exact TreeSHAP identifies character length, whitespace word count and hashtag ratio as the strongest named descriptors by mean absolute contribution to class margins. Their largest contributions are for Neutral. This matches the length pattern in EDA and points to a monitoring question: a change in tweet format or hashtag usage can affect the alert mix. Each PCA basis is fitted within a fold, so the figure ranks named descriptors and does not align individual PCA axes across folds. SHAP values describe how the fitted model uses inputs, while the ablation measures the combined effect of adding descriptors under this experiment.

Member 2’s four-model error analysis found 152 shared mistakes out of 1,222 development tweets, before including XGBoost. Finance terms occurred in 39.5% of these cases and 26.4% of the other cases. The case study for tweet ID 806 describes a sharp market fall that carries a Positive label, plausibly because a generic lexicon reads *Nifty* as a positive adjective. An analyst-facing tool should show the original cleaned text and model confidence alongside aggregates, and sample reviewed cases for domain-aware relabelling. The supplied labels and topic-filtered sample support a discussion-monitoring prototype. Validation against financial interpretations or returns would require separate evidence.

Insert [`m3_shap_global.png`](../../figures/m3_shap_global.png) in Section 5.3 and use [`m3_oof_confusion.png`](../../figures/m3_oof_confusion.png) as an appendix figure if space allows. For the four-model error analysis and case 806, see [`../docs/report_drafts/m2_final_sections.md`](m2_final_sections.md), Section 4.3.

Suggested references: Chen and Guestrin (2016), *XGBoost: A Scalable Tree Boosting System*, doi:10.1145/2939672.2939785. Lundberg and Lee (2017), *A Unified Approach to Interpreting Model Predictions*, NeurIPS 30. Reimers and Gurevych (2019), *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*, doi:10.18653/v1/D19-1410.
