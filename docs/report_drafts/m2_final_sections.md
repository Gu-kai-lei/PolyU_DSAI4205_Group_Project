# 2 号负责的期末报告章节（11/20 定稿草稿）

给 6 号统一格式用。章节号按课程报告模板；模板里有编号的表（表 1、2、4、8）沿用模板编号，其他表和图写成 `[n]`，由 6 号统一编号。数字来自 `clean_v2`、`folds_v1`、`results/m2_knn.csv`、`notebooks/04_analysis/31_label_noise_errors.ipynb`（11/18 运行）。

要用到的图：

| 位置 | 文件 |
|---|---|
| 2.1 标签来源 | `figures/cleaning_label_by_language.png` |
| 3.1 清洗 | `figures/cleaning_funnel.png` |
| 3.4 kNN | `figures/m2_knn_k_curve.png`、`figures/m2_knn_confusion.png` |
| 4.3 错误分析 | `figures/analysis_errors_vader.png` |

Submission check: Section 4.3 uses the specified four development prediction sets. Confirm the reserve wording with the group before merging the final Word file.

---

## 1.3 Dataset description

We use Dataset 3 of the course, "Huge crash in the Stock market". According to its description, the tweets were collected from Twitter under the hashtag #stockmarketcrash during the 2022 stock market crash, and each tweet carries one of three sentiment labels. The file has 1,698 rows and two columns, the tweet text and its label. Besides #stockmarketcrash (1,009 tweets), 720 tweets also carry #bearmarket.

The file has no timestamp or user fields. A few tweets mention dates in their text, and the content fits 2022: 56 tweets use hashtags that contain "2022" (for example #BearMarket2022), 92 mention Russia or Ukraine, and 26 mention the LUNA/Terra cryptocurrency, which collapsed in May 2022.

The description does not say how the labels were assigned; Section 2.1 tests whether they come from the VADER sentiment lexicon. 123 tweets (7.2%) are in other languages or cannot be read because the file damaged their characters. After cleaning (Section 3.1) the experiments use 1,525 tweets, and all 1,698 rows stay in `clean_v2_all.csv` with the reason for each exclusion.

Table 1  Dataset overview

| Attribute | Description |
|---|---|
| Dataset and version | Dataset 3 "Huge crash in the Stock market", file `stock_market_crash_2022.csv` as provided by the course; cleaned version `clean_v2` |
| Source and time coverage | Tweets collected from Twitter under #stockmarketcrash during the 2022 crash (dataset description); no timestamp field |
| Unit of analysis | One tweet |
| Size and language | 1,698 tweets, 1,525 after cleaning; 123 tweets (7.2%) are in other languages or unreadable |
| Target labels | `text_sentiment`: 0 Neutral (575 tweets), 1 Positive (627), 2 Negative (496); the labelling method is not stated |
| Usage conditions | Provided by the course for this assignment, with no licence stated. The tweets are public posts but contain user names, which we replace with `@user` |

Table 2  Data dictionary

| Field | Type | Description |
|---|---|---|
| `text` (renamed `text_raw`) | Text | Original tweet, unchanged; used for EDA and error analysis |
| `text_sentiment` (renamed `label`) | Integer 0, 1, 2 | Target: 0 Neutral, 1 Positive, 2 Negative; `label_name` holds the class name |
| `id` | Integer | Row number in the raw file, never renumbered; links the data, the split, the embeddings and the predictions |
| `group_id` | Integer | Duplicate and near-duplicate tweets share one value, so the split keeps them on the same side |
| `text_clean` | Text | Repaired tweet with case and punctuation kept; input for Sentence-BERT |
| `text_norm` | Text | Lower-cased, lemmatised tweet without stopwords, negations kept; input for TF-IDF and FastText |
| `emoji_lost` | Integer 0, 1 | 1 if the source file destroyed characters in the tweet, usually emoji (397 retained tweets) |
| 13 feature columns | Integer or decimal | Length, counts of hashtags, mentions, cashtags, `!` and `?`, upper-case ratio, hashtag ratio, a hashtag-spam flag and market flags (crypto, India, US) |

---

## 2.1 Data quality: where the labels come from (paragraphs for member 3's section)

The dataset description does not say how the tweets were labelled. If a sentiment lexicon produced the labels, our models would learn to imitate that lexicon, and its score could not be used as a feature. We therefore tested VADER (Hutto & Gilbert, 2014) with its recommended ±0.05 threshold on the compound score. The test uses the 1,666 readable tweets: the 1,525 retained tweets plus the 141 non-English and word-less tweets, which belong to the evidence. VADER reproduces 93.0% of the labels (Cohen's κ = 0.89), while TextBlob, a different lexicon, reproduces 60.2% (κ = 0.40). On the cleaned text in this 1,666-tweet check, VADER agrees with 92.1% of the labels. The submission notebook reports 92.06% on the raw text of the 1,222 development tweets, a separate calculation with a different denominator.

Other observations point the same way. VADER scores unknown words as zero, and 72.5% of the 102 non-English tweets are labelled Neutral, against 28.4% of the retained tweets (Figure [n]); of the 28 non-English tweets labelled Positive or Negative, 27 get the same label from VADER, through English words or words that VADER reads as English, such as Spanish *sin*. All 39 tweets without words are Neutral. The source file also lost the apostrophe in some negations (*don__ panic*). Restoring the negation changes VADER's label for 14 retained tweets, and in 13 of them the given label follows the damaged text. Tweets with lost characters agree slightly less often (91.2% against 93.7%), which these observations do not explain.

The labels are therefore highly consistent with VADER-style scoring of the text as distributed, but the original process is unknown, so we treat label origin as a working hypothesis. It leads to four decisions: VADER scores are never a model input; the experiments use tweets with English text; the 112 retained tweets (7.3%) on which VADER and the given label disagree are treated as uncertain labels; and our scores measure agreement with these labels, which may differ from how investors would read the tweets.

---

## 3.1 Data cleaning

Table 4 lists the cleaning steps in the order they run. The cleaning log (`data/clean_v2/clean_v2_log.md`) records each step's rule, the number of rows it changed and a before-and-after example. Excluded rows stay in `clean_v2_all.csv` with a `keep` flag and a `drop_reason`, so every exclusion can be checked or reversed.

Table 4  Data cleaning operations

| Operation | Rationale | Records affected |
|---|---|---|
| Check missing values | Neither column has missing or empty values, so nothing is imputed | 0 |
| Remove exact duplicates | The same text appears more than once with the same label; the first copy is kept | 9 removed |
| Repair encoding and HTML entities | Double-encoded text such as *estÃ©s* is repaired with ftfy; `&amp;` and `&gt;` are unescaped | 155 modified |
| Remove unreadable tweets | Non-Latin scripts whose bytes were lost cannot be repaired or read | 21 removed |
| Replace user mentions with `@user` | User names are personal data and carry no sentiment | 273 modified |
| Restore verified contractions | The file turned an apostrophe and the next letter into `__` (*it__* for *it's*). Only patterns checked by hand against every occurrence are restored | 166 modified |
| Remove lost characters | Remaining `__` runs (emoji, symbols, possessive *'s*) cannot be recovered; they are deleted and the tweet is flagged with `emoji_lost` | 453 modified |
| Set aside non-English tweets | The experiments cover English text, and an English lexicon cannot check these labels (Section 2.1). Detected by script, stopwords of other languages and a language detector; tweets the rules could not judge were checked by hand | 102 set aside |
| Set aside tweets without words | Only hashtags, mentions or cashtags, all labelled Neutral; keeping them could teach a "no words means Neutral" shortcut | 39 set aside |
| Normalise hashtags and whitespace | The `#` is removed but hashtags are kept whole (*StockMarketCrash*); line breaks are merged | all rows |
| Remove content duplicates | Tweets identical after lower-casing and removing mentions, punctuation and lost characters | 2 removed |
| Group related tweets | Near-duplicates (word overlap of 70% or more) and tweets with identical normalised text share a `group_id` and stay in the same fold | 20 tweets in 8 groups |
| Add features | Lengths, counts, ratios and market flags for EDA and the feature-based models | 13 columns |
| Final retained dataset | English, readable, non-duplicate tweets that contain words | 1,525 of 1,698 |

Several steps depend on each other, so the order is fixed. Unreadable tweets are found before lost characters are deleted, because the damaged bytes are the evidence. Mentions are replaced before underscores are interpreted, since many user names contain them. The language is detected before the `#` symbols are removed, because hashtag words such as *stockmarketcrash* confuse the detector. Content duplicates are searched only after the text is repaired.

We keep hashtags whole because of the label analysis. VADER treats a hashtag as one unknown word. On the retained tweets it agrees with the labels 91.9% of the time when hashtags stay whole, 88.1% when they are split at capital letters and 71.1% when *stockmarketcrash* becomes *stock market crash*, since the split adds the word *crash*, which VADER would score as negative.

Table [n]  Examples of cleaned text

| Raw tweet | `text_clean` | `text_norm` |
|---|---|---|
| `I don__ see it. #stockmarketcrash` | `I don't see it. stockmarketcrash` | `not see stockmarketcrash` |
| `It__ finally here! #StockMarketcrash` | `It's finally here! StockMarketcrash` | `finally stockmarketcrash` |
| `#ETH is gone_¦ð_#Ethereum #cryptocurrencies #bearmarket` | `ETH is gone Ethereum cryptocurrencies bearmarket` | `eth go ethereum cryptocurrencies bearmarket` |

No exact or content duplicate had conflicting labels. One near-duplicate pair, the same tweet sent to two different users, is labelled Positive in one copy and Negative in the other; both copies are kept, in the same group. We removed no tweet for its length: a single-hashtag tweet and a 400-character thread are both ordinary tweets.

Most excluded rows were Neutral (142 of 173). The retained data has 607 Positive (39.8%), 485 Negative (31.8%) and 433 Neutral tweets (28.4%), a largest-to-smallest ratio of 1.40. We use stratified, grouped folds and macro-averaged metrics for this mild imbalance, and Section [n] compares class weighting and oversampling.

---

## 3.3 Text representation (cleaning and FastText part)

Cleaning produces two versions of each tweet. `text_clean` keeps case and punctuation (23 words on average), because Sentence-BERT uses them. `text_norm` serves the TF-IDF baseline and FastText: contractions are expanded (*don't* becomes *do not*), the text is lower-cased, only letters are kept, words are lemmatised with WordNet according to their part of speech (*selling* becomes *sell*), and NLTK English stopwords (Bird et al., 2009) are removed except negations and direction words. A standard stopword list would delete *not* (in 307 retained tweets), *down* (102), *up* (84), *more* (81) and *below* (23), and turn "not bullish" into "bullish". A `text_norm` tweet has 14.3 words on average (median 13, maximum 39).

Characters destroyed in the source file are not replaced by a placeholder, because Sentence-BERT would read the placeholder as a word. The `emoji_lost` flag (397 retained tweets, 26.0%) keeps the information for the feature-based models and the error analysis.

FastText (Bojanowski et al., 2017) learns vectors for words and for their character n-grams, so it can build a vector for a rare or misspelled word from its parts. It is our embedding trained from scratch. We train it with gensim on `text_norm`: skip-gram, 100 dimensions, a context window of 5, character n-grams of 3 to 5 letters, words that appear at least twice, 50 epochs, one worker and seed 42, so every run with the same software versions gives the same vectors. We use 200,000 hash buckets instead of gensim's default of two million, which would take about 800 MB per model.

FastText never sees the labels, but it still follows the split. If the validation tweets of fold *k* were in its training text, every model that uses it would get a slightly optimistic score. We therefore train six models: for each fold, one trained only on the other four folds (975 to 980 tweets, 1,386 to 1,420 words in the vocabulary), and one trained on all 1,222 development tweets (1,645 words) to encode the test tweets for the final predictions. Test tweets never enter FastText training.

Each FastText model gives three representations. The mean vector averages the word vectors. The SIF vector (Arora et al., 2017) gives frequent words less weight, a/(a + p(w)) with a = 0.001, and removes the first principal component; word frequencies and the component come from the training part only. The token sequence keeps the word vectors in order, padded or cut to 31 tokens with a mask (8 retained tweets are cut); TextCNN and BiLSTM use it. We do not weight word vectors with TF-IDF, because the brief allows TF-IDF only in the baseline.

The nearest neighbours in the full model show the limits of a small corpus. Some are meaningful (*bull*: *bullrun*, *bullish*, *bullmarket*; *recession*: *depression*), but many only share spelling (*dip*: *sip*, *sit*, *rip*). With about 1,000 training tweets the character n-grams dominate, which limits how well these vectors separate the classes (Section 3.4).

---

## 3.4 Model 2: k-nearest neighbours (paragraph for this section)

kNN gives a tweet the majority label of the *k* development tweets whose vectors are closest to it. It learns nothing beyond storing the vectors, so its score shows how well an embedding alone separates the three classes. The embedding and the classifier are separate models. The embeddings are the frozen Sentence-BERT model `paraphrase-multilingual-mpnet-base-v2` (Reimers & Gurevych, 2019), computed with member 5's script at the same model revision, and our FastText mean and SIF vectors. For fold *k* we use the FastText model trained without fold *k*. All vectors are scaled to length 1 and compared by cosine distance.

We tuned the number of neighbours (odd values from 3 to 51) and the vote weighting (equal votes, or closer neighbours count more) by 5-fold cross-validated macro-F1 on the development set. Table [n] shows the best setting for each embedding. The choice of embedding matters more than *k*: over the whole grid, Sentence-BERT ranges from 0.450 to 0.539, FastText mean vectors from 0.447 to 0.473 and SIF vectors from 0.412 to 0.453 (Figure [n]). Because the setting was chosen on the same folds that score it, 0.539 is optimistic. A nested check, which chooses the setting inside each outer training part, gives 0.522; the chosen *k* stayed between 27 and 43, always with distance weighting. Cross-validation and refitting take 0.25 seconds on a CPU. The test predictions are saved for stacking and are not scored here.

Table [n]  kNN cross-validated results by embedding

| Embedding | Best k | Weighting | Accuracy | Macro-F1 | F1 Neutral | F1 Positive | F1 Negative |
|---|---|---|---|---|---|---|---|
| Sentence-BERT | 41 | distance | 0.547 | 0.539 | 0.543 | 0.589 | 0.486 |
| FastText, mean | 9 | equal | 0.481 | 0.473 | 0.451 | 0.524 | 0.445 |
| FastText, SIF | 21 | distance | 0.468 | 0.453 | 0.390 | 0.540 | 0.429 |

SBERT kNN scored 0.5395 after same-fold parameter selection and about 0.522 in the nested check. It trails both the fixed TF-IDF baseline (0.5820) and the nested tuned baseline (0.6135). The labels appear to follow a word-level lexicon, and logistic regression on word counts can learn a weight for each sentiment word, while kNN votes among tweets that are close in overall meaning and need not share those words. Negative tweets are the hardest: kNN finds 157 of the 388 (recall 0.40) and predicts 114 as Neutral and 117 as Positive (Figure [n]). The FastText vectors, learned from about 1,000 tweets, do worse than Sentence-BERT.

We also tested whether questionable labels hurt kNN by removing tweets from the training folds only. Confident learning (Northcutt et al., 2021) flags a third of each training part (33.4%). Over the five folds it flags 72% of the VADER disagreements at least once, but also 51% of the other tweets. Training without the flagged tweets lowers the score to 0.494, and training without the VADER disagreements lowers it to 0.521, so kNN trains on all tweets.

---

## 4.3 Error analysis

We compared four frozen sets of out-of-fold predictions: TextCNN, the fixed TF-IDF baseline, SBERT kNN and the archived FastText MLP run. This sample of models keeps Table 8 and its counts tied to the same predictions. All four misclassify 152 of the 1,222 development tweets (12.4%), while 1,070 tweets (87.6%) are classified correctly by at least one model, which gives us a reason to compare the models’ errors before trying fusion.

The tweets that every model gets wrong more often have a questionable label. VADER disagrees with the given label for 13.2% of them, against 3.4% of the tweets that no model gets wrong, and the share rises with the number of models that fail (Figure [n]). Finance terms are also more common: 39.5% of these tweets contain words such as *red*, *green*, *bearish*, *bullish*, *Fed* or *Nifty*, against 26.4% of the others (odds ratio 1.81, Fisher's exact test p = 0.001). The counts for single words are small, so we rely on the combined test. A third group of errors comes from the damaged file: 9.9% of these tweets contain a negation whose apostrophe was lost (*don__ panic*), against 3.6% of the others. A lexicon reads such a tweet without the negation, while our cleaned text restores it, so the models and the label disagree.

The most common error is predicting Positive for a Neutral or Negative tweet (64 of the 152, and 15 more where Positive ties with another label). Table 8 shows typical cases. In several of them the label agrees with VADER but not with what the tweet says, so a high score means agreement with lexicon-style labels more than with an investor's reading of the market.

Table 8  Representative classification errors (predicted label = most common prediction of the four models; a tie lists both)

| Case | Given label | Predicted label | Explanation |
|---|---|---|---|
| 1574 | Negative | Positive | "Don__ panic. Look at the bigger picture." The file lost the apostrophe of *don't*, so a lexicon reads *panic* without the negation (VADER −0.51; +0.40 with *don't* restored). Our cleaning restores *don't*, and the models read the encouragement. |
| 806 | Positive | Negative or Neutral (2 models each) | "Blood bath continues in Indian stock Market ... NIFTY: 16,842 -308 ... Market tanks over 2000 points." The only lexicon word is *Nifty* (+1.7), so the label is Positive although the tweet reports a fall. |
| 1160 | Positive | Neutral | "@user sad but true #bearmarket." VADER weights the clause after *but* more heavily, so *true* outweighs *sad* (+0.39). |
| 1455 | Negative | Positive | "Time to purchase more stock during this #stockmarketcrash ... I buy low and sell high." The lexicon scores *yes* (+1.7) before *but* and *low* (−1.1) after it; its *but* rule makes *low* dominate (−0.20). The tweet is optimistic. |
| 1134 | Positive | Neutral | "Yay #stockmarketcrash." Sarcasm. The label follows *yay* literally, and the tweet is too short for the models. |
| 1649 | Negative | Neutral | "My anxiety level is the only thing up in this bear market." The label is plausible; the models miss a mild sentiment word among market words. |

These errors suggest four changes for a later version: treat index names such as *Nifty* as names, flag tweets whose lexicon score flips when a lost negation is restored, combine the models by stacking, and label a sample of the hardest tweets by hand to measure how much of the remaining error is label noise.

---

## 5.5 Limitations and future work

The labels agree closely with the VADER lexicon, but the original labelling process is unknown. If they were produced this way, our models learn to imitate a general-purpose lexicon, which gives no score to *bearish*, *bullish* or *short* and reads the index name *Nifty* as praise. In 112 of the 1,525 retained tweets (7.3%) VADER disagrees with the given label, and the error analysis shows that these tweets are over-represented among the tweets that every model gets wrong.

The source file damaged emoji and apostrophes. We restored only the apostrophe patterns we could verify and removed the rest, but 397 retained tweets (26.0%) lost characters, and emoji often carry sentiment. Lemmatisation also loses some meaning: it turns *Fed* into *feed* in 12 of the 37 retained tweets that contain the word, and `text_norm` drops prices and percentages, which remain only in `text_clean`.

Each training fold has fewer than 1,000 tweets. Earlier BiLSTM and imbalance experiments used the labels on the 303 reserve rows to report scores. The technical submission leaves those rows unscored and reports the exploratory ensemble on the development folds. Any decision to include the old reserve scores belongs in Section 4.2 with their original protocols. Scores of tuned models are optimistic when the setting is chosen on the folds that score it; for kNN the nested estimate is 0.017 lower. The small corpus also limits embeddings trained from scratch, as the FastText neighbours show.

The tweets come from one year and were collected through crash-related hashtags, so they centre on market falls. Crypto terms appear in 31.5% of the retained tweets and Indian market terms in 13.5%, against 7.3% for the US market. The experiments use English text only, so the results do not cover the 102 non-English tweets or the users who wrote them.

The tweets are public, but they can contain names and opinions about people. We replaced every user mention with `@user`, the file has no user IDs, and the data stays in a private repository used only for this course. If the labels come from an English lexicon, the models inherit its blind spots for finance vocabulary and for English mixed with other languages.

Future work, in order of priority:

1. Label a random sample of about 200 tweets by hand, two people per tweet, to measure how often the given labels match human judgement, since every score in this report is measured against those labels.
2. Obtain a copy of the tweets with emoji and apostrophes intact and compare the models on it.
3. Collect tweets from other periods and markets to test whether the results generalise.
4. Try a finance-specific encoder such as FinBERT (Araci, 2019) as the embedding model, to see whether it reduces the errors on market vocabulary.

---

## Appendix C Project development

C.1 Schedule (status on 20 November)

| Planned date | Milestone | Owner | Status |
|---|---|---|---|
| 26 Oct | Shared repository, environment and helper code | 2 | Done 26 Oct |
| 26 Oct | Data cleaning (`clean_v2`) and cleaning log | 2 | Done 26 Oct |
| 26 Oct | Task division published, revised to version 2 the same day | 2 | Done 26 Oct |
| 28 Oct | First team meeting | All | Held 28 Oct, all six members |
| 29 Oct | Fixed data split `folds_v1` | 4 | Generated 27 Oct by member 2; member 4's own implementation matched it fold by fold on 4 Nov |
| 29 Oct | FastText embeddings and first kNN | 2 | Done 27 Oct |
| 1 Nov | Baseline results and EDA figures | 1, 3 | Baseline code 27 Oct, official run 2 Nov; EDA uploaded 3 Nov |
| 6 Nov | Interim report | 2 | Assembled 4 Nov, submitted |
| 7 Nov | Consultation with the lecturer | All | Held 7 Nov |
| 8 Nov | Sentence-BERT vectors, PyTorch training template, shared report document | 5, 4, 6 | Template uploaded 3 Nov and completed in the shared code 4 Nov; Sentence-BERT vectors with TextCNN, 7 Nov on a branch and 14 Nov on main |
| 13 Nov | First version of the six models; imbalance comparison | All, 6 | TextCNN 14 Nov, BiLSTM and imbalance 18 Nov (on a branch 4 Nov), kNN with Sentence-BERT 18 Nov, MLP 4 Nov; SVM and XGBoost outstanding |
| 15 Nov | Out-of-fold and test predictions | All | kNN and TextCNN shared in the repository |
| 17 Nov | Error analysis | 2 | Done 18 Nov on four models |
| 18 Nov | Experiments frozen, model comparison, final model | 1 | Outstanding |
| 19 to 21 Nov | Report sections, slides and merged notebook; rehearsal on 21 Nov | 6, 5, 4 | In progress |
| 22 Nov | Report, notebook and slides submitted | 6 | Due |

C.2 Individual responsibilities

| Member | Model | Shared tasks | Report sections |
|---|---|---|---|
| 1 FENG Dingran | SVM | Baseline, stacking, final model comparison | 3.4 (baseline), 4.1, 4.2, 5.1 |
| 2 GU Kailei | kNN | Cleaning and FastText, error analysis, coordination, interim report | 1.3, 3.1, 3.3, 4.3, 5.5, Appendix C |
| 3 YANG Qitian | XGBoost | EDA, SHAP, business insights | 2, 5.3 |
| 4 CAO Taihe | MLP | Data split, PyTorch training template, merged notebook | 3.2, 3.5, 5.4, Appendix A |
| 5 YU Guanlin | TextCNN | Sentence-BERT vectors and token sequences, slide design | 3.3 (Sentence-BERT) |
| 6 CHOI CHUN MAN | BiLSTM with attention | Imbalance handling, report formatting, submission | Abstract, 1.1, 1.2, 5.2, 5.6 |

Every member also writes the Section 3.4 paragraph on their own model and presents their own part of the slides.

C.3 Iterations

| Date | Change | Reason and evidence |
|---|---|---|
| 26 Oct | Hashtags kept whole | VADER agrees with the labels on 91.9% of retained tweets with whole hashtags and on 71.1% when *stockmarketcrash* is split, because the split adds the word *crash* |
| 26 Oct | Lost characters removed and flagged instead of replaced by a placeholder | A transformer would read the placeholder as a word |
| 26 Oct | Non-English and word-less tweets set aside | Most are labelled Neutral and an English lexicon cannot check their labels; all rows stay in `clean_v2_all.csv` |
| 26 Oct | Second language check added | 16 French, Spanish, Italian and Romanian tweets had passed the first rule; the retained set went from 1,541 to 1,525 tweets |
| 26 Oct | FastText made smaller and reproducible | 200,000 hash buckets instead of two million, one worker and a fixed seed |
| 27 Oct | Plan to reserve the test split until the experiment freeze | Later BiLSTM and imbalance runs reported scores on it. The technical submission records this history and does not score it again. |
| 27 Oct | Two label-noise filters for kNN tested and rejected | With FastText vectors both lowered macro-F1 (0.461 and 0.456 against 0.473) |
| 2 Nov | Official baseline run on `folds_v1` | Macro-F1 0.582; member 1's own rerun on 4 Nov gave the same score |
| 4 Nov | MLP with three configurations | In member 4's environment both refinements lowered macro-F1; in the team environment the regularized version scored highest. The report gives both |
| 4 Nov | Interim report wording revised after review | Label origin described as a hypothesis; set-aside tweets described as kept in `clean_v2_all.csv` |
| 12 to 14 Nov | TextCNN candidates compared with nested selection | Configurations C0, C1, C3 and C7 compared; C3 adopted (macro-F1 0.618) |
| 18 Nov | Historical BiLSTM and imbalance experiments | The BiLSTM score of 0.550 used the scoring fold for early stopping. The original imbalance run gave 0.5092, 0.5274 and 0.5268 for no adjustment, weights and oversampling. |
| 18 Nov | kNN with Sentence-BERT and a nested check | 0.539 tuned, 0.522 nested; both label-noise filters lowered the score again |
| 18 Nov | Error analysis on four models | 152 tweets wrong in every model; labels follow the damaged text in 13 of 14 negation cases |
| 19 Nov | Member 5's deliverables reduced to one notebook | Intermediate evidence removed from the repository |
| 21 Nov | Final development review | Nested baseline 0.6135, corrected BiLSTM 0.5211 and exploratory equal-weight fusion 0.6639. The imbalance rerun gave 0.5164, 0.5230 and 0.5354 with its regenerated FastText cache. |

C.4 Collaboration

All code, data versions and results are in one private GitHub repository. Members upload their work to their own `MemberX_file` folder; member 2 moves shared pieces such as the split and the training template into the common folders and checks their numbers against the shared split. Every model reads the same cleaned data, the same split file and the same evaluation functions, and every experiment has a `run_id` that links its scores in `results/`, its parameters and its saved predictions. Minutes of the team meeting (28 Oct) and the consultation (7 Nov) are in `docs/meetings/`, and report drafts are in `docs/report_drafts/`.

---

## References added by these sections

- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. arXiv:1908.10063.
- Arora, S., Liang, Y., & Ma, T. (2017). A simple but tough-to-beat baseline for sentence embeddings. *International Conference on Learning Representations*.
- Bird, S., Klein, E., & Loper, E. (2009). *Natural language processing with Python*. O'Reilly Media.
- Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. *Transactions of the Association for Computational Linguistics, 5*, 135-146.
- Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. In *Proceedings of the Eighth International AAAI Conference on Weblogs and Social Media* (pp. 216-225).
- Northcutt, C., Jiang, L., & Chuang, I. (2021). Confident learning: Estimating uncertainty in dataset labels. *Journal of Artificial Intelligence Research, 70*, 1373-1411.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. In *Proceedings of EMNLP-IJCNLP 2019* (pp. 3982-3992).
