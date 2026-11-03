# Report draft: label origin, cleaning and text representation (member 2)

Draft text for the interim and final reports (sections 2.1, 3.1 and 3.3), following the section numbers of the report template. All numbers come from `data/clean_v2/clean_v2.py` (shown in `notebooks/01_data/01_cleaning.ipynb`) and `notebooks/04_analysis/31_label_noise_errors.ipynb` on data version `clean_v2`; the full step-by-step log is in `data/clean_v2/clean_v2_log.md` (copied to `docs/cleaning_log.md` for the appendix). Figures are in `figures/`.

---

## For 2.1 Data quality: label origin

The dataset description does not say how the tweets were labelled. Because a lexicon-based labeller would make our models imitate that tool, we tested whether the labels can be reproduced by VADER, a widely used rule-based sentiment analyser, with its recommended ±0.05 threshold on the compound score. The test uses the 1,666 readable, non-duplicate tweets: the 1,525 retained ones plus the non-English and word-less tweets removed later, because those removed groups are part of the evidence.

VADER applied to the raw tweet text reproduces 93.0% of the labels (Cohen's κ = 0.89), while TextBlob, a different sentiment lexicon, reproduces only 60.2% (κ = 0.40). Agreement falls to 92.1% when VADER reads our cleaned text, which indicates that the labels were computed on the original text. Three further observations fit the same explanation:

- Non-English tweets agree with VADER 98.0% of the time and are labelled *Neutral* 72.5% of the time, against 25.4% for retained English tweets (Figure *cleaning_label_by_language*). VADER scores words it does not know as 0. Of the 28 non-English tweets labelled Positive or Negative, 27 get that label from VADER, through English words mixed into romanised Hindi (*profit*, *fun*) or through "false friends" that VADER reads as English: Spanish *sin* ('without'), German *die* ('the'), Italian *dire* ('to say').
- Tweets without any words (only hashtags or mentions) are all labelled *Neutral*.
- Tweets whose emoji were lost in the file agree slightly less often (91.2% vs 93.7%). VADER scores emoji, so this fits labels computed before the emoji were destroyed.

The results are **highly consistent with VADER-style rule-based labelling, but the original labelling process is unknown**. We therefore treat label origin as a working hypothesis with four consequences: if it holds, the task effectively means learning to imitate a sentiment lexicon; VADER scores are never used as a model feature, since that could leak the label; the current experiments are limited to tweets with English text, while the non-English and word-less tweets stay in `clean_v2_all.csv` for a later comparison run; and the 112 retained tweets where VADER and the given label disagree are treated as uncertain labels in the error analysis.

## 3.1 Data cleaning

**Table 4 Data cleaning operations**

| Operation | Rationale | Records affected |
|---|---|---|
| Check missing values | Neither column has missing or whitespace-only values, so nothing is imputed | 0 |
| Remove exact duplicates | The same tweet text appears more than once; all copies carry the same label, the first is kept | 9 removed |
| Repair encoding and HTML entities | Double-encoded UTF-8 such as *estÃ©s* is repaired with ftfy; `&amp;`, `&gt;` are unescaped | 155 modified |
| Remove unreadable tweets | Non-Latin scripts whose original bytes were lost; ftfy cannot repair them and no model can read them | 21 removed |
| Replace user mentions with `@user` | User names carry no sentiment, are personal data and often contain underscores that would interfere with the next step | 273 modified |
| Restore verified contractions | The file replaced the apostrophe ’ and the following character with `__` (*it’s* → *it__*). Only patterns checked by hand against every occurrence are restored (*'s*, *n't*, *'re/'ve*, *I'm*, *'ll*, *y'all*, *I've*) | 166 modified (214 restorations) |
| Remove lost characters | Remaining `__` runs (emoji, symbols or possessive *'s*, which cannot be told apart) and leftover mojibake are deleted, and the tweet is flagged with `emoji_lost = 1` | 453 modified |
| Set aside non-English tweets | The current experiments cover English text; an English lexicon cannot check these labels, which are mostly Neutral (see 2.1). Kept in `clean_v2_all.csv` for a comparison run. Rules: non-Latin script, romanised Hindi function words, stopwords of another language, or a confident language detector agreeing with an English word list; short or romanised tweets the rules cannot judge were checked by hand | 102 removed |
| Set aside tweets without words | Only hashtags, mentions or cashtags; all labelled *Neutral*, so keeping them could teach a "no text = Neutral" shortcut; kept in `clean_v2_all.csv` | 39 removed |
| Normalise hashtags and whitespace | `#` removed but hashtags **not split** into words (*#StockMarketCrash* → *StockMarketCrash*), because VADER treats each hashtag as one unknown token; line breaks merged | all rows |
| Remove content duplicates | Tweets identical after lowercasing and removing mentions, punctuation and lost characters | 2 removed |
| Group related tweets | Near-duplicates (token Jaccard ≥ 0.7) and tweets with identical normalised text share a `group_id` and are kept in the same fold | 20 tweets in 8 groups |
| Engineer features | Lengths, hashtag / mention / cashtag counts, `!` and `?`, upper-case ratio, hashtag ratio and market flags (crypto, India, US) for EDA and XGBoost | 13 columns |
| **Final retained dataset** | English, readable, non-duplicate tweets that contain words | **1,525 of 1,698** |

The steps run in a fixed order because several depend on each other: unreadable tweets are detected before lost characters are deleted (the mojibake is the evidence), mentions are replaced before underscores are interpreted, language is detected before the `#` symbols are removed (otherwise hashtag words such as *stockmarketcrash* confuse the detector), and content duplicates are searched only after the text has been repaired. Every step logs the rule, the number of rows it affected and a before/after example; Appendix C reproduces the log. Removed rows are not lost: `clean_v2_all.csv` keeps all 1,698 tweets with a `keep` flag and a `drop_reason`, so every removal can be reversed in an ablation experiment.

*Why hashtags are not split.* On the retained tweets, VADER agrees with the labels 91.9% of the time when hashtags stay whole, 88.1% when they are split on capital letters, and 71.1% when *stockmarketcrash* is split into *stock market crash*. Splitting adds the word *crash*, which VADER would not score in the raw text.

*Before → after examples*

| Raw tweet | `text_clean` | `text_norm` |
|---|---|---|
| `I don__ see it. #stockmarketcrash` | `I don't see it. stockmarketcrash` | `not see stockmarketcrash` |
| `It__ finally here!  #StockMarketcrash` | `It's finally here! StockMarketcrash` | `finally stockmarketcrash` |
| `#ETH is gone_¦ð_#Ethereum #cryptocurrencies #bearmarket` | `ETH is gone Ethereum cryptocurrencies bearmarket` | `eth go ethereum cryptocurrencies bearmarket` |

*Conflicting labels.* No exact or content duplicate had conflicting labels, so no row was removed for that reason. One near-duplicate pair (the same tweet addressed to two different users) is labelled *Positive* in one copy and *Negative* in the other; both are kept, grouped, and listed among the label-noise candidates.

*What was not removed.* No tweet was removed for its length or as an outlier: a one-hashtag tweet and a 400-character thread are both legitimate tweets. There are no missing values in either column.

*Class balance.* Cleaning removed mostly *Neutral* rows (142 of the 173 removed tweets). The final distribution is Positive 607 (39.8%), Neutral 433 (28.4%) and Negative 485 (31.8%), a largest-to-smallest ratio of 1.40. This mild imbalance is handled with stratified splitting, macro-averaged metrics and class weights.

## 3.3 Text representation (normalisation part)

Two versions of each tweet are produced, because the models need different inputs:

- **`text_clean`** keeps case and punctuation (23 words on average). Transformer encoders such as Sentence-BERT use capitalisation and punctuation, so we only repair the text.
- **`text_norm`** is fully normalised for the TF-IDF baseline and the self-trained FastText: contractions are expanded (*don't* → *do not*), text is lowercased, only letters are kept (punctuation and digits removed), words are lemmatised with WordNet using part-of-speech tags (*selling* → *sell*), and NLTK English stopwords are removed **except negations and direction words**. Standard stopword lists would delete *not* (in 307 retained tweets), *down* (102), *up* (84), *more* (81) and *below* (23), turning "not bullish" into "bullish" and "down 20%" into nothing. A `text_norm` tweet has 14.3 tokens on average (median 13, maximum 39).

Characters destroyed in the source file are not replaced by a placeholder token, because transformers would read the placeholder as text. Instead, the `emoji_lost` flag (397 retained tweets) keeps that information for the feature-based models and the error analysis.

### Self-trained word embedding: FastText

FastText (Bojanowski et al., 2017) learns a vector for every word and for its character n-grams, so it can still build a vector for a rare or misspelled word from its parts. It is our embedding trained from scratch: we train it with gensim on `text_norm` (skip-gram, 100 dimensions, context window 5, character n-grams of 3–5 letters, words that appear at least twice, 50 epochs, fixed seed 42).

Although FastText never sees the labels, it must follow the same split as the classifiers. If the validation tweets of fold *k* were part of its training text, every model that uses it would get a slightly optimistic cross-validation score. We therefore train six FastText models:

- for each fold *k*, one model trained only on the other four folds (975–980 tweets, vocabulary of 1,386–1,420 words), which encodes all training tweets for that fold;
- one `full` model trained on all 1,222 training tweets (vocabulary of 1,645 words), which encodes the training and test tweets for the final test predictions.

Test tweets are never used to train any FastText model. Each model gives three representations of a tweet:

- **mean vector**: the average of its word vectors (100 numbers);
- **SIF vector** (Arora et al., 2017): a weighted average in which frequent words get less weight, weight *a* / (*a* + *p*(*w*)) with *a* = 0.001, followed by removing the first principal component shared by all tweets. Word frequencies and the principal component are computed on the training part only;
- **token sequence**: the word vectors in order, padded or cut to 31 tokens (the 99th percentile of tweet length; 8 tweets are cut), with a mask marking the padding. These are the FastText inputs of TextCNN and BiLSTM.

We do not weight word vectors with TF-IDF, because the project brief allows TF-IDF only in the baseline.

The nearest neighbours in the `full` model show what such a small corpus can learn. Some neighbours are meaningful (*bull*: *bullrun*, *bullish*, *bullmarket*; *recession*: *depression*), but many only share spelling (*dip*: *sip*, *sit*, *rip*; *bear*: *bearish*, *beat*, *swear*). With about 1,000 training tweets, the character n-grams dominate, which limits how well these vectors separate the sentiment classes (Section 4.1).
