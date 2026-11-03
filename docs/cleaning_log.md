# clean_v2 cleaning log

Produced by `data/clean_v2/clean_v2.py` from `data/raw/stock_market_crash_2022.csv`. Re-running the script reproduces these numbers exactly.

## Summary

- Raw tweets: 1,698; retained: **1,525**; removed: 173.
- Removed rows stay in `clean_v2_all.csv` with `keep = 0` and a `drop_reason`, so every removal can be audited or reversed in an ablation experiment.

| Label | Raw | Retained | Retained share |
|---|---|---|---|
| Positive | 627 | 607 | 39.8% |
| Neutral | 575 | 433 | 28.4% |
| Negative | 496 | 485 | 31.8% |

Largest / smallest class after cleaning: 1.40.

## Steps

| Step | Operation | Rule | Rationale | Rows affected | Rows remaining | Example | Details |
|---|---|---|---|---|---|---|---|
| 0 | Load raw data | read the course CSV; id = row number in the raw file; label_name from the dataset description | stable ids let every later file (splits, embeddings, predictions) align on the same tweet | 1698 | 1698 |  |  |
| 1 | Check missing and empty values | isna() on both columns; whitespace-only text | required by the brief; none found, so nothing is imputed or removed | 0 | 1698 |  |  |
| 2 | Remove exact duplicates | identical raw text; first copy kept (labels of all copies agree) | duplicates inflate counts and leak between train and test | 9 | 1689 | #stockmarketcrash |  |
| 3 | Repair encoding and HTML entities | ftfy.fix_text, then html.unescape | double-encoded UTF-8 ('estÃ©s') and '&amp;' are scraping artefacts, not user text | 155 | 1689 | Mientras estÃ©s vendiendo en el bear market, algui… → Mientras estés vendiendo en el bear market, alguie… |  |
| 4 | Remove unreadable tweets | >= 2 mojibake residues, < 4 readable words and < 50 % Latin letters | non-Latin scripts whose bytes were lost in the source file; no model can read them | 21 | 1668 | à¤¹à¤®à¥_ à¤¤à à¤¶àà¤¤° à¤¨à à¤²àà¤_¤¾... ⏎ à¤²à¤¡à¤¼àà¤¿à¤¯… |  |
| 5 | Anonymise mentions | @handle -> @user (count kept in n_mentions) | handles carry no sentiment, often contain underscores, and are personal data | 273 | 1668 | …. Why, just why?!! ⏎ @opensea #BearMarket #BearOpe… → …. Why, just why?!! ⏎ @user #BearMarket #BearOpenSe… |  |
| 6 | Restore verified contractions | it__ -> it's, don__ -> don't, you__e -> you're / you've (by the next word) | the source replaced the apostrophe and the next character with '__'; restored text keeps negations | 166 | 1668 | It__ finally here!  #StockMarketcrash → It's finally here!  #StockMarketcrash | 's: 80, n't: 77, 're/'ve: 27, I'm: 16, 'll: 9, y'all: 3, I've: 2 |
| 7 | Remove lost characters | delete '_' runs and mojibake residue outside #/@/$ tokens; flag emoji_lost = 1; merge newlines and repeated spaces | the original emoji and symbols cannot be recovered; a placeholder would be read as text by transformers | 453 | 1668 | #ETH is gone_¦ð_#Ethereum #cryptocurrencies #bearm… → #ETH is gone #Ethereum #cryptocurrencies #bearmark… |  |
| 8a | Remove non-English tweets (rules) | non-Latin script > 30 % of letters; OR >= 2 Hinglish function words (>= 15 % of words); OR >= 3 stopwords of one other language and more than twice the English stopwords; OR langdetect non-English with p >= 0.80 and < 50 % of words found in an English word list | experiments are limited to English text; an English lexicon cannot check these labels (mostly Neutral); rows stay in clean_v2_all.csv for a comparison run | 84 | 1584 | Mientras estÃ©s vendiendo en el bear market, alguien estarÃ¡… | label mix: {'Neutral': 59, 'Positive': 15, 'Negative': 10} |
| 8b | Remove non-English tweets (manual list) | 20 ids listed in clean_v2.py, each read in full | language detection is unreliable on short or romanised text | 18 | 1566 | Pura market down..  ⏎  ⏎ Aek akela #TCS Upper ____ ⏎  ⏎ #sto… | label mix: {'Neutral': 15, 'Positive': 3} |
| 9 | Remove tweets without words | no word of >= 2 letters once hashtags, mentions and cashtags are removed | no English text to classify; all are labelled Neutral, so keeping them could teach a 'no text = Neutral' shortcut; rows stay in clean_v2_all.csv | 39 | 1527 | #stockmarketcrash | label mix: {'Neutral': 39} |
| 10 | Normalise hashtags | '#StockMarketCrash' -> 'StockMarketCrash' (no word splitting); stray '#' removed; list kept in hashtags | VADER treats each hashtag as one unknown token; splitting adds words (e.g. 'crash') that VADER would not score in the raw text | 1527 | 1527 | #bearmarket i expect retest of 30k by end of jan → bearmarket i expect retest of 30k by end of jan |  |
| 11 | Remove content duplicates | same text after lowercasing and removing @user, punctuation and lost characters; first copy kept | copies that differ only by tagged account, emoji or punctuation | 2 | 1525 | yes please and thank you  #stockmarketcrash | 0 rows in groups with conflicting labels |
| 12 | Build text_norm | expand contractions; lowercase; letters only; POS-aware WordNet lemmatisation; NLTK stopwords removed except negations and direction / intensity words | bag-of-words input for TF-IDF and FastText; negations and up/down/below flip the meaning of market tweets | 1525 | 1525 | I don't see it. stockmarketcrash → not see stockmarketcrash |  |
| 13 | Remove empty text_norm | nothing left after normalisation | no input for bag-of-words models | 0 | 1525 |  |  |
| 14 | Group related tweets | union of: token Jaccard >= 0.7 (>= 5 tokens), identical text_norm, and duplicates with their first copy; group_id = smallest raw id in the group | 02_splits keeps a group inside one fold, so templates such as 'Candle of day … closed' never straddle train and test | 20 | 1525 |  | 8 groups among retained tweets |
| 15 | Engineer features | lengths, hashtag / mention / cashtag counts, ! and ?, caps_ratio, hashtag_ratio, is_hashtag_spam (>= 0.6), domain_crypto / india / us from hashtags and cashtags | signals that normalisation removes, for XGBoost and EDA; VADER is not a feature (label leakage) | 1525 | 1525 |  |  |
| F | Final dataset | English, readable, non-duplicate tweets that contain words | label distribution: {'Positive': 607, 'Negative': 485, 'Neutral': 433} | 0 | 1525 |  |  |

## Removed rows

| Reason | Rows | Examples |
|---|---|---|
| non_english | 84 | Mientras estÃ©s vendiendo en el bear market, …; Bhai sare Fintwits ne toh paisa banaya hai aj… |
| no_content | 39 | #stockmarketcrash; @CoinFlashx #JusticeForJohnnyDepp  ⏎ #stockma… |
| unreadable_encoding | 21 | à¤¹à¤®à¥_ à¤¤à à¤¶àà¤¤° à¤¨à à¤²àà¤_¤¾... ⏎ à…; #stockmarketcrash à´´±àµ_´±à´¿à_àµ__àµ |
| non_english_manual | 18 | Pura market down..  ⏎  ⏎ Aek akela #TCS Upper…; @JasonBo68354014 Ladki ka chakkar babu bhaiya… |
| exact_duplicate | 9 | #stockmarketcrash; #stockmarketcrash |
| content_duplicate | 2 | yes please and thank you  #stockmarketcrash; US economy going towards double digit inflati… |

## Columns of clean_v2.csv

| Column | Type | Meaning | Used by (member) |
|---|---|---|---|
| id | int | Row number in data/raw/stock_market_crash_2022.csv; never renumbered | all files align on it |
| group_id | int | Related tweets (duplicates, near-duplicates, identical text_norm) share one id | 4: 02_splits |
| label | int | Given label: 0 Neutral, 1 Positive, 2 Negative | all models |
| label_name | str | Label as text | EDA, charts |
| text_raw | str | Original tweet, unchanged | EDA, error analysis |
| text_clean | str | Encoding repaired, @user, contractions restored, lost characters removed, '#' removed (hashtags not split); case and punctuation kept | 1: Sentence-BERT; 5: token sequences |
| text_norm | str | text_clean lowercased, letters only, lemmatised, stopwords removed except negations and direction words | 1: TF-IDF baseline; 2: FastText |
| hashtags | str | Hashtags of the tweet, lowercase, space-separated | 3: EDA, features |
| lang | str | en, or short (too short to detect); other values only appear in clean_v2_all.csv | EDA |
| emoji_lost | int | 1 if the source file destroyed characters in this tweet (usually emoji) | 3: features, error analysis |
| char_len / word_len | int | Length of text_clean in characters / words | 3: EDA, XGBoost |
| n_hashtags / n_mentions / n_cashtags | int | Counts in the repaired text before anonymisation | 3: XGBoost |
| n_exclaim / n_question | int | Number of ! and ? in text_clean | 3: XGBoost |
| caps_ratio | float | Upper-case share of the letters in text_clean | 3: XGBoost |
| hashtag_ratio / is_hashtag_spam | float / int | Hashtags per token; flag when >= 0.6 | 3: XGBoost, EDA |
| domain_crypto / domain_india / domain_us | int | Market mentioned in hashtags or cashtags | 3: business insight |

`clean_v2_all.csv` has the same columns plus `keep` (1/0) and `drop_reason`, for all 1,698 raw tweets.

## Evidence behind the design decisions

**Hashtags are not split.** Agreement between VADER and the given labels on the retained tweets:

| Text given to VADER | Agreement |
|---|---|
| raw text | 92.7% |
| text_clean (hashtags kept whole) | 91.9% |
| hashtags split on CamelCase (clean_v1 style) | 88.1% |
| 'stockmarketcrash' split into words | 71.1% |

Splitting hashtags adds words such as *crash* that VADER would not score in the raw text, so agreement drops. The small gap between raw text and text_clean comes mostly from restored contractions: of the 15 tweets where VADER matches the label on the raw text but not on text_clean, 13 contain a restored contraction. On the raw text VADER reads 'shouldn__ worry' without the negation (Negative), while the cleaned text says 'shouldn't worry'. If the labels were computed on the raw text, these tweets carry label noise from the corrupted source rather than cleaning errors.

**Non-English tweets are set aside.** The current experiments cover tweets with English text. Label mix of the non-English tweets: {'Neutral': 74, 'Positive': 18, 'Negative': 10}. VADER scores words it does not know as 0, which becomes Neutral. Of the 28 removed non-English tweets labelled Positive or Negative, 27 get the same label from VADER on the raw text: their sentiment comes from words VADER reads as English, such as Spanish *sin* ('without'), German *die* ('the'), Italian *dire* ('to say') and Romanian *care* ('which'). These patterns are consistent with lexicon-based labels, but they do not show that every given label is wrong; the rows stay in `clean_v2_all.csv` for a comparison run.

**Tweets without words are set aside.** Label mix: {'Neutral': 39}.

**Negations and direction words are kept in text_norm.** Retained tweets whose text_norm contains: *not* 307, *no* 69, *never* 22, *up* 84, *down* 102, *below* 23, *more* 81. Standard NLTK stopword lists would delete all of them.

**Template tweets are grouped.** 20 retained tweets fall into 8 groups (near-duplicates or identical text_norm, e.g. 'Candle of day … closed'); a group must stay inside one fold.

VADER is used only for this analysis and must never be a model feature.

## Changes from clean_v1

- Columns renamed: `text_light` → `text_clean`, `text_full` → `text_norm`; added `label_name`, `hashtags`, `emoji_lost` and 13 feature columns; dropped `n_restored` and `n_unk`.
- Hashtags are no longer split on CamelCase.
- Lost characters are removed and flagged instead of being replaced by `[UNK]`.
- Non-English tweets and tweets without words are set aside for the current experiments (they stay in `clean_v2_all.csv`).
- Direction and intensity words (up, down, below, more …) survive stopword removal.
- `group_id` also joins tweets with identical `text_norm`, which fixes template tweets crossing train and test.
- The unreadable-tweet rule also requires < 50 % Latin letters, so 'Rug pull 100% Ahead' is kept.

## Reproduce

```bash
conda activate dsai4205
python data/clean_v2/clean_v2.py
```
