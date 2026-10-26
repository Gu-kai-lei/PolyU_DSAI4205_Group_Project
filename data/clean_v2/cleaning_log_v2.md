# Data Cleaning Log — clean_v2

**Input:** `stock_market_crash_2022.csv` 1698 rows · **Output:** `clean_v2.csv` 1523 rows · **Script:** `clean_v2.py` (seed 42)

## 1. Summary

| Step | Operation | Type | Rows affected | Rows remaining |
|---|---|---|---:|---:|
| 0 | Load raw data | transform | 1698 | 1698 |
| 0b | Check missing / empty values | validate | 0 | 1698 |
| 1a | Fix mojibake | transform | 73 | 1698 |
| 1b | Decode HTML entities | transform | 85 | 1698 |
| 1c | Unicode NFC + whitespace | transform | 1036 | 1698 |
| 2a | Restore contractions | transform | 167 | 1698 |
| 2b | Remove emoji residue | transform | 480 | 1698 |
| 3a | Remove retweet marker | transform | 1 | 1698 |
| 3b | Anonymise mentions | transform | 278 | 1698 |
| 3c | Hashtag symbol | transform | 1698 | 1698 |
| 4a | Remove exact duplicates | remove | 9 | 1689 |
| 4b | Remove near duplicates | remove | 23 | 1666 |
| 4c | Resolve duplicate label conflicts | remove | 0 | 1666 |
| 5 | Remove content-free tweets | remove | 28 | 1638 |
| 6a | Remove non-English tweets (rules) | remove | 96 | 1542 |
| 6b | Remove non-English tweets (manual review) | remove | 19 | 1523 |
| 7 | Flag hashtag-spam outliers | flag | 28 | 1523 |
| 8a | Create text\_norm | transform | 1523 | 1523 |
| 8b | Remove empty normalised text | remove | 0 | 1523 |
| 9a | Engineer features | transform | 1523 | 1523 |
| 9b | Stratified split | transform | 1523 | 1523 |
| V | Validation: label provenance check | validate | 112 | 1523 |
| F | Final retained dataset | validate | 1523 | 1523 |

## 2. Removed rows by reason and label

| Reason | Negative | Neutral | Positive | Total |
|---|---:|---:|---:|---:|
| exact\_duplicate | 0 | 9 | 0 | 9 |
| near\_duplicate | 1 | 21 | 1 | 23 |
| no\_content\_hashtag\_or\_mention\_only | 0 | 28 | 0 | 28 |
| non\_english | 12 | 68 | 16 | 96 |
| non\_english\_manual\_review | 0 | 16 | 3 | 19 |
| kept (final dataset) | 483 | 433 | 607 | 1523 |

Of the 175 removed rows, 142 were labelled Neutral: invalid or non-English text had been defaulted to Neutral by the lexicon labeller.

## 3. Final dataset

| Split | Negative | Neutral | Positive | Total |
|---|---:|---:|---:|---:|
| train | 338 | 303 | 425 | 1066 |
| val | 72 | 65 | 91 | 228 |
| test | 73 | 65 | 91 | 229 |
| **total** | **483** | **433** | **607** | **1523** |

Imbalance ratio (largest / smallest class): 1.40

## 4. Step details

### Step 0 — Load raw data

- **Rule:** read CSV; add raw\_id; map label 1/0/2 -> Positive/Neutral/Negative
- **Rationale:** traceable IDs and readable labels; macOS junk files in the zip (\_\_MACOSX, .DS\_Store) ignored
- **Rows affected:** 1698 · **Rows remaining:** 1698

### Step 0b — Check missing / empty values

- **Rule:** isna() on text and label; whitespace-only text
- **Rationale:** required by brief; none found, so no imputation or removal needed
- **Rows affected:** 0 · **Rows remaining:** 1698

### Step 1a — Fix mojibake

- **Rule:** ftfy.fix\_text (double-encoded UTF-8, e.g. 'estÃ©s' -> 'estés')
- **Rationale:** restore original characters so tokens and language detection work
- **Rows affected:** 73 · **Rows remaining:** 1698
- **Before:** `Mientras estÃ©s vendiendo en el bear market, alguien estarÃ¡ comprando. ⏎ #bearmarket`
- **After:** `Mientras estés vendiendo en el bear market, alguien estará comprando. ⏎ #bearmarket`

### Step 1b — Decode HTML entities

- **Rule:** html.unescape ('&amp;' -> '&', '&gt;' -> '>')
- **Rationale:** entities are scraping artefacts, not user text
- **Rows affected:** 85 · **Rows remaining:** 1698
- **Before:** `They say make money in compounding, &amp; here I'm loosing money in compounding__ ⏎ . ⏎ . ⏎ . ⏎ . ⏎ . ⏎ #stockmarketcrash #StockMarket #Nifty #sensex #marketcrash`
- **After:** `They say make money in compounding, & here I'm loosing money in compounding__ ⏎ . ⏎ . ⏎ . ⏎ . ⏎ . ⏎ #stockmarketcrash #StockMarket #Nifty #sensex #marketcrash`

### Step 1c — Unicode NFC + whitespace

- **Rule:** NFC normalisation; newlines/tabs/multiple spaces -> one space
- **Rationale:** consistent representation; NFC used instead of NFKC because NFKC turns residue '¾' into digits '3⁄4'
- **Rows affected:** 1036 · **Rows remaining:** 1698
- **Before:** `Mientras estés vendiendo en el bear market, alguien estará comprando. ⏎ #bearmarket`
- **After:** `Mientras estés vendiendo en el bear market, alguien estará comprando. #bearmarket`

### Step 2a — Restore contractions

- **Rule:** word\_\_ -> apostrophe form via rule map (don\_\_ -> don't, it\_\_ -> it's, you\_\_e -> you're)
- **Rationale:** lost apostrophes break negation ('don\_\_' -> don't -> 'not'); text restored to its true meaning although labels were made on the corrupted text (see step V)
- **Rows affected:** 167 · **Rows remaining:** 1698
- **Before:** `It__ finally here! #StockMarketcrash`
- **After:** `It's finally here! #StockMarketcrash`

### Step 2b — Remove emoji residue

- **Rule:** remove '\_' runs and attached broken symbol bytes outside #/@/$ tokens; set emoji\_lost=1
- **Rationale:** the original emoji are unrecoverable; residue would create junk tokens. Flag kept because lost emoji explain part of label noise
- **Rows affected:** 480 · **Rows remaining:** 1698
- **Before:** `#ETH is gone_¦ð_#Ethereum #cryptocurrencies #bearmarket`
- **After:** `#ETH is gone #Ethereum #cryptocurrencies #bearmarket`

### Step 3a — Remove retweet marker

- **Rule:** leading 'RT '
- **Rationale:** platform artefact
- **Rows affected:** 1 · **Rows remaining:** 1698

### Step 3b — Anonymise mentions

- **Rule:** @handle -> @user (count kept in n\_mentions)
- **Rationale:** handles are identifiers with no sentiment; anonymisation also limits personal data
- **Rows affected:** 278 · **Rows remaining:** 1698
- **Before:** `OpenSea removed the activity chart from all #NFT #NFTs collections... Why, just why?!! @opensea #BearMarket #BearOpenSea #ETH #NGMI`
- **After:** `OpenSea removed the activity chart from all #NFT #NFTs collections... Why, just why?!! @user #BearMarket #BearOpenSea #ETH #NGMI`

### Step 3c — Hashtag symbol

- **Rule:** '#bearmarket' -> 'bearmarket'; no word segmentation; hashtag list kept in column hashtags
- **Rationale:** labels come from VADER which scored each hashtag as one unknown token; segmenting would add sentiment the labeller never saw
- **Rows affected:** 1698 · **Rows remaining:** 1698
- **Before:** `#bearmarket i expect retest of 30k by end of jan`
- **After:** `bearmarket i expect retest of 30k by end of jan`

### Step 4a — Remove exact duplicates

- **Rule:** identical raw text (keep first); label conflicts checked: 0
- **Rationale:** duplicates inflate counts and leak between train/test
- **Rows affected:** 9 · **Rows remaining:** 1689
- **Before:** `#stockmarketcrash`

### Step 4b — Remove near duplicates

- **Rule:** same text after lowercasing, removing @user/punctuation/emoji residue (keep first)
- **Rationale:** copies that differ only by tagged account or emoji
- **Rows affected:** 23 · **Rows remaining:** 1666
- **Before:** `#bearmarket`

### Step 4c — Resolve duplicate label conflicts

- **Rule:** drop whole group if duplicates disagree on label
- **Rationale:** no reliable way to pick the right label; 0 groups found
- **Rows affected:** 0 · **Rows remaining:** 1666

### Step 5 — Remove content-free tweets

- **Rule:** no word of >=2 letters left after removing hashtags, mentions, cashtags, numbers, symbols
- **Rationale:** nothing to classify; all such tweets were labelled Neutral by default, which teaches a 'no text = Neutral' shortcut
- **Rows affected:** 28 · **Rows remaining:** 1638

### Step 6a — Remove non-English tweets (rules)

- **Rule:** non-Latin script >30% of letters, OR >=2 broken Devanagari/Gujarati byte pairs, OR >=2 Hinglish markers (>=15% of words), OR langdetect non-en with p>=0.80 and <50% of words more frequent in English than in that language (wordfreq)
- **Rationale:** labeller (VADER) is English-only, so these labels are artefacts (removed label mix: {'Neutral': 68, 'Positive': 16, 'Negative': 12}); rows kept in all\_flagged file for ablation
- **Rows affected:** 96 · **Rows remaining:** 1542

### Step 6b — Remove non-English tweets (manual review)

- **Rule:** raw\_id in MANUAL\_NON\_EN (19 ids, listed in clean\_v2.py)
- **Rationale:** language detection is unreliable on short / romanised text; borderline rows checked by hand
- **Rows affected:** 19 · **Rows remaining:** 1523

### Step 7 — Flag hashtag-spam outliers

- **Rule:** hashtag\_ratio = hashtags / tokens >= 0.6 -> is\_hashtag\_spam=1 (not removed)
- **Rationale:** extreme but genuine tweets; flag lets models/analysis treat them separately
- **Rows affected:** 28 · **Rows remaining:** 1523

### Step 8a — Create text\_norm

- **Rule:** lowercase; spaCy en\_core\_web\_sm tokenise + lemmatise; drop punctuation, numeric tokens (prices, dates), stopwords; KEEP negations/intensifiers
- **Rationale:** brief requires normalisation + lemmatisation; negation words kept because they flip sentiment
- **Rows affected:** 1523 · **Rows remaining:** 1523
- **Before:** `I am a trader, who don't even opened a single trade ( spot + lev ) without SL ever. bearmarket btc bitcoin`
- **After:** `trader not open single trade spot lev without sl bearmarket btc bitcoin`

### Step 8b — Remove empty normalised text

- **Rule:** text\_norm empty
- **Rationale:** no features for TF-IDF
- **Rows affected:** 0 · **Rows remaining:** 1523

### Step 9a — Engineer features

- **Rule:** char\_len, word\_len, n\_exclaim, n\_question, caps\_ratio, n\_hashtags, n\_mentions, n\_cashtags, emoji\_lost, domain\_\* from hashtags
- **Rationale:** signals removed by normalisation (!, caps) kept as numeric features; domains support business insight. VADER deliberately NOT added (label leakage)
- **Rows affected:** 1523 · **Rows remaining:** 1523

### Step 9b — Stratified split

- **Rule:** train/val/test = 70/15/15 stratified by label, random\_state=42, after deduplication
- **Rationale:** dedup before split prevents leakage; fit TF-IDF/vocab on train only
- **Rows affected:** 1523 · **Rows remaining:** 1523

### Step V — Validation: label provenance check

- **Rule:** VADER (thresholds +/-0.05) on retained rows: raw text vs text\_clean vs hashtags segmented
- **Rationale:** agreement raw=92.65%, clean=91.92%, segmented=71.18%. Labels are VADER-generated on the ALREADY-CORRUPTED text (restoring don't/isn't lowers agreement: e.g. 'don't panic' is labelled Negative). Clean text kept; mismatches treated as label noise. Segmenting hashtags would drop agreement sharply, confirming step 3c.
- **Rows affected:** 112 · **Rows remaining:** 1523

### Step F — Final retained dataset

- **Rule:** English, non-duplicate, contains real words, non-empty text\_norm
- **Rationale:** label distribution: {'Positive': 607, 'Negative': 483, 'Neutral': 433}
- **Rows affected:** 1523 · **Rows remaining:** 1523

## 5. Output columns

| Column | Description |
|---|---|
| raw_id | row index in the raw CSV (traceability) |
| split | train / val / test (stratified 70/15/15) |
| label, label_name | 1 Positive, 0 Neutral, 2 Negative |
| text_raw | original tweet |
| text_clean | encoding-fixed, contractions restored, emoji residue removed, @user, '#' dropped; case and punctuation kept (for BERT) |
| text_norm | lowercased, lemmatised, punctuation/numbers/stopwords removed, negations kept (for TF-IDF, Word2Vec) |
| hashtags | lower-cased hashtag list |
| char_len, word_len, n_hashtags, n_mentions, n_cashtags, n_exclaim, n_question, caps_ratio, hashtag_ratio | engineered numeric features |
| is_hashtag_spam, emoji_lost, domain_crypto, domain_india, domain_us | engineered flags |
| lang, lang_prob, en_vocab_ratio | language-detection diagnostics |

Manually reviewed non-English raw_ids (step 6b): 14, 35, 63, 69, 70, 187, 232, 241, 248, 360, 409, 444, 462, 526, 535, 553, 664, 676, 1155
