# Report draft: dataset description (member 2)

Draft for section 1.3 of the final report (Table 1 and Table 2). Sources: `docs/materials/Dataset3_Description.docx`, `data/raw/stock_market_crash_2022.csv` and `data/clean_v2/clean_v2_log.md`.

---

## 1.3 Dataset description

We use Dataset 3 of the course, "Huge crash in the Stock market". According to its description, the tweets were collected from Twitter under the hashtag #stockmarketcrash during the 2022 stock market crash, and each tweet has one of three sentiment labels. The file has 1,698 rows and two columns: the tweet text and its label. Besides #stockmarketcrash (1,009 tweets), many tweets also carry #bearmarket (720 tweets).

The file contains no timestamps and no user information, so the collection period cannot be checked directly. The content is consistent with 2022: 56 tweets use hashtags that contain "2022" (for example #BearMarket2022), 92 mention Russia or Ukraine, and 26 mention the LUNA/Terra cryptocurrency, which collapsed in May 2022.

The description does not explain how the labels were assigned. Section 2.1 shows that they were very probably produced automatically by the VADER sentiment tool, which matters for how we interpret every result in this report. Most tweets are in English; 123 tweets (7.2%) are in other languages or could not be read because their characters were damaged in the file. These are removed during cleaning (Section 3.1), which leaves 1,525 tweets.

**Table 1 Dataset overview**

| Attribute | Description |
|---|---|
| Dataset and version | Dataset 3 "Huge crash in the Stock market", file `stock_market_crash_2022.csv` as provided by the course; cleaned version `clean_v2` |
| Source and time coverage | Tweets collected from Twitter under #stockmarketcrash during the 2022 market crash (dataset description); the file has no timestamps |
| Unit of analysis | One tweet |
| Size and language | 1,698 tweets, 1,525 after cleaning; mostly English, 123 tweets (7.2%) not in English or unreadable |
| Target labels | `text_sentiment`: 0 Neutral (575 tweets), 1 Positive (627), 2 Negative (496); the labelling method is not stated |
| Usage conditions | Provided by the course for this assignment; the description states no licence. The tweets are public posts but contain user names, which we replace with `@user` |

**Table 2 Data dictionary**

The raw file has the first two fields. The cleaned file `clean_v2.csv`, which every model reads, keeps them under new names and adds the fields below.

| Field | Type | Description |
|---|---|---|
| `text` → `text_raw` | Text | Original tweet, unchanged; used for EDA and error analysis |
| `text_sentiment` → `label` | Integer (0, 1, 2) | Target: 0 Neutral, 1 Positive, 2 Negative; `label_name` holds the same value as text |
| `id` | Integer | Identifier: row number in the raw file, never renumbered; links the data, the split, the embeddings and the predictions |
| `group_id` | Integer | Group: duplicate and near-duplicate tweets share one value, so the split keeps them on the same side |
| `text_clean` | Text | Repaired tweet with case and punctuation kept; input for Sentence-BERT and token sequences |
| `text_norm` | Text | Lower-cased, lemmatised tweet without stopwords (negations and direction words kept); input for TF-IDF and FastText |
| `emoji_lost` | Integer (0, 1) | 1 if the source file destroyed characters in the tweet, usually emoji (397 retained tweets) |
| 13 feature columns | Integer or decimal | Length, counts of hashtags, mentions, cashtags, `!` and `?`, upper-case ratio, hashtag ratio, a hashtag-spam flag and market flags (crypto, India, US); inputs for XGBoost and EDA |

The full column list with explanations is in `data/clean_v2/clean_v2_log.md`.
