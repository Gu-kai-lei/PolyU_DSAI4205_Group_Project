# Data cleaning log

| Step | Operation | Rationale | Rows affected | Rows remaining | Example (before → after) | Details |
|---|---|---|---|---|---|---|
| 0 | Load raw data | Original CSV; one row is one tweet; id = row number in the raw file | 0 | 1698 |  |  |
| 1 | Remove exact duplicates | The same tweet text appears more than once (labels agree); first copy kept | 9 | 1689 | #stockmarketcrash |  |
| 2 | Repair encoding and HTML entities | Double-encoded UTF-8 repaired with ftfy (estÃ©s → estés); HTML entities unescaped (&amp; → &) | 155 | 1689 | Mientras estÃ©s vendiendo en el bear market, alguien es… → Mientras estés vendiendo en el bear market, alguien est… |  |
| 3 | Replace user mentions | @username → @user: names carry no sentiment and often contain underscores | 277 | 1689 | …. Why, just why?!! ⏎ @opensea #BearMarket #BearOpenSea … → …. Why, just why?!! ⏎ @user #BearMarket #BearOpenSea #ET… |  |
| 4 | Restore verified contractions | In the source file the apostrophe ’ and the next character became '__' (it’s → it__, you’re → you__e). Only patterns checked by hand against every occurrence are restored | 166 | 1689 | It__ finally here!  #StockMarketcrash → It's finally here!  #StockMarketcrash | 's: 80, n't: 77, 're/'ve: 27, I'm: 16, 'll: 9, y'all: 3, I've: 2 |
| 5 | Remove unreadable tweets | Non-Latin scripts whose original bytes were lost in the source file (ftfy cannot repair them): ≥ 2 mojibake residues and < 4 readable words | 22 | 1667 | à¤¹à¤®à¥_ à¤¤à à¤¶àà¤¤° à¤¨à à¤²àà¤_¤¾... ⏎ à¤²à¤¡à¤¼àà¤¿à¤¯à¥_ à¤à¤ à… |  |
| 6 | Replace unrecoverable characters with [UNK] | Remaining '__' runs (emoji, symbols or possessive 's, which cannot be told apart) and leftover mojibake | 460 | 1667 | #ETH is gone_¦ð_#Ethereum #cryptocurrencies #bearmarket → #ETH is gone [UNK] #Ethereum #cryptocurrencies #bearmar… | 1233 replacements |
| 7 | Detect language (flag only) | langdetect on the tweet body without hashtags and mentions; < 15 letters → 'und'. Readable non-English tweets are kept and flagged in the lang column | 258 | 1667 |  | en: 1409, und: 133, es: 23, fr: 17, it: 12, id: 10, no: 7, et: 6 |
| 8 | Normalise hashtags and whitespace | Remove '#', split CamelCase hashtags (#StockMarketCrash → Stock Market Crash); merge newlines and spaces | 1667 | 1667 | #bearmarket i expect retest of 30k by end of jan → bearmarket i expect retest of 30k by end of jan |  |
| 9 | Remove content duplicates | Tweets identical after lowercasing and removing mentions, placeholders and punctuation (e.g. '@user #bearmarket' vs '#bearmarket'); first copy kept | 13 | 1654 | #bearmarket |  |
| 10 | Build text_full | Expand contractions (don't → do not), lowercase, keep letters only (drops punctuation and digits), POS-aware WordNet lemmatisation, remove NLTK stopwords except negations (no, not, never…) | 1654 | 1654 | bearmarket i expect retest of 30k by end of jan → bearmarket expect retest end jan |  |
| 11 | Remove tweets with no words left | Only mentions, emoji or symbols remained; nothing to classify | 0 | 1654 |  |  |
| 12 | Group near-duplicates | Token Jaccard ≥ 0.7 (tweets with ≥ 5 tokens) on the content key; a group must stay in one fold | 10 | 1654 |  | 5 groups |
| 13 | Final dataset | Columns: id, group_id, text_raw, text_full, text_light, label, lang, n_restored, n_unk | 0 | 1654 |  |  |

## Notes

- Data version: clean_v1. Produced by notebooks/01_data/01_cleaning.ipynb with src/cleaning.py.
- [UNK] marks a character that cannot be recovered from the source file: an emoji, a symbol, or a possessive 's (the source replaced the apostrophe and the following character with '__'). In text_full it appears as the token 'unk'.
- Readable non-English tweets are kept and flagged in the lang column (247 rows, of which 122 are too short to tell).
- group_id links near-duplicate tweets; 02_splits must keep every group inside one fold.
- Extra columns: lang (detected language), n_restored (contractions restored), n_unk (characters replaced by [UNK]).
