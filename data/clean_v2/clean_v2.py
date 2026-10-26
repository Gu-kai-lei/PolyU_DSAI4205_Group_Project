"""
DSAI4205 Group Project (DataVinci) - Dataset 3: Huge crash in the Stock market
Data cleaning pipeline, version 2 (clean_v2)

Input : stock_market_crash_2022.csv  (columns: text, text_sentiment; 1=Positive, 0=Neutral, 2=Negative)
Output: clean_v2.csv                 retained rows, cleaned text + engineered features + split
        clean_v2_all_flagged.csv     all 1698 raw rows with keep flag and drop_reason (for ablations)
        cleaning_log_v2.csv          step-by-step log (operation, rule, rationale, rows affected)

Key design decisions (see cleaning_log_v2.csv for counts):
  * Labels agree with VADER on the raw text for 93.2% of rows, so they are lexicon-generated weak labels.
    Therefore: hashtags are NOT word-segmented (VADER saw '#stockmarketcrash' as one token, score 0;
    'stock market crash' would score -0.40), negations are preserved, and VADER is never used as a feature.
  * Non-English tweets are removed from the main set: VADER only knows English, so their 'Neutral'
    labels are artefacts. They stay in clean_v2_all_flagged.csv for an ablation experiment.

Usage: python clean_v2.py <raw_csv> <out_dir>
"""
import html
import re
import sys
import unicodedata
from pathlib import Path

import ftfy
import numpy as np
import pandas as pd
import spacy
from langdetect import DetectorFactory, detect_langs
from wordfreq import available_languages, zipf_frequency
from sklearn.model_selection import train_test_split

SEED = 42
DetectorFactory.seed = 0
LABEL_NAMES = {1: "Positive", 0: "Neutral", 2: "Negative"}

RAW = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("stock_market_crash_2022.csv")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(".")
OUT.mkdir(parents=True, exist_ok=True)

LOG = []


def log(step, operation, rule, rationale, affected, remaining, before="", after=""):
    LOG.append(dict(step=step, operation=operation, rule=rule, rationale=rationale,
                    rows_affected=int(affected), rows_remaining=int(remaining),
                    example_before=before[:200], example_after=after[:200]))


def first_change(before: pd.Series, after: pd.Series):
    m = before != after
    if not m.any():
        return "", ""
    i = m.idxmax()
    return before[i], after[i]


# --------------------------------------------------------------------------------------
# Step 0. Load
# --------------------------------------------------------------------------------------
df = pd.read_csv(RAW)
df.insert(0, "raw_id", np.arange(len(df)))
df = df.rename(columns={"text": "text_raw", "text_sentiment": "label"})
df["label_name"] = df["label"].map(LABEL_NAMES)
df["keep"] = True
df["drop_reason"] = ""
N0 = len(df)
assert df["label"].isin(LABEL_NAMES).all()
log("0", "Load raw data", "read CSV; add raw_id; map label 1/0/2 -> Positive/Neutral/Negative",
    "traceable IDs and readable labels; macOS junk files in the zip (__MACOSX, .DS_Store) ignored",
    N0, N0)
n_missing = int(df[["text_raw", "label"]].isna().any(axis=1).sum()
                + (df["text_raw"].astype(str).str.strip() == "").sum())
log("0b", "Check missing / empty values", "isna() on text and label; whitespace-only text",
    "required by brief; none found, so no imputation or removal needed", n_missing, N0)

t = df["text_raw"].astype(str)

# --------------------------------------------------------------------------------------
# Step 1. Encoding repair
# --------------------------------------------------------------------------------------
t1 = t.map(lambda x: ftfy.fix_text(x, unescape_html=False))
b, a = first_change(t, t1)
log("1a", "Fix mojibake", "ftfy.fix_text (double-encoded UTF-8, e.g. 'estÃ©s' -> 'estés')",
    "restore original characters so tokens and language detection work", (t != t1).sum(), N0, b, a)
t2 = t1.map(html.unescape)
b, a = first_change(t1, t2)
log("1b", "Decode HTML entities", "html.unescape ('&amp;' -> '&', '&gt;' -> '>')",
    "entities are scraping artefacts, not user text", (t1 != t2).sum(), N0, b, a)
t3 = t2.map(lambda s: unicodedata.normalize("NFC", s))
t3 = t3.str.replace(r"\s+", " ", regex=True).str.strip()
b, a = first_change(t2, t3)
log("1c", "Unicode NFC + whitespace", "NFC normalisation; newlines/tabs/multiple spaces -> one space",
    "consistent representation; NFC used instead of NFKC because NFKC turns residue '¾' into digits '3⁄4'",
    (t2 != t3).sum(), N0, b, a)

# --------------------------------------------------------------------------------------
# Step 2. Underscore corruption: curly apostrophes and emoji were replaced by '_' during scraping
# --------------------------------------------------------------------------------------
S_STEMS = {"it", "that", "what", "there", "here", "he", "she", "who", "how", "where", "let",
           "everyone", "everything", "nobody", "something", "one"}
NT_STEMS = {"don", "can", "isn", "doesn", "won", "didn", "haven", "couldn", "shouldn", "ain",
            "wasn", "aren", "wouldn", "hasn", "weren", "mustn", "hadn"}
PRON = {"i", "you", "we", "they", "he", "she", "it", "that", "there", "who"}


def fix_contraction(m):
    stem, suf = m.group(1), m.group(2)
    s, x = stem.lower(), suf.lower()
    if x == "" and s in NT_STEMS:
        return stem + "'t"
    if x == "" and s in S_STEMS:
        return stem + "'s"
    if x == "" and s == "i":
        return stem + "'m"
    if s in PRON and x == "e":          # you__e: "you're" (could also be "you've"; 're is more frequent)
        return stem + ("'ve" if s == "i" else "'re")
    if s in PRON and x in {"l", "ll"}:  # I__l -> I'll
        return stem + "'ll"
    if s in PRON and x == "d":
        return stem + "'d"
    if s == "y" and x == "ll":          # y__ll -> y'all
        return "y'all"
    return None


CONTR = re.compile(r"(?<![\w#@])([A-Za-z]+)__(?!_)([A-Za-z]{0,2})(?![A-Za-z0-9_])")


def repair_contractions(s):
    def rep(m):
        r = fix_contraction(m)
        if r is None:
            return m.group(0)
        return r.upper() if m.group(1).isupper() and len(m.group(1)) > 1 else r  # DON__ -> DON'T
    return CONTR.sub(rep, s)


t4 = t3.map(repair_contractions)
b, a = first_change(t3, t4)
log("2a", "Restore contractions", "word__ -> apostrophe form via rule map (don__ -> don't, it__ -> it's, you__e -> you're)",
    "lost apostrophes break negation ('don__' -> don't -> 'not'); text restored to its true meaning although labels were made on the corrupted text (see step V)",
    (t3 != t4).sum(), N0, b, a)

# residue = runs containing '_' plus adjacent Latin-1 symbols / combining marks left by broken emoji bytes
RESIDUE = re.compile(r"[ -¿×÷ïð̀-ͯ⁄€]*_+"
                     r"[_ -¿×÷ïð̀-ͯ⁄€]*")
TOKEN = re.compile(r"([#@$][A-Za-z0-9_]+)")


def strip_residue(s):
    parts = TOKEN.split(s)
    n = 0
    out = []
    for i, p in enumerate(parts):
        if i % 2 == 1:  # protected hashtag / mention / cashtag: keep inner '_' (e.g. #ES_F, @cz_binance)
            core = p[0] + p[1:].strip("_")
            n += int(core != p)
            out.append(core if len(core) > 1 else " ")
        else:
            p2, k = RESIDUE.subn(" ", p)
            n += k
            out.append(p2)
    s2 = "".join(out)
    s2, k2 = re.subn(r"\u00e2(?=[\s\u0080-\u00bf]|$)", " ", s2)  # orphan 'â' = first byte of a broken dash/quote
    s2 = re.sub(r"\s+", " ", s2).strip()
    return s2, n + k2


res = t4.map(strip_residue)
t5 = res.map(lambda x: x[0])
df["emoji_lost"] = (res.map(lambda x: x[1]) > 0).astype(int)
b, a = first_change(t4, t5)
log("2b", "Remove emoji residue", "remove '_' runs and attached broken symbol bytes outside #/@/$ tokens; set emoji_lost=1",
    "the original emoji are unrecoverable; residue would create junk tokens. Flag kept because lost emoji explain part of label noise",
    (t4 != t5).sum(), N0, b, a)

# --------------------------------------------------------------------------------------
# Step 3. Twitter-specific standardisation
# --------------------------------------------------------------------------------------
t6 = t5.str.replace(r"^RT\s+", "", regex=True)
log("3a", "Remove retweet marker", "leading 'RT '", "platform artefact", (t5 != t6).sum(), N0)
df["n_mentions"] = t6.str.count(r"@\w+")
df["n_hashtags"] = t6.str.count(r"#\w+")
df["n_cashtags"] = t6.str.count(r"\$[A-Za-z]{1,6}\b")
t7 = t6.str.replace(r"@\w+", "@user", regex=True)
b, a = first_change(t6, t7)
log("3b", "Anonymise mentions", "@handle -> @user (count kept in n_mentions)",
    "handles are identifiers with no sentiment; anonymisation also limits personal data",
    (t6 != t7).sum(), N0, b, a)
df["hashtags"] = t7.str.findall(r"#(\w+)").map(lambda xs: " ".join(x.lower() for x in xs))
df["text_hashtag_kept"] = t7  # cleaned text still containing '#', used by the language/content checks
t8 = t7.str.replace(r"#(\w+)", r"\1", regex=True)
b, a = first_change(t7, t8)
log("3c", "Hashtag symbol", "'#bearmarket' -> 'bearmarket'; no word segmentation; hashtag list kept in column hashtags",
    "labels come from VADER which scored each hashtag as one unknown token; segmenting would add sentiment the labeller never saw",
    (t7 != t8).sum(), N0, b, a)
df["text_clean"] = t8

# --------------------------------------------------------------------------------------
# Step 4. Duplicates
# --------------------------------------------------------------------------------------
def drop(mask, reason):
    m = mask & df["keep"]
    df.loc[m, "keep"] = False
    df.loc[m, "drop_reason"] = reason
    return int(m.sum())


def conflict_groups(key):
    g = df[df["keep"]].groupby(key)["label"].nunique()
    return set(g[g > 1].index)


exact = df.duplicated(subset=["text_raw"], keep="first")
assert not conflict_groups(df["text_raw"]), "exact duplicates with conflicting labels"
k = drop(exact, "exact_duplicate")
log("4a", "Remove exact duplicates", "identical raw text (keep first); label conflicts checked: 0",
    "duplicates inflate counts and leak between train/test", k, df["keep"].sum(),
    df.loc[exact.idxmax(), "text_raw"] if k else "")

df["dedup_key"] = (df["text_clean"].str.lower().str.replace("@user", " ", regex=False)
                   .str.replace(r"[^a-z0-9$ ]", " ", regex=True).str.split().str.join(" "))
conf = conflict_groups(df["dedup_key"])
near = df["keep"] & df.duplicated(subset=["dedup_key"], keep="first") & ~df["dedup_key"].isin(conf)
k = drop(near, "near_duplicate")
k_conf = drop(df["dedup_key"].isin(conf), "duplicate_label_conflict")
log("4b", "Remove near duplicates", "same text after lowercasing, removing @user/punctuation/emoji residue (keep first)",
    "copies that differ only by tagged account or emoji", k, df["keep"].sum(),
    df.loc[near.idxmax(), "text_raw"] if k else "")
log("4c", "Resolve duplicate label conflicts", "drop whole group if duplicates disagree on label",
    "no reliable way to pick the right label; 0 groups found", k_conf, df["keep"].sum())

# --------------------------------------------------------------------------------------
# Step 5. Content-free tweets
# --------------------------------------------------------------------------------------
body = (df["text_hashtag_kept"].str.replace(r"[#@$]\w+", " ", regex=True)
        .str.replace(r"[^A-Za-zÀ-ɏͰ-￿]+", " ", regex=True))
df["n_body_words"] = body.str.split().map(lambda ws: sum(len(w) >= 2 for w in ws))
k = drop(df["n_body_words"] == 0, "no_content_hashtag_or_mention_only")
log("5", "Remove content-free tweets", "no word of >=2 letters left after removing hashtags, mentions, cashtags, numbers, symbols",
    "nothing to classify; all such tweets were labelled Neutral by default, which teaches a 'no text = Neutral' shortcut",
    k, df["keep"].sum())

# --------------------------------------------------------------------------------------
# Step 6. Language
# --------------------------------------------------------------------------------------
HINGLISH = {"hai", "hain", "nahi", "nahin", "kya", "bhai", "toh", "ke", "ka", "ki", "ko", "mein", "ye", "yeh",
            "bhi", "aur", "kar", "karo", "raha", "rha", "rhe", "hota", "tak", "sab", "abhi", "jo", "kuch",
            "paisa", "paise", "yaar", "mera", "meri", "apna", "kaise", "kyu", "kyun", "gaya", "gya", "wala",
            "wale", "se", "hua", "hue", "diya", "liya", "naam", "chiz", "agar", "lekin"}


WF_LANGS = set(available_languages())


def en_vocab_ratio(words, lang):
    """share of words that are more frequent in English than in the detected language (wordfreq)"""
    words = [w for w in words if len(w) >= 2]
    if not words:
        return 1.0
    if lang in WF_LANGS and lang != "en":
        hits = [zipf_frequency(w, "en") >= zipf_frequency(w, lang) and zipf_frequency(w, "en") > 0 for w in words]
    else:
        hits = [zipf_frequency(w, "en") >= 3.5 for w in words]
    return sum(hits) / len(words)


def lang_info(s):
    s = re.sub(r"[#@$]\w+", " ", s)
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return "short", 1.0, 1.0
    non_latin = sum(not (c.isascii() or "\u00c0" <= c <= "\u024f") for c in letters) / len(letters)
    words = re.findall(r"[a-z\u00c0-\u024f']+", s.lower())
    hing = sum(w in HINGLISH for w in words)
    if non_latin > 0.3:
        return "non_latin", 1.0, 0.0
    if len(re.findall(r"\u00e0[\u00a4\u00a5\u00aa\u00ab]", s)) >= 2:  # 'à¤'/'àª' = broken Devanagari/Gujarati bytes
        return "garbled_indic", 1.0, 0.0
    if hing >= 2 and hing / max(len(words), 1) >= 0.15:
        return "hinglish", 1.0, round(en_vocab_ratio(words, "en"), 3)
    if len(letters) < 20:
        return "short", 1.0, round(en_vocab_ratio(words, "en"), 3)
    try:
        top = detect_langs(s)[0]
        lang, prob = top.lang, round(top.prob, 3)
    except Exception:
        lang, prob = "unk", 0.0
    return lang, prob, round(en_vocab_ratio(words, lang), 3)


li = df["text_hashtag_kept"].map(lang_info)
df["lang"] = li.map(lambda x: x[0])
df["lang_prob"] = li.map(lambda x: x[1])
df["en_vocab_ratio"] = li.map(lambda x: x[2])
non_en = (df["lang"].isin(["non_latin", "garbled_indic", "hinglish"])
          | (~df["lang"].isin(["en", "short", "unk", "non_latin", "garbled_indic", "hinglish"])
             & (df["lang_prob"] >= 0.80) & (df["en_vocab_ratio"] < 0.50)))
df["is_english"] = (~non_en).astype(int)
dist = df.loc[df["keep"] & non_en, "label_name"].value_counts().to_dict()
k = drop(non_en, "non_english")
log("6a", "Remove non-English tweets (rules)",
    "non-Latin script >30% of letters, OR >=2 broken Devanagari/Gujarati byte pairs, OR >=2 Hinglish markers (>=15% of words), OR langdetect non-en with p>=0.80 and <50% of words more frequent in English than in that language (wordfreq)",
    f"labeller (VADER) is English-only, so these labels are artefacts (removed label mix: {dist}); rows kept in all_flagged file for ablation",
    k, df["keep"].sum())

# Manual review: every kept row not detected as 'en' (short texts, low-confidence detections) and every
# 'en' row containing Hindi function words was read by a team member. These raw_ids are clearly
# non-English (romanised Hindi/Punjabi, Spanish, Italian, French, Tagalog) but too short for the rules.
# Code-mixed tweets whose sentiment-bearing part is English (e.g. raw_id 1091) are kept.
MANUAL_NON_EN = {14, 35, 63, 69, 70, 187, 232, 241, 248, 360, 409, 444, 462, 526, 535, 553, 664, 676, 1155}
man = df["raw_id"].isin(MANUAL_NON_EN)
df.loc[man, "lang"] = "manual_non_en"
df.loc[man, "is_english"] = 0
k = drop(man, "non_english_manual_review")
log("6b", "Remove non-English tweets (manual review)", f"raw_id in MANUAL_NON_EN ({len(MANUAL_NON_EN)} ids, listed in clean_v2.py)",
    "language detection is unreliable on short / romanised text; borderline rows checked by hand", k, df["keep"].sum())

# --------------------------------------------------------------------------------------
# Step 7. Outlier flags (kept, not removed)
# --------------------------------------------------------------------------------------
tok = df["text_hashtag_kept"].str.split().str.len()
df["hashtag_ratio"] = (df["n_hashtags"] / tok).round(3)
df["is_hashtag_spam"] = (df["hashtag_ratio"] >= 0.6).astype(int)
log("7", "Flag hashtag-spam outliers", "hashtag_ratio = hashtags / tokens >= 0.6 -> is_hashtag_spam=1 (not removed)",
    "extreme but genuine tweets; flag lets models/analysis treat them separately",
    (df["keep"] & (df["is_hashtag_spam"] == 1)).sum(), df["keep"].sum())

# --------------------------------------------------------------------------------------
# Step 8. Normalised text for TF-IDF / Word2Vec: lowercase, punctuation & number removal,
#         tokenisation + lemmatisation (spaCy), stopwords removed EXCEPT negations
# --------------------------------------------------------------------------------------
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
NEG = {"not", "no", "nor", "never", "none", "nothing", "nobody", "neither", "nowhere", "without",
       "cannot", "against", "down", "up", "off", "over", "very", "too", "less", "more", "most"}
STOP = (nlp.Defaults.stop_words | {"user", "amp", "rt"}) - NEG


def normalise(s):
    s = s.replace("@user", " ")
    s = re.sub(r"\$([A-Za-z]{1,6})\b", r" \1 ", s)  # cashtag $TSLA -> tsla
    out = []
    for tk in nlp(s.lower()):
        lem = tk.lemma_.lower().strip()
        if tk.text in {"n't", "nt"} or lem == "not":
            out.append("not")
            continue
        if tk.is_punct or tk.is_space or tk.like_num or not re.search(r"[a-z]", lem) or re.match(r"\d", lem):
            continue  # drop numbers/prices ('30k', '16160'); keep alnum hashtags ('nifty50', 'bearmarket2022')
        if tk.text.endswith("ies") and lem.endswith("ie"):
            lem = lem[:-2] + "y"  # spaCy: 'cryptocurrencies' -> 'cryptocurrencie'; fix to 'cryptocurrency'
        lem = re.sub(r"[^a-z0-9']", "", lem).strip("'")
        if len(lem) < 2 or lem in STOP:
            continue
        out.append(lem)
    return " ".join(out)


kept = df["keep"]
df["text_norm"] = ""
df.loc[kept, "text_norm"] = df.loc[kept, "text_clean"].map(normalise)
ex = df.loc[kept & (df["text_clean"].str.contains("n't")), ["text_clean", "text_norm"]].iloc[0]
log("8a", "Create text_norm", "lowercase; spaCy en_core_web_sm tokenise + lemmatise; drop punctuation, numeric tokens (prices, dates), stopwords; KEEP negations/intensifiers",
    "brief requires normalisation + lemmatisation; negation words kept because they flip sentiment",
    kept.sum(), kept.sum(), ex["text_clean"], ex["text_norm"])
k = drop(df["text_norm"].str.strip() == "", "empty_after_normalisation")
log("8b", "Remove empty normalised text", "text_norm empty", "no features for TF-IDF", k, df["keep"].sum())

# --------------------------------------------------------------------------------------
# Step 9. Engineered features (on text_clean) and stratified split
# --------------------------------------------------------------------------------------
tc = df["text_clean"]
df["char_len"] = tc.str.len()
df["word_len"] = tc.str.split().str.len()
df["n_exclaim"] = tc.str.count("!")
df["n_question"] = tc.str.count(r"\?")
letters = tc.str.count(r"[A-Za-z]")
df["caps_ratio"] = (tc.str.count(r"[A-Z]") / letters.replace(0, np.nan)).fillna(0).round(3)
CRYPTO = r"btc|bitcoin|eth|ethereum|crypto|nft|doge|altcoin|defi|blockchain|solana|luna|terra|binance|bnb|xrp|cardano|web3|shib|usdt|coin|tezos|hodl"
INDIA = r"nifty|sensex|nse|bse|india|sgx|zerodha|dalal"
USSTK = r"sp500|spx|spy|nasdaq|dow|tsla|aapl|amzn|nvda|wallstreet|nyse|fed\b|qqq|es_f|nq_f"
for name, pat in [("domain_crypto", CRYPTO), ("domain_india", INDIA), ("domain_us", USSTK)]:
    df[name] = df["hashtags"].str.contains(pat, regex=True).astype(int)
log("9a", "Engineer features", "char_len, word_len, n_exclaim, n_question, caps_ratio, n_hashtags, n_mentions, n_cashtags, emoji_lost, domain_* from hashtags",
    "signals removed by normalisation (!, caps) kept as numeric features; domains support business insight. VADER deliberately NOT added (label leakage)",
    df["keep"].sum(), df["keep"].sum())

df["split"] = ""
idx = df.index[df["keep"]]
tr, tmp = train_test_split(idx, test_size=0.30, stratify=df.loc[idx, "label"], random_state=SEED)
va, te = train_test_split(tmp, test_size=0.50, stratify=df.loc[tmp, "label"], random_state=SEED)
df.loc[tr, "split"], df.loc[va, "split"], df.loc[te, "split"] = "train", "val", "test"
log("9b", "Stratified split", f"train/val/test = 70/15/15 stratified by label, random_state={SEED}, after deduplication",
    "dedup before split prevents leakage; fit TF-IDF/vocab on train only",
    df["keep"].sum(), df["keep"].sum())

final = df[df["keep"]]
try:  # validation only - VADER is NOT used as a model feature (it would leak the label)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _v = SentimentIntensityAnalyzer()
    _lab = lambda x: (lambda c: 1 if c >= 0.05 else (2 if c <= -0.05 else 0))(_v.polarity_scores(x)["compound"])
    agr_raw = (final["text_raw"].map(_lab) == final["label"]).mean()
    agr_cln = (final["text_clean"].map(_lab) == final["label"]).mean()
    agr_seg = (final["text_clean"].str.replace(r"(?i)stockmarketcrash", "stock market crash", regex=True)
               .map(_lab) == final["label"]).mean()
    log("V", "Validation: label provenance check",
        "VADER (thresholds +/-0.05) on retained rows: raw text vs text_clean vs hashtags segmented",
        f"agreement raw={agr_raw:.2%}, clean={agr_cln:.2%}, segmented={agr_seg:.2%}. Labels are VADER-generated on the ALREADY-CORRUPTED text "
        "(restoring don't/isn't lowers agreement: e.g. 'don't panic' is labelled Negative). Clean text kept; mismatches treated as label noise. "
        "Segmenting hashtags would drop agreement sharply, confirming step 3c.",
        int((final["text_raw"].map(_lab) != final["label"]).sum()), len(final))
except ImportError:
    pass
log("F", "Final retained dataset",
    "English, non-duplicate, contains real words, non-empty text_norm",
    "label distribution: " + str(final["label_name"].value_counts().to_dict()),
    len(final), len(final))

# --------------------------------------------------------------------------------------
# Save
# --------------------------------------------------------------------------------------
cols = ["raw_id", "split", "label", "label_name", "text_raw", "text_clean", "text_norm", "hashtags",
        "char_len", "word_len", "n_hashtags", "n_mentions", "n_cashtags", "n_exclaim", "n_question",
        "caps_ratio", "hashtag_ratio", "is_hashtag_spam", "emoji_lost",
        "domain_crypto", "domain_india", "domain_us", "lang", "lang_prob", "en_vocab_ratio"]
final[cols].to_csv(OUT / "clean_v2.csv", index=False)
df[["keep", "drop_reason"] + [c for c in cols if c != "split"] + ["split", "is_english"]].to_csv(
    OUT / "clean_v2_all_flagged.csv", index=False)
pd.DataFrame(LOG).to_csv(OUT / "cleaning_log_v2.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame(LOG)[["step", "operation", "rows_affected", "rows_remaining"]].to_string(index=False))
print(final["label_name"].value_counts())
print(final["split"].value_counts())
