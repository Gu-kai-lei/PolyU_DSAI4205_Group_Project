"""clean_v2: data-cleaning recipe for DSAI4205 Dataset 3 ("Huge crash in the Stock market").

Run from the repository root (inside the dsai4205 conda environment):

    python data/clean_v2/clean_v2.py

It reads data/raw/stock_market_crash_2022.csv and writes three files next to this script:

    clean_v2.csv       retained tweets; every model reads it through src.data.load_clean()
    clean_v2_all.csv   all 1,698 raw tweets with keep / drop_reason, for ablations and auditing
    clean_v2_log.md    step-by-step log, data dictionary and the evidence behind each design decision

The recipe is frozen with this data version: a later version gets its own folder and script. It only uses
packages from environment.yml (pandas, ftfy, langdetect, nltk, vaderSentiment) and is deterministic, so
running it twice produces byte-identical files.

Design decisions (numbers are recomputed and written to clean_v2_log.md on every run):
  * The labels are very likely VADER compound scores on the raw text. Hashtags are therefore NOT split into
    words ('#StockMarketCrash' stays one token) because VADER never saw the split words.
  * Non-English tweets and tweets without any words are removed: VADER cannot read them, so their labels are
    almost always Neutral by default. They stay in clean_v2_all.csv for ablation experiments.
  * Characters the source file destroyed (emoji, curly quotes) are removed and flagged in emoji_lost instead
    of being replaced by a placeholder token.
  * Negations and direction words (not, up, down, below ...) survive stopword removal in text_norm.
  * group_id joins near-duplicates AND tweets with identical text_norm (template tweets), so a group never
    straddles train and test.
"""
from __future__ import annotations

import html
import re
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW_CSV = ROOT / "data" / "raw" / "stock_market_crash_2022.csv"
NLTK_DIR = ROOT / "data" / "nltk_data"
NLTK_PACKAGES = ("stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger_eng", "words")
VERSION = "clean_v2"
LABEL_NAMES = {0: "Neutral", 1: "Positive", 2: "Negative"}

KEY_COLUMNS = ["id", "group_id", "label", "label_name"]
TEXT_COLUMNS = ["text_raw", "text_clean", "text_norm", "hashtags"]
FEATURE_COLUMNS = ["lang", "emoji_lost", "char_len", "word_len", "n_hashtags", "n_mentions", "n_cashtags",
                   "n_exclaim", "n_question", "caps_ratio", "hashtag_ratio", "is_hashtag_spam",
                   "domain_crypto", "domain_india", "domain_us"]
COLUMNS = KEY_COLUMNS + TEXT_COLUMNS + FEATURE_COLUMNS

# --------------------------------------------------------------------------------------------------------------
# Encoding and Twitter tokens
# --------------------------------------------------------------------------------------------------------------
MENTION = re.compile(r"@\w+")
HASHTAG = re.compile(r"#(\w+)")
CASHTAG = re.compile(r"\$[A-Za-z]{1,6}\b")
PROTECTED = re.compile(r"([#@$][A-Za-z0-9_]+)")  # hashtag / mention / cashtag tokens keep inner underscores


def fix_encoding(text: str) -> str:
    """Repair double-encoded UTF-8 ('estÃ©s' -> 'estés') and unescape HTML entities ('&amp;' -> '&')."""
    import ftfy

    return html.unescape(ftfy.fix_text(text))


# Runs of Latin-1 Supplement characters (U+0080-U+00FF) that contain a C1 control character, or are at least two
# characters long and contain a symbol ('à¤', 'ï¸', '¤£'), are leftovers of bytes the source file destroyed.
# A single 'é', '«', '¿' or '£' is a real character and is kept.
LATIN1_RUN = re.compile("[\u0080-ÿ]+")


def _is_residue(run: str) -> bool:
    has_c1 = any("\u0080" <= ch <= "\u009f" for ch in run)
    has_symbol = any(" " <= ch <= "¿" for ch in run)
    return has_c1 or (len(run) >= 2 and has_symbol)


def count_residue(text: str) -> int:
    return sum(_is_residue(m.group()) for m in LATIN1_RUN.finditer(text))


def is_unreadable(text: str) -> bool:
    """A tweet whose script was destroyed: >= 2 residues, < 4 readable words and < 50 % Latin letters."""
    body = re.sub(r"[#@$]\w+", " ", text)
    letters = [ch for ch in body if ch.isalpha()]
    latin_share = sum(ch.isascii() for ch in letters) / max(len(letters), 1)
    words = re.findall(r"\b[A-Za-z]{2,}\b", body)
    return count_residue(text) >= 2 and len(words) < 4 and latin_share < 0.5


# --------------------------------------------------------------------------------------------------------------
# Contractions: the source replaced the apostrophe AND the following character with '__'
# (it’s -> it__, don’t -> don__, you’re -> you__e). Only patterns checked against every occurrence are restored.
# --------------------------------------------------------------------------------------------------------------
NEGATION_STEMS = ("don", "doesn", "didn", "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn",
                  "couldn", "shouldn", "wouldn", "won", "can", "ain", "mustn", "needn")
S_STEMS = ("it", "that", "what", "there", "let", "he", "she", "who", "how", "here", "where")
LL_STEMS = ("i", "you", "we", "they", "he", "she", "it", "that", "there")
PARTICIPLES = {"been", "got", "gotten", "lost", "seen", "done", "had", "made", "broken", "taken", "given",
               "known", "become", "gone", "bought", "held", "paid", "felt", "heard", "left", "told",
               "thought", "kept", "begun", "fallen", "grown", "shown", "written", "chosen", "missed"}
SKIPPABLE_ADVERBS = {"already", "just", "never", "ever", "finally", "all", "really", "also", "probably", "definitely"}
_END = r"(?=[^A-Za-z0-9_]|$)"


def _join(stem: str, suffix: str) -> str:
    """Keep the case of shouted words: DON__ -> DON'T, don__ -> don't, I__ -> I'm."""
    return stem + (suffix.upper() if stem.isupper() and len(stem) > 1 else suffix)


CONTRACTION_RULES = [
    ("n't", re.compile(rf"\b({'|'.join(NEGATION_STEMS)})__{_END}", re.I), lambda m: _join(m.group(1), "'t")),
    ("I'm", re.compile(rf"\b(i)__{_END}", re.I), lambda m: _join(m.group(1), "'m")),
    ("'s", re.compile(rf"\b({'|'.join(S_STEMS)})__{_END}", re.I), lambda m: _join(m.group(1), "'s")),
    ("'ll", re.compile(rf"\b({'|'.join(LL_STEMS)})__l{_END}", re.I), lambda m: _join(m.group(1), "'ll")),
    ("I've", re.compile(rf"\b(i)__e{_END}", re.I), lambda m: _join(m.group(1), "'ve")),
    ("y'all", re.compile(rf"\b(y)__ll{_END}", re.I), lambda m: _join(m.group(1), "'all")),
]
RE_VE_RULE = re.compile(r"\b(you|we|they)__e\b", re.I)


def _re_or_ve(match: re.Match) -> str:
    """you__e / we__e / they__e -> 've when the next word (after adverbs) is a past participle, else 're."""
    words = [w.lower() for w in re.findall(r"[A-Za-z]+", match.string[match.end():match.end() + 60])]
    while words and words[0] in SKIPPABLE_ADVERBS:
        words.pop(0)
    nxt = words[0] if words else ""
    participle = nxt in PARTICIPLES or (len(nxt) > 3 and nxt.endswith("ed"))
    return _join(match.group(1), "'ve" if participle else "'re")


def restore_contractions(text: str) -> tuple[str, Counter]:
    hits = Counter()
    for name, pattern, repl in CONTRACTION_RULES:
        text, n = pattern.subn(repl, text)
        hits[name] += n
    text, n = RE_VE_RULE.subn(_re_or_ve, text)
    hits["'re/'ve"] += n
    return text, hits


def strip_lost_characters(text: str) -> tuple[str, int]:
    """Remove '_' runs and mojibake residue; return (text, number of removed pieces).

    Hashtag, mention and cashtag tokens keep single inner underscores ('#ES_F'); runs of two or more are lost
    characters ('#Bitcoin__rs' -> '#Bitcoin rs'), and a token that is only underscores ('#____') is removed.
    """
    parts = PROTECTED.split(text)
    removed = 0
    out = []
    for i, part in enumerate(parts):
        if i % 2 == 1:
            inner = part[1:]
            cleaned = re.sub(r"_{2,}", " ", inner).strip("_ ")
            removed += cleaned != inner
            out.append(part[0] + cleaned if cleaned else " ")
            continue
        part, n_underscore = re.subn(r"_+", " ", part)
        n_residue = count_residue(part)
        part = LATIN1_RUN.sub(lambda m: " " if _is_residue(m.group()) else m.group(), part)
        removed += n_underscore + n_residue
        out.append(part)
    return " ".join("".join(out).split()), removed


def normalize_hashtags(text: str) -> str:
    """'#StockMarketCrash' -> 'StockMarketCrash' (not split: VADER saw one unknown token); stray '#' removed."""
    text = HASHTAG.sub(lambda m: " " + m.group(1), text)
    return " ".join(text.replace("#", " ").split())


def content_key(text: str) -> str:
    """Key for 'same content': lowercase, no @user, letters and digits only."""
    text = text.lower().replace("@user", " ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


def body_words(text: str) -> int:
    """Words of >= 2 letters once hashtags, mentions and cashtags are removed."""
    body = re.sub(r"[#@$]\w+", " ", text)
    return sum(len(w) >= 2 for w in re.findall(r"[^\W\d_]+", body))


# --------------------------------------------------------------------------------------------------------------
# Language
# --------------------------------------------------------------------------------------------------------------
HINGLISH = {"hai", "hain", "nahi", "nahin", "kya", "bhai", "toh", "ke", "ka", "ki", "ko", "mein", "ye", "yeh",
            "bhi", "aur", "kar", "karo", "raha", "rha", "rhe", "hota", "tak", "sab", "abhi", "jo", "kuch",
            "paisa", "paise", "yaar", "mera", "meri", "apna", "kaise", "kyu", "kyun", "gaya", "gya", "wala",
            "wale", "se", "hua", "hue", "diya", "liya", "naam", "chiz", "agar", "lekin"}
# Short or romanised tweets that the rules cannot judge. Each was read in full and is not English
# (romanised Hindi / Punjabi / Urdu, Spanish, Italian, French, Tagalog).
MANUAL_NON_ENGLISH = {14, 35, 63, 69, 70, 187, 232, 241, 248, 360, 395, 409, 444, 462, 526, 535, 553, 664, 676, 1155}
# Languages of the stopword vote (NLTK stopword lists), with the code written to the lang column
STOPWORD_LANGUAGES = {"french": "fr", "spanish": "es", "italian": "it", "portuguese": "pt", "german": "de",
                      "dutch": "nl", "romanian": "ro", "indonesian": "id", "swedish": "sv", "norwegian": "no",
                      "danish": "da"}


class LanguageTagger:
    """Language decision per tweet: 'en' and 'short' are kept, any other value is removed.

    Other values: non_latin, hinglish, or the code of the language found by the stopword vote or langdetect.
    """

    def __init__(self):
        from nltk.corpus import stopwords, wordnet, words

        self._wordnet = wordnet
        self.english_stop = set(stopwords.words("english"))
        self.english = {w.lower() for w in words.words()} | self.english_stop
        # Words that are stopwords in English too (a, me, no ...) are no evidence for another language
        self.foreign_stop = {code: set(stopwords.words(name)) - self.english_stop
                             for name, code in STOPWORD_LANGUAGES.items()}

    def english_share(self, tokens: list[str]) -> float:
        tokens = [t for t in tokens if len(t) >= 2]
        if not tokens:
            return 1.0
        known = sum(t in self.english or self._wordnet.morphy(t) is not None for t in tokens)
        return known / len(tokens)

    def stopword_vote(self, tokens: list[str]) -> tuple[str, int, int]:
        """Short-word language identification: (best other language, its stopword hits, English stopword hits).

        The English word list alone cannot separate French or Spanish from English: it contains le, la, de,
        en, se ... Function words can, because every language has its own.
        """
        words = re.findall(r"[a-zà-ɏ]+", " ".join(tokens))  # split c'est -> c est, as the stopword lists do
        code = max(self.foreign_stop, key=lambda c: sum(w in self.foreign_stop[c] for w in words))
        return code, sum(w in self.foreign_stop[code] for w in words), sum(w in self.english_stop for w in words)

    def __call__(self, text: str) -> str:
        from langdetect import DetectorFactory, LangDetectException, detect_langs

        body = re.sub(r"[#@$]\w+", " ", text)
        letters = [ch for ch in body if ch.isalpha()]
        tokens = re.findall(r"[a-zÀ-ɏ']+", body.lower())
        if not letters:
            return "short"
        if sum(not (ch.isascii() or "À" <= ch <= "ɏ") for ch in letters) / len(letters) > 0.3:
            return "non_latin"
        hinglish = sum(t in HINGLISH for t in tokens)
        if tokens and hinglish >= 2 and hinglish / len(tokens) >= 0.15:
            return "hinglish"
        if len(letters) < 15:
            return "short"
        code, foreign, english = self.stopword_vote(tokens)
        if foreign >= 3 and foreign > 2 * english:
            return code
        DetectorFactory.seed = 0  # langdetect is random unless seeded
        try:
            top = detect_langs(body)[0]
        except (LangDetectException, TypeError) as err:  # TypeError: rare internal langdetect error
            warnings.warn(f"langdetect failed ({err!r}) on {body[:50]!r}; tagged as 'short'")
            return "short"
        # langdetect alone mislabels short English tweets ('Stocks are on sale!' -> sv), so it also needs
        # most words to be missing from the English word list
        if top.lang != "en" and top.prob >= 0.80 and self.english_share(tokens) < 0.5:
            return top.lang
        return "en"


# --------------------------------------------------------------------------------------------------------------
# text_norm: expand contractions -> lowercase -> letters only -> POS-aware lemmatisation -> stopwords
# --------------------------------------------------------------------------------------------------------------
NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "nobody", "neither", "nowhere", "cannot", "without"}
# Direction and intensity words carry the meaning of market tweets ('market down', 'below 30k', 'too much red')
KEEP_WORDS = {"up", "down", "over", "under", "off", "out", "above", "below", "more", "most", "less", "least",
              "very", "too", "against", "few"}
EXPANSIONS = [(r"\bcan't\b", "can not"), (r"\bwon't\b", "will not"), (r"\bain't\b", "is not"),
              (r"n't\b", " not"), (r"'re\b", " are"), (r"'m\b", " am"), (r"'ll\b", " will"),
              (r"'ve\b", " have"), (r"'d\b", " would"), (r"'s\b", "")]


def _wordnet_pos(tag: str) -> str:
    return {"J": "a", "V": "v", "R": "r"}.get(tag[:1], "n")


class NormBuilder:
    def __init__(self):
        from nltk import pos_tag
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        self._pos_tag = pos_tag
        self._lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words("english")) - NEGATIONS - KEEP_WORDS

    def __call__(self, text_clean: str) -> str:
        text = text_clean.replace("’", "'").lower().replace("@user", " ")
        for pattern, repl in EXPANSIONS:
            text = re.sub(pattern, repl, text)
        tokens = re.findall(r"[a-z]+", text)
        if not tokens:
            return ""
        lemmas = [self._lemmatizer.lemmatize(tok, _wordnet_pos(tag)) for tok, tag in self._pos_tag(tokens)]
        return " ".join(w for w in lemmas if len(w) > 1 and w not in self.stopwords)


# --------------------------------------------------------------------------------------------------------------
# Groups for leakage-safe splitting
# --------------------------------------------------------------------------------------------------------------
class UnionFind:
    def __init__(self, items):
        self.parent = {i: i for i in items}

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def jaccard_pairs(ids, keys, threshold=0.7, min_tokens=5):
    """Pairs of ids whose content-key token sets have Jaccard similarity >= threshold."""
    sets = [(i, set(k.split())) for i, k in zip(ids, keys)]
    sets = [(i, s) for i, s in sets if len(s) >= min_tokens]
    for a in range(len(sets)):
        id_a, set_a = sets[a]
        for b in range(a + 1, len(sets)):
            id_b, set_b = sets[b]
            inter = len(set_a & set_b)
            if inter and inter / len(set_a | set_b) >= threshold:
                yield id_a, id_b


# --------------------------------------------------------------------------------------------------------------
# Features (VADER is deliberately not one of them: it would leak the label)
# --------------------------------------------------------------------------------------------------------------
DOMAINS = {
    "domain_crypto": r"btc|bitcoin|eth|ethereum|crypto|nft|doge|altcoin|defi|blockchain|solana|luna|terra|binance"
                     r"|bnb|xrp|cardano|web3|shib|usdt|coin|tezos|hodl",
    "domain_india": r"nifty|sensex|nse|bse|india|sgx|zerodha|dalal",
    "domain_us": r"sp500|spx|spy|nasdaq|dow|tsla|aapl|amzn|nvda|wallstreet|nyse|fed\b|qqq|es_f|nq_f",
}


def add_features(df: pd.DataFrame) -> None:
    clean = df["text_clean"]
    letters = clean.str.count(r"[A-Za-z]")
    tokens = df["text_fixed"].str.split().str.len().clip(lower=1)
    df["char_len"] = clean.str.len()
    df["word_len"] = clean.str.split().str.len()
    df["n_hashtags"] = df["text_fixed"].str.count(r"#\w+")
    df["n_mentions"] = df["text_fixed"].str.count(r"@\w+")
    df["n_cashtags"] = df["text_fixed"].str.count(CASHTAG.pattern)
    df["n_exclaim"] = clean.str.count("!")
    df["n_question"] = clean.str.count(r"\?")
    df["caps_ratio"] = (clean.str.count(r"[A-Z]") / letters.replace(0, np.nan)).fillna(0).round(3)
    df["hashtag_ratio"] = (df["n_hashtags"] / tokens).round(3)
    df["is_hashtag_spam"] = (df["hashtag_ratio"] >= 0.6).astype(int)
    tags = df["hashtags"] + " " + df["text_fixed"].str.findall(CASHTAG.pattern).str.join(" ").str.lower()
    for name, pattern in DOMAINS.items():
        df[name] = tags.str.contains(pattern, regex=True).astype(int)


# --------------------------------------------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------------------------------------------
def setup_nltk() -> None:
    import nltk

    NLTK_DIR.mkdir(parents=True, exist_ok=True)
    if str(NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DIR))
    for package in NLTK_PACKAGES:
        nltk.download(package, download_dir=str(NLTK_DIR), quiet=True)


def _short(text, n=60) -> str:
    text = str(text).replace("\n", " ⏎ ")
    return text if len(text) <= n else text[:n] + "…"


def _first_change(before: pd.Series, after: pd.Series) -> str:
    changed = before != after
    if not changed.any():
        return ""
    i = changed[changed].index[0]
    a, b = str(before[i]), str(after[i])
    start = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    start = max(0, start - 20)
    prefix = "…" if start else ""
    return f"{prefix}{_short(a[start:], 50)} → {prefix}{_short(b[start:], 50)}"


def build(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], dict]:
    """Clean the raw table (columns text, text_sentiment). Returns (all rows with keep/drop_reason, log, evidence)."""
    setup_nltk()
    log, evidence = [], {}
    df = pd.DataFrame({"id": np.arange(len(raw)), "text_raw": raw["text"].astype(str),
                       "label": raw["text_sentiment"].astype(int)})
    df["label_name"] = df["label"].map(LABEL_NAMES)
    df["keep"], df["drop_reason"] = True, ""

    def kept() -> int:
        return int(df["keep"].sum())

    def record(step, operation, rule, rationale, affected=0, example="", details=""):
        log.append({"step": step, "operation": operation, "rule": rule, "rationale": rationale,
                    "rows_affected": int(affected), "rows_remaining": kept(), "example": example, "details": details})

    def drop(mask: pd.Series, reason: str) -> pd.DataFrame:
        hit = mask & df["keep"]
        df.loc[hit, "keep"] = False
        df.loc[hit, "drop_reason"] = reason
        return df[hit]

    def drop_example(removed: pd.DataFrame) -> str:
        return _short(removed["text_raw"].iloc[0]) if len(removed) else ""

    def changed(before: pd.Series, after: pd.Series) -> tuple[int, str]:
        """Rows still kept whose text changed, and an example of the first change."""
        live = df["keep"]
        return int(((before != after) & live).sum()), _first_change(before[live], after[live])

    assert df["label"].isin(LABEL_NAMES).all(), "unknown label in raw data"
    record("0", "Load raw data", "read the course CSV; id = row number in the raw file; label_name from the dataset description",
           "stable ids let every later file (splits, embeddings, predictions) align on the same tweet", len(df))
    empty = df["text_raw"].str.strip() == ""
    record("1", "Check missing and empty values", "isna() on both columns; whitespace-only text",
           "required by the brief; none found, so nothing is imputed or removed", int(empty.sum()))

    removed = drop(df.duplicated("text_raw", keep="first"), "exact_duplicate")
    record("2", "Remove exact duplicates", "identical raw text; first copy kept (labels of all copies agree)",
           "duplicates inflate counts and leak between train and test", len(removed), drop_example(removed))

    df["text_fixed"] = df["text_raw"].map(fix_encoding)
    record("3", "Repair encoding and HTML entities", "ftfy.fix_text, then html.unescape",
           "double-encoded UTF-8 ('estÃ©s') and '&amp;' are scraping artefacts, not user text",
           *changed(df["text_raw"], df["text_fixed"]))

    removed = drop(df["text_fixed"].map(is_unreadable), "unreadable_encoding")
    record("4", "Remove unreadable tweets", ">= 2 mojibake residues, < 4 readable words and < 50 % Latin letters",
           "non-Latin scripts whose bytes were lost in the source file; no model can read them",
           len(removed), drop_example(removed))

    text = df["text_fixed"].map(lambda s: MENTION.sub("@user", s))
    record("5", "Anonymise mentions", "@handle -> @user (count kept in n_mentions)",
           "handles carry no sentiment, often contain underscores, and are personal data",
           *changed(df["text_fixed"], text))

    restored = text.map(restore_contractions)
    after = restored.map(lambda r: r[0])
    hits = sum((r[1] for r in restored), Counter())
    record("6", "Restore verified contractions", "it__ -> it's, don__ -> don't, you__e -> you're / you've (by the next word)",
           "the source replaced the apostrophe and the next character with '__'; restored text keeps negations",
           *changed(text, after), ", ".join(f"{k}: {v}" for k, v in hits.most_common()))
    evidence["contractions"] = dict(hits)
    text = after

    stripped = text.map(strip_lost_characters)
    after = stripped.map(lambda r: r[0])
    df["emoji_lost"] = (stripped.map(lambda r: r[1]) > 0).astype(int)
    lost = df["emoji_lost"].astype(bool) & df["keep"]
    record("7", "Remove lost characters", "delete '_' runs and mojibake residue outside #/@/$ tokens; flag emoji_lost = 1; "
           "merge newlines and repeated spaces",
           "the original emoji and symbols cannot be recovered; a placeholder would be read as text by transformers",
           int(lost.sum()), _first_change(text[lost], after[lost]))
    text = after
    df["text_hashtagged"] = text

    tagger = LanguageTagger()
    df["lang"] = df["text_hashtagged"].map(tagger)
    non_en = ~df["lang"].isin(["en", "short"])
    mix = df.loc[non_en & df["keep"], "label_name"].value_counts().to_dict()
    removed = drop(non_en, "non_english")
    record("8a", "Remove non-English tweets (rules)",
           "non-Latin script > 30 % of letters; OR >= 2 Hinglish function words (>= 15 % of words); OR >= 3 "
           "stopwords of one other language and more than twice the English stopwords; OR langdetect non-English "
           "with p >= 0.80 and < 50 % of words found in an English word list",
           "VADER only reads English, so these labels are artefacts", len(removed), drop_example(removed), f"label mix: {mix}")
    manual = df["id"].isin(MANUAL_NON_ENGLISH)
    df.loc[manual & df["keep"], "lang"] = "manual_non_en"
    mix = df.loc[manual & df["keep"], "label_name"].value_counts().to_dict()
    removed = drop(manual, "non_english_manual")
    record("8b", "Remove non-English tweets (manual list)", f"{len(MANUAL_NON_ENGLISH)} ids listed in clean_v2.py, each read in full",
           "language detection is unreliable on short or romanised text", len(removed), drop_example(removed), f"label mix: {mix}")
    evidence["non_english_mix"] = df.loc[df["drop_reason"].str.startswith("non_english"), "label_name"].value_counts().to_dict()

    no_words = df["text_hashtagged"].map(body_words) == 0
    mix = df.loc[no_words & df["keep"], "label_name"].value_counts().to_dict()
    removed = drop(no_words, "no_content")
    record("9", "Remove tweets without words", "no word of >= 2 letters once hashtags, mentions and cashtags are removed",
           "nothing to classify; these tweets are Neutral by default, which would teach a 'no text = Neutral' shortcut",
           len(removed), drop_example(removed), f"label mix: {mix}")
    evidence["no_content_mix"] = mix

    df["hashtags"] = df["text_hashtagged"].str.findall(HASHTAG).map(lambda tags: " ".join(t.lower() for t in tags))
    df["text_clean"] = df["text_hashtagged"].map(normalize_hashtags)
    record("10", "Normalise hashtags", "'#StockMarketCrash' -> 'StockMarketCrash' (no word splitting); stray '#' removed; list kept in hashtags",
           "VADER scored each hashtag as one unknown token; splitting would add sentiment the labeller never saw",
           *changed(df["text_hashtagged"], df["text_clean"]))

    key = df["text_clean"].map(content_key)
    live = df["keep"]
    conflict_keys = set(df[live].groupby(key[live])["label"].nunique().loc[lambda s: s > 1].index)
    removed_conflict = drop(key.isin(conflict_keys), "duplicate_label_conflict")
    live = df["keep"]
    removed = drop(live & key.where(live).duplicated(keep="first") & key.notna(), "content_duplicate")
    record("11", "Remove content duplicates", "same text after lowercasing and removing @user, punctuation and lost characters; first copy kept",
           "copies that differ only by tagged account, emoji or punctuation", len(removed) + len(removed_conflict),
           drop_example(removed), f"{len(removed_conflict)} rows in groups with conflicting labels")

    builder = NormBuilder()
    df["text_norm"] = df["text_clean"].map(builder)
    i = df.index[df["keep"] & df["text_clean"].str.contains("n't")][0]
    record("12", "Build text_norm", "expand contractions; lowercase; letters only; POS-aware WordNet lemmatisation; "
           "NLTK stopwords removed except negations and direction / intensity words",
           "bag-of-words input for TF-IDF and FastText; negations and up/down/below flip the meaning of market tweets",
           kept(), f"{_short(df.at[i, 'text_clean'], 50)} → {_short(df.at[i, 'text_norm'], 50)}")
    removed = drop(df["text_norm"].str.strip() == "", "empty_text_norm")
    record("13", "Remove empty text_norm", "nothing left after normalisation", "no input for bag-of-words models",
           len(removed), drop_example(removed))

    # Groups: duplicates join their first copy; near-duplicates (Jaccard) and identical text_norm join each other.
    uf = UnionFind(df["id"])
    for key_series in (df["text_raw"], key):
        first = {}
        for rid, k in zip(df["id"], key_series):
            if not k:  # an empty key would chain unrelated tweets together
                continue
            if k in first:
                uf.union(first[k], rid)
            else:
                first[k] = rid
    readable = df["drop_reason"] != "unreadable_encoding"
    for a, b in jaccard_pairs(df.loc[readable, "id"], key[readable]):
        uf.union(a, b)
    by_norm = {}
    for rid, norm in zip(df.loc[readable, "id"], df.loc[readable, "text_norm"]):
        if norm.strip():
            by_norm.setdefault(norm, []).append(rid)
    for members in by_norm.values():
        for rid in members[1:]:
            uf.union(members[0], rid)
    df["group_id"] = df["id"].map(uf.find)
    sizes = df.loc[df["keep"], "group_id"].value_counts()
    grouped = df["keep"] & df["group_id"].map(sizes).gt(1)
    record("14", "Group related tweets", "union of: token Jaccard >= 0.7 (>= 5 tokens), identical text_norm, and "
           "duplicates with their first copy; group_id = smallest raw id in the group",
           "02_splits keeps a group inside one fold, so templates such as 'Candle of day … closed' never straddle train and test",
           int(grouped.sum()), "", f"{df.loc[grouped, 'group_id'].nunique()} groups among retained tweets")
    evidence["groups"] = (int(grouped.sum()), int(df.loc[grouped, "group_id"].nunique()))

    add_features(df)
    record("15", "Engineer features", "lengths, hashtag / mention / cashtag counts, ! and ?, caps_ratio, hashtag_ratio, "
           "is_hashtag_spam (>= 0.6), domain_crypto / india / us from hashtags and cashtags",
           "signals that normalisation removes, for XGBoost and EDA; VADER is not a feature (label leakage)", kept())

    evidence["vader"] = _vader_evidence(df)
    evidence["keep_words"] = {w: int(df.loc[df["keep"], "text_norm"].str.contains(rf"\b{w}\b").sum())
                              for w in ("not", "no", "never", "up", "down", "below", "more")}
    record("F", "Final dataset", "English, readable, non-duplicate tweets that contain words",
           "label distribution: " + str(df.loc[df["keep"], "label_name"].value_counts().to_dict()))
    return df, log, evidence


def _vader_evidence(df: pd.DataFrame) -> dict:
    """Agreement between VADER (+/-0.05) and the given labels for different versions of the retained text.

    Analysis only: shows which text the labeller saw and why hashtags are not split.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()

    def hits(texts: pd.Series) -> pd.Series:
        scores = texts.map(lambda s: sia.polarity_scores(s)["compound"])
        pred = np.where(scores >= 0.05, 1, np.where(scores <= -0.05, 2, 0))
        return pd.Series(pred == df.loc[texts.index, "label"].to_numpy(), index=texts.index)

    kept = df[df["keep"]]
    camel = kept["text_clean"].map(lambda s: re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s))
    split_crash = kept["text_clean"].str.replace(r"(?i)stockmarketcrash", "stock market crash", regex=True)
    raw_hit, clean_hit = hits(kept["text_raw"]), hits(kept["text_clean"])
    gap = raw_hit & ~clean_hit
    polar_foreign = df[df["drop_reason"].str.startswith("non_english") & (df["label"] != 0)]
    return {"agreement": {"raw text": round(float(raw_hit.mean()), 3),
                          "text_clean (hashtags kept whole)": round(float(clean_hit.mean()), 3),
                          "hashtags split on CamelCase (clean_v1 style)": round(float(hits(camel).mean()), 3),
                          "'stockmarketcrash' split into words": round(float(hits(split_crash).mean()), 3)},
            "gap_rows": int(gap.sum()),
            "gap_with_contraction": int((gap & kept["text_raw"].str.contains(r"[A-Za-z]__")).sum()),
            "non_english_polar": (len(polar_foreign), int(hits(polar_foreign["text_raw"]).sum()))}


# --------------------------------------------------------------------------------------------------------------
# Outputs
# --------------------------------------------------------------------------------------------------------------
DICTIONARY = [
    ("id", "int", "Row number in data/raw/stock_market_crash_2022.csv; never renumbered", "all files align on it"),
    ("group_id", "int", "Related tweets (duplicates, near-duplicates, identical text_norm) share one id", "4: 02_splits"),
    ("label", "int", "Given label: 0 Neutral, 1 Positive, 2 Negative", "all models"),
    ("label_name", "str", "Label as text", "EDA, charts"),
    ("text_raw", "str", "Original tweet, unchanged", "EDA, error analysis"),
    ("text_clean", "str", "Encoding repaired, @user, contractions restored, lost characters removed, '#' removed "
     "(hashtags not split); case and punctuation kept", "1: Sentence-BERT; 5: token sequences"),
    ("text_norm", "str", "text_clean lowercased, letters only, lemmatised, stopwords removed except negations "
     "and direction words", "1: TF-IDF baseline; 2: FastText"),
    ("hashtags", "str", "Hashtags of the tweet, lowercase, space-separated", "3: EDA, features"),
    ("lang", "str", "en, or short (too short to detect); other values only appear in clean_v2_all.csv", "EDA"),
    ("emoji_lost", "int", "1 if the source file destroyed characters in this tweet (usually emoji)", "3: features, error analysis"),
    ("char_len / word_len", "int", "Length of text_clean in characters / words", "3: EDA, XGBoost"),
    ("n_hashtags / n_mentions / n_cashtags", "int", "Counts in the repaired text before anonymisation", "3: XGBoost"),
    ("n_exclaim / n_question", "int", "Number of ! and ? in text_clean", "3: XGBoost"),
    ("caps_ratio", "float", "Upper-case share of the letters in text_clean", "3: XGBoost"),
    ("hashtag_ratio / is_hashtag_spam", "float / int", "Hashtags per token; flag when >= 0.6", "3: XGBoost, EDA"),
    ("domain_crypto / domain_india / domain_us", "int", "Market mentioned in hashtags or cashtags", "3: business insight"),
]


def _md_table(rows: list[list], header: list[str]) -> list[str]:
    def cell(v) -> str:
        return str(v).replace("|", r"\|").replace("\n", " ")

    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]


def render_log(df: pd.DataFrame, log: list[dict], evidence: dict) -> str:
    kept = df[df["keep"]]
    before = df["label_name"].value_counts()
    after = kept["label_name"].value_counts()
    reasons = df.loc[~df["keep"], "drop_reason"].value_counts()
    lines = [f"# {VERSION} cleaning log", "",
             "Produced by `data/clean_v2/clean_v2.py` from `data/raw/stock_market_crash_2022.csv`. "
             "Re-running the script reproduces these numbers exactly.", "",
             "## Summary", "",
             f"- Raw tweets: {len(df):,}; retained: **{len(kept):,}**; removed: {int((~df['keep']).sum())}.",
             "- Removed rows stay in `clean_v2_all.csv` with `keep = 0` and a `drop_reason`, so every removal can be "
             "audited or reversed in an ablation experiment.", ""]
    lines += _md_table([[name, int(before.get(name, 0)), int(after.get(name, 0)), f"{after.get(name, 0) / len(kept):.1%}"]
                        for name in ("Positive", "Neutral", "Negative")],
                       ["Label", "Raw", "Retained", "Retained share"])
    lines += ["", f"Largest / smallest class after cleaning: {after.max() / after.min():.2f}.", "",
              "## Steps", ""]
    lines += _md_table([[r["step"], r["operation"], r["rule"], r["rationale"], r["rows_affected"], r["rows_remaining"],
                         r["example"], r["details"]] for r in log],
                       ["Step", "Operation", "Rule", "Rationale", "Rows affected", "Rows remaining", "Example", "Details"])
    lines += ["", "## Removed rows", ""]
    lines += _md_table([[reason, int(n), "; ".join(_short(t, 45) for t in df.loc[df["drop_reason"] == reason, "text_raw"].head(2))]
                        for reason, n in reasons.items()], ["Reason", "Rows", "Examples"])
    lines += ["", "## Columns of clean_v2.csv", ""]
    lines += _md_table([list(r) for r in DICTIONARY], ["Column", "Type", "Meaning", "Used by (member)"])
    lines += ["", "`clean_v2_all.csv` has the same columns plus `keep` (1/0) and `drop_reason`, for all 1,698 raw tweets.", "",
              "## Evidence behind the design decisions", "",
              "**Hashtags are not split.** Agreement between VADER and the given labels on the retained tweets:", ""]
    vader = evidence["vader"]
    lines += _md_table([[k, f"{v:.1%}"] for k, v in vader["agreement"].items()], ["Text given to VADER", "Agreement"])
    lines += ["", "Splitting hashtags adds words such as *crash* that the labeller never saw, so agreement drops. "
              f"The small gap between raw text and text_clean comes mostly from restored contractions: of the "
              f"{vader['gap_rows']} tweets where VADER matches the label on the raw text but not on text_clean, "
              f"{vader['gap_with_contraction']} contain a restored contraction. The labeller saw 'shouldn__ worry' "
              "(no negation, so Negative); the cleaned text says 'shouldn't worry'. These tweets are label noise "
              "caused by the corrupted source text, not cleaning errors.", "",
              f"**Non-English tweets are removed.** Label mix of the removed non-English tweets: {evidence['non_english_mix']}. "
              "VADER scores text it cannot read as 0, which becomes Neutral. Of the "
              f"{vader['non_english_polar'][0]} removed non-English tweets labelled Positive or Negative, "
              f"{vader['non_english_polar'][1]} get the same label from VADER on the raw text: their sentiment comes "
              "from words VADER reads as English, such as Spanish *sin* ('without'), German *die* ('the'), Italian "
              "*dire* ('to say') and Romanian *care* ('which').", "",
              f"**Tweets without words are removed.** Label mix: {evidence['no_content_mix']}.", "",
              "**Negations and direction words are kept in text_norm.** Retained tweets whose text_norm contains: "
              + ", ".join(f"*{w}* {n}" for w, n in evidence["keep_words"].items()) + ". "
              "Standard NLTK stopword lists would delete all of them.", "",
              f"**Template tweets are grouped.** {evidence['groups'][0]} retained tweets fall into {evidence['groups'][1]} groups "
              "(near-duplicates or identical text_norm, e.g. 'Candle of day … closed'); a group must stay inside one fold.", "",
              "VADER is used only for this analysis and must never be a model feature.", "",
              "## Changes from clean_v1", "",
              "- Columns renamed: `text_light` → `text_clean`, `text_full` → `text_norm`; added `label_name`, `hashtags`, "
              "`emoji_lost` and 13 feature columns; dropped `n_restored` and `n_unk`.",
              "- Hashtags are no longer split on CamelCase.",
              "- Lost characters are removed and flagged instead of being replaced by `[UNK]`.",
              "- Non-English tweets and tweets without words are removed (they stay in `clean_v2_all.csv`).",
              "- Direction and intensity words (up, down, below, more …) survive stopword removal.",
              "- `group_id` also joins tweets with identical `text_norm`, which fixes template tweets crossing train and test.",
              "- The unreadable-tweet rule also requires < 50 % Latin letters, so 'Rug pull 100% Ahead' is kept.", "",
              "## Reproduce", "", "```bash", "conda activate dsai4205", "python data/clean_v2/clean_v2.py", "```", ""]
    return "\n".join(lines)


def main() -> None:
    raw = pd.read_csv(RAW_CSV, encoding="utf-8")
    df, log, evidence = build(raw)
    full = df.assign(keep=df["keep"].astype(int))[["keep", "drop_reason"] + COLUMNS]
    full.to_csv(HERE / f"{VERSION}_all.csv", index=False, encoding="utf-8", lineterminator="\n")
    full.loc[full["keep"] == 1, COLUMNS].to_csv(HERE / f"{VERSION}.csv", index=False, encoding="utf-8", lineterminator="\n")
    (HERE / f"{VERSION}_log.md").write_text(render_log(df, log, evidence), encoding="utf-8", newline="\n")
    summary = pd.DataFrame(log)[["step", "operation", "rows_affected", "rows_remaining"]]
    print(summary.to_string(index=False))
    print(f"\nretained {int(df['keep'].sum())} of {len(df)} tweets -> {HERE}")


if __name__ == "__main__":
    main()
