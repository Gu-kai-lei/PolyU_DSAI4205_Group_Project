"""clean_v1 的清洗流程（已冻结，只为复现 data/clean_v1/clean_v1.csv 而保留）。

当前版本 clean_v2 的清洗脚本是 data/clean_v2/clean_v2.py，列名和规则都有变化，新代码不要再调用这里的函数。

clean_v1 生成两套文本：
- text_light：只修复编码和格式问题，保留大小写、标点和占位符 [UNK]，给 Transformer（SBERT 等）用
- text_full：在 text_light 基础上展开缩写、小写、去标点和数字、词形还原、去停用词（保留否定词），
  给 TF-IDF baseline 和 FastText 用

函数一览：
- run_cleaning(raw)              完整清洗流程：按顺序执行下面各步，返回 (清洗后数据, 每一步的日志, 被删除的行及原因)
- fix_encoding(text)             用 ftfy 修复双重编码乱码，并还原 HTML 实体（&amp; → &）
- replace_mentions(text)         @用户名 → @user
- restore_contractions(text)     还原人工核对过的缩写模式（如 don__ → don't），返回 (文本, 各规则命中次数)
- mark_unknown(text)             其余无法还原的下划线串和乱码残留 → 占位符 [UNK]，返回 (文本, 占位符个数)
- is_unreadable(text)            判断是否为无法修复的乱码推文（原始字节已丢失，几乎没有可读单词）
- normalize_hashtags(text)       去掉 #，按驼峰拆词（#StockMarketCrash → Stock Market Crash）
- normalize_whitespace(text)     换行和连续空白合并成一个空格
- content_key(text)              判断"内容相同"用的键：小写、去 @user 和占位符、只保留字母和数字
- detect_language(text)          识别语言（去掉 hashtag 和 @user 后判断，文字太少返回 "und"）
- FullTextBuilder()(text_light)  从 text_light 生成 text_full
- near_duplicate_groups(...)     按 token 的 Jaccard 相似度把近似重复的推文分到同一组，返回 group_id

维护人：2 号。
"""
from __future__ import annotations

import html
import re
import warnings
from collections import Counter

import pandas as pd

from .config import PROJECT_ROOT

UNK = "[UNK]"  # 无法还原的字符（emoji、符号或所有格 's）的占位符；text_full 里会变成 unk
V1_COLUMNS = ["id", "group_id", "text_raw", "text_full", "text_light", "label"]  # clean_v1 的列，和当前接口不同
NLTK_DIR = PROJECT_ROOT / "data" / "nltk_data"  # NLTK 语料下载到仓库内（data/ 不进 Git）
NLTK_PACKAGES = ("stopwords", "wordnet", "omw-1.4", "averaged_perceptron_tagger_eng")

# ---------------------------------------------------------------------------
# 1. 编码和格式
# ---------------------------------------------------------------------------


def fix_encoding(text: str) -> str:
    """修复双重编码乱码（如 estÃ©s → estés），并还原 HTML 实体（&amp; → &，&gt; → >）。"""
    import ftfy

    return html.unescape(ftfy.fix_text(text))


def replace_mentions(text: str) -> str:
    """@用户名 → @user。用户名本身不带情感信息，而且里面常有下划线，要在处理下划线之前替换。"""
    return re.sub(r"@\w+", "@user", text)


# ---------------------------------------------------------------------------
# 2. 下划线：原始文件里弯引号 ’ 连同它后面的一个字符被替换成了 "__"
#    （it’s → it__，don’t → don__，you’re → you__e）。下面只还原逐条核对过的模式。
# ---------------------------------------------------------------------------

NEGATION_STEMS = ("don", "doesn", "didn", "isn", "aren", "wasn", "weren", "hasn", "haven", "hadn",
                  "couldn", "shouldn", "wouldn", "won", "can", "ain", "mustn", "needn")
S_STEMS = ("it", "that", "what", "there", "let", "he", "she", "who", "how", "here", "where")
LL_STEMS = ("i", "you", "we", "they", "he", "she", "it", "that", "there")
# you__e / we__e / they__e 既可能是 're 也可能是 've：后面跟过去分词时是 've
PARTICIPLES = {"been", "got", "gotten", "lost", "seen", "done", "had", "made", "broken", "taken", "given",
               "known", "become", "gone", "bought", "held", "paid", "felt", "heard", "left", "told",
               "thought", "kept", "begun", "fallen", "grown", "shown", "written", "chosen", "missed"}
SKIPPABLE_ADVERBS = {"already", "just", "never", "ever", "finally", "all", "really", "also", "probably", "definitely"}

_W = r"(?=[^A-Za-z0-9_]|$)"  # 词尾：后面不是字母、数字或下划线
CONTRACTION_RULES = [
    ("n't", re.compile(rf"\b({'|'.join(NEGATION_STEMS)})__{_W}", re.I), lambda m: m.group(1) + "'t"),
    ("I'm", re.compile(rf"\b(i)__{_W}", re.I), lambda m: m.group(1) + "'m"),
    ("'s", re.compile(rf"\b({'|'.join(S_STEMS)})__{_W}", re.I), lambda m: m.group(1) + "'s"),
    ("'ll", re.compile(rf"\b({'|'.join(LL_STEMS)})__l{_W}", re.I), lambda m: m.group(1) + "'ll"),
    ("I've", re.compile(rf"\b(i)__e{_W}", re.I), lambda m: m.group(1) + "'ve"),
    ("y'all", re.compile(rf"\b(y)__ll{_W}", re.I), lambda m: m.group(1) + "'all"),
]
RE_VE_RULE = re.compile(r"\b(you|we|they)__e\b", re.I)


def _re_or_ve(match: re.Match) -> str:
    """you__e / we__e / they__e：跳过 already、just 等副词后，下一个词是过去分词就还原成 've，否则 're。"""
    following = re.findall(r"[A-Za-z]+", match.string[match.end():match.end() + 60])
    words = [w.lower() for w in following]
    while words and words[0] in SKIPPABLE_ADVERBS:
        words.pop(0)
    nxt = words[0] if words else ""
    is_participle = nxt in PARTICIPLES or (len(nxt) > 3 and nxt.endswith("ed"))
    return match.group(1) + ("'ve" if is_participle else "'re")


def restore_contractions(text: str) -> tuple[str, Counter]:
    """还原核对过的缩写，返回 (还原后的文本, 每条规则的命中次数)。"""
    hits = Counter()
    for name, pattern, repl in CONTRACTION_RULES:
        text, n = pattern.subn(repl, text)
        hits[name] += n
    text, n = RE_VE_RULE.subn(_re_or_ve, text)
    hits["'re/'ve"] += n
    return text, hits


# ftfy 修不好的乱码残留：连续的 Latin-1 补充字符（U+0080–U+00FF），里面含 C1 控制字符，
# 或者长度 ≥ 2 且含符号（如 à¤、ï¸、¤ï、¦ð）。单独一个 é、«、¿、£ 是正常字符，不算残留。
_LATIN1_RUN = re.compile("[\u0080-ÿ]+")


def _is_residue(run: str) -> bool:
    has_c1 = any("\u0080" <= ch <= "\u009f" for ch in run)
    has_symbol = any(" " <= ch <= "¿" for ch in run)
    return has_c1 or (len(run) >= 2 and has_symbol)


def count_residue(text: str) -> int:
    """数一数文本里有几处乱码残留。"""
    return sum(_is_residue(m.group()) for m in _LATIN1_RUN.finditer(text))


def mark_unknown(text: str) -> tuple[str, int]:
    """把还原之后剩下的下划线串和乱码残留换成占位符 [UNK]，连续的占位符合并成一个，返回 (文本, 替换次数)。

    单词中间的单个下划线（#Bear_Market）是真实字符，只替换成空格。
    """
    text = re.sub(r"(?<=[A-Za-z0-9])_(?=[A-Za-z0-9])", " ", text)
    text, n_underscore = re.subn(r"_+", f" {UNK} ", text)
    n_residue = count_residue(text)
    text = _LATIN1_RUN.sub(lambda m: f" {UNK} " if _is_residue(m.group()) else m.group(), text)
    text = re.sub(rf"(?:\s*{re.escape(UNK)})+", f" {UNK}", text)
    return text, n_underscore + n_residue


def is_unreadable(text: str, min_residue: int = 2, min_words: int = 4) -> bool:
    """修复编码后仍有 ≥ min_residue 处乱码残留、且可读的拉丁字母单词少于 min_words 个，就判为不可读。

    这类推文（多为印地语、马拉雅拉姆语、韩语）的原始字节在数据集里已经丢失，任何模型都读不懂。
    text 应该是 fix_encoding() 之后、mark_unknown() 之前的文本。
    """
    residue = count_residue(text)
    body = re.sub(r"[#@]\w+", " ", text)
    words = re.findall(r"\b[A-Za-z]{2,}\b", body)
    return residue >= min_residue and len(words) < min_words


# ---------------------------------------------------------------------------
# 3. hashtag 和空白
# ---------------------------------------------------------------------------


def normalize_hashtags(text: str) -> str:
    """去掉 #，按驼峰拆词：#StockMarketCrash → Stock Market Crash。全小写的 #bearmarket 保持原样。

    前面补一个空格，避免和紧挨着的文字粘在一起（nope!#Tag → nope! Tag）。
    """
    def split(match: re.Match) -> str:
        return " " + re.sub(r"(?<=[a-z])(?=[A-Z])", " ", match.group(1))

    return re.sub(r"#(\w+)", split, text)


def normalize_whitespace(text: str) -> str:
    """换行、制表符和连续空白合并成一个空格，并去掉首尾空白。"""
    return " ".join(text.split())


def content_key(text: str) -> str:
    """判断"内容相同"用的键：小写，去掉 @user 和 [UNK]，只保留字母和数字。"""
    text = text.lower().replace("@user", " ").replace(UNK.lower(), " ")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())


# ---------------------------------------------------------------------------
# 4. 语言识别
# ---------------------------------------------------------------------------


def detect_language(text: str, min_letters: int = 15) -> str:
    """识别正文语言。先去掉 hashtag、@user 和 [UNK]，剩下的字母少于 min_letters 时返回 "und"（无法判断）。"""
    from langdetect import DetectorFactory, LangDetectException, detect

    DetectorFactory.seed = 0  # langdetect 默认带随机性，固定种子保证结果可复现
    body = re.sub(r"#\w+|@user|\[UNK\]", " ", text)
    if sum(ch.isalpha() for ch in body) < min_letters:
        return "und"
    try:
        return detect(body)
    except LangDetectException:
        return "und"
    except TypeError as err:  # langdetect 偶发的内部错误，不让整个流程中断，但要提示出来
        warnings.warn(f"langdetect 内部错误，按 und 处理：{err!r}，文本：{body[:60]!r}")
        return "und"


# ---------------------------------------------------------------------------
# 5. text_full：展开缩写 → 分词 → 词性标注 → 词形还原 → 去停用词
# ---------------------------------------------------------------------------

NEGATIONS = {"no", "not", "nor", "never", "none", "nothing", "nobody", "neither", "nowhere", "cannot"}
_CONTRACTION_EXPANSIONS = [
    (r"\bcan't\b", "can not"), (r"\bwon't\b", "will not"), (r"\bain't\b", "is not"),
    (r"n't\b", " not"), (r"'re\b", " are"), (r"'m\b", " am"), (r"'ll\b", " will"),
    (r"'ve\b", " have"), (r"'d\b", " would"), (r"'s\b", ""),
]


def setup_nltk() -> None:
    """把 make_full() 需要的 NLTK 语料下载到 data/nltk_data（已下载过就跳过）。"""
    import nltk

    NLTK_DIR.mkdir(parents=True, exist_ok=True)
    if str(NLTK_DIR) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DIR))
    for package in NLTK_PACKAGES:
        nltk.download(package, download_dir=str(NLTK_DIR), quiet=True)


def _wordnet_pos(tag: str) -> str:
    return {"J": "a", "V": "v", "R": "r"}.get(tag[:1], "n")


class FullTextBuilder:
    """生成 text_full。初始化时加载停用词和词形还原器，之后对每条文本调用实例本身。"""

    def __init__(self):
        from nltk import pos_tag
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer

        setup_nltk()
        self._pos_tag = pos_tag
        self._lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words("english")) - NEGATIONS

    def __call__(self, text_light: str) -> str:
        text = text_light.replace("’", "'").lower()
        text = text.replace("@user", " ").replace(UNK.lower(), " unk ")
        for pattern, repl in _CONTRACTION_EXPANSIONS:
            text = re.sub(pattern, repl, text)
        tokens = re.findall(r"[a-z]+", text)  # 只保留字母：去掉标点、数字和其他文字
        if not tokens:
            return ""
        lemmas = [self._lemmatizer.lemmatize(tok, _wordnet_pos(tag)) for tok, tag in self._pos_tag(tokens)]
        return " ".join(w for w in lemmas if len(w) > 1 and w not in self.stopwords)


# ---------------------------------------------------------------------------
# 6. 近似重复分组
# ---------------------------------------------------------------------------


def near_duplicate_groups(ids, keys, threshold: float = 0.7, min_tokens: int = 5) -> dict:
    """把 token 集合 Jaccard 相似度 ≥ threshold 的推文连成一组（并查集），返回 {id: group_id}。

    group_id 取组内最小的 id；没有近似重复的推文自成一组。少于 min_tokens 个 token 的短文本不参与比较，
    避免 "#bearmarket" 这类极短推文被误判。
    """
    ids = list(ids)
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    token_sets = [set(k.split()) for k in keys]
    candidates = [(i, s) for i, s in zip(ids, token_sets) if len(s) >= min_tokens]
    for a in range(len(candidates)):
        id_a, set_a = candidates[a]
        for b in range(a + 1, len(candidates)):
            id_b, set_b = candidates[b]
            inter = len(set_a & set_b)
            if inter and inter / len(set_a | set_b) >= threshold:
                parent[find(id_b)] = find(id_a)

    groups = {}
    for i in ids:
        groups.setdefault(find(i), []).append(i)
    return {i: min(members) for members in groups.values() for i in members}


# ---------------------------------------------------------------------------
# 7. 完整流程：每一步记一条日志，被删除的行连同原因一起返回
# ---------------------------------------------------------------------------


def _short(text: str, n: int = 70) -> str:
    text = str(text).replace("\n", " ⏎ ")
    return text if len(text) <= n else text[:n] + "…"


def run_cleaning(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """完整清洗流程。raw 是 src.data.load_raw() 的输出（id、text、label 三列）。

    返回：
    - clean：clean_v1 格式的清洗数据（V1_COLUMNS），另附 lang、n_restored、n_unk 三列
    - log：每一步一条记录（step、operation、rationale、rows_affected、rows_remaining、example、details）
    - dropped：被删除的行（id、text_raw、label、reason）
    """
    log, dropped = [], []
    df = raw[["id", "text", "label"]].rename(columns={"text": "text_raw"}).copy()

    def record(operation, rationale, affected=0, example="", details=""):
        log.append({"step": len(log), "operation": operation, "rationale": rationale,
                    "rows_affected": int(affected), "rows_remaining": len(df),
                    "example": example, "details": details})

    def change_example(before: pd.Series, after: pd.Series) -> tuple[int, str]:
        """返回 (改动的行数, 第一处改动附近的前后对比)。"""
        changed = before != after
        if not changed.any():
            return 0, ""
        i = changed[changed].index[0]
        a, b = before[i], after[i]
        start = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
        start = max(0, start - 20)
        prefix = "…" if start else ""
        return int(changed.sum()), f"{prefix}{_short(a[start:], 55)} → {prefix}{_short(b[start:], 55)}"

    def drop(mask: pd.Series, reason: str) -> pd.DataFrame:
        nonlocal df
        removed = df[mask]
        dropped.append(removed[["id", "text_raw", "label"]].assign(reason=reason))
        df = df[~mask].copy()
        return removed

    record("Load raw data", "Original CSV; one row is one tweet; id = row number in the raw file")

    removed = drop(df.duplicated("text_raw", keep="first"), "exact duplicate")
    record("Remove exact duplicates", "The same tweet text appears more than once (labels agree); first copy kept",
           len(removed), _short(removed["text_raw"].iloc[0]) if len(removed) else "")

    df["fixed"] = df["text_raw"].map(fix_encoding)
    n, ex = change_example(df["text_raw"], df["fixed"])
    record("Repair encoding and HTML entities",
           "Double-encoded UTF-8 repaired with ftfy (estÃ©s → estés); HTML entities unescaped (&amp; → &)", n, ex)

    df["text"] = df["fixed"].map(replace_mentions)
    n, ex = change_example(df["fixed"], df["text"])
    record("Replace user mentions", "@username → @user: names carry no sentiment and often contain underscores", n, ex)

    before = df["text"]
    restored = before.map(restore_contractions)
    df["text"] = restored.map(lambda r: r[0])
    df["n_restored"] = restored.map(lambda r: sum(r[1].values()))
    hits = sum((r[1] for r in restored), Counter())
    n, ex = change_example(before, df["text"])
    record("Restore verified contractions",
           "In the source file the apostrophe ’ and the next character became '__' (it’s → it__, you’re → you__e). "
           "Only patterns checked by hand against every occurrence are restored",
           n, ex, ", ".join(f"{k}: {v}" for k, v in hits.most_common()))

    removed = drop(df["fixed"].map(is_unreadable), "unreadable mojibake")
    record("Remove unreadable tweets",
           "Non-Latin scripts whose original bytes were lost in the source file (ftfy cannot repair them): "
           "≥ 2 mojibake residues and < 4 readable words",
           len(removed), _short(removed["text_raw"].iloc[0]) if len(removed) else "")

    before = df["text"]
    marked = before.map(mark_unknown)
    df["text"] = marked.map(lambda r: r[0])
    df["n_unk"] = marked.map(lambda r: r[1])
    n, ex = change_example(before, df["text"])
    record(f"Replace unrecoverable characters with {UNK}",
           "Remaining '__' runs (emoji, symbols or possessive 's, which cannot be told apart) and leftover mojibake",
           n, ex, f"{int(df['n_unk'].sum())} replacements")

    df["lang"] = df["text"].map(detect_language)
    counts = df["lang"].value_counts()
    record("Detect language (flag only)",
           "langdetect on the tweet body without hashtags and mentions; < 15 letters → 'und'. "
           "Readable non-English tweets are kept and flagged in the lang column",
           int((df["lang"] != "en").sum()), "",
           ", ".join(f"{k}: {v}" for k, v in counts.head(8).items()))

    before = df["text"]
    df["text_light"] = before.map(normalize_hashtags).map(normalize_whitespace)
    n, ex = change_example(before, df["text_light"])
    record("Normalise hashtags and whitespace",
           "Remove '#', split CamelCase hashtags (#StockMarketCrash → Stock Market Crash); merge newlines and spaces",
           n, ex)

    key = df["text_light"].map(content_key)
    conflicting = key.map(df.groupby(key)["label"].nunique()) > 1
    removed_conflict = drop(conflicting, "content duplicate with conflicting labels")
    key = key[~conflicting]
    removed = drop(key.duplicated(keep="first"), "content duplicate")
    record("Remove content duplicates",
           "Tweets identical after lowercasing and removing mentions, placeholders and punctuation "
           "(e.g. '@user #bearmarket' vs '#bearmarket'); first copy kept",
           len(removed) + len(removed_conflict), _short(removed["text_raw"].iloc[0]) if len(removed) else "",
           f"{len(removed_conflict)} rows with conflicting labels removed" if len(removed_conflict) else "")

    builder = FullTextBuilder()
    df["text_full"] = df["text_light"].map(builder)
    i = df.index[0]
    record("Build text_full",
           "Expand contractions (don't → do not), lowercase, keep letters only (drops punctuation and digits), "
           "POS-aware WordNet lemmatisation, remove NLTK stopwords except negations (no, not, never…)",
           len(df), f"{_short(df.at[i, 'text_light'])} → {_short(df.at[i, 'text_full'])}")

    no_words = df["text_full"].map(lambda s: not [w for w in s.split() if w != "unk"])
    removed = drop(no_words, "no words left")
    record("Remove tweets with no words left", "Only mentions, emoji or symbols remained; nothing to classify",
           len(removed), _short(removed["text_raw"].iloc[0]) if len(removed) else "")

    keys = df["text_light"].map(content_key)
    groups = near_duplicate_groups(df["id"], keys)
    df["group_id"] = df["id"].map(groups)
    sizes = df["group_id"].map(df["group_id"].value_counts())
    in_groups = int((sizes > 1).sum())
    record("Group near-duplicates",
           "Token Jaccard ≥ 0.7 (tweets with ≥ 5 tokens) on the content key; a group must stay in one fold",
           in_groups, "", f"{df.loc[sizes > 1, 'group_id'].nunique()} groups")

    clean = df[V1_COLUMNS + ["lang", "n_restored", "n_unk"]].reset_index(drop=True)
    record("Final dataset", "Columns: " + ", ".join(clean.columns))
    dropped_rows = pd.concat(dropped, ignore_index=True) if dropped else pd.DataFrame()
    return clean, log, dropped_rows


def log_to_markdown(log: list[dict], title: str = "Data cleaning log", notes: list[str] | None = None) -> str:
    """把 run_cleaning() 的日志转成 Markdown 表格（英文，可以直接放进报告的表 4 和附录），notes 附在表格后面。"""
    def cell(value) -> str:
        return str(value).replace("|", r"\|").replace("\n", " ")

    lines = [f"# {title}", "",
             "| Step | Operation | Rationale | Rows affected | Rows remaining | Example (before → after) | Details |",
             "|---|---|---|---|---|---|---|"]
    for r in log:
        lines.append("| " + " | ".join(cell(r[k]) for k in
                     ("step", "operation", "rationale", "rows_affected", "rows_remaining", "example", "details")) + " |")
    if notes:
        lines += ["", "## Notes", ""] + [f"- {note}" for note in notes]
    return "\n".join(lines) + "\n"
