# Report draft: limitations and future work (member 2)

Draft for section 5.5 of the final report. Numbers come from `data/clean_v2/clean_v2_log.md`, `notebooks/04_analysis/31_label_noise_errors.ipynb` and `splits/folds_v1.csv`. Before submission, add any limitation that the final results reveal (for example, if one class stays much harder for every model).

---

## 5.5 Limitations and future work

**Labels.** The given labels agree closely with the VADER sentiment tool (93.0% on the raw text), but the original labelling process is unknown. If the labels were produced this way, our models learn to imitate a general-purpose sentiment lexicon rather than human judgement or market sentiment. VADER was not built for finance: it gives no score at all to market words such as *bearish*, *bullish*, *short* or *dip*, so a model can score well on these labels and still misjudge what an investor would mean. In 112 of the 1,525 retained tweets (7.3%) VADER disagrees with the given label; these labels are uncertain. Tweets that lost characters in the file disagree more often (9.8% against 6.5%), which suggests that the labels were computed before the emoji were destroyed.

**Data quality.** The source file damaged emoji and curly apostrophes. We restored only the apostrophe patterns we could verify and removed the rest, but 397 retained tweets (26.0%) lost characters, and emoji often carry sentiment. Restoring contractions also changes some tweets compared with the damaged text in the file: in 13 tweets, VADER matches the label on the damaged text but not after a contraction such as *shouldn't* is restored.

**Size and evaluation.** After cleaning, 1,525 tweets remain; each training fold has fewer than 1,000 tweets and the test split has 303. An accuracy measured on 303 tweets has a 95% confidence interval of about ±5.5 percentage points, so differences of a few points between models on the test split should not be over-interpreted. The small corpus also limits embeddings trained from scratch: FastText learned many neighbours that only share spelling (*dip* and *sip*).

**Generalisation.** The tweets come from one year (2022) and were collected through crash-related hashtags, so their topics centre on market falls. Crypto hashtags or tickers appear in 31.5% of the retained tweets and Indian ones in 13.5%, against 7.3% for the US market. We set aside 102 non-English tweets, so the current results apply only to tweets with English text. Results may not hold for other periods, other markets or neutral market news.

**Privacy and bias.** The tweets are public, but they can contain names and opinions about people. We replaced every user mention with `@user` and keep the data in a private repository used only for this course. Because the labels follow a general English lexicon, the models also inherit its blind spots, for example for romanised Hindi mixed with English, which we kept in six tweets.

**Future work, in order of priority.**

1. Label a random sample of about 200 tweets by hand, with two people per tweet, to measure how often the given labels match human judgement. Every other result depends on this.
2. Obtain a copy of the tweets with emoji and apostrophes intact, and compare the models on the repaired text.
3. Collect more tweets, from other periods and other market hashtags, to test whether the results generalise.
4. Try a finance-specific embedding such as FinBERT (Araci, 2019), if the brief allows it, to see whether financial context reduces the errors on market vocabulary.

## Reference used here

- Araci, D. (2019). FinBERT: Financial sentiment analysis with pre-trained language models. arXiv:1908.10063.
