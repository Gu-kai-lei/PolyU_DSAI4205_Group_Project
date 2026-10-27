# Slides draft: member 2 (2:00–4:00 of the presentation)

Content for two slides and a two-minute script (about 240 words, plus one sentence on the error analysis). Put it into member 5's slide template once it is shared (11/10) and hand it in by 11/19. The last point of slide 2 is filled in after the error analysis (11/17).

---

## Slide 1: Cleaning: 1,698 → 1,525 tweets

Figure: `figures/cleaning_funnel.png`

- Repaired the text: 155 tweets with broken encoding, 166 with restored contractions (*it__* → *it's*)
- Removed 173 tweets: 102 not in English, 39 without any words, 21 unreadable, 11 duplicates
- Kept hashtags whole: splitting *#stockmarketcrash* adds the word *crash*
- Near-duplicate tweets share a `group_id` and always stay in the same fold

## Slide 2: The labels were most likely made by VADER

Figure: `figures/cleaning_label_by_language.png` (or `figures/analysis_vader_vs_labels.png`)

- VADER, a word-list sentiment tool, reproduces 93.0% of the labels; TextBlob only 60.2%
- Non-English tweets are 73% Neutral, because VADER cannot read them
- So our models learn to imitate VADER, and VADER is never used as a feature
- **[After 11/17]** Tweets that every model gets wrong: one finding from the error analysis

## Script

**Slide 1.** "Before modelling, we had to repair the data. The file we received had broken characters: curly apostrophes and the letter after them had turned into two underscores, so *it's* became *it__*. We restored only the patterns we could check by hand, which fixed 166 tweets, and repaired the encoding of 155 more. We then removed 173 tweets: duplicates, tweets we could not read, tweets with no words, and 102 tweets that are not in English. One decision needs a reason: we did not split hashtags into words. You will see why on the next slide."

**Slide 2.** "The dataset description does not say how the tweets were labelled. When we ran VADER, a common rule-based sentiment tool, it gave the same label as the dataset for 93% of the tweets. Another tool, TextBlob, matched only 60%. The non-English tweets show the same pattern: VADER cannot read them, scores them zero, and almost three quarters of them are labelled Neutral. Splitting hashtags would add words such as *crash* that VADER never saw, and the match drops from 92% to 71%. This has three consequences. Our task is really to imitate VADER. VADER's score can never be a model input, because that would leak the answer. And where VADER disagrees with the label, the label itself may be wrong. [Error analysis finding.] Now member 4 will explain how we kept the comparison between models fair."

## Numbers used (for questions)

| Number | Source |
|---|---|
| 1,698 → 1,525 tweets; 173 removed (102 non-English, 39 no words, 21 unreadable, 9 exact + 2 content duplicates) | `data/clean_v2/clean_v2_log.md` |
| 155 encoding repairs, 166 tweets with restored contractions | `clean_v2_log.md`, steps 3 and 6 |
| VADER 93.0% (Cohen's κ 0.89), TextBlob 60.2%, on 1,666 readable tweets | `31_label_noise_errors`, A1 |
| Non-English tweets 73% Neutral | `figures/cleaning_label_by_language.png` |
| Hashtags kept whole 91.9%, *stockmarketcrash* split 71.1% (retained tweets) | `clean_v2_log.md`, evidence section |
