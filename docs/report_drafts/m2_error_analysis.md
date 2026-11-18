# 错误分析（2 号，11/18）

来源：`notebooks/04_analysis/31_label_noise_errors.ipynb` Part B。用的是各模型在开发集上的交叉验证（OOF）预测，测试集没有动。

## 给 3 号和 6 号的要点

- **范围**：目前有预测的 4 个模型——TextCNN（5 号，0.618）、Baseline（0.582）、kNN（2 号，SBERT，0.539；参数在同一套折上调，嵌套估计 0.522）、MLP（4 号设计，团队环境 FastText 版，0.535）。BiLSTM、SVM、XGBoost 的预测到了以后重跑一次，数字会变。
- **152 条推文（1,222 条中的 12.4%）4 个模型全错**；另一方面，1,070 条（87.6%）至少有一个模型判对，说明模型之间互补，对 1 号的 Stacking 是好消息。
- **错得越多，标签越可疑**：被 0 个模型判错的推文里，VADER 与标签不一致的只占 3.4%；被全部 4 个模型判错的占 13.2%（图 `figures/analysis_errors_vader.png`）。
- **金融用语是第二个原因**：全错的推文里 39.5% 含金融词，其他推文是 26.4%（odds ratio 1.81，p = 0.001）。按单个词看，全错比例最高的是 green（4/11）、red（8/27）、bearish（6/21）、puts（2/7）、bullish（5/21）、fed（7/30），但每个词的样本都很少，结论只看总体检验；crash 和 dip 反而不容易错（VADER 本来就给 crash 打负分）。
- **源文件丢掉的撇号**：*don't panic* 在文件里是 *don__ panic*，词典只看到 *panic*，标签就成了 Negative；我们清洗时已经把它还原成 *don't*，模型读到的是"别慌"，于是和标签对不上。全错的推文里 9.9% 有这种情况，其他推文是 3.6%。另外，在"恢复否定词会改变 VADER 判断"的 14 条推文里，13 条的标签和丢字后的文本一致，只有 1 条和恢复后的一致——这进一步支持"标签是在我们收到的文本上用 VADER 类方法算的"这个假设（可以补进报告 2.4 节）。
- **最常见的错法**：把 Neutral 或 Negative 判成 Positive（152 条里 64 条，42%；另有 15 条是 Positive 和别的标签平票）。
- **商业洞察（给 3 号）**：通用情感词典会误读市场语言——指数名 *Nifty* 被当成褒义形容词，"指数大跌"的行情播报因此被标成 Positive；red/green、bull/bear 这类行话词典不认识；抄底式的乐观（*buy low*）被读成负面；还有反讽。直接用词典信号做投资者情绪面板，会在下跌行情里给出偏乐观的信号。
- **对报告的影响（给 6 号）**：多数难例（87%）并不是 VADER 不一致的样本。一部分难例的标签和 VADER 一致，但和推文本意相反（见下表的 806、1160、1134），这类问题用 VADER 一致率检查不出来。所以我们的分数衡量的是"和这套标签一致的程度"，局限性里要写清楚。

## Draft for report Section 4.3 Error analysis

We compared the out-of-fold predictions of the four models that have them so far (TextCNN, the TF-IDF baseline, kNN on Sentence-BERT and the FastText MLP, rerun in the team environment). 152 of the 1,222 development tweets (12.4%) are misclassified by all four, while 1,070 (87.6%) are classified correctly by at least one model, so the models are complementary. The tweets that every model gets wrong are not random. First, the given label is more often questionable: VADER disagrees with the label for 13.2% of them, against 3.4% of the tweets that no model gets wrong, and the share rises steadily with the number of models that fail. Second, finance terms are over-represented: 39.5% of these tweets contain words such as *red*, *green*, *bearish*, *bullish*, *Fed* or *Nifty*, against 26.4% of the others (odds ratio 1.81, Fisher's exact test p = 0.001). Third, 9.9% contain a negation whose apostrophe was lost in the source file (*don__ panic*), against 3.6% of the others: a lexicon reads such a tweet without the negation, while our cleaned text restores it, so the models and the label disagree. Where restoring the negation changes VADER's label (14 tweets), the given label follows the damaged text in 13 cases, which is consistent with labels computed on the text as distributed. The most common error is predicting Positive for a Neutral or Negative tweet (64 of the 152, and 15 more where Positive ties with another label). Table 8 shows representative cases; several labels agree with VADER but not with what the tweet says, so a high score measures agreement with lexicon-style labels rather than with investors' reading of the market.

**Table 8 Representative classification errors** (predicted = most common prediction of the four models; a tie lists both labels)

| Case | True label | Predicted label | Explanation |
|---|---|---|---|
| 1574 | Negative | Positive | "Don__ panic. Look at the bigger picture." The file lost the apostrophe of *don't*, so a lexicon reads *panic* without the negation (VADER −0.51; +0.40 with *don't* restored). Our cleaning restores *don't*, and the models read the encouragement. |
| 806 | Positive | Negative / Neutral (2 models each) | "Blood bath continues in Indian stock Market … NIFTY: 16,842 -308 … Market tanks over 2000 points." The only lexicon word is *Nifty* (+1.7), so the label is Positive although the tweet reports a fall; two models read it as Negative. |
| 1160 | Positive | Neutral | "@user sad but true #bearmarket." VADER gives more weight to the clause after *but*, so *true* outweighs *sad* (+0.39) and the label is Positive. |
| 1455 | Negative | Positive | "Time to purchase more stock during this #stockmarketcrash … I buy low and sell high." Buying-the-dip optimism. The lexicon scores *yes* (+1.7) and *low* (−1.1); its *but* rule halves the clause before *but* and boosts the one after it, so *low* dominates and the label is Negative (−0.20). |
| 1134 | Positive | Neutral | "Yay #stockmarketcrash." Sarcasm: the label follows *yay* literally, and the tweet is too short for the models. |
| 1649 | Negative | Neutral | "My anxiety level is the only thing up in this bear market." The label is plausible; the models miss a mild sentiment word (*anxiety*) among market words. A genuine model error. |

Remedies for the final phase: treat index names such as *Nifty* as entities rather than sentiment words, mark tweets whose lexicon score flips when a lost negation is restored as uncertain labels, combine the complementary models by stacking, and relabel a small sample of the tweets that every model gets wrong to estimate how much of the remaining error is label noise.
