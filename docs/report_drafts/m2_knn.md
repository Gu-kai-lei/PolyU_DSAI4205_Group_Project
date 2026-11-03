# Report draft: kNN model (member 2)

Draft of the kNN paragraph for section 3.4 and its results for section 4.1. Numbers come from `notebooks/03_models/21_m2_knn.ipynb` and `results/m2_knn.csv` (data `clean_v2`, split `folds_v1`). Sentence-BERT results are added after member 5 delivers the vectors (11/8); places to update are marked **[SBERT]**.

---

## For 3.4: Model 2, k-nearest neighbours (kNN)

kNN gives a tweet the majority label of the *k* training tweets whose sentence vectors are most similar to it. It has no real training step, so its score shows how well an embedding alone separates the three classes. This makes it a useful reference point for the other five models, which learn a decision function on top of the same vectors.

**Integration of embedding and classifier.** The embedding and the classifier are two separate steps. The embedding model turns each tweet into a fixed vector without using the labels: our self-trained FastText (mean and SIF vectors, Section 3.3) and **[SBERT]** the frozen Sentence-BERT vectors. The kNN classifier then only compares these vectors. For fold *k* we use the FastText model trained without fold *k*, so validation tweets never influence the vectors used to classify them.

**Settings.** All vectors are scaled to length 1 and compared with cosine distance. We tune two hyperparameters by 5-fold cross-validated macro-F1 on the training split: the number of neighbours *k* (odd values from 3 to 51) and the vote weighting (every neighbour counts the same, or closer neighbours count more). The best setting of each embedding is then refitted on the whole training split to predict the test split for the ensemble; the test split is scored only once, after the experiment freeze.

## For 4.1: kNN results

| Embedding | Best *k* | Weighting | CV accuracy | CV macro-F1 | F1 Neutral | F1 Positive | F1 Negative |
|---|---|---|---|---|---|---|---|
| FastText, mean vectors | 9 | uniform | 0.481 | **0.473** | 0.451 | 0.524 | 0.445 |
| FastText, SIF vectors | 21 | distance | 0.468 | 0.453 | 0.390 | 0.540 | 0.429 |
| **[SBERT]** Sentence-BERT | | | | | | | |

The choice of *k* matters less than the choice of embedding: across all values of *k* and both weightings, mean vectors stay between 0.447 and 0.473 and SIF vectors between 0.412 and 0.453.

**Comparison with the baseline.** On the same folds, the TF-IDF + logistic regression baseline reaches a macro-F1 of 0.582 (member 1; confirm with the official record in `results/baseline.csv`). kNN on FastText vectors is therefore about 0.11 lower. We see three likely reasons:

1. FastText was trained on only about 1,000 tweets per fold, so many nearest neighbours share spelling rather than meaning (*dip* is closest to *sip* and *sit*).
2. Averaging word vectors loses word order and dilutes short but decisive words such as *not*.
3. The labels agree closely with VADER, which scores individual sentiment words. If they were produced this way, a linear model on word counts can copy such a word-level rule more directly than a vote among similar tweets.

The largest errors in the out-of-fold confusion matrix (`figures/m2_knn_confusion.png`) are Negative tweets predicted as Positive (167) and Positive tweets predicted as Negative (143).

**Label-noise check.** We tested whether questionable labels hurt kNN by removing tweets from the training folds only, while the validation folds keep every tweet:

| Training folds | CV macro-F1 |
|---|---|
| all tweets | 0.473 |
| without tweets flagged by confident learning (Northcutt et al., 2021) | 0.461 |
| without the tweets where VADER disagrees with the given label | 0.456 |

Both filters lower the score, so we train on all tweets. Confident learning flags 38.6% of each training part, far more than the 7% of retained tweets on which VADER disagrees with the label, because it trusts the model's own predictions and this model is weak. It flags 73% of the VADER disagreements but also 69% of all other tweets, so with FastText vectors it cannot single out the questionable labels. **[SBERT]** Repeat both checks with Sentence-BERT vectors.

## References used here

- Arora, S., Liang, Y., & Ma, T. (2017). A simple but tough-to-beat baseline for sentence embeddings. *ICLR*.
- Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with subword information. *Transactions of the ACL*, 5, 135–146.
- Northcutt, C., Jiang, L., & Chuang, I. (2021). Confident learning: Estimating uncertainty in dataset labels. *Journal of Artificial Intelligence Research*, 70, 1373–1411.
