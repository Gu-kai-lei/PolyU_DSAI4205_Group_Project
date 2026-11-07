# Member 5 report draft — YU Guanlin

## Section 3.3: Frozen contextual representations

We extracted contextual representations from `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` at revision `4328cf26390c98c5e3c738b4460a05b95f4911f5`. Although the repository name contains MPNet, the multilingual checkpoint uses an XLM-RoBERTa backbone. The encoder was externally pretrained and frozen; no dataset labels or dataset-specific gradient updates were used for representation extraction. We used `text_clean`, preserving case and punctuation. The sequence limit was set to 82 tokenizer positions from the 99th percentile of development-text lengths, capped at the model's default limit of 128. Test text was encoded only by this fixed transformation.

Sentence vectors were obtained by attention-masked mean pooling, including attended special tokens, followed by L2 normalization. Each vector has 768 dimensions. For sequence classifiers, special tokens were removed, remaining contextual token vectors were compacted in order, and sequences were right-padded with zeros. The resulting token cache has shape (1525, 82, 768), stored as float16 with a separate boolean mask and stable IDs. Encoder inference used float32. All exports are aligned by original tweet ID; the manifest records input hashes, encoder revision, pooling and truncation settings, and output checksums. This representation implements the sentence-embedding approach associated with Sentence-BERT (Reimers & Gurevych, 2019), using the specified multilingual checkpoint rather than retraining the original paper's architecture.

## Section 3.4: TextCNN classifier

Following the convolution-and-pooling approach of Kim (2014), the independent TextCNN classifier uses convolution widths 2, 3, 4 and 5, with 100 filters per width, ReLU, max-over-time pooling, dropout 0.5, and a three-logit output layer. Classifier parameters are randomly initialized; the contextual encoder remains frozen. Padding is zeroed before convolution and excluded from pooling. For a nonempty sequence shorter than a kernel, one zero-extended window is allowed. An empty input contributes zero pooled features. Unit checks cover padding invariance and sequences of lengths 0, 1, 2, 4 and 8.

We compare frozen contextual token inputs against the team's independently trained FastText token sequences (100 dimensions). For each outer fold, FastText is fitted only on that fold's training corpus. FastText hyperparameters and tokenization follow `src/fasttext_emb.py`; the sequence limit is 31 words, determined using development texts only. Inner-stop texts are included in the outer-training FastText corpus, so inner selection is transductive with respect to unsupervised representation fitting; outer validation and test text remain excluded from FastText fitting for their corresponding evaluation models.

The official five grouped folds are used without modification. Within each outer training partition, one additional grouped split selects the epoch count using validation Macro-F1, with a 12-epoch ceiling and patience 3. We then reinitialize the classifier and refit it on all outer training rows for the selected number of epochs before producing outer validation probabilities. Optimization uses AdamW, learning rate 0.001, weight decay 0.01, batch size 32 and gradient clipping at 1.0, with seed 42. After cross-validation, the median selected epoch count is used for a full-development refit and test probability export. Test labels are not scored by this implementation.

## Interpretation limits

These are fixed-configuration MVP experiments, not exhaustive hyperparameter tuning. Report pooled outer OOF Macro-F1 as the primary metric, separately from the arithmetic mean and sample standard deviation across folds. Selecting a representation after observing these results can introduce selection optimism. The supplied labels may reflect a lexicon-based annotation process, so classification agreement does not establish human sentiment validity or investment performance. Other team experiments had already evaluated the reserved test partition; the team must agree how to describe or replace that holdout before final claims.

## References

- Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP. https://aclanthology.org/D14-1181/
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP. https://aclanthology.org/D19-1410/
- Model card: https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2

<!-- RUN_RESULTS -->
## First actual results (7 November 2026)

| Representation | OOF Accuracy | OOF Macro-F1 | Difference from baseline | Fold F1 sample SD |
|---|---:|---:|---:|---:|
| TF-IDF + LR (team reference) | 0.592471 | 0.581996 | — | — |
| fasttext_tok + TextCNN | 0.534370 | 0.536124 | -0.045872 | 0.056843 |
| transformer_tokens + TextCNN | 0.583470 | 0.579576 | -0.002420 | 0.027189 |

Frozen contextual token inputs improve substantially over this scratch-FastText sequence experiment, but the contextual TextCNN remains slightly below the TF-IDF + LR baseline (0.579576 versus 0.581996). The absolute Macro-F1 gap is 0.002420; this single-seed result does not establish a meaningful performance difference. There is no held-out test performance claim. Possible explanations include the small training corpus and the label-generation mechanism; these require further evidence. Further refinements should be selected using development data only.
