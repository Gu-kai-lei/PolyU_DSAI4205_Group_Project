# Member 5 · Final report contribution

## 3.3 Frozen contextual representations

We encoded `text_clean` with the frozen `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` checkpoint at revision `4328cf26390c98c5e3c738b4460a05b95f4911f5`. Sentence vectors use attention-masked mean pooling followed by L2 normalisation. For the sequence models, we removed special tokens, kept the remaining contextual vectors in order and padded the sequences with zeros. We set the maximum length to 82 tokenizer positions using the 99th percentile of development-text lengths. The resulting cache contains 1,525 sentence vectors of dimension 768, a float16 token array of size 1,525 × 82 × 768, and a boolean padding mask. Tweet IDs link the representations to `clean_v2` and `folds_v1`.

## 3.4 TextCNN classifier and refinement

The independently trained TextCNN applies convolution widths of 2, 3, 4 and 5 to the frozen contextual token vectors, followed by ReLU, masked max pooling, dropout and a three-class output. The final C3 configuration uses 64 filters per width, dropout of 0.5, AdamW with learning rate 0.0003 and weight decay of 0.01, and batch size 32. Its 689,155 trainable parameters were refitted for nine epochs on the full development split at seed 42.

To evaluate the selection procedure, we used the team's five outer grouped folds. For each outer fold, three inner grouped folds and three seeds compared four settings that vary filter count, learning rate or class weighting. The selected model and matched C0 were each refitted on the outer training rows for their own selected epoch count. The mean pooled out-of-fold Macro-F1 across the three seeds was 0.6167 for the selection procedure, compared with 0.6070 for matched C0 and 0.5820 for the team TF-IDF and logistic-regression baseline. The gain over C0 was positive in all three seeds, but all paired group-bootstrap intervals included zero. The 0.6167 estimate applies to the selection procedure. The fixed C3 checkpoint produced probabilities for 303 reserved rows, whose labels were not scored in this experiment.

## References

- Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. EMNLP. https://aclanthology.org/D14-1181/
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. EMNLP-IJCNLP. https://aclanthology.org/D19-1410/
