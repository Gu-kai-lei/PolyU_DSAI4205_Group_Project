# Data folder

Nothing here is committed except this file. The real data lives in the shared Google Drive folder.

Point the `DATA_ROOT` environment variable at your copy of that folder (see the main README). If `DATA_ROOT` is not set, the code falls back to this `data/` folder, so you can also copy the files here for a quick local run.

Expected layout:

```
DATA_ROOT/
├─ raw/stock_market_crash_2022.csv   original dataset, never edited
├─ clean/clean_v1.csv                output of notebooks/01_cleaning.ipynb
├─ emb/                              cached embeddings (.npy)
└─ preds/                            OOF and test probabilities (.npy)
```
