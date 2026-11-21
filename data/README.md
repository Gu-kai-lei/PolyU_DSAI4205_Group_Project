# Data

The supplied [raw CSV](raw/stock_market_crash_2022.csv) contains 1,698 tweets. [clean_v2.csv](clean_v2/clean_v2.csv) contains the 1,525 retained rows. [clean_v2_all.csv](clean_v2/clean_v2_all.csv) records the inclusion decision for every raw row.

The [cleaning log](clean_v2/clean_v2_log.md) lists the rules, affected rows and reasons for exclusion. Run `python data/clean_v2/clean_v2.py` from the repository root to regenerate the files. The earlier dataset remains in [clean_v1/](clean_v1/).

To use a shared cache, set `DATA_ROOT` to a directory with the same layout. Embeddings, local predictions, weights and NLTK corpora live outside Git. Use pandas or JupyterLab to read tweet fields as text.
