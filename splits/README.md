# Grouped folds

[folds_v1.csv](folds_v1.csv) assigns each cleaned tweet to one split using its `id` and `group_id`. The 1,222 development rows have fold numbers 0 through 4. The 303 reserve rows have fold number -1.

Every tweet in a related-text group stays in the same fold or reserve. The [split notebook](../notebooks/01_data/02_splits.ipynb) can reproduce the assignment from `clean_v2` with seed 42.
