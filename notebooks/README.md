# Working notebooks

The [submission notebook](../submission/DataVinci.ipynb) runs on its own. These notebooks are for reproducing individual stages in the repository.

| Directory | Run order |
|---|---|
| [01_data](01_data/) | EDA, cleaning, grouped split and baseline |
| [02_embeddings](02_embeddings/) | SBERT sentence vectors, fold-specific FastText and SBERT token vectors |
| [03_models](03_models/) | Six classifier experiments |
| [04_analysis](04_analysis/) | Fusion, label diagnostics and class imbalance |

Run the embedding notebook needed by a model before retraining it. See the [SBERT handoff](../experiments/Member5_file/README.md) for the weight and cache requirements.
