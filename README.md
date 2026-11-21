# DataVinci: Sentiment in stock market crash tweets

DSAI4205 Big Data Analytics · The Hong Kong Polytechnic University · Dataset 3

We classify tweets about the 2022 stock market crash as neutral, positive or negative. We used 1,525 of the 1,698 supplied tweets after cleaning. Near-duplicate tweets stay in the same group so that a validation fold cannot contain a close copy of one of its training examples.

## Submission

Open [DataVinci.ipynb](submission/DataVinci.ipynb). It contains the data, code and saved results. You can download the notebook by itself.

To rerun it from the repository root:

```sh
conda env create -f environment.yml
conda activate dsai4205
jupyter lab submission/DataVinci.ipynb
```

In JupyterLab, select **Restart Kernel and Run All Cells**. The default run rebuilds the cleaning outputs, checks the fixed split, fits both baselines, and reads the saved scores and predictions for the final comparison. The appendix has switches for model retraining. The SBERT experiments need the pinned weights or a verified embedding cache.

## Results

These scores come from five grouped folds on the 1,222 development tweets.

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + logistic regression | 59.25% | 58.20% |
| Tuned TF-IDF + logistic regression | 61.46% | 61.35% |
| SBERT + TextCNN | 61.78% | 61.79% |
| Tuned baseline + TextCNN, equal weights | 66.37% | 66.39% |

The fusion classified 811 tweets correctly. We chose that combination after reviewing the development results, so it still needs an independent evaluation. This submission leaves the 303 reserve labels unscored. The notebook gives the full model comparison and confusion matrix.

## Repository

| Path | Contents |
|---|---|
| [DataVinci.ipynb](submission/DataVinci.ipynb) | Single-file technical submission |
| [data/](data/) and [splits/](splits/) | Supplied tweets, cleaning records and the fixed folds |
| [src/](src/) and [tests/](tests/) | Shared Python code and verification tests |
| [notebooks/](notebooks/) | Working notebooks in experiment order |
| [experiments/](experiments/) and [results/](results/) | Member handoffs, predictions and score tables |
| [figures/](figures/) | Figures from individual experiments |
| [docs/](docs/) and [reports/](reports/) | Course materials, project notes and report drafts |

## Team

| Name | Student ID | Model | Other responsibilities |
|---|---|---|---|
| FENG Dingran | 24128996D | SVM | TF-IDF baseline, model comparison and ensemble experiments |
| GU Kailei | 24102618D | kNN | Data cleaning, fold-specific FastText, error analysis, project coordination and the interim report |
| CHOI CHUN MAN | 24133223D | BiLSTM with attention | Class imbalance experiments, report formatting and submission coordination |
| YANG Qitian | 24128112D | XGBoost | Exploratory analysis, SHAP plots and interpretation of the results |
| CAO Taihe | 24101734D | MLP | Grouped data split, shared PyTorch training code and notebook integration |
| YU Guanlin | 24107216D | TextCNN | SBERT sentence and token embeddings and slide design |
