# 每个人负责的实验

数字表示项目阶段。每位组员运行自己负责的 Notebook，并先确认前置数据或缓存已准备好；不同模型可以并行开发。

| 阶段 | 文件夹 | 内容 |
|---|---|---|
| 1 | [01_data/](01_data/) | EDA、清洗、固定划分和基线 |
| 2 | [02_embeddings/](02_embeddings/) | SBERT、FastText 和 token 向量 |
| 3 | [03_models/](03_models/) | 六种分类模型 |
| 4 | [04_analysis/](04_analysis/) | 集成、标签噪声和不平衡分析 |
| 5 | [05_final/](05_final/) | 整合最终提交的单个 Notebook |

| 组员编号 | 主要入口 |
|---|---|
| 1 号 | `03_baseline`、`20_m1_svm`、`30_ensemble` |
| 2 号 | `01_cleaning`、`11_emb_fasttext`、`21_m2_knn`、`31_label_noise_errors` |
| 3 号 | `00_eda`、`22_m3_xgboost` |
| 4 号 | `02_splits`、`23_m4_mlp`、`99_final` |
| 5 号 | `10_emb_sbert`、`12_emb_tokens`、`24_m5_textcnn` |
| 6 号 | `25_m6_bilstm`、`32_imbalance` |

报告和 PPT 每人写自己负责的部分，分配见[项目首页的分工](../README.md#分工)。

每个 Notebook 开头的任务清单说明负责人、工作内容和产出。评分与预测使用全组统一接口，详细规则见[项目首页](../README.md)。

