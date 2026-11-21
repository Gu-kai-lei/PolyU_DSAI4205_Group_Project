# 2 号的交付文件

成员：GU Kailei　负责：kNN、错误分析、项目协调

## 已交付：kNN 预测（给 1 号做 Stacking 和总对比）

由 `notebooks/03_models/21_m2_knn.ipynb` 生成，对应 `results/m2_knn.csv` 中交叉验证分数最高的一次（SBERT 句向量，k=41，按距离加权，CV macro-F1 0.539；参数在同一套折上调，嵌套估计 0.522）。重跑 notebook 会在 `local_artifacts/knn/<run_id>/` 生成独立预测文件。以下文件保留为归档交接证据。

| 文件 | 内容 |
|---|---|
| `predictions/knn_oof.csv` | 开发集 1,222 条的交叉验证预测：`id, group_id, fold, label, prediction, prob_0, prob_1, prob_2` |
| `predictions/knn_test.csv` | 测试集 303 条的预测概率（不含标签，这里不评分）：`id, prediction, prob_0, prob_1, prob_2` |

概率列的顺序是 0 Neutral、1 Positive、2 Negative。

把你的交付文件**直接放进这个文件夹**即可，不用管它们最后放在仓库的哪个位置，也不用先改格式。最后由 2 号（GU Kailei）统一整理到仓库的对应位置，并规范格式。

## 可以放

- notebook（`.ipynb`）和代码（`.py`）
- 报告章节草稿、PPT 页、实验记录
- 图片（`.png`）和结果表（`.csv`）

## 不要放

- 数据缓存、嵌入向量和模型权重（`.npy`、`.npz`、`.pt`、`.pth` 等），这些文件太大
- 密钥、密码和账号信息（例如 `.env`）

## 怎么上传

在 GitHub 上打开这个文件夹，点 **Add file → Upload files**，把文件拖进去。提交说明用中文写清楚是什么，例如"2 号：kNN notebook 第一版"。只放进自己的文件夹，不要改别人的文件夹。
