# 6 号的交付文件

成员：Brain & Yang Qitian　负责：BiLSTM + Attention、不平衡处理、报告统一

把你的交付文件**直接放进这个文件夹**即可，不用管它们最后放在仓库的哪个位置，也不用先改格式。最后由 2 号（GU Kailei）统一整理到仓库的对应位置，并规范格式。

## 可以放

- notebook（`.ipynb`）和代码（`.py`）
- 报告章节草稿、PPT 页、实验记录
- 图片（`.png`）和结果表（`.csv`）

## 不要放

- 数据缓存、嵌入向量和模型权重（`.npy`、`.npz`、`.pt`、`.pth` 等），这些文件太大
- 密钥、密码和账号信息（例如 `.env`）

## 怎么上传

在 GitHub 上打开这个文件夹，点 **Add file → Upload files**，把文件拖进去。提交说明用中文写清楚是什么，例如"6 号：BiLSTM notebook 第一版"。只放进自己的文件夹，不要改别人的文件夹。

## Submission result

Use [the corrected BiLSTM notebook](../../notebooks/03_models/25_m6_bilstm.ipynb) and [its OOF scores](../../results/m6_bilstm_grouped.csv). It selects the epoch inside each outer training partition, then refits before predicting the grouped validation fold. The previous 0.550 score is retained as experiment history. The corrected development Macro-F1 is 0.5211.

The [current imbalance notebook](../../notebooks/04_analysis/32_imbalance.ipynb) scores only the development folds. Its regenerated FastText cache gives 0.5164, 0.5230 and 0.5354 without adjustment, with class weights and with oversampling, respectively. The [original ledger](../../results/imbalance.csv) also contains previously reported reserve scores.
