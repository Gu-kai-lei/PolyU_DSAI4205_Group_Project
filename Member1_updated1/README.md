# Member 1 — Baseline 完整交付包

本文件夹包含查看和运行 TF-IDF + Logistic Regression baseline 所需的代码、清洗数据、统一划分和正式实验结果。下载仓库后进入本文件夹即可运行，无需合并到团队根目录。

## 查看内容

- [正式结果与方法说明](member1/README.md)
- [Notebook](notebooks/01_data/03_baseline.ipynb)
- [实现代码](member1/baseline.py)
- [实验记录](docs/experiments/m1.md)
- [评分结果](results/baseline.csv)
- [图表](figures/)
- [验证证据](member1/evidence/baseline/)
- [Methodology 历史草稿](docs/report_drafts/m1_baseline.md)：可能仍含 pending，以正式结果为准。

正式 pooled OOF Accuracy 为 59.2471%，Macro-F1 为 58.1996%。测试集未参与训练或评分。SVM、调参及 Stacking 尚未完成。

## 下载后运行

在本文件夹（包含本 README 的 Member1_update1）中打开终端，使用 Python 3.11：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r member1/requirements-baseline.txt
python member1/baseline.py --check
python -m unittest member1.test_baseline member1.test_split_recovery -v
python member1/baseline.py --mode team
```

Windows 激活命令为 `.venv\Scripts\activate`。使用 Notebook 时额外安装 `jupyter`，选择此 Python 环境，再打开 `notebooks/01_data/03_baseline.ipynb`。GitHub 网页用于查看，实际执行需下载到本地。

如已设置 `DATA_ROOT` 环境变量，请先取消以使用包内数据。每次训练创建新 run_id，结果写到本包中的 results、figures 和 member1/outputs，预测缓存写到 data/preds。

## 来源与上传

`src/` 为团队公共模块快照，清洗数据为团队 clean_v2，划分与 Member 4 原始哈希一致，见 [划分来源](member1/split_provenance.json)。本包不包含虚拟环境或预测缓存。

上传整个 `Member1_update1` 文件夹，保留这一层目录。以后更新同一路径即可。团队根目录共享实验记录不会因本包上传而自动更新。
