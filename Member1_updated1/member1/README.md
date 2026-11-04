# Member 1 — 正式 baseline 交付

2026-11-04 已接入 Member 4 的统一划分并完成正式五折运行。现有报告没有改动。

- Run ID：`baseline__tfidf__20261104-222636-b258`
- 数据：clean_v2，1,525 条；开发集 1,222 条，保留测试集 303 条。
- 模型：TF-IDF unigram (`min_df=1`) + Logistic Regression (`C=1`, `max_iter=2000`)，随机种子 42。
- 正式 pooled OOF：Accuracy **59.2471%**、Macro-Precision **60.5349%**、Macro-Recall **57.6738%**、Macro-F1 **58.1996%**。
- 五折均值 ± 样本标准差：Accuracy **59.2676% ± 4.1677 个百分点**；Macro-F1 **58.1647% ± 4.2273 个百分点**。
- pooled OOF 指标与 Member 4 的独立 baseline reference 一致。各折均值和 pooled OOF 是不同统计口径。
- 测试集没有参与训练、调参或评分；本次也未生成测试预测。OOF 预测缓存在本地 `data/preds/`，不上传 GitHub。

正式结果见 [evidence/baseline](evidence/baseline)，团队评分见 [../results/baseline.csv](../results/baseline.csv)。图表在仓库 `figures/`，文件名包含本次 run_id。

## 运行

在仓库根目录、Python 3.11 环境中执行：

```bash
python -m pip install -r member1/requirements-baseline.txt
python member1/baseline.py --check
python -m unittest member1.test_baseline member1.test_split_recovery -v
python member1/baseline.py --mode team
```

`requirements-baseline.txt` 与团队 environment.yml 的 baseline 依赖版本一致；`requirements-baseline-tested.txt` 记录本次包括间接依赖在内的实测版本。Notebook 入口是 `notebooks/01_data/03_baseline.ipynb`，需要额外安装团队 Jupyter 环境并选择正确内核。脚本已实跑；Notebook 本次通过格式和代码语法检查，未另做 fresh-kernel 执行。

环境安装与运行步骤见 [交付包首页](../README.md)。环境不随 Git 上传。

## 划分来源

上传的 Member 4 包未附 `folds_v1.csv`。我们从其 OOF 的 id/group_id/fold 恢复开发集，clean_v2 中其余 ID 为测试集。恢复文件的 SHA-256 与 Member 4 记录**逐字节一致**，不是重新随机划分。

- 原始来源与 SHA-256：`split_provenance.json`。
- 可复查的最小证据：`evidence/member4/`，已去除模型预测概率，仅保留划分字段及签名。
- 恢复工具：`python member1/recover_member4_splits.py`。正常运行已有的正式划分无需调用此工具。
- 校验会拒绝不完整 OOF、改动后的文本、折号、标签或分组，以及不匹配的哈希。

## 测试与后续

8 项测试通过，包括组泄漏、训练词表隔离、缺少依赖时拒绝运行、团队结果/缓存接口和划分恢复。两张图已目视检查。

下一步是 SVM：仍需 Member 5 的 SBERT 句向量及 ID，和/或 Member 2 按正式折训练的 FastText 句向量及 ID。当前上传包没有这些向量。Baseline 调参、SVM、Stacking 和最终模型对比尚未完成。

历史临时运行输出和过时的 Notebook 草稿已清理。历史报告草稿可能仍写着 pending，以本交付记录为准。
