# 成员 1 Baseline 交付包

此文件夹是 5 个交付文件的副本，保留它们在团队仓库中的目录结构。

## 合并方式

请将本目录中的 `notebooks/`、`member1/`、`docs/` 下的文件合并到团队仓库根目录的对应路径。仅将整个 `member1_file/` 上传可供阅读，但它不是独立运行的项目；不要直接在此副本中运行 Notebook。合并时只更新列出的文件，不要覆盖其他成员的目录或文件。

- `notebooks/01_data/03_baseline.ipynb`：Baseline 入口。
- `member1/baseline.py`：Notebook 调用的实现，必须一起合并。
- `member1/test_baseline.py`：Data Leakage 与保存接口检查。
- `docs/report_drafts/m1_baseline.md`：Methodology 初稿。
- `docs/experiments/m1.md`：Experiment Log。

## 当前状态

已完成 provisional 验证和团队接口接入。正式 Results 等待成员 4 的 `splits/folds_v1.csv` 和 `environment.yml` 指定的 Environment。尚未执行 Test Set Evaluation，也尚未完成 Refinement 或 Ensemble。

依赖团队现有 `src/`、`environment.yml` 和 `data/clean_v2/clean_v2.csv`，本交付包不重复包含这些文件。provisional Results 和 Prediction Cache 未打包。

## 合并后的运行方式

在团队仓库根目录、激活团队 Environment 后运行：

```bash
python member1/baseline.py --check
python -m unittest member1.test_baseline -v
```

待 Preflight 通过后运行：

```bash
python member1/baseline.py --mode team
```

或打开 `notebooks/01_data/03_baseline.ipynb`。Notebook 依赖 `member1/baseline.py`，最终提交单个 Notebook 时由成员 4 一并合并实现。

本目录为静态副本，原文件后续修改不会自动同步。
