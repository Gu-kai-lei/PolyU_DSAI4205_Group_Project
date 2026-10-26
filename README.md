# DSAI4205 Group Project · DataVinci

DSAI4205 Big Data Analytics（2026/27 第一学期）小组项目，占课程总分 25%。

我们组分到 **Dataset 3：Huge crash in the Stock market**，内容是 2022 年 #stockmarketcrash / #bearmarket 推文，任务是做一个文本情感三分类的决策支持系统。

> 这里只是摘要，具体要求以 [docs/materials/](docs/materials/) 里的原文为准。

## 截止日期

| 日期 | 交付物 | 占比 |
|---|---|---|
| **11/6（周五）23:55** | 中期报告：EDA、清洗、baseline 结果 | 20% |
| 第 10/11 周周六 14:00–18:00 | 中期答疑会面（需要提前预约） | 含在上一项 |
| **11/22（周日）23:55** | 展示用 PPT（12 分钟展示） | 20% |
| **11/22（周日）23:55** | 最终报告 .docx + **单个** .ipynb | 60% |

## 任务要求

1. **EDA**：数据类型、粒度、分布、异常值、缺失和无效值、变量之间的相关性。
2. **数据清洗**：缺失值、异常值、无效值、去重、标准化、变量变换。**每一步都要记进清洗日志。**
3. **建模**（N = 组员人数 = 6）：
   - i. Baseline：TF-IDF + 逻辑回归。**TF-IDF 只能用在 baseline 里。**
   - ii. 6 个分析模型，每个都要搭配一个**独立的**词嵌入模型。逻辑回归不算在这 6 个里；同一个模型不能既做嵌入又做分类；TF-IDF 不算词嵌入。
   - iii. 6 个模型分别和 baseline 比较，并解释差异。
   - iv. 改进模型（调参、集成），然后再比较一次。
   - v. 至少 6 张数据统计图。
4. **可视化讲故事**：EDA 和最终结果都要用图来说明。

最终报告（60%）的评分构成：

| 评分项 | 占比 | 要点 |
|---|---|---|
| EDA 与清洗 | 20% | 粒度、分布、类别不平衡；小写、去标点、词形还原或词干化；清洗日志 |
| Baseline | 10% | TF-IDF + LR，报告 Accuracy / Precision / Recall / F1 |
| 高级建模 | 30% | 6 个"嵌入 + 分类器"组合，写清整合思路；用课外模型、从零训练会加分；模型同质化会扣分 |
| Refinement | 10% | 调参证据，不平衡处理 |
| 创新 | 10% | 特征工程、新颖的方法组合、深入的商业洞察 |
| 项目开发与文档 | 20% | 详细时间表、个人分工、实验和迭代过程的证据 |

报告必须按老师给的模板写：[DSAI4205_Final_Report_Template.pdf](docs/materials/DSAI4205_Final_Report_Template.pdf)。

## 数据集要点

- 共 1,698 条推文，两列：`text` 和 `text_sentiment`，其中 **0 = Neutral，1 = Positive，2 = Negative**
- 类别分布：Positive 627 / Neutral 575 / Negative 496，轻度不平衡
- TF-IDF + LR 初步测试：5 折交叉验证准确率约 62%
- 已知问题：
  - 标签和 VADER 打分的一致率是 93.2%，很可能是自动生成的标签
  - emoji 和引号被替换成了 `__`，还有一部分双重编码造成的乱码
  - 混有西班牙语、印地语、阿拉伯语等多种语言
  - 9 条完全重复，规范化文本后还有约 71 条近似重复
  - Neutral 类的文本明显比其他两类短

## 分工

| # | 成员 | 文本表示 | 分类器 | 公共职责 |
|---|---|---|---|---|
| 1 | 待填 | 多语言 SBERT、英文 SBERT、FastText 平均句向量 | SVM | Baseline 和改进版 baseline，汇总 3iii / 3iv 对比，抽取 SBERT 句向量 |
| 2 | 待填 | 同上 | kNN | 数据清洗和清洗日志，训练自有的 FastText / Word2Vec，标签噪声和错误分析 |
| 3 | 待填 | 同上，加手工特征 | XGBoost | EDA（至少 6 张图），特征工程，SHAP，商业洞察 |
| 4 | 待填 | 同上 | MLP | 评估框架和固定数据划分，不平衡处理，合并最终 notebook |
| 5 | 待填 | 冻结 Transformer 的 token 序列、FastText 词向量序列 | TextCNN | 抽取 token 序列，Stacking 集成 |
| 6 | 待填 | 同上 | BiLSTM + Attention | 项目管理，预约答疑，报告和 PPT 主编 |

每个人对应的 notebook 见下面的仓库结构，每个 notebook 开头都列了具体任务清单。

## 仓库结构

```
├─ docs/
│  ├─ materials/        课程原始资料：项目说明、报告模板、数据集说明
│  ├─ meetings/         每周会议记录（复制 _template.md）
│  └─ experiments/      每个人的实验记录（复制 _template.md）
├─ notebooks/
│  ├─ 00_eda                     3 号
│  ├─ 01_cleaning                2 号
│  ├─ 02_splits                  4 号
│  ├─ 03_baseline                1 号
│  ├─ 10_emb_sbert               1 号
│  ├─ 11_emb_fasttext            2 号
│  ├─ 12_emb_tokens              5 号
│  ├─ 20_m1_svm … 25_m6_bilstm   各自的模型
│  ├─ 30_ensemble                5 号
│  ├─ 31_label_noise_errors      2 号
│  ├─ 32_imbalance               4 号
│  └─ 99_final                   4 号：最终提交的单个 notebook
├─ src/                 共享代码：config.py（路径、常量）、data.py（读数据）、evaluate.py（统一评分）
├─ splits/              固定的数据划分 folds.csv（4 号生成）
├─ results/             每个模型各一个结果 csv
├─ figures/             报告和 PPT 用的图
├─ data/                本地数据，不进 Git
└─ environment.yml
```

## 快速开始

**1. 克隆仓库**

```bash
git clone https://github.com/Gu-kai-lei/DSAI4205_Group_Project.git
cd DSAI4205_Group_Project
```

**2. 创建环境**

```bash
conda env create -f environment.yml
conda activate dsai4205
nbstripout --install
```

`nbstripout --install` 会在提交时自动清掉 notebook 的输出，避免合并冲突。每个人 clone 之后都要运行一次。

有 NVIDIA 显卡的同学，再把 PyTorch 换成 CUDA 版本：

```bash
pip install torch==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124
```

**3. 准备数据**

数据、嵌入缓存和预测结果都放在共享的 Google Drive 文件夹里。把环境变量 `DATA_ROOT` 设成你本地这个文件夹的路径（PowerShell）：

```powershell
setx DATA_ROOT "G:\我的云端硬盘\DSAI4205-Data3"
```

设置后要重开终端才生效。Colab 用户在 notebook 里加一行：`os.environ["DATA_ROOT"] = "/content/drive/MyDrive/DSAI4205-Data3"`。共享文件夹要先在 Drive 里"添加快捷方式到我的云端硬盘"，Colab 才能读到。

如果不设置 `DATA_ROOT`，代码会默认读取仓库里的 `data/` 文件夹，目录结构见 [data/README.md](data/README.md)。

**4. 打开自己的 notebook**

```bash
jupyter lab
```

## 协作规则

- 每个人在自己的分支上开发（例如 `m3-xgboost`），通过 Pull Request 合入 `main`。
- 只改自己负责的 notebook。要改 `src/` 里的公共代码，先在群里说一声。
- 结果统一用 `evaluate()` 计算、用 `save_result(row, "m3_xgboost")` 保存到自己的结果文件里，不要多人共用一个 csv。
- 每个模型都要输出训练集的 OOF 概率和测试集概率（`.npy`，N × 3），放在 `DATA_ROOT/preds/`，集成时要用。
- 不提交数据、嵌入缓存、模型权重。`.gitignore` 已经排除了这些文件。
- 每次实验记进 `docs/experiments/mN.md`，每周会议记进 `docs/meetings/`。这些是评分里"迭代过程证据"的来源。

**不能做的事**

- 把 VADER 分数当特征：标签本身就是 VADER 打的，这样做等于泄露标签。
- 把 Transformer 和分类器一起微调：Transformer 必须冻结，否则可能被认定为同一个模型兼做嵌入和分类。
- 在划分之前做 SMOTE 或过采样：只能在训练折内做。
- 先划分再去重：必须先去重，否则重复样本会同时出现在训练集和测试集里。
- 在 baseline 以外用 TF-IDF，包括用 TF-IDF 权重给词向量加权平均。

## 原始资料

| 文件 | 内容 |
|---|---|
| [DSAI4205_Project_Description.pdf](docs/materials/DSAI4205_Project_Description.pdf) | 项目说明和评分标准 |
| [DSAI4205_Final_Report_Template.pdf](docs/materials/DSAI4205_Final_Report_Template.pdf) | 老师给的最终报告模板 |
| [Dataset3_Description.docx](docs/materials/Dataset3_Description.docx) | 数据集说明，包括标签编码 |
