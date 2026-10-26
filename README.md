# DSAI4205 小组项目 · DataVinci

项目说明、分工和运行方法。

| 姓名 | 学号 |
|---|---|
| FENG Dingran | 24128996D |
| GU Kailei | 24102618D |
| CHOI CHUN MAN | 24133223D |
| YANG Qitian | 24128112D |
| CAO Taihe | 24101734D |
| YU Guanlin | 24107216D |

DSAI4205 Big Data Analytics小组项目，占总分 25%。

我们组分到 **Dataset 3：Huge crash in the Stock market**，内容是 2022 年 #stockmarketcrash / #bearmarket 推文，任务是做一个文本情感三分类的决策支持系统。

> 这里只是摘要，具体要求以 [docs/materials/](docs/materials/) 里的原文为准。

## 截止日期

| 日期 | 交付物 | 占比 |
|---|---|---|
| **11/6（周五）23:55** | 中期报告：EDA、清洗、baseline 结果 | 20% |
| 第 10/11 周周六 14:00–18:00 | 中期答疑会面（WK10） | 含在上一项 |
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
- 下面这些来自建仓库之前的初步分析，**还没有在仓库里复现**。由 3 号（`00_eda`）和 2 号（`31_label_noise_errors`）在 notebook 里重新计算，并记录方法和数据版本：
  - TF-IDF + LR 初步测试：5 折交叉验证准确率约 62%
  - 标签和 VADER 打分的一致率约 93.2%，标签可能由 VADER 自动生成（待验证的假设）
  - 部分字符被替换成了 `__`（疑似弯引号和 emoji，具体来源待抽样核对），还有一部分双重编码造成的乱码
  - 混有西班牙语、印地语、阿拉伯语等多种语言
  - 9 条完全重复，规范化文本后还有约 71 条近似重复
  - Neutral 类的文本明显比其他两类短

## 分工

| # | 成员 | 文本表示 | 分类器 | 公共职责 |
|---|---|---|---|---|
| 1 | 待填 | 多语言 SBERT、FastText 平均句向量（英文 SBERT 选做） | SVM | Baseline 和改进版 baseline，汇总 3iii / 3iv 对比，抽取 SBERT 句向量 |
| 2 | 待填 | 同上 | kNN | 数据清洗和清洗日志，训练自有的 FastText（Word2Vec 选做），标签噪声和错误分析 |
| 3 | 待填 | 同上，加手工特征 | XGBoost | EDA（至少 6 张图），特征工程，SHAP，商业洞察 |
| 4 | 待填 | 同上 | MLP | 评估框架和固定数据划分，不平衡处理，合并最终 notebook |
| 5 | 待填 | 冻结 Transformer 的 token 序列、FastText 词向量序列 | TextCNN | 抽取 token 序列，Stacking 集成 |
| 6 | 待填 | 同上 | BiLSTM + Attention | 统一各组员报告和 PPT 主编 |

每个人对应的 notebook 见 [notebooks/README.md](notebooks/README.md)，每个 notebook 开头都列了具体任务清单。标"（选做）"的任务等必做部分完成后再做。

## 仓库结构

每个文件夹里都有自己的 README，点进去可以看到详细说明（例如 notebook 的负责人对照表在 `notebooks/` 里）。

```
PolyU_DSAI4205_Group_Project/
├─ README.md               项目说明、分工和运行方法
├─ environment.yml         conda 环境配置，所有包固定版本
├─ .gitignore              排除数据、嵌入和权重
├─ .gitattributes          统一文本换行和文件类型
├─ docs/                   课程资料、会议和实验记录、清洗日志、报告草稿
├─ notebooks/              按 5 个阶段编号的 notebook，每人只改自己负责的
├─ src/                    全组共用函数：路径和版本号、读取数据、评分、缓存、清洗、FastText
├─ splits/          [4 号] 固定数据划分 folds_v1.csv，生成后不再修改
├─ results/                每个模型的评分结果，每次实验一行，带 run_id
├─ figures/                报告和 PPT 用图
├─ reports/         [6 号] 已提交的报告和 PPT 存档
└─ data/                   原始数据、clean_v1 和清洗日志；嵌入与预测缓存不进 Git
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

原始数据已放在 `data/raw/stock_market_crash_2022.csv`，当前清洗数据与日志放在 `data/clean_v1/`。下载仓库后，默认直接读取这些文件。嵌入缓存和预测结果仍保留在本地或共享 Google Drive。

如果要使用共享网盘，在网盘中保持相同目录结构，并设置 `DATA_ROOT`（PowerShell）：

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

- 提交说明请使用中文，简要说明本次改动，方便组员在目录页和提交记录中查看。

- 每个人在自己的分支上开发（例如 `m3-xgboost`），通过 Pull Request 合入 `main`。
- 只改自己负责的 notebook。要改 `src/` 里的公共代码，先在群里说一声。
- 每次实验先用 `new_run_id()` 生成一个 run_id。评分用 `evaluate(..., run_id=run_id, params=...)` 计算、`save_result(row, "m3_xgboost")` 保存；OOF 概率和测试集概率用 `save_preds(run_id, ...)` 保存。同一个 run_id 把成绩、超参数和预测文件绑在一起，重跑也不会覆盖旧结果。
- 仓库仅保留指定的原始 CSV、`clean_v1.csv` 和版本清洗日志；嵌入缓存、预测结果、模型权重及其他本地数据不提交。
- 每次实验记进 `docs/experiments/mN.md`，每周会议记进 `docs/meetings/`。这些是评分里"迭代过程证据"的来源。

**数据和实验版本**

当前版本号写在 `src/config.py`：`CLEAN_VERSION`（清洗数据）和 `SPLIT_VERSION`（数据划分）。每条评分记录和每个预测文件都会自动记下这两个版本和随机种子。

| 改了什么 | 要做什么 |
|---|---|
| 清洗删了行、加了行，或者近似重复的分组（`group_id`）变了 | 升级 `CLEAN_VERSION` 和 `SPLIT_VERSION`，4 号重新生成划分，**所有模型重跑** |
| 只改了文本处理方式，行和 `group_id` 都没变 | 升级 `CLEAN_VERSION`，划分沿用；用到这列文本的嵌入和模型重跑 |
| 只改了自己模型的超参数 | 不升级版本，用新的 run_id 重跑自己的模型 |

`load_splits(clean=...)` 会自动检查划分和清洗数据是否对得上，对不上会直接报错。

**不能做的事**

- 把 VADER 分数当特征：标签可能就是 VADER 生成的（待验证），用它当特征等于走捷径，相当于泄露标签。
- 让近似重复的推文跨训练集和测试集：同一个 `group_id` 必须落在同一折。
- 在划分之前做 SMOTE 或过采样：只能在训练折内做。
- 用包含验证折的文本训练自训嵌入：FastText 等要按折训练，测试集文本任何时候都不参与训练。
- 先划分再去重：必须先去重，否则重复样本会同时出现在训练集和测试集里。
- 在 baseline 以外用 TF-IDF，包括用 TF-IDF 权重给词向量加权平均。

**我们的实施方案**

- Transformer 类模型（SBERT 等）只用来抽取嵌入、保持冻结，不和分类器一起微调。项目要求嵌入模型和分类模型相互独立，冻结能让这一点没有争议。是否允许微调，答疑时向老师确认。
- 文本里的 `__` 只还原人工抽样核对过的模式，其余统一替换成占位符，每条规则都记进清洗日志。

## 原始资料

| 文件 | 内容 |
|---|---|
| [DSAI4205_Project_Description.pdf](docs/materials/DSAI4205_Project_Description.pdf) | 项目说明和评分标准 |
| [DSAI4205_Final_Report_Template.pdf](docs/materials/DSAI4205_Final_Report_Template.pdf) | 最终报告模板 |
| [Dataset3_Description.docx](docs/materials/Dataset3_Description.docx) | 数据集说明，包括标签编码 |

