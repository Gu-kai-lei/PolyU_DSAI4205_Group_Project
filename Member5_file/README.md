# Member 5 当前采用版 · YU Guanlin

当前采用的是截图所指的 **C3 v1**，不是后续本地 C1。版本、权重摘要和证据目录统一由 [selected_version.json](selected_version.json) 指定。

入口：[Member5_TextCNN.ipynb](Member5_TextCNN.ipynb)。在普通 Jupyter 中打开，执行 **Restart & Run All** 查看并重算记录；默认无需权重、缓存、GPU 或账号授权，不启动训练或下载。

## 为什么采用这版

| 已完成的流程 | 三种子平均 Macro-F1 |
|---|---:|
| **当前采用的 v1** | **0.6167** |
| 同轮 C0 | 0.6070 |
| 后续本地选模流程（归档） | 0.6059 |
| 后续本地 C0（不同协议，参考） | 0.6155 |
| 官方 baseline（参考） | 0.5820 |

v1 记录分数较高，相对自身 C0 三个种子均为正。可以作为当前工作版本；但 +0.0097 未达预定 +0.010，三个分组 bootstrap 区间均跨零，不能确认稳定优势，也不把不同协议的点估计当公平独立头对头比较。

**0.6167 是完整选模流程的 OOF 均值，不是固定 C3 的独立测试分数。** 最终已有 C3 权重为：64 filters、lr=0.0003、dropout=0.5、weight decay=0.01、无类别权重、9轮、seed42，689155个参数。本次只是采用该原始权重，没有重训。

采用记录：[ADOPTION_ZH.md](evidence/refinement_v1_20261112/ADOPTION_ZH.md)。后续六配置实验仍保留为 [负结果归档](evidence/local_refinement_20261113/REPORT_ZH.md)，不删除或改写其分数。

## 环境与数值复核

已验证 Python 3.12.8／PyTorch 2.11.0。在与 Jupyter 内核相同的环境安装：

```bash
python -m pip install -r Member5_file/requirements.txt
python Member5_file/release.py
```

Notebook 也可用 `%pip install -r <仓库路径>/Member5_file/requirements.txt`，安装后重启内核。所有子进程使用当前 Python，不指定个人环境或云端目录。

默认核验从六份逐条 OOF 重算 Accuracy、各类 Precision／Recall／F1、混淆矩阵、对比表、类别取舍及配对分组 bootstrap。历史249次拟合的完整关联审计见 acceptance_forensics.json；本次对原始 C3 的再次 CPU 重载见 current_model_reload.json：303条预测类别一致，最大概率差3.58e-7，未使用保留集标签评分。历史完整训练审计不冒充本发布代码的一次新训练。

## 使用已有 C3 模型

由 YU Guanlin 本地保管的原始交付包括：

| 文件 | 放置于 DATA_ROOT 下 | 校验来源 |
|---|---|---|
| 原始 C3 权重 | models/textcnn_c3_v1.safetensors | selected_version.json 中的 weights_sha256 |
| 原始六个冻结数组及 ID／mask | emb/ | evidence/embedding_manifest.json |
| 对应完整概率包 | preds/textcnn_refinement_colab_latest_20261112.npz | selected_version.json 中的 team_predictions.sha256 |

接收文件后，复制到自己的 DATA_ROOT，不覆盖其他成员的缓存。**加载这个已有模型需要原始已核验数组，不能重新编码后冒充相同输入**：不同平台的浮点字节可能不同，已有权重的输入合同仍按原始哈希核验。无需再次下载编码器。

默认 DATA_ROOT 是仓库 data/。如使用其他目录，先复制 raw／clean_v2 和上述文件，在新内核最先设置 DATA_ROOT 环境变量，再导入项目模块；固定 splits/folds_v1.csv 留在仓库。

从仓库根目录或 Notebook 设置完成后可执行：

```python
from Member5_file.adopted_model import load_adopted_model, predict_cached, verify_team_archive
model = load_adopted_model()  # CPU；核对原始权重、模型类、数据和划分
saved = verify_team_archive()  # 核对对应概率包的哈希和版本
probabilities = predict_cached(saved['test_ids'].tolist(), device='cpu')
```

不自动下载模型，缺少文件时明确报错。大缓存／权重不提交 Git，默认查看模式不依赖这些大文件。可明确指定 CPU／MPS／CUDA 推理；设备不可用时不会偷偷切换后端。

## 团队接口与后续训练

冻结编码器为 sentence-transformers/paraphrase-multilingual-mpnet-base-v2，revision 固定在 embedding_manifest.json：句向量1525×768、L2归一化；token序列最长82×768、float16、去特殊token、右补零；mask为布尔值，每组均有配套ID。

数据／划分为 clean_v2／folds_v1，概率列顺序 Neutral／Positive／Negative（0／1／2）。团队汇总行 [results/m5_textcnn_adopted.csv](../results/m5_textcnn_adopted.csv) 对应同名概率包，是 **seed42 流程 F1=0.617926**；三种子均值仍为0.616686，不混淆两者。

本次采用既有训练版本，不再重跑历史训练。原始训练逻辑与完整源码快照供审计；历史预算／失败记录机制的局限不能伪称已修复。后续优化应另开新实验，使用已有的 run_local.py 六配置诊断及耐久记录入口；那是后续研究，不是当前 C3 的复现或自动替换。新生成缓存也只能按新的来源和签名进入新实验。

原始单种子 MVP 仍可在 [Member5_MVP.ipynb](Member5_MVP.ipynb) 查看。4号还需将成员交付整合成课程要求的最终单个 Notebook。

## 保留的限制

历史试跑按带验证早停的时长选择四配置预算，可能间接受到标签影响；长度82沿用完整开发文本的无标签统计；历史OOF和团队保留集曾被接触；没有独立人工标签验证。因此不宣称全新独立测试、已证明模型架构上限或稳定统计优势。没有解冻编码器或把 VADER 分数当特征。
