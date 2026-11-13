# Member 5 首版交付 · YU Guanlin

入口：[Member5_TextCNN.ipynb](Member5_TextCNN.ipynb)。在普通 Jupyter 中打开，选择安装了项目依赖的 Python 内核，执行 **Restart & Run All** 即可查看并核对已完成结果。默认不训练、不下载模型，不需要嵌入缓存或云端账号。

[Member5_MVP.ipynb](Member5_MVP.ipynb) 保留最初 SBERT／FastText + TextCNN 实验的复现入口；后续完整调参记录在 [本地实验报告](evidence/local_refinement_20261113/REPORT_ZH.md)。

## 环境

已验证 Python 3.12.8、PyTorch 2.11.0，以及 requirements.txt 中的数值包版本。在与 Jupyter 内核相同的环境中安装：

```bash
python -m pip install -r Member5_file/requirements.txt
```

Notebook 内也可以使用 `%pip install -r <仓库路径>/Member5_file/requirements.txt`；安装后重启内核。代码使用当前内核的 `sys.executable` 启动子进程，不指定个人电脑路径、云端目录或固定虚拟环境。支持 CPU，并在可用时使用 MPS／CUDA；明确指定的设备不可用会报错，不混用后端续训。

## 结果与验收

最新实验完成 235 次正式拟合、五个固定外折及三个随机种子：

| 本轮方案 | 平均 Macro-F1 |
|---|---:|
| 自适应选模流程 | 0.6059 |
| 同协议 C0 | **0.6155** |
| 历史官方 baseline（参考） | 0.5820 |

本轮选模没有超过 C0，三个种子差值均为负；保留真实结果，不作为“调参成功”的证据。最终 C1 是该流程选出的实验模型，不能把流程 OOF 当作固定 C1 的独立测试成绩。

Notebook 从逐条 OOF 重新计算 Accuracy、各类 Precision／Recall／F1、混淆矩阵、对比表、结论及分组 bootstrap；无需加载大权重即可检查评分。也可运行：

```bash
python Member5_file/release.py
```

历史完整训练审计与 303 条预测的权重重载曾通过，记录在 independent_verification.json；原始源码快照、逐轮日志和权重由 YU Guanlin 本地保管。本发布版移除了云端执行路径，具有新的源码签名；不把历史审计宣称为对发布代码的重新训练证明。公开整理范围见 PUBLICATION.json。

## 向量和团队接口

冻结编码器为 sentence-transformers/paraphrase-multilingual-mpnet-base-v2；版本、数据／划分哈希及文件校验和在 evidence/embedding_manifest.json。

- 句向量：sbert_multi.npy，1525 × 768，L2 归一化，配套 ID 文件。
- token 序列：tokens_multi.npy，float16，最长 82 个 token、768 维；去除特殊 token，右侧补零。
- mask：tokens_mask.npy，布尔值；每组数组均有同顺序 ID 文件。
- 固定划分：clean_v2／folds_v1；概率列顺序为 Neutral、Positive、Negative（0、1、2）。

团队可通过 src.cache.load_embedding(name, ids=...) 读取数组；训练输出为 preds/<run_id>.npz，包含开发集 OOF 与 303 条保留集概率。大缓存、权重、虚拟环境不提交 Git。

## 可选复现训练

首次需要本地冻结缓存。已有缓存时按上述 manifest 检查；没有缓存时可通过 mvp.py 的 encode 子命令生成。首次下载预训练模型需要联网。新生成缓存使用本地 m5_transformer_manifest.json 绑定当前数据、固定编码器版本及实际文件哈希；允许跨平台浮点存储差异，但不会混用不同来源缓存或回写历史结果。

为保护团队缓存，准备新的独立 DATA_ROOT，并复制版本化原始／清洗数据。以下代码可在仓库内的普通 Jupyter 单元格执行：

```python
from pathlib import Path
import os, shutil, subprocess, sys
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'src/config.py').exists())
work = ROOT / 'data/local_artifacts/m5_reproduction_inputs'
if (work / 'emb').exists() and any((work / 'emb').iterdir()):
    raise FileExistsError('缓存目录非空；请先核验或选择新的目录')
for name in ['raw', 'clean_v2']:
    if not (work / name).exists():
        shutil.copytree(ROOT / 'data' / name, work / name)
env = {**os.environ, 'DATA_ROOT': str(work)}
subprocess.run([sys.executable, str(ROOT/'Member5_file/mvp.py'),
                'encode', '--device', 'cpu',
                '--revision', '4328cf26390c98c5e3c738b4460a05b95f4911f5'],
               cwd=ROOT, env=env, check=True)
subprocess.run([sys.executable, str(ROOT/'Member5_file/run_local.py'),
                '--device', 'auto', '--run-id', 'm5_first_release_reproduction'],
               cwd=ROOT, env=env, check=True)
```

也可把准备好的缓存放到 Notebook 可选训练单元格所读的 DATA_ROOT，并明确开启 RUN_TRAINING。整个流程使用当前环境和本地文件，不需要特定 Notebook 托管平台。

训练顺序为审计、合成计时、记忆测试／学习曲线、六配置内层筛选及多种子复核、五外折、最终模型、独立核验。C0 使用自身九次内层最佳轮数的中位数。累计运行预算两小时，诊断十五分钟、记忆测试两分钟；尝试与失败保留，已完成拟合经校验后复用。源码、输入、环境或后端改变必须使用新的、明确记录的实验，不通过换编号追分或绕过预算。

## 解释限制

历史 OOF 已被查看，团队保留集也曾被接触；没有全新独立测试集声明，也没有人工独立标签验证。训练／验证差距与过拟合相符，但不能唯一归因于容量。没有解冻编码器、把情感分类器当嵌入，或将 VADER 分数作为特征。

本文件夹为成员交付，4 号仍需整合到课程要求的最终单个 Notebook；不改动其他成员公共模块。
