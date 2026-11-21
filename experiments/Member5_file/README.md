# Member 5 · YU Guanlin

| Notebook | 用途 |
|---|---|
| [10_emb_sbert.ipynb](../../notebooks/02_embeddings/10_emb_sbert.ipynb) | 核对或生成 SBERT 句向量，供句向量模型使用 |
| [12_emb_tokens.ipynb](../../notebooks/02_embeddings/12_emb_tokens.ipynb) | 核对或生成 SBERT token 序列和 padding mask |
| [24_m5_textcnn.ipynb](../../notebooks/03_models/24_m5_textcnn.ipynb) | 查看 C3 v1 成绩、加载已保存权重、可选训练固定 C3 或重跑四配置选模 |

在团队仓库里打开普通 Jupyter。需要建立环境时，在仓库根目录执行 `python -m pip install -r Member5_file/requirements.txt`。嵌入文件从 5 号的交接包解压到 `data/emb/`，也可使用自定义 `DATA_ROOT`。要重新提取向量，在任一嵌入 Notebook 设置 `GENERATE_IF_MISSING = True`。两种表示共用同一次编码，生成的缓存可直接读入团队的 `src.cache`。

供 4 号合并：这里的 10 和 12 接入同一个 [缓存入口](sbert_cache.py) 和 [SBERT 编码实现](mvp.py)。24 的训练调用 [C3 训练与选模代码](c3_workflow.py)。整合到课程要求的单一 Notebook 时，将这些实现和用到的 `src` 函数一并嵌入。嵌入训练选项和设备设置写在对应单元格中。

[Seed 42 逐条预测](evidence/refinement_v1_20261112/selected_seed42_oof.csv)供错误分析与集成使用，对应[评分表](../../results/m5_textcnn_adopted.csv)。向量、原始 C3 权重和完整预测包由 YU Guanlin 另行交接。复跑输出在 `DATA_ROOT/local_artifacts/m5_c3/` 下。

[实验摘要](../../docs/experiments/m5.md) · [英文报告 3.3、3.4](../../docs/report_drafts/m5_sections.md)
