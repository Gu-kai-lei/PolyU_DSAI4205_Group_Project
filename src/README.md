# 全组共用函数

Notebook 通过这里的函数读取数据、计算指标和保存缓存。

| 文件 | 用途 |
|---|---|
| `config.py` | 路径、随机种子、标签和版本号 |
| `data.py` | 读取数据并检查清洗结果与划分是否一致 |
| `evaluate.py` | 生成 run_id、统一评分、保存结果和绘制混淆矩阵 |
| `cache.py` | 按样本 ID 对齐嵌入，按 run_id 保存预测概率 |
| `cleaning.py` | 数据清洗流程、清洗日志、`text_full` 生成和近似重复分组 |
| `fasttext_emb.py` | 按折训练 FastText，生成平均和 SIF 句向量及 token 向量序列 |

修改公共函数的参数或返回值前，请先与维护人协调，并同步受影响的 Notebook。

