# 1 号的 baseline 代码

`notebooks/01_data/03_baseline.ipynb` 调用这里的实现。最终提交单个 notebook 时，由 4 号把这里的函数一并合并进去。

| 文件 | 内容 |
|---|---|
| `baseline.py` | TF-IDF + 逻辑回归：按折训练、OOF 预测、评分、混淆矩阵、每类权重最高的词 |
| `test_baseline.py` | 检查 id 和分组是否对齐、同组是否跨折、词表是否只来自训练折，以及保存流程 |
| `outputs/` | 每次运行的完整输出，按 run_id 分文件夹，不进 Git |

在仓库根目录、激活 `dsai4205` 环境后运行：

```bash
python member1/baseline.py --check
python -m unittest member1.test_baseline -v
python member1/baseline.py --mode team
```

`--check` 只检查清洗数据、正式划分和包版本，不训练。`--mode provisional` 用临时划分调试，结果不写进 `results/` 和预测缓存。
