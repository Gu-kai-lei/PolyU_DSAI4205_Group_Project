# 原始数据、清洗数据和缓存

下载仓库后，默认从这里读取原始数据和当前清洗版本。

```text
data/
├─ raw/
│  └─ stock_market_crash_2022.csv  原始数据，保持不变
└─ clean_v1/
   ├─ clean_v1.csv                清洗后的第 1 版数据
   └─ cleaning_log.md             第 1 版清洗日志
```

原始 CSV 来自课程提供的数据包；`clean_v1.csv` 和日志按已提供的文件原样保存。最新日志在 `docs/cleaning_log.md` 保留副本，便于撰写报告。

`src/config.py` 的 `CLEAN_VERSION` 指定当前清洗版本。`load_clean(1)` 读取 `clean_v1/clean_v1.csv`；生成新版本时使用对应的 `clean_v版本号/` 文件夹。

嵌入缓存 `emb/`、预测结果 `preds/`、模型权重和 NLTK 语料不提交到 Git。需要使用共享网盘时设置 `DATA_ROOT`，并在网盘中保持相同目录结构。

不要直接用EXCEL打开
