# 原始数据、清洗数据和缓存

下载仓库后，默认从这里读取原始数据和当前清洗版本（`src/config.py` 的 `CLEAN_VERSION`，现在是 2）。

```text
data/
├─ raw/
│  └─ stock_market_crash_2022.csv  原始数据，保持不变
├─ clean_v2/                       当前版本
│  ├─ clean_v2.csv                 清洗后的数据，load_clean() 读的就是它
│  ├─ clean_v2_all.csv             全部 1,698 行原始推文，带 keep 和 drop_reason，用于对比实验和核查
│  ├─ clean_v2_log.md              清洗日志、列说明和每条规则的依据
│  └─ clean_v2.py                  生成以上三个文件的脚本
└─ clean_v1/                       上一版，存档
   ├─ clean_v1.csv
   └─ cleaning_log.md
```

每个版本的文件夹自带生成它的脚本或日志；`docs/cleaning_log.md` 保留当前版本日志的副本，便于撰写报告。重新生成当前版本：在仓库根目录运行 `python data/clean_v2/clean_v2.py`，结果和仓库里的文件逐字节一致。

`load_clean()` 读取当前版本；`load_clean(1)` 读取 `clean_v1/clean_v1.csv`（列名是旧的 `text_full` / `text_light`）。生成新版本时新建 `clean_v版本号/` 文件夹，并在 `.gitignore` 里放行它。

嵌入缓存 `emb/`、预测结果 `preds/`、模型权重和 NLTK 语料不提交到 Git。需要使用共享网盘时设置 `DATA_ROOT`，并在网盘中保持相同目录结构。

不要直接用 Excel 打开或保存这些 CSV：会出现乱码，`@user` 开头的单元格会被当成公式。查看请用 JupyterLab 或 pandas。
