"""全局路径、常量和版本号。所有 notebook 都从这里导入，不要在代码里写死路径。

- 原始数据和各版本清洗数据（data/clean_vN/）随仓库保存；嵌入和预测缓存放在本地或共享网盘
- splits / results / figures 是小文件，跟着仓库一起做版本管理
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # 仓库根目录

# 版本号：升级规则见 README 的"数据和实验版本"一节
CLEAN_VERSION = 2  # 当前使用的清洗数据 -> DATA_ROOT/clean_v2/clean_v2.csv
SPLIT_VERSION = 1  # 当前使用的数据划分 -> splits/folds_v1.csv


# 默认读取仓库里的 data/；需要使用共享网盘时设置环境变量 DATA_ROOT
DATA_ROOT = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))
RAW_CSV = DATA_ROOT / "raw" / "stock_market_crash_2022.csv"  # 原始数据，只读，不要修改
CLEAN_DIR = DATA_ROOT / f"clean_v{CLEAN_VERSION}"  # 清洗后的数据（2 号产出）
EMB_DIR = DATA_ROOT / "emb"      # 嵌入缓存（1、2、5 号产出）
PRED_DIR = DATA_ROOT / "preds"   # 每次实验的 OOF 概率和测试集概率，文件名是 run_id

# 小文件跟着仓库走
SPLITS_DIR = PROJECT_ROOT / "splits"    # 固定数据划分（4 号产出）
RESULTS_DIR = PROJECT_ROOT / "results"  # 每个模型的评分结果
FIGURES_DIR = PROJECT_ROOT / "figures"  # 报告和 PPT 用图


SEED = 42        # 全组统一的随机种子
N_FOLDS = 5      # 交叉验证折数
TEST_SIZE = 0.2  # 测试集比例

# 标签编码，来自数据集说明；预测概率的列顺序也统一按这个顺序
LABELS = {0: "Neutral", 1: "Positive", 2: "Negative"}

# 每条评分记录和每个预测文件都会附上这些信息，用来确认结果来自哪一版数据和划分
VERSION_TAGS = {
    "data_version": f"clean_v{CLEAN_VERSION}",
    "split_version": f"folds_v{SPLIT_VERSION}",
    "seed": SEED,
}
