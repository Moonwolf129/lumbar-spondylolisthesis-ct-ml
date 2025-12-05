# config.py
from pathlib import Path

# -------- 基本路径 --------
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# 自动建目录（第一次运行有用）
for _d in [RESULTS_DIR, FIGURES_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

# -------- 随机种子和划分比例 --------
RANDOM_STATE = 42
TEST_SIZE = 0.30

# -------- 目标变量 --------
TARGET_SLIP_BIN = "slip_bin"
TARGET_SLIP_GRADE = "slip_grade"
TARGET_SLIP_LEVEL = "slip_level"

# -------- 关键分层变量 --------
GROUP_SEX = "sex"
GROUP_AGE = "age"
GROUP_PARS = "pars_defect"

# -------- 默认数据文件名 --------
DEFAULT_DATA_FILE = DATA_DIR / "lumbar_spondylolisthesis_dataset.csv"

# -------- 数值特征列表（可根据你的真实数据改） --------
NUMERIC_FEATURES = [
    "age", "height_m", "weight_kg", "BMI",
    "L3_SMA", "L3_SMI",
    "L3_SATA", "L3_SATI",
    "L3_VATA", "L3_VATI",
    "L3_IMATA", "L3_IMATI",
    "L3_SMD", "SATD", "VATD", "IMATD",
    "PV_muscle_area", "PV_fat_area",
    "PV_muscle_density", "PV_fat_density",
    "PMFI"
]

# -------- 类别特征（0/1 或类别编码） --------
CATEGORICAL_FEATURES = [
    "sex",        # F/M 或 0/1
    "pars_defect" # 0/1
]

# -------- 部分相关中重点关注的变量 --------
PARTIAL_CORR_VARS = [
    "L3_IMATI",
    "L3_SMD",
    "PMFI",
    "PV_fat_area"
]

# -------- 机器学习使用的特征（可以和 NUMERIC+CAT 合并） --------
ML_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
