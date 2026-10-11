# -*- coding: utf-8 -*-
"""
00_config.py —— 全局配置
=========================
项目：企业数字化转型的文本测度构建及其对全要素生产率的影响
用途：统一路径、随机种子、样本口径与超参数，供五步流水线脚本共享。

运行环境：Python 3.10+，依赖见 requirements.txt
"""
from __future__ import annotations

from pathlib import Path

# ------------------------------------------------------------------
# 1. 路径
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]

DIR_CODE = ROOT / "code"
DIR_DICT = ROOT / "data" / "dictionaries"
DIR_RAW = ROOT / "data" / "raw"
DIR_SAMPLE = DIR_RAW / "annual_reports_sample"
DIR_PROC = ROOT / "data" / "processed"
DIR_DOCS = ROOT / "docs"
DIR_TABLES = ROOT / "output" / "tables"
DIR_FIGURES = ROOT / "output" / "figures"

for _d in (DIR_DICT, DIR_RAW, DIR_SAMPLE, DIR_PROC, DIR_DOCS, DIR_TABLES, DIR_FIGURES):
    _d.mkdir(parents=True, exist_ok=True)

# 字典 / 停用词
PATH_DIGITAL_DICT = DIR_DICT / "digital_dict.csv"
PATH_USERDICT = DIR_DICT / "digital_userdict.txt"        # jieba 自定义词典（由脚本生成）
PATH_STOP_GENERAL = DIR_DICT / "stopwords_general.txt"
PATH_STOP_FINANCE = DIR_DICT / "stopwords_finance.txt"

# 中间产物
PATH_CORPUS = DIR_RAW / "mda_corpus.csv"                  # 原始语料（步骤一）
PATH_CLEAN = DIR_PROC / "mda_clean.csv"                   # 清洗后文本（步骤二）
PATH_TOKENS = DIR_PROC / "tokens_freq.csv"                # 分词与词频（步骤二）
PATH_WORDFREQ = DIR_PROC / "word_freq_digital.csv"        # 词典命中明细（步骤三）
PATH_PANEL = DIR_PROC / "panel_firm_year.csv"             # 企业-年份面板（步骤四，核心交付）
PATH_PANEL_XLSX = DIR_PROC / "panel_firm_year.xlsx"       # 面板 Excel 版（步骤四）

# ------------------------------------------------------------------
# 2. 随机种子（保证全流程可复现）
# ------------------------------------------------------------------
SEED = 20261011

# ------------------------------------------------------------------
# 3. 样本口径（与论文"3.1 样本选择与数据来源"一致）
# ------------------------------------------------------------------
N_FIRMS = 300                     # 公司数
YEARS = list(range(2017, 2024))   # 2017—2023 年，共 7 年
N_OBS_EXPECTED = N_FIRMS * len(YEARS)   # 2,100 个企业-年度观测

# ------------------------------------------------------------------
# 4. 文本与测度超参数
# ------------------------------------------------------------------
SENT_MIN, SENT_MAX = 62, 88       # 单份 MD&A 句数区间
DIGI_BASE_P = 0.06                # 数字化句子的基准概率
DIGI_SLOPE = 0.18                 # 数字化句子概率对潜变量的敏感度
HITS_PER_SENT = (2, 4)            # 每个数字化句子命中的词典词数区间
TEXT_SAMPLE_N = 20                # 导出为 .txt 的样例年报份数

# ------------------------------------------------------------------
# 5. 分析设定
# ------------------------------------------------------------------
WINSOR_LO, WINSOR_HI = 0.01, 0.99   # 缩尾分位
MAIN_DV = "TFP_LP"                  # 主被解释变量
ROBUST_DV = "TFP_OP"                # 稳健性被解释变量
MAIN_IV = "Digital"                 # 主解释变量 = ln(1+词典命中词频)
