# -*- coding: utf-8 -*-
"""
04_measure_construction.py —— 第四步：指标构建（Measure Construction）
====================================================================
对应论文"3.5 指标构建"。

三个层次：
    L1 原始计数   RawCount  = 词典命中总词数
    L2 标准化比率 Ratio     = 命中词数 ÷ 有效词数 × 100（消除篇幅影响）
    L3 复合测度   Digital   = ln(1 + 命中词数)      ← 本文主测度
                  Digital_A = ln(1 + 底层技术运用词频)
                  Digital_B = ln(1 + 技术实践应用词频)

三个必须写进论文的决策：
    决策一 分母口径 —— 全文有效词数（去停用词后）；稳健性中改用总句数
    决策二 是否加权 —— 主测度等权；稳健性中以 TF-IDF 权重加权重构
    决策三 变量命名 —— 体现"构念 + 计算方式"

输出：
    data/processed/panel_firm_year.csv    企业-年份面板（最终测度结果表）
    data/processed/panel_firm_year.xlsx   Excel 版本
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
cfg = importlib.import_module("00_config")


def winsorize(s: pd.Series, lo: float, hi: float) -> pd.Series:
    return s.clip(lower=s.quantile(lo), upper=s.quantile(hi))


# ------------------------------------------------------------------
# 1. 合并"文本特征"与"财务面板"
# ------------------------------------------------------------------
feat = pd.read_csv(cfg.DIR_PROC / "doc_features.csv", encoding="utf-8-sig")
fin = pd.read_csv(cfg.DIR_RAW / "firm_financials.csv", encoding="utf-8-sig")
panel = feat.merge(fin, on=["firm_id", "year"], how="inner", validate="one_to_one")
print(f"[步骤4] 合并后面板：{len(panel)} 个企业-年度观测，{panel['firm_id'].nunique()} 家公司，"
      f"{panel['year'].nunique()} 个年度")

# ------------------------------------------------------------------
# 2. L1 / L2 / L3 三层测度
# ------------------------------------------------------------------
panel["RawCount"] = panel["hits_total"]                                   # L1
panel["Digital_ratio"] = panel["hits_total"] / panel["n_tokens"] * 100    # L2（分母=有效词数）
panel["Digital_sent_ratio"] = panel["hits_total"] / panel["n_sent"] * 100 # L2'（分母=句数）
panel["Digital"] = np.log1p(panel["hits_total"])                          # L3（主测度）
panel["Digital_A"] = np.log1p(panel["hits_A"])
panel["Digital_B"] = np.log1p(panel["hits_B"])
panel["Digital_dum"] = (panel["hits_total"] > panel["hits_total"].median()).astype(int)
panel["Digital_vec"] = panel["cos_digital"]
panel["Digital_topic"] = panel["topic_digital"]

# 复合测度：两维度标准化后等权加总
zA = (panel["Digital_A"] - panel["Digital_A"].mean()) / panel["Digital_A"].std()
zB = (panel["Digital_B"] - panel["Digital_B"].mean()) / panel["Digital_B"].std()
panel["Digital_comp"] = (0.5 * zA + 0.5 * zB)

# ------------------------------------------------------------------
# 3. 变量命名与整理
# ------------------------------------------------------------------
panel = panel.rename(columns={"n_tokens": "WordCnt", "n_sent": "SentCnt"})
CONTROLS = ["Size", "Lev", "ROA", "Growth", "Age", "Top1", "Dual", "Board"]
NUMERIC = ["RawCount", "Digital_ratio", "Digital_sent_ratio", "Digital", "Digital_A",
           "Digital_B", "Digital_comp", "Digital_vec", "Digital_topic",
           "WordCnt", "SentCnt", *CONTROLS, "RD", "Intangible", "ITexec",
           "DigitalInvest", "TFP_LP", "TFP_OP"]
for c in NUMERIC:
    panel[c] = winsorize(panel[c].astype(float), cfg.WINSOR_LO, cfg.WINSOR_HI)

panel = panel.sort_values(["firm_id", "year"]).reset_index(drop=True)
panel.to_csv(cfg.PATH_PANEL, index=False, encoding="utf-8-sig")
try:
    panel.to_excel(cfg.PATH_PANEL_XLSX, index=False)
    print(f"[步骤4] 已输出 Excel 版：{cfg.PATH_PANEL_XLSX.name}")
except Exception as e:  # pragma: no cover
    print(f"[步骤4] Excel 输出跳过（{e}）")

# ------------------------------------------------------------------
# 4. 测度摘要
# ------------------------------------------------------------------
summary = pd.DataFrame({
    "指标": ["命中词频 L1(RawCount)", "标准化比率 L2(Digital_ratio)", "对数测度 L3(Digital)",
             "底层技术运用(Digital_A)", "技术实践应用(Digital_B)", "复合测度(Digital_comp)"],
    "均值": [panel["RawCount"].mean(), panel["Digital_ratio"].mean(), panel["Digital"].mean(),
             panel["Digital_A"].mean(), panel["Digital_B"].mean(), panel["Digital_comp"].mean()],
    "标准差": [panel["RawCount"].std(), panel["Digital_ratio"].std(), panel["Digital"].std(),
               panel["Digital_A"].std(), panel["Digital_B"].std(), panel["Digital_comp"].std()],
    "最小值": [panel["RawCount"].min(), panel["Digital_ratio"].min(), panel["Digital"].min(),
               panel["Digital_A"].min(), panel["Digital_B"].min(), panel["Digital_comp"].min()],
    "最大值": [panel["RawCount"].max(), panel["Digital_ratio"].max(), panel["Digital"].max(),
               panel["Digital_A"].max(), panel["Digital_B"].max(), panel["Digital_comp"].max()],
}).round(3)
summary.to_csv(cfg.DIR_TABLES / "tab04_measure_summary.csv", index=False, encoding="utf-8-sig")

print(f"[步骤4] 主测度 Digital：均值 {panel['Digital'].mean():.3f}，"
      f"标准差 {panel['Digital'].std():.3f}，"
      f"分布 P25={panel['Digital'].quantile(.25):.2f} / P50={panel['Digital'].median():.2f} "
      f"/ P75={panel['Digital'].quantile(.75):.2f}")
print(f"[步骤4] 面板已输出：{cfg.PATH_PANEL.name}")
