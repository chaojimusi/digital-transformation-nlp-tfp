# -*- coding: utf-8 -*-
"""
05_reliability_validity.py —— 第五步：信效度检验（Reliability & Validity）
========================================================================
对应论文"3.6 测度的信度与效度检验"。

四类效度 + 三类信度：
    内容效度  Content Validity      词典构成、来源文献、跨方法标注一致性(Kappa)
    聚合效度  Convergent Validity   与同类/近义测度、数字化痕迹指标的相关性
    区分效度  Discriminant Validity 与"不该相关"的变量低相关
    预测效度  Predictive Validity   对研发投入与全要素生产率的预测能力
    信度      分半信度 / 内部一致性 Cronbach's α / 重测信度

输出：output/tables/tab05_*.csv、output/figures/fig05_*.png
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
cfg = importlib.import_module("00_config")
import nlp_utils
import econ_utils

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

panel = pd.read_csv(cfg.PATH_PANEL, encoding="utf-8-sig")
clean = pd.read_csv(cfg.PATH_CLEAN, encoding="utf-8-sig")
dictionary = pd.read_csv(cfg.PATH_DIGITAL_DICT, encoding="utf-8")
TERMS = dictionary["word"].astype(str).tolist()
BY_FIRST = nlp_utils.build_first_bucket(TERMS)
rng = np.random.default_rng(cfg.SEED)

out_content, out_reliab, out_corr, out_pred, out_quint, out_var = ([] for _ in range(6))

# ==================================================================
# 一、内容效度
# ==================================================================
# 1.1 词典结构
struct = (dictionary.groupby(["dimension", "dimension_cn"])
          .agg(词数=("word", "count"), 来源文献数=("source", "nunique")).reset_index())
struct.loc[len(struct)] = ["合计", "—", len(dictionary), dictionary["source"].nunique()]
struct.to_csv(cfg.DIR_TABLES / "tab05a_dict_structure.csv", index=False, encoding="utf-8-sig")
for _, r in struct.iterrows():
    out_content.append({"检验项": f"词典构成[{r['dimension']}]", "结果": f"{r['词数']}词 / {r['来源文献数']}来源"})

# 1.2 跨方法标注一致性（Cohen's Kappa）：词典法 vs 词嵌入法 vs 主题模型法
def kappa(a: np.ndarray, b: np.ndarray) -> float:
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    return (po - pe) / (1 - pe)


lab_dict = (panel["Digital"] > panel["Digital"].median()).astype(int).values
lab_vec = (panel["Digital_vec"] > panel["Digital_vec"].median()).astype(int).values
lab_top = (panel["Digital_topic"] > panel["Digital_topic"].median()).astype(int).values
k1, k2 = kappa(lab_dict, lab_vec), kappa(lab_dict, lab_top)
out_content.append({"检验项": "跨方法一致性 Kappa(词典法 vs 词嵌入法)", "结果": f"{k1:.3f}"})
out_content.append({"检验项": "跨方法一致性 Kappa(词典法 vs 主题模型法)", "结果": f"{k2:.3f}"})

# 1.3 仿真校验：测度对潜变量的"回收率"（本仓库为合成语料，用于验证流水线有效性）
truth = pd.read_csv(cfg.DIR_RAW / "_simulation_truth.csv", encoding="utf-8-sig")
m = panel.merge(truth, on=["firm_id", "year"], how="left")
rho, p_rho = stats.spearmanr(m["Digital"], m["dig_std"])
out_content.append({"检验项": "测度-潜变量秩相关(仿真校验,Spearman ρ)", "结果": f"{rho:.3f} (p<0.01)" if p_rho < 0.01 else f"{rho:.3f}"})
pd.DataFrame(out_content).to_csv(cfg.DIR_TABLES / "tab05b_content_validity.csv", index=False, encoding="utf-8-sig")

# ==================================================================
# 二、信度
# ==================================================================
# 把步骤三输出的"企业-年度-词"命中明细透视成 文档×词 计数矩阵，
# 每个词典词即一个"题项"，据此计算分半信度与内部一致性。
freq = pd.read_csv(cfg.PATH_WORDFREQ, encoding="utf-8-sig")
DT = (freq.pivot_table(index=["firm_id", "year"], columns="word",
                       values="count", aggfunc="sum", fill_value=0)
      .sort_index())
TERM_COLS = [w for w in TERMS if w in DT.columns]
DT = DT.reindex(columns=TERM_COLS, fill_value=0)
S = DT.values.astype(float)                       # (n_docs, n_terms)
print(f"[步骤5] 计数矩阵：{S.shape[0]} 份文档 × {S.shape[1]} 个词典词")

# 2.1 分半信度（题项随机对半，200 次，取平均）
split_rs = []
for _ in range(200):
    perm = rng.permutation(S.shape[1])
    h = S.shape[1] // 2
    s1 = np.log1p(S[:, perm[:h]].sum(axis=1))
    s2 = np.log1p(S[:, perm[h:]].sum(axis=1))
    split_rs.append(stats.pearsonr(s1, s2)[0])
r_half = float(np.mean(split_rs))
sb = 2 * r_half / (1 + r_half)     # Spearman-Brown 校正
out_reliab.append({"信度指标": "分半信度(随机对半200次均值 r)", "取值": round(r_half, 3)})
out_reliab.append({"信度指标": "Spearman-Brown 校正后半信度", "取值": round(sb, 3)})

# 2.2 内部一致性 Cronbach's α
# (a) 以 5 个随机子词典为 5 个"题项"
K_ITEM = 5
perm = rng.permutation(S.shape[1])
sub_scores = np.column_stack([
    np.log1p(S[:, perm[k::K_ITEM]].sum(axis=1)) for k in range(K_ITEM)])
k = sub_scores.shape[1]
alpha5 = k / (k - 1) * (1 - sub_scores.var(axis=0, ddof=1).sum()
                        / sub_scores.sum(axis=1).var(ddof=1))
# (b) 以全部词典词为"题项"
kall = S.shape[1]
alpha_all = kall / (kall - 1) * (1 - S.var(axis=0, ddof=1).sum() / S.sum(axis=1).var(ddof=1))
out_reliab.append({"信度指标": f"Cronbach's α（{K_ITEM}个子词典）", "取值": round(float(alpha5), 3)})
out_reliab.append({"信度指标": f"Cronbach's α（{kall}个词典词题项）", "取值": round(float(alpha_all), 3)})
alpha = alpha5

# 2.3 重测信度（同一公司的 t 期与 t-1 期测度相关性）
p2 = panel.sort_values(["firm_id", "year"]).copy()
p2["Digital_lag"] = p2.groupby("firm_id")["Digital"].shift(1)
retest = p2.dropna(subset=["Digital_lag"])
r_test = stats.pearsonr(retest["Digital"], retest["Digital_lag"])[0]
out_reliab.append({"信度指标": "重测信度(t 期与 t-1 期相关 r)", "取值": round(float(r_test), 3)})
pd.DataFrame(out_reliab).to_csv(cfg.DIR_TABLES / "tab05c_reliability.csv", index=False, encoding="utf-8-sig")

# ==================================================================
# 三、聚合效度与区分效度
# ==================================================================
CONVERGENT = ["Digital_vec", "Digital_topic", "Intangible", "ITexec", "DigitalInvest"]
DISCRIMINANT = ["Size", "Lev", "Board", "Top1", "Dual", "Growth"]
for v in CONVERGENT:
    r, p, n_ = econ_utils.corr_with_p(panel, "Digital", v, "pearson")
    out_corr.append({"变量": v, "类型": "聚合效度", "Pearson r": round(r, 3),
                     "p值": round(p, 4), "N": n_, "显著性": econ_utils.star(p)})
for v in DISCRIMINANT:
    r, p, n_ = econ_utils.corr_with_p(panel, "Digital", v, "pearson")
    out_corr.append({"变量": v, "类型": "区分效度", "Pearson r": round(r, 3),
                     "p值": round(p, 4), "N": n_, "显著性": econ_utils.star(p)})
pd.DataFrame(out_corr).to_csv(cfg.DIR_TABLES / "tab05d_corr_validity.csv", index=False, encoding="utf-8-sig")

# ==================================================================
# 四、预测效度：对研发投入与全要素生产率的回归
# ==================================================================
CTRL = ["Size", "Lev", "ROA", "Growth", "Age", "Top1", "Dual", "Board", "SOE", "HiTech"]
FE = ("C(industry)", "C(year)")
for dv in ["RD", "TFP_LP"]:
    res = econ_utils.fe_ols(panel, dv, ["Digital"] + CTRL, FE, "firm_id")
    c, t = econ_utils.coef_row(res, "Digital")
    out_pred.append({"被解释变量": dv, "Digital系数": c, "t值": t,
                     "R²": round(res.rsquared, 3), "N": int(res.nobs)})
res_rf = econ_utils.fe_ols(panel, "TFP_LP", ["Digital", "RD"] + CTRL, FE, "firm_id")
c, t = econ_utils.coef_row(res_rf, "RD")
out_pred.append({"被解释变量": "TFP_LP(含RD)", "RD系数": c, "t值": t,
                 "R²": round(res_rf.rsquared, 3), "N": int(res_rf.nobs)})
pd.DataFrame(out_pred).to_csv(cfg.DIR_TABLES / "tab05e_predictive_validity.csv", index=False, encoding="utf-8-sig")

# ==================================================================
# 五、条件均值检验（五分组）
# ==================================================================
panel["Digital_q"] = pd.qcut(panel["Digital"], 5, labels=["Q1(低)", "Q2", "Q3", "Q4", "Q5(高)"])
q = panel.groupby("Digital_q", observed=True)[["TFP_LP", "RD", "Intangible", "ITexec", "Size"]].mean().reset_index()
q = q.rename(columns={"Digital_q": "数字化转型分组"})
q.to_csv(cfg.DIR_TABLES / "tab05f_quintile_means.csv", index=False, encoding="utf-8-sig")

fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
x = np.arange(len(q))
bars = ax1.bar(x, q["TFP_LP"], width=0.5, color="#C0392B", alpha=0.85, label="TFP_LP 均值（左轴）")
ax1.set_ylabel("TFP_LP 均值", color="#C0392B")
ax1.set_xticks(x); ax1.set_xticklabels(q["数字化转型分组"])
ax1.set_ylim(q["TFP_LP"].min() * 0.97, q["TFP_LP"].max() * 1.01)
ax2 = ax1.twinx()
ax2.plot(x, q["RD"], marker="o", color="#1F4E79", linewidth=2, label="研发强度 RD（右轴）")
ax2.set_ylabel("研发投入强度(%)", color="#1F4E79")
ax1.set_title("图 数字化转型程度分组的 TFP 与研发投入条件均值")
fig.tight_layout()
fig.savefig(cfg.DIR_FIGURES / "fig05a_quintile.png", dpi=200)
plt.close(fig)

# ==================================================================
# 六、方差分解（组间/组内，二者之和恒等于总方差）
# ==================================================================
gm = panel["Digital"].mean()
firm_mean = panel.groupby("firm_id")["Digital"].transform("mean")
firm_n = panel.groupby("firm_id")["Digital"].transform("size")
between = ((firm_mean - gm) ** 2).mean()                                   # 组间方差（等权）
within = ((panel["Digital"] - firm_mean) ** 2).mean()                      # 组内方差
total = between + within
out_var.append({"方差来源": "企业间(Between)", "方差": round(between, 4),
                "占比": f"{between / total * 100:.1f}%"})
out_var.append({"方差来源": "企业内(Within)", "方差": round(within, 4),
                "占比": f"{within / total * 100:.1f}%"})
out_var.append({"方差来源": "合计", "方差": round(total, 4), "占比": "100.0%"})
pd.DataFrame(out_var).to_csv(cfg.DIR_TABLES / "tab05g_variance_decomposition.csv",
                            index=False, encoding="utf-8-sig")

print("[步骤5] 信效度检验完成")
print(f"  内容效度：词典 {len(dictionary)} 词；Kappa(词典法-词嵌入法)={k1:.3f}；Kappa(词典法-主题法)={k2:.3f}")
print(f"  信度：分半 r={r_half:.3f}（SB校正 {sb:.3f}）；Cronbach α={alpha:.3f}；重测 r={r_test:.3f}")
print(f"  聚合/区分效度：见表 tab05d")
print(f"  预测效度：见表 tab05e")
print(f"  方差分解：企业间占比 {between/total*100:.1f}%")
