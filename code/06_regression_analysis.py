# -*- coding: utf-8 -*-
"""
06_regression_analysis.py —— 实证分析：基准回归 / 稳健性 / 中介 / 异质性
=======================================================================
对应论文"第四章 实证结果与分析"。

    * 描述性统计与相关系数矩阵
    * 基准回归（逐步加入控制变量与固定效应）
    * 稳健性检验（替换被解释变量、替换核心解释变量、更换分母、滞后、剔除疫情年、
      企业固定效应、PSM 倾向得分匹配、工具变量 2SLS）
    * 中介效应（研发投入渠道，逐步法 + Sobel + Bootstrap）
    * 异质性分析（产权性质、高新技术资质、测度维度分解）

输出：output/tables/tab0*.csv、output/figures/fig0*.png
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
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parent))
cfg = importlib.import_module("00_config")
import econ_utils as eu

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

panel = pd.read_csv(cfg.PATH_PANEL, encoding="utf-8-sig")
CTRL = ["Size", "Lev", "ROA", "Growth", "Age", "Top1", "Dual", "Board", "SOE", "HiTech"]
FE = ("C(industry)", "C(year)")
DV, IV = cfg.MAIN_DV, cfg.MAIN_IV
KEY_VARS = ["TFP_LP", "TFP_OP", "Digital", "Digital_A", "Digital_B", "Digital_ratio",
            "RD", "Intangible", "ITexec"] + CTRL


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


results_md = []

# ==================================================================
# 一、描述性统计
# ==================================================================
desc = pd.DataFrame({
    "变量": KEY_VARS,
    "N": [panel[v].notna().sum() for v in KEY_VARS],
    "均值": [panel[v].mean() for v in KEY_VARS],
    "标准差": [panel[v].std() for v in KEY_VARS],
    "最小值": [panel[v].min() for v in KEY_VARS],
    "P25": [panel[v].quantile(.25) for v in KEY_VARS],
    "中位数": [panel[v].median() for v in KEY_VARS],
    "P75": [panel[v].quantile(.75) for v in KEY_VARS],
    "最大值": [panel[v].max() for v in KEY_VARS],
}).round(3)
desc.to_csv(cfg.DIR_TABLES / "tab02_descriptive.csv", index=False, encoding="utf-8-sig")
results_md.append("### 表2 主要变量描述性统计\n\n" + md_table(desc))

# ==================================================================
# 二、相关系数矩阵（下三角 Pearson / 上三角 Spearman）
# ==================================================================
pear = panel[KEY_VARS].corr(method="pearson")
spear = panel[KEY_VARS].corr(method="spearman")
mat = pear.copy()
for i, a in enumerate(KEY_VARS):
    for j, b in enumerate(KEY_VARS):
        if j > i:
            mat.loc[a, b] = spear.loc[a, b]
        if i == j:
            mat.loc[a, b] = 1.0
mat.round(3).to_csv(cfg.DIR_TABLES / "tab03_correlation.csv", encoding="utf-8-sig")

# ==================================================================
# 三、基准回归
# ==================================================================
specs = [
    ("(1) 仅核心解释变量", [IV], ()),
    ("(2) 加入控制变量", [IV] + CTRL, ()),
    ("(3) 加入行业/年份固定效应", [IV] + CTRL, FE),
]
base_rows, models = [], {}
for name, xs, fe in specs:
    res = eu.fe_ols(panel, DV, xs, fe, "firm_id")
    models[name] = res
    c, t = eu.coef_row(res, IV)
    base_rows.append({"变量": IV, name: f"{c} {t}", "R²": round(res.rsquared, 3), "N": int(res.nobs)})

# (4) 企业 + 年份 双向固定效应（SOE/HiTech 为不随时间变化的公司特征，
#     会被企业固定效应完全吸收，故直接从控制变量中剔除）
CTRL_TV = [c for c in CTRL if c not in ("SOE", "HiTech")]
res_firm = eu.fe_ols(panel, DV, [IV] + CTRL_TV, ("C(firm_id)", "C(year)"), "firm_id")
models["(4) 企业+年份固定效应"] = res_firm
c, t = eu.coef_row(res_firm, IV)
base_rows.append({"变量": IV, "(4) 企业+年份固定效应": f"{c} {t}",
                  "R²": round(res_firm.rsquared, 3), "N": int(res_firm.nobs)})

base_tbl = pd.DataFrame(base_rows)
base_tbl.to_csv(cfg.DIR_TABLES / "tab06a_baseline.csv", index=False, encoding="utf-8-sig")
results_md.append("### 表6 基准回归结果（被解释变量：TFP_LP）\n\n" + md_table(base_tbl))

main_res = models["(3) 加入行业/年份固定效应"]
BETA_MAIN, T_MAIN = main_res.params[IV], main_res.tvalues[IV]

# ==================================================================
# 四、稳健性检验
# ==================================================================
rob = []

# 4.1 替换被解释变量
r = eu.fe_ols(panel, "TFP_OP", [IV] + CTRL, FE, "firm_id")
c, t = eu.coef_row(r, IV)
rob.append({"检验": "替换被解释变量：TFP_OP（OP法）", "核心系数": c, "t值": t, "N": int(r.nobs)})

# 4.2 替换核心解释变量口径
for alt, lab in [("Digital_ratio", "更换口径：词频占有效词数比率×100"),
                 ("Digital_sent_ratio", "更换分母：词频占总句数"),
                 ("RawCount", "更换层次：L1 原始词频"),
                 ("Digital_dum", "更换形式：数字化强度虚拟变量（高于中位数=1）"),
                 ("Digital_comp", "复合测度：两维度标准化加总")]:
    r = eu.fe_ols(panel, DV, [alt] + CTRL, FE, "firm_id")
    c, t = eu.coef_row(r, alt)
    rob.append({"检验": lab, "核心系数": c, "t值": t, "N": int(r.nobs)})

# 4.3 更换测度构造路线
for alt, lab in [("Digital_vec", "更换技术路线：词嵌入法余弦相似度"),
                 ("Digital_topic", "更换技术路线：LDA 主题比例")]:
    r = eu.fe_ols(panel, DV, [alt] + CTRL, FE, "firm_id")
    c, t = eu.coef_row(r, alt)
    rob.append({"检验": lab, "核心系数": c, "t值": t, "N": int(r.nobs)})

# 4.4 滞后一期自变量（缓解反向因果）
pl = panel.sort_values(["firm_id", "year"]).copy()
pl["Digital_lag"] = pl.groupby("firm_id")["Digital"].shift(1)
pl1 = pl.dropna(subset=["Digital_lag"])
r = eu.fe_ols(pl1, DV, ["Digital_lag"] + CTRL, FE, "firm_id")
c, t = eu.coef_row(r, "Digital_lag")
rob.append({"检验": "解释变量滞后一期（t-1 期文本）", "核心系数": c, "t值": t, "N": int(r.nobs)})

# 4.5 剔除 2020 年（疫情冲击）
r = eu.fe_ols(panel[panel["year"] != 2020], DV, [IV] + CTRL, ("C(industry)", "C(year)"), "firm_id")
c, t = eu.coef_row(r, IV)
rob.append({"检验": "剔除 2020 年（新冠疫情）样本", "核心系数": c, "t值": t, "N": int(r.nobs)})

# 4.6 工具变量 2SLS（同行业-同年度其他公司数字化均值）
g = panel.groupby(["industry", "year"])["Digital"]
panel["Digital_iv"] = (g.transform("sum") - panel["Digital"]) / (g.transform("count") - 1)
D = panel.dropna(subset=["Digital_iv"] + [DV] + CTRL).copy()
dum = pd.get_dummies(D[["industry", "year"]].astype(str), drop_first=True).astype(float)
X = sm.add_constant(D[[IV] + CTRL].astype(float))
X = pd.concat([X, dum], axis=1)
Z = sm.add_constant(D[["Digital_iv"] + CTRL].astype(float))
Z = pd.concat([Z, dum], axis=1)
# ----- 两阶段最小二乘（手工实现，便于控制标准误与汇报一阶段 F）-----
# 第一阶段：Digital ~ Z（含工具变量与全部外生变量），取拟合值
fs = sm.OLS(X[IV].astype(float), Z.astype(float)).fit()
D["Digital_hat"] = fs.fittedvalues.values
first_F = float(fs.tvalues["Digital_iv"] ** 2)
# 第二阶段：TFP ~ Digital_hat + 外生变量（标准误按公司聚类）
X2 = sm.add_constant(D[["Digital_hat"] + CTRL].astype(float))
X2 = pd.concat([X2, dum], axis=1)
iv_res = sm.OLS(D[DV].astype(float), X2.astype(float)).fit(
    cov_type="cluster", cov_kwds={"groups": D["firm_id"]})
iv_b = float(iv_res.params["Digital_hat"])
iv_t = float(iv_res.tvalues["Digital_hat"])
iv_p = float(iv_res.pvalues["Digital_hat"])
rob.append({"检验": f"工具变量 2SLS（行业-年度均值，一阶段F={first_F:.1f}）",
            "核心系数": f"{iv_b:.3f}{eu.star(iv_p)}",
            "t值": f"({iv_t:.2f})", "N": int(iv_res.nobs)})

# 4.7 PSM 倾向得分匹配
Dm = panel.copy()
Dm["treat"] = (Dm["Digital"] > Dm["Digital"].median()).astype(int)
ps_model = LogisticRegression(max_iter=2000).fit(Dm[CTRL], Dm["treat"])
Dm["pscore"] = ps_model.predict_proba(Dm[CTRL])[:, 1]
cal = 0.05 * Dm["pscore"].std()
treated = Dm[Dm["treat"] == 1]
control = Dm[Dm["treat"] == 0]
nn = NearestNeighbors(n_neighbors=1).fit(control[["pscore"]].values)
dist, idx = nn.kneighbors(treated[["pscore"]].values)
keep = dist[:, 0] <= cal
matched = pd.concat([treated[keep], control.iloc[idx[keep, 0]]], ignore_index=True)
r = eu.fe_ols(matched, DV, ["treat"] + CTRL, FE, "firm_id")
c, t = eu.coef_row(r, "treat")
rob.append({"检验": f"PSM 倾向得分匹配（1:1 近邻，保留 {len(matched)} 观测）",
            "核心系数": c, "t值": t, "N": int(r.nobs)})

rob_tbl = pd.DataFrame(rob)
rob_tbl.to_csv(cfg.DIR_TABLES / "tab06b_robustness.csv", index=False, encoding="utf-8-sig")
results_md.append("### 表7 稳健性检验\n\n" + md_table(rob_tbl))

# ==================================================================
# 五、中介效应：研发投入渠道
# ==================================================================
med = []
r1 = eu.fe_ols(panel, DV, [IV] + CTRL, FE, "firm_id")          # c
r2 = eu.fe_ols(panel, "RD", [IV] + CTRL, FE, "firm_id")        # a
r3 = eu.fe_ols(panel, DV, [IV, "RD"] + CTRL, FE, "firm_id")    # c', b
med.append({"模型": "步骤一 TFP←Digital (总效应 c)", "Digital": f"{eu.coef_row(r1, IV)[0]} {eu.coef_row(r1, IV)[1]}"})
med.append({"模型": "步骤二 RD←Digital (路径 a)", "Digital": f"{eu.coef_row(r2, IV)[0]} {eu.coef_row(r2, IV)[1]}"})
med.append({"模型": "步骤三 TFP←Digital+RD (路径 c'/b)",
            "Digital": f"{eu.coef_row(r3, IV)[0]} {eu.coef_row(r3, IV)[1]}",
            "RD": f"{eu.coef_row(r3, 'RD')[0]} {eu.coef_row(r3, 'RD')[1]}"})
a_, sa = r2.params[IV], r2.bse[IV]
b_, sb = r3.params["RD"], r3.bse["RD"]
ind, z, p = eu.sobel(a_, sa, b_, sb)
med.append({"模型": f"Sobel 检验：间接效应 a×b={ind:.4f}，z={z:.2f}，p={p:.4f}",
            "Digital": f"中介占比 {ind/r1.params[IV]*100:.1f}%"})
boot_mean, blo, bhi, nb = eu.bootstrap_mediation(panel, DV, IV, "RD", CTRL, FE, n_boot=300, seed=cfg.SEED)
med.append({"模型": f"Bootstrap 中介（{nb}次）：间接效应={boot_mean:.4f}，95%CI=[{blo:.4f}, {bhi:.4f}]"})
med_tbl = pd.DataFrame(med).fillna("")
med_tbl.to_csv(cfg.DIR_TABLES / "tab07_mediation.csv", index=False, encoding="utf-8-sig")
results_md.append("### 表8 中介效应检验（研发投入渠道）\n\n" + md_table(med_tbl))

# ==================================================================
# 六、异质性分析
# ==================================================================
het = []
for var, lab, hi in [("SOE", "产权性质", 1), ("HiTech", "高新技术资质", 1)]:
    # 分组回归时剔除该分组变量本身（组内为常数，会造成完全共线）
    ctrl_sub = [c for c in CTRL if c != var]
    sub1 = panel[panel[var] == hi]
    sub0 = panel[panel[var] == 0]
    r_1 = eu.fe_ols(sub1, DV, [IV] + ctrl_sub, FE, "firm_id")
    r_0 = eu.fe_ols(sub0, DV, [IV] + ctrl_sub, FE, "firm_id")
    panel["_inter"] = panel[IV] * panel[var]
    r_int = eu.fe_ols(panel, DV, [IV, var, "_inter"] + ctrl_sub, FE, "firm_id")
    c1, t1 = eu.coef_row(r_1, IV)
    c0, t0 = eu.coef_row(r_0, IV)
    ci, ti = eu.coef_row(r_int, "_inter")
    name1 = f"{lab}={hi}" + ("（国企）" if var == "SOE" and hi == 1 else ("（高新技术）" if hi == 1 else ""))
    name0 = f"{lab}=0" + ("（非国企）" if var == "SOE" else ("（非高新技术）" if var == "HiTech" else ""))
    het.append({"分组": name1, "Digital系数": c1, "t值": t1, "N": int(r_1.nobs)})
    het.append({"分组": name0, "Digital系数": c0, "t值": t0, "N": int(r_0.nobs)})
    het.append({"分组": f"交互项检验 Digital×{var}", "Digital系数": ci, "t值": ti, "N": int(r_int.nobs)})

# 维度分解
for alt, lab in [("Digital_A", "底层技术运用维度"), ("Digital_B", "技术实践应用维度")]:
    r = eu.fe_ols(panel, DV, [alt] + CTRL, FE, "firm_id")
    c, t = eu.coef_row(r, alt)
    het.append({"分组": f"维度分解：{lab}", "Digital系数": c, "t值": t, "N": int(r.nobs)})
het_tbl = pd.DataFrame(het)
het_tbl.to_csv(cfg.DIR_TABLES / "tab08_heterogeneity.csv", index=False, encoding="utf-8-sig")
results_md.append("### 表9 异质性与维度分解\n\n" + md_table(het_tbl))

# ==================================================================
# 七、图形
# ==================================================================
# 7.1 年度趋势
trend = panel.groupby("year").agg(Digital=("Digital", "mean"), TFP=("TFP_LP", "mean")).reset_index()
fig, ax1 = plt.subplots(figsize=(7.2, 4.2))
ax1.plot(trend["year"], trend["Digital"], marker="s", color="#C0392B", linewidth=2, label="Digital 均值（左轴）")
ax1.set_xlabel("年度"); ax1.set_ylabel("Digital（ln(1+词频)）均值", color="#C0392B")
ax2 = ax1.twinx()
ax2.plot(trend["year"], trend["TFP"], marker="o", color="#1F4E79", linewidth=2, label="TFP_LP 均值（右轴）")
ax2.set_ylabel("TFP_LP 均值", color="#1F4E79")
ax1.set_title("图 2017—2023 年企业数字化转型程度与全要素生产率的年度趋势")
fig.tight_layout()
fig.savefig(cfg.DIR_FIGURES / "fig06a_trend.png", dpi=200)
plt.close(fig)

# 7.2 系数森林图
items = [("基准回归", BETA_MAIN, main_res.bse[IV]),
         ("TFP_OP", eu.fe_ols(panel, "TFP_OP", [IV] + CTRL, FE, "firm_id").params[IV],
          eu.fe_ols(panel, "TFP_OP", [IV] + CTRL, FE, "firm_id").bse[IV]),
         ("比率口径", eu.fe_ols(panel, DV, ["Digital_ratio"] + CTRL, FE, "firm_id").params["Digital_ratio"],
          eu.fe_ols(panel, DV, ["Digital_ratio"] + CTRL, FE, "firm_id").bse["Digital_ratio"]),
         ("词嵌入法", eu.fe_ols(panel, DV, ["Digital_vec"] + CTRL, FE, "firm_id").params["Digital_vec"],
          eu.fe_ols(panel, DV, ["Digital_vec"] + CTRL, FE, "firm_id").bse["Digital_vec"]),
         ("主题模型法", eu.fe_ols(panel, DV, ["Digital_topic"] + CTRL, FE, "firm_id").params["Digital_topic"],
          eu.fe_ols(panel, DV, ["Digital_topic"] + CTRL, FE, "firm_id").bse["Digital_topic"])]
labels = [i[0] for i in items]
betas = np.array([i[1] for i in items])
ses = np.array([i[2] for i in items])
fig, ax = plt.subplots(figsize=(7.2, 3.8))
y = np.arange(len(labels))
ax.errorbar(betas, y, xerr=1.96 * ses, fmt="o", color="#C0392B", ecolor="#7F8C8D", capsize=4)
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
ax.set_yticks(y); ax.set_yticklabels(labels)
ax.set_xlabel("核心解释变量系数（95% 置信区间）")
ax.set_title("图 主效应系数在不同测度口径下的稳健性")
fig.tight_layout()
fig.savefig(cfg.DIR_FIGURES / "fig06b_forest.png", dpi=200)
plt.close(fig)

# ==================================================================
# 八、输出关键结果摘要
# ==================================================================
summary = (
    f"基准回归：Digital 系数 = {BETA_MAIN:.4f}（t = {T_MAIN:.2f}），"
    f"R² = {main_res.rsquared:.3f}，N = {int(main_res.nobs)}\n"
    f"中介效应：a = {a_:.4f}，b = {b_:.4f}，间接效应 = {ind:.4f}（Sobel z = {z:.2f}，p = {p:.4f}），"
    f"中介占比 {ind/r1.params[IV]*100:.1f}%\n"
    f"Bootstrap 中介 95% CI = [{blo:.4f}, {bhi:.4f}]\n"
    f"工具变量一阶段 F = {first_F:.1f}\n"
)
(cfg.DIR_TABLES / "results_summary.txt").write_text(summary, encoding="utf-8")
(cfg.DIR_DOCS / ".." / "output" / "tables" / "results_tables.md").write_text(
    "# 实证结果表（自动生成）\n\n" + "\n\n".join(results_md) + "\n", encoding="utf-8")

print("[步骤6] 实证分析完成")
print(summary)
