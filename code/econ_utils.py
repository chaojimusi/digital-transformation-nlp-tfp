# -*- coding: utf-8 -*-
"""
econ_utils.py —— 计量分析公共函数
=================================
被 05_reliability_validity.py / 06_regression_analysis.py 共同调用。

包含：
    * fe_ols        面板固定效应 OLS（行业/年份虚拟变量），公司层面聚类稳健标准误
    * star          显著性星号
    * coef_row      把回归结果格式化为论文表格的一行
    * corr_matrix   相关系数矩阵（Pearson 下三角 / Spearman 上三角）
    * sobel         Sobel 中介效应检验
    * bootstrap_mediation  Bootstrap 中介效应检验
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def fe_ols(df: pd.DataFrame, y: str, xs: list[str],
           fe: tuple[str, ...] = ("C(industry)", "C(year)"),
           cluster: str = "firm_id", weights: str | None = None):
    """固定效应 OLS + 聚类稳健标准误。"""
    rhs = " + ".join(xs + list(fe)) if fe else " + ".join(xs)
    formula = f"{y} ~ {rhs}"
    model = smf.ols(formula, data=df)
    return model.fit(cov_type="cluster", cov_kwds={"groups": df[cluster]})


def star(p: float) -> str:
    return "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))


def coef_row(res, var: str, digits: int = 3) -> tuple[str, str]:
    """返回 (系数字符串, t值字符串)。"""
    b = res.params.get(var, np.nan)
    t = res.tvalues.get(var, np.nan)
    p = res.pvalues.get(var, np.nan)
    return f"{b:.{digits}f}{star(p)}", f"({t:.2f})"


def corr_matrix(df: pd.DataFrame, cols: list[str], method: str = "pearson") -> pd.DataFrame:
    return df[cols].corr(method=method)


def corr_with_p(df: pd.DataFrame, x: str, y: str, method: str = "pearson"):
    from scipy import stats
    d = df[[x, y]].dropna()
    if method == "pearson":
        r, p = stats.pearsonr(d[x], d[y])
    else:
        r, p = stats.spearmanr(d[x], d[y])
    return r, p, len(d)


def sobel(a: float, sa: float, b: float, sb: float) -> tuple[float, float, float]:
    """Sobel 检验：返回 (间接效应, z 值, p 值)。"""
    from scipy import stats
    ind = a * b
    se = np.sqrt(a ** 2 * sb ** 2 + b ** 2 * sa ** 2)
    z = ind / se if se > 0 else np.nan
    p = 2 * (1 - stats.norm.cdf(abs(z))) if not np.isnan(z) else np.nan
    return ind, z, p


def bootstrap_mediation(df: pd.DataFrame, y: str, x: str, m: str,
                        controls: list[str], fe: tuple[str, ...],
                        n_boot: int = 500, seed: int = 20261011):
    """Bootstrap 中介效应（百分位法，返回间接效应 95% 置信区间）。"""
    rng = np.random.default_rng(seed)
    firms = df["firm_id"].unique()
    inds = []
    for _ in range(n_boot):
        pick = rng.choice(firms, size=len(firms), replace=True)
        boot = pd.concat([df[df["firm_id"] == f] for f in pick], ignore_index=True)
        if boot["firm_id"].nunique() < 5:
            continue
        try:
            a = fe_ols(boot, m, [x] + controls, fe, cluster="firm_id").params[x]
            b = fe_ols(boot, y, [x, m] + controls, fe, cluster="firm_id").params[m]
            inds.append(a * b)
        except Exception:
            continue
    inds = np.array(inds)
    lo, hi = np.percentile(inds, [2.5, 97.5]) if len(inds) else (np.nan, np.nan)
    return float(np.mean(inds)) if len(inds) else np.nan, float(lo), float(hi), len(inds)
