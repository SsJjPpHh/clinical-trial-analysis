# epidemiology.py
"""
流行病学分析模块（重构版）

直接执行：
-------------------------------------
import streamlit as st
from epidemiology import epidemiology_ui
epidemiology_ui()
-------------------------------------
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
from datetime import datetime
from typing import Dict, Tuple, List

# -------------------------------------------------
# 会话级数据集工具（与前两模块一致）
# -------------------------------------------------
def _session_dataset_key(name: str) -> str:
    return f"dataset_{name}"

def list_datasets() -> Dict[str, pd.DataFrame]:
    ds = {}
    for k, v in st.session_state.items():
        if k.startswith("dataset_") and isinstance(v, dict) and "data" in v:
            ds[v["name"]] = v["data"]
    return ds

def save_dataset(name: str, df: pd.DataFrame) -> None:
    st.session_state[_session_dataset_key(name)] = {
        "name": name,
        "data": df,
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# -------------------------------------------------
# 示例数据
# -------------------------------------------------
@st.cache_data(show_spinner=False)
def generate_epidemiology_sample(seed: int = 7) -> pd.DataFrame:
    """
    生成一个用于演示的队列研究数据集
      exposure  (1/0)
      outcome   (1/0)
      sex       (男/女)  — 用作分层变量示例
      age       连续
    """
    rng = np.random.default_rng(seed)
    n = 500
    exposure = rng.binomial(1, 0.45, n)
    baseline_risk = 0.08
    rr_true = 2.3
    outcome = rng.binomial(1, baseline_risk * np.where(exposure == 1, rr_true, 1), n)
    sex = rng.choice(["男", "女"], n)
    age = rng.integers(20, 70, n)
    return pd.DataFrame(
        {
            "exposure": exposure,
            "outcome": outcome,
            "sex": sex,
            "age": age,
        }
    )

# -------------------------------------------------
# 统计核心函数
# -------------------------------------------------
def _conf_int(log_est: float, se: float, alpha: float = 0.05) -> Tuple[float, float]:
    z = stats.norm.ppf(1 - alpha / 2)
    low, high = np.exp(log_est - z * se), np.exp(log_est + z * se)
    return low, high

# ---------- Cohort ----------
def cohort_analysis(df: pd.DataFrame, exp: str, out: str) -> Dict[str, float]:
    tab = pd.crosstab(df[exp], df[out])
    if tab.shape != (2, 2):
        raise ValueError("暴露或结局必须为二分类 (0/1)")
    risk_exp = tab.loc[1, 1] / tab.loc[1].sum()
    risk_un  = tab.loc[0, 1] / tab.loc[0].sum()
    rd = risk_exp - risk_un
    rr = risk_exp / risk_un if risk_un > 0 else np.nan
    # RD CI (Wald)
    se_rd = np.sqrt(
        risk_exp * (1 - risk_exp) / tab.loc[1].sum()
        + risk_un * (1 - risk_un) / tab.loc[0].sum()
    )
    rd_ci = (rd - 1.96 * se_rd, rd + 1.96 * se_rd)
    # RR CI (log method)
    se_rr = np.sqrt(1 / tab.loc[1, 1] - 1 / tab.loc[1].sum()
                    + 1 / tab.loc[0, 1] - 1 / tab.loc[0].sum())
    rr_ci = _conf_int(np.log(rr), se_rr) if rr > 0 else (np.nan, np.nan)
    # χ² or Fisher
    chi2, p, _, exp_freq = stats.chi2_contingency(tab)
    if (exp_freq < 5).any():
        _, p = stats.fisher_exact(tab)
    return {
        "Risk_exposed": risk_exp,
        "Risk_unexposed": risk_un,
        "Risk Difference": rd,
        "RD 95%CI low": rd_ci[0],
        "RD 95%CI high": rd_ci[1],
        "Risk Ratio": rr,
        "RR 95%CI low": rr_ci[0],
        "RR 95%CI high": rr_ci[1],
        "p value": p,
    }

# ---------- Case-Control ----------
def case_control_analysis(df: pd.DataFrame, exp: str, case: str) -> Dict[str, float]:
    """
    exp: 暴露变量 0/1
    case: 病例 (1) / 对照 (0)
    """
    tab = pd.crosstab(df[case], df[exp])  # 行: 病例
    if tab.shape != (2, 2):
        raise ValueError("暴露或病例变量必须为二分类 (0/1)")
    or_est = (tab.loc[1, 1] * tab.loc[0, 0]) / (tab.loc[1, 0] * tab.loc[0, 1])
    se_or = np.sqrt(np.sum(1 / tab.values))
    or_ci = _conf_int(np.log(or_est), se_or)
    chi2, p, _, exp_freq = stats.chi2_contingency(tab)
    if (exp_freq < 5).any():
        _, p = stats.fisher_exact(tab)
    return {
        "Odds Ratio": or_est,
        "OR 95%CI low": or_ci[0],
        "OR 95%CI high": or_ci[1],
        "p value": p,
    }

# ---------- Cross-Sectional ----------
def cross_sectional_analysis(df: pd.DataFrame, exp: str, dis: str) -> Dict[str, float]:
    """
    计算患病率比 (PR) / 患病率差 (PD)
    """
    tab = pd.crosstab(df[exp], df[dis])
    if tab.shape != (2, 2):
        raise ValueError("暴露或疾病变量必须为二分类 (0/1)")
    prev_exp = tab.loc[1, 1] / tab.loc[1].sum()
    prev_un  = tab.loc[0, 1] / tab.loc[0].sum()
    pdiff = prev_exp - prev_un
    pr = prev_exp / prev_un if prev_un > 0 else np.nan
    # CI
    se_pd = np.sqrt(
        prev_exp * (1 - prev_exp) / tab.loc[1].sum()
        + prev_un * (1 - prev_un) / tab.loc[0].sum()
    )
    pd_ci = (pdiff - 1.96 * se_pd, pdiff + 1.96 * se_pd)
    se_pr = np.sqrt(1 / tab.loc[1, 1] - 1 / tab.loc[1].sum()
                    + 1 / tab.loc[0, 1] - 1 / tab.loc[0].sum())
    pr_ci = _conf_int(np.log(pr), se_pr) if pr > 0 else (np.nan, np.nan)
    chi2, p, _, exp_freq = stats.chi2_contingency(tab)
    if (exp_freq < 5).any():
        _, p = stats.fisher_exact(tab)
    return {
        "Prevalence_exposed": prev_exp,
        "Prevalence_unexposed": prev_un,
        "Prevalence Difference": pdiff,
        "PD 95%CI low": pd_ci[0],
        "PD 95%CI high": pd_ci[1],
        "Prevalence Ratio": pr,
        "PR 95%CI low": pr_ci[0],
        "PR 95%CI high": pr_ci[1],
        "p value": p,
    }

# ---------- Mantel-Haenszel ----------
def mantel_haenszel_rr_or(
    df: pd.DataFrame, exp: str, out: str, strat: str, study_type: str = "cohort"
) -> Dict[str, float]:
    """
    基于多个分层表计算 MH 合并 RR 或 OR
    study_type: 'cohort' or 'casecontrol'
    """
    a, b, c, d = 0, 0, 0, 0  # a: exposed&disease
    for level, sub in df.groupby(strat):
        tab = pd.crosstab(sub[exp], sub[out])
        if tab.shape != (2, 2):
            continue
        a_i, b_i = tab.loc[1, 1], tab.loc[1, 0]
        c_i, d_i = tab.loc[0, 1], tab.loc[0, 0]
        a += a_i
        b += b_i
        c += c_i
        d += d_i
    if study_type == "cohort":
        rr_mh = (a / (a + b)) / (c / (c + d))
        se = np.sqrt((b / (a * (a + b))) + (d / (c * (c + d))))
        ci = _conf_int(np.log(rr_mh), se)
        return {"MH RR": rr_mh, "95%CI low": ci[0], "95%CI high": ci[1]}
    else:
        or_mh = (a * d) / (b * c)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        ci = _conf_int(np.log(or_mh), se)
        return {"MH OR": or_mh, "95%CI low": ci[0], "95%CI high": ci[1]}

# -------------------------------------------------
# UI 组件
# -------------------------------------------------
def epidemiology_ui() -> None:
    st.set_page_config("流行病学分析", "🦠", layout="wide")
    st.markdown("# 🦠 流行病学分析")
    st.sidebar.markdown("## 研究设计")
    design = st.sidebar.radio("选择研究设计", ["Cohort 队列", "Case-Control 病例-对照", "Cross-Sectional 横断面"])

    # 数据集
    datasets = list_datasets()
    if not datasets:
        st.warning("当前会话尚未加载数据")
        if st.button("🎲 生成示例数据"):
            save_dataset("epi_sample", generate_epidemiology_sample())
            st.experimental_rerun()
        return
    name = st.selectbox("选择数据集", list(datasets.keys()))
    df = datasets[name]

    with st.expander("👁️ 数据预览", False):
        st.dataframe(df.head())

    # 变量选择
    binary_cols = [c for c in df.columns if df[c].dropna().isin([0, 1]).all()]
    if not binary_cols:
        st.error("当前数据集不包含 0/1 二分类变量，无法继续")
        return

    if design.startswith("Cohort"):
        exp = st.selectbox("暴露变量 (0/1)", binary_cols, key="exp1")
        out = st.selectbox("结局变量 (0/1)", binary_cols, key="out1")
        strat = st.selectbox("可选分层变量 (None)", [None] + list(df.columns), index=0)
        if st.button("🚀 运行分析"):
            try:
                res = cohort_analysis(df, exp, out)
                st.success("计算完成")
                st.table(pd.DataFrame(res, index=["结果"]).T)

                # 可视化
                risks = {
                    "暴露组": res["Risk_exposed"],
                    "非暴露组": res["Risk_unexposed"],
                }
                fig = px.bar(x=list(risks.keys()), y=list(risks.values()),
                             labels={"x": "组别", "y": "Risk"},
                             text=[f"{v:.2%}" for v in risks.values()])
                st.plotly_chart(fig, use_container_width=True)

                # 分层
                if strat and strat != exp and strat != out:
                    mh = mantel_haenszel_rr_or(df, exp, out, strat, "cohort")
                    st.info(f"MH 合并 RR: {mh['MH RR']:.3f} (95%CI {mh['95%CI low']:.3f}–{mh['95%CI high']:.3f})")

            except Exception as e:
                st.error(f"分析失败: {e}")

    elif design.startswith("Case"):
        exp = st.selectbox("暴露变量 (0/1)", binary_cols, key="exp2")
        case = st.selectbox("病例变量 (1=病例,0=对照)", binary_cols, key="case2")
        strat = st.selectbox("可选分层变量 (None)", [None] + list(df.columns), index=0)
        if st.button("🚀 运行分析"):
            try:
                res = case_control_analysis(df, exp, case)
                st.table(pd.DataFrame(res, index=["结果"]).T)
                fig = px.histogram(df, x=exp, color=case, barmode="group",
                                   labels={exp: "暴露", case: "病例"}, category_orders={exp: [0, 1]})
                st.plotly_chart(fig, use_container_width=True)

                if strat and strat not in (exp, case):
                    mh = mantel_haenszel_rr_or(df, exp, case, strat, "casecontrol")
                    st.info(f"MH 合并 OR: {mh['MH OR']:.3f} (95%CI {mh['95%CI low']:.3f}–{mh['95%CI high']:.3f})")

            except Exception as e:
                st.error(f"分析失败: {e}")

    else:  # Cross-Sectional
        exp = st.selectbox("暴露变量 (0/1)", binary_cols, key="exp3")
        dis = st.selectbox("疾病/指标变量 (0/1)", binary_cols, key="dis3")
        if st.button("🚀 运行分析"):
            try:
                res = cross_sectional_analysis(df, exp, dis)
                st.table(pd.DataFrame(res, index=["结果"]).T)
                prevs = {
                    "暴露组": res["Prevalence_exposed"],
                    "非暴露组": res["Prevalence_unexposed"],
                }
                fig = px.bar(x=list(prevs.keys()), y=list(prevs.values()),
                             text=[f"{v:.2%}" for v in prevs.values()],
                             labels={"x": "组别", "y": "Prevalence"})
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"分析失败: {e}")

    # 下载结果按钮（自动合并表为一行）
    if "res" in locals():
        csv = pd.DataFrame(res, index=[0]).to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 下载结果 CSV", csv, file_name="epi_result.csv", mime="text/csv")

# -------------------------------------------------
if __name__ == "__main__":
    epidemiology_ui()
