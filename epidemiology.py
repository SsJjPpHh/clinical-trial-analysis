# epidemiology.py  ───────────────────────────────────────────────
"""
流行病学分析模块（重构版）
Author : H
Date   : 2025-07-11
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
import statsmodels.api as sm
from typing import Tuple, List, Dict
from datetime import datetime


# ╭─────────────────── SessionState 数据接口 ───────────────────╮
def get_dataset() -> Tuple[pd.DataFrame | None, str]:
    ds = st.session_state.get("dataset_current")
    if ds:
        return ds["data"], ds["name"]
    return None, ""


# ╭───────────────────────── 工具函数 ───────────────────────────╮
def two_by_two(df: pd.DataFrame, exposure: str, outcome: str) -> pd.DataFrame:
    """
    返回 2×2 列联表:
                 outcome=1 | outcome=0
    exposure=1
    exposure=0
    """
    ct = pd.crosstab(df[exposure], df[outcome])
    # 确保行列顺序
    ct = ct.reindex(index=[1, 0], columns=[1, 0]).fillna(0).astype(int)
    return ct


def compute_rr(ct: pd.DataFrame) -> Tuple[float, Tuple[float, float]]:
    """相对危险度 RR 及 95%CI"""
    a, b = ct.loc[1, 1], ct.loc[1, 0]
    c, d = ct.loc[0, 1], ct.loc[0, 0]
    rr = (a / (a + b)) / (c / (c + d))
    se = np.sqrt(1 / a - 1 / (a + b) + 1 / c - 1 / (c + d))
    l, u = np.exp(np.log(rr) - 1.96 * se), np.exp(np.log(rr) + 1.96 * se)
    return rr, (l, u)


def compute_or(ct: pd.DataFrame) -> Tuple[float, Tuple[float, float]]:
    """比值比 OR 及 95%CI"""
    a, b = ct.loc[1, 1], ct.loc[1, 0]
    c, d = ct.loc[0, 1], ct.loc[0, 0]
    or_ = (a * d) / (b * c)
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    l, u = np.exp(np.log(or_) - 1.96 * se), np.exp(np.log(or_) + 1.96 * se)
    return or_, (l, u)


# ╭──────────────────── ① Cohort Study Analysis ──────────────────╮
def cohort_study_analysis(df: pd.DataFrame, exposure: str, outcome: str) -> None:
    st.markdown("#### 📈 队列研究分析")
    ct = two_by_two(df, exposure, outcome)
    rr, (l, u) = compute_rr(ct)
    chi2, p, _, _ = stats.chi2_contingency(ct)

    c1, c2 = st.columns(2)
    c1.metric("RR", f"{rr:.2f}", f"[{l:.2f}, {u:.2f}] 95%CI")
    c2.metric("χ² / P", f"{chi2:.2f}", f"P={p:.3g}")

    st.write("2×2 列联表")
    st.dataframe(ct)

    st.plotly_chart(
        px.bar(ct.reset_index().melt(id_vars=exposure, var_name=outcome, value_name="Count"),
               x=exposure, y="Count", color=outcome, barmode="group",
               title="发生率分布"), use_container_width=True
    )


# ╭─────────────────── ② Case-Control Study Analysis ─────────────╮
def case_control_analysis(df: pd.DataFrame, exposure: str, outcome: str) -> None:
    st.markdown("#### 🎲 病例-对照研究分析")
    ct = two_by_two(df, exposure, outcome)
    or_, (l, u) = compute_or(ct)
    chi2, p, _, _ = stats.chi2_contingency(ct)

    c1, c2 = st.columns(2)
    c1.metric("OR", f"{or_:.2f}", f"[{l:.2f}, {u:.2f}] 95%CI")
    c2.metric("χ² / P", f"{chi2:.2f}", f"P={p:.3g}")

    st.write("2×2 列联表")
    st.dataframe(ct)

    st.plotly_chart(
        px.bar(ct.T.reset_index().melt(id_vars=outcome, var_name=exposure, value_name="Count"),
               x=outcome, y="Count", color=exposure, barmode="group",
               title="暴露分布"), use_container_width=True
    )


# ╭─────────────────── ③ Cross-Sectional Study Analysis ──────────╮
def cross_sectional_analysis(df: pd.DataFrame, exposure: str, outcome: str) -> None:
    st.markdown("#### 🗂️ 横断面研究分析")
    ct = two_by_two(df, exposure, outcome)
    pr, (l, u) = compute_rr(ct)  # 横断面常用 PR，与 RR 公式相同
    chi2, p, _, _ = stats.chi2_contingency(ct)

    c1, c2 = st.columns(2)
    c1.metric("PR", f"{pr:.2f}", f"[{l:.2f}, {u:.2f}] 95%CI")
    c2.metric("χ² / P", f"{chi2:.2f}", f"P={p:.3g}")

    st.write("2×2 列联表")
    st.dataframe(ct)

    st.plotly_chart(
        px.bar(ct.reset_index().melt(id_vars=exposure, var_name=outcome, value_name="Count"),
               x=exposure, y="Count", color=outcome, barmode="group",
               title="患病率分布"), use_container_width=True
    )


# ╭─────────────────── ④ Logistic/Cox Regression ─────────────────╮
def multivariable_logistic(df: pd.DataFrame, outcome: str, covars: List[str]) -> None:
    st.markdown("#### 🧮 多因素 Logistic 回归")
    X = pd.get_dummies(df[covars], drop_first=True)
    X = sm.add_constant(X)
    y = df[outcome]

    model = sm.Logit(y, X).fit(disp=False)
    st.write(model.summary())

    or_ci = np.exp(model.conf_int().assign(OR=np.exp(model.params)))
    st.dataframe(or_ci.rename(columns={0: "2.5%", 1: "97.5%"}))


# ╭────────────────────────── UI 界面 ────────────────────────────╮
def epidemiology_ui() -> None:
    st.title("🩺 流行病学分析")
    st.markdown("*支持 Cohort / Case-Control / Cross-Sectional / Logistic*")

    df, name = get_dataset()
    if df is None:
        st.warning("请先在数据管理中心导入并选择数据集")
        return

    st.sidebar.header("🎛️ 设置")
    study_type = st.sidebar.selectbox(
        "研究设计",
        ("队列研究 (Cohort)", "病例-对照 (Case-Control)", "横断面 (Cross-Sectional)", "Logistic 回归")
    )

    # 字段选择
    binary_cols = [c for c in df.columns if df[c].dropna().isin([0, 1]).all()]
    if len(binary_cols) < 2:
        st.error("数据集中需至少存在两个二值变量（0/1）以作暴露和结局。")
        return

    exposure = st.sidebar.selectbox("暴露变量 (0/1)", binary_cols)
    outcome = st.sidebar.selectbox("结局变量 (0/1)", [c for c in binary_cols if c != exposure])

    if study_type == "队列研究 (Cohort)":
        cohort_study_analysis(df, exposure, outcome)
    elif study_type == "病例-对照 (Case-Control)":
        case_control_analysis(df, exposure, outcome)
    elif study_type == "横断面 (Cross-Sectional)":
        cross_sectional_analysis(df, exposure, outcome)
    else:
        # Logistic
        available_covars = [c for c in df.columns if c not in (outcome,)]
        covars = st.multiselect("协变量选择", available_covars)
        if covars:
            multivariable_logistic(df.dropna(subset=[outcome] + covars), outcome, covars)
        else:
            st.info("请选择 ≥1 个协变量后运行模型")


# ╭─────────────────────────── 调试入口 ─────────────────────────╮
if __name__ == "__main__":
    st.set_page_config(page_title="流行病学分析", layout="wide")
    epidemiology_ui()
