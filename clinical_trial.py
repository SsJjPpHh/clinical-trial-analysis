# clinical_trial.py  ───────────────────────────────────────────────
"""
临床试验分析模块（重构版）
Author : H
Date   : 2025-07-11
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import scipy.stats as stats
from typing import Dict, List, Tuple
from datetime import datetime

# ╭───────────────────── 工具函数区域 ─────────────────────╮
@st.cache_data(show_spinner=False)
def get_available_datasets() -> Dict[str, Dict]:
    """
    从 session_state 中检索所有 `dataset_*` 数据集
    Returns
    -------
    dict : {display_name: {"data": DataFrame, ...}, …}
    """
    datasets: Dict[str, Dict] = {}
    for key, val in st.session_state.items():
        if key.startswith("dataset_") and isinstance(val, dict) and "data" in val:
            display_name = val.get("name", key.replace("dataset_", ""))
            datasets[display_name] = val
    return datasets


def validate_clinical_data(df: pd.DataFrame) -> bool:
    """
    基础校验：空表 / 行列阈值 / 必要列预警
    """
    if df.empty:
        st.error("❌ 数据为空，请检查数据源。")
        return False
    if len(df) < 10:
        st.warning("⚠️ 样本量 < 10，统计结果可能不稳定。")
    return True


def split_variables(df: pd.DataFrame, cat_th: int = 10) -> Tuple[List[str], List[str]]:
    """
    根据数据类型与唯一值数量，自动分为分类 / 连续变量
    cat_th : 若唯一值 ≤ cat_th 或 dtype=object，则视作分类
    """
    cat_vars, cont_vars = [], []
    for col in df.columns:
        if df[col].dtype == "O" or df[col].nunique(dropna=True) <= cat_th:
            cat_vars.append(col)
        else:
            cont_vars.append(col)
    return cat_vars, cont_vars


# ╭─────────────────── 统计分析子模块区域 ──────────────────╮
def baseline_characteristics(df: pd.DataFrame, group_col: str) -> None:
    """
    1. 分类变量   →  频数 + 卡方/Fisher
    2. 连续变量   →  均值±SD + t 检验 / Mann–Whitney
    """
    st.subheader("📊 基线特征分析")

    cat_vars, cont_vars = split_variables(df.drop(columns=[group_col]))
    grp_values = df[group_col].dropna().unique().tolist()
    if len(grp_values) != 2:
        st.error("目前仅支持两个组的比较，请确认分组列。")
        return

    # 分类变量
    if cat_vars:
        st.markdown("#### 1️⃣ 分类变量 (Cardinalities)")
        cat_table = []
        for v in cat_vars:
            tbl = pd.crosstab(df[v], df[group_col])
            chi2, p, _, _ = stats.chi2_contingency(tbl)
            cat_table.append({
                "变量": v,
                "卡方 χ²": round(chi2, 2),
                "P 值": f"{p:.3g}"
            })
        st.dataframe(pd.DataFrame(cat_table))

    # 连续变量
    if cont_vars:
        st.markdown("#### 2️⃣ 连续变量 (Means ± SD)")
        cont_table = []
        for v in cont_vars:
            g1, g2 = (df[df[group_col] == grp_values[0]][v].dropna(),
                      df[df[group_col] == grp_values[1]][v].dropna())
            # 正态性检验
            if stats.shapiro(g1).pvalue > .05 and stats.shapiro(g2).pvalue > .05:
                stat, p = stats.ttest_ind(g1, g2, equal_var=False)
                test = "t-test"
            else:
                stat, p = stats.mannwhitneyu(g1, g2, alternative="two-sided")
                test = "Mann-Whitney"
            cont_table.append({
                "变量": v,
                f"{grp_values[0]} 均值±SD": f"{g1.mean():.2f} ± {g1.std():.2f}",
                f"{grp_values[1]} 均值±SD": f"{g2.mean():.2f} ± {g2.std():.2f}",
                test: round(stat, 2),
                "P 值": f"{p:.3g}"
            })
        st.dataframe(pd.DataFrame(cont_table))


def primary_endpoint(df: pd.DataFrame, group_col: str, endpoint_col: str) -> None:
    """
    主要终点分析：连续型终点 → 均值差；二分类终点 → RR & χ²
    """
    st.subheader("🎯 主要终点分析")

    if df[endpoint_col].dtype == "O" or df[endpoint_col].nunique() <= 2:
        # 二分类终点
        tbl = pd.crosstab(df[group_col], df[endpoint_col])
        rr = (tbl.iloc[1, 1] / tbl.iloc[1].sum()) / (tbl.iloc[0, 1] / tbl.iloc[0].sum())
        chi2, p, _, _ = stats.chi2_contingency(tbl)
        st.write("**风险比 RR:**", f"{rr:.2f}")
        st.write("**卡方检验 χ² / P:**", f"{chi2:.2f} / {p:.3g}")
        st.dataframe(tbl)
    else:
        # 连续型终点
        groups = df[group_col].unique().tolist()
        g1 = df[df[group_col] == groups[0]][endpoint_col].dropna()
        g2 = df[df[group_col] == groups[1]][endpoint_col].dropna()
        diff = g1.mean() - g2.mean()
        stat, p = stats.ttest_ind(g1, g2, equal_var=False)
        st.metric("均值差", f"{diff:.2f}")
        st.write("t-test", f"{stat:.2f} (P={p:.3g})")
        st.plotly_chart(
            px.box(df, x=group_col, y=endpoint_col, points="all",
                   color=group_col, title="主要终点分布")
        )


# ── 其余分析入口（次要终点 / 安全性 / …）保留占位 ─────────────
def secondary_endpoint(*_):      st.info("次要终点分析待实现…")
def safety_analysis(*_):         st.info("安全性分析待实现…")
def subgroup_analysis(*_):       st.info("亚组分析待实现…")
def time_trend_analysis(*_):     st.info("时间趋势分析待实现…")
def sensitivity_analysis(*_):    st.info("敏感性分析待实现…")
def trial_summary_report(*_):    st.info("试验总结报告待实现…")


# ╭───────────────────────── UI 主入口 ─────────────────────────╮
def clinical_trial_analysis() -> None:
    st.title("🧬 临床试验分析")
    st.markdown("*专业的临床试验数据分析工具*")

    # ── 侧边栏导航
    with st.sidebar:
        st.header("🔧 分析模块")
        analysis_type = st.selectbox(
            "选择分析类型",
            ("基线特征分析", "主要终点分析", "次要终点分析",
             "安全性分析", "亚组分析", "时间趋势分析",
             "敏感性分析", "试验总结报告")
        )

    # ── 数据源
    datasets = get_available_datasets()
    if not datasets:
        st.warning("请先在数据管理模块导入临床试验数据。")
        return

    selected_name = st.selectbox("📂 选择数据集", options=list(datasets.keys()))
    df = datasets[selected_name]["data"]

    if not validate_clinical_data(df):
        return

    # ── 选择分组 & 终点列
    group_col = st.selectbox("🧑‍🤝‍🧑 分组列", df.columns, index=0)
    endpoint_col = None
    if analysis_type in ("主要终点分析", "次要终点分析"):
        endpoint_col = st.selectbox("🎯 终点列", df.columns)

    # ── 调用对应分析
    if analysis_type == "基线特征分析":
        baseline_characteristics(df, group_col)
    elif analysis_type == "主要终点分析":
        primary_endpoint(df, group_col, endpoint_col)          # type: ignore[arg-type]
    elif analysis_type == "次要终点分析":
        secondary_endpoint()
    elif analysis_type == "安全性分析":
        safety_analysis()
    elif analysis_type == "亚组分析":
        subgroup_analysis()
    elif analysis_type == "时间趋势分析":
        time_trend_analysis()
    elif analysis_type == "敏感性分析":
        sensitivity_analysis()
    elif analysis_type == "试验总结报告":
        trial_summary_report()


# ╭───────────────────────── 调试入口 ─────────────────────────╮
if __name__ == "__main__":
    st.set_page_config(page_title="临床试验分析", layout="wide")
    clinical_trial_analysis()


            
