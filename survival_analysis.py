# survival_analysis.py  ──────────────────────────────────────────────
"""
生存分析模块（重构版）
Author : Your Name
Date   : 2025-07-11
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import Tuple, List
import plotly.express as px
from lifelines import KaplanMeierFitter, CoxPHFitter, statistics

# ╭─────────────────── Session 数据接口 ────────────────────╮
def get_dataset() -> Tuple[pd.DataFrame | None, str]:
    ds = st.session_state.get("dataset_current")
    if ds:
        return ds["data"], ds["name"]
    return None, ""


# ╭─────────────────── 绘图工具 ───────────────────────────╮
def plot_km(df: pd.DataFrame, time: str, status: str, group: str | None = None):
    kmf = KaplanMeierFitter()
    if group:
        fig = None
        for name, sub in df.groupby(group):
            kmf.fit(sub[time], sub[status], label=str(name))
            cur = kmf.plot_survival_function(ci_show=False)
            fig = cur.get_figure()
    else:
        kmf.fit(df[time], df[status], label="All")
        fig = kmf.plot_survival_function(ci_show=False).get_figure()
    return fig


# ╭─────────────────── UI ────────────────────────────╮
def survival_ui() -> None:
    st.title("⏳ 生存分析")
    df, name = get_dataset()
    if df is None:
        st.warning("请先在数据管理中心导入数据")
        return

    st.sidebar.header("设置")
    time_col = st.sidebar.selectbox("生存时间列", df.columns)
    status_col = st.sidebar.selectbox("结局状态列 (1=事件,0=删失)", df.columns)
    group_col = st.sidebar.selectbox("分组列 (可选)", ["(不分组)"] + list(df.columns))

    st.markdown(f"当前数据集：**{name}**")
    if st.button("绘制 KM 曲线"):
        grp = None if group_col == "(不分组)" else group_col
        fig = plot_km(df, time_col, status_col, grp)
        st.pyplot(fig)

        # Log-rank
        if grp:
            groups = df[grp].dropna().unique()
            if len(groups) == 2:
                ix = df[grp] == groups[0]
                res = statistics.logrank_test(
                    df[ix][time_col], df[~ix][time_col],
                    df[ix][status_col], df[~ix][status_col]
                )
                st.info(f"Log-rank P 值：{res.p_value:.4f}")
            else:
                st.info("分组数 >2，暂未计算 Log-rank。")

    # Cox 回归
    st.subheader("Cox 比例风险模型")
    covars = st.multiselect("协变量选择", [c for c in df.columns if c not in (time_col, status_col)])
    if covars and st.button("拟合 Cox 模型"):
        cph = CoxPHFitter()
        use_df = df[[time_col, status_col] + covars].dropna()
        cph.fit(use_df, duration_col=time_col, event_col=status_col)
        st.write(cph.summary)

if __name__ == "__main__":
    st.set_page_config(page_title="生存分析", layout="wide")
    survival_ui()
