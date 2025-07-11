# survival_analysis.py
"""
生存分析模块（Kaplan-Meier & Cox）

直接运行：
-------------------------------------
import streamlit as st
from survival_analysis import survival_ui
survival_ui()
-------------------------------------
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats as stats
from lifelines import KaplanMeierFitter, CoxPHFitter, statistics
from datetime import datetime
from typing import Dict, Tuple, List

# -------------------------------------------------
# Session 数据集工具（与其他模块一致）
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
def generate_survival_sample(seed: int = 2024) -> pd.DataFrame:
    """
    简单的两组生存数据：
      - time: 生存/随访时间
      - event: 1=死亡 / 0=删失
      - group: 0=对照 1=治疗
      - age, sex 作为协变量
    """
    rng = np.random.default_rng(seed)
    n = 200
    group = rng.binomial(1, 0.5, n)
    # 基线风险
    baseline_scale = 10
    hr_true = 0.6   # 处理组真 HR
    lam = baseline_scale * np.where(group == 1, hr_true, 1)
    time = rng.exponential(lam)
    censor = rng.exponential(15, n)
    observed_time = np.minimum(time, censor).round(1)
    event = (time <= censor).astype(int)
    age = rng.integers(40, 75, n)
    sex = rng.choice(["男", "女"], n)
    return pd.DataFrame(
        dict(time=observed_time, event=event, group=group, age=age, sex=sex)
    )

# -------------------------------------------------
# 核心函数
# -------------------------------------------------
def km_fit(df: pd.DataFrame, time: str, event: str, group: str | None = None):
    """
    返回 KM 拟合器字典 {label: kmf}
    """
    km_dict = {}
    if group:
        for level, sub in df.groupby(group):
            kmf = KaplanMeierFitter()
            kmf.fit(sub[time], sub[event], label=str(level))
            km_dict[str(level)] = kmf
    else:
        kmf = KaplanMeierFitter()
        kmf.fit(df[time], df[event], label="All")
        km_dict["All"] = kmf
    return km_dict


def km_plot(km_dict: Dict[str, KaplanMeierFitter]) -> go.Figure:
    fig = go.Figure()
    for label, kmf in km_dict.items():
        fig.add_scatter(
            x=kmf.survival_function_.index,
            y=kmf.survival_function_[label],
            mode="lines",
            name=label,
            step="post",
        )
    fig.update_layout(
        xaxis_title="Time",
        yaxis_title="Survival Probability",
        yaxis_range=[0, 1],
        template="plotly_white",
    )
    return fig


def logrank_p(df: pd.DataFrame, time: str, event: str, group: str) -> float:
    levels = df[group].unique()
    if len(levels) != 2:
        raise ValueError("Log-rank 仅支持 2 组比较")
    g0, g1 = levels
    res = statistics.logrank_test(
        df[df[group] == g0][time],
        df[df[group] == g1][time],
        df[df[group] == g0][event],
        df[df[group] == g1][event],
    )
    return res.p_value


def cox_fit(
    df: pd.DataFrame,
    time: str,
    event: str,
    covariates: List[str],
    strata: List[str] | None = None,
) -> CoxPHFitter:
    """
    返回拟合好的 CoxPHFitter
    """
    # 处理分类变量 → one-hot
    cat_cols = [c for c in covariates if df[c].dtype == "object" or df[c].dtype.name == "category"]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        covariates = [c for c in df.columns if c not in (time, event)]
    cph = CoxPHFitter()
    cph.fit(df[[time, event] + covariates], duration_col=time, event_col=event, strata=strata)
    return cph

# -------------------------------------------------
# UI
# -------------------------------------------------
def survival_ui() -> None:
    st.set_page_config("生存分析", "⏱️", layout="wide")
    st.markdown("# ⏱️ 生存分析")

    # 数据集
    datasets = list_datasets()
    if not datasets:
        st.warning("未检测到数据集")
        if st.button("🎲 生成示例数据"):
            save_dataset("surv_sample", generate_survival_sample())
            st.experimental_rerun()
        return

    name = st.selectbox("选择数据集", list(datasets.keys()))
    df = datasets[name]

    with st.expander("👁️ 数据预览", False):
        st.dataframe(df.head())

    # 变量选择
    time_col = st.selectbox("生存时间变量", df.columns)
    event_col = st.selectbox("结局事件变量 (1=事件,0=删失)", [c for c in df.columns if df[c].dropna().isin([0, 1]).all()])
    group_col = st.selectbox("分组变量 (可选)", [None] + list(df.columns), index=0)

    st.markdown("## Kaplan-Meier 曲线")
    if st.button("🚀 生成 KM 曲线"):
        try:
            km_dict = km_fit(df, time_col, event_col, group_col)
            fig = km_plot(km_dict)
            st.plotly_chart(fig, use_container_width=True)

            if group_col:
                p = logrank_p(df, time_col, event_col, group_col)
                st.info(f"Log-rank p-value = {p:.4f}")

        except Exception as e:
            st.error(f"KM 曲线生成失败: {e}")

    st.markdown("---")
    st.markdown("## Cox 比例风险模型")
    covariate_cols = st.multiselect(
        "选择协变量（至少 1 个）", [c for c in df.columns if c not in (time_col, event_col)]
    )
    strata_cols = st.multiselect("选择分层变量 (可选)", [c for c in df.columns if c not in covariate_cols])

    if st.button("🚀 拟合 Cox 模型"):
        if not covariate_cols:
            st.warning("请至少选择一个协变量")
        else:
            try:
                cph = cox_fit(df.copy(), time_col, event_col, covariate_cols, strata_cols or None)
                st.success("模型拟合完成")
                st.write("### HR 结果")
                st.dataframe(cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]])

                # 检验比例风险假设
                with st.expander("⏱️ PH 假设检验"):
                    ph = cph.check_assumptions(df, p_value_threshold=0.1, show_plots=False)
                    if ph is None:
                        st.success("通过：未发现显著比例风险违背")
            except Exception as e:
                st.error(f"Cox 模型拟合失败: {e}")

    # 下载
    if "cph" in locals():
        csv = cph.summary.reset_index().to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 下载 HR 结果 CSV", csv, file_name="cox_result.csv", mime="text/csv")

# -------------------------------------------------
if __name__ == "__main__":
    survival_ui()
