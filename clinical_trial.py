# clinical_trial.py
"""
临床试验分析模块 (全新重构版)

可直接在 Streamlit 运行:
-------------------------------------------------
import streamlit as st
from clinical_trial import clinical_trial_analysis
clinical_trial_analysis()
-------------------------------------------------
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from io import BytesIO
import scipy.stats as stats
from typing import Dict, List, Tuple

# ============ 公共工具函数 ============ #

@st.cache_data(show_spinner=False)
def generate_clinical_trial_sample_data(seed: int = 42) -> pd.DataFrame:
    """
    生成一个简单的临床试验示例数据集
    包含：
        - id
        - treatment  (0=对照, 1=试验)
        - age        连续
        - sex        分类
        - bmi        连续
        - primary_y  连续型主要终点
        - primary_b  二分类主要终点 (0/1)
    """
    rng = np.random.default_rng(seed)
    n = 120
    treatment = rng.integers(0, 2, n)
    age = rng.normal(55, 10, n).round(1)
    sex = rng.choice(["男", "女"], n)
    bmi = rng.normal(25, 4, n).round(1)
    # 连续型主要终点：假设试验组降低更多
    primary_y = rng.normal(120, 15, n) - treatment * rng.normal(8, 3, n)
    # 二分类主要终点：事件发生率
    baseline_risk = 0.30
    treatment_effect = 0.65  # 35% 相对风险降低
    primary_b = rng.binomial(
        1, p=np.where(treatment == 1, baseline_risk * treatment_effect, baseline_risk)
    )
    df = pd.DataFrame(
        {
            "id": range(1, n + 1),
            "treatment": treatment,
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "primary_y": primary_y.round(1),
            "primary_b": primary_b,
        }
    )
    return df


def _session_dataset_key(dataset_name: str) -> str:
    return f"dataset_{dataset_name}"


def save_dataset_to_session(name: str, df: pd.DataFrame) -> None:
    st.session_state[_session_dataset_key(name)] = {
        "name": name,
        "data": df,
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_available_datasets() -> Dict[str, pd.DataFrame]:
    """
    从 session_state 中提取所有数据集
    """
    ds = {}
    for k, v in st.session_state.items():
        if k.startswith("dataset_") and isinstance(v, dict) and "data" in v:
            ds[v["name"]] = v["data"]
    return ds


def to_csv_download(df: pd.DataFrame) -> Tuple[bytes, str]:
    """
    将 DataFrame 转成 csv bytes 及文件名
    """
    csv_bytes = df.to_csv(index=False).encode("utf-8-sig")
    file_name = f"result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return csv_bytes, file_name


def identify_baseline_variables(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    自动识别连续型 & 分类型基线变量
    - 连续变量: number & unique>5
    - 分类变量: object/category 或 unique<=5
    """
    continuous, categorical = [], []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].nunique(dropna=True) > 5:
                continuous.append(col)
            else:
                categorical.append(col)
        else:
            categorical.append(col)
    # 排除 id / primary / treatment
    exclude = {"id", "primary_y", "primary_b"}
    continuous = [c for c in continuous if c not in exclude]
    categorical = [c for c in categorical if c not in exclude | {"treatment"}]
    return continuous, categorical


# ============ 基线特征分析 ============ #


def analyze_continuous_baseline(
    df: pd.DataFrame, cont_vars: List[str], group_var: str = "treatment"
) -> pd.DataFrame:
    """
    对连续型基线变量比较：正态→t 检验，非正态→Mann-Whitney U
    返回长格式结果
    """
    results = []
    for var in cont_vars:
        group0 = df[df[group_var] == 0][var].dropna()
        group1 = df[df[group_var] == 1][var].dropna()
        # 正态性检验
        p_norm0 = stats.shapiro(group0)[1] if len(group0) >= 3 else 0
        p_norm1 = stats.shapiro(group1)[1] if len(group1) >= 3 else 0
        normal = (p_norm0 > 0.05) and (p_norm1 > 0.05)
        if normal:
            stat, p = stats.ttest_ind(group0, group1, equal_var=False)
            test = "t 检验"
        else:
            stat, p = stats.mannwhitneyu(group0, group1, alternative="two-sided")
            test = "Mann-Whitney U"
        results.append(
            {
                "变量": var,
                "组0 均值±SD": f"{group0.mean():.2f} ± {group0.std():.2f}",
                "组1 均值±SD": f"{group1.mean():.2f} ± {group1.std():.2f}",
                "检验": test,
                "p 值": round(p, 4),
            }
        )
    return pd.DataFrame(results)


def analyze_categorical_baseline(
    df: pd.DataFrame, cat_vars: List[str], group_var: str = "treatment"
) -> pd.DataFrame:
    """
    对分类型基线变量比较：行列 <2 或期望频数 <5 → Fisher，否则卡方
    """
    results = []
    for var in cat_vars:
        ct = pd.crosstab(df[var], df[group_var])
        # 只有一个水平 => 跳过
        if ct.shape[0] <= 1:
            continue
        chi2, p, dof, exp = stats.chi2_contingency(ct, correction=False)
        if (exp < 5).any():
            # Fisher 仅支持 2x2
            if ct.shape == (2, 2):
                stat, p = stats.fisher_exact(ct)
                test = "Fisher 精确检验"
            else:
                test = "卡方 (期望<5)"
        else:
            test = "Pearson 卡方"
        results.append(
            {
                "变量": var,
                "卡方/Fisher": test,
                "p 值": round(p, 4),
            }
        )
    return pd.DataFrame(results)


# ============ 主要终点分析 ============ #


def analyze_primary_continuous(
    df: pd.DataFrame, outcome: str = "primary_y", group_var: str = "treatment"
) -> Dict[str, float]:
    """
    两独立样本均值差 + 95% CI
    """
    g0 = df[df[group_var] == 0][outcome].dropna()
    g1 = df[df[group_var] == 1][outcome].dropna()
    diff = g1.mean() - g0.mean()
    se = np.sqrt(g0.var(ddof=1) / len(g0) + g1.var(ddof=1) / len(g1))
    ci_low, ci_high = stats.t.interval(0.95, df=len(g0) + len(g1) - 2, loc=diff, scale=se)
    t_stat, p_val = stats.ttest_ind(g1, g0, equal_var=False)
    return {
        "均值差": diff,
        "95% CI 下限": ci_low,
        "95% CI 上限": ci_high,
        "p 值": p_val,
    }


def analyze_primary_binary(
    df: pd.DataFrame, outcome: str = "primary_b", group_var: str = "treatment"
) -> Dict[str, float]:
    """
    风险比 + 95% CI（Wald）
    """
    tab = pd.crosstab(df[group_var], df[outcome])
    # 试验组在行1
    risk_treat = tab.loc[1, 1] / tab.loc[1].sum()
    risk_ctrl = tab.loc[0, 1] / tab.loc[0].sum()
    rr = risk_treat / risk_ctrl
    # 95% CI（ln(RR)±1.96*SE）
    se = np.sqrt(1 / tab.loc[1, 1] - 1 / tab.loc[1].sum() + 1 / tab.loc[0, 1] - 1 / tab.loc[0].sum())
    ci_low, ci_high = np.exp(np.log(rr) + np.array([-1, 1]) * 1.96 * se)
    # 卡方检验
    chi2, p_val, _, _ = stats.chi2_contingency(tab)
    return {
        "风险比": rr,
        "95% CI 下限": ci_low,
        "95% CI 上限": ci_high,
        "p 值": p_val,
    }


# ============ 主 UI 函数 ============ #


def clinical_trial_analysis() -> None:
    st.set_page_config(page_title="临床试验分析", layout="wide", page_icon="🧪")
    st.markdown("# 🧪 临床试验分析")
    st.markdown("*专业的临床试验数据分析工具*")

    # -------- 侧边栏导航 -------- #
    with st.sidebar:
        st.markdown("## 🧪 分析模块")
        analysis_type = st.radio(
            "选择分析类型",
            [
                "📊 基线特征分析",
                "🎯 主要终点分析",
                "📈 次要终点分析 (TODO)",
                "🛡️ 安全性分析 (TODO)",
                "🗂️ 亚组分析 (TODO)",
            ],
        )

    # -------- 数据集检查/选择/上传 -------- #
    datasets = get_available_datasets()
    if not datasets:
        st.warning("⚠️ 未检测到已加载的数据集")
        if st.button("🎲 生成示例数据", use_container_width=True):
            df_sample = generate_clinical_trial_sample_data()
            save_dataset_to_session("临床试验示例数据", df_sample)
            st.success("✅ 示例数据已生成并载入")
            st.rerun()
        return

    dataset_name = st.selectbox("选择数据集", list(datasets.keys()), index=0)
    df = datasets[dataset_name]

    with st.expander("👁️ 数据预览", expanded=False):
        st.dataframe(df.head())

    # -------- 各分析功能 -------- #
    if analysis_type == "📊 基线特征分析":
        baseline_tab(df)
    elif analysis_type == "🎯 主要终点分析":
        primary_endpoint_tab(df)
    else:
        st.info("🚧 敬请期待更多分析模块...")


# ============ 子页签函数 ============ #


def baseline_tab(df: pd.DataFrame) -> None:
    st.subheader("📊 基线特征分析")

    cont_vars, cat_vars = identify_baseline_variables(df)

    st.markdown("### 变量识别")
    st.write(f"连续型变量: {cont_vars}")
    st.write(f"分类变量: {cat_vars}")

    # 计算 & 展示
    st.markdown("### 结果")
    cont_res = analyze_continuous_baseline(df, cont_vars)
    cat_res = analyze_categorical_baseline(df, cat_vars)

    st.markdown("#### 连续变量比较")
    st.dataframe(cont_res, use_container_width=True)
    st.markdown("#### 分类变量比较")
    st.dataframe(cat_res, use_container_width=True)

    # 下载
    csv_bytes, fname = to_csv_download(
        pd.concat({"continuous": cont_res, "categorical": cat_res})
    )
    st.download_button("📥 下载结果 csv", csv_bytes, file_name=fname, mime="text/csv")


def primary_endpoint_tab(df: pd.DataFrame) -> None:
    st.subheader("🎯 主要终点分析")

    outcome_type = st.radio("选择主要终点类型", ["连续型", "二分类"], horizontal=True)

    if outcome_type == "连续型":
        outcome_col = st.selectbox("选择连续型主要终点变量", [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
        if outcome_col:
            res = analyze_primary_continuous(df, outcome=outcome_col)
            st.table(pd.DataFrame(res, index=["结果"]).T)

            # 绘图
            fig = px.box(df, x="treatment", y=outcome_col, points="all", color="treatment",
                         labels={"treatment": "Treatment group", outcome_col: outcome_col})
            st.plotly_chart(fig, use_container_width=True)

    else:  # 二分类
        outcome_col = st.selectbox("选择二分类主要终点变量", [c for c in df.columns if df[c].nunique() == 2])
        if outcome_col:
            res = analyze_primary_binary(df, outcome=outcome_col)
            st.table(pd.DataFrame(res, index=["结果"]).T)

            # 风险柱状图
            tab = pd.crosstab(df["treatment"], df[outcome_col], normalize="index")
            fig = go.Figure()
            fig.add_bar(x=["对照组", "试验组"], y=tab[1], name="事件发生率")
            fig.update_layout(yaxis_title="Risk", xaxis_title="组别")
            st.plotly_chart(fig, use_container_width=True)


# ============ 入口保护 ============ #
if __name__ == "__main__":
    clinical_trial_analysis()


            
