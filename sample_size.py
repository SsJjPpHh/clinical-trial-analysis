# sample_size.py  ────────────────────────────────────────────────
"""
样本量计算模块（重构版）
Author : Your Name
Date   : 2025-07-11
"""
from __future__ import annotations
import streamlit as st
import numpy as np
from scipy.stats import norm
from statsmodels.stats.power import TTestIndPower, NormalIndPower, GofChisquarePower

# ╭─────────────────── 基础函数 ────────────────────╮
def two_mean_sample_size(alpha: float, power: float,
                         delta: float, sd: float, ratio: float = 1.0) -> tuple[int, int]:
    """
    双样本均值比较样本量
    """
    analysis = TTestIndPower()
    n1 = analysis.solve_power(effect_size=delta / sd,
                              alpha=alpha, power=power,
                              ratio=ratio, alternative="two-sided")
    n1 = int(np.ceil(n1))
    n2 = int(np.ceil(n1 * ratio))
    return n1, n2


def two_prop_sample_size(alpha: float, power: float,
                         p1: float, p2: float, ratio: float = 1.0) -> tuple[int, int]:
    """
    双比例相比样本量（无序列试验，常规近似）
    """
    effect = abs(p1 - p2)
    pooled = (p1 + p2) / 2
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    sd = np.sqrt(2 * pooled * (1 - pooled))
    n1 = (z_alpha + z_beta) ** 2 * sd**2 / effect**2
    n1 = int(np.ceil(n1))
    n2 = int(np.ceil(n1 * ratio))
    return n1, n2


# ╭─────────────────── UI ────────────────────────────╮
def sample_size_ui() -> None:
    st.title("📏 样本量计算")
    st.sidebar.header("参数设置")
    method = st.sidebar.selectbox("设计类型",
                                  ("两独立样本均值", "两独立样本比例", "单组率"))

    alpha = st.sidebar.number_input("α (显著性水平)", 0.001, 0.2, 0.05, 0.001)
    power = st.sidebar.number_input("1-β (检验功效)", 0.5, 0.99, 0.8, 0.01)
    ratio = st.sidebar.number_input("样本量比例 (组2/组1)", 0.1, 5.0, 1.0, 0.1)

    if method == "两独立样本均值":
        delta = st.number_input("期望差值 Δ", 0.01, 1e3, 5.0)
        sd = st.number_input("标准差 σ", 0.01, 1e3, 10.0)
        if st.button("计算样本量"):
            n1, n2 = two_mean_sample_size(alpha, power, delta, sd, ratio)
            st.success(f"组1：{n1}；组2：{n2} (总计 {n1+n2})")

    elif method == "两独立样本比例":
        p1 = st.number_input("p1 (对照组)", 0.0, 1.0, 0.5, 0.01)
        p2 = st.number_input("p2 (试验组)", 0.0, 1.0, 0.6, 0.01)
        if st.button("计算样本量"):
            n1, n2 = two_prop_sample_size(alpha, power, p1, p2, ratio)
            st.success(f"组1：{n1}；组2：{n2} (总计 {n1+n2})")

    else:  # 单组率
        p0 = st.number_input("参考率 p0", 0.0, 1.0, 0.5, 0.01)
        pA = st.number_input("期望率 pA", 0.0, 1.0, 0.6, 0.01)
        if st.button("计算样本量"):
            analysis = NormalIndPower()
            n = analysis.solve_power(effect_size=abs(pA - p0) /
                                     np.sqrt(p0 * (1 - p0)),
                                     alpha=alpha, power=power,
                                     alternative="two-sided")
            n = int(np.ceil(n))
            st.success(f"所需样本量：{n}")

if __name__ == "__main__":
    st.set_page_config(page_title="样本量计算", layout="wide")
    sample_size_ui()
