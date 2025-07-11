# sample_size.py
"""
样本量计算模块

场景：
1. 两样本均数差 (t 检验)
2. 两比例差 (χ² / Z)
3. 两组生存 (log-rank, HR)
"""

from __future__ import annotations

import streamlit as st
import numpy as np
from scipy.stats import norm
from statsmodels.stats.power import (
    TTestIndPower,
    NormalIndPower,
)

# ---------- 公式 ---------- #
def two_mean_sample_size(delta: float, sd: float, alpha: float, power: float, ratio: float) -> float:
    obj = TTestIndPower()
    return obj.solve_power(
        effect_size=delta / sd, alpha=alpha, power=power, ratio=ratio, alternative="two-sided"
    )

def two_prop_sample_size(p1: float, p2: float, alpha: float, power: float, ratio: float) -> float:
    obj = NormalIndPower()
    effect = abs(p1 - p2)
    pooled = (p1 + ratio * p2) / (1 + ratio)
    sd = np.sqrt(pooled * (1 - pooled) * (1 + 1 / ratio))
    return obj.solve_power(effect_size=effect / sd, alpha=alpha, power=power, ratio=ratio)

def logrank_sample_size(hr: float, alpha: float, power: float, allocation: float = 1.0) -> float:
    """
    Freedman 公式:  n_each = (Zα/2 + Zβ)^2 / ( (log HR)^2 * allocation_factor )
    allocation = n_treat / n_control
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    k = allocation
    factor = (k + 1) ** 2 / (k)
    n_events = (z_alpha + z_beta) ** 2 / ((np.log(hr)) ** 2) * factor
    return n_events  # 需再除以预期事件率得到总例数

# ---------- UI ---------- #
def sample_size_ui() -> None:
    st.set_page_config("样本量计算", "📐", layout="wide")
    st.markdown("# 📐 样本量计算")

    design = st.radio("研究设计", ["均数差", "比例差", "生存 HR (log-rank)"])
    alpha = st.number_input("α（显著性水平）", 0.0, 0.2, 0.05, 0.01)
    power = st.number_input("1-β（检验效能）", 0.7, 0.99, 0.8, 0.01)
    ratio = st.number_input("组间样本量比 (治疗/对照)", 0.1, 5.0, 1.0, 0.1)

    if design == "均数差":
        delta = st.number_input("期望均数差 Δ", 0.0, 1e3, 5.0)
        sd = st.number_input("组内标准差 σ", 0.0001, 1e3.0, 10.0)
        if st.button("🧮 计算样本量"):
            n_control = two_mean_sample_size(delta, sd, alpha, power, ratio)
            n_treat = n_control * ratio
            st.success(f"对照组 ≈ {np.ceil(n_control):.0f} 例，试验组 ≈ {np.ceil(n_treat):.0f} 例，总计 ≈ {np.ceil(n_control + n_treat):.0f}")

    elif design == "比例差":
        p1 = st.number_input("对照组事件率 p₀", 0.0, 1.0, 0.4, 0.01)
        p2 = st.number_input("试验组事件率 p₁", 0.0, 1.0, 0.25, 0.01)
        if st.button("🧮 计算样本量"):
            n_control = two_prop_sample_size(p1, p2, alpha, power, ratio)
            n_treat = n_control * ratio
            st.success(f"对照组 ≈ {np.ceil(n_control):.0f} 例，试验组 ≈ {np.ceil(n_treat):.0f} 例，总计 ≈ {np.ceil(n_control + n_treat):.0f}")

    else:  # 生存
        hr = st.number_input("预期 HR", 0.1, 2.0, 0.7, 0.01)
        event_rate = st.number_input("整体事件率 (如 0.6)", 0.05, 1.0, 0.6, 0.01)
        if st.button("🧮 计算样本量"):
            n_events = logrank_sample_size(hr, alpha, power, ratio)
            total_n = n_events / event_rate
            n_control = total_n / (1 + ratio)
            n_treat = total_n - n_control
            st.success(f"需事件数 ≈ {np.ceil(n_events):.0f}\n对照组 ≈ {np.ceil(n_control):.0f}，试验组 ≈ {np.ceil(n_treat):.0f}，总计 ≈ {np.ceil(total_n):.0f}")

if __name__ == "__main__":
    sample_size_ui()

