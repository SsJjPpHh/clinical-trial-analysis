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
import plotly.express as px
import plotly.graph_objects as go

# ---------- 公式 ---------- #
def two_mean_sample_size(delta: float, sd: float, alpha: float, power: float, ratio: float) -> float:
    """两样本均数差的样本量计算"""
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    n1 = ((z_alpha + z_beta) * sd / delta) ** 2 * (1 + 1/ratio)
    return n1

def two_prop_sample_size(p1: float, p2: float, alpha: float, power: float, ratio: float) -> float:
    """两比例差的样本量计算"""
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    p_pooled = (p1 + ratio * p2) / (1 + ratio)
    
    n1 = (z_alpha * np.sqrt(p_pooled * (1 - p_pooled) * (1 + 1/ratio)) +
          z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio)) ** 2 / (p1 - p2) ** 2
    
    return n1

def survival_sample_size(hr: float, alpha: float, power: float, ratio: float,
                        p1_event: float = 0.5) -> tuple:
    """生存分析样本量计算（基于事件数）"""
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    # 所需事件数
    events = ((z_alpha + z_beta) / np.log(hr)) ** 2 * (1 + ratio) ** 2 / (ratio)
    
    # 总样本量（假设事件发生率）
    n_total = events / p1_event
    n1 = n_total / (1 + ratio)
    n2 = n_total * ratio / (1 + ratio)
    
    return n1, n2, events

# ---------- 主UI函数 ---------- #
def sample_size_ui():
    """样本量计算主界面"""
    st.title("🔢 样本量计算")
    st.markdown("临床试验样本量计算工具")
    
    # 选择计算类型
    calc_type = st.selectbox(
        "选择计算类型",
        options=["两样本均数比较", "两样本比例比较", "生存分析比较"]
    )
    
    # 通用参数
    st.header("📋 基本参数")
    
    col1, col2 = st.columns(2)
    
    with col1:
        alpha = st.selectbox("显著性水平 (α)", options=[0.05, 0.01, 0.001], value=0.05)
        power = st.selectbox("检验效能 (1-β)", options=[0.80, 0.85, 0.90, 0.95], value=0.80)
    
    with col2:
        ratio = st.number_input("样本量比例 (n2/n1)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        two_sided = st.checkbox("双侧检验", value=True)
    
    # 根据类型显示特定参数
    if calc_type == "两样本均数比较":
        st.header("📊 均数比较参数")
        
        col1, col2 = st.columns(2)
        
        with col1:
            delta = st.number_input("期望差值 (μ1 - μ2)", value=5.0, help="两组均数的期望差值")
            sd = st.number_input("标准差 (σ)", min_value=0.1, value=10.0, help="假设两组标准差相等")
        
        with col2:
            # 效应量
            effect_size = delta / sd if sd > 0 else 0
            st.metric("Cohen's d (效应量)", f"{effect_size:.3f}")
            
            if effect_size < 0.2:
                st.warning("效应量很小 (< 0.2)")
            elif effect_size < 0.5:
                st.info("效应量较小 (0.2-0.5)")
            elif effect_size < 0.8:
                st.success("效应量中等 (0.5-0.8)")
            else:
                st.success("效应量较大 (> 0.8)")
        
        # 计算样本量
        if st.button("计算样本量", type="primary"):
            try:
                n1 = two_mean_sample_size(abs(delta), sd, alpha, power, ratio)
                n2 = n1 * ratio
                
                st.success("✅ 计算完成")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("组1样本量", f"{int(np.ceil(n1))}")
                with col2:
                    st.metric("组2样本量", f"{int(np.ceil(n2))}")
                with col3:
                    st.metric("总样本量", f"{int(np.ceil(n1 + n2))}")
                
                # 绘制功效曲线
                st.subheader("📈 功效曲线")
                
                effect_sizes = np.linspace(0.1, 1.5, 50)
                powers = []
                
                for es in effect_sizes:
                    try:
                        z = es * np.sqrt(n1 * ratio / (1 + ratio) / 2)
                        power_calc = 1 - norm.cdf(norm.ppf(1 - alpha/2) - z) + norm.cdf(-norm.ppf(1 - alpha/2) - z)
                        powers.append(power_calc)
                    except:
                        powers.append(0)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=effect_sizes, y=powers, mode='lines', name='功效曲线'))
                fig.add_hline(y=power, line_dash="dash", line_color="red", annotation_text=f"目标功效 = {power}")
                fig.add_vline(x=effect_size, line_dash="dash", line_color="green", annotation_text=f"当前效应量 = {effect_size:.3f}")
                
                fig.update_layout(
                    title="统计功效 vs 效应量",
                    xaxis_title="效应量 (Cohen's d)",
                    yaxis_title="统计功效",
                    yaxis=dict(range=[0, 1])
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ 计算失败：{str(e)}")
    
    elif calc_type == "两样本比例比较":
        st.header("📊 比例比较参数")
        
        col1, col2 = st.columns(2)
        
        with col1:
            p1 = st.number_input("组1比例 (p1)", min_value=0.01, max_value=0.99, value=0.30, step=0.01)
            p2 = st.number_input("组2比例 (p2)", min_value=0.01, max_value=0.99, value=0.20, step=0.01)
