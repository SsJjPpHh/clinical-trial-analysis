"""
样本量计算模块 (sample_size.py)
提供各种统计分析的样本量计算功能
"""

import streamlit as st
import numpy as np
import pandas as pd
import scipy.stats as stats
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import fsolve
import math

def sample_size_calculator():
    """样本量计算主函数"""
    st.markdown("# 🧮 样本量计算器")
    st.markdown("*为各种统计分析设计提供科学的样本量计算*")
    
    # 侧边栏 - 计算类型选择
    with st.sidebar:
        st.markdown("### 📊 计算类型")
        calc_type = st.selectbox(
            "选择分析类型",
            [
                "🔢 均数比较",
                "📊 比例比较", 
                "📈 相关性分析",
                "🧪 方差分析(ANOVA)",
                "🔄 卡方检验",
                "⚖️ 生存分析",
                "📉 回归分析",
                "🎯 非劣效性检验",
                "⚡ 功效分析",
                "📋 多重比较"
            ]
        )
        
        st.markdown("### ⚙️ 通用参数")
        alpha = st.slider("显著性水平 (α)", 0.01, 0.10, 0.05, 0.01)
        power = st.slider("统计功效 (1-β)", 0.70, 0.99, 0.80, 0.01)
        
        st.markdown("### 📈 可视化选项")
        show_power_curve = st.checkbox("显示功效曲线", value=True)
        show_sample_curve = st.checkbox("显示样本量曲线", value=True)
    
    # 根据选择的类型调用相应函数
    if calc_type == "🔢 均数比较":
        mean_comparison_sample_size(alpha, power, show_power_curve, show_sample_curve)
    elif calc_type == "📊 比例比较":
        proportion_comparison_sample_size(alpha, power, show_power_curve, show_sample_curve)
    elif calc_type == "📈 相关性分析":
        correlation_sample_size(alpha, power, show_power_curve, show_sample_curve)
    elif calc_type == "🧪 方差分析(ANOVA)":
        anova_sample_size(alpha, power, show_power_curve, show_sample_curve)
    elif calc_type == "🔄 卡方检验":
        chi_square_sample_size(alpha, power, show_power_curve, show_sample_curve)
    elif calc_type == "⚖️ 生存分析":
        survival_sample_size(alpha, power, show_power_curve, show_sample_curve)
    elif calc_type == "📉 回归分析":
        regression_sample_size(alpha, power, show_power_curve, show_sample_curve)
    elif calc_type == "🎯 非劣效性检验":
        non_inferiority_sample_size(alpha, power, show_power_curve, show_sample_curve)
    elif calc_type == "⚡ 功效分析":
        power_analysis(alpha, show_power_curve, show_sample_curve)
    elif calc_type == "📋 多重比较":
        multiple_comparison_sample_size(alpha, power, show_power_curve, show_sample_curve)

def mean_comparison_sample_size(alpha, power, show_power_curve, show_sample_curve):
    """均数比较的样本量计算"""
    st.markdown("## 🔢 均数比较样本量计算")
    st.markdown("*适用于t检验、配对t检验等均数比较研究*")
    
    # 参数输入
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 研究设计参数")
        
        study_design = st.selectbox(
            "研究设计类型",
            ["两独立样本t检验", "配对t检验", "单样本t检验"]
        )
        
        test_type = st.selectbox(
            "检验类型",
            ["双侧检验", "单侧检验"]
        )
        
        if study_design == "两独立样本t检验":
            allocation_ratio = st.number_input(
                "分组比例 (试验组:对照组)",
                min_value=0.1, max_value=5.0, value=1.0, step=0.1
            )
        else:
            allocation_ratio = 1.0
    
    with col2:
        st.markdown("### 📊 效应量参数")
        
        effect_input_method = st.selectbox(
            "效应量输入方式",
            ["直接输入效应量", "输入均数和标准差"]
        )
        
        if effect_input_method == "直接输入效应量":
            effect_size = st.number_input(
                "效应量 (Cohen's d)",
                min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                help="小效应: 0.2, 中等效应: 0.5, 大效应: 0.8"
            )
        else:
            if study_design == "两独立样本t检验":
                mu1 = st.number_input("试验组均数", value=10.0)
                mu2 = st.number_input("对照组均数", value=8.0)
                sigma = st.number_input("总体标准差", min_value=0.1, value=2.0)
                effect_size = abs(mu1 - mu2) / sigma
            elif study_design == "配对t检验":
                mean_diff = st.number_input("配对差值的均数", value=2.0)
                sd_diff = st.number_input("配对差值的标准差", min_value=0.1, value=3.0)
                effect_size = abs(mean_diff) / sd_diff
            else:  # 单样本t检验
                sample_mean = st.number_input("样本均数", value=10.0)
                pop_mean = st.number_input("总体均数", value=8.0)
                sigma = st.number_input("总体标准差", min_value=0.1, value=2.0)
                effect_size = abs(sample_mean - pop_mean) / sigma
        
        st.info(f"💡 计算得到的效应量: {effect_size:.3f}")
    
    # 样本量计算
    try:
        if study_design == "两独立样本t检验":
            sample_size = calculate_two_sample_t_test_size(
                effect_size, alpha, power, allocation_ratio, test_type
            )
            
            n1 = int(np.ceil(sample_size))
            n2 = int(np.ceil(sample_size * allocation_ratio))
            total_n = n1 + n2
            
        elif study_design == "配对t检验":
            sample_size = calculate_paired_t_test_size(
                effect_size, alpha, power, test_type
            )
            
            n1 = int(np.ceil(sample_size))
            n2 = n1  # 配对设计
            total_n = n1
            
        else:  # 单样本t检验
            sample_size = calculate_one_sample_t_test_size(
                effect_size, alpha, power, test_type
            )
            
            n1 = int(np.ceil(sample_size))
            n2 = 0
            total_n = n1
        
        # 结果显示
        display_sample_size_results(
            study_design, n1, n2, total_n, effect_size, alpha, power, test_type
        )
        
        # 敏感性分析
        sensitivity_analysis_means(
            study_design, effect_size, alpha, power, test_type, allocation_ratio,
            show_power_curve, show_sample_curve
        )
        
        # 样本量表格
        generate_sample_size_table_means(
            study_design, effect_size, alpha, test_type, allocation_ratio
        )
    
    except Exception as e:
        st.error(f"❌ 样本量计算失败: {str(e)}")

def calculate_two_sample_t_test_size(effect_size, alpha, power, ratio, test_type):
    """计算两独立样本t检验的样本量"""
    
    # 获取临界值
    if test_type == "双侧检验":
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(power)
    
    # 样本量公式
    k = ratio  # 分组比例
    n1 = ((z_alpha + z_beta)**2 * (1 + 1/k)) / (effect_size**2)
    
    return n1

def calculate_paired_t_test_size(effect_size, alpha, power, test_type):
    """计算配对t检验的样本量"""
    
    if test_type == "双侧检验":
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(power)
    
    # 配对t检验样本量公式
    n = ((z_alpha + z_beta) / effect_size)**2
    
    return n

def calculate_one_sample_t_test_size(effect_size, alpha, power, test_type):
    """计算单样本t检验的样本量"""
    
    if test_type == "双侧检验":
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(power)
    
    # 单样本t检验样本量公式
    n = ((z_alpha + z_beta) / effect_size)**2
    
    return n

def display_sample_size_results(study_design, n1, n2, total_n, effect_size, alpha, power, test_type):
    """显示样本量计算结果"""
    st.markdown("### 🎯 样本量计算结果")
    
    # 创建结果展示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if study_design == "两独立样本t检验":
            st.metric("试验组样本量", n1)
        elif study_design == "配对t检验":
            st.metric("配对数", n1)
        else:
            st.metric("样本量", n1)
    
    with col2:
        if study_design == "两独立样本t检验":
            st.metric("对照组样本量", n2)
        elif study_design == "配对t检验":
            st.metric("总观测数", n1 * 2)
        else:
            st.metric("", "")
    
    with col3:
        st.metric("总样本量", total_n)
    
    # 详细参数表
    st.markdown("### 📋 计算参数摘要")
    
    params_df = pd.DataFrame({
        '参数': [
            '研究设计',
            '检验类型', 
            '效应量',
            '显著性水平(α)',
            '统计功效(1-β)',
            '预期检出率'
        ],
        '数值': [
            study_design,
            test_type,
            f"{effect_size:.3f}",
            f"{alpha:.3f}",
            f"{power:.3f}",
            f"{power*100:.1f}%"
        ]
    })
    
    st.dataframe(params_df, hide_index=True)
    
    # 解释说明
    st.markdown("### 📝 结果解释")
    
    if study_design == "两独立样本t检验":
        st.markdown(f"""
        **样本量计算结果解释:**
        - 试验组需要 **{n1}** 名受试者
        - 对照组需要 **{n2}** 名受试者  
        - 总计需要 **{total_n}** 名受试者
        - 在α={alpha}，功效={power}的条件下，能够检出效应量为{effect_size:.3f}的差异
        """)
    elif study_design == "配对t检验":
        st.markdown(f"""
        **样本量计算结果解释:**
        - 需要 **{n1}** 对配对观测
        - 总计需要 **{n1*2}** 次观测
        - 在α={alpha}，功效={power}的条件下，能够检出效应量为{effect_size:.3f}的配对差异
        """)
    else:
        st.markdown(f"""
        **样本量计算结果解释:**
        - 需要 **{n1}** 名受试者
        - 在α={alpha}，功效={power}的条件下，能够检出效应量为{effect_size:.3f}的差异
        """)

def sensitivity_analysis_means(study_design, effect_size, alpha, power, test_type, ratio, show_power_curve, show_sample_curve):
    """均数比较的敏感性分析"""
    st.markdown("### 📈 敏感性分析")
    
    tab1, tab2 = st.tabs(["功效曲线", "样本量曲线"])
    
    with tab1:
        if show_power_curve:
            st.markdown("#### 🔋 统计功效曲线")
            
            # 效应量范围
            effect_range = np.linspace(0.1, 1.5, 50)
            powers = []
            
            for es in effect_range:
                if study_design == "两独立样本t检验":
                    n = calculate_two_sample_t_test_size(es, alpha, 0.8, ratio, test_type)
                elif study_design == "配对t检验":
                    n = calculate_paired_t_test_size(es, alpha, 0.8, test_type)
                else:
                    n = calculate_one_sample_t_test_size(es, alpha, 0.8, test_type)
                
                # 计算实际功效
                actual_power = calculate_actual_power_t_test(n, es, alpha, test_type, study_design, ratio)
                powers.append(actual_power)
            
            fig_power = go.Figure()
            
            fig_power.add_trace(go.Scatter(
                x=effect_range,
                y=powers,
                mode='lines',
                name='统计功效',
                line=dict(color='blue', width=3)
            ))
            
            # 添加参考线
            fig_power.add_hline(y=0.8, line_dash="dash", line_color="red", 
                               annotation_text="目标功效 (80%)")
            fig_power.add_vline(x=effect_size, line_dash="dash", line_color="green",
                               annotation_text=f"当前效应量 ({effect_size:.3f})")
            
            fig_power.update_layout(
                title="统计功效随效应量变化曲线",
                xaxis_title="效应量 (Cohen's d)",
                yaxis_title="统计功效",
                yaxis=dict(range=[0, 1]),
                height=400
            )
            
            st.plotly_chart(fig_power, use_container_width=True)
    
    with tab2:
        if show_sample_curve:
            st.markdown("#### 📊 样本量需求曲线")
            
            # 功效范围
            power_range = np.linspace(0.5, 0.99, 50)
            sample_sizes = []
            
            for p in power_range:
                if study_design == "两独立样本t检验":
                    n = calculate_two_sample_t_test_size(effect_size, alpha, p, ratio, test_type)
                    total_n = n * (1 + ratio)
                elif study_design == "配对t检验":
                    n = calculate_paired_t_test_size(effect_size, alpha, p, test_type)
                    total_n = n
                else:
                    n = calculate_one_sample_t_test_size(effect_size, alpha, p, test_type)
                    total_n = n
                
                sample_sizes.append(total_n)
            
            fig_sample = go.Figure()
            
            fig_sample.add_trace(go.Scatter(
                x=power_range,
                y=sample_sizes,
                mode='lines',
                name='总样本量',
                line=dict(color='orange', width=3)
            ))
            
            # 添加参考线
            fig_sample.add_vline(x=power, line_dash="dash", line_color="red",
                                annotation_text=f"目标功效 ({power:.0%})")
            
            fig_sample.update_layout(
                title="样本量需求随统计功效变化曲线",
                xaxis_title="统计功效",
                yaxis_title="总样本量",
                height=400
            )
            
            st.plotly_chart(fig_sample, use_container_width=True)

def calculate_actual_power_t_test(n, effect_size, alpha, test_type, study_design, ratio=1.0):
    """计算t检验的实际统计功效"""
    try:
        if test_type == "双侧检验":
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        if study_design == "两独立样本t检验":
            se = np.sqrt((1 + 1/ratio) / n)
        else:
            se = 1 / np.sqrt(n)
        
        z_beta = effect_size / se - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return min(max(power, 0), 1)
    
    except:
        return 0.5

def generate_sample_size_table_means(study_design, effect_size, alpha, test_type, ratio):
    """生成样本量对照表"""
    st.markdown("### 📊 样本量对照表")
    
    # 创建不同参数组合的样本量表
    power_levels = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
    alpha_levels = [0.01, 0.05, 0.10]
    
    table_data = []
    
    for a in alpha_levels:
        for p in power_levels:
            if study_design == "两独立样本t检验":
                n = calculate_two_sample_t_test_size(effect_size, a, p, ratio, test_type)
                total_n = int(np.ceil(n * (1 + ratio)))
            elif study_design == "配对t检验":
                n = calculate_paired_t_test_size(effect_size, a, p, test_type)
                total_n = int(np.ceil(n))
            else:
                n = calculate_one_sample_t_test_size(effect_size, a, p, test_type)
                total_n = int(np.ceil(n))
            
            table_data.append({
                '显著性水平(α)': f"{a:.2f}",
                '统计功效(1-β)': f"{p:.2f}",
                '总样本量': total_n
            })
    
    table_df = pd.DataFrame(table_data)
    
    # 透视表格式
    pivot_table = table_df.pivot(index='显著性水平(α)', 
                                columns='统计功效(1-β)', 
                                values='总样本量')
    
    st.dataframe(pivot_table)
    
    st.markdown(f"""
    **表格说明:**
    - 基于效应量 = {effect_size:.3f}
    - 检验类型: {test_type}
    - 研究设计: {study_design}
    """)

def proportion_comparison_sample_size(alpha, power, show_power_curve, show_sample_curve):
    """比例比较的样本量计算"""
    st.markdown("## 📊 比例比较样本量计算")
    st.markdown("*适用于两组率的比较、卡方检验等*")
    
    # 参数输入
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 研究设计参数")
        
        study_design = st.selectbox(
            "研究设计类型",
            ["两独立样本比例比较", "配对比例比较", "单样本比例检验"]
        )
        
        test_type = st.selectbox(
            "检验类型",
            ["双侧检验", "单侧检验"]
        )
        
        if study_design == "两独立样本比例比较":
            allocation_ratio = st.number_input(
                "分组比例 (试验组:对照组)",
                min_value=0.1, max_value=5.0, value=1.0, step=0.1
            )
        else:
            allocation_ratio = 1.0
    
    with col2:
        st.markdown("### 📊 比例参数")
        
        if study_design == "两独立样本比例比较":
            p1 = st.number_input(
                "试验组预期比例 (p1)",
                min_value=0.01, max_value=0.99, value=0.6, step=0.01
            )
            p2 = st.number_input(
                "对照组预期比例 (p2)", 
                min_value=0.01, max_value=0.99, value=0.4, step=0.01
            )
            
            # 计算效应量
            effect_size = abs(p1 - p2)
            
        elif study_design == "配对比例比较":
            p_discordant = st.number_input(
                "不一致对的比例",
                min_value=0.01, max_value=0.50, value=0.2, step=0.01
            )
            p_diff = st.number_input(
                "配对差异比例",
                min_value=0.01, max_value=1.0, value=0.1, step=0.01
            )
            p1, p2 = p_discordant, p_diff
            effect_size = p_diff
            
        else:  # 单样本比例检验
            p_sample = st.number_input(
                "样本预期比例",
                min_value=0.01, max_value=0.99, value=0.6, step=0.01
            )
            p_null = st.number_input(
                "原假设比例",
                min_value=0.01, max_value=0.99, value=0.5, step=0.01
            )
            p1, p2 = p_sample, p_null
            effect_size = abs(p_sample - p_null)
        
        st.info(f"💡 效应量: {effect_size:.3f}")
    
    # 样本量计算
    try:
        if study_design == "两独立样本比例比较":
            sample_size = calculate_two_proportion_test_size(
                p1, p2, alpha, power, allocation_ratio, test_type
            )
            
            n1 = int(np.ceil(sample_size))
            n2 = int(np.ceil(sample_size * allocation_ratio))
            total_n = n1 + n2
            
        elif study_design == "配对比例比较":
            sample_size = calculate_paired_proportion_test_size(
                p_discordant, p_diff, alpha, power, test_type
            )
            
            n1 = int(np.ceil(sample_size))
            n2 = n1
            total_n = n1
            
        else:  # 单样本比例检验
            sample_size = calculate_one_proportion_test_size(
                p_sample, p_null, alpha, power, test_type
            )
            
            n1 = int(np.ceil(sample_size))
            n2 = 0
            total_n = n1
        
        # 结果显示
        display_proportion_results(
            study_design, n1, n2, total_n, p1, p2, alpha, power, test_type
        )
        
        # 敏感性分析
        sensitivity_analysis_proportions(
            study_design, p1, p2, alpha, power, test_type, allocation_ratio,
            show_power_curve, show_sample_curve
        )
    
    except Exception as e:
        st.error(f"❌ 样本量计算失败: {str(e)}")

def calculate_two_proportion_test_size(p1, p2, alpha, power, ratio, test_type):
    """计算两比例比较的样本量"""
    
    if test_type == "双侧检验":
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(power)
    
    # 合并比例
    p_pooled = (p1 + ratio * p2) / (1 + ratio)
    
    # 样本量公式
    numerator = (z_alpha * np.sqrt(p_pooled * (1 - p_pooled) * (1 + 1/ratio)) + 
                z_beta * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2) / ratio))**2
    
    denominator = (p1 - p2)**2
    
    n1 = numerator / denominator
    
    return n1

def calculate_paired_proportion_test_size(p_discordant, p_diff, alpha, power, test_type):
    """计算配对比例比较的样本量"""
    
    if test_type == "双侧检验":
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(power)
    
    # McNemar检验样本量公式
    n = ((z_alpha + z_beta)**2 * p_discordant) / (p_diff**2)
    
    return n

def calculate_one_proportion_test_size(p_sample, p_null, alpha, power, test_type):
    """计算单样本比例检验的样本量"""
    
    if test_type == "双侧检验":
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(power)
    
    # 单样本比例检验样本量公式
    numerator = (z_alpha * np.sqrt(p_null * (1 - p_null)) + 
                z_beta * np.sqrt(p_sample * (1 - p_sample)))**2
    
    denominator = (p_sample - p_null)**2
    
    n = numerator / denominator
    
    return n

def display_proportion_results(study_design, n1, n2, total_n, p1, p2, alpha, power, test_type):
    """显示比例比较样本量结果"""
    st.markdown("### 🎯 样本量计算结果")
    
    # 结果展示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if study_design == "两独立样本比例比较":
            st.metric("试验组样本量", n1)
        elif study_design == "配对比例比较":
            st.metric("配对数", n1)
        else:
            st.metric("样本量", n1)
    
    with col2:
        if study_design == "两独立样本比例比较":
            st.metric("对照组样本量", n2)
        elif study_design == "配对比例比较":
            st.metric("总观测数", n1 * 2)
        else:
            st.metric("", "")
    
    with col3:
        st.metric("总样本量", total_n)
    
        # 详细参数表
    st.markdown("### 📋 计算参数摘要")
    
    if study_design == "两独立样本比例比较":
        params_df = pd.DataFrame({
            '参数': [
                '研究设计',
                '检验类型',
                '试验组比例(p1)',
                '对照组比例(p2)',
                '效应量(|p1-p2|)',
                '显著性水平(α)',
                '统计功效(1-β)'
            ],
            '数值': [
                study_design,
                test_type,
                f"{p1:.3f}",
                f"{p2:.3f}",
                f"{abs(p1-p2):.3f}",
                f"{alpha:.3f}",
                f"{power:.3f}"
            ]
        })
    elif study_design == "配对比例比较":
        params_df = pd.DataFrame({
            '参数': [
                '研究设计',
                '检验类型',
                '不一致对比例',
                '配对差异比例',
                '显著性水平(α)',
                '统计功效(1-β)'
            ],
            '数值': [
                study_design,
                test_type,
                f"{p1:.3f}",
                f"{p2:.3f}",
                f"{alpha:.3f}",
                f"{power:.3f}"
            ]
        })
    else:
        params_df = pd.DataFrame({
            '参数': [
                '研究设计',
                '检验类型',
                '样本比例',
                '原假设比例',
                '效应量',
                '显著性水平(α)',
                '统计功效(1-β)'
            ],
            '数值': [
                study_design,
                test_type,
                f"{p1:.3f}",
                f"{p2:.3f}",
                f"{abs(p1-p2):.3f}",
                f"{alpha:.3f}",
                f"{power:.3f}"
            ]
        })
    
    st.dataframe(params_df, hide_index=True)

def sensitivity_analysis_proportions(study_design, p1, p2, alpha, power, test_type, ratio, show_power_curve, show_sample_curve):
    """比例比较的敏感性分析"""
    st.markdown("### 📈 敏感性分析")
    
    tab1, tab2 = st.tabs(["功效曲线", "样本量曲线"])
    
    with tab1:
        if show_power_curve:
            st.markdown("#### 🔋 统计功效曲线")
            
            if study_design == "两独立样本比例比较":
                # 固定p2，变化p1
                p1_range = np.linspace(max(0.01, p2-0.4), min(0.99, p2+0.4), 50)
                powers = []
                
                for p1_var in p1_range:
                    if abs(p1_var - p2) > 0.001:  # 避免除零
                        n = calculate_two_proportion_test_size(p1_var, p2, alpha, 0.8, ratio, test_type)
                        actual_power = calculate_actual_power_proportion(n, p1_var, p2, alpha, test_type, study_design, ratio)
                        powers.append(actual_power)
                    else:
                        powers.append(0.05)  # 接近原假设时功效接近α
                
                fig_power = go.Figure()
                
                fig_power.add_trace(go.Scatter(
                    x=p1_range,
                    y=powers,
                    mode='lines',
                    name='统计功效',
                    line=dict(color='blue', width=3)
                ))
                
                fig_power.add_hline(y=0.8, line_dash="dash", line_color="red", 
                                   annotation_text="目标功效 (80%)")
                fig_power.add_vline(x=p1, line_dash="dash", line_color="green",
                                   annotation_text=f"当前p1 ({p1:.3f})")
                
                fig_power.update_layout(
                    title="统计功效随试验组比例变化曲线",
                    xaxis_title="试验组比例 (p1)",
                    yaxis_title="统计功效",
                    yaxis=dict(range=[0, 1]),
                    height=400
                )
            
            st.plotly_chart(fig_power, use_container_width=True)
    
    with tab2:
        if show_sample_curve:
            st.markdown("#### 📊 样本量需求曲线")
            
            power_range = np.linspace(0.5, 0.99, 50)
            sample_sizes = []
            
            for p in power_range:
                if study_design == "两独立样本比例比较":
                    n = calculate_two_proportion_test_size(p1, p2, alpha, p, ratio, test_type)
                    total_n = n * (1 + ratio)
                elif study_design == "配对比例比较":
                    n = calculate_paired_proportion_test_size(p1, p2, alpha, p, test_type)
                    total_n = n
                else:
                    n = calculate_one_proportion_test_size(p1, p2, alpha, p, test_type)
                    total_n = n
                
                sample_sizes.append(total_n)
            
            fig_sample = go.Figure()
            
            fig_sample.add_trace(go.Scatter(
                x=power_range,
                y=sample_sizes,
                mode='lines',
                name='总样本量',
                line=dict(color='orange', width=3)
            ))
            
            fig_sample.add_vline(x=power, line_dash="dash", line_color="red",
                                annotation_text=f"目标功效 ({power:.0%})")
            
            fig_sample.update_layout(
                title="样本量需求随统计功效变化曲线",
                xaxis_title="统计功效",
                yaxis_title="总样本量",
                height=400
            )
            
            st.plotly_chart(fig_sample, use_container_width=True)

def calculate_actual_power_proportion(n, p1, p2, alpha, test_type, study_design, ratio=1.0):
    """计算比例检验的实际统计功效"""
    try:
        if test_type == "双侧检验":
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        if study_design == "两独立样本比例比较":
            p_pooled = (p1 + ratio * p2) / (1 + ratio)
            se_null = np.sqrt(p_pooled * (1 - p_pooled) * (1 + 1/ratio) / n)
            se_alt = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / (n * ratio))
        else:
            se_null = np.sqrt(p2 * (1 - p2) / n)
            se_alt = np.sqrt(p1 * (1 - p1) / n)
        
        z_beta = (abs(p1 - p2) / se_alt) - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return min(max(power, 0), 1)
    
    except:
        return 0.5

def correlation_sample_size(alpha, power, show_power_curve, show_sample_curve):
    """相关性分析样本量计算"""
    st.markdown("## 📈 相关性分析样本量计算")
    st.markdown("*适用于Pearson相关、Spearman相关等*")
    
    # 参数输入
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 研究设计参数")
        
        correlation_type = st.selectbox(
            "相关分析类型",
            ["Pearson相关", "Spearman相关", "偏相关"]
        )
        
        test_type = st.selectbox(
            "检验类型",
            ["双侧检验", "单侧检验"]
        )
        
        if correlation_type == "偏相关":
            control_vars = st.number_input(
                "控制变量个数",
                min_value=1, max_value=10, value=2, step=1
            )
        else:
            control_vars = 0
    
    with col2:
        st.markdown("### 📊 效应量参数")
        
        effect_input_method = st.selectbox(
            "效应量输入方式",
            ["直接输入相关系数", "根据效应大小选择"]
        )
        
        if effect_input_method == "直接输入相关系数":
            expected_r = st.number_input(
                "预期相关系数 (r)",
                min_value=0.01, max_value=0.99, value=0.3, step=0.01
            )
        else:
            effect_size_level = st.selectbox(
                "效应大小",
                ["小效应 (r=0.1)", "中等效应 (r=0.3)", "大效应 (r=0.5)"]
            )
            
            effect_mapping = {
                "小效应 (r=0.1)": 0.1,
                "中等效应 (r=0.3)": 0.3,
                "大效应 (r=0.5)": 0.5
            }
            expected_r = effect_mapping[effect_size_level]
        
        null_r = st.number_input(
            "原假设相关系数",
            min_value=0.0, max_value=0.99, value=0.0, step=0.01
        )
        
        st.info(f"💡 效应量 (r): {expected_r:.3f}")
    
    # 样本量计算
    try:
        sample_size = calculate_correlation_sample_size(
            expected_r, null_r, alpha, power, test_type, control_vars
        )
        
        n = int(np.ceil(sample_size))
        
        # 结果显示
        display_correlation_results(
            n, expected_r, null_r, alpha, power, test_type, correlation_type, control_vars
        )
        
        # 敏感性分析
        sensitivity_analysis_correlation(
            expected_r, null_r, alpha, power, test_type, control_vars,
            show_power_curve, show_sample_curve
        )
    
    except Exception as e:
        st.error(f"❌ 样本量计算失败: {str(e)}")

def calculate_correlation_sample_size(r1, r0, alpha, power, test_type, control_vars=0):
    """计算相关分析的样本量"""
    
    if test_type == "双侧检验":
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(power)
    
    # Fisher's z变换
    z1 = 0.5 * np.log((1 + r1) / (1 - r1))
    z0 = 0.5 * np.log((1 + r0) / (1 - r0))
    
    # 样本量公式（考虑控制变量）
    n = ((z_alpha + z_beta) / (z1 - z0))**2 + 3 + control_vars
    
    return n

def display_correlation_results(n, r1, r0, alpha, power, test_type, correlation_type, control_vars):
    """显示相关分析样本量结果"""
    st.markdown("### 🎯 样本量计算结果")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("所需样本量", n)
    
    with col2:
        st.metric("预期相关系数", f"{r1:.3f}")
    
    with col3:
        st.metric("效应量", f"{abs(r1-r0):.3f}")
    
    # 详细参数表
    st.markdown("### 📋 计算参数摘要")
    
    params_data = [
        ['相关分析类型', correlation_type],
        ['检验类型', test_type],
        ['预期相关系数(r1)', f"{r1:.3f}"],
        ['原假设相关系数(r0)', f"{r0:.3f}"],
        ['显著性水平(α)', f"{alpha:.3f}"],
        ['统计功效(1-β)', f"{power:.3f}"]
    ]
    
    if control_vars > 0:
        params_data.append(['控制变量个数', str(control_vars)])
    
    params_df = pd.DataFrame(params_data, columns=['参数', '数值'])
    st.dataframe(params_df, hide_index=True)
    
    # 结果解释
    st.markdown("### 📝 结果解释")
    st.markdown(f"""
    **样本量计算结果解释:**
    - 需要 **{n}** 名受试者进行相关性分析
    - 在α={alpha}，功效={power}的条件下，能够检出相关系数为{r1:.3f}的关联
    - 使用{correlation_type}进行分析
    """)

def sensitivity_analysis_correlation(r1, r0, alpha, power, test_type, control_vars, show_power_curve, show_sample_curve):
    """相关分析的敏感性分析"""
    st.markdown("### 📈 敏感性分析")
    
    tab1, tab2 = st.tabs(["功效曲线", "样本量曲线"])
    
    with tab1:
        if show_power_curve:
            st.markdown("#### 🔋 统计功效曲线")
            
            r_range = np.linspace(0.05, 0.8, 50)
            powers = []
            
            for r in r_range:
                n = calculate_correlation_sample_size(r, r0, alpha, 0.8, test_type, control_vars)
                actual_power = calculate_actual_power_correlation(n, r, r0, alpha, test_type, control_vars)
                powers.append(actual_power)
            
            fig_power = go.Figure()
            
            fig_power.add_trace(go.Scatter(
                x=r_range,
                y=powers,
                mode='lines',
                name='统计功效',
                line=dict(color='blue', width=3)
            ))
            
            fig_power.add_hline(y=0.8, line_dash="dash", line_color="red", 
                               annotation_text="目标功效 (80%)")
            fig_power.add_vline(x=r1, line_dash="dash", line_color="green",
                               annotation_text=f"当前相关系数 ({r1:.3f})")
            
            fig_power.update_layout(
                title="统计功效随相关系数变化曲线",
                xaxis_title="相关系数 (r)",
                yaxis_title="统计功效",
                yaxis=dict(range=[0, 1]),
                height=400
            )
            
            st.plotly_chart(fig_power, use_container_width=True)
    
    with tab2:
        if show_sample_curve:
            st.markdown("#### 📊 样本量需求曲线")
            
            power_range = np.linspace(0.5, 0.99, 50)
            sample_sizes = []
            
            for p in power_range:
                n = calculate_correlation_sample_size(r1, r0, alpha, p, test_type, control_vars)
                sample_sizes.append(n)
            
            fig_sample = go.Figure()
            
            fig_sample.add_trace(go.Scatter(
                x=power_range,
                y=sample_sizes,
                mode='lines',
                name='样本量',
                line=dict(color='orange', width=3)
            ))
            
            fig_sample.add_vline(x=power, line_dash="dash", line_color="red",
                                annotation_text=f"目标功效 ({power:.0%})")
            
            fig_sample.update_layout(
                title="样本量需求随统计功效变化曲线",
                xaxis_title="统计功效",
                yaxis_title="样本量",
                height=400
            )
            
            st.plotly_chart(fig_sample, use_container_width=True)

def calculate_actual_power_correlation(n, r1, r0, alpha, test_type, control_vars=0):
    """计算相关分析的实际统计功效"""
    try:
        if test_type == "双侧检验":
            z_alpha = stats.norm.ppf(1 - alpha/2)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
        
        # Fisher's z变换
        z1 = 0.5 * np.log((1 + r1) / (1 - r1))
        z0 = 0.5 * np.log((1 + r0) / (1 - r0))
        
        # 标准误
        se = 1 / np.sqrt(n - 3 - control_vars)
        
        z_beta = (z1 - z0) / se - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return min(max(power, 0), 1)
    
    except:
        return 0.5

def anova_sample_size(alpha, power, show_power_curve, show_sample_curve):
    """方差分析样本量计算"""
    st.markdown("## 🧪 方差分析(ANOVA)样本量计算")
    st.markdown("*适用于单因素方差分析、多因素方差分析*")
    
    # 参数输入
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 研究设计参数")
        
        anova_type = st.selectbox(
            "方差分析类型",
            ["单因素方差分析", "双因素方差分析", "重复测量方差分析"]
        )
        
        num_groups = st.number_input(
            "组别数量",
            min_value=2, max_value=10, value=3, step=1
        )
        
        if anova_type == "双因素方差分析":
            factor_a_levels = st.number_input(
                "因子A水平数",
                min_value=2, max_value=5, value=2, step=1
            )
            factor_b_levels = st.number_input(
                "因子B水平数", 
                min_value=2, max_value=5, value=2, step=1
            )
            num_groups = factor_a_levels * factor_b_levels
        
        elif anova_type == "重复测量方差分析":
            num_timepoints = st.number_input(
                "测量时间点数",
                min_value=2, max_value=8, value=3, step=1
            )
            correlation = st.slider(
                "重复测量相关系数",
                0.0, 0.9, 0.5, 0.1
            )
    
    with col2:
        st.markdown("### 📊 效应量参数")
        
        effect_input_method = st.selectbox(
            "效应量输入方式",
            ["直接输入效应量f", "输入均数和标准差", "根据效应大小选择"]
        )
        
        if effect_input_method == "直接输入效应量f":
            effect_size_f = st.number_input(
                "效应量 f",
                min_value=0.1, max_value=1.0, value=0.25, step=0.05,
                help="小效应: 0.1, 中等效应: 0.25, 大效应: 0.4"
            )
        
        elif effect_input_method == "根据效应大小选择":
            effect_level = st.selectbox(
                "效应大小",
                ["小效应 (f=0.1)", "中等效应 (f=0.25)", "大效应 (f=0.4)"]
            )
            
            effect_mapping = {
                "小效应 (f=0.1)": 0.1,
                "中等效应 (f=0.25)": 0.25,
                "大效应 (f=0.4)": 0.4
            }
            effect_size_f = effect_mapping[effect_level]
        
        else:  # 输入均数和标准差
            st.markdown("**各组均数:**")
            group_means = []
            for i in range(num_groups):
                mean = st.number_input(
                    f"第{i+1}组均数",
                    value=10.0 + i * 2.0,
                    key=f"mean_{i}"
                )
                group_means.append(mean)
            
            pooled_sd = st.number_input(
                "组内标准差",
                min_value=0.1, value=3.0, step=0.1
            )
            
            # 计算效应量f
            grand_mean = np.mean(group_means)
            sum_squares_between = sum([(m - grand_mean)**2 for m in group_means])
            effect_size_f = np.sqrt(sum_squares_between / (num_groups * pooled_sd**2))
        
        st.info(f"💡 效应量 f: {effect_size_f:.3f}")
    
    # 样本量计算
    try:
        if anova_type == "单因素方差分析":
            sample_size_per_group = calculate_one_way_anova_sample_size(
                effect_size_f, alpha, power, num_groups
            )
            total_n = sample_size_per_group * num_groups
            
        elif anova_type == "双因素方差分析":
            sample_size_per_cell = calculate_two_way_anova_sample_size(
                effect_size_f, alpha, power, factor_a_levels, factor_b_levels
            )
            total_n = sample_size_per_cell * num_groups
            
        else:  # 重复测量方差分析
            sample_size = calculate_repeated_measures_anova_sample_size(
                effect_size_f, alpha, power, num_timepoints, correlation
            )
            sample_size_per_group = sample_size
            total_n = sample_size
        
        # 结果显示
        display_anova_results(
            anova_type, sample_size_per_group, total_n, num_groups, 
            effect_size_f, alpha, power
        )
        
        # 敏感性分析
        sensitivity_analysis_anova(
            anova_type, effect_size_f, alpha, power, num_groups,
            show_power_curve, show_sample_curve
        )
    
    except Exception as e:
        st.error(f"❌ 样本量计算失败: {str(e)}")

def calculate_one_way_anova_sample_size(effect_size_f, alpha, power, num_groups):
    """计算单因素方差分析样本量"""
    
    # 自由度
    df_between = num_groups - 1
    
    # 非中心参数
    lambda_param = effect_size_f**2
    
    # 使用迭代方法求解样本量
    def power_function(n_per_group):
        df_within = num_groups * (n_per_group - 1)
        ncp = lambda_param * n_per_group * num_groups
        
        # F分布临界值
        f_crit = stats.f.ppf(1 - alpha, df_between, df_within)
        
        # 计算功效
        power_calc = 1 - stats.ncf.cdf(f_crit, df_between, df_within, ncp)
        
        return power_calc - power
    
    # 求解样本量
    try:
        n_per_group = fsolve(power_function, 10)[0]
        return max(2, int(np.ceil(n_per_group)))
    except:
        # 备用公式
        n_per_group = ((stats.norm.ppf(1-alpha) + stats.norm.ppf(power))**2) / (effect_size_f**2)
        return max(2, int(np.ceil(n_per_group)))

def calculate_two_way_anova_sample_size(effect_size_f, alpha, power, factor_a, factor_b):
    """计算双因素方差分析样本量"""
    
    # 简化计算，使用单因素公式的调整版本
    total_groups = factor_a * factor_b
    n_per_cell = calculate_one_way_anova_sample_size(effect_size_f, alpha, power, total_groups)
    
    return n_per_cell

def calculate_repeated_measures_anova_sample_size(effect_size_f, alpha, power, num_timepoints, correlation):
    """计算重复测量方差分析样本量"""
    
    # 考虑相关性的调整
    epsilon = 1 - correlation  # 球形度假设调整
    adjusted_effect = effect_size_f / np.sqrt(epsilon)
    
    # 使用调整后的效应量计算样本量
    n = calculate_one_way_anova_sample_size(adjusted_effect, alpha, power, num_timepoints)
    
    return n

def display_anova_results(anova_type, n_per_group, total_n, num_groups, effect_size_f, alpha, power):
    """显示方差分析样本量结果"""
    st.markdown("### 🎯 样本量计算结果")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if anova_type == "重复测量方差分析":
            st.metric("所需受试者数", n_per_group)
        else:
            st.metric("每组样本量", n_per_group)
    
    with col2:
        st.metric("组别数量", num_groups)
    
    with col3:
        st.metric("总样本量", total_n)
    
    # 详细参数表
    st.markdown("### 📋 计算参数摘要")
    
    params_df = pd.DataFrame({
        '参数': [
            '方差分析类型',
            '效应量(f)',
            '显著性水平(α)',
            '统计功效(1-β)',
            '组别数量'
        ],
        '数值': [
            anova_type,
            f"{effect_size_f:.3f}",
            f"{alpha:.3f}",
            f"{power:.3f}",
            str(num_groups)
        ]
    })
    
    st.dataframe(params_df, hide_index=True)

def sensitivity_analysis_anova(anova_type, effect_size_f, alpha, power, num_groups, show_power_curve, show_sample_curve):
    """方差分析的敏感性分析"""
    st.markdown("### 📈 敏感性分析")
    
    tab1, tab2 = st.tabs(["功效曲线", "样本量曲线"])
    
    with tab1:
        if show_power_curve:
            st.markdown("#### 🔋 统计功效曲线")
            
            effect_range = np.linspace(0.05, 0.8, 50)
            powers = []
            
            for f in effect_range:
                n = calculate_one_way_anova_sample_size(f, alpha, 0.8, num_groups)
                actual_power = calculate_actual_power_anova(n, f, alpha, num_groups)
                powers.append(actual_power)
            
            fig_power = go.Figure()
            
            fig_power.add_trace(go.Scatter(
                x=effect_range,
                y=powers,
                mode='lines',
                name='统计功效',
                line=dict(color='blue', width=3)
            ))
            
            fig_power.add_hline(y=0.8, line_dash="dash", line_color="red", 
                               annotation_text="目标功效 (80%)")
            fig_power.add_vline(x=effect_size_f, line_dash="dash", line_color="green",
                               annotation_text=f"当前效应量 ({effect_size_f:.3f})")
            
            fig_power.update_layout(
                title="统计功效随效应量变化曲线",
                xaxis_title="效应量 (f)",
                yaxis_title="统计功效",
                yaxis=dict(range=[0, 1]),
                height=400
            )
                        st.plotly_chart(fig_power, use_container_width=True)
    
    with tab2:
        if show_sample_curve:
            st.markdown("#### 📊 样本量需求曲线")
            
            power_range = np.linspace(0.5, 0.99, 50)
            sample_sizes = []
            
            for p in power_range:
                n = calculate_one_way_anova_sample_size(effect_size_f, alpha, p, num_groups)
                total_n = n * num_groups
                sample_sizes.append(total_n)
            
            fig_sample = go.Figure()
            
            fig_sample.add_trace(go.Scatter(
                x=power_range,
                y=sample_sizes,
                mode='lines',
                name='总样本量',
                line=dict(color='orange', width=3)
            ))
            
            fig_sample.add_vline(x=power, line_dash="dash", line_color="red",
                                annotation_text=f"目标功效 ({power:.0%})")
            
            fig_sample.update_layout(
                title="样本量需求随统计功效变化曲线",
                xaxis_title="统计功效",
                yaxis_title="总样本量",
                height=400
            )
            
            st.plotly_chart(fig_sample, use_container_width=True)

def calculate_actual_power_anova(n_per_group, effect_size_f, alpha, num_groups):
    """计算方差分析的实际统计功效"""
    try:
        df_between = num_groups - 1
        df_within = num_groups * (n_per_group - 1)
        ncp = effect_size_f**2 * n_per_group * num_groups
        
        f_crit = stats.f.ppf(1 - alpha, df_between, df_within)
        power = 1 - stats.ncf.cdf(f_crit, df_between, df_within, ncp)
        
        return min(max(power, 0), 1)
    except:
        return 0.5

def survival_sample_size(alpha, power, show_power_curve, show_sample_curve):
    """生存分析样本量计算"""
    st.markdown("## ⚖️ 生存分析样本量计算")
    st.markdown("*适用于Log-rank检验、Cox回归等生存分析*")
    
    # 参数输入
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 研究设计参数")
        
        survival_design = st.selectbox(
            "生存分析类型",
            ["Log-rank检验", "Cox回归分析", "指数生存模型"]
        )
        
        test_type = st.selectbox(
            "检验类型",
            ["双侧检验", "单侧检验"]
        )
        
        allocation_ratio = st.number_input(
            "分组比例 (试验组:对照组)",
            min_value=0.1, max_value=5.0, value=1.0, step=0.1
        )
        
        study_duration = st.number_input(
            "研究持续时间 (年)",
            min_value=0.5, max_value=10.0, value=3.0, step=0.5
        )
        
        accrual_period = st.number_input(
            "入组期 (年)",
            min_value=0.5, max_value=5.0, value=2.0, step=0.5
        )
    
    with col2:
        st.markdown("### 📊 生存参数")
        
        param_input_method = st.selectbox(
            "参数输入方式",
            ["输入风险比", "输入中位生存时间", "输入生存率"]
        )
        
        if param_input_method == "输入风险比":
            hazard_ratio = st.number_input(
                "风险比 (HR)",
                min_value=0.1, max_value=5.0, value=0.7, step=0.1,
                help="HR<1表示试验组优于对照组"
            )
            
            # 估算对照组事件率
            control_event_rate = st.slider(
                "对照组预期事件率",
                0.1, 0.9, 0.6, 0.05
            )
            
        elif param_input_method == "输入中位生存时间":
            median_control = st.number_input(
                "对照组中位生存时间 (年)",
                min_value=0.1, max_value=10.0, value=2.0, step=0.1
            )
            
            median_treatment = st.number_input(
                "试验组中位生存时间 (年)",
                min_value=0.1, max_value=10.0, value=3.0, step=0.1
            )
            
            # 计算风险比
            hazard_ratio = median_control / median_treatment
            
            # 估算事件率
            control_event_rate = 1 - np.exp(-np.log(2) * study_duration / median_control)
            
        else:  # 输入生存率
            survival_rate_control = st.number_input(
                "对照组生存率",
                min_value=0.01, max_value=0.99, value=0.4, step=0.01
            )
            
            survival_rate_treatment = st.number_input(
                "试验组生存率",
                min_value=0.01, max_value=0.99, value=0.6, step=0.01
            )
            
            # 计算风险比
            hazard_control = -np.log(survival_rate_control) / study_duration
            hazard_treatment = -np.log(survival_rate_treatment) / study_duration
            hazard_ratio = hazard_treatment / hazard_control
            
            control_event_rate = 1 - survival_rate_control
        
        st.info(f"💡 计算得到的风险比: {hazard_ratio:.3f}")
        st.info(f"💡 对照组事件率: {control_event_rate:.3f}")
    
    # 样本量计算
    try:
        if survival_design == "Log-rank检验":
            total_events, total_sample = calculate_logrank_sample_size(
                hazard_ratio, alpha, power, allocation_ratio, 
                control_event_rate, test_type
            )
            
        elif survival_design == "Cox回归分析":
            total_events, total_sample = calculate_cox_regression_sample_size(
                hazard_ratio, alpha, power, allocation_ratio,
                control_event_rate, test_type
            )
            
        else:  # 指数生存模型
            total_events, total_sample = calculate_exponential_survival_sample_size(
                hazard_ratio, alpha, power, allocation_ratio,
                control_event_rate, study_duration, accrual_period
            )
        
        # 结果显示
        display_survival_results(
            survival_design, total_sample, total_events, hazard_ratio,
            alpha, power, allocation_ratio, control_event_rate
        )
        
        # 敏感性分析
        sensitivity_analysis_survival(
            hazard_ratio, alpha, power, allocation_ratio, control_event_rate,
            show_power_curve, show_sample_curve
        )
    
    except Exception as e:
        st.error(f"❌ 样本量计算失败: {str(e)}")

def calculate_logrank_sample_size(hr, alpha, power, ratio, event_rate, test_type):
    """计算Log-rank检验样本量"""
    
    if test_type == "双侧检验":
        z_alpha = stats.norm.ppf(1 - alpha/2)
    else:
        z_alpha = stats.norm.ppf(1 - alpha)
    
    z_beta = stats.norm.ppf(power)
    
    # Log-rank检验所需事件数
    p1 = 1 / (1 + ratio)  # 试验组比例
    p2 = ratio / (1 + ratio)  # 对照组比例
    
    # 所需事件数
    events_needed = ((z_alpha + z_beta) / np.log(hr))**2 / (p1 * p2)
    
    # 总样本量
    total_sample = events_needed / event_rate
    
    return int(np.ceil(events_needed)), int(np.ceil(total_sample))

def calculate_cox_regression_sample_size(hr, alpha, power, ratio, event_rate, test_type):
    """计算Cox回归分析样本量"""
    
    # Cox回归与Log-rank检验类似，但需要考虑协变量
    events_needed, total_sample = calculate_logrank_sample_size(
        hr, alpha, power, ratio, event_rate, test_type
    )
    
    # Cox回归通常需要更多样本（考虑协变量调整）
    inflation_factor = 1.1  # 10%的样本量增加
    
    return int(np.ceil(events_needed * inflation_factor)), int(np.ceil(total_sample * inflation_factor))

def calculate_exponential_survival_sample_size(hr, alpha, power, ratio, event_rate, 
                                             study_duration, accrual_period):
    """计算指数生存模型样本量"""
    
    # 基础Log-rank样本量
    events_needed, base_sample = calculate_logrank_sample_size(
        hr, alpha, power, ratio, event_rate, "双侧检验"
    )
    
    # 考虑入组时间和随访时间的调整
    follow_up_time = study_duration - accrual_period
    
    if follow_up_time > 0:
        # 调整因子基于入组模式和随访时间
        adjustment_factor = 1 + (accrual_period / (2 * follow_up_time))
        total_sample = base_sample * adjustment_factor
    else:
        total_sample = base_sample * 1.5  # 保守估计
    
    return events_needed, int(np.ceil(total_sample))

def display_survival_results(survival_design, total_sample, total_events, hr, 
                           alpha, power, ratio, event_rate):
    """显示生存分析样本量结果"""
    st.markdown("### 🎯 样本量计算结果")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总样本量", total_sample)
    
    with col2:
        st.metric("所需事件数", total_events)
    
    with col3:
        treatment_n = int(total_sample / (1 + ratio))
        st.metric("试验组样本量", treatment_n)
    
    with col4:
        control_n = total_sample - treatment_n
        st.metric("对照组样本量", control_n)
    
    # 详细参数表
    st.markdown("### 📋 计算参数摘要")
    
    params_df = pd.DataFrame({
        '参数': [
            '生存分析类型',
            '风险比(HR)',
            '对照组事件率',
            '分组比例',
            '显著性水平(α)',
            '统计功效(1-β)'
        ],
        '数值': [
            survival_design,
            f"{hr:.3f}",
            f"{event_rate:.3f}",
            f"1:{ratio:.1f}",
            f"{alpha:.3f}",
            f"{power:.3f}"
        ]
    })
    
    st.dataframe(params_df, hide_index=True)
    
    # 结果解释
    st.markdown("### 📝 结果解释")
    st.markdown(f"""
    **生存分析样本量结果解释:**
    - 总计需要 **{total_sample}** 名受试者
    - 需要观察到 **{total_events}** 个事件
    - 试验组: **{int(total_sample / (1 + ratio))}** 人
    - 对照组: **{total_sample - int(total_sample / (1 + ratio))}** 人
    - 在α={alpha}，功效={power}的条件下，能够检出风险比为{hr:.3f}的差异
    """)

def sensitivity_analysis_survival(hr, alpha, power, ratio, event_rate, show_power_curve, show_sample_curve):
    """生存分析的敏感性分析"""
    st.markdown("### 📈 敏感性分析")
    
    tab1, tab2 = st.tabs(["功效曲线", "样本量曲线"])
    
    with tab1:
        if show_power_curve:
            st.markdown("#### 🔋 统计功效曲线")
            
            hr_range = np.linspace(0.3, 1.5, 50)
            powers = []
            
            for hr_val in hr_range:
                if abs(hr_val - 1.0) > 0.01:  # 避免HR=1的情况
                    events, total_n = calculate_logrank_sample_size(
                        hr_val, alpha, 0.8, ratio, event_rate, "双侧检验"
                    )
                    actual_power = calculate_actual_power_survival(
                        total_n, hr_val, alpha, ratio, event_rate
                    )
                    powers.append(actual_power)
                else:
                    powers.append(0.05)  # HR=1时功效接近α
            
            fig_power = go.Figure()
            
            fig_power.add_trace(go.Scatter(
                x=hr_range,
                y=powers,
                mode='lines',
                name='统计功效',
                line=dict(color='blue', width=3)
            ))
            
            fig_power.add_hline(y=0.8, line_dash="dash", line_color="red", 
                               annotation_text="目标功效 (80%)")
            fig_power.add_vline(x=hr, line_dash="dash", line_color="green",
                               annotation_text=f"当前HR ({hr:.3f})")
            fig_power.add_vline(x=1.0, line_dash="dot", line_color="gray",
                               annotation_text="无效应 (HR=1)")
            
            fig_power.update_layout(
                title="统计功效随风险比变化曲线",
                xaxis_title="风险比 (HR)",
                yaxis_title="统计功效",
                yaxis=dict(range=[0, 1]),
                height=400
            )
            
            st.plotly_chart(fig_power, use_container_width=True)
    
    with tab2:
        if show_sample_curve:
            st.markdown("#### 📊 样本量需求曲线")
            
            power_range = np.linspace(0.5, 0.99, 50)
            sample_sizes = []
            
            for p in power_range:
                events, total_n = calculate_logrank_sample_size(
                    hr, alpha, p, ratio, event_rate, "双侧检验"
                )
                sample_sizes.append(total_n)
            
            fig_sample = go.Figure()
            
            fig_sample.add_trace(go.Scatter(
                x=power_range,
                y=sample_sizes,
                mode='lines',
                name='总样本量',
                line=dict(color='orange', width=3)
            ))
            
            fig_sample.add_vline(x=power, line_dash="dash", line_color="red",
                                annotation_text=f"目标功效 ({power:.0%})")
            
            fig_sample.update_layout(
                title="样本量需求随统计功效变化曲线",
                xaxis_title="统计功效",
                yaxis_title="总样本量",
                height=400
            )
            
            st.plotly_chart(fig_sample, use_container_width=True)

def calculate_actual_power_survival(n, hr, alpha, ratio, event_rate):
    """计算生存分析的实际统计功效"""
    try:
        z_alpha = stats.norm.ppf(1 - alpha/2)
        
        p1 = 1 / (1 + ratio)
        p2 = ratio / (1 + ratio)
        
        events = n * event_rate
        z_beta = np.log(hr) * np.sqrt(events * p1 * p2) - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return min(max(power, 0), 1)
    except:
        return 0.5

def power_analysis(alpha, show_power_curve, show_sample_curve):
    """功效分析"""
    st.markdown("## ⚡ 功效分析")
    st.markdown("*已知样本量，计算统计功效*")
    
    # 参数输入
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📋 研究设计参数")
        
        analysis_type = st.selectbox(
            "分析类型",
            ["t检验功效分析", "比例检验功效分析", "相关分析功效分析", "方差分析功效分析"]
        )
        
        sample_size = st.number_input(
            "已知样本量",
            min_value=5, max_value=10000, value=100, step=5
        )
        
        if analysis_type in ["t检验功效分析", "比例检验功效分析"]:
            allocation_ratio = st.number_input(
                "分组比例 (如适用)",
                min_value=0.1, max_value=5.0, value=1.0, step=0.1
            )
    
    with col2:
        st.markdown("### 📊 效应量参数")
        
        if analysis_type == "t检验功效分析":
            effect_size = st.number_input(
                "效应量 (Cohen's d)",
                min_value=0.1, max_value=2.0, value=0.5, step=0.1
            )
            
        elif analysis_type == "比例检验功效分析":
            p1 = st.number_input("组1比例", min_value=0.01, max_value=0.99, value=0.6, step=0.01)
            p2 = st.number_input("组2比例", min_value=0.01, max_value=0.99, value=0.4, step=0.01)
            effect_size = abs(p1 - p2)
            
        elif analysis_type == "相关分析功效分析":
            effect_size = st.number_input(
                "相关系数 (r)",
                min_value=0.01, max_value=0.99, value=0.3, step=0.01
            )
            
        else:  # 方差分析功效分析
            effect_size = st.number_input(
                "效应量 (f)",
                min_value=0.1, max_value=1.0, value=0.25, step=0.05
            )
            num_groups = st.number_input(
                "组别数量",
                min_value=2, max_value=10, value=3, step=1
            )
    
    # 功效计算
    try:
        if analysis_type == "t检验功效分析":
            calculated_power = calculate_power_t_test(sample_size, effect_size, alpha, allocation_ratio)
            
        elif analysis_type == "比例检验功效分析":
            calculated_power = calculate_power_proportion_test(sample_size, p1, p2, alpha, allocation_ratio)
            
        elif analysis_type == "相关分析功效分析":
            calculated_power = calculate_power_correlation(sample_size, effect_size, alpha)
            
        else:  # 方差分析功效分析
            calculated_power = calculate_power_anova(sample_size, effect_size, alpha, num_groups)
        
        # 结果显示
        display_power_analysis_results(
            analysis_type, sample_size, effect_size, alpha, calculated_power
        )
        
        # 功效曲线
        if show_power_curve:
            display_power_curves(analysis_type, sample_size, effect_size, alpha)
    
    except Exception as e:
        st.error(f"❌ 功效计算失败: {str(e)}")

def calculate_power_t_test(n, effect_size, alpha, ratio=1.0):
    """计算t检验的统计功效"""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    se = np.sqrt((1 + 1/ratio) / n)
    z_beta = effect_size / se - z_alpha
    power = stats.norm.cdf(z_beta)
    return min(max(power, 0), 1)

def calculate_power_proportion_test(n, p1, p2, alpha, ratio=1.0):
    """计算比例检验的统计功效"""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    p_pooled = (p1 + ratio * p2) / (1 + ratio)
    
    se_null = np.sqrt(p_pooled * (1 - p_pooled) * (1 + 1/ratio) / n)
    se_alt = np.sqrt(p1 * (1 - p1) / n + p2 * (1 - p2) / (n * ratio))
    
    z_beta = abs(p1 - p2) / se_alt - z_alpha
    power = stats.norm.cdf(z_beta)
    return min(max(power, 0), 1)

def calculate_power_correlation(n, r, alpha):
    """计算相关分析的统计功效"""
    z_alpha = stats.norm.ppf(1 - alpha/2)
    z_r = 0.5 * np.log((1 + r) / (1 - r))
    se = 1 / np.sqrt(n - 3)
    
    z_beta = z_r / se - z_alpha
    power = stats.norm.cdf(z_beta)
    return min(max(power, 0), 1)

def calculate_power_anova(n_per_group, effect_size_f, alpha, num_groups):
    """计算方差分析的统计功效"""
    df_between = num_groups - 1
    df_within = num_groups * (n_per_group - 1)
    ncp = effect_size_f**2 * n_per_group * num_groups
    
    f_crit = stats.f.ppf(1 - alpha, df_between, df_within)
    power = 1 - stats.ncf.cdf(f_crit, df_between, df_within, ncp)
    return min(max(power, 0), 1)

def display_power_analysis_results(analysis_type, sample_size, effect_size, alpha, power):
    """显示功效分析结果"""
    st.markdown("### ⚡ 功效分析结果")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("样本量", sample_size)
    
    with col2:
        st.metric("效应量", f"{effect_size:.3f}")
    
    with col3:
        st.metric("显著性水平", f"{alpha:.3f}")
    
    with col4:
        st.metric("统计功效", f"{power:.3f}", f"{power*100:.1f}%")
    
    # 功效解释
    st.markdown("### 📝 功效解释")
    
    if power >= 0.8:
        st.success(f"✅ 统计功效充足 ({power:.1%})，能够有效检测到预期效应")
    elif power >= 0.6:
        st.warning(f"⚠️ 统计功效中等 ({power:.1%})，可能无法充分检测到预期效应")
    else:
        st.error(f"❌ 统计功效不足 ({power:.1%})，建议增加样本量")
    
    # 建议
    st.markdown("### 💡 建议")
    
    if power < 0.8:
        # 计算达到80%功效所需的样本量
        if analysis_type == "t检验功效分析":
            recommended_n = calculate_two_sample_t_test_size(effect_size, alpha, 0.8, 1.0, "双侧检验")
        elif analysis_type == "相关分析功效分析":
            recommended_n = calculate_correlation_sample_size(effect_size, 0, alpha, 0.8, "双侧检验")
        else:
            recommended_n = sample_size * 1.5  # 简单估计
        
        st.info(f"💡 建议样本量: {int(np.ceil(recommended_n))} (达到80%功效)")

def display_power_curves(analysis_type, sample_size, effect_size, alpha):
    """显示功效曲线"""
    st.markdown("### 📈 功效曲线分析")
    
    if analysis_type == "t检验功效分析":
        effect_range = np.linspace(0.1, 1.5, 50)
        powers = [calculate_power_t_test(sample_size, es, alpha) for es in effect_range]
        x_label = "效应量 (Cohen's d)"
        
    elif analysis_type == "相关分析功效分析":
        effect_range = np.linspace(0.05, 0.8, 50)
        powers = [calculate_power_correlation(sample_size, r, alpha) for r in effect_range]
        x_label = "相关系数 (r)"
        
    else:
        return  # 其他类型暂不显示
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=effect_range,
        y=powers,
        mode='lines',
        name='统计功效',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                  annotation_text="目标功效 (80%)")
    fig.add_vline(x=effect_size, line_dash="dash", line_color="green",
                  annotation_text=f"当前效应量 ({effect_size:.3f})")
    
    fig.update_layout(
        title=f"统计功效曲线 (n={sample_size})",
        xaxis_title=x_label,
        yaxis_title="统计功效",
        yaxis=dict(range=[0, 1]),
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# 主函数调用
if __name__ == "__main__":
    sample_size_calculator()

