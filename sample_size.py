import streamlit as st
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
import pandas as pd
import plotly.graph_objects as go

def sample_size_ui():
    st.header("📊 样本量计算")
    
    # 研究类型选择
    study_type = st.selectbox(
        "选择研究类型",
        ["两组均数比较", "两组率比较", "相关性研究", "生存分析"]
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("参数设置")
        
        if study_type == "两组均数比较":
            mean1 = st.number_input("组1均数", value=10.0)
            mean2 = st.number_input("组2均数", value=12.0)
            sd = st.number_input("标准差", value=3.0, min_value=0.1)
            test_type = st.selectbox("检验类型", ["双侧检验", "单侧检验"])
            
        elif study_type == "两组率比较":
            prop1 = st.number_input("组1率(%)", value=20.0, min_value=0.0, max_value=100.0)
            prop2 = st.number_input("组2率(%)", value=30.0, min_value=0.0, max_value=100.0)
            test_type = st.selectbox("检验类型", ["双侧检验", "单侧检验"])
            
        elif study_type == "相关性研究":
            correlation = st.number_input("预期相关系数", value=0.3, min_value=-1.0, max_value=1.0)
            
        elif study_type == "生存分析":
            median1 = st.number_input("组1中位生存时间", value=12.0)
            median2 = st.number_input("组2中位生存时间", value=18.0)
            accrual_time = st.number_input("入组时间", value=2.0)
            follow_time = st.number_input("随访时间", value=3.0)
    
    with col2:
        st.subheader("统计参数")
        
        alpha = st.number_input("α水平", value=0.05, min_value=0.01, max_value=0.1, step=0.01)
        power = st.number_input("检验效能(1-β)", value=0.8, min_value=0.5, max_value=0.99, step=0.01)
        ratio = st.number_input("组间比例", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
    
    if st.button("🔢 计算样本量", type="primary"):
        try:
            if study_type == "两组均数比较":
                result = calculate_sample_size_means(mean1, mean2, sd, alpha, power, ratio, test_type)
            elif study_type == "两组率比较":
                result = calculate_sample_size_proportions(prop1/100, prop2/100, alpha, power, ratio, test_type)
            elif study_type == "相关性研究":
                result = calculate_sample_size_correlation(correlation, alpha, power)
            elif study_type == "生存分析":
                result = calculate_sample_size_survival(median1, median2, alpha, power, ratio)
            
            # 显示结果
            st.subheader("📋 计算结果")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("组1样本量", result['n1'])
            with col2:
                st.metric("组2样本量", result['n2'])
            with col3:
                st.metric("总样本量", result['total'])
            
            # 详细结果
            st.subheader("📝 详细信息")
            for key, value in result['details'].items():
                st.write(f"**{key}:** {value}")
            
            # 功效曲线
            if study_type in ["两组均数比较", "两组率比较"]:
                plot_power_curve(result, study_type)
                
        except Exception as e:
            st.error(f"计算错误: {str(e)}")

def calculate_sample_size_means(mean1, mean2, sd, alpha, power, ratio, test_type):
    """计算两组均数比较的样本量"""
    
    # 效应量
    effect_size = abs(mean1 - mean2) / sd
    
    # 临界值
    if test_type == "双侧检验":
        z_alpha = norm.ppf(1 - alpha/2)
    else:
        z_alpha = norm.ppf(1 - alpha)
    
    z_beta = norm.ppf(power)
    
    # 样本量计算
    n1 = ((z_alpha + z_beta) ** 2 * (1 + 1/ratio) * sd**2) / (mean1 - mean2)**2
    n1 = int(np.ceil(n1))
    n2 = int(np.ceil(n1 / ratio))
    
    return {
        'n1': n1,
        'n2': n2,
        'total': n1 + n2,
        'details': {
            '效应量(Cohen\'s d)': f"{effect_size:.4f}",
            '检验类型': test_type,
            'α水平': alpha,
            '检验效能': power,
            '组间比例': f"{ratio}:1"
        }
    }

def calculate_sample_size_proportions(p1, p2, alpha, power, ratio, test_type):
    """计算两组率比较的样本量"""
    
    # 合并比例
    p_pooled = (p1 + ratio * p2) / (1 + ratio)
    
    # 临界值
    if test_type == "双侧检验":
        z_alpha = norm.ppf(1 - alpha/2)
    else:
        z_alpha = norm.ppf(1 - alpha)
    
    z_beta = norm.ppf(power)
    
    # 样本量计算
    numerator = (z_alpha * np.sqrt((1 + ratio) * p_pooled * (1 - p_pooled)) + 
                z_beta * np.sqrt(p1 * (1 - p1) + ratio * p2 * (1 - p2)))**2
    denominator = ratio * (p1 - p2)**2
    
    n1 = int(np.ceil(numerator / denominator))
    n2 = int(np.ceil(n1 / ratio))
    
    return {
        'n1': n1,
        'n2': n2,
        'total': n1 + n2,
        'details': {
            '组1率': f"{p1*100:.1f}%",
            '组2率': f"{p2*100:.1f}%",
            '率差': f"{abs(p1-p2)*100:.1f}%",
            '检验类型': test_type,
            'α水平': alpha,
            '检验效能': power,
            '组间比例': f"{ratio}:1"
        }
    }

def calculate_sample_size_correlation(r, alpha, power):
    """计算相关性研究的样本量"""
    
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    # Fisher's z变换
    z_r = 0.5 * np.log((1 + r) / (1 - r))
    
    # 样本量计算
    n = ((z_alpha + z_beta) / z_r)**2 + 3
    n = int(np.ceil(n))
    
    return {
        'n1': n,
        'n2': 0,
        'total': n,
        'details': {
            '预期相关系数': f"{r:.3f}",
            'Fisher\'s z': f"{z_r:.4f}",
            'α水平': alpha,
            '检验效能': power
        }
    }

def calculate_sample_size_survival(median1, median2, alpha, power, ratio):
    """计算生存分析的样本量（简化版本）"""
    
    # 转换为风险比
    hr = median1 / median2
    log_hr = np.log(hr)
    
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    # 所需事件数
    events = ((z_alpha + z_beta) / log_hr)**2
    
    # 假设事件率为50%
    event_rate = 0.5
    
    n1 = int(np.ceil(events / (event_rate * (1 + 1/ratio))))
    n2 = int(np.ceil(n1 / ratio))
    
    return {
        'n1': n1,
        'n2': n2,
        'total': n1 + n2,
        'details': {
            '组1中位生存时间': f"{median1}",
            '组2中位生存时间': f"{median2}",
            '风险比': f"{hr:.3f}",
            '所需事件数': f"{int(events)}",
            'α水平': alpha,
            '检验效能': power
        }
    }

def plot_power_curve(result, study_type):
    """绘制功效曲线"""
    
    st.subheader("📈 功效曲线")
    
    # 创建不同样本量下的功效值
    n_range = np.arange(max(10, result['total']//2), result['total']*2, 5)
    power_values = []
    
    for n in n_range:
        # 简化的功效计算
        if study_type == "两组均数比较":
            power = 1 - norm.cdf(1.96 - np.sqrt(n/4) * 0.5)  # 简化计算
                elif study_type == "两组率比较":
            power = 1 - norm.cdf(1.96 - np.sqrt(n/4) * 0.3)  # 简化计算
        
        power_values.append(min(power, 0.99))
    
    # 绘制功效曲线
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=n_range,
        y=power_values,
        mode='lines',
        name='功效曲线',
        line=dict(color='blue', width=2)
    ))
    
    # 添加目标点
    fig.add_trace(go.Scatter(
        x=[result['total']],
        y=[0.8],
        mode='markers',
        name='目标样本量',
        marker=dict(color='red', size=10)
    ))
    
    fig.add_hline(y=0.8, line_dash="dash", line_color="red", 
                  annotation_text="目标功效=0.8")
    
    fig.update_layout(
        title="样本量-功效关系图",
        xaxis_title="总样本量",
        yaxis_title="检验效能",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
