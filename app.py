import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import sys
import os

# 添加本地模块路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入自定义模块 - 修复导入方式
try:
    from clinical_trial import clinical_trial_analysis
    # 为其他模块创建占位符函数，避免导入错误
    def statistical_analysis_placeholder():
        st.info("统计分析模块正在开发中...")
    
    def data_management_placeholder():
        st.info("数据管理模块正在开发中...")
        
    def epidemiology_placeholder():
        st.info("流行病学分析模块正在开发中...")
        
    def survival_analysis_placeholder():
        st.info("生存分析模块正在开发中...")
        
    def sample_size_placeholder():
        st.info("样本量计算模块正在开发中...")
        
except ImportError as e:
    st.error(f"模块导入错误: {e}")
    st.info("请确保所有必需的模块文件都在正确位置")

# 页面配置
st.set_page_config(
    page_title="临床试验统计分析系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: bold;
}
.sub-header {
    font-size: 1.5rem;
    color: #2c3e50;
    margin: 1rem 0;
}
.info-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
    margin: 1rem 0;
}
.success-box {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
    margin: 1rem 0;
}
.warning-box {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffc107;
    margin: 1rem 0;
}
.sidebar .sidebar-content {
    background-color: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

def main():
    """主函数"""
    
    # 主标题
    st.markdown('<h1 class="main-header">🏥 临床试验统计分析系统</h1>', unsafe_allow_html=True)
    
    # 侧边栏导航
    with st.sidebar:
        st.markdown("### 📊 功能导航")
        
        page = st.selectbox(
            "选择功能模块",
            [
                "🏠 首页概览",
                "🧪 临床试验分析",  # 修改为与实际函数对应
                "📊 统计分析",
                "💾 数据管理",
                "📈 流行病学分析", 
                "⏱️ 生存分析",
                "🎯 样本量计算",
                "📋 报告生成",
                "⚙️ 系统设置"
            ]
        )
        
        # 侧边栏信息
        st.markdown("---")
        st.markdown("### ℹ️ 系统信息")
        st.info(f"当前时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 快速统计
        if 'data' in st.session_state and st.session_state.data is not None:
            st.success(f"已加载数据: {st.session_state.data.shape[0]} 行 × {st.session_state.data.shape[1]} 列")
        else:
            st.warning("未加载数据")
    
    # 初始化会话状态
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = {}
    
    # 页面路由 - 修复调用方式
    if page == "🏠 首页概览":
        show_home_page()
    elif page == "🧪 临床试验分析":
        clinical_trial_analysis()  # 直接调用函数
    elif page == "📊 统计分析":
        statistical_analysis_placeholder()
    elif page == "💾 数据管理":
        data_management_placeholder()
    elif page == "📈 流行病学分析":
        epidemiology_placeholder()
    elif page == "⏱️ 生存分析":
        survival_analysis_placeholder()
    elif page == "🎯 样本量计算":
        sample_size_placeholder()
    elif page == "📋 报告生成":
        show_report_page()
    elif page == "⚙️ 系统设置":
        show_settings_page()

def show_home_page():
    """首页概览"""
    st.markdown('<h2 class="sub-header">🏠 系统概览</h2>', unsafe_allow_html=True)
    
    # 欢迎信息
    st.markdown("""
    <div class="info-box">
    <h3>欢迎使用临床试验统计分析系统！</h3>
    <p>本系统提供全面的临床试验设计、数据管理和统计分析功能，帮助研究人员高效完成临床研究工作。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 功能模块展示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🧪 试验分析")
        st.markdown("""
        - 基线特征分析
        - 主要终点分析
        - 次要终点分析
        - 安全性分析
        """)
        
    with col2:
        st.markdown("### 📊 统计分析")
        st.markdown("""
        - 描述性统计
        - 假设检验
        - 回归分析
        - 多元分析
        """)
        
    with col3:
        st.markdown("### 📈 专业分析")
        st.markdown("""
        - 生存分析
        - 流行病学分析
        - 样本量计算
        - 效应量评估
        """)
    
    # 快速开始指南
    st.markdown("---")
    st.markdown("### 🚀 快速开始")
    
    with st.expander("📖 使用指南", expanded=True):
        st.markdown("""
        1. **数据准备**: 在"数据管理"模块上传或输入您的数据
        2. **探索性分析**: 使用"统计分析"模块进行初步数据探索
        3. **专业分析**: 根据研究需要选择相应的专业分析模块
        4. **结果导出**: 在"报告生成"模块导出分析结果
        """)
    
    # 示例数据
    if st.button("🎯 加载示例数据"):
        sample_data = generate_sample_data()
        st.session_state.data = sample_data
        st.success("示例数据已加载！")
        st.dataframe(sample_data.head())

def generate_sample_data():
    """生成示例数据"""
    np.random.seed(42)
    n = 200
    
    data = {
        '受试者ID': [f'S{i:03d}' for i in range(1, n+1)],
        '治疗组': np.random.choice(['试验组', '对照组'], n),
        '年龄': np.random.normal(55, 12, n).astype(int),
        '性别': np.random.choice(['男', '女'], n),
        '体重': np.random.normal(70, 15, n),
        '基线血压_收缩压': np.random.normal(140, 20, n),
        '基线血压_舒张压': np.random.normal(90, 10, n),
        '主要终点_有效率': np.random.choice([0, 1], n, p=[0.3, 0.7]),
        '次要终点_改善程度': np.random.normal(5, 2, n),
        '不良事件': np.random.choice([0, 1], n, p=[0.8, 0.2])
    }
    
    return pd.DataFrame(data)

def show_report_page():
    """报告生成页面"""
    st.markdown("### 📋 报告生成")
    st.info("报告生成功能正在开发中...")

def show_settings_page():
    """系统设置页面"""
    st.markdown("### ⚙️ 系统设置")
    st.info("系统设置功能正在开发中...")

if __name__ == "__main__":
    main()
