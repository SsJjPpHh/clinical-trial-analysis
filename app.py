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

# 导入自定义模块
try:
    from clinical_trial import ClinicalTrialModule
    from statistical_analysis import StatisticalAnalysisModule
    from data_management import DataManagementModule
    from epidemiology import EpidemiologyModule
    from survival_analysis import SurvivalAnalysisModule
    from sample_size import SampleSizeModule
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
                "🧪 临床试验设计",
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
    
    # 页面路由
    if page == "🏠 首页概览":
        show_home_page()
    elif page == "🧪 临床试验设计":
        show_clinical_trial_page()
    elif page == "📊 统计分析":
        show_statistical_analysis_page()
    elif page == "💾 数据管理":
        show_data_management_page()
    elif page == "📈 流行病学分析":
        show_epidemiology_page()
    elif page == "⏱️ 生存分析":
        show_survival_analysis_page()
    elif page == "🎯 样本量计算":
        show_sample_size_page()
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
        st.markdown("### 🧪 试验设计")
        st.markdown("""
        - 试验方案制定
        - 随机化设计
        - 盲法设置
        - 质量控制
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

def show_clinical_trial_page():
    """临床试验设计页面"""
    st.markdown('<h2 class="sub-header">🧪 临床试验设计</h2>', unsafe_allow_html=True)
    
    try:
        clinical_module = ClinicalTrialModule()
        clinical_module.render()
    except NameError:
        st.error("临床试验模块未正确导入")
        show_placeholder_clinical_trial()

def show_statistical_analysis_page():
    """统计分析页面"""
    st.markdown('<h2 class="sub-header">📊 统计分析</h2>', unsafe_allow_html=True)
    
    try:
        stats_module = StatisticalAnalysisModule()
        stats_module.render()
    except NameError:
        st.error("统计分析模块未正确导入")
        show_placeholder_statistical_analysis()

def show_data_management_page():
    """数据管理页面"""
    st.markdown('<h2 class="sub-header">💾 数据管理</h2>', unsafe_allow_html=True)
    
    try:
        data_module = DataManagementModule()
        data_module.render()
    except NameError:
        st.error("数据管理模块未正确导入")
        show_placeholder_data_management()

def show_epidemiology_page():
    """流行病学分析页面"""
    st.markdown('<h2 class="sub-header">📈 流行病学分析</h2>', unsafe_allow_html=True)
    
    try:
        epi_module = EpidemiologyModule()
        epi_module.render()
    except NameError:
        st.error("流行病学模块未正确导入")
        show_placeholder_epidemiology()

def show_survival_analysis_page():
    """生存分析页面"""
    st.markdown('<h2 class="sub-header">⏱️ 生存分析</h2>', unsafe_allow_html=True)
    
    try:
        survival_module = SurvivalAnalysisModule()
        survival_module.render()
    except NameError:
        st.error("生存分析模块未正确导入")
        show_placeholder_survival_analysis()

def show_sample_size_page():
    """样本量计算页面"""
    st.markdown('<h2 class="sub-header">🎯 样本量计算</h2>', unsafe_allow_html=True)
    
    try:
        sample_module = SampleSizeModule()
        sample_module.render()
    except NameError:
        st.error("样本量计算模块未正确导入")
        show_placeholder_sample_size()

def show_report_page():
    """报告生成页面"""
    st.markdown('<h2 class="sub-header">📋 报告生成</h2>', unsafe_allow_html=True)
    
    st.info("报告生成功能正在开发中...")
    
    if st.session_state.data is not None:
        st.markdown("### 📊 数据概览报告")
        
        # 基本信息
        st.markdown(f"**数据维度**: {st.session_state.data.shape[0]} 行 × {st.session_state.data.shape[1]} 列")
        
        # 数据类型统计
        st.markdown("**数据类型分布**:")
        dtype_counts = st.session_state.data.dtypes.value_counts()
        st.bar_chart(dtype_counts)
        
        # 缺失值统计
        missing_data = st.session_state.data.isnull().sum()
        if missing_data.sum() > 0:
            st.markdown("**缺失值统计**:")
            st.bar_chart(missing_data[missing_data > 0])
    else:
        st.warning("请先加载数据")

def show_settings_page():
    """系统设置页面"""
    st.markdown('<h2 class="sub-header">⚙️ 系统设置</h2>', unsafe_allow_html=True)
    
    # 显示设置
    st.markdown("### 🎨 显示设置")
    theme = st.selectbox("选择主题", ["默认", "深色", "浅色"])
    
    # 分析设置
    st.markdown("### 📊 分析设置")
    significance_level = st.slider("显著性水平", 0.01, 0.10, 0.05, 0.01)
    confidence_level = st.slider("置信水平", 0.90, 0.99, 0.95, 0.01)
    
    # 数据设置
    st.markdown("### 💾 数据设置")
    max_rows = st.number_input("最大显示行数", 100, 10000, 1000)
    
    if st.button("💾 保存设置"):
        st.success("设置已保存！")

# 占位符函数（当模块导入失败时使用）
def show_placeholder_clinical_trial():
    """临床试验设计占位符"""
    st.info("临床试验设计模块正在加载...")
    
    st.markdown("### 🎯 试验类型选择")
    trial_type = st.selectbox("选择试验类型", 
                             ["随机对照试验(RCT)", "队列研究", "病例对照研究", "横断面研究"])
    
    st.markdown("### 👥 受试者管理")
    col1, col2 = st.columns(2)
    with col1:
        st.number_input("目标样本量", min_value=1, value=100)
    with col2:
        st.number_input("实际入组", min_value=0, value=0)

def show_placeholder_statistical_analysis():
    """统计分析占位符"""
    st.info("统计分析模块正在加载...")
    
    if st.session_state.data is not None:
        st.markdown("### 📊 描述性统计")
        st.dataframe(st.session_state.data.describe())
    else:
        st.warning("请先在数据管理模块加载数据")

def show_placeholder_data_management():
    """数据管理占位符"""
    st.info("数据管理模块正在加载...")
    
    st.markdown("### 📤 数据上传")
    uploaded_file = st.file_uploader("选择CSV或Excel文件", 
                                    type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            st.session_state.data = data
            st.success(f"数据上传成功！形状: {data.shape}")
            st.dataframe(data.head())
            
        except Exception as e:
            st.error(f"数据加载失败: {e}")

def show_placeholder_epidemiology():
    """流行病学分析占位符"""
    st.info("流行病学分析模块正在加载...")
    st.markdown("### 📈 流行病学指标计算")
    st.markdown("- 发病率计算")
    st.markdown("- 患病率计算") 
    st.markdown("- 相对危险度(RR)")
    st.markdown("- 比值比(OR)")

def show_placeholder_survival_analysis():
    """生存分析占位符"""
    st.info("生存分析模块正在加载...")
    st.markdown("### ⏱️ 生存分析方法")
    st.markdown("- Kaplan-Meier生存曲线")
    st.markdown("- Cox比例风险模型")
    st.markdown("- Log-rank检验")

def show_placeholder_sample_size():
    """样本量计算占位符"""
    st.info("样本量计算模块正在加载...")
    
    st.markdown("### 🎯 样本量计算")
    
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.number_input("α (第一类错误)", 0.01, 0.10, 0.05)
        power = st.number_input("统计功效 (1-β)", 0.70, 0.99, 0.80)
    
    with col2:
        effect_size = st.number_input("效应量", 0.1, 2.0, 0.5)
        
    if st.button("计算样本量"):
        # 简单的样本量计算示例
        import scipy.stats as stats
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        n = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        st.success(f"估计样本量: {int(n)} (每组)")

def generate_sample_data():
    """生成示例数据"""
    np.random.seed(42)
    n = 200
    
    data = {
        '患者ID': [f'P{i:03d}' for i in range(1, n+1)],
        '年龄': np.random.normal(65, 12, n).astype(int),
        '性别': np.random.choice(['男', '女'], n),
        '治疗组': np.random.choice(['试验组', '对照组'], n),
        '基线血压': np.random.normal(140, 20, n),
        '治疗后血压': np.random.normal(130, 18, n),
        '不良事件': np.random.choice(['无', '轻度', '中度', '重度'], n, p=[0.7, 0.2, 0.08, 0.02]),
        '随访时间': np.random.exponential(12, n),
        '事件发生': np.random.choice([0, 1], n, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    main()
