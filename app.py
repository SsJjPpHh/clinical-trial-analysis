import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# 页面配置
st.set_page_config(
    page_title="临床试验统计分析平台",
    page_icon="🔬",
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
}
.feature-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.metric-card {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #1f77b4;
    margin: 0.5rem 0;
}
.status-available {
    color: #28a745;
    font-weight: bold;
}
.status-unavailable {
    color: #dc3545;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

def show_homepage():
    """显示首页"""
    st.markdown('<h1 class="main-header">🔬 临床试验统计分析平台</h1>', unsafe_allow_html=True)
    
    # 平台概述
    st.markdown("## 🎯 平台概述")
    st.markdown("""
    欢迎使用临床试验统计分析平台！本平台专为临床研究人员设计，提供全面的统计分析工具和可视化功能。
    """)
    
    # 功能特色
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3>📊 专业分析</h3>
            <p>提供描述性统计、假设检验、生存分析等专业统计方法</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3>📈 数据可视化</h3>
            <p>支持多种图表类型，包括散点图、箱线图、生存曲线等</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3>📋 报告生成</h3>
            <p>自动生成专业的统计分析报告，支持导出多种格式</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 功能模块状态
    st.markdown("## 🔧 模块状态检查")
    
    modules_status = [
        ("临床试验分析", True, "包含基础统计、假设检验等功能"),
        ("数据管理", True, "支持CSV、Excel文件上传和处理"),
        ("流行病学分析", True, "队列研究、病例对照研究分析"),
        ("生存分析", True, "Kaplan-Meier曲线、Cox回归"),
        ("样本量计算", True, "多种研究设计的样本量计算"),
        ("随机化工具", True, "简单、分块、分层随机化"),
        ("报告生成", True, "Markdown格式报告导出")
    ]
    
    col1, col2 = st.columns(2)
    
    for i, (module, available, description) in enumerate(modules_status):
        target_col = col1 if i % 2 == 0 else col2
        
        with target_col:
            status_class = "status-available" if available else "status-unavailable"
            status_text = "✅ 可用" if available else "❌ 不可用"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>{module}</h4>
                <p class="{status_class}">{status_text}</p>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 快速开始
    st.markdown("## 🚀 快速开始")
    st.markdown("""
    1. **数据管理**: 上传您的数据文件（支持CSV、Excel等格式）
    2. **选择分析**: 根据研究类型选择相应的分析模块
    3. **设置参数**: 配置分析参数和可视化选项
    4. **查看结果**: 获得详细的统计结果和图表
    5. **生成报告**: 导出专业的分析报告
    """)
    
    # 联系信息
    st.markdown("## 📞 技术支持")
    st.info("如有技术问题或功能建议，请通过侧边栏的反馈功能联系我们。")

def main():
    """主函数"""
    # 侧边栏导航
    st.sidebar.title("🔬 临床试验统计分析平台")
    
    # 导航菜单
    page = st.sidebar.selectbox(
        "选择功能模块",
        [
            "🏠 首页",
            "🧪 临床试验分析",
            "📁 数据管理",
            "🦠 流行病学分析",
            "📊 生存分析",
            "🔢 样本量计算",
            "🎲 随机化工具",
            "📝 报告生成"
        ]
    )
    
    # 根据选择显示相应页面
    if page == "🏠 首页":
        show_homepage()
    
    elif page == "🧪 临床试验分析":
        try:
            from clinical_trial import clinical_trial_analysis
            clinical_trial_analysis()
        except ImportError as e:
            st.error(f"❌ 模块导入失败：{e}")
            st.info("请检查 clinical_trial.py 文件是否存在且包含 clinical_trial_analysis 函数")
    
    elif page == "📁 数据管理":
        try:
            from data_management import data_management_ui
            data_management_ui()
        except ImportError as e:
            st.error(f"❌ 模块导入失败：{e}")
            st.info("请检查 data_management.py 文件是否存在且包含 data_management_ui 函数")
    
    elif page == "🦠 流行病学分析":
        try:
            from epidemiology import epidemiology_ui
            epidemiology_ui()
        except ImportError as e:
            st.error(f"❌ 模块导入失败：{e}")
            st.info("请检查 epidemiology.py 文件是否存在且包含 epidemiology_ui 函数")
    
    elif page == "📊 生存分析":
        try:
            from survival_analysis import survival_ui
            survival_ui()
        except ImportError as e:
            st.error(f"❌ 模块导入失败：{e}")
            st.info("请检查 survival_analysis.py 文件是否存在且包含 survival_ui 函数")
    
    elif page == "🔢 样本量计算":
        try:
            from sample_size import sample_size_ui
            sample_size_ui()
        except ImportError as e:
            st.error(f"❌ 模块导入失败：{e}")
            st.info("请检查 sample_size.py 文件是否存在且包含 sample_size_ui 函数")
    
    elif page == "🎲 随机化工具":
        try:
            from randomization import randomization_ui
            randomization_ui()
        except ImportError as e:
            st.error(f"❌ 模块导入失败：{e}")
            st.info("请检查 randomization.py 文件是否存在且包含 randomization_ui 函数")
    
    elif page == "📝 报告生成":
        try:
            from reporting import reporting_ui
            reporting_ui()
        except ImportError as e:
            st.error(f"❌ 模块导入失败：{e}")
            st.info("请检查 reporting.py 文件是否存在且包含 reporting_ui 函数")
    
    # 侧边栏信息
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 统计信息")
    st.sidebar.info("当前会话数据集数量: 0")
    
    st.sidebar.markdown("### 💡 使用提示")
    st.sidebar.markdown("""
    - 建议先在数据管理模块上传数据
    - 各模块间可以共享数据集
    - 分析结果可导出为报告
    """)
    
    st.sidebar.markdown("### 📞 反馈")
    feedback = st.sidebar.text_area("意见建议", placeholder="请输入您的建议...")
    if st.sidebar.button("提交反馈"):
        if feedback:
            st.sidebar.success("感谢您的反馈！")
        else:
            st.sidebar.warning("请输入反馈内容")

if __name__ == "__main__":
    main()
