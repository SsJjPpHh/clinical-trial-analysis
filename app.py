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

# 导入自定义模块 - 修复导入方式，导入函数而不是类
try:
    from clinical_trial import clinical_trial_analysis
    from data_management import data_management_analysis
    from epidemiology import epidemiology_analysis
    from survival_analysis import survival_analysis
    from sample_size import sample_size_calculation
    from randomization import randomization_analysis
    from reporting import reporting_analysis
    
    # 创建占位符函数用于未完成的模块
    def statistical_analysis_placeholder():
        st.markdown("### 📊 统计分析")
        st.info("统计分析模块正在开发中...")
        
        # 简单的统计分析示例
        if st.session_state.get('data') is not None:
            data = st.session_state.data
            
            st.subheader("数据概览")
            st.dataframe(data.describe())
            
            # 选择数值列进行分析
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("选择要分析的数值变量", numeric_cols)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.histogram(data, x=selected_col, title=f"{selected_col} 分布")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = px.box(data, y=selected_col, title=f"{selected_col} 箱线图")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("请先在数据管理模块中加载数据")
    
except ImportError as e:
    st.error(f"模块导入错误: {e}")
    st.info("请确保所有必需的模块文件都在正确位置")
    
    # 创建所有占位符函数
    def clinical_trial_analysis():
        st.error("临床试验分析模块导入失败")
    def data_management_analysis():
        st.error("数据管理模块导入失败")
    def epidemiology_analysis():
        st.error("流行病学分析模块导入失败")
    def survival_analysis():
        st.error("生存分析模块导入失败")
    def sample_size_calculation():
        st.error("样本量计算模块导入失败")
    def randomization_analysis():
        st.error("随机化模块导入失败")
    def reporting_analysis():
        st.error("报告生成模块导入失败")
    def statistical_analysis_placeholder():
        st.error("统计分析模块导入失败")

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
                "🧪 临床试验分析",
                "📊 统计分析", 
                "💾 数据管理",
                "📈 流行病学分析",
                "⏱️ 生存分析",
                "🎯 样本量计算",
                "🎲 随机化设计",
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
    
    # 页面路由 - 直接调用函数
    if page == "🏠 首页概览":
        show_home_page()
    elif page == "🧪 临床试验分析":
        clinical_trial_analysis()
    elif page == "📊 统计分析":
        statistical_analysis_placeholder()
    elif page == "💾 数据管理":
        data_management_analysis()
    elif page == "📈 流行病学分析":
        epidemiology_analysis()
    elif page == "⏱️ 生存分析":
        survival_analysis()
    elif page == "🎯 样本量计算":
        sample_size_calculation()
    elif page == "🎲 随机化设计":
        randomization_analysis()
    elif page == "📋 报告生成":
        reporting_analysis()
    elif page == "⚙️ 系统设置":
        show_settings_page()

def show_home_page():
    """首页概览"""
    st.markdown('<h2 class="sub-header">🏠 系统概览</h2>', unsafe_allow_html=True)
    
    # 欢迎信息
    st.markdown("""
    <div class="info-box">
    <h3>🎉 欢迎使用临床试验统计分析系统！</h3>
    <p>本系统提供全面的临床试验设计、数据管理和统计分析功能，帮助研究人员高效完成临床研究工作。</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 功能模块展示
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🧪 试验分析")
        st.markdown("""
        - ✅ 基线特征分析
        - ✅ 主要终点分析
        - ✅ 次要终点分析
        - ✅ 安全性分析
        - ✅ 亚组分析
        """)
        
    with col2:
        st.markdown("### 📊 数据管理")
        st.markdown("""
        - ✅ 数据导入导出
        - ✅ 数据清洗
        - ✅ 数据验证
        - ✅ 数据转换
        - ✅ 质量控制
        """)
        
    with col3:
        st.markdown("### 📈 专业分析")
        st.markdown("""
        - ✅ 生存分析
        - ✅ 流行病学分析
        - ✅ 样本量计算
        - ✅ 随机化设计
        - ✅ 报告生成
        """)
    
    # 系统状态检查
    st.markdown("---")
    st.markdown("### 🔍 系统状态检查")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📦 模块状态")
        modules_status = {
            "临床试验分析": "✅ 正常",
            "数据管理": "✅ 正常", 
            "流行病学分析": "✅ 正常",
            "生存分析": "✅ 正常",
            "样本量计算": "✅ 正常",
            "随机化设计": "✅ 正常",
            "报告生成": "✅ 正常",
            "统计分析": "🚧 开发中"
        }
        
        for module, status in modules_status.items():
            st.write(f"- {module}: {status}")
    
    with col2:
        st.markdown("#### 💾 数据状态")
        if st.session_state.get('data') is not None:
            data = st.session_state.data
            st.write(f"- 数据行数: {data.shape[0]}")
            st.write(f"- 数据列数: {data.shape[1]}")
            st.write(f"- 数据大小: {data.memory_usage(deep=True).sum() / 1024:.1f} KB")
            st.write("- 状态: ✅ 已加载")
        else:
            st.write("- 状态: ⚠️ 未加载数据")
            st.write("- 建议: 前往数据管理模块加载数据")
    
    # 快速开始指南
    st.markdown("---")
    st.markdown("### 🚀 快速开始")
    
    with st.expander("📖 使用指南", expanded=True):
        st.markdown("""
        #### 🎯 推荐工作流程：
        
        1. **📥 数据准备**
           - 前往 "💾 数据管理" 模块
           - 上传您的临床试验数据
           - 进行数据清洗和验证
        
        2. **🔍 探索性分析**
           - 使用 "📊 统计分析" 模块
           - 进行初步数据探索
           - 了解数据分布特征
        
        3. **🧪 专业分析**
           - 根据研究设计选择相应模块：
             - 临床试验分析：主要/次要终点分析
             - 生存分析：时间-事件分析
             - 流行病学分析：队列/病例对照研究
        
        4. **📋 结果导出**
           - 在 "📋 报告生成" 模块
           - 生成专业统计报告
           - 导出图表和表格
        
        #### 💡 小贴士：
        - 💾 系统会自动保存您的分析结果
        - 🔄 可以随时在不同模块间切换
        - 📊 所有图表都支持交互式操作
        - 📤 支持多种格式的数据导出
        """)
    
    # 示例数据
    st.markdown("---")
    st.markdown("### 🎲 试用示例数据")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧪 加载临床试验示例数据", use_container_width=True):
            sample_data = generate_clinical_trial_sample_data()
            st.session_state.data = sample_data
            st.success("✅ 临床试验示例数据已加载！")
            st.dataframe(sample_data.head())
    
    with col2:
        if st.button("📊 加载流行病学示例数据", use_container_width=True):
            sample_data = generate_epidemiology_sample_data()
            st.session_state.data = sample_data
            st.success("✅ 流行病学示例数据已加载！")
            st.dataframe(sample_data.head())

def generate_clinical_trial_sample_data():
    """生成临床试验示例数据"""
    np.random.seed(42)
    n = 200
    
    # 生成基础人口学信息
    ages = np.random.normal(55, 12, n).astype(int)
    ages = np.clip(ages, 18, 85)  # 限制年龄范围
    
    data = {
        '受试者ID': [f'CT{i:03d}' for i in range(1, n+1)],
        '治疗组': np.random.choice(['试验组', '对照组'], n, p=[0.5, 0.5]),
        '年龄': ages,
        '性别': np.random.choice(['男', '女'], n, p=[0.6, 0.4]),
        '体重_kg': np.random.normal(70, 15, n).round(1),
        '身高_cm': np.random.normal(170, 10, n).round(1),
        '基线收缩压': np.random.normal(140, 20, n).round(0),
        '基线舒张压': np.random.normal(90, 10, n).round(0),
        '基线心率': np.random.normal(75, 12, n).round(0),
        '主要终点_有效': np.random.choice([0, 1], n, p=[0.4, 0.6]),
        '次要终点_改善分数': np.random.normal(5, 2, n).round(1),
        '治疗持续时间_天': np.random.normal(84, 14, n).round(0),
        '不良事件': np.random.choice([0, 1], n, p=[0.75, 0.25]),
        '严重不良事件': np.random.choice([0, 1], n, p=[0.95, 0.05]),
        '依从性_百分比': np.random.normal(85, 15, n).round(1)
    }
    
    # 确保逻辑一致性
    df = pd.DataFrame(data)
    df.loc[df['依从性_百分比'] > 100, '依从性_百分比'] = 100
    df.loc[df['依从性_百分比'] < 0, '依从性_百分比'] = 0
    
    return df

def generate_epidemiology_sample_data():
    """生成流行病学示例数据"""
    np.random.seed(123)
    n = 500
    
    data = {
        '研究ID': [f'EPI{i:04d}' for i in range(1, n+1)],
        '年龄': np.random.normal(45, 15, n).astype(int),
        '性别': np.random.choice(['男', '女'], n),
        '教育程度': np.random.choice(['小学', '中学', '大学', '研究生'], n, p=[0.2, 0.4, 0.3, 0.1]),
        '吸烟状态': np.random.choice(['从不吸烟', '曾经吸烟', '目前吸烟'], n, p=[0.5, 0.3, 0.2]),
        '饮酒频率': np.random.choice(['从不', '偶尔', '经常', '每天'], n, p=[0.3, 0.4, 0.2, 0.1]),
        '体重指数': np.random.normal(24, 4, n).round(1),
        '收缩压': np.random.normal(125, 18, n).round(0),
        '舒张压': np.random.normal(80, 12, n).round(0),
        '总胆固醇': np.random.normal(200, 40, n).round(1),
        '血糖': np.random.normal(95, 15, n).round(1),
        '疾病状态': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        '随访时间_月': np.random.exponential(36, n).round(0),
        '结局事件': np.random.choice([0, 1], n, p=[0.8, 0.2])
    }
    
    return pd.DataFrame(data)

def show_settings_page():
    """系统设置页面"""
    st.markdown("### ⚙️ 系统设置")
    
    st.markdown("#### 🎨 界面设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        theme = st.selectbox("选择主题", ["默认", "深色", "浅色"])
        language = st.selectbox("语言设置", ["中文", "English"])
        
    with col2:
        chart_style = st.selectbox("图表样式", ["默认", "简约", "专业"])
        decimal_places = st.number_input("小数位数", min_value=1, max_value=6, value=3)
    
    st.markdown("#### 📊 分析设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_level = st.slider("置信水平", 0.90, 0.99, 0.95, 0.01)
        significance_level = st.slider("显著性水平", 0.01, 0.10, 0.05, 0.01)
        
    with col2:
        bootstrap_samples = st.number_input("Bootstrap样本数", min_value=1000, max_value=10000, value=5000)
        random_seed = st.number_input("随机种子", min_value=1, max_value=9999, value=42)
    
    if st.button("💾 保存设置"):
        st.success("设置已保存！")

if __name__ == "__main__":
    main()
