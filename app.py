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
        ("流行病学分析", False, "正在开发中"),
        ("生存分析", False, "计划中的功能"),
        ("样本量计算", False, "计划中的功能"),
        ("随机化工具", False, "计划中的功能"),
        ("报告生成", True, "基础报告功能可用")
    ]
    
    col1, col2 = st.columns(2)
    
    for i, (module_name, status, description) in enumerate(modules_status):
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            status_icon = "✅" if status else "❌"
            status_class = "status-available" if status else "status-unavailable"
            status_text = "可用" if status else "不可用"
            
            st.markdown(f"""
            <div class="metric-card">
                <h4>{status_icon} {module_name}</h4>
                <p class="{status_class}">状态: {status_text}</p>
                <p><small>{description}</small></p>
            </div>
            """, unsafe_allow_html=True)
    
    # 快速开始指南
    st.markdown("---")
    st.markdown("## 🚀 快速开始")
    
    st.markdown("""
    1. **上传数据**: 在"基础统计"页面上传您的CSV文件
    2. **选择变量**: 选择要分析的变量和分组
    3. **查看结果**: 系统自动生成统计结果和图表
    4. **导出报告**: 下载分析结果和可视化图表
    """)

def show_basic_stats():
    """基础统计分析页面"""
    st.title("📊 基础统计分析")
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "上传数据文件",
        type=['csv', 'xlsx', 'xls'],
        help="支持CSV和Excel格式文件"
    )
    
    if uploaded_file is not None:
        try:
            # 读取文件
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ 成功上传文件: {uploaded_file.name}")
            st.markdown(f"**数据维度**: {df.shape[0]} 行 × {df.shape[1]} 列")
            
            # 数据预览
            st.markdown("### 📋 数据预览")
            st.dataframe(df.head(10))
            
            # 数据基本信息
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📈 数据类型")
                data_types = df.dtypes.to_frame('数据类型')
                data_types['非空值数量'] = df.count()
                data_types['缺失值数量'] = df.isnull().sum()
                st.dataframe(data_types)
            
            with col2:
                st.markdown("### 📊 描述性统计")
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    st.dataframe(df[numeric_cols].describe())
                else:
                    st.info("没有发现数值型变量")
            
            # 变量选择和分析
            if len(numeric_cols) > 0:
                st.markdown("---")
                st.markdown("### 🔍 统计分析")
                
                analysis_col1, analysis_col2 = st.columns(2)
                
                with analysis_col1:
                    selected_var = st.selectbox("选择分析变量", numeric_cols)
                
                with analysis_col2:
                    group_cols = ['无分组'] + list(df.columns)
                    selected_group = st.selectbox("选择分组变量", group_cols)
                
                if st.button("🔬 执行分析"):
                    perform_statistical_analysis(df, selected_var, selected_group)
        
        except Exception as e:
            st.error(f"❌ 文件读取错误: {str(e)}")
    
    else:
        # 显示示例数据
        st.markdown("### 🎯 示例数据演示")
        st.info("👆 请上传您的数据文件，或查看下方示例数据分析")
        
        # 生成示例数据
        generate_sample_data_analysis()

def perform_statistical_analysis(df, variable, group_var):
    """执行统计分析"""
    st.markdown("#### 📊 分析结果")
    
    if group_var == '无分组':
        # 单变量分析
        data = df[variable].dropna()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("样本量", len(data))
        with col2:
            st.metric("均值", f"{data.mean():.2f}")
        with col3:
            st.metric("标准差", f"{data.std():.2f}")
        with col4:
            st.metric("中位数", f"{data.median():.2f}")
        
        # 正态性检验
        if len(data) > 3:
            stat, p_value = stats.shapiro(data)
            st.markdown(f"**正态性检验 (Shapiro-Wilk)**: 统计量 = {stat:.4f}, p值 = {p_value:.4f}")
            
            if p_value > 0.05:
                st.success("✅ 数据符合正态分布 (p > 0.05)")
            else:
                st.warning("⚠️ 数据不符合正态分布 (p ≤ 0.05)")
    
    else:
        # 分组分析
        if group_var in df.columns:
            groups = df.groupby(group_var)[variable].apply(list)
            
            st.markdown("##### 📋 分组描述统计")
            group_stats = df.groupby(group_var)[variable].agg(['count', 'mean', 'std', 'median'])
            st.dataframe(group_stats)
            
            # 组间比较
            if len(groups) == 2:
                group_names = list(groups.index)
                group1_data = np.array(groups.iloc[0])
                group2_data = np.array(groups.iloc[1])
                
                # t检验
                t_stat, t_p = ttest_ind(group1_data, group2_data)
                st.markdown(f"**独立样本t检验**: t = {t_stat:.4f}, p = {t_p:.4f}")
                
                # Mann-Whitney U检验
                u_stat, u_p = mannwhitneyu(group1_data, group2_data)
                st.markdown(f"**Mann-Whitney U检验**: U = {u_stat:.4f}, p = {u_p:.4f}")
                
                if t_p < 0.05:
                    st.success(f"✅ 组间差异显著 (p < 0.05)")
                else:
                    st.info("ℹ️ 组间差异不显著 (p ≥ 0.05)")

def generate_sample_data_analysis():
    """生成示例数据分析"""
    # 生成示例数据
    np.random.seed(42)
    n_samples = 200
    
    sample_data = pd.DataFrame({
        'PatientID': range(1, n_samples + 1),
        'Treatment': np.random.choice(['药物A', '药物B', '安慰剂'], n_samples, p=[0.4, 0.4, 0.2]),
        'Age': np.random.normal(55, 12, n_samples),
        'Gender': np.random.choice(['男', '女'], n_samples),
        'BaselineScore': np.random.normal(50, 15, n_samples),
        'EndpointScore': np.random.normal(60, 18, n_samples),
        'Response': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'SideEffects': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    })
    
    # 添加一些逻辑关系
    sample_data.loc[sample_data['Treatment'] == '药物A', 'EndpointScore'] += 5
    sample_data.loc[sample_data['Treatment'] == '药物B', 'EndpointScore'] += 3
    
    st.markdown("#### 📋 示例数据")
    st.dataframe(sample_data.head(10))
    
    # 示例分析
    st.markdown("#### 📊 示例分析结果")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### 治疗组基线特征")
        baseline_stats = sample_data.groupby('Treatment')['Age'].agg(['count', 'mean', 'std'])
        st.dataframe(baseline_stats)
    
    with col2:
        st.markdown("##### 疗效终点分析")
        endpoint_stats = sample_data.groupby('Treatment')['EndpointScore'].agg(['count', 'mean', 'std'])
        st.dataframe(endpoint_stats)

def show_data_visualization():
    """数据可视化页面"""
    st.title("📈 数据可视化")
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "上传数据文件用于可视化",
        type=['csv', 'xlsx', 'xls'],
        key="viz_upload"
    )
    
    if uploaded_file is not None:
        try:
            # 读取文件
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ 数据加载成功: {df.shape[0]} 行 × {df.shape[1]} 列")
            
            # 图表类型选择
            chart_type = st.selectbox(
                "选择图表类型",
                ["散点图", "箱线图", "直方图", "条形图", "热力图"]
            )
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if chart_type == "散点图" and len(numeric_cols) >= 2:
                col1, col2, col3 = st.columns(3)
                with col1:
                    x_var = st.selectbox("X轴变量", numeric_cols)
                with col2:
                    y_var = st.selectbox("Y轴变量", numeric_cols)
                with col3:
                    color_var = st.selectbox("颜色分组", ['无'] + categorical_cols)
                
                if st.button("生成散点图"):
                    fig = px.scatter(
                        df, x=x_var, y=y_var,
                        color=color_var if color_var != '无' else None,
                        title=f"{x_var} vs {y_var}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "箱线图" and len(numeric_cols) >= 1:
                col1, col2 = st.columns(2)
                with col1:
                    y_var = st.selectbox("数值变量", numeric_cols)
                with col2:
                    x_var = st.selectbox("分组变量", ['无'] + categorical_cols)
                
                if st.button("生成箱线图"):
                    fig = px.box(
                        df, y=y_var,
                        x=x_var if x_var != '无' else None,
                        title=f"{y_var} 的分布"
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            elif chart_type == "直方图" and len(numeric_cols) >= 1:
                col1, col2 = st.columns(2)
                with col1:
                    var = st.selectbox("选择变量", numeric_cols)
                with col2:
                    bins = st.slider("分组数量", 10, 50, 20)
                
                if st.button("生成直方图"):
                    fig = px.histogram(df, x=var, nbins=bins, title=f"{var} 的分布")
                    st.plotly_chart(fig, use_container_width=True)
            
            else:
                st.info("请确保数据包含适当类型的变量以生成选定的图表")
        
        except Exception as e:
            st.error(f"❌ 数据处理错误: {str(e)}")
    
    else:
        # 示例可视化
        st.markdown("### 🎯 示例可视化")
        show_sample_visualizations()

def show_sample_visualizations():
    """显示示例可视化"""
    # 生成示例数据
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'Treatment': np.random.choice(['Treatment A', 'Treatment B', 'Control'], 150),
        'Age': np.random.normal(50, 15, 150),
        'Response': np.random.normal(75, 20, 150),
        'Gender': np.random.choice(['Male', 'Female'], 150)
    })
    
    # 创建示例图表
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📊 治疗组响应分布")
        fig1 = px.box(sample_data, x='Treatment', y='Response',
                     title='不同治疗组的响应分布')
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.markdown("#### 📈 年龄与响应关系")
        fig2 = px.scatter(sample_data, x='Age', y='Response',
                         color='Treatment', title='年龄与治疗响应的关系')
        st.plotly_chart(fig2, use_container_width=True)

def main():
    """主函数"""
    # 侧边栏导航
    st.sidebar.title("📋 功能导航")
    st.sidebar.markdown("---")
    
    # 导航菜单
    menu_options = [
        "🏠 首页",
        "📊 基础统计",
        "📈 数据可视化",
        "📋 报告生成",
        "ℹ️ 帮助文档"
    ]
    
    selected = st.sidebar.selectbox("选择功能模块", menu_options)
    
    # 侧边栏信息
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📞 联系信息")
    st.sidebar.info("""
    **技术支持**: support@clinicaltrial.com
    **版本**: v1.0.0
    **更新日期**: 2024年
    """)
    
    # 根据选择显示不同页面
    if selected == "🏠 首页":
        show_homepage()
    elif selected == "📊 基础统计":
        show_basic_stats()
    elif selected == "📈 数据可视化":
        show_data_visualization()
    elif selected == "📋 报告生成":
        st.title("📋 报告生成")
        st.info("🚧 报告生成功能正在开发中，敬请期待！")
        st.markdown("""
        ### 计划功能:
        - 📄 PDF报告导出
        - 📊 图表批量导出
        - 📈 统计结果汇总
        - 📋 自定义报告模板
        """)
    elif selected == "ℹ️ 帮助文档":
        st.title("ℹ️ 帮助文档")
        st.markdown("""
        ### 📚 使用指南
        
        #### 1. 数据准备
        - 支持CSV和Excel格式文件
        - 确保数据格式正确，包含列标题
        - 建议数据清洗后再上传
        
        #### 2. 统计分析
        - 选择合适的统计方法
        - 检查数据分布和假设条件
        - 解读统计结果的临床意义
        
        #### 3. 可视化
        - 根据数据类型选择合适的图表
        - 注意图表的可读性和美观性
        - 添加适当的标题和标签
        
        #### 4. 常见问题
        - **Q**: 支持哪些文件格式？
        - **A**: 目前支持CSV、Excel (.xlsx, .xls) 格式
        
        - **Q**: 如何处理缺失值？
        - **A**: 系统会自动识别并在分析中排除缺失值
        
        - **Q**: 统计结果如何解读？
        - **A**: 建议结合临床背景和统计学知识进行解读
        """)

if __name__ == "__main__":
    main()
