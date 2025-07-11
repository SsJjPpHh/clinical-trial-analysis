import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go

def main():
    st.set_page_config(
        page_title="临床试验统计分析系统",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 临床试验统计分析系统")
    st.markdown("---")
    
    # 侧边栏导航
    st.sidebar.title("📋 功能导航")
    
    menu_options = [
        "🏠 首页",
        "📊 数据管理",
        "🧪 临床试验分析", 
        "🦠 流行病学分析",
        "🎲 随机化",
        "📈 生存分析",
        "🧮 样本量计算",
        "📄 报告生成"
    ]
    
    selected = st.sidebar.selectbox("选择功能模块", menu_options)
    
    # 根据选择显示相应页面
    if selected == "🏠 首页":
        show_home_page()
    elif selected == "📊 数据管理":
        data_management_page()
    elif selected == "🧪 临床试验分析":
        clinical_trial_page()
    elif selected == "🦠 流行病学分析":
        epidemiology_page()
    elif selected == "🎲 随机化":
        randomization_page()
    elif selected == "📈 生存分析":
        survival_analysis_page()
    elif selected == "🧮 样本量计算":
        sample_size_page()
    elif selected == "📄 报告生成":
        reporting_page()

def show_home_page():
    st.header("🏠 欢迎使用临床试验统计分析系统")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 系统功能")
        st.markdown("""
        - **📊 数据管理**: 数据导入、清洗、预处理
        - **🧪 临床试验分析**: 基线特征、疗效评估
        - **🦠 流行病学分析**: 队列研究、病例对照研究
        - **🎲 随机化**: 随机分组、分层随机化
        - **📈 生存分析**: Kaplan-Meier、Cox回归
        - **🧮 样本量计算**: 各种研究设计的样本量估算
        - **📄 报告生成**: 自动生成统计分析报告
        """)
    
    with col2:
        st.subheader("📋 使用说明")
        st.markdown("""
        1. 从左侧菜单选择需要的功能模块
        2. 上传或输入您的数据
        3. 选择合适的统计方法
        4. 查看分析结果和图表
        5. 生成并下载报告
        
        **注意事项:**
        - 确保数据格式正确
        - 选择适当的统计方法
        - 注意样本量要求
        """)
    
    st.markdown("---")
    st.info("💡 提示：请从左侧菜单选择具体的功能模块开始使用")
    
    # 添加演示数据
    st.subheader("📈 演示数据")
    demo_data = pd.DataFrame({
        '患者ID': range(1, 101),
        '年龄': np.random.normal(65, 12, 100),
        '性别': np.random.choice(['男', '女'], 100),
        '治疗组': np.random.choice(['试验组', '对照组'], 100),
        '疗效评分': np.random.normal(75, 15, 100)
    })
    st.dataframe(demo_data.head())

def data_management_page():
    st.header("📊 数据管理")
    
    uploaded_file = st.file_uploader("上传数据文件", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success("数据上传成功！")
            st.subheader("数据预览")
            st.dataframe(df.head())
            
            st.subheader("数据基本信息")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("行数", df.shape[0])
            with col2:
                st.metric("列数", df.shape[1])
            with col3:
                st.metric("缺失值", df.isnull().sum().sum())
            
            st.subheader("描述性统计")
            st.dataframe(df.describe())
            
        except Exception as e:
            st.error(f"数据读取错误: {str(e)}")
    else:
        st.info("请上传数据文件开始分析")

def clinical_trial_page():
    st.header("🧪 临床试验分析")
    
    # 生成示例数据
    np.random.seed(42)
    n_patients = st.slider("患者数量", 50, 500, 200)
    
    data = pd.DataFrame({
        '患者ID': range(1, n_patients + 1),
        '年龄': np.random.normal(65, 12, n_patients),
        '性别': np.random.choice(['男', '女'], n_patients),
        '治疗组': np.random.choice(['试验组', '对照组'], n_patients),
        '基线评分': np.random.normal(50, 10, n_patients),
        '治疗后评分': np.random.normal(60, 12, n_patients),
        '不良事件': np.random.choice(['无', '轻度', '中度', '重度'], n_patients, p=[0.6, 0.25, 0.1, 0.05])
    })
    
    st.subheader("基线特征分析")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots()
        data['年龄'].hist(bins=20, ax=ax)
        ax.set_title('年龄分布')
        ax.set_xlabel('年龄')
        ax.set_ylabel('频次')
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots()
        data['性别'].value_counts().plot(kind='pie', ax=ax, autopct='%1.1f%%')
        ax.set_title('性别分布')
        st.pyplot(fig)
    
    st.subheader("疗效分析")
    treatment_effect = data.groupby('治疗组')['治疗后评分'].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("试验组平均评分", f"{treatment_effect['试验组']:.2f}")
    with col2:
        st.metric("对照组平均评分", f"{treatment_effect['对照组']:.2f}")
    
    # t检验
    trial_group = data[data['治疗组'] == '试验组']['治疗后评分']
    control_group = data[data['治疗组'] == '对照组']['治疗后评分']
    t_stat, p_value = stats.ttest_ind(trial_group, control_group)
    
    st.subheader("统计检验结果")
    st.write(f"t统计量: {t_stat:.4f}")
    st.write(f"p值: {p_value:.4f}")
    
    if p_value < 0.05:
        st.success("结果具有统计学意义 (p < 0.05)")
    else:
        st.info("结果无统计学意义 (p ≥ 0.05)")

def epidemiology_page():
    st.header("🦠 流行病学分析")
    st.info("流行病学分析功能正在开发中...")
    
    # 简单的2x2表分析
    st.subheader("2×2列联表分析")
    
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("暴露+疾病+", value=20, min_value=0)
        c = st.number_input("暴露+疾病-", value=80, min_value=0)
    with col2:
        b = st.number_input("暴露-疾病+", value=10, min_value=0)
        d = st.number_input("暴露-疾病-", value=90, min_value=0)
    
    if st.button("计算风险比和比值比"):
        # 计算风险比
        risk_exposed = a / (a + c)
        risk_unexposed = b / (b + d)
        risk_ratio = risk_exposed / risk_unexposed if risk_unexposed > 0 else float('inf')
        
        # 计算比值比
        odds_ratio = (a * d) / (b * c) if (b * c) > 0 else float('inf')
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("风险比 (RR)", f"{risk_ratio:.3f}")
        with col2:
            st.metric("比值比 (OR)", f"{odds_ratio:.3f}")

def randomization_page():
    st.header("🎲 随机化")
    
    st.subheader("简单随机化")
    n_subjects = st.number_input("受试者数量", value=100, min_value=1)
    group_ratio = st.selectbox("分组比例", ["1:1", "2:1", "3:1"])
    
    if st.button("生成随机分组"):
        if group_ratio == "1:1":
            groups = np.random.choice(['A组', 'B组'], n_subjects)
        elif group_ratio == "2:1":
            groups = np.random.choice(['A组', 'B组'], n_subjects, p=[2/3, 1/3])
        else:  # 3:1
            groups = np.random.choice(['A组', 'B组'], n_subjects, p=[3/4, 1/4])
        
        result_df = pd.DataFrame({
            '受试者ID': range(1, n_subjects + 1),
            '分组': groups
        })
        
        st.subheader("随机化结果")
        st.dataframe(result_df)
        
        # 分组统计
        group_counts = pd.Series(groups).value_counts()
        st.subheader("分组统计")
        for group, count in group_counts.items():
            st.write(f"{group}: {count}人 ({count/n_subjects*100:.1f}%)")

def survival_analysis_page():
    st.header("📈 生存分析")
    st.info("生存分析功能正在开发中...")
    
    # 简单的生存时间演示
    st.subheader("生存时间数据演示")
    
    np.random.seed(42)
    n_patients = 100
    survival_time = np.random.exponential(12, n_patients)  # 平均生存时间12个月
    censored = np.random.choice([0, 1], n_patients, p=[0.3, 0.7])  # 30%删失
    
    survival_data = pd.DataFrame({
        '患者ID': range(1, n_patients + 1),
        '生存时间(月)': survival_time,
        '事件发生': censored,
        '治疗组': np.random.choice(['试验组', '对照组'], n_patients)
    })
    
    st.dataframe(survival_data.head())
    
    # 简单的生存曲线图
    fig, ax = plt.subplots()
    for group in ['试验组', '对照组']:
        group_data = survival_data[survival_data['治疗组'] == group]
        sorted_times = np.sort(group_data['生存时间(月)'])
        survival_prob = np.arange(len(sorted_times), 0, -1) / len(sorted_times)
        ax.step(sorted_times, survival_prob, label=group, where='post')
    
    ax.set_xlabel('时间(月)')
    ax.set_ylabel('生存概率')
    ax.set_title('生存曲线')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

def sample_size_page():
    st.header("🧮 样本量计算")
    
    st.subheader("两组均数比较的样本量计算")
    
    col1, col2 = st.columns(2)
    with col1:
        alpha = st.selectbox("显著性水平 (α)", [0.05, 0.01, 0.001], index=0)
        power = st.selectbox("检验效能 (1-β)", [0.8, 0.9, 0.95], index=0)
    
    with col2:
        effect_size = st.number_input("效应量", value=0.5, min_value=0.1, max_value=2.0, step=0.1)
        ratio = st.selectbox("组间比例", ["1:1", "2:1", "3:1"], index=0)
    
    if st.button("计算样本量"):
        from scipy.stats import norm
        
        z_alpha = norm.ppf(1 - alpha/2)
        z_beta = norm.ppf(power)
        
        if ratio == "1:1":
            k = 1
        elif ratio == "2:1":
            k = 2
        else:
            k = 3
        
        n1 = ((z_alpha + z_beta) ** 2 * (1 + 1/k)) / (effect_size ** 2)
        n2 = n1 / k
        
        st.subheader("样本量计算结果")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("第一组样本量", f"{int(np.ceil(n1))}")
        with col2:
            st.metric("第二组样本量", f"{int(np.ceil(n2))}")
        with col3:
            st.metric("总样本量", f"{int(np.ceil(n1 + n2))}")

def reporting_page():
    st.header("📄 报告生成")
    st.info("报告生成功能正在开发中...")
    
    st.subheader("报告模板")
    report_type = st.selectbox("选择报告类型", [
        "临床试验统计分析报告",
        "流行病学研究报告", 
        "生存分析报告",
        "样本量计算报告"
    ])
    
    if st.button("生成报告"):
        st.success(f"已生成 {report_type}")
        st.markdown("""
        ### 统计分析报告
        
        **研究背景**: 本研究旨在评估新药物的疗效和安全性。
        
        **研究方法**: 采用随机对照试验设计，将患者随机分为试验组和对照组。
        
        **统计分析**: 使用t检验比较两组间的疗效差异。
        
        **结果**: 
        - 试验组平均疗效评分: XX.XX
        - 对照组平均疗效评分: XX.XX
        - p值: X.XXX
        
        **结论**: 根据统计分析结果...
        """)

if __name__ == "__main__":
    main()
