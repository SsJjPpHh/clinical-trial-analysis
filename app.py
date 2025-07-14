import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 尝试导入自定义模块，如果失败则使用内置功能
try:
    from clinical_trial import clinical_trial_analysis as clinical_trial_module
    CLINICAL_TRIAL_AVAILABLE = True
except ImportError:
    CLINICAL_TRIAL_AVAILABLE = False

try:
    from epidemiology import epidemiology_analysis as epidemiology_module
    EPIDEMIOLOGY_AVAILABLE = True
except ImportError:
    EPIDEMIOLOGY_AVAILABLE = False

try:
    from survival import survival_analysis as survival_module
    SURVIVAL_AVAILABLE = True
except ImportError:
    SURVIVAL_AVAILABLE = False

# 设置页面配置
st.set_page_config(
    page_title="临床试验数据分析平台",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def clinical_trial_analysis():
    """临床试验数据分析模块"""
    if CLINICAL_TRIAL_AVAILABLE:
        # 使用导入的模块
        clinical_trial_module()
    else:
        # 使用内置功能
        st.header("🧪 临床试验数据分析")
        
        # 侧边栏配置
        st.sidebar.subheader("分析配置")
        analysis_type = st.sidebar.selectbox(
            "选择分析类型",
            ["基础统计分析", "疗效对比分析", "安全性分析", "亚组分析"]
        )
        
        # 数据上传
        uploaded_file = st.file_uploader("上传临床试验数据", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            # 读取数据
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"数据上传成功！共 {len(df)} 行，{len(df.columns)} 列")
                
                # 显示数据预览
                with st.expander("数据预览", expanded=True):
                    st.dataframe(df.head())
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("总样本数", len(df))
                    with col2:
                        st.metric("变量数", len(df.columns))
                    with col3:
                        st.metric("缺失值", df.isnull().sum().sum())
                
                # 根据分析类型执行相应分析
                if analysis_type == "基础统计分析":
                    basic_statistics_analysis(df)
                elif analysis_type == "疗效对比分析":
                    efficacy_analysis(df)
                elif analysis_type == "安全性分析":
                    safety_analysis(df)
                elif analysis_type == "亚组分析":
                    subgroup_analysis(df)
                    
            except Exception as e:
                st.error(f"数据读取失败: {str(e)}")
        else:
            # 显示示例数据
            st.info("请上传数据文件，或使用下面的示例数据进行演示")
            if st.button("生成示例数据"):
                df = generate_sample_data()
                st.session_state.sample_data = df
                st.success("示例数据生成成功！")
            
            if 'sample_data' in st.session_state:
                df = st.session_state.sample_data
                st.dataframe(df.head())
                
                if analysis_type == "基础统计分析":
                    basic_statistics_analysis(df)
                elif analysis_type == "疗效对比分析":
                    efficacy_analysis(df)
                elif analysis_type == "安全性分析":
                    safety_analysis(df)
                elif analysis_type == "亚组分析":
                    subgroup_analysis(df)

def generate_sample_data():
    """生成示例临床试验数据"""
    np.random.seed(42)
    n_patients = 200
    
    data = {
        'patient_id': range(1, n_patients + 1),
        'age': np.random.normal(65, 12, n_patients).astype(int),
        'gender': np.random.choice(['男', '女'], n_patients),
        'treatment_group': np.random.choice(['试验组', '对照组'], n_patients),
        'baseline_score': np.random.normal(50, 10, n_patients),
        'endpoint_score': np.random.normal(45, 12, n_patients),
        'adverse_events': np.random.choice(['无', '轻度', '中度', '重度'], n_patients, p=[0.6, 0.25, 0.1, 0.05]),
        'duration_days': np.random.normal(90, 15, n_patients).astype(int)
    }
    
    # 添加一些逻辑关系
    df = pd.DataFrame(data)
    
    # 试验组疗效更好
    trial_mask = df['treatment_group'] == '试验组'
    df.loc[trial_mask, 'endpoint_score'] -= np.random.normal(5, 2, trial_mask.sum())
    
    # 计算疗效改善
    df['improvement'] = df['baseline_score'] - df['endpoint_score']
    df['response'] = df['improvement'] > 10
    
    return df

def basic_statistics_analysis(df):
    """基础统计分析"""
    st.subheader("📊 基础统计分析")
    
    # 描述性统计
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**数值变量描述性统计**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.dataframe(df[numeric_cols].describe())
    
    with col2:
        st.write("**分类变量频数统计**")
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols[:3]:  # 显示前3个分类变量
            st.write(f"**{col}**")
            st.write(df[col].value_counts())
    
    # 可视化
    st.subheader("数据可视化")
    
    if len(numeric_cols) > 0:
        # 选择要可视化的变量
        viz_col = st.selectbox("选择要可视化的数值变量", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 直方图
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.hist(df[viz_col].dropna(), bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_title(f'{viz_col} 分布直方图')
            ax.set_xlabel(viz_col)
            ax.set_ylabel('频数')
            st.pyplot(fig)
        
        with col2:
            # 箱线图
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.boxplot(df[viz_col].dropna())
            ax.set_title(f'{viz_col} 箱线图')
            ax.set_ylabel(viz_col)
            st.pyplot(fig)

def efficacy_analysis(df):
    """疗效对比分析"""
    st.subheader("🎯 疗效对比分析")
    
    # 检查是否有治疗组变量
    group_cols = [col for col in df.columns if 'group' in col.lower() or 'treatment' in col.lower()]
    
    if len(group_cols) == 0:
        st.warning("未找到治疗组变量，请确保数据中包含治疗组信息")
        return
    
    group_col = st.selectbox("选择治疗组变量", group_cols)
    
    # 选择疗效指标
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.warning("未找到数值型疗效指标")
        return
    
    efficacy_col = st.selectbox("选择疗效指标", numeric_cols)
    
    # 按组统计
    group_stats = df.groupby(group_col)[efficacy_col].agg(['count', 'mean', 'std', 'median']).round(2)
    
    st.write("**各组疗效指标统计**")
    st.dataframe(group_stats)
    
    # 统计检验
    groups = df[group_col].unique()
    if len(groups) == 2:
        group1_data = df[df[group_col] == groups[0]][efficacy_col].dropna()
        group2_data = df[df[group_col] == groups[1]][efficacy_col].dropna()
        
        # t检验
        t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
        
        st.write("**统计检验结果**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("t统计量", f"{t_stat:.4f}")
        with col2:
            st.metric("p值", f"{p_value:.4f}")
        with col3:
            significance = "显著" if p_value < 0.05 else "不显著"
            st.metric("统计显著性", significance)
    
    # 可视化对比
    col1, col2 = st.columns(2)
    
    with col1:
        # 箱线图对比
        fig, ax = plt.subplots(figsize=(8, 6))
        df.boxplot(column=efficacy_col, by=group_col, ax=ax)
        ax.set_title(f'{efficacy_col} 按 {group_col} 分组对比')
        plt.suptitle('')  # 移除默认标题
        st.pyplot(fig)
    
    with col2:
        # 小提琴图
        fig, ax = plt.subplots(figsize=(8, 6))
        for i, group in enumerate(groups):
            data = df[df[group_col] == group][efficacy_col].dropna()
            ax.violinplot([data], positions=[i], showmeans=True)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups)
        ax.set_title(f'{efficacy_col} 分布对比')
        ax.set_ylabel(efficacy_col)
        st.pyplot(fig)

def safety_analysis(df):
    """安全性分析"""
    st.subheader("🛡️ 安全性分析")
    
    # 查找不良事件相关列
    ae_cols = [col for col in df.columns if any(keyword in col.lower()
               for keyword in ['adverse', 'ae', 'event', '不良', '副作用'])]
    
    if len(ae_cols) == 0:
        st.warning("未找到不良事件相关变量")
        return
    
    ae_col = st.selectbox("选择不良事件变量", ae_cols)
    
    # 不良事件统计
    ae_counts = df[ae_col].value_counts()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**不良事件发生情况**")
        st.dataframe(ae_counts.to_frame('频数'))
        
        # 计算发生率
        total_patients = len(df)
        ae_rates = (ae_counts / total_patients * 100).round(2)
        st.write("**不良事件发生率 (%)**")
        st.dataframe(ae_rates.to_frame('发生率(%)'))
    
    with col2:
        # 饼图
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(ae_counts.values, labels=ae_counts.index, autopct='%1.1f%%', startangle=90)
        ax.set_title('不良事件分布')
        st.pyplot(fig)
    
    # 按治疗组分析安全性
    group_cols = [col for col in df.columns if 'group' in col.lower() or 'treatment' in col.lower()]
    
    if len(group_cols) > 0:
        group_col = st.selectbox("选择治疗组变量进行安全性对比", group_cols)
        
        # 交叉表分析
        crosstab = pd.crosstab(df[group_col], df[ae_col], margins=True)
        
        st.write("**各组不良事件对比**")
        st.dataframe(crosstab)
        
        # 卡方检验
        chi2, p_value, dof, expected = stats.chi2_contingency(crosstab.iloc[:-1, :-1])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("卡方统计量", f"{chi2:.4f}")
        with col2:
            st.metric("p值", f"{p_value:.4f}")
        with col3:
            significance = "显著" if p_value < 0.05 else "不显著"
            st.metric("组间差异", significance)

def subgroup_analysis(df):
    """亚组分析"""
    st.subheader("👥 亚组分析")
    
    # 选择亚组变量
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) == 0:
        st.warning("未找到分类变量进行亚组分析")
        return
    
    subgroup_col = st.selectbox("选择亚组变量", categorical_cols)
    
    # 选择分析指标
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.warning("未找到数值型分析指标")
        return
    
    analysis_col = st.selectbox("选择分析指标", numeric_cols)
    
    # 亚组统计
    subgroup_stats = df.groupby(subgroup_col)[analysis_col].agg([
        'count', 'mean', 'std', 'median', 'min', 'max'
    ]).round(2)
    
    st.write(f"**{analysis_col} 按 {subgroup_col} 亚组分析**")
    st.dataframe(subgroup_stats)
    
    # 可视化
    col1, col2 = st.columns(2)
    
    with col1:
        # 条形图
        fig, ax = plt.subplots(figsize=(8, 6))
        subgroup_means = df.groupby(subgroup_col)[analysis_col].mean()
        ax.bar(range(len(subgroup_means)), subgroup_means.values, color='lightcoral')
        ax.set_xticks(range(len(subgroup_means)))
        ax.set_xticklabels(subgroup_means.index, rotation=45)
        ax.set_title(f'{analysis_col} 各亚组均值对比')
        ax.set_ylabel(f'{analysis_col} 均值')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        # 箱线图
        fig, ax = plt.subplots(figsize=(8, 6))
        df.boxplot(column=analysis_col, by=subgroup_col, ax=ax)
        ax.set_title(f'{analysis_col} 亚组分布对比')
        plt.suptitle('')
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    
    # 方差分析
    subgroups = [df[df[subgroup_col] == group][analysis_col].dropna()
                for group in df[subgroup_col].unique()]
    
    if len(subgroups) > 2:
        f_stat, p_value = stats.f_oneway(*subgroups)
        
        st.write("**方差分析结果**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("F统计量", f"{f_stat:.4f}")
        with col2:
            st.metric("p值", f"{p_value:.4f}")
        with col3:
            significance = "显著" if p_value < 0.05 else "不显著"
            st.metric("组间差异", significance)

def epidemiology_analysis():
    """流行病学分析模块"""
    if EPIDEMIOLOGY_AVAILABLE:
        epidemiology_module()
    else:
        st.header("📈 流行病学分析")
        st.info("流行病学分析模块正在开发中...")

def survival_analysis():
    """生存分析模块"""
    if SURVIVAL_AVAILABLE:
        survival_module()
    else:
        st.header("⏱️ 生存分析")
        st.info("生存分析模块正在开发中...")

def main():
    """主函数"""
    st.title("🏥 临床试验数据分析平台")
    st.markdown("---")
    
    # 侧边栏导航
    st.sidebar.title("📋 分析模块")
    page = st.sidebar.radio(
        "选择分析模块",
        ["临床试验分析", "流行病学分析", "生存分析"]
    )
    
    # 根据选择显示相应模块
    if page == "临床试验分析":
        clinical_trial_analysis()
    elif page == "流行病学分析":
        epidemiology_analysis()
    elif page == "生存分析":
        survival_analysis()
    
    # 页脚
    st.markdown("---")
    st.markdown("💡 **使用说明**: 请上传您的数据文件，或使用示例数据进行分析演示")

if __name__ == "__main__":
    main()
