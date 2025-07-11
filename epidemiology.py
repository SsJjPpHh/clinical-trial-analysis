
"""
流行病学分析模块 (epidemiology.py)
提供专门的流行病学分析功能，包括队列研究、病例对照研究、横断面研究等
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, ttest_ind
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def epidemiology_analysis():
    """流行病学分析主函数"""
    st.markdown("# 🦠 流行病学分析模块")
    st.markdown("*专业的流行病学研究分析工具*")
    
    # 侧边栏 - 分析类型选择
    with st.sidebar:
        st.markdown("### 📋 分析类型")
        analysis_type = st.selectbox(
            "选择分析类型",
            [
                "📊 描述性流行病学",
                "🔬 队列研究分析", 
                "🎯 病例对照研究",
                "📈 横断面研究",
                "🌍 疾病监测分析",
                "📉 流行趋势分析",
                "🗺️ 空间流行病学",
                "⚡ 疫情暴发调查",
                "🧬 分子流行病学",
                "📊 筛查试验评价"
            ]
        )
    
    # 数据上传
    uploaded_file = st.file_uploader(
        "📁 上传流行病学数据",
        type=['csv', 'xlsx', 'xls'],
        help="支持CSV和Excel格式的流行病学数据"
    )
    
    if uploaded_file is not None:
        try:
            # 读取数据
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ 数据加载成功！共 {len(df)} 行，{len(df.columns)} 列")
            
            # 数据预览
            with st.expander("👀 数据预览", expanded=False):
                st.dataframe(df.head(10))
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("总样本量", len(df))
                with col2:
                    st.metric("变量数", len(df.columns))
                with col3:
                    missing_rate = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                    st.metric("缺失率", f"{missing_rate:.1f}%")
            
            # 根据选择的分析类型调用相应函数
            if analysis_type == "📊 描述性流行病学":
                descriptive_epidemiology(df)
            elif analysis_type == "🔬 队列研究分析":
                cohort_study_analysis(df)
            elif analysis_type == "🎯 病例对照研究":
                case_control_analysis(df)
            elif analysis_type == "📈 横断面研究":
                cross_sectional_analysis(df)
            elif analysis_type == "🌍 疾病监测分析":
                disease_surveillance(df)
            elif analysis_type == "📉 流行趋势分析":
                trend_analysis(df)
            elif analysis_type == "🗺️ 空间流行病学":
                spatial_epidemiology(df)
            elif analysis_type == "⚡ 疫情暴发调查":
                outbreak_investigation(df)
            elif analysis_type == "🧬 分子流行病学":
                molecular_epidemiology(df)
            elif analysis_type == "📊 筛查试验评价":
                screening_test_evaluation(df)
                
        except Exception as e:
            st.error(f"❌ 数据读取失败: {str(e)}")
    
    else:
        # 显示示例数据格式
        show_example_data_formats()

def show_example_data_formats():
    """显示示例数据格式"""
    st.markdown("### 📋 数据格式要求")
    
    tab1, tab2, tab3, tab4 = st.tabs(["队列研究", "病例对照", "横断面研究", "疾病监测"])
    
    with tab1:
        st.markdown("#### 队列研究数据格式示例")
        cohort_example = pd.DataFrame({
            '受试者ID': ['P001', 'P002', 'P003', 'P004', 'P005'],
            '年龄': [45, 52, 38, 61, 29],
            '性别': ['男', '女', '男', '女', '男'],
            '暴露状态': ['暴露', '未暴露', '暴露', '未暴露', '暴露'],
            '随访时间(年)': [5.2, 4.8, 6.1, 3.9, 5.5],
            '结局发生': ['是', '否', '是', '否', '否'],
            '结局时间(年)': [3.1, 4.8, 2.8, 3.9, 5.5]
        })
        st.dataframe(cohort_example)
    
    with tab2:
        st.markdown("#### 病例对照研究数据格式示例")
        case_control_example = pd.DataFrame({
            '受试者ID': ['C001', 'C002', 'C003', 'C004', 'C005'],
            '年龄': [55, 48, 62, 39, 51],
            '性别': ['女', '男', '女', '男', '女'],
            '病例对照': ['病例', '对照', '病例', '对照', '病例'],
            '暴露史': ['有', '无', '有', '无', '有'],
            '吸烟史': ['是', '否', '是', '否', '是'],
            '家族史': ['有', '无', '有', '无', '无']
        })
        st.dataframe(case_control_example)
    
    with tab3:
        st.markdown("#### 横断面研究数据格式示例")
        cross_sectional_example = pd.DataFrame({
            '受试者ID': ['S001', 'S002', 'S003', 'S004', 'S005'],
            '年龄': [34, 45, 28, 56, 41],
            '性别': ['男', '女', '男', '女', '男'],
            '地区': ['城市', '农村', '城市', '农村', '城市'],
            '疾病状态': ['患病', '未患病', '未患病', '患病', '未患病'],
            '危险因素1': ['有', '无', '有', '有', '无'],
            '危险因素2': ['高', '低', '中', '高', '低']
        })
        st.dataframe(cross_sectional_example)
    
    with tab4:
        st.markdown("#### 疾病监测数据格式示例")
        surveillance_example = pd.DataFrame({
            '日期': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05'],
            '地区': ['北京', '上海', '广州', '深圳', '杭州'],
            '病例数': [12, 8, 15, 6, 9],
            '死亡数': [1, 0, 2, 0, 1],
            '人口数': [2000000, 2500000, 1800000, 1300000, 1200000],
            '年龄组': ['全年龄', '全年龄', '全年龄', '全年龄', '全年龄']
        })
        st.dataframe(surveillance_example)

def descriptive_epidemiology(df):
    """描述性流行病学分析"""
    st.markdown("### 📊 描述性流行病学分析")
    st.markdown("*分析疾病的人群、时间、地区分布特征*")
    
    # 变量选择
    col1, col2 = st.columns(2)
    
    with col1:
        # 疾病/结局变量
        outcome_vars = df.columns.tolist()
        outcome_var = st.selectbox("选择疾病/结局变量", outcome_vars)
    
    with col2:
        # 分析维度
        analysis_dimension = st.selectbox(
            "选择分析维度",
            ["人群特征分析", "时间分布分析", "地区分布分析", "综合分析"]
        )
    
    if outcome_var:
        if analysis_dimension == "人群特征分析":
            person_analysis(df, outcome_var)
        elif analysis_dimension == "时间分布分析":
            time_analysis(df, outcome_var)
        elif analysis_dimension == "地区分布分析":
            place_analysis(df, outcome_var)
        elif analysis_dimension == "综合分析":
            comprehensive_descriptive_analysis(df, outcome_var)

def person_analysis(df, outcome_var):
    """人群特征分析"""
    st.markdown("#### 👥 人群特征分析")
    
    # 识别人群特征变量
    person_vars = identify_person_variables(df)
    
    if not person_vars:
        st.warning("⚠️ 未识别到人群特征变量（如年龄、性别等）")
        return
    
    # 选择人群变量
    selected_person_vars = st.multiselect(
        "选择人群特征变量",
        person_vars,
        default=person_vars[:3] if len(person_vars) >= 3 else person_vars
    )
    
    if not selected_person_vars:
        return
    
    # 分析每个人群特征
    for person_var in selected_person_vars:
        st.markdown(f"##### 📊 按{person_var}分布")
        
        # 创建交叉表
        if df[outcome_var].dtype in ['object', 'category'] or df[outcome_var].nunique() <= 10:
            # 分类结局变量
            crosstab = pd.crosstab(df[person_var], df[outcome_var], margins=True)
            
            # 计算率
            if '患病' in df[outcome_var].values or '是' in df[outcome_var].values:
                disease_col = '患病' if '患病' in df[outcome_var].values else '是'
                if disease_col in crosstab.columns:
                    crosstab['患病率(%)'] = (crosstab[disease_col] / crosstab['All'] * 100).round(2)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(crosstab)
            
            with col2:
                # 可视化
                if '患病率(%)' in crosstab.columns:
                    # 排除总计行
                    plot_data = crosstab[crosstab.index != 'All']
                    
                    fig = px.bar(
                        x=plot_data.index,
                        y=plot_data['患病率(%)'],
                        title=f"{outcome_var}按{person_var}的患病率分布",
                        labels={'x': person_var, 'y': '患病率(%)'}
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            # 连续结局变量
            summary_stats = df.groupby(person_var)[outcome_var].agg([
                'count', 'mean', 'std', 'median', 'min', 'max'
            ]).round(3)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(summary_stats)
            
            with col2:
                # 箱线图
                fig = px.box(
                    df, x=person_var, y=outcome_var,
                    title=f"{outcome_var}按{person_var}的分布"
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # 统计检验
        perform_person_analysis_test(df, person_var, outcome_var)

def perform_person_analysis_test(df, person_var, outcome_var):
    """执行人群特征分析的统计检验"""
    st.markdown("**统计检验结果:**")
    
    try:
        if df[outcome_var].dtype in ['object', 'category'] or df[outcome_var].nunique() <= 10:
            # 分类结局变量 - 卡方检验
            crosstab = pd.crosstab(df[person_var], df[outcome_var])
            
            if crosstab.shape[0] >= 2 and crosstab.shape[1] >= 2:
                if crosstab.shape == (2, 2) and crosstab.min().min() < 5:
                    # Fisher精确检验
                    _, p_value = fisher_exact(crosstab)
                    test_method = "Fisher精确检验"
                else:
                    # 卡方检验
                    chi2, p_value, _, _ = chi2_contingency(crosstab)
                    test_method = "卡方检验"
                
                st.write(f"• 检验方法: {test_method}")
                st.write(f"• P值: {p_value:.4f}")
                
                if p_value < 0.05:
                    st.success("✅ 不同人群特征间存在显著差异")
                else:
                    st.info("ℹ️ 不同人群特征间无显著差异")
        
        else:
            # 连续结局变量
            groups = df[person_var].unique()
            
            if len(groups) == 2:
                # 两组比较 - t检验
                group1_data = df[df[person_var] == groups[0]][outcome_var].dropna()
                group2_data = df[df[person_var] == groups[1]][outcome_var].dropna()
                
                t_stat, p_value = ttest_ind(group1_data, group2_data)
                st.write(f"• 检验方法: 独立样本t检验")
                st.write(f"• t统计量: {t_stat:.4f}")
                st.write(f"• P值: {p_value:.4f}")
                
                if p_value < 0.05:
                    st.success("✅ 两组间存在显著差异")
                else:
                    st.info("ℹ️ 两组间无显著差异")
            
            else:
                # 多组比较 - 方差分析
                group_data = [df[df[person_var] == group][outcome_var].dropna() for group in groups]
                f_stat, p_value = stats.f_oneway(*group_data)
                
                st.write(f"• 检验方法: 单因素方差分析")
                st.write(f"• F统计量: {f_stat:.4f}")
                st.write(f"• P值: {p_value:.4f}")
                
                if p_value < 0.05:
                    st.success("✅ 各组间存在显著差异")
                else:
                    st.info("ℹ️ 各组间无显著差异")
    
    except Exception as e:
        st.warning(f"⚠️ 统计检验失败: {str(e)}")

def identify_person_variables(df):
    """识别人群特征变量"""
    person_keywords = [
        '年龄', '性别', '职业', '教育', '收入', '婚姻', '民族', 
        'age', 'sex', 'gender', 'occupation', 'education', 'income', 'marital'
    ]
    
    person_vars = []
    
    for col in df.columns:
        # 检查列名是否包含人群特征关键词
        if any(keyword in col.lower() for keyword in person_keywords):
            person_vars.append(col)
        
        # 检查是否为典型的分类变量
        elif df[col].dtype in ['object', 'category'] and df[col].nunique() <= 20:
            person_vars.append(col)
    
    return person_vars

def time_analysis(df, outcome_var):
    """时间分布分析"""
    st.markdown("#### ⏰ 时间分布分析")
    
    # 识别时间变量
    time_vars = identify_time_variables(df)
    
    if not time_vars:
        st.warning("⚠️ 未识别到时间变量")
        return
    
    # 选择时间变量
    time_var = st.selectbox("选择时间变量", time_vars)
    
    if not time_var:
        return
    
    # 时间分析类型
    time_analysis_type = st.selectbox(
        "选择时间分析类型",
        ["时间趋势分析", "季节性分析", "周期性分析", "时间聚集性分析"]
    )
    
    if time_analysis_type == "时间趋势分析":
        temporal_trend_analysis(df, time_var, outcome_var)
    elif time_analysis_type == "季节性分析":
        seasonal_analysis(df, time_var, outcome_var)
    elif time_analysis_type == "周期性分析":
        cyclical_analysis(df, time_var, outcome_var)
    elif time_analysis_type == "时间聚集性分析":
        temporal_clustering_analysis(df, time_var, outcome_var)

def temporal_trend_analysis(df, time_var, outcome_var):
    """时间趋势分析"""
    st.markdown("##### 📈 时间趋势分析")
    
    try:
        # 确保时间变量为日期格式
        if df[time_var].dtype != 'datetime64[ns]':
            df[time_var] = pd.to_datetime(df[time_var], errors='coerce')
        
        # 按时间聚合数据
        time_grouping = st.selectbox(
            "选择时间聚合方式",
            ["日", "周", "月", "季度", "年"]
        )
        
        # 创建时间分组
        if time_grouping == "日":
            df['时间组'] = df[time_var].dt.date
        elif time_grouping == "周":
            df['时间组'] = df[time_var].dt.to_period('W')
        elif time_grouping == "月":
            df['时间组'] = df[time_var].dt.to_period('M')
        elif time_grouping == "季度":
            df['时间组'] = df[time_var].dt.to_period('Q')
        elif time_grouping == "年":
            df['时间组'] = df[time_var].dt.to_period('Y')
        
        # 计算时间趋势
        if df[outcome_var].dtype in ['object', 'category'] or df[outcome_var].nunique() <= 10:
            # 分类结局变量 - 计算发病率
            if '患病' in df[outcome_var].values or '是' in df[outcome_var].values:
                disease_value = '患病' if '患病' in df[outcome_var].values else '是'
                
                time_trend = df.groupby('时间组').agg({
                    outcome_var: ['count', lambda x: (x == disease_value).sum()]
                }).round(3)
                
                time_trend.columns = ['总数', '病例数']
                time_trend['发病率(%)'] = (time_trend['病例数'] / time_trend['总数'] * 100).round(2)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.dataframe(time_trend)
                
                with col2:
                    # 趋势图
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=time_trend.index.astype(str),
                        y=time_trend['发病率(%)'],
                        mode='lines+markers',
                        name='发病率',
                        line=dict(color='red', width=2),
                        marker=dict(size=6)
                    ))
                    
                    fig.update_layout(
                        title=f"{outcome_var}时间趋势图",
                        xaxis_title="时间",
                        yaxis_title="发病率(%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            # 连续结局变量
            time_trend = df.groupby('时间组')[outcome_var].agg([
                'count', 'mean', 'std', 'median'
            ]).round(3)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(time_trend)
            
            with col2:
                # 趋势图
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=time_trend.index.astype(str),
                    y=time_trend['mean'],
                    mode='lines+markers',
                    name='均值',
                    line=dict(color='blue', width=2),
                    marker=dict(size=6)
                ))
                
                fig.update_layout(
                    title=f"{outcome_var}时间趋势图",
                    xaxis_title="时间",
                    yaxis_title=outcome_var,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # 趋势检验
        perform_trend_test(time_trend, outcome_var)
        
    except Exception as e:
        st.error(f"❌ 时间趋势分析失败: {str(e)}")

def perform_trend_test(time_trend, outcome_var):
    """执行趋势检验"""
    st.markdown("**趋势检验结果:**")
    
    try:
        # Mann-Kendall趋势检验
        if '发病率(%)' in time_trend.columns:
            data_series = time_trend['发病率(%)'].dropna()
        elif 'mean' in time_trend.columns:
            data_series = time_trend['mean'].dropna()
        else:
            return
        
        if len(data_series) < 3:
            st.warning("⚠️ 数据点太少，无法进行趋势检验")
            return
        
        # 简化的Mann-Kendall检验
        n = len(data_series)
        s = 0
        
        for i in range(n-1):
            for j in range(i+1, n):
                if data_series.iloc[j] > data_series.iloc[i]:
                    s += 1
                elif data_series.iloc[j] < data_series.iloc[i]:
                    s -= 1
        
        # 计算方差
        var_s = n * (n - 1) * (2 * n + 5) / 18
        
        # 计算Z统计量
        if s > 0:
            z = (s - 1) / np.sqrt(var_s)
        elif s < 0:
            z = (s + 1) / np.sqrt(var_s)
        else:
            z = 0
        
        # 计算p值
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        st.write(f"• 检验方法: Mann-Kendall趋势检验")
        st.write(f"• S统计量: {s}")
        st.write(f"• Z统计量: {z:.4f}")
        st.write(f"• P值: {p_value:.4f}")
        
        if p_value < 0.05:
            if s > 0:
                st.success("✅ 存在显著的上升趋势")
            else:
                st.warning("⚠️ 存在显著的下降趋势")
        else:
            st.info("ℹ️ 无显著的时间趋势")
    
    except Exception as e:
        st.warning(f"⚠️ 趋势检验失败: {str(e)}")

def identify_time_variables(df):
    """识别时间变量"""
    time_vars = []
    
    for col in df.columns:
        # 检查列名
        time_keywords = ['时间', '日期', '年', '月', '日', 'date', 'time', 'year', 'month', 'day']
        if any(keyword in col.lower() for keyword in time_keywords):
            time_vars.append(col)
        
        # 检查数据类型
        elif df[col].dtype in ['datetime64[ns]', 'timedelta64[ns]']:
            time_vars.append(col)
        
        # 检查是否可以转换为日期
        elif df[col].dtype == 'object':
            try:
                pd.to_datetime(df[col].head(), errors='raise')
                time_vars.append(col)
            except:
                pass
    
    return time_vars

def cohort_study_analysis(df):
    """队列研究分析"""
    st.markdown("### 🔬 队列研究分析")
    st.markdown("*前瞻性队列研究的风险评估和因果推断*")
    
    # 变量选择
    col1, col2, col3 = st.columns(3)
    
    with col1:
        exposure_var = st.selectbox("选择暴露变量", df.columns.tolist())
    
    with col2:
        outcome_var = st.selectbox("选择结局变量", df.columns.tolist())
    
    with col3:
        time_var = st.selectbox("选择随访时间变量", df.columns.tolist(), help="随访时间或生存时间")
    
    if not all([exposure_var, outcome_var, time_var]):
        return
    
    # 分析类型选择
    analysis_type = st.selectbox(
        "选择分析类型",
        ["风险比分析", "发病密度分析", "归因风险分析", "生存分析", "多因素分析"]
    )
    
    if analysis_type == "风险比分析":
        risk_ratio_analysis(df, exposure_var, outcome_var, time_var)
    elif analysis_type == "发病密度分析":
        incidence_density_analysis(df, exposure_var, outcome_var, time_var)
    elif analysis_type == "归因风险分析":
        attributable_risk_analysis(df, exposure_var, outcome_var, time_var)
    elif analysis_type == "生存分析":
        cohort_survival_analysis(df, exposure_var, outcome_var, time_var)
    elif analysis_type == "多因素分析":
        multivariable_cohort_analysis(df, exposure_var, outcome_var, time_var)

def risk_ratio_analysis(df, exposure_var, outcome_var, time_var):
    """风险比分析"""
    st.markdown("#### 📊 风险比(RR)分析")
    
    try:
        # 创建2x2表
        crosstab = pd.crosstab(df[exposure_var], df[outcome_var], margins=True)
        
        st.markdown("##### 📋 2×2列联表")
        st.dataframe(crosstab)
        
        # 计算风险比
        if crosstab.shape == (3, 3):  # 包含margins的3x3表
            # 提取2x2核心数据
            a = crosstab.iloc[0, 0]  # 暴露+结局+
            b = crosstab.iloc[0, 1]  # 暴露+结局-
            c = crosstab.iloc[1, 0]  # 暴露-结局+
            d = crosstab.iloc[1, 1]  # 暴露-结局-
            
            # 计算风险
            risk_exposed = a / (a + b)
            risk_unexposed = c / (c + d)
            
            # 计算风险比
            if risk_unexposed > 0:
                rr = risk_exposed / risk_unexposed
            else:
                rr = float('inf')
            
            # 计算95%置信区间
            if a > 0 and c > 0:
                                log_rr = np.log(rr)
                se_log_rr = np.sqrt(1/a - 1/(a+b) + 1/c - 1/(c+d))
                ci_lower = np.exp(log_rr - 1.96 * se_log_rr)
                ci_upper = np.exp(log_rr + 1.96 * se_log_rr)
            else:
                ci_lower, ci_upper = np.nan, np.nan
            
            # 显示结果
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 📊 风险估计")
                results_df = pd.DataFrame({
                    '指标': ['暴露组风险', '非暴露组风险', '风险比(RR)', '95%CI下限', '95%CI上限'],
                    '数值': [
                        f"{risk_exposed:.4f} ({risk_exposed*100:.2f}%)",
                        f"{risk_unexposed:.4f} ({risk_unexposed*100:.2f}%)",
                        f"{rr:.4f}",
                        f"{ci_lower:.4f}" if not np.isnan(ci_lower) else "N/A",
                        f"{ci_upper:.4f}" if not np.isnan(ci_upper) else "N/A"
                    ]
                })
                st.dataframe(results_df, hide_index=True)
            
            with col2:
                # 风险比可视化
                fig = go.Figure()
                
                # 添加风险比点估计
                fig.add_trace(go.Scatter(
                    x=[rr], y=['风险比'],
                    mode='markers',
                    marker=dict(size=12, color='red'),
                    name='点估计'
                ))
                
                # 添加置信区间
                if not np.isnan(ci_lower) and not np.isnan(ci_upper):
                    fig.add_trace(go.Scatter(
                        x=[ci_lower, ci_upper], y=['风险比', '风险比'],
                        mode='lines',
                        line=dict(color='red', width=3),
                        name='95%CI'
                    ))
                
                # 添加无效线
                fig.add_vline(x=1, line_dash="dash", line_color="gray")
                
                fig.update_layout(
                    title="风险比及其95%置信区间",
                    xaxis_title="风险比",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # 统计检验
            perform_rr_test(a, b, c, d)
            
            # 结果解释
            interpret_risk_ratio(rr, ci_lower, ci_upper)
        
        else:
            st.warning("⚠️ 数据格式不适合2×2分析，请检查暴露和结局变量")
    
    except Exception as e:
        st.error(f"❌ 风险比分析失败: {str(e)}")

def perform_rr_test(a, b, c, d):
    """执行风险比的统计检验"""
    st.markdown("##### 🧮 统计检验")
    
    try:
        # 卡方检验
        observed = np.array([[a, b], [c, d]])
        chi2, p_value, _, _ = chi2_contingency(observed)
        
        # Fisher精确检验（如果样本量小）
        if min(a, b, c, d) < 5:
            _, fisher_p = fisher_exact(observed)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**卡方检验:**")
                st.write(f"• χ² = {chi2:.4f}")
                st.write(f"• P值 = {p_value:.4f}")
            
            with col2:
                st.write("**Fisher精确检验:**")
                st.write(f"• P值 = {fisher_p:.4f}")
                
                if fisher_p < 0.05:
                    st.success("✅ 差异具有统计学意义")
                else:
                    st.info("ℹ️ 差异无统计学意义")
        else:
            st.write("**卡方检验:**")
            st.write(f"• χ² = {chi2:.4f}")
            st.write(f"• P值 = {p_value:.4f}")
            
            if p_value < 0.05:
                st.success("✅ 差异具有统计学意义")
            else:
                st.info("ℹ️ 差异无统计学意义")
    
    except Exception as e:
        st.warning(f"⚠️ 统计检验失败: {str(e)}")

def interpret_risk_ratio(rr, ci_lower, ci_upper):
    """解释风险比结果"""
    st.markdown("##### 💡 结果解释")
    
    if np.isnan(rr):
        st.warning("⚠️ 无法计算风险比")
        return
    
    # 基本解释
    if rr > 1:
        if not np.isnan(ci_lower) and ci_lower > 1:
            st.success(f"✅ 暴露是危险因素，暴露者的风险是非暴露者的 {rr:.2f} 倍")
        else:
            st.info(f"ℹ️ 暴露可能是危险因素，但置信区间包含1，需谨慎解释")
    elif rr < 1:
        if not np.isnan(ci_upper) and ci_upper < 1:
            st.success(f"✅ 暴露是保护因素，可降低 {(1-rr)*100:.1f}% 的风险")
        else:
            st.info(f"ℹ️ 暴露可能是保护因素，但置信区间包含1，需谨慎解释")
    else:
        st.info("ℹ️ 暴露与结局无关联")
    
    # 临床意义评估
    if rr > 2 or rr < 0.5:
        st.warning("⚠️ 关联强度较强，具有重要的临床或公共卫生意义")
    elif rr > 1.5 or rr < 0.67:
        st.info("ℹ️ 关联强度中等，需结合其他证据评估")
    else:
        st.info("ℹ️ 关联强度较弱，临床意义有限")

def incidence_density_analysis(df, exposure_var, outcome_var, time_var):
    """发病密度分析"""
    st.markdown("#### 📈 发病密度分析")
    
    try:
        # 按暴露状态分组计算发病密度
        exposure_groups = df[exposure_var].unique()
        
        density_results = []
        
        for group in exposure_groups:
            group_data = df[df[exposure_var] == group]
            
            # 计算病例数
            if df[outcome_var].dtype in ['object', 'category']:
                cases = len(group_data[group_data[outcome_var].isin(['是', '患病', 'Yes', '1', 1])])
            else:
                cases = len(group_data[group_data[outcome_var] == 1])
            
            # 计算人时
            person_time = group_data[time_var].sum()
            
            # 计算发病密度
            if person_time > 0:
                incidence_density = cases / person_time
                # 转换为每1000人年
                incidence_density_1000 = incidence_density * 1000
            else:
                incidence_density = 0
                incidence_density_1000 = 0
            
            density_results.append({
                '暴露状态': group,
                '病例数': cases,
                '人时': person_time,
                '发病密度': incidence_density,
                '发病密度(‰人年)': incidence_density_1000
            })
        
        # 显示结果
        density_df = pd.DataFrame(density_results)
        st.dataframe(density_df.round(4))
        
        # 计算发病密度比(IDR)
        if len(density_results) == 2:
            idr = density_results[0]['发病密度'] / density_results[1]['发病密度'] if density_results[1]['发病密度'] > 0 else float('inf')
            
            # 计算IDR的置信区间
            cases1, cases2 = density_results[0]['病例数'], density_results[1]['病例数']
            pt1, pt2 = density_results[0]['人时'], density_results[1]['人时']
            
            if cases1 > 0 and cases2 > 0:
                log_idr = np.log(idr)
                se_log_idr = np.sqrt(1/cases1 + 1/cases2)
                ci_lower = np.exp(log_idr - 1.96 * se_log_idr)
                ci_upper = np.exp(log_idr + 1.96 * se_log_idr)
                
                st.markdown("##### 📊 发病密度比(IDR)")
                idr_results = pd.DataFrame({
                    '指标': ['发病密度比(IDR)', '95%CI下限', '95%CI上限'],
                    '数值': [f"{idr:.4f}", f"{ci_lower:.4f}", f"{ci_upper:.4f}"]
                })
                st.dataframe(idr_results, hide_index=True)
                
                # IDR可视化
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=[idr], y=['发病密度比'],
                    mode='markers',
                    marker=dict(size=12, color='blue'),
                    name='点估计'
                ))
                
                fig.add_trace(go.Scatter(
                    x=[ci_lower, ci_upper], y=['发病密度比', '发病密度比'],
                    mode='lines',
                    line=dict(color='blue', width=3),
                    name='95%CI'
                ))
                
                fig.add_vline(x=1, line_dash="dash", line_color="gray")
                
                fig.update_layout(
                    title="发病密度比及其95%置信区间",
                    xaxis_title="发病密度比",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 解释结果
                interpret_incidence_density_ratio(idr, ci_lower, ci_upper)
        
        # 发病密度的可视化比较
        fig = px.bar(
            density_df, x='暴露状态', y='发病密度(‰人年)',
            title="不同暴露状态的发病密度比较",
            color='暴露状态'
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"❌ 发病密度分析失败: {str(e)}")

def interpret_incidence_density_ratio(idr, ci_lower, ci_upper):
    """解释发病密度比结果"""
    st.markdown("##### 💡 结果解释")
    
    if idr > 1:
        if ci_lower > 1:
            st.success(f"✅ 暴露组的发病密度是非暴露组的 {idr:.2f} 倍")
        else:
            st.info("ℹ️ 暴露组发病密度可能更高，但置信区间包含1")
    elif idr < 1:
        if ci_upper < 1:
            st.success(f"✅ 暴露可降低发病密度 {(1-idr)*100:.1f}%")
        else:
            st.info("ℹ️ 暴露可能降低发病密度，但置信区间包含1")
    else:
        st.info("ℹ️ 两组发病密度相似")

def case_control_analysis(df):
    """病例对照研究分析"""
    st.markdown("### 🎯 病例对照研究分析")
    st.markdown("*回顾性病例对照研究的比值比分析*")
    
    # 变量选择
    col1, col2 = st.columns(2)
    
    with col1:
        case_control_var = st.selectbox("选择病例对照变量", df.columns.tolist())
    
    with col2:
        exposure_var = st.selectbox("选择暴露变量", df.columns.tolist())
    
    if not all([case_control_var, exposure_var]):
        return
    
    # 分析类型
    analysis_type = st.selectbox(
        "选择分析类型",
        ["比值比分析", "匹配分析", "分层分析", "条件Logistic回归", "多因素分析"]
    )
    
    if analysis_type == "比值比分析":
        odds_ratio_analysis(df, case_control_var, exposure_var)
    elif analysis_type == "匹配分析":
        matched_analysis(df, case_control_var, exposure_var)
    elif analysis_type == "分层分析":
        stratified_analysis(df, case_control_var, exposure_var)
    elif analysis_type == "条件Logistic回归":
        conditional_logistic_analysis(df, case_control_var, exposure_var)
    elif analysis_type == "多因素分析":
        multivariable_case_control_analysis(df, case_control_var, exposure_var)

def odds_ratio_analysis(df, case_control_var, exposure_var):
    """比值比分析"""
    st.markdown("#### 📊 比值比(OR)分析")
    
    try:
        # 创建2x2表
        crosstab = pd.crosstab(df[exposure_var], df[case_control_var], margins=True)
        
        st.markdown("##### 📋 2×2列联表")
        st.dataframe(crosstab)
        
        # 计算比值比
        if crosstab.shape == (3, 3):
            # 提取2x2核心数据
            a = crosstab.iloc[0, 0]  # 暴露+病例+
            b = crosstab.iloc[0, 1]  # 暴露+对照+
            c = crosstab.iloc[1, 0]  # 暴露-病例+
            d = crosstab.iloc[1, 1]  # 暴露-对照+
            
            # 计算比值比
            if b > 0 and c > 0:
                or_value = (a * d) / (b * c)
            else:
                or_value = float('inf')
            
            # 计算95%置信区间
            if all(x > 0 for x in [a, b, c, d]):
                log_or = np.log(or_value)
                se_log_or = np.sqrt(1/a + 1/b + 1/c + 1/d)
                ci_lower = np.exp(log_or - 1.96 * se_log_or)
                ci_upper = np.exp(log_or + 1.96 * se_log_or)
            else:
                ci_lower, ci_upper = np.nan, np.nan
            
            # 显示结果
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 📊 比值比估计")
                results_df = pd.DataFrame({
                    '指标': ['比值比(OR)', '95%CI下限', '95%CI上限'],
                    '数值': [
                        f"{or_value:.4f}",
                        f"{ci_lower:.4f}" if not np.isnan(ci_lower) else "N/A",
                        f"{ci_upper:.4f}" if not np.isnan(ci_upper) else "N/A"
                    ]
                })
                st.dataframe(results_df, hide_index=True)
                
                # 计算暴露比例
                exposed_cases = a / (a + c) * 100
                exposed_controls = b / (b + d) * 100
                
                st.write(f"• 病例中暴露比例: {exposed_cases:.1f}%")
                st.write(f"• 对照中暴露比例: {exposed_controls:.1f}%")
            
            with col2:
                # 比值比可视化
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=[or_value], y=['比值比'],
                    mode='markers',
                    marker=dict(size=12, color='green'),
                    name='点估计'
                ))
                
                if not np.isnan(ci_lower) and not np.isnan(ci_upper):
                    fig.add_trace(go.Scatter(
                        x=[ci_lower, ci_upper], y=['比值比', '比值比'],
                        mode='lines',
                        line=dict(color='green', width=3),
                        name='95%CI'
                    ))
                
                fig.add_vline(x=1, line_dash="dash", line_color="gray")
                
                fig.update_layout(
                    title="比值比及其95%置信区间",
                    xaxis_title="比值比",
                    height=300,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # 统计检验
            perform_or_test(a, b, c, d)
            
            # 结果解释
            interpret_odds_ratio(or_value, ci_lower, ci_upper)
        
        else:
            st.warning("⚠️ 数据格式不适合2×2分析")
    
    except Exception as e:
        st.error(f"❌ 比值比分析失败: {str(e)}")

def perform_or_test(a, b, c, d):
    """执行比值比的统计检验"""
    st.markdown("##### 🧮 统计检验")
    
    try:
        observed = np.array([[a, b], [c, d]])
        
        # 卡方检验
        chi2, p_chi2, _, _ = chi2_contingency(observed)
        
        # Fisher精确检验
        _, p_fisher = fisher_exact(observed)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**卡方检验:**")
            st.write(f"• χ² = {chi2:.4f}")
            st.write(f"• P值 = {p_chi2:.4f}")
        
        with col2:
            st.write("**Fisher精确检验:**")
            st.write(f"• P值 = {p_fisher:.4f}")
        
        # 选择合适的检验结果
        if min(a, b, c, d) < 5:
            p_value = p_fisher
            test_name = "Fisher精确检验"
        else:
            p_value = p_chi2
            test_name = "卡方检验"
        
        if p_value < 0.05:
            st.success(f"✅ {test_name}显示关联具有统计学意义")
        else:
            st.info(f"ℹ️ {test_name}显示关联无统计学意义")
    
    except Exception as e:
        st.warning(f"⚠️ 统计检验失败: {str(e)}")

def interpret_odds_ratio(or_value, ci_lower, ci_upper):
    """解释比值比结果"""
    st.markdown("##### 💡 结果解释")
    
    if np.isnan(or_value):
        st.warning("⚠️ 无法计算比值比")
        return
    
    if or_value > 1:
        if not np.isnan(ci_lower) and ci_lower > 1:
            st.success(f"✅ 暴露是危险因素，暴露者患病的几率是非暴露者的 {or_value:.2f} 倍")
        else:
            st.info("ℹ️ 暴露可能是危险因素，但置信区间包含1")
    elif or_value < 1:
        if not np.isnan(ci_upper) and ci_upper < 1:
            st.success(f"✅ 暴露是保护因素，可降低患病几率 {(1-or_value)*100:.1f}%")
        else:
            st.info("ℹ️ 暴露可能是保护因素，但置信区间包含1")
    else:
        st.info("ℹ️ 暴露与疾病无关联")
    
    # 关联强度评估
    if or_value > 3 or or_value < 0.33:
        st.warning("⚠️ 关联强度强，具有重要意义")
    elif or_value > 2 or or_value < 0.5:
        st.info("ℹ️ 关联强度中等")
    else:
        st.info("ℹ️ 关联强度弱")

def cross_sectional_analysis(df):
    """横断面研究分析"""
    st.markdown("### 📈 横断面研究分析")
    st.markdown("*横断面研究的患病率和关联性分析*")
    
    # 变量选择
    col1, col2 = st.columns(2)
    
    with col1:
        outcome_var = st.selectbox("选择疾病/结局变量", df.columns.tolist())
    
    with col2:
        exposure_var = st.selectbox("选择暴露/危险因素变量", df.columns.tolist())
    
    if not all([outcome_var, exposure_var]):
        return
    
    # 分析类型
    analysis_type = st.selectbox(
        "选择分析类型",
        ["患病率分析", "患病率比分析", "关联性分析", "多因素分析", "趋势分析"]
    )
    
    if analysis_type == "患病率分析":
        prevalence_analysis(df, outcome_var, exposure_var)
    elif analysis_type == "患病率比分析":
        prevalence_ratio_analysis(df, outcome_var, exposure_var)
    elif analysis_type == "关联性分析":
        cross_sectional_association_analysis(df, outcome_var, exposure_var)
    elif analysis_type == "多因素分析":
        multivariable_cross_sectional_analysis(df, outcome_var, exposure_var)
    elif analysis_type == "趋势分析":
        cross_sectional_trend_analysis(df, outcome_var, exposure_var)

def prevalence_analysis(df, outcome_var, exposure_var):
    """患病率分析"""
    st.markdown("#### 📊 患病率分析")
    
    try:
        # 总体患病率
        if df[outcome_var].dtype in ['object', 'category']:
            disease_cases = len(df[df[outcome_var].isin(['是', '患病', 'Yes', '1', 1])])
        else:
            disease_cases = len(df[df[outcome_var] == 1])
        
        total_subjects = len(df)
        overall_prevalence = disease_cases / total_subjects * 100
        
        st.markdown("##### 📋 总体患病率")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("总人数", total_subjects)
        with col2:
            st.metric("患病人数", disease_cases)
        with col3:
            st.metric("患病率", f"{overall_prevalence:.2f}%")
        
        # 按暴露状态分层的患病率
        st.markdown("##### 📊 分层患病率")
        
        exposure_groups = df[exposure_var].unique()
        prevalence_results = []
        
        for group in exposure_groups:
            group_data = df[df[exposure_var] == group]
            group_total = len(group_data)
            
            if df[outcome_var].dtype in ['object', 'category']:
                group_cases = len(group_data[group_data[outcome_var].isin(['是', '患病', 'Yes', '1', 1])])
            else:
                group_cases = len(group_data[group_data[outcome_var] == 1])
            
            group_prevalence = group_cases / group_total * 100 if group_total > 0 else 0
            
            # 计算95%置信区间
            if group_total > 0 and group_cases > 0:
                p = group_cases / group_total
                se = np.sqrt(p * (1 - p) / group_total)
                ci_lower = max(0, (p - 1.96 * se) * 100)
                ci_upper = min(100, (p + 1.96 * se) * 100)
            else:
                ci_lower, ci_upper = 0, 0
            
            prevalence_results.append({
                '暴露状态': group,
                '总人数': group_total,
                '患病人数': group_cases,
                '患病率(%)': group_prevalence,
                '95%CI下限': ci_lower,
                '95%CI上限': ci_upper
            })
        
        prevalence_df = pd.DataFrame(prevalence_results)
        st.dataframe(prevalence_df.round(2))
        
        # 患病率可视化
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=prevalence_df['暴露状态'],
            y=prevalence_df['患病率(%)'],
            name='患病率',
            error_y=dict(
                type='data',
                symmetric=False,
                array=prevalence_df['95%CI上限'] - prevalence_df['患病率(%)'],
                arrayminus=prevalence_df['患病率(%)'] - prevalence_df['95%CI下限']
            )
        ))
        
        fig.update_layout(
            title=f"{outcome_var}在不同{exposure_var}中的患病率",
            xaxis_title=exposure_var,
            yaxis_title="患病率(%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 统计检验
        if len(exposure_groups) >= 2:
            perform_prevalence_test(df, outcome_var, exposure_var)
    
    except Exception as e:
        st.error(f"❌ 患病率分析失败: {str(e)}")

def perform_prevalence_test(df, outcome_var, exposure_var):
    """执行患病率差异检验"""
    st.markdown("##### 🧮 患病率差异检验")
    
    try:
        # 创建列联表
        crosstab = pd.crosstab(df[exposure_var], df[outcome_var])
        
        # 卡方检验
        chi2, p_value, _, _ = chi2_contingency(crosstab)
        
        st.write(f"• 检验方法: 卡方检验")
        st.write(f"• χ² = {chi2:.4f}")
        st.write(f"• P值 = {p_value:.4f}")
        st.write(f"• 自由度 = {(crosstab.shape[0]-1) * (crosstab.shape[1]-1)}")
        
        if p_value < 0.05:
            st.success("✅ 不同暴露状态间患病率存在显著差异")
        else:
            st.info("ℹ️ 不同暴露状态间患病率无显著差异")
    
    except Exception as e:
        st.warning(f"⚠️ 统计检验失败: {str(e)}")

def disease_surveillance(df):
    """疾病监测分析"""
    st.markdown("### 🌍 疾病监测分析")
    st.markdown("*疾病监测数据的流行病学分析*")
    
    # 变量识别和选择
    col1, col2, col3 = st.columns(3)
    
    with col1:
        time_var = st.selectbox("选择时间变量", df.columns.tolist())
    
    with col2:
        case_var = st.selectbox("选择病例数变量", df.columns.tolist())
    
    with col3:
        area_var = st.selectbox("选择地区变量", df.columns.tolist(), help="可选")
    
    if not all([time_var, case_var]):
        return
    
    # 监测分析类型
        surveillance_type = st.selectbox(
        "选择监测分析类型",
        ["时间序列分析", "疫情预警分析", "发病率监测", "异常检测", "趋势预测"]
    )
    
    if surveillance_type == "时间序列分析":
        surveillance_time_series(df, time_var, case_var, area_var)
    elif surveillance_type == "疫情预警分析":
        outbreak_alert_analysis(df, time_var, case_var, area_var)
    elif surveillance_type == "发病率监测":
        incidence_surveillance(df, time_var, case_var, area_var)
    elif surveillance_type == "异常检测":
        anomaly_detection(df, time_var, case_var, area_var)
    elif surveillance_type == "趋势预测":
        trend_prediction(df, time_var, case_var, area_var)

def surveillance_time_series(df, time_var, case_var, area_var):
    """监测时间序列分析"""
    st.markdown("#### 📈 时间序列分析")
    
    try:
        # 确保时间变量为日期格式
        if df[time_var].dtype != 'datetime64[ns]':
            df[time_var] = pd.to_datetime(df[time_var], errors='coerce')
        
        # 按时间排序
        df_sorted = df.sort_values(time_var)
        
        # 时间聚合选择
        time_unit = st.selectbox("选择时间聚合单位", ["日", "周", "月", "季度"])
        
        # 数据聚合
        if time_unit == "日":
            df_sorted['时间组'] = df_sorted[time_var].dt.date
        elif time_unit == "周":
            df_sorted['时间组'] = df_sorted[time_var].dt.to_period('W')
        elif time_unit == "月":
            df_sorted['时间组'] = df_sorted[time_var].dt.to_period('M')
        elif time_unit == "季度":
            df_sorted['时间组'] = df_sorted[time_var].dt.to_period('Q')
        
        # 聚合病例数
        if area_var and area_var != time_var:
            # 按地区和时间聚合
            time_series_data = df_sorted.groupby(['时间组', area_var])[case_var].sum().reset_index()
            
            # 可视化不同地区的时间序列
            fig = px.line(
                time_series_data, 
                x='时间组', 
                y=case_var, 
                color=area_var,
                title=f"各地区{case_var}时间序列",
                markers=True
            )
            
            fig.update_layout(
                xaxis_title="时间",
                yaxis_title=case_var,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 总体时间序列
            total_series = df_sorted.groupby('时间组')[case_var].sum().reset_index()
            
        else:
            # 总体时间序列
            total_series = df_sorted.groupby('时间组')[case_var].sum().reset_index()
        
        # 显示总体时间序列
        st.markdown("##### 📊 总体时间序列")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(total_series.tail(10))
        
        with col2:
            # 总体趋势图
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=total_series['时间组'].astype(str),
                y=total_series[case_var],
                mode='lines+markers',
                name='病例数',
                line=dict(color='red', width=2),
                marker=dict(size=6)
            ))
            
            fig.update_layout(
                title=f"总体{case_var}时间趋势",
                xaxis_title="时间",
                yaxis_title=case_var,
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # 时间序列统计分析
        perform_time_series_analysis(total_series, case_var)
        
    except Exception as e:
        st.error(f"❌ 时间序列分析失败: {str(e)}")

def perform_time_series_analysis(data, case_var):
    """执行时间序列统计分析"""
    st.markdown("##### 📊 时间序列统计特征")
    
    try:
        values = data[case_var].values
        
        # 基本统计
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("均值", f"{np.mean(values):.2f}")
        with col2:
            st.metric("标准差", f"{np.std(values):.2f}")
        with col3:
            st.metric("最大值", f"{np.max(values)}")
        with col4:
            st.metric("最小值", f"{np.min(values)}")
        
        # 趋势分析
        if len(values) >= 3:
            # 简单线性趋势
            x = np.arange(len(values))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
            
            st.markdown("**趋势分析:**")
            st.write(f"• 线性趋势斜率: {slope:.4f}")
            st.write(f"• 相关系数: {r_value:.4f}")
            st.write(f"• P值: {p_value:.4f}")
            
            if p_value < 0.05:
                if slope > 0:
                    st.success("✅ 存在显著上升趋势")
                else:
                    st.warning("⚠️ 存在显著下降趋势")
            else:
                st.info("ℹ️ 无显著时间趋势")
        
        # 季节性检测（如果数据点足够）
        if len(values) >= 12:
            detect_seasonality(values)
    
    except Exception as e:
        st.warning(f"⚠️ 时间序列分析失败: {str(e)}")

def detect_seasonality(values):
    """检测季节性模式"""
    st.markdown("**季节性分析:**")
    
    try:
        # 简单的季节性检测 - 使用自相关
        n = len(values)
        
        # 检测12个月的季节性（如果数据足够）
        if n >= 24:
            # 计算12期滞后的自相关
            lag_12_corr = np.corrcoef(values[:-12], values[12:])[0, 1]
            
            st.write(f"• 12期滞后自相关: {lag_12_corr:.4f}")
            
            if abs(lag_12_corr) > 0.3:
                st.success("✅ 检测到可能的年度季节性模式")
            else:
                st.info("ℹ️ 未检测到明显的年度季节性")
        
        # 检测其他周期
        for lag in [3, 4, 6]:
            if n >= lag * 2:
                corr = np.corrcoef(values[:-lag], values[lag:])[0, 1]
                st.write(f"• {lag}期滞后自相关: {corr:.4f}")
    
    except Exception as e:
        st.warning(f"⚠️ 季节性检测失败: {str(e)}")

def outbreak_alert_analysis(df, time_var, case_var, area_var):
    """疫情预警分析"""
    st.markdown("#### ⚡ 疫情预警分析")
    
    try:
        # 预警阈值设置
        col1, col2 = st.columns(2)
        
        with col1:
            threshold_method = st.selectbox(
                "选择预警阈值方法",
                ["历史均值+2SD", "历史均值+3SD", "百分位数法", "自定义阈值"]
            )
        
        with col2:
            if threshold_method == "百分位数法":
                percentile = st.slider("选择百分位数", 75, 99, 95)
            elif threshold_method == "自定义阈值":
                custom_threshold = st.number_input("输入预警阈值", value=10.0)
        
        # 计算预警阈值
        historical_data = df[case_var].dropna()
        
        if threshold_method == "历史均值+2SD":
            threshold = historical_data.mean() + 2 * historical_data.std()
        elif threshold_method == "历史均值+3SD":
            threshold = historical_data.mean() + 3 * historical_data.std()
        elif threshold_method == "百分位数法":
            threshold = np.percentile(historical_data, percentile)
        elif threshold_method == "自定义阈值":
            threshold = custom_threshold
        
        st.write(f"**预警阈值: {threshold:.2f}**")
        
        # 识别预警事件
        df_alert = df.copy()
        if df_alert[time_var].dtype != 'datetime64[ns]':
            df_alert[time_var] = pd.to_datetime(df_alert[time_var], errors='coerce')
        
        df_alert['预警状态'] = df_alert[case_var] > threshold
        alert_events = df_alert[df_alert['预警状态'] == True]
        
        # 显示预警统计
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("总观察期数", len(df_alert))
        with col2:
            st.metric("预警次数", len(alert_events))
        with col3:
            alert_rate = len(alert_events) / len(df_alert) * 100 if len(df_alert) > 0 else 0
            st.metric("预警率", f"{alert_rate:.1f}%")
        
        # 预警事件列表
        if len(alert_events) > 0:
            st.markdown("##### 🚨 预警事件详情")
            
            alert_display = alert_events[[time_var, case_var]]
            if area_var and area_var in alert_events.columns:
                alert_display = alert_events[[time_var, area_var, case_var]]
            
            st.dataframe(alert_display.sort_values(time_var, ascending=False))
        
        # 预警时间序列可视化
        fig = go.Figure()
        
        # 添加病例数时间序列
        fig.add_trace(go.Scatter(
            x=df_alert[time_var],
            y=df_alert[case_var],
            mode='lines+markers',
            name='病例数',
            line=dict(color='blue'),
            marker=dict(size=4)
        ))
        
        # 添加预警阈值线
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"预警阈值: {threshold:.1f}"
        )
        
        # 标记预警点
        if len(alert_events) > 0:
            fig.add_trace(go.Scatter(
                x=alert_events[time_var],
                y=alert_events[case_var],
                mode='markers',
                name='预警事件',
                marker=dict(color='red', size=8, symbol='triangle-up')
            ))
        
        fig.update_layout(
            title="疫情预警监测图",
            xaxis_title="时间",
            yaxis_title=case_var,
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 预警性能评估
        evaluate_alert_performance(df_alert, case_var, threshold)
        
    except Exception as e:
        st.error(f"❌ 疫情预警分析失败: {str(e)}")

def evaluate_alert_performance(df, case_var, threshold):
    """评估预警性能"""
    st.markdown("##### 📊 预警性能评估")
    
    try:
        # 计算预警性能指标
        true_alerts = len(df[df[case_var] > threshold])
        false_alerts = 0  # 简化处理
        missed_alerts = 0  # 需要真实疫情数据来计算
        
        # 敏感性和特异性（需要真实标签）
        st.markdown("**预警统计:**")
        st.write(f"• 触发预警次数: {true_alerts}")
        st.write(f"• 预警触发率: {true_alerts/len(df)*100:.1f}%")
        
        # 预警间隔分析
        alert_dates = df[df[case_var] > threshold].index
        if len(alert_dates) > 1:
            intervals = np.diff(alert_dates)
            avg_interval = np.mean(intervals)
            st.write(f"• 平均预警间隔: {avg_interval:.1f} 个观察期")
        
        # 阈值敏感性分析
        st.markdown("**阈值敏感性分析:**")
        
        thresholds = np.linspace(df[case_var].min(), df[case_var].max(), 10)
        alert_rates = []
        
        for t in thresholds:
            rate = len(df[df[case_var] > t]) / len(df) * 100
            alert_rates.append(rate)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=thresholds,
            y=alert_rates,
            mode='lines+markers',
            name='预警率'
        ))
        
        fig.add_vline(x=threshold, line_dash="dash", line_color="red")
        
        fig.update_layout(
            title="预警阈值敏感性分析",
            xaxis_title="阈值",
            yaxis_title="预警率(%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.warning(f"⚠️ 预警性能评估失败: {str(e)}")

def screening_test_evaluation(df):
    """筛查试验评价"""
    st.markdown("### 📊 筛查试验评价")
    st.markdown("*诊断试验的敏感性、特异性等指标评价*")
    
    # 变量选择
    col1, col2 = st.columns(2)
    
    with col1:
        test_result_var = st.selectbox("选择筛查试验结果变量", df.columns.tolist())
    
    with col2:
        gold_standard_var = st.selectbox("选择金标准结果变量", df.columns.tolist())
    
    if not all([test_result_var, gold_standard_var]):
        return
    
    # 分析类型
    evaluation_type = st.selectbox(
        "选择评价类型",
        ["诊断试验评价", "ROC曲线分析", "多个试验比较", "截断值优化", "预测值分析"]
    )
    
    if evaluation_type == "诊断试验评价":
        diagnostic_test_evaluation(df, test_result_var, gold_standard_var)
    elif evaluation_type == "ROC曲线分析":
        roc_curve_analysis(df, test_result_var, gold_standard_var)
    elif evaluation_type == "多个试验比较":
        multiple_tests_comparison(df, test_result_var, gold_standard_var)
    elif evaluation_type == "截断值优化":
        cutoff_optimization(df, test_result_var, gold_standard_var)
    elif evaluation_type == "预测值分析":
        predictive_value_analysis(df, test_result_var, gold_standard_var)

def diagnostic_test_evaluation(df, test_var, gold_var):
    """诊断试验评价"""
    st.markdown("#### 🔬 诊断试验评价")
    
    try:
        # 创建2x2表
        crosstab = pd.crosstab(df[test_var], df[gold_var], margins=True)
        
        st.markdown("##### 📋 诊断试验2×2表")
        st.dataframe(crosstab)
        
        # 提取2x2表数据
        if crosstab.shape == (3, 3):
            # 假设阳性结果在第一行/列
            a = crosstab.iloc[0, 0]  # 真阳性
            b = crosstab.iloc[0, 1]  # 假阳性
            c = crosstab.iloc[1, 0]  # 假阴性
            d = crosstab.iloc[1, 1]  # 真阴性
            
            # 计算诊断指标
            sensitivity = a / (a + c) if (a + c) > 0 else 0
            specificity = d / (b + d) if (b + d) > 0 else 0
            ppv = a / (a + b) if (a + b) > 0 else 0  # 阳性预测值
            npv = d / (c + d) if (c + d) > 0 else 0  # 阴性预测值
            accuracy = (a + d) / (a + b + c + d) if (a + b + c + d) > 0 else 0
            
            # 似然比
            lr_positive = sensitivity / (1 - specificity) if specificity < 1 else float('inf')
            lr_negative = (1 - sensitivity) / specificity if specificity > 0 else float('inf')
            
            # 约登指数
            youden_index = sensitivity + specificity - 1
            
            # 显示结果
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 📊 诊断性能指标")
                
                metrics_df = pd.DataFrame({
                    '指标': [
                        '敏感性(Sensitivity)',
                        '特异性(Specificity)', 
                        '阳性预测值(PPV)',
                        '阴性预测值(NPV)',
                        '准确性(Accuracy)',
                        '阳性似然比(LR+)',
                        '阴性似然比(LR-)',
                        '约登指数'
                    ],
                    '数值': [
                        f"{sensitivity:.4f} ({sensitivity*100:.1f}%)",
                        f"{specificity:.4f} ({specificity*100:.1f}%)",
                        f"{ppv:.4f} ({ppv*100:.1f}%)",
                        f"{npv:.4f} ({npv*100:.1f}%)",
                        f"{accuracy:.4f} ({accuracy*100:.1f}%)",
                        f"{lr_positive:.2f}" if lr_positive != float('inf') else "∞",
                        f"{lr_negative:.4f}" if lr_negative != float('inf') else "∞",
                        f"{youden_index:.4f}"
                    ]
                })
                
                st.dataframe(metrics_df, hide_index=True)
            
            with col2:
                # 性能指标可视化
                metrics_viz = ['敏感性', '特异性', '阳性预测值', '阴性预测值', '准确性']
                values_viz = [sensitivity, specificity, ppv, npv, accuracy]
                
                fig = go.Figure()
                
                fig.add_trace(go.Bar(
                    x=metrics_viz,
                    y=values_viz,
                    marker_color=['red', 'blue', 'green', 'orange', 'purple']
                ))
                
                fig.update_layout(
                    title="诊断性能指标",
                    yaxis_title="数值",
                    yaxis=dict(range=[0, 1]),
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # 计算置信区间
            calculate_diagnostic_ci(a, b, c, d, sensitivity, specificity, ppv, npv)
            
            # 结果解释
            interpret_diagnostic_results(sensitivity, specificity, lr_positive, lr_negative)
        
        else:
            st.warning("⚠️ 数据格式不适合2×2分析")
    
    except Exception as e:
        st.error(f"❌ 诊断试验评价失败: {str(e)}")

def calculate_diagnostic_ci(a, b, c, d, sensitivity, specificity, ppv, npv):
    """计算诊断指标的置信区间"""
    st.markdown("##### 📊 95%置信区间")
    
    try:
        # 敏感性置信区间
        if (a + c) > 0:
            sens_ci = proportion_confint(a, a + c, alpha=0.05, method='wilson')
        else:
            sens_ci = (0, 0)
        
        # 特异性置信区间
        if (b + d) > 0:
            spec_ci = proportion_confint(d, b + d, alpha=0.05, method='wilson')
        else:
            spec_ci = (0, 0)
        
        # PPV置信区间
        if (a + b) > 0:
            ppv_ci = proportion_confint(a, a + b, alpha=0.05, method='wilson')
        else:
            ppv_ci = (0, 0)
        
        # NPV置信区间
        if (c + d) > 0:
            npv_ci = proportion_confint(d, c + d, alpha=0.05, method='wilson')
        else:
            npv_ci = (0, 0)
        
        ci_df = pd.DataFrame({
            '指标': ['敏感性', '特异性', '阳性预测值', '阴性预测值'],
            '点估计': [
                f"{sensitivity:.3f}",
                f"{specificity:.3f}",
                f"{ppv:.3f}",
                f"{npv:.3f}"
            ],
            '95%CI下限': [
                f"{sens_ci[0]:.3f}",
                f"{spec_ci[0]:.3f}",
                f"{ppv_ci[0]:.3f}",
                f"{npv_ci[0]:.3f}"
            ],
            '95%CI上限': [
                f"{sens_ci[1]:.3f}",
                f"{spec_ci[1]:.3f}",
                f"{ppv_ci[1]:.3f}",
                f"{npv_ci[1]:.3f}"
            ]
        })
        
        st.dataframe(ci_df, hide_index=True)
    
    except ImportError:
        st.warning("⚠️ 需要安装statsmodels库计算置信区间")
    except Exception as e:
        st.warning(f"⚠️ 置信区间计算失败: {str(e)}")

def proportion_confint(count, nobs, alpha=0.05, method='wilson'):
    """计算比例的置信区间（简化版本）"""
    p = count / nobs
    z = 1.96  # 95% CI
    
    if method == 'wilson':
        # Wilson方法
        n = nobs
        p_adj = (count + z**2/2) / (n + z**2)
        margin = z * np.sqrt(p_adj * (1 - p_adj) / (n + z**2))
        return (max(0, p_adj - margin), min(1, p_adj + margin))
    else:
        # 正态近似
        se = np.sqrt(p * (1 - p) / nobs)
        margin = z * se
        return (max(0, p - margin), min(1, p + margin))

def interpret_diagnostic_results(sensitivity, specificity, lr_pos, lr_neg):
    """解释诊断结果"""
    st.markdown("##### 💡 结果解释")
    
    # 敏感性解释
    if sensitivity >= 0.9:
        st.success(f"✅ 敏感性优秀({sensitivity*100:.1f}%)，能很好地识别患病者")
    elif sensitivity >= 0.8:
        st.info(f"ℹ️ 敏感性良好({sensitivity*100:.1f}%)，能较好地识别患病者")
    else:
        st.warning(f"⚠️ 敏感性较低({sensitivity*100:.1f}%)，可能遗漏较多患病者")
    
    # 特异性解释
    if specificity >= 0.9:
        st.success(f"✅ 特异性优秀({specificity*100:.1f}%)，能很好地排除非患病者")
    elif specificity >= 0.8:
        st.info(f"ℹ️ 特异性良好({specificity*100:.1f}%)，能较好地排除非患病者")
    else:
        st.warning(f"⚠️ 特异性较低({specificity*100:.1f}%)，可能误诊较多非患病者")
    
    # 似然比解释
    if lr_pos > 10:
        st.success(f"✅ 阳性似然比很高({lr_pos:.1f})，阳性结果有很强的诊断价值")
    elif lr_pos > 5:
        st.info(f"ℹ️ 阳性似然比较高({lr_pos:.1f})，阳性结果有一定诊断价值")
    elif lr_pos > 2:
        st.info(f"ℹ️ 阳性似然比中等({lr_pos:.1f})，阳性结果有轻微诊断价值")
    else:
        st.warning(f"⚠️ 阳性似然比较低({lr_pos:.1f})，阳性结果诊断价值有限")
    
    if lr_neg < 0.1:
        st.success(f"✅ 阴性似然比很低({lr_neg:.3f})，阴性结果能很好地排除疾病")
    elif lr_neg < 0.2:
        st.info(f"ℹ️ 阴性似然比较低({lr_neg:.3f})，阴性结果能较好地排除疾病")
    elif lr_neg < 0.5:
        st.info(f"ℹ️ 阴性似然比中等({lr_neg:.3f})，阴性结果有一定排除价值")
    else:
        st.warning(f"⚠️ 阴性似然比较高({lr_neg:.3f})，阴性结果排除价值有限")

# 主函数调用
if __name__ == "__main__":
    epidemiology_analysis()


