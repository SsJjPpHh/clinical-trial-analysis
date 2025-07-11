import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency, fisher_exact
import plotly.express as px
import plotly.graph_objects as go

def epidemiology_ui():
    st.header("🦠 流行病学分析")
    
    # 分析类型选择
    analysis_type = st.selectbox(
        "选择分析类型",
        ["研究设计", "队列研究分析", "病例对照研究分析", "横断面研究分析"]
    )
    
    if analysis_type == "研究设计":
        study_design_ui()
    elif analysis_type == "队列研究分析":
        cohort_study_analysis()
    elif analysis_type == "病例对照研究分析":
        case_control_analysis()
    elif analysis_type == "横断面研究分析":
        cross_sectional_analysis()

def study_design_ui():
    st.subheader("📋 流行病学研究设计")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**研究类型选择**")
        study_type = st.selectbox(
            "选择研究类型",
            ["横断面研究", "队列研究", "病例对照研究", "临床试验"]
        )
        
        if study_type == "队列研究":
            st.write("**队列研究参数**")
            follow_up_time = st.number_input("随访时间（年）", value=5.0, min_value=0.1, max_value=50.0)
            expected_incidence = st.number_input("预期发病率（%）", value=10.0, min_value=0.1, max_value=100.0)
            
        elif study_type == "病例对照研究":
            st.write("**病例对照研究参数**")
            case_control_ratio = st.number_input("对照与病例比例", value=1, min_value=1, max_value=10)
            expected_or = st.number_input("预期比值比", value=2.0, min_value=0.1, max_value=10.0)
    
    with col2:
        st.write("**样本量估算**")
        alpha = st.number_input("α水平", value=0.05, min_value=0.01, max_value=0.1, step=0.01)
        power = st.number_input("检验效能(1-β)", value=0.8, min_value=0.5, max_value=0.99, step=0.01)
        
        if st.button("🔢 计算样本量", type="primary"):
            try:
                if study_type == "队列研究":
                    sample_size = calculate_cohort_sample_size(
                        expected_incidence/100, alpha, power, follow_up_time
                    )
                elif study_type == "病例对照研究":
                    sample_size = calculate_case_control_sample_size(
                        expected_or, alpha, power, case_control_ratio
                    )
                else:
                    sample_size = calculate_cross_sectional_sample_size(alpha, power)
                
                display_sample_size_results(sample_size, study_type)
                
            except Exception as e:
                st.error(f"计算失败: {str(e)}")

def calculate_cohort_sample_size(incidence_rate, alpha, power, follow_up_time):
    """计算队列研究样本量"""
    from scipy.stats import norm
    
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    # 简化的样本量计算
    n = ((z_alpha + z_beta)**2 * (1 + 1/incidence_rate)) / (np.log(2)**2)
    n = int(np.ceil(n))
    
    return {
        'total_sample_size': n,
        'exposed_group': n // 2,
        'unexposed_group': n // 2,
        'details': {
            '预期发病率': f"{incidence_rate*100:.1f}%",
            '随访时间': f"{follow_up_time}年",
            'α水平': alpha,
            '检验效能': power
        }
    }

def calculate_case_control_sample_size(odds_ratio, alpha, power, ratio):
    """计算病例对照研究样本量"""
    from scipy.stats import norm
    
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    # 假设暴露率为30%
    p1 = 0.3
    p2 = (odds_ratio * p1) / (1 + p1 * (odds_ratio - 1))
    
    n_cases = ((z_alpha + z_beta)**2 * (1/p1 + 1/p2 + 1/(ratio*p1) + 1/(ratio*p2))) / ((p2 - p1)**2)
    n_cases = int(np.ceil(n_cases))
    n_controls = n_cases * ratio
    
    return {
        'total_sample_size': n_cases + n_controls,
        'cases': n_cases,
        'controls': n_controls,
        'details': {
            '预期比值比': odds_ratio,
            '对照病例比': f"{ratio}:1",
            'α水平': alpha,
            '检验效能': power
        }
    }

def calculate_cross_sectional_sample_size(alpha, power):
    """计算横断面研究样本量"""
    from scipy.stats import norm
    
    z_alpha = norm.ppf(1 - alpha/2)
    z_beta = norm.ppf(power)
    
    # 假设患病率为15%
    prevalence = 0.15
    n = ((z_alpha + z_beta)**2 * prevalence * (1 - prevalence)) / (0.05**2)
    n = int(np.ceil(n))
    
    return {
        'total_sample_size': n,
        'details': {
            '预期患病率': f"{prevalence*100:.1f}%",
            '精度': "±5%",
            'α水平': alpha,
            '检验效能': power
        }
    }

def display_sample_size_results(results, study_type):
    """显示样本量计算结果"""
    
    st.subheader("📊 样本量计算结果")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("总样本量", results['total_sample_size'])
    
    if study_type == "队列研究":
        with col2:
            st.metric("暴露组", results['exposed_group'])
        with col3:
            st.metric("非暴露组", results['unexposed_group'])
    elif study_type == "病例对照研究":
        with col2:
            st.metric("病例数", results['cases'])
        with col3:
            st.metric("对照数", results['controls'])
    
    st.write("**详细信息:**")
    for key, value in results['details'].items():
        st.write(f"- **{key}:** {value}")

def cohort_study_analysis():
    st.subheader("👥 队列研究分析")
    
    if st.session_state.cleaned_data is None:
        st.warning("请先导入并清理数据")
        return
    
    df = st.session_state.cleaned_data
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**变量选择**")
        
        # 暴露变量
        categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
        exposure_var = st.selectbox("暴露变量", ["请选择"] + categorical_vars)
        
        # 结局变量
        outcome_var = st.selectbox("结局变量", ["请选择"] + categorical_vars)
        
        # 时间变量（可选）
        numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
        time_var = st.selectbox("时间变量（可选）", ["无"] + numeric_vars)
        
        run_cohort = st.button("🚀 运行队列分析", type="primary")
    
    with col2:
        if run_cohort and exposure_var != "请选择" and outcome_var != "请选择":
            try:
                results = perform_cohort_analysis(df, exposure_var, outcome_var, time_var)
                display_cohort_results(results, exposure_var, outcome_var)
                
            except Exception as e:
                st.error(f"分析失败: {str(e)}")

def perform_cohort_analysis(df, exposure_var, outcome_var, time_var):
    """执行队列研究分析"""
    
    # 创建2x2表
    crosstab = pd.crosstab(df[exposure_var], df[outcome_var], margins=True)
    
    # 获取四格表数据
    exposed_outcome = crosstab.iloc[1, 1]  # a
    exposed_no_outcome = crosstab.iloc[1, 0]  # b
    unexposed_outcome = crosstab.iloc[0, 1]  # c
    unexposed_no_outcome = crosstab.iloc[0, 0]  # d
    
    # 计算发病率
    incidence_exposed = exposed_outcome / (exposed_outcome + exposed_no_outcome)
    incidence_unexposed = unexposed_outcome / (unexposed_outcome + unexposed_no_outcome)
    
    # 计算相对危险度(RR)
    relative_risk = incidence_exposed / incidence_unexposed if incidence_unexposed > 0 else np.inf
    
    # 计算归因危险度(AR)
    attributable_risk = incidence_exposed - incidence_unexposed
    
    # 计算归因危险度百分比(AR%)
    attributable_risk_percent = (attributable_risk / incidence_exposed) * 100 if incidence_exposed > 0 else 0
    
    # 计算人群归因危险度(PAR)
    total_incidence = crosstab.iloc[2, 1] / crosstab.iloc[2, 2]
    population_attributable_risk = total_incidence - incidence_unexposed
    
    # 计算人群归因危险度百分比(PAR%)
    population_attributable_risk_percent = (population_attributable_risk / total_incidence) * 100 if total_incidence > 0 else 0
    
    # 卡方检验
    chi2, p_value, dof, expected = chi2_contingency(crosstab.iloc[:-1, :-1])
    
    # 计算95%置信区间（简化版本）
    rr_ci_lower = relative_risk * np.exp(-1.96 * np.sqrt(1/exposed_outcome + 1/unexposed_outcome - 1/(exposed_outcome + exposed_no_outcome) - 1/(unexposed_outcome + unexposed_no_outcome)))
    rr_ci_upper = relative_risk * np.exp(1.96 * np.sqrt(1/exposed_outcome + 1/unexposed_outcome - 1/(exposed_outcome + exposed_no_outcome) - 1/(unexposed_outcome + unexposed_no_outcome)))
    
    results = {
        'crosstab': crosstab,
        'incidence_exposed': incidence_exposed,
        'incidence_unexposed': incidence_unexposed,
        'relative_risk': relative_risk,
        'rr_ci': (rr_ci_lower, rr_ci_upper),
        'attributable_risk': attributable_risk,
        'attributable_risk_percent': attributable_risk_percent,
        'population_attributable_risk': population_attributable_risk,
        'population_attributable_risk_percent': population_attributable_risk_percent,
        'chi2_test': {'chi2': chi2, 'p_value': p_value}
    }
    
    return results

def display_cohort_results(results, exposure_var, outcome_var):
    """显示队列研究结果"""
    
    # 2x2表
    st.write("**2×2列联表**")
    st.dataframe(results['crosstab'])
    
    # 主要指标
    st.write("**主要流行病学指标**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("暴露组发病率", f"{results['incidence_exposed']:.4f}")
        st.metric("非暴露组发病率", f"{results['incidence_unexposed']:.4f}")
    
    with col2:
        st.metric("相对危险度(RR)", f"{results['relative_risk']:.4f}")
        st.write(f"95%CI: ({results['rr_ci'][0]:.4f}, {results['rr_ci'][1]:.4f})")
        
    with col3:
        st.metric("归因危险度(AR)", f"{results['attributable_risk']:.4f}")
        st.metric("归因危险度%", f"{results['attributable_risk_percent']:.2f}%")
    
    # 人群指标
    st.write("**人群水平指标**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("人群归因危险度(PAR)", f"{results['population_attributable_risk']:.4f}")
    
    with col2:
        st.metric("人群归因危险度%(PAR%)", f"{results['population_attributable_risk_percent']:.2f}%")
    
    # 统计检验
    st.write("**统计检验**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("卡方统计量", f"{results['chi2_test']['chi2']:.4f}")
    
    with col2:
        st.metric("P值", f"{results['chi2_test']['p_value']:.4f}")
    
    with col3:
        significance = "显著" if results['chi2_test']['p_value'] < 0.05 else "不显著"
        st.metric("统计学意义", significance)

def case_control_analysis():
    st.subheader("🔍 病例对照研究分析")
    
    if st.session_state.cleaned_data is None:
        st.warning("请先导入并清理数据")
        return
    
    df = st.session_state.cleaned_data
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**变量选择**")
        
        # 病例对照变量
        categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
        case_control_var = st.selectbox("病例对照变量", ["请选择"] + categorical_vars)
        
        # 暴露变量
        exposure_var = st.selectbox("暴露变量", ["请选择"] + categorical_vars, key="cc_exposure")
        
        # 匹配变量（可选）
        matching_vars = st.multiselect("匹配变量（可选）", categorical_vars)
        
        run_case_control = st.button("🚀 运行病例对照分析", type="primary")
    
    with col2:
        if run_case_control and case_control_var != "请选择" and exposure_var != "请选择":
            try:
                results = perform_case_control_analysis(df, case_control_var, exposure_var, matching_vars)
                display_case_control_results(results, case_control_var, exposure_var)
                
            except Exception as e:
                st.error(f"分析失败: {str(e)}")

def perform_case_control_analysis(df, case_control_var, exposure_var, matching_vars):
    """执行病例对照研究分析"""
    
    # 创建2x2表
    crosstab = pd.crosstab(df[case_control_var], df[exposure_var], margins=True)
    
    # 获取四格表数据（假设病例为第二个类别，暴露为第二个类别）
    case_exposed = crosstab.iloc[1, 1]  # a
    case_unexposed = crosstab.iloc[1, 0]  # b
    control_exposed = crosstab.iloc[0, 1]  # c
    control_unexposed = crosstab.iloc[0, 0]  # d
    
    # 计算比值比(OR)
    odds_ratio = (case_exposed * control_unexposed) / (case_unexposed * control_exposed) if (case_unexposed * control_exposed) > 0 else np.inf
    
    # 计算95%置信区间
    log_or = np.log(odds_ratio) if odds_ratio > 0 and odds_ratio != np.inf else 0
    se_log_or = np.sqrt(1/case_exposed + 1/case_unexposed + 1/control_exposed + 1/control_unexposed) if all(x > 0 for x in [case_exposed, case_unexposed, control_exposed, control_unexposed]) else 0
    
    or_ci_lower = np.exp(log_or - 1.96 * se_log_or)
    or_ci_upper = np.exp(log_or + 1.96 * se_log_or)
    
    # 计算暴露率
    exposure_rate_cases = case_exposed / (case_exposed + case_unexposed)
    exposure_rate_controls = control_exposed / (control_exposed + control_unexposed)
    
    # 统计检验
    if crosstab.iloc[:-1, :-1].shape == (2, 2):
        # Fisher精确检验
        oddsratio_fisher, p_value_fisher = fisher_exact(crosstab.iloc[:-1, :-1])
        
        # 卡方检验
        chi2, p_value_chi2, dof, expected = chi2_contingency(crosstab.iloc[:-1, :-1])
    else:
        p_value_fisher = np.nan
        chi2, p_value_chi2 = np.nan, np.nan
    
    results = {
        'crosstab': crosstab,
        'odds_ratio': odds_ratio,
        'or_ci': (or_ci_lower, or_ci_upper),
        'exposure_rate_cases': exposure_rate_cases,
        'exposure_rate_controls': exposure_rate_controls,
        'fisher_test': {'p_value': p_value_fisher},
        'chi2_test': {'chi2': chi2, 'p_value': p_value_chi2}
    }
    
    return results

def display_case_control_results(results, case_control_var, exposure_var):
    """显示病例对照研究结果"""
    
    # 2x2表
    st.write("**2×2列联表**")
    st.dataframe(results['crosstab'])
    
    # 主要指标
    st.write("**主要流行病学指标**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("病例组暴露率", f"{results['exposure_rate_cases']:.4f}")
        st.metric("对照组暴露率", f"{results['exposure_rate_controls']:.4f}")
    
    with col2:
        st.metric("比值比(OR)", f"{results['odds_ratio']:.4f}")
        st.write(f"95%CI: ({results['or_ci'][0]:.4f}, {results['or_ci'][1]:.4f})")
    
    with col3:
        # OR解释
        if results['odds_ratio'] > 1:
            interpretation = "暴露增加疾病风险"
        elif results['odds_ratio'] < 1:
            interpretation = "暴露降低疾病风险"
        else:
            interpretation = "暴露与疾病无关联"
        
        st.metric("关联强度", interpretation)
    
    # 统计检验
    st.write("**统计检验**")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Fisher精确检验P值", f"{results['fisher_test']['p_value']:.4f}" if not np.isnan(results['fisher_test']['p_value']) else "N/A")
    
    with col2:
        st.metric("卡方统计量", f"{results['chi2_test']['chi2']:.4f}" if not np.isnan(results['chi2_test']['chi2']) else "N/A")
    
    with col3:
        st.metric("卡方检验P值", f"{results['chi2_test']['p_value']:.4f}" if not np.isnan(results['chi2_test']['p_value']) else "N/A")
    
    with col4:
        p_val = results['fisher_test']['p_value'] if not np.isnan(results['fisher_test']['p_value']) else results['chi2_test']['p_value']
        significance = "显著" if not np.isnan(p_val) and p_val < 0.05 else "不显著"
        st.metric("统计学意义", significance)

def cross_sectional_analysis():
    st.subheader("📊 横断面研究分析")
    
    if st.session_state.cleaned_data is None:
        st.warning("请先导入并清理数据")
        return
    
    df = st.session_state.cleaned_data
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.write("**变量选择**")
        
        # 疾病/结局变量
        categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
        disease_var = st.selectbox("疾病/结局变量", ["请选择"] + categorical_vars)
        
        # 暴露/危险因素变量
        exposure_var = st.selectbox("暴露/危险因素变量", ["请选择"] + categorical_vars, key="cs_exposure")
        
        # 分层变量（可选）
        stratify_var = st.selectbox("分层变量（可选）", ["无"] + categorical_vars)
        
        run_cross_sectional = st.button("🚀 运行横断面分析", type="primary")
    
    with col2:
        if run_cross_sectional and disease_var != "请选择" and exposure_var != "请选择":
            try:
                results = perform_cross_sectional_analysis(df, disease_var, exposure_var, stratify_var)
                display_cross_sectional_results(results, disease_var, exposure_var)
                
            except Exception as e:
                st.error(f"分析失败: {str(e)}")

def perform_cross_sectional_analysis(df, disease_var, exposure_var, stratify_var):
    """执行横断面研究分析"""
    
    results = {
        'overall': {},
        'stratified': {}
    }
    
    # 总体分析
    crosstab = pd.crosstab(df[exposure_var], df[disease_var], margins=True)
    
    # 计算患病率
    exposed_diseased = crosstab.iloc[1, 1]
    exposed_total = crosstab.iloc[1, 2]
    unexposed_diseased = crosstab.iloc[0, 1]
    unexposed_total = crosstab.iloc[0, 2]
    
    prevalence_exposed = exposed_diseased / exposed_total if exposed_total > 0 else 0
    prevalence_unexposed = unexposed_diseased / unexposed_total if unexposed_total > 0 else 0
    
    # 计算患病率比(PR)
    prevalence_ratio = prevalence_exposed / prevalence_unexposed if prevalence_unexposed > 0 else np.inf
    
    # 计算患病率差(PD)
    prevalence_difference = prevalence_exposed - prevalence_unexposed
    
    # 计算比值比(OR)
    exposed_not_diseased = exposed_total - exposed_diseased
    unexposed_not_diseased = unexposed_total - unexposed_diseased
    
    odds_ratio = (exposed_diseased * unexposed_not_diseased) / (exposed_not_diseased * unexposed_diseased) if (exposed_not_diseased * unexposed_diseased) > 0 else np.inf
    
    # 统计检验
    chi2, p_value, dof, expected = chi2_contingency(crosstab.iloc[:-1, :-1])
    
    results['overall'] = {
        'crosstab': crosstab,
        'prevalence_exposed': prevalence_exposed,
        'prevalence_unexposed': prevalence_unexposed,
        'prevalence_ratio': prevalence_ratio,
        'prevalence_difference': prevalence_difference,
        'odds_ratio': odds_ratio,
        'chi2_test': {'chi2': chi2, 'p_value': p_value}
    }
    
    # 分层分析
    if stratify_var != "无":
        strata = df[stratify_var].unique()
        
        for stratum in strata:
            stratum_data = df[df[stratify_var] == stratum]
            stratum_crosstab = pd.crosstab(stratum_data[exposure_var], stratum_data[disease_var], margins=True)
            
            # 计算分层指标（简化版本）
            if stratum_crosstab.shape[0] >= 3 and stratum_crosstab.shape[1] >= 3:
                s_exposed_diseased = stratum_crosstab.iloc[1, 1]
                s_exposed_total = stratum_crosstab.iloc[1, 2]
                s_unexposed_diseased = stratum_crosstab.iloc[0, 1]
                s_unexposed_total = stratum_crosstab.iloc[0, 2]
                
                s_prevalence_exposed = s_exposed_diseased / s_exposed_total if s_exposed_total > 0 else 0
                s_prevalence_unexposed = s_unexposed_diseased / s_unexposed_total if s_unexposed_total > 0 else 0
                s_prevalence_ratio = s_prevalence_exposed / s_prevalence_unexposed if s_prevalence_unexposed > 0 else np.inf
                
                results['stratified'][str(stratum)] = {
                    'crosstab': stratum_crosstab,
                    'prevalence_exposed': s_prevalence_exposed,
                    'prevalence_unexposed': s_prevalence_unexposed,
                    'prevalence_ratio': s_prevalence_ratio
                }
    
    return results

def display_cross_sectional_results(results, disease_var, exposure_var):
    """显示横断面研究结果"""
    
    # 总体分析结果
    st.write("**总体分析结果**")
    
    # 2x2表
    st.write("2×2列联表:")
    st.dataframe(results['overall']['crosstab'])
    
    # 主要指标
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("暴露组患病率", f"{results['overall']['prevalence_exposed']:.4f}")
    
    with col2:
        st.metric("非暴露组患病率", f"{results['overall']['prevalence_unexposed']:.4f}")
    
    with col3:
        st.metric("患病率比(PR)", f"{results['overall']['prevalence_ratio']:.4f}")
    
    with col4:
        st.metric("患病率差(PD)", f"{results['overall']['prevalence_difference']:.4f}")
    
    # 其他指标
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("比值比(OR)", f"{results['overall']['odds_ratio']:.4f}")
    
    with col2:
        st.metric("卡方统计量", f"{results['overall']['chi2_test']['chi2']:.4f}")
    
    with col3:
        st.metric("P值", f"{results['overall']['chi2_test']['p_value']:.4f}")
    
    # 分层分析结果
    if results['stratified']:
        st.write("**分层分析结果**")
        
        stratified_results = []
        for stratum, data in results['stratified'].items():
            stratified_results.append({
                '分层': stratum,
                '暴露组患病率': f"{data['prevalence_exposed']:.4f}",
                '非暴露组患病率': f"{data['prevalence_unexposed']:.4f}",
                '患病率比': f"{data['prevalence_ratio']:.4f}"
            })
        
        stratified_df = pd.DataFrame(stratified_results)
        st.dataframe(stratified_df, use_container_width=True)
    
    # 可视化
    st.write("**数据可视化**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 患病率比较柱状图
        prevalence_data = pd.DataFrame({
            '组别': ['暴露组', '非暴露组'],
            '患病率': [results['overall']['prevalence_exposed'], results['overall']['prevalence_unexposed']]
        })
        
        fig = px.bar(prevalence_data, x='组别', y='患病率', 
                    title="暴露组与非暴露组患病率比较")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 2x2表热力图
        crosstab_values = results['overall']['crosstab'].iloc[:-1, :-1]
        fig = px.imshow(crosstab_values, 
                       title="2×2列联表热力图",
                       labels=dict(x=disease_var, y=exposure_var))
        st.plotly_chart(fig, use_container_width=True)

