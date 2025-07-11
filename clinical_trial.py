import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, ttest_ind
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

def baseline_analysis_ui():
    st.header("📊 基线特征分析")
    
    if st.session_state.cleaned_data is None:
        st.warning("请先导入并清理数据")
        return
    
    df = st.session_state.cleaned_data
    
    # 变量选择
    st.subheader("🎯 变量选择")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 分组变量
        categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
        group_var = st.selectbox("分组变量", ["无分组"] + categorical_vars)
    
    with col2:
        # 连续变量
        numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
        continuous_vars = st.multiselect("连续变量", numeric_vars)
    
    with col3:
        # 分类变量
        categorical_analysis_vars = st.multiselect("分类变量", categorical_vars)
    
    # 分析设置
    st.subheader("⚙️ 分析设置")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        continuous_test = st.selectbox(
            "连续变量检验",
            ["t检验", "Mann-Whitney U检验", "方差分析", "Kruskal-Wallis检验"]
        )
    
    with col2:
        categorical_test = st.selectbox(
            "分类变量检验",
            ["卡方检验", "Fisher精确检验"]
        )
    
    with col3:
        conf_level = st.number_input("置信水平", value=0.95, min_value=0.8, max_value=0.99, step=0.01)
    
    with col4:
        paired_test = st.checkbox("配对检验", False)
    
    # 运行分析
    if st.button("🚀 运行基线分析", type="primary"):
        if not continuous_vars and not categorical_analysis_vars:
            st.warning("请至少选择一个分析变量")
            return
        
        try:
            results = perform_baseline_analysis(
                df, group_var, continuous_vars, categorical_analysis_vars,
                continuous_test, categorical_test, conf_level, paired_test
            )
            
            display_baseline_results(results, df, group_var, continuous_vars, categorical_analysis_vars)
            
        except Exception as e:
            st.error(f"分析失败: {str(e)}")

def perform_baseline_analysis(df, group_var, continuous_vars, categorical_vars, 
                            continuous_test, categorical_test, conf_level, paired_test):
    """执行基线特征分析"""
    
    results = {
        'descriptive': {},
        'tests': {},
        'table_one': []
    }
    
    # 如果有分组变量
    if group_var != "无分组":
        groups = df[group_var].unique()
        
        # 连续变量分析
        for var in continuous_vars:
            var_data = df[var].dropna()
            group_data = df[group_var][var_data.index]
            
            # 描述统计
            desc_stats = {}
            for group in groups:
                group_values = var_data[group_data == group]
                desc_stats[group] = {
                    'n': len(group_values),
                    'mean': np.mean(group_values),
                    'std': np.std(group_values, ddof=1),
                    'median': np.median(group_values),
                    'q25': np.percentile(group_values, 25),
                    'q75': np.percentile(group_values, 75),
                    'min': np.min(group_values),
                    'max': np.max(group_values)
                }
            
            results['descriptive'][var] = desc_stats
            
            # 统计检验
            if len(groups) == 2:
                group1_data = var_data[group_data == groups[0]]
                group2_data = var_data[group_data == groups[1]]
                
                if continuous_test == "t检验":
                    stat, p_value = ttest_ind(group1_data, group2_data)
                    test_name = "独立样本t检验"
                elif continuous_test == "Mann-Whitney U检验":
                    stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                    test_name = "Mann-Whitney U检验"
                
                results['tests'][var] = {
                    'test': test_name,
                    'statistic': stat,
                    'p_value': p_value,
                    'significant': p_value < (1 - conf_level)
                }
            
            # 添加到Table 1
            if len(groups) == 2:
                results['table_one'].append({
                    '变量': var,
                    '类型': '连续变量',
                    f'{groups[0]} (n={desc_stats[groups[0]]["n"]})': 
                        f"{desc_stats[groups[0]]['mean']:.2f} ± {desc_stats[groups[0]]['std']:.2f}",
                    f'{groups[1]} (n={desc_stats[groups[1]]["n"]})': 
                        f"{desc_stats[groups[1]]['mean']:.2f} ± {desc_stats[groups[1]]['std']:.2f}",
                    'P值': f"{results['tests'][var]['p_value']:.4f}" if var in results['tests'] else "N/A"
                })
        
        # 分类变量分析
        for var in categorical_vars:
            if var == group_var:
                continue
                
            # 交叉表
            crosstab = pd.crosstab(df[var], df[group_var], margins=True)
            results['descriptive'][var] = crosstab
            
            # 统计检验
            if categorical_test == "卡方检验":
                chi2, p_value, dof, expected = chi2_contingency(crosstab.iloc[:-1, :-1])
                test_name = "卡方检验"
                stat = chi2
            elif categorical_test == "Fisher精确检验":
                if crosstab.shape == (3, 3):  # 2x2表
                    oddsratio, p_value = fisher_exact(crosstab.iloc[:-1, :-1])
                    test_name = "Fisher精确检验"
                    stat = oddsratio
                else:
                    # 对于大于2x2的表，使用卡方检验
                    chi2, p_value, dof, expected = chi2_contingency(crosstab.iloc[:-1, :-1])
                    test_name = "卡方检验"
                    stat = chi2
            
            results['tests'][var] = {
                'test': test_name,
                'statistic': stat,
                'p_value': p_value,
                'significant': p_value < (1 - conf_level)
            }
            
            # 添加到Table 1
            for category in crosstab.index[:-1]:  # 排除Total行
                row_data = {'变量': f"{var} - {category}", '类型': '分类变量'}
                for group in groups:
                    count = crosstab.loc[category, group]
                    total = crosstab.loc['All', group]
                    percentage = (count / total * 100) if total > 0 else 0
                    row_data[f'{group} (n={total})'] = f"{count} ({percentage:.1f}%)"
                
                row_data['P值'] = f"{results['tests'][var]['p_value']:.4f}" if category == crosstab.index[0] else ""
                results['table_one'].append(row_data)
    
    else:
        # 无分组的描述性分析
        for var in continuous_vars:
            var_data = df[var].dropna()
            results['descriptive'][var] = {
                'overall': {
                    'n': len(var_data),
                    'mean': np.mean(var_data),
                    'std': np.std(var_data, ddof=1),
                    'median': np.median(var_data),
                    'q25': np.percentile(var_data, 25),
                    'q75': np.percentile(var_data, 75),
                    'min': np.min(var_data),
                    'max': np.max(var_data)
                }
            }
        
        for var in categorical_vars:
            value_counts = df[var].value_counts()
            results['descriptive'][var] = value_counts
    
    return results

def display_baseline_results(results, df, group_var, continuous_vars, categorical_vars):
    """显示基线分析结果"""
    
    # 创建标签页
    tab1, tab2, tab3, tab4 = st.tabs(["📊 描述统计", "🔬 统计检验", "📈 可视化", "📋 Table 1"])
    
    with tab1:
        st.subheader("描述统计结果")
        
        # 连续变量描述统计
        if continuous_vars:
            st.write("**连续变量:**")
            for var in continuous_vars:
                st.write(f"**{var}**")
                if group_var != "无分组":
                    desc_df = pd.DataFrame(results['descriptive'][var]).T
                    st.dataframe(desc_df.round(3))
                else:
                    desc_data = results['descriptive'][var]['overall']
                    desc_df = pd.DataFrame([desc_data])
                    st.dataframe(desc_df.round(3))
                st.write("---")
        
        # 分类变量描述统计
        if categorical_vars:
            st.write("**分类变量:**")
            for var in categorical_vars:
                if var != group_var:
                    st.write(f"**{var}**")
                    if isinstance(results['descriptive'][var], pd.DataFrame):
                        st.dataframe(results['descriptive'][var])
                    else:
                        st.write(results['descriptive'][var])
                    st.write("---")
    
    with tab2:
        st.subheader("统计检验结果")
        
        if results['tests']:
            test_results = []
            for var, test_result in results['tests'].items():
                test_results.append({
                    '变量': var,
                    '检验方法': test_result['test'],
                    '统计量': f"{test_result['statistic']:.4f}",
                    'P值': f"{test_result['p_value']:.4f}",
                    '是否显著': "是" if test_result['significant'] else "否"
                })
            
            test_df = pd.DataFrame(test_results)
            st.dataframe(test_df, use_container_width=True)
        else:
            st.info("无统计检验结果")
    
    with tab3:
        st.subheader("数据可视化")
        
        if group_var != "无分组":
            # 连续变量可视化
            if continuous_vars:
                st.write("**连续变量分布:**")
                
                for var in continuous_vars[:4]:  # 限制显示数量
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # 箱线图
                        fig = px.box(df, x=group_var, y=var, title=f"{var} 箱线图")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # 小提琴图
                        fig = px.violin(df, x=group_var, y=var, title=f"{var} 分布")
                        st.plotly_chart(fig, use_container_width=True)
            
            # 分类变量可视化
            if categorical_vars:
                st.write("**分类变量分布:**")
                
                for var in categorical_vars[:4]:  # 限制显示数量
                    if var != group_var:
                        crosstab = pd.crosstab(df[var], df[group_var])
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # 堆叠柱状图
                            fig = px.bar(crosstab, title=f"{var} 分组分布")
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # 热力图
                            fig = px.imshow(crosstab, title=f"{var} 交叉表热力图")
                            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("标准化基线特征表格 (Table 1)")
        
        if results['table_one']:
            table_one_df = pd.DataFrame(results['table_one'])
            st.dataframe(table_one_df, use_container_width=True)
            
            # 下载按钮
            csv = table_one_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下载 Table 1",
                data=csv,
                file_name="baseline_characteristics_table1.csv",
                mime="text/csv"
            )
        else:
            st.info("无Table 1数据")
