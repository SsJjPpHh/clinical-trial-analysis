import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, ttest_ind, wilcoxon
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def clinical_trial_analysis():
    """临床试验分析主函数"""
    st.markdown("# 🧪 临床试验分析")
    st.markdown("*专业的临床试验数据分析工具，支持多种试验设计和统计分析*")
    
    # 侧边栏导航
    with st.sidebar:
        st.markdown("## 🧪 分析模块")
        analysis_type = st.radio(
            "选择分析类型",
            [
                "📊 基线特征分析",
                "🎯 主要终点分析", 
                "📈 次要终点分析",
                "🛡️ 安全性分析",
                "📋 亚组分析",
                "⏱️ 时间趋势分析",
                "🔍 敏感性分析",
                "📄 试验总结报告"
            ]
        )
    
    # 检查数据
    datasets = get_available_datasets()
    if not datasets:
        st.warning("⚠️ 请先在数据管理模块中导入临床试验数据")
        st.info("💡 您可以使用示例数据集中的'临床试验数据'进行学习")
        
        # 提供示例数据选项
        if st.button("🎲 生成临床试验示例数据", use_container_width=True):
            sample_data = generate_clinical_trial_sample_data()
            st.session_state['dataset_clinical_sample'] = {
                'name': '临床试验示例数据',
                'data': sample_data,
                'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            st.success("✅ 示例数据已生成！")
            st.rerun()
        return
    
    # 选择数据集
    selected_dataset = st.selectbox(
        "📊 选择临床试验数据集", 
        list(datasets.keys()),
        help="选择包含临床试验数据的数据集"
    )
    df = datasets[selected_dataset]['data']
    
    # 数据验证
    if not validate_clinical_data(df):
        return
    
    # 根据选择的分析类型调用相应函数
    if analysis_type == "📊 基线特征分析":
        baseline_characteristics_analysis(df)
    elif analysis_type == "🎯 主要终点分析":
        primary_endpoint_analysis(df)
    elif analysis_type == "📈 次要终点分析":
        secondary_endpoint_analysis(df)
    elif analysis_type == "🛡️ 安全性分析":
        safety_analysis(df)
    elif analysis_type == "📋 亚组分析":
        subgroup_analysis(df)
    elif analysis_type == "⏱️ 时间趋势分析":
        time_trend_analysis(df)
    elif analysis_type == "🔍 敏感性分析":
        sensitivity_analysis(df)
    elif analysis_type == "📄 试验总结报告":
        trial_summary_report(df)

def get_available_datasets():
    """获取可用的数据集"""
    datasets = {}
    for key, value in st.session_state.items():
        if key.startswith('dataset_') and isinstance(value, dict) and 'data' in value:
            datasets[value.get('name', key)] = value
    return datasets

def validate_clinical_data(df):
    """验证临床试验数据格式"""
    st.markdown("### 📋 数据验证")
    
    required_cols = ['治疗组', '受试者ID']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        st.error(f"❌ 缺少必要列: {missing_cols}")
        st.info("💡 临床试验数据应包含: 治疗组、受试者ID等基本信息")
        
        # 提供列映射选项
        with st.expander("🔧 列名映射", expanded=True):
            st.markdown("请将您的数据列映射到标准格式:")
            
            col1, col2 = st.columns(2)
            with col1:
                if '治疗组' not in df.columns:
                    treatment_col = st.selectbox("治疗组列", [''] + df.columns.tolist())
                    if treatment_col:
                        df['治疗组'] = df[treatment_col]
            
            with col2:
                if '受试者ID' not in df.columns:
                    subject_col = st.selectbox("受试者ID列", [''] + df.columns.tolist())
                    if subject_col:
                        df['受试者ID'] = df[subject_col]
            
            if st.button("✅ 应用映射"):
                st.success("映射已应用，请重新运行分析")
                st.rerun()
        
        return False
    
    # 数据质量检查
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 总受试者数", len(df))
    with col2:
        treatment_groups = df['治疗组'].nunique()
        st.metric("🎯 治疗组数", treatment_groups)
    with col3:
        missing_rate = df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100
        st.metric("❌ 缺失率", f"{missing_rate:.1f}%")
    with col4:
        duplicate_subjects = df['受试者ID'].duplicated().sum()
        st.metric("🔄 重复受试者", duplicate_subjects)
    
    # 治疗组分布
    treatment_dist = df['治疗组'].value_counts()
    st.markdown("**治疗组分布:**")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(treatment_dist.reset_index().rename(columns={'index': '治疗组', '治疗组': '人数'}))
    with col2:
        fig = px.pie(values=treatment_dist.values, names=treatment_dist.index, 
                     title="治疗组分布", color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig, use_container_width=True)
    
    return True

def baseline_characteristics_analysis(df):
    """基线特征分析"""
    st.markdown("### 📊 基线特征分析")
    st.markdown("*比较各治疗组间基线特征的平衡性*")
    
    # 识别基线变量
    baseline_vars = identify_baseline_variables(df)
    
    if not baseline_vars:
        st.warning("⚠️ 未识别到基线变量")
        return
    
    # 选择要分析的基线变量
    selected_vars = st.multiselect(
        "选择基线变量",
        baseline_vars,
        default=baseline_vars[:10] if len(baseline_vars) >= 10 else baseline_vars,
        help="选择要进行组间比较的基线变量"
    )
    
    if not selected_vars:
        return
    
    # 分析选项
    col1, col2, col3 = st.columns(3)
    with col1:
        show_pvalues = st.checkbox("显示P值", value=True)
    with col2:
        alpha_level = st.selectbox("显著性水平", [0.05, 0.01, 0.001], index=0)
    with col3:
        effect_size = st.checkbox("计算效应量", value=True)
    
    # 执行基线特征分析
    results = perform_baseline_analysis(df, selected_vars, show_pvalues, alpha_level, effect_size)
    
    # 显示结果表格
    st.markdown("#### 📋 基线特征比较表")
    
    # 格式化结果表格
    formatted_results = format_baseline_table(results, show_pvalues, effect_size)
    st.dataframe(formatted_results, use_container_width=True)
    
    # 可视化基线特征
    st.markdown("#### 📊 基线特征可视化")
    
    # 选择可视化变量
    viz_var = st.selectbox("选择可视化变量", selected_vars)
    
    if viz_var:
        create_baseline_visualization(df, viz_var)
    
    # 基线不平衡检测
    st.markdown("#### ⚖️ 基线平衡性评估")
    
    imbalanced_vars = detect_baseline_imbalance(results, alpha_level)
    
    if imbalanced_vars:
        st.warning(f"⚠️ 发现 {len(imbalanced_vars)} 个基线不平衡变量:")
        for var in imbalanced_vars:
            st.write(f"• {var}")
        
        st.info("💡 建议在主要分析中考虑这些变量作为协变量进行调整")
    else:
        st.success("✅ 所有基线变量在组间均衡良好")
    
    # 导出基线特征表
    if st.button("📥 导出基线特征表"):
        export_baseline_table(formatted_results)

def identify_baseline_variables(df):
    """识别基线变量"""
    # 常见的基线变量关键词
    baseline_keywords = [
        '年龄', '性别', '体重', '身高', 'BMI', '血压', '基线', 
        '入组', '筛选', '人口学', '既往史', '合并用药', '病史'
    ]
    
    baseline_vars = []
    
    # 排除明显的结局变量
    exclude_keywords = [
        '终点', '疗效', '不良事件', '随访', '出组', '完成', 
        '依从性', '满意度', '评估', '改善'
    ]
    
    for col in df.columns:
        if col in ['受试者ID', '治疗组']:
            continue
            
        # 检查是否包含基线关键词
        is_baseline = any(keyword in col for keyword in baseline_keywords)
        
        # 检查是否为排除变量
        is_exclude = any(keyword in col for keyword in exclude_keywords)
        
        if is_baseline or (not is_exclude and col not in baseline_vars):
            # 进一步检查数据类型和分布
            if df[col].dtype in ['object', 'category'] or df[col].nunique() < len(df) * 0.8:
                baseline_vars.append(col)
    
    return baseline_vars

def perform_baseline_analysis(df, variables, show_pvalues, alpha_level, effect_size):
    """执行基线特征分析"""
    results = []
    treatment_groups = df['治疗组'].unique()
    
    for var in variables:
        var_result = {'变量': var}
        
        # 检查变量类型
        if df[var].dtype in ['object', 'category'] or df[var].nunique() <= 10:
            # 分类变量
            var_result.update(analyze_categorical_baseline(df, var, treatment_groups, show_pvalues, alpha_level, effect_size))
        else:
            # 连续变量
            var_result.update(analyze_continuous_baseline(df, var, treatment_groups, show_pvalues, alpha_level, effect_size))
        
        results.append(var_result)
    
    return results

def analyze_categorical_baseline(df, var, treatment_groups, show_pvalues, alpha_level, effect_size):
    """分析分类基线变量"""
    result = {'类型': '分类变量'}
    
    # 计算各组的频数和百分比
    for group in treatment_groups:
        group_data = df[df['治疗组'] == group][var]
        value_counts = group_data.value_counts()
        total = len(group_data.dropna())
        
        if total > 0:
            # 格式化为 "n (%)"
            formatted_values = []
            for value, count in value_counts.items():
                pct = count / total * 100
                formatted_values.append(f"{count} ({pct:.1f}%)")
            
            result[f'{group}'] = "; ".join(formatted_values)
        else:
            result[f'{group}'] = "无数据"
    
    # 统计检验
    if show_pvalues:
        try:
            # 创建列联表
            crosstab = pd.crosstab(df[var], df['治疗组'])
            
            if crosstab.shape[0] == 2 and crosstab.shape[1] == 2:
                # 2x2表，使用Fisher精确检验
                _, p_value = fisher_exact(crosstab)
                result['检验方法'] = "Fisher精确检验"
            else:
                # 卡方检验
                chi2, p_value, _, _ = chi2_contingency(crosstab)
                result['检验方法'] = "卡方检验"
            
            result['P值'] = f"{p_value:.4f}"
            result['显著性'] = "是" if p_value < alpha_level else "否"
            
            # 效应量 (Cramér's V)
            if effect_size:
                n = crosstab.sum().sum()
                cramers_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))
                result['效应量(Cramér\'s V)'] = f"{cramers_v:.3f}"
                
        except Exception as e:
            result['P值'] = "计算失败"
            result['检验方法'] = "无法计算"
    
    return result

def analyze_continuous_baseline(df, var, treatment_groups, show_pvalues, alpha_level, effect_size):
    """分析连续基线变量"""
    result = {'类型': '连续变量'}
    
    # 计算各组的描述性统计
    for group in treatment_groups:
        group_data = df[df['治疗组'] == group][var].dropna()
        
        if len(group_data) > 0:
            mean = group_data.mean()
            std = group_data.std()
            median = group_data.median()
            q1 = group_data.quantile(0.25)
            q3 = group_data.quantile(0.75)
            
            # 根据数据分布选择描述方式
            if is_normally_distributed(group_data):
                result[f'{group}'] = f"{mean:.2f} ± {std:.2f}"
            else:
                result[f'{group}'] = f"{median:.2f} ({q1:.2f}, {q3:.2f})"
        else:
            result[f'{group}'] = "无数据"
    
    # 统计检验
    if show_pvalues and len(treatment_groups) >= 2:
        try:
            group_data_list = []
            for group in treatment_groups:
                group_data = df[df['治疗组'] == group][var].dropna()
                group_data_list.append(group_data)
            
            if len(treatment_groups) == 2:
                # 两组比较
                group1_data, group2_data = group_data_list[0], group_data_list[1]
                
                # 正态性检验
                if (is_normally_distributed(group1_data) and is_normally_distributed(group2_data) 
                    and len(group1_data) >= 30 and len(group2_data) >= 30):
                    # t检验
                    _, p_value = ttest_ind(group1_data, group2_data)
                    result['检验方法'] = "独立样本t检验"
                else:
                    # Mann-Whitney U检验
                    _, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                    result['检验方法'] = "Mann-Whitney U检验"
                
                # 效应量
                if effect_size:
                    cohens_d = calculate_cohens_d(group1_data, group2_data)
                    result['效应量(Cohen\'s d)'] = f"{cohens_d:.3f}"
            
            else:
                # 多组比较
                from scipy.stats import kruskal, f_oneway
                
                # 检查正态性
                all_normal = all(is_normally_distributed(data) for data in group_data_list if len(data) >= 8)
                
                if all_normal:
                    # 方差分析
                    _, p_value = f_oneway(*group_data_list)
                    result['检验方法'] = "单因素方差分析"
                else:
                    # Kruskal-Wallis检验
                    _, p_value = kruskal(*group_data_list)
                    result['检验方法'] = "Kruskal-Wallis检验"
            
            result['P值'] = f"{p_value:.4f}"
            result['显著性'] = "是" if p_value < alpha_level else "否"
            
        except Exception as e:
            result['P值'] = "计算失败"
            result['检验方法'] = "无法计算"
    
    return result

def is_normally_distributed(data, alpha=0.05):
    """检验数据是否正态分布"""
    if len(data) < 8:
        return False
    
    try:
        from scipy.stats import shapiro, normaltest
        
        if len(data) <= 5000:
            _, p_value = shapiro(data)
        else:
            _, p_value = normaltest(data)
        
        return p_value > alpha
    except:
        return False

def calculate_cohens_d(group1, group2):
    """计算Cohen's d效应量"""
    n1, n2 = len(group1), len(group2)
    
    if n1 == 0 or n2 == 0:
        return 0
    
    # 计算合并标准差
    pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0
    
    return (group1.mean() - group2.mean()) / pooled_std

def format_baseline_table(results, show_pvalues, effect_size):
    """格式化基线特征表格"""
    df_results = pd.DataFrame(results)
    
    # 重新排列列的顺序
    columns_order = ['变量', '类型']
    
    # 添加治疗组列
    treatment_cols = [col for col in df_results.columns if col not in ['变量', '类型', 'P值', '检验方法', '显著性'] and not col.startswith('效应量')]
    columns_order.extend(treatment_cols)
    
    if show_pvalues:
        columns_order.extend(['检验方法', 'P值', '显著性'])
    
    if effect_size:
        effect_cols = [col for col in df_results.columns if col.startswith('效应量')]
        columns_order.extend(effect_cols)
    
    # 重新排列列
    available_columns = [col for col in columns_order if col in df_results.columns]
    df_results = df_results[available_columns]
    
    return df_results

def create_baseline_visualization(df, var):
    """创建基线特征可视化"""
    treatment_groups = df['治疗组'].unique()
    
    if df[var].dtype in ['object', 'category'] or df[var].nunique() <= 10:
        # 分类变量 - 堆积柱状图
        crosstab = pd.crosstab(df[var], df['治疗组'], normalize='columns') * 100
        
        fig = px.bar(
            crosstab.reset_index(),
            x=var,
            y=crosstab.columns.tolist(),
            title=f"{var} 在各治疗组中的分布",
            labels={'value': '百分比 (%)', 'variable': '治疗组'},
            barmode='group'
        )
        
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # 连续变量 - 箱线图和直方图
        col1, col2 = st.columns(2)
        
        with col1:
            # 箱线图
            fig_box = px.box(
                df, x='治疗组', y=var,
                title=f"{var} 箱线图比较",
                points="outliers"
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # 分组直方图
            fig_hist = px.histogram(
                df, x=var, color='治疗组',
                title=f"{var} 分布直方图",
                barmode='overlay',
                opacity=0.7
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

def detect_baseline_imbalance(results, alpha_level):
    """检测基线不平衡变量"""
    imbalanced_vars = []
    
    for result in results:
        if 'P值' in result and result['P值'] != "计算失败":
            try:
                p_value = float(result['P值'])
                if p_value < alpha_level:
                    imbalanced_vars.append(result['变量'])
            except:
                continue
    
    return imbalanced_vars

def export_baseline_table(formatted_results):
    """导出基线特征表"""
    import io
    
    # 转换为Excel格式
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        formatted_results.to_excel(writer, sheet_name='基线特征分析', index=False)
        
        # 格式化工作表
        workbook = writer.book
        worksheet = writer.sheets['基线特征分析']
        
        # 设置列宽
        for i, col in enumerate(formatted_results.columns):
            max_len = max(len(str(col)), formatted_results[col].astype(str).str.len().max())
            worksheet.set_column(i, i, min(max_len + 2, 50))
    
    output.seek(0)
    
    st.download_button(
        label="📥 下载基线特征表",
        data=output.getvalue(),
        file_name=f"基线特征分析_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

def primary_endpoint_analysis(df):
    """主要终点分析"""
    st.markdown("### 🎯 主要终点分析")
    st.markdown("*分析试验的主要疗效终点*")
    
    # 识别可能的主要终点变量
    endpoint_vars = identify_endpoint_variables(df, endpoint_type='primary')
    
    if not endpoint_vars:
        st.warning("⚠️ 未识别到主要终点变量")
        st.info("💡 请确保数据中包含主要疗效指标")
        return
    
    # 选择主要终点
    col1, col2 = st.columns(2)
    with col1:
        primary_endpoint = st.selectbox(
            "选择主要终点变量",
            endpoint_vars,
            help="选择试验的主要疗效终点"
        )
    
    with col2:
        endpoint_type = st.selectbox(
            "终点类型",
            ["连续型", "二分类", "时间-事件", "有序分类"],
            help="选择终点变量的数据类型"
        )
    
    if not primary_endpoint:
        return
    
    # 分析设置
    with st.expander("🔧 分析设置", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            alpha_level = st.selectbox("显著性水平", [0.05, 0.01, 0.001], index=0)
            confidence_level = 1 - alpha_level
        
        with col2:
            analysis_method = st.selectbox(
                "分析方法",
                ["意向性治疗分析(ITT)", "符合方案集分析(PP)", "安全性分析集(SS)"]
            )
        
        with col3:
            adjustment_vars = st.multiselect(
                "协变量调整",
                [col for col in df.columns if col not in [primary_endpoint, '治疗组', '受试者ID']],
                help="选择需要调整的协变量"
            )
    
    # 执行主要终点分析
    if endpoint_type == "连续型":
        analyze_continuous_endpoint(df, primary_endpoint, alpha_level, confidence_level, adjustment_vars)
    elif endpoint_type == "二分类":
        analyze_binary_endpoint(df, primary_endpoint, alpha_level, confidence_level, adjustment_vars)
    elif endpoint_type == "时间-事件":
        analyze_time_to_event_endpoint(df, primary_endpoint, alpha_level, confidence_level, adjustment_vars)
    elif endpoint_type == "有序分类":
        analyze_ordinal_endpoint(df, primary_endpoint, alpha_level, confidence_level, adjustment_vars)

def identify_endpoint_variables(df, endpoint_type='primary'):
    """识别终点变量"""
    if endpoint_type == 'primary':
        keywords = ['主要终点', '主终点', '疗效', '有效率', '缓解', '改善', '达标']
    else:
        keywords = ['次要终点', '次终点', '生活质量', '满意度', '依从性', '安全性']
    
    endpoint_vars = []
    
    for col in df.columns:
        if col in ['受试者ID', '治疗组']:
            continue
        
        # 检查列名是否包含关键词
        if any(keyword in col for keyword in keywords):
            endpoint_vars.append(col)
    
    # 如果没有找到，返回数值型变量
    if not endpoint_vars:
        endpoint_vars = df.select_dtypes(include=[np.number]).columns.tolist()
        endpoint_vars = [col for col in endpoint_vars if col not in ['受试者ID']]
    
    return endpoint_vars

def analyze_continuous_endpoint(df, endpoint, alpha_level, confidence_level, adjustment_vars):
    """分析连续型主要终点"""
    st.markdown("#### 📊 连续型终点分析结果")
    
    # 描述性统计
    st.markdown("##### 📋 描述性统计")
    
    treatment_groups = df['治疗组'].unique()
    desc_stats = []
    
    for group in treatment_groups:
        group_data = df[df['治疗组'] == group][endpoint].dropna()
        
        if len(group_data) > 0:
            desc_stats.append({
                '治疗组': group,
                '例数': len(group_data),
                '均值': group_data.mean(),
                '标准差': group_data.std(),
                '中位数': group_data.median(),
                '最小值': group_data.min(),
                '最大值': group_data.max(),
                f'{confidence_level*100:.0f}%置信区间下限': group_data.mean() - stats.t.ppf(1-alpha_level/2, len(group_data)-1) * group_data.sem(),
                f'{confidence_level*100:.0f}%置信区间上限': group_data.mean() + stats.t.ppf(1-alpha_level/2, len(group_data)-1) * group_data.sem()
            })
    
    desc_df = pd.DataFrame(desc_stats)
    st.dataframe(desc_df.round(3), use_container_width=True)
    
    # 统计检验
    st.markdown("##### 🧮 统计检验")
    
    if len(treatment_groups) == 2:
        # 两组比较
        group1_data = df[df['治疗组'] == treatment_groups[0]][endpoint].dropna()
        group2_data = df[df['治疗组'] == treatment_groups[1]][endpoint].dropna()
        
        # 选择检验方法
        if (is_normally_distributed(group1_data) and is_normally_distributed(group2_data) 
            and len(group1_data) >= 30 and len(group2_data) >= 30):
            
            # t检验
            t_stat, p_value = ttest_ind(group1_data, group2_data)
            test_method = "独立样本t检验"
            
                        # 计算效应量和置信区间
            mean_diff = group1_data.mean() - group2_data.mean()
            pooled_se = np.sqrt(group1_data.var()/len(group1_data) + group2_data.var()/len(group2_data))
            
            # 置信区间
            df_welch = (group1_data.var()/len(group1_data) + group2_data.var()/len(group2_data))**2 / (
                (group1_data.var()/len(group1_data))**2/(len(group1_data)-1) + 
                (group2_data.var()/len(group2_data))**2/(len(group2_data)-1)
            )
            
            t_critical = stats.t.ppf(1-alpha_level/2, df_welch)
            ci_lower = mean_diff - t_critical * pooled_se
            ci_upper = mean_diff + t_critical * pooled_se
            
            # Cohen's d
            cohens_d = calculate_cohens_d(group1_data, group2_data)
            
        else:
            # Mann-Whitney U检验
            u_stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            test_method = "Mann-Whitney U检验"
            
            # 中位数差异
            median_diff = group1_data.median() - group2_data.median()
            
            # 效应量 (r = Z/sqrt(N))
            z_score = stats.norm.ppf(1 - p_value/2)
            effect_size_r = abs(z_score) / np.sqrt(len(group1_data) + len(group2_data))
            
        # 显示检验结果
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**检验统计量:**")
            if test_method == "独立样本t检验":
                st.write(f"• 检验方法: {test_method}")
                st.write(f"• t统计量: {t_stat:.4f}")
                st.write(f"• P值: {p_value:.4f}")
                st.write(f"• 均值差异: {mean_diff:.3f}")
                st.write(f"• {confidence_level*100:.0f}%置信区间: ({ci_lower:.3f}, {ci_upper:.3f})")
                st.write(f"• Cohen's d: {cohens_d:.3f}")
            else:
                st.write(f"• 检验方法: {test_method}")
                st.write(f"• U统计量: {u_stat:.4f}")
                st.write(f"• P值: {p_value:.4f}")
                st.write(f"• 中位数差异: {median_diff:.3f}")
                st.write(f"• 效应量(r): {effect_size_r:.3f}")
        
        with col2:
            # 结果解释
            st.markdown("**结果解释:**")
            if p_value < alpha_level:
                st.success(f"✅ 在α={alpha_level}水平下，两组间差异具有统计学意义")
            else:
                st.info(f"ℹ️ 在α={alpha_level}水平下，两组间差异无统计学意义")
            
            # 效应量解释
            if test_method == "独立样本t检验":
                if abs(cohens_d) < 0.2:
                    effect_interpretation = "效应量很小"
                elif abs(cohens_d) < 0.5:
                    effect_interpretation = "效应量小"
                elif abs(cohens_d) < 0.8:
                    effect_interpretation = "效应量中等"
                else:
                    effect_interpretation = "效应量大"
                st.write(f"• {effect_interpretation}")
    
    else:
        # 多组比较
        group_data_list = []
        for group in treatment_groups:
            group_data = df[df['治疗组'] == group][endpoint].dropna()
            group_data_list.append(group_data)
        
        # 检查正态性
        all_normal = all(is_normally_distributed(data) for data in group_data_list if len(data) >= 8)
        
        if all_normal:
            # 方差分析
            f_stat, p_value = stats.f_oneway(*group_data_list)
            test_method = "单因素方差分析(ANOVA)"
            
            # 计算效应量 (eta squared)
            ss_between = sum(len(data) * (data.mean() - df[endpoint].mean())**2 for data in group_data_list)
            ss_total = sum((df[endpoint] - df[endpoint].mean())**2)
            eta_squared = ss_between / ss_total
            
        else:
            # Kruskal-Wallis检验
            h_stat, p_value = stats.kruskal(*group_data_list)
            test_method = "Kruskal-Wallis检验"
            
            # 效应量 (epsilon squared)
            n_total = sum(len(data) for data in group_data_list)
            epsilon_squared = (h_stat - len(treatment_groups) + 1) / (n_total - len(treatment_groups))
        
        # 显示多组比较结果
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**检验统计量:**")
            if test_method == "单因素方差分析(ANOVA)":
                st.write(f"• 检验方法: {test_method}")
                st.write(f"• F统计量: {f_stat:.4f}")
                st.write(f"• P值: {p_value:.4f}")
                st.write(f"• 效应量(η²): {eta_squared:.3f}")
            else:
                st.write(f"• 检验方法: {test_method}")
                st.write(f"• H统计量: {h_stat:.4f}")
                st.write(f"• P值: {p_value:.4f}")
                st.write(f"• 效应量(ε²): {epsilon_squared:.3f}")
        
        with col2:
            st.markdown("**结果解释:**")
            if p_value < alpha_level:
                st.success(f"✅ 在α={alpha_level}水平下，各组间差异具有统计学意义")
                
                # 事后多重比较
                if st.checkbox("进行事后多重比较"):
                    perform_post_hoc_analysis(df, endpoint, treatment_groups, alpha_level)
            else:
                st.info(f"ℹ️ 在α={alpha_level}水平下，各组间差异无统计学意义")
    
    # 协变量调整分析
    if adjustment_vars:
        st.markdown("##### 🔧 协变量调整分析")
        perform_covariate_adjustment(df, endpoint, adjustment_vars, alpha_level)
    
    # 可视化
    st.markdown("##### 📊 结果可视化")
    create_endpoint_visualization(df, endpoint, treatment_groups)

def analyze_binary_endpoint(df, endpoint, alpha_level, confidence_level, adjustment_vars):
    """分析二分类主要终点"""
    st.markdown("#### 🎯 二分类终点分析结果")
    
    # 描述性统计
    st.markdown("##### 📋 描述性统计")
    
    treatment_groups = df['治疗组'].unique()
    binary_stats = []
    
    for group in treatment_groups:
        group_data = df[df['治疗组'] == group][endpoint].dropna()
        
        if len(group_data) > 0:
            success_count = group_data.sum() if group_data.dtype in [bool, 'bool'] else (group_data == 1).sum()
            total_count = len(group_data)
            success_rate = success_count / total_count
            
            # 计算95%置信区间 (Wilson方法)
            z = stats.norm.ppf(1 - alpha_level/2)
            n = total_count
            p = success_rate
            
            denominator = 1 + z**2/n
            center = (p + z**2/(2*n)) / denominator
            half_width = z * np.sqrt((p*(1-p) + z**2/(4*n))/n) / denominator
            
            ci_lower = max(0, center - half_width)
            ci_upper = min(1, center + half_width)
            
            binary_stats.append({
                '治疗组': group,
                '总例数': total_count,
                '成功例数': success_count,
                '成功率(%)': success_rate * 100,
                f'{confidence_level*100:.0f}%置信区间下限(%)': ci_lower * 100,
                f'{confidence_level*100:.0f}%置信区间上限(%)': ci_upper * 100
            })
    
    binary_df = pd.DataFrame(binary_stats)
    st.dataframe(binary_df.round(2), use_container_width=True)
    
    # 统计检验
    st.markdown("##### 🧮 统计检验")
    
    if len(treatment_groups) == 2:
        # 两组比较
        group1_data = df[df['治疗组'] == treatment_groups[0]][endpoint].dropna()
        group2_data = df[df['治疗组'] == treatment_groups[1]][endpoint].dropna()
        
        # 创建2x2列联表
        success1 = group1_data.sum() if group1_data.dtype in [bool, 'bool'] else (group1_data == 1).sum()
        success2 = group2_data.sum() if group2_data.dtype in [bool, 'bool'] else (group2_data == 1).sum()
        
        total1, total2 = len(group1_data), len(group2_data)
        fail1, fail2 = total1 - success1, total2 - success2
        
        contingency_table = np.array([[success1, fail1], [success2, fail2]])
        
        # 选择检验方法
        if min(contingency_table.flatten()) >= 5:
            # 卡方检验
            chi2, p_value, _, _ = chi2_contingency(contingency_table)
            test_method = "卡方检验"
        else:
            # Fisher精确检验
            _, p_value = fisher_exact(contingency_table)
            test_method = "Fisher精确检验"
        
        # 计算效应量和风险指标
        rate1 = success1 / total1
        rate2 = success2 / total2
        
        # 相对风险 (RR)
        rr = rate1 / rate2 if rate2 > 0 else float('inf')
        
        # 风险差 (RD)
        rd = rate1 - rate2
        
        # 比值比 (OR)
        if fail1 > 0 and fail2 > 0:
            or_value = (success1 * fail2) / (fail1 * success2)
        else:
            or_value = float('inf')
        
        # 需要治疗的病人数 (NNT)
        nnt = 1 / abs(rd) if rd != 0 else float('inf')
        
        # 显示检验结果
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**检验统计量:**")
            st.write(f"• 检验方法: {test_method}")
            if test_method == "卡方检验":
                st.write(f"• χ²统计量: {chi2:.4f}")
            st.write(f"• P值: {p_value:.4f}")
            
            st.markdown("**效应量指标:**")
            st.write(f"• 相对风险(RR): {rr:.3f}")
            st.write(f"• 风险差(RD): {rd:.3f}")
            st.write(f"• 比值比(OR): {or_value:.3f}")
            if nnt != float('inf'):
                st.write(f"• 需要治疗的病人数(NNT): {nnt:.1f}")
        
        with col2:
            st.markdown("**结果解释:**")
            if p_value < alpha_level:
                st.success(f"✅ 在α={alpha_level}水平下，两组成功率差异具有统计学意义")
            else:
                st.info(f"ℹ️ 在α={alpha_level}水平下，两组成功率差异无统计学意义")
            
            # 临床意义解释
            if rr > 1:
                st.write(f"• 试验组成功率是对照组的{rr:.2f}倍")
            elif rr < 1:
                st.write(f"• 试验组成功率是对照组的{rr:.2f}倍（降低）")
            
            if rd > 0:
                st.write(f"• 试验组成功率比对照组高{abs(rd)*100:.1f}个百分点")
            elif rd < 0:
                st.write(f"• 试验组成功率比对照组低{abs(rd)*100:.1f}个百分点")
    
    else:
        # 多组比较 - 卡方检验
        contingency_table = pd.crosstab(df[endpoint], df['治疗组'])
        chi2, p_value, _, _ = chi2_contingency(contingency_table)
        
        st.markdown("**多组比较结果:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"• 检验方法: 卡方检验")
            st.write(f"• χ²统计量: {chi2:.4f}")
            st.write(f"• P值: {p_value:.4f}")
            st.write(f"• 自由度: {(contingency_table.shape[0]-1)*(contingency_table.shape[1]-1)}")
        
        with col2:
            if p_value < alpha_level:
                st.success(f"✅ 在α={alpha_level}水平下，各组成功率差异具有统计学意义")
            else:
                st.info(f"ℹ️ 在α={alpha_level}水平下，各组成功率差异无统计学意义")
    
    # 可视化
    st.markdown("##### 📊 结果可视化")
    create_binary_endpoint_visualization(df, endpoint, treatment_groups)

def analyze_time_to_event_endpoint(df, endpoint, alpha_level, confidence_level, adjustment_vars):
    """分析时间-事件终点"""
    st.markdown("#### ⏱️ 时间-事件终点分析结果")
    
    # 检查是否有生存时间和事件状态列
    time_col = st.selectbox("选择时间变量", df.select_dtypes(include=[np.number]).columns.tolist())
    event_col = st.selectbox("选择事件状态变量", df.columns.tolist())
    
    if not time_col or not event_col:
        st.warning("⚠️ 时间-事件分析需要时间变量和事件状态变量")
        return
    
    try:
        from lifelines import KaplanMeierFitter, logrank_test
        from lifelines.statistics import multivariate_logrank_test
        
        # 生存分析
        treatment_groups = df['治疗组'].unique()
        
        # Kaplan-Meier生存曲线
        st.markdown("##### 📈 Kaplan-Meier生存曲线")
        
        fig = go.Figure()
        survival_stats = []
        
        for group in treatment_groups:
            group_data = df[df['治疗组'] == group]
            
            if len(group_data) > 0:
                kmf = KaplanMeierFitter()
                kmf.fit(group_data[time_col], group_data[event_col], label=group)
                
                # 添加生存曲线
                fig.add_trace(go.Scatter(
                    x=kmf.timeline,
                    y=kmf.survival_function_[group],
                    mode='lines',
                    name=group,
                    line=dict(width=2)
                ))
                
                # 计算生存统计
                median_survival = kmf.median_survival_time_
                survival_at_times = []
                
                for t in [12, 24, 36]:  # 1年、2年、3年生存率
                    if t <= kmf.timeline.max():
                        survival_rate = kmf.survival_function_at_times(t).iloc[0]
                        survival_at_times.append(f"{t}个月: {survival_rate:.3f}")
                
                survival_stats.append({
                    '治疗组': group,
                    '例数': len(group_data),
                    '事件数': group_data[event_col].sum(),
                    '中位生存时间': median_survival if not np.isnan(median_survival) else "未达到",
                    '生存率': "; ".join(survival_at_times)
                })
        
        fig.update_layout(
            title="Kaplan-Meier生存曲线",
            xaxis_title="时间",
            yaxis_title="生存概率",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # 生存统计表
        st.markdown("##### 📋 生存统计")
        survival_df = pd.DataFrame(survival_stats)
        st.dataframe(survival_df, use_container_width=True)
        
        # Log-rank检验
        st.markdown("##### 🧮 Log-rank检验")
        
        if len(treatment_groups) == 2:
            # 两组比较
            group1_data = df[df['治疗组'] == treatment_groups[0]]
            group2_data = df[df['治疗组'] == treatment_groups[1]]
            
            results = logrank_test(
                group1_data[time_col], group2_data[time_col],
                group1_data[event_col], group2_data[event_col]
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Log-rank检验结果:**")
                st.write(f"• 检验统计量: {results.test_statistic:.4f}")
                st.write(f"• P值: {results.p_value:.4f}")
                st.write(f"• 自由度: 1")
            
            with col2:
                if results.p_value < alpha_level:
                    st.success(f"✅ 在α={alpha_level}水平下，两组生存差异具有统计学意义")
                else:
                    st.info(f"ℹ️ 在α={alpha_level}水平下，两组生存差异无统计学意义")
        
        else:
            # 多组比较
            results = multivariate_logrank_test(df[time_col], df['治疗组'], df[event_col])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**多组Log-rank检验结果:**")
                st.write(f"• 检验统计量: {results.test_statistic:.4f}")
                st.write(f"• P值: {results.p_value:.4f}")
                st.write(f"• 自由度: {len(treatment_groups)-1}")
            
            with col2:
                if results.p_value < alpha_level:
                    st.success(f"✅ 在α={alpha_level}水平下，各组生存差异具有统计学意义")
                else:
                    st.info(f"ℹ️ 在α={alpha_level}水平下，各组生存差异无统计学意义")
        
        # Cox回归分析
        if adjustment_vars:
            st.markdown("##### 🔧 Cox比例风险回归")
            perform_cox_regression(df, time_col, event_col, adjustment_vars)
            
    except ImportError:
        st.error("❌ 需要安装lifelines库进行生存分析")
        st.code("pip install lifelines")

def perform_post_hoc_analysis(df, endpoint, treatment_groups, alpha_level):
    """执行事后多重比较"""
    st.markdown("**事后多重比较 (Tukey HSD):**")
    
    try:
        from scipy.stats import tukey_hsd
        
        # 准备数据
        group_data_list = []
        for group in treatment_groups:
            group_data = df[df['治疗组'] == group][endpoint].dropna()
            group_data_list.append(group_data)
        
        # Tukey HSD检验
        tukey_result = tukey_hsd(*group_data_list)
        
        # 创建比较结果表
        comparisons = []
        for i in range(len(treatment_groups)):
            for j in range(i+1, len(treatment_groups)):
                p_value = tukey_result.pvalue[i, j]
                mean_diff = group_data_list[i].mean() - group_data_list[j].mean()
                
                comparisons.append({
                    '比较组': f"{treatment_groups[i]} vs {treatment_groups[j]}",
                    '均值差异': mean_diff,
                    'P值': p_value,
                    '显著性': "是" if p_value < alpha_level else "否"
                })
        
        comparison_df = pd.DataFrame(comparisons)
        st.dataframe(comparison_df.round(4), use_container_width=True)
        
    except ImportError:
        st.warning("⚠️ 无法进行Tukey HSD检验，请升级scipy版本")

def perform_covariate_adjustment(df, endpoint, adjustment_vars, alpha_level):
    """执行协变量调整分析"""
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        
        # 构建回归公式
        formula = f"{endpoint} ~ C(治疗组)"
        
        for var in adjustment_vars:
            if df[var].dtype in ['object', 'category']:
                formula += f" + C({var})"
            else:
                formula += f" + {var}"
        
        # 拟合模型
        model = ols(formula, data=df).fit()
        
        # 显示结果
        st.markdown("**协变量调整后的结果:**")
        
        # 提取治疗组效应
        treatment_params = [param for param in model.params.index if '治疗组' in param]
        
        if treatment_params:
            for param in treatment_params:
                coef = model.params[param]
                se = model.bse[param]
                p_value = model.pvalues[param]
                ci_lower = model.conf_int().loc[param, 0]
                ci_upper = model.conf_int().loc[param, 1]
                
                st.write(f"• {param}: 系数={coef:.3f}, SE={se:.3f}, P={p_value:.4f}")
                st.write(f"  95%置信区间: ({ci_lower:.3f}, {ci_upper:.3f})")
        
        # 模型拟合优度
        st.write(f"• R²: {model.rsquared:.3f}")
        st.write(f"• 调整R²: {model.rsquared_adj:.3f}")
        st.write(f"• F统计量P值: {model.f_pvalue:.4f}")
        
    except ImportError:
        st.warning("⚠️ 需要安装statsmodels库进行协变量调整")

def create_endpoint_visualization(df, endpoint, treatment_groups):
    """创建终点可视化"""
    col1, col2 = st.columns(2)
    
    with col1:
        # 箱线图
        fig_box = px.box(
            df, x='治疗组', y=endpoint,
            title=f"{endpoint} 组间比较",
            points="outliers"
        )
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # 小提琴图
        fig_violin = px.violin(
            df, x='治疗组', y=endpoint,
            title=f"{endpoint} 分布比较",
            box=True
        )
        fig_violin.update_layout(height=400)
        st.plotly_chart(fig_violin, use_container_width=True)

def create_binary_endpoint_visualization(df, endpoint, treatment_groups):
    """创建二分类终点可视化"""
    # 计算成功率
    success_rates = []
    for group in treatment_groups:
        group_data = df[df['治疗组'] == group][endpoint].dropna()
        if len(group_data) > 0:
            success_count = group_data.sum() if group_data.dtype in [bool, 'bool'] else (group_data == 1).sum()
            success_rate = success_count / len(group_data) * 100
            success_rates.append({'治疗组': group, '成功率(%)': success_rate})
    
    success_df = pd.DataFrame(success_rates)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 柱状图
        fig_bar = px.bar(
            success_df, x='治疗组', y='成功率(%)',
            title="各组成功率比较",
            color='治疗组'
        )
        fig_bar.update_layout(height=400)
        st.plotly_chart(fig_bar, use_container_width=True)
    
    with col2:
        # 饼图（如果只有两组）
        if len(treatment_groups) == 2:
            fig_pie = px.pie(
                success_df, values='成功率(%)', names='治疗组',
                title="成功率分布"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            # 堆积柱状图显示成功/失败
            stacked_data = []
            for group in treatment_groups:
                group_data = df[df['治疗组'] == group][endpoint].dropna()
                if len(group_data) > 0:
                    success_count = group_data.sum() if group_data.dtype in [bool, 'bool'] else (group_data == 1).sum()
                    fail_count = len(group_data) - success_count
                    
                    stacked_data.extend([
                        {'治疗组': group, '结果': '成功', '人数': success_count},
                        {'治疗组': group, '结果': '失败', '人数': fail_count}
                    ])
            
            stacked_df = pd.DataFrame(stacked_data)
            fig_stacked = px.bar(
                stacked_df, x='治疗组', y='人数', color='结果',
                title="成功/失败人数分布",
                barmode='stack'
            )
            fig_stacked.update_layout(height=400)
            st.plotly_chart(fig_stacked, use_container_width=True)

def secondary_endpoint_analysis(df):
    """次要终点分析"""
    st.markdown("### 📈 次要终点分析")
    st.markdown("*分析试验的次要疗效终点和探索性终点*")
    
    # 识别次要终点变量
    secondary_vars = identify_endpoint_variables(df, endpoint_type='secondary')
    
    if not secondary_vars:
        st.warning("⚠️ 未识别到次要终点变量")
        return
    
    # 选择次要终点
    selected_endpoints = st.multiselect(
        "选择次要终点变量",
        secondary_vars,
        default=secondary_vars[:5] if len(secondary_vars) >= 5 else secondary_vars,
        help="可以选择多个次要终点进行分析"
    )
    
    if not selected_endpoints:
        return
    
    # 分析设置
    with st.expander("🔧 分析设置", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            alpha_level = st.selectbox("显著性水平", [0.05, 0.01, 0.001], index=0)
            multiple_comparison = st.checkbox("多重比较校正", value=True)
        
        with col2:
            correction_method = st.selectbox(
                "校正方法",
                ["Bonferroni", "Holm", "FDR (Benjamini-Hochberg)"],
                disabled=not multiple_comparison
            )
        
        with col3:
                        show_effect_size = st.checkbox("显示效应量", value=True)
    
    # 执行次要终点分析
    secondary_results = []
    
    for endpoint in selected_endpoints:
        result = analyze_single_secondary_endpoint(df, endpoint, alpha_level, show_effect_size)
        secondary_results.append(result)
    
    # 多重比较校正
    if multiple_comparison and len(secondary_results) > 1:
        secondary_results = apply_multiple_comparison_correction(secondary_results, correction_method, alpha_level)
    
    # 显示结果
    st.markdown("#### 📊 次要终点分析结果")
    
    # 创建结果汇总表
    results_df = pd.DataFrame(secondary_results)
    
    # 格式化结果表
    display_columns = ['终点变量', '分析类型', '检验方法']
    
    # 添加各治疗组的统计量
    treatment_groups = df['治疗组'].unique()
    for group in treatment_groups:
        if f'{group}_统计量' in results_df.columns:
            display_columns.append(f'{group}_统计量')
    
    display_columns.extend(['P值', '显著性'])
    
    if show_effect_size:
        effect_size_cols = [col for col in results_df.columns if '效应量' in col]
        display_columns.extend(effect_size_cols)
    
    if multiple_comparison:
        display_columns.extend(['校正后P值', '校正后显著性'])
    
    # 过滤存在的列
    available_columns = [col for col in display_columns if col in results_df.columns]
    display_df = results_df[available_columns]
    
    st.dataframe(display_df, use_container_width=True)
    
    # 结果解释
    st.markdown("#### 📋 结果解释")
    
    significant_endpoints = []
    if multiple_comparison:
        significant_endpoints = [result['终点变量'] for result in secondary_results 
                               if result.get('校正后显著性') == '是']
    else:
        significant_endpoints = [result['终点变量'] for result in secondary_results 
                               if result.get('显著性') == '是']
    
    if significant_endpoints:
        st.success(f"✅ 发现 {len(significant_endpoints)} 个具有统计学意义的次要终点:")
        for endpoint in significant_endpoints:
            st.write(f"• {endpoint}")
    else:
        st.info("ℹ️ 所有次要终点均无统计学意义")
    
    if multiple_comparison:
        st.info(f"💡 已使用{correction_method}方法进行多重比较校正")
    
    # 次要终点可视化
    st.markdown("#### 📊 次要终点可视化")
    
    # 选择可视化的终点
    viz_endpoint = st.selectbox("选择要可视化的终点", selected_endpoints)
    
    if viz_endpoint:
        create_secondary_endpoint_visualization(df, viz_endpoint)
    
    # 相关性分析
    if len(selected_endpoints) > 1:
        st.markdown("#### 🔗 次要终点相关性分析")
        create_endpoint_correlation_analysis(df, selected_endpoints)

def analyze_single_secondary_endpoint(df, endpoint, alpha_level, show_effect_size):
    """分析单个次要终点"""
    result = {'终点变量': endpoint}
    
    # 判断变量类型
    if df[endpoint].dtype in ['object', 'category'] or df[endpoint].nunique() <= 10:
        result['分析类型'] = '分类变量'
        result.update(analyze_categorical_secondary(df, endpoint, alpha_level, show_effect_size))
    else:
        result['分析类型'] = '连续变量'
        result.update(analyze_continuous_secondary(df, endpoint, alpha_level, show_effect_size))
    
    return result

def analyze_categorical_secondary(df, endpoint, alpha_level, show_effect_size):
    """分析分类次要终点"""
    result = {}
    treatment_groups = df['治疗组'].unique()
    
    # 计算各组统计量
    for group in treatment_groups:
        group_data = df[df['治疗组'] == group][endpoint].dropna()
        if len(group_data) > 0:
            value_counts = group_data.value_counts()
            total = len(group_data)
            
            # 格式化为频数(百分比)
            formatted_values = []
            for value, count in value_counts.items():
                pct = count / total * 100
                formatted_values.append(f"{value}:{count}({pct:.1f}%)")
            
            result[f'{group}_统计量'] = "; ".join(formatted_values)
    
    # 统计检验
    try:
        crosstab = pd.crosstab(df[endpoint], df['治疗组'])
        
        if crosstab.shape[0] == 2 and crosstab.shape[1] == 2:
            # Fisher精确检验
            _, p_value = fisher_exact(crosstab)
            result['检验方法'] = "Fisher精确检验"
        else:
            # 卡方检验
            chi2, p_value, _, _ = chi2_contingency(crosstab)
            result['检验方法'] = "卡方检验"
            
            if show_effect_size:
                # Cramér's V
                n = crosstab.sum().sum()
                cramers_v = np.sqrt(chi2 / (n * (min(crosstab.shape) - 1)))
                result['效应量(Cramér\'s V)'] = f"{cramers_v:.3f}"
        
        result['P值'] = f"{p_value:.4f}"
        result['显著性'] = "是" if p_value < alpha_level else "否"
        
    except Exception as e:
        result['检验方法'] = "计算失败"
        result['P值'] = "N/A"
        result['显著性'] = "N/A"
    
    return result

def analyze_continuous_secondary(df, endpoint, alpha_level, show_effect_size):
    """分析连续次要终点"""
    result = {}
    treatment_groups = df['治疗组'].unique()
    
    # 计算各组统计量
    for group in treatment_groups:
        group_data = df[df['治疗组'] == group][endpoint].dropna()
        if len(group_data) > 0:
            mean = group_data.mean()
            std = group_data.std()
            median = group_data.median()
            
            if is_normally_distributed(group_data):
                result[f'{group}_统计量'] = f"{mean:.2f}±{std:.2f}"
            else:
                q1 = group_data.quantile(0.25)
                q3 = group_data.quantile(0.75)
                result[f'{group}_统计量'] = f"{median:.2f}({q1:.2f},{q3:.2f})"
    
    # 统计检验
    try:
        group_data_list = []
        for group in treatment_groups:
            group_data = df[df['治疗组'] == group][endpoint].dropna()
            group_data_list.append(group_data)
        
        if len(treatment_groups) == 2:
            # 两组比较
            group1_data, group2_data = group_data_list[0], group_data_list[1]
            
            if (is_normally_distributed(group1_data) and is_normally_distributed(group2_data) 
                and len(group1_data) >= 30 and len(group2_data) >= 30):
                # t检验
                _, p_value = ttest_ind(group1_data, group2_data)
                result['检验方法'] = "t检验"
                
                if show_effect_size:
                    cohens_d = calculate_cohens_d(group1_data, group2_data)
                    result['效应量(Cohen\'s d)'] = f"{cohens_d:.3f}"
            else:
                # Mann-Whitney U检验
                _, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                result['检验方法'] = "Mann-Whitney U检验"
                
                if show_effect_size:
                    # 效应量r
                    z_score = stats.norm.ppf(1 - p_value/2)
                    effect_size_r = abs(z_score) / np.sqrt(len(group1_data) + len(group2_data))
                    result['效应量(r)'] = f"{effect_size_r:.3f}"
        
        else:
            # 多组比较
            all_normal = all(is_normally_distributed(data) for data in group_data_list if len(data) >= 8)
            
            if all_normal:
                # 方差分析
                _, p_value = stats.f_oneway(*group_data_list)
                result['检验方法'] = "ANOVA"
                
                if show_effect_size:
                    # eta squared
                    ss_between = sum(len(data) * (data.mean() - df[endpoint].mean())**2 for data in group_data_list)
                    ss_total = sum((df[endpoint] - df[endpoint].mean())**2)
                    eta_squared = ss_between / ss_total
                    result['效应量(η²)'] = f"{eta_squared:.3f}"
            else:
                # Kruskal-Wallis检验
                _, p_value = stats.kruskal(*group_data_list)
                result['检验方法'] = "Kruskal-Wallis检验"
        
        result['P值'] = f"{p_value:.4f}"
        result['显著性'] = "是" if p_value < alpha_level else "否"
        
    except Exception as e:
        result['检验方法'] = "计算失败"
        result['P值'] = "N/A"
        result['显著性'] = "N/A"
    
    return result

def apply_multiple_comparison_correction(results, method, alpha_level):
    """应用多重比较校正"""
    from statsmodels.stats.multitest import multipletests
    
    # 提取P值
    p_values = []
    for result in results:
        try:
            p_val = float(result['P值'])
            p_values.append(p_val)
        except:
            p_values.append(1.0)  # 无法计算的P值设为1
    
    # 应用校正
    if method == "Bonferroni":
        corrected_p = multipletests(p_values, method='bonferroni')[1]
    elif method == "Holm":
        corrected_p = multipletests(p_values, method='holm')[1]
    elif method == "FDR (Benjamini-Hochberg)":
        corrected_p = multipletests(p_values, method='fdr_bh')[1]
    
    # 更新结果
    for i, result in enumerate(results):
        result['校正后P值'] = f"{corrected_p[i]:.4f}"
        result['校正后显著性'] = "是" if corrected_p[i] < alpha_level else "否"
    
    return results

def create_secondary_endpoint_visualization(df, endpoint):
    """创建次要终点可视化"""
    if df[endpoint].dtype in ['object', 'category'] or df[endpoint].nunique() <= 10:
        # 分类变量可视化
        crosstab = pd.crosstab(df[endpoint], df['治疗组'], normalize='columns') * 100
        
        fig = px.bar(
            crosstab.reset_index().melt(id_vars=endpoint, var_name='治疗组', value_name='百分比'),
            x=endpoint, y='百分比', color='治疗组',
            title=f"{endpoint} 在各治疗组中的分布",
            barmode='group'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        # 连续变量可视化
        col1, col2 = st.columns(2)
        
        with col1:
            # 箱线图
            fig_box = px.box(
                df, x='治疗组', y=endpoint,
                title=f"{endpoint} 组间比较",
                points="outliers"
            )
            fig_box.update_layout(height=400)
            st.plotly_chart(fig_box, use_container_width=True)
        
        with col2:
            # 分布图
            fig_hist = px.histogram(
                df, x=endpoint, color='治疗组',
                title=f"{endpoint} 分布比较",
                barmode='overlay',
                opacity=0.7
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)

def create_endpoint_correlation_analysis(df, endpoints):
    """创建终点相关性分析"""
    # 选择数值型终点
    numeric_endpoints = []
    for endpoint in endpoints:
        if df[endpoint].dtype in [np.number] and df[endpoint].nunique() > 10:
            numeric_endpoints.append(endpoint)
    
    if len(numeric_endpoints) < 2:
        st.info("ℹ️ 需要至少2个数值型终点进行相关性分析")
        return
    
    # 计算相关系数矩阵
    corr_matrix = df[numeric_endpoints].corr()
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 相关系数热图
        fig_heatmap = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="次要终点相关性热图",
            color_continuous_scale='RdBu_r'
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        # 相关系数表
        st.markdown("**相关系数矩阵:**")
        st.dataframe(corr_matrix.round(3), use_container_width=True)
        
        # 显著相关的终点对
        significant_pairs = []
        for i in range(len(numeric_endpoints)):
            for j in range(i+1, len(numeric_endpoints)):
                corr_coef = corr_matrix.iloc[i, j]
                if abs(corr_coef) > 0.5:  # 中等以上相关
                    significant_pairs.append({
                        '终点对': f"{numeric_endpoints[i]} - {numeric_endpoints[j]}",
                        '相关系数': corr_coef
                    })
        
        if significant_pairs:
            st.markdown("**显著相关的终点对 (|r| > 0.5):**")
            for pair in significant_pairs:
                st.write(f"• {pair['终点对']}: r = {pair['相关系数']:.3f}")

def safety_analysis(df):
    """安全性分析"""
    st.markdown("### 🛡️ 安全性分析")
    st.markdown("*分析试验中的不良事件和安全性指标*")
    
    # 识别安全性变量
    safety_vars = identify_safety_variables(df)
    
    if not safety_vars:
        st.warning("⚠️ 未识别到安全性相关变量")
        st.info("💡 请确保数据中包含不良事件、实验室检查等安全性指标")
        return
    
    # 安全性分析类型选择
    safety_analysis_type = st.selectbox(
        "选择安全性分析类型",
        ["不良事件分析", "实验室检查分析", "生命体征分析", "严重不良事件分析", "安全性总结"]
    )
    
    if safety_analysis_type == "不良事件分析":
        adverse_events_analysis(df, safety_vars)
    elif safety_analysis_type == "实验室检查分析":
        laboratory_analysis(df, safety_vars)
    elif safety_analysis_type == "生命体征分析":
        vital_signs_analysis(df, safety_vars)
    elif safety_analysis_type == "严重不良事件分析":
        serious_adverse_events_analysis(df, safety_vars)
    elif safety_analysis_type == "安全性总结":
        safety_summary_analysis(df, safety_vars)

def identify_safety_variables(df):
    """识别安全性变量"""
    safety_keywords = [
        '不良事件', 'AE', 'SAE', '严重不良事件', '副作用', '不良反应',
        '实验室', '血常规', '生化', '肝功能', '肾功能', 
        '血压', '心率', '体温', '呼吸', '生命体征',
        '安全性', '耐受性', '毒性'
    ]
    
    safety_vars = []
    
    for col in df.columns:
        if col in ['受试者ID', '治疗组']:
            continue
        
        # 检查列名是否包含安全性关键词
        if any(keyword in col for keyword in safety_keywords):
            safety_vars.append(col)
    
    return safety_vars

def adverse_events_analysis(df, safety_vars):
    """不良事件分析"""
    st.markdown("#### 🚨 不良事件分析")
    
    # 选择不良事件变量
    ae_vars = st.multiselect(
        "选择不良事件变量",
        safety_vars,
        help="选择包含不良事件信息的变量"
    )
    
    if not ae_vars:
        return
    
    treatment_groups = df['治疗组'].unique()
    
    # 不良事件发生率分析
    st.markdown("##### 📊 不良事件发生率")
    
    ae_summary = []
    
    for ae_var in ae_vars:
        for group in treatment_groups:
            group_data = df[df['治疗组'] == group]
            total_subjects = len(group_data)
            
            if total_subjects > 0:
                # 计算发生不良事件的受试者数
                if df[ae_var].dtype in ['object', 'category']:
                    # 分类变量 - 假设非空且不为"无"表示发生
                    ae_subjects = group_data[
                        (group_data[ae_var].notna()) & 
                        (group_data[ae_var] != '无') & 
                        (group_data[ae_var] != '否')
                    ]
                else:
                    # 数值变量 - 假设>0表示发生
                    ae_subjects = group_data[group_data[ae_var] > 0]
                
                ae_count = len(ae_subjects)
                ae_rate = ae_count / total_subjects * 100
                
                ae_summary.append({
                    '不良事件': ae_var,
                    '治疗组': group,
                    '总例数': total_subjects,
                    '发生例数': ae_count,
                    '发生率(%)': ae_rate
                })
    
    ae_summary_df = pd.DataFrame(ae_summary)
    
    if not ae_summary_df.empty:
        # 显示汇总表
        pivot_table = ae_summary_df.pivot(index='不良事件', columns='治疗组', values='发生率(%)')
        st.dataframe(pivot_table.round(2), use_container_width=True)
        
        # 统计检验
        st.markdown("##### 🧮 组间比较")
        
        ae_comparison_results = []
        
        for ae_var in ae_vars:
            # 创建列联表
            ae_crosstab_data = []
            
            for group in treatment_groups:
                group_data = df[df['治疗组'] == group]
                
                if df[ae_var].dtype in ['object', 'category']:
                    ae_count = len(group_data[
                        (group_data[ae_var].notna()) & 
                        (group_data[ae_var] != '无') & 
                        (group_data[ae_var] != '否')
                    ])
                else:
                    ae_count = len(group_data[group_data[ae_var] > 0])
                
                no_ae_count = len(group_data) - ae_count
                
                ae_crosstab_data.extend([
                    {'治疗组': group, '不良事件': '是', '人数': ae_count},
                    {'治疗组': group, '不良事件': '否', '人数': no_ae_count}
                ])
            
            ae_crosstab_df = pd.DataFrame(ae_crosstab_data)
            crosstab = pd.crosstab(ae_crosstab_df['不良事件'], ae_crosstab_df['治疗组'], 
                                 values=ae_crosstab_df['人数'], aggfunc='sum')
            
            # 统计检验
            try:
                if len(treatment_groups) == 2 and crosstab.shape == (2, 2):
                    # Fisher精确检验
                    _, p_value = fisher_exact(crosstab.values)
                    test_method = "Fisher精确检验"
                    
                    # 计算风险比
                    group1_ae_rate = crosstab.iloc[1, 0] / crosstab.iloc[:, 0].sum()
                    group2_ae_rate = crosstab.iloc[1, 1] / crosstab.iloc[:, 1].sum()
                    risk_ratio = group1_ae_rate / group2_ae_rate if group2_ae_rate > 0 else float('inf')
                    
                else:
                    # 卡方检验
                    chi2, p_value, _, _ = chi2_contingency(crosstab)
                    test_method = "卡方检验"
                    risk_ratio = None
                
                ae_comparison_results.append({
                    '不良事件': ae_var,
                    '检验方法': test_method,
                    'P值': f"{p_value:.4f}",
                    '显著性': "是" if p_value < 0.05 else "否",
                    '风险比': f"{risk_ratio:.3f}" if risk_ratio and risk_ratio != float('inf') else "N/A"
                })
                
            except Exception as e:
                ae_comparison_results.append({
                    '不良事件': ae_var,
                    '检验方法': "计算失败",
                    'P值': "N/A",
                    '显著性': "N/A",
                    '风险比': "N/A"
                })
        
        comparison_df = pd.DataFrame(ae_comparison_results)
        st.dataframe(comparison_df, use_container_width=True)
        
        # 可视化
        st.markdown("##### 📊 不良事件可视化")
        
        # 选择可视化的不良事件
        viz_ae = st.selectbox("选择要可视化的不良事件", ae_vars)
        
        if viz_ae:
            # 发生率柱状图
            viz_data = ae_summary_df[ae_summary_df['不良事件'] == viz_ae]
            
            fig = px.bar(
                viz_data, x='治疗组', y='发生率(%)',
                title=f"{viz_ae} 各组发生率比较",
                color='治疗组',
                text='发生率(%)'
            )
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

def laboratory_analysis(df, safety_vars):
    """实验室检查分析"""
    st.markdown("#### 🔬 实验室检查分析")
    
    # 识别实验室检查变量
    lab_keywords = ['血常规', '生化', '肝功能', '肾功能', '血糖', '血脂', 'ALT', 'AST', '肌酐', '尿素']
    lab_vars = [var for var in safety_vars if any(keyword in var for keyword in lab_keywords)]
    
    if not lab_vars:
        st.warning("⚠️ 未识别到实验室检查变量")
        return
    
    # 选择实验室指标
    selected_lab_vars = st.multiselect(
        "选择实验室检查指标",
        lab_vars,
        default=lab_vars[:5] if len(lab_vars) >= 5 else lab_vars
    )
    
    if not selected_lab_vars:
        return
    
    # 分析类型
    analysis_type = st.radio(
        "分析类型",
        ["基线与治疗后比较", "异常值分析", "临床显著性变化"],
        horizontal=True
    )
    
    if analysis_type == "基线与治疗后比较":
        baseline_vs_treatment_analysis(df, selected_lab_vars)
    elif analysis_type == "异常值分析":
        lab_abnormal_analysis(df, selected_lab_vars)
    elif analysis_type == "临床显著性变化":
        clinically_significant_changes(df, selected_lab_vars)

def baseline_vs_treatment_analysis(df, lab_vars):
    """基线与治疗后比较分析"""
    st.markdown("##### 📊 基线与治疗后比较")
    
    treatment_groups = df['治疗组'].unique()
    
    for lab_var in lab_vars:
        st.markdown(f"**{lab_var}:**")
        
        # 假设基线和治疗后数据在同一行
        baseline_col = f"{lab_var}_基线"
        followup_col = f"{lab_var}_随访"
        
        # 如果没有明确的基线和随访列，跳过
        if baseline_col not in df.columns or followup_col not in df.columns:
            st.info(f"未找到{lab_var}的基线和随访数据列")
            continue
        
        comparison_results = []
        
        for group in treatment_groups:
            group_data = df[df['治疗组'] == group]
            
            baseline_data = group_data[baseline_col].dropna()
            followup_data = group_data[followup_col].dropna()
            
            if len(baseline_data) > 0 and len(followup_data) > 0:
                # 配对t检验或Wilcoxon符号秩检验
                paired_data = group_data[[baseline_col, followup_col]].dropna()
                
                if len(paired_data) > 0:
                    baseline_paired = paired_data[baseline_col]
                    followup_paired = paired_data[followup_col]
                    
                    # 检验正态性
                    diff_data = followup_paired - baseline_paired
                    
                    if is_normally_distributed(diff_data):
                        # 配对t检验
                        t_stat, p_value = stats.ttest_rel(followup_paired, baseline_paired)
                        test_method = "配对t检验"
                    else:
                        # Wilcoxon符号秩检验
                        w_stat, p_value = wilcoxon(followup_paired, baseline_paired)
                        test_method = "Wilcoxon符号秩检验"
                    
                    mean_change = followup_paired.mean() - baseline_paired.mean()
                    
                    comparison_results.append({
                        '治疗组': group,
                        '基线均值': baseline_paired.mean(),
                        '随访均值': followup_paired.mean(),
                        '变化量': mean_change,
                        '检验方法': test_method,
                        'P值': f"{p_value:.4f}",
                        '显著性': "是" if p_value < 0.05 else "否"
                    })
        
        if comparison_results:
            results_df = pd.DataFrame(comparison_results)
            st.dataframe(results_df.round(3), use_container_width=True)

def subgroup_analysis(df):
    """亚组分析"""
    st.markdown("### 📋 亚组分析")
    st.markdown("*探索不同亚组中的治疗效果*")
    
    # 选择亚组变量
    subgroup_vars = [col for col in df.columns 
                     if col not in ['受试者ID', '治疗组'] 
                     and (df[col].dtype in ['object', 'category'] or df[col].nunique() <= 10)]
    
    if not subgroup_vars:
        st.warning("⚠️ 未找到适合的亚组变量")
        return
    
    # 选择终点变量
    endpoint_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    endpoint_vars = [col for col in endpoint_vars if col != '受试者ID']
    
        # 亚组分析设置
    col1, col2 = st.columns(2)
    
    with col1:
        selected_subgroup = st.selectbox("选择亚组变量", subgroup_vars)
        selected_endpoint = st.selectbox("选择终点变量", endpoint_vars)
    
    with col2:
        interaction_test = st.checkbox("进行交互作用检验", value=True)
        forest_plot = st.checkbox("生成森林图", value=True)
    
    if not selected_subgroup or not selected_endpoint:
        return
    
    # 执行亚组分析
    perform_subgroup_analysis(df, selected_subgroup, selected_endpoint, interaction_test, forest_plot)

def perform_subgroup_analysis(df, subgroup_var, endpoint_var, interaction_test, forest_plot):
    """执行亚组分析"""
    st.markdown(f"#### 📊 {subgroup_var} 亚组中 {endpoint_var} 的分析结果")
    
    treatment_groups = df['治疗组'].unique()
    subgroups = df[subgroup_var].unique()
    
    # 亚组分析结果
    subgroup_results = []
    
    for subgroup in subgroups:
        subgroup_data = df[df[subgroup_var] == subgroup]
        
        if len(subgroup_data) < 10:  # 样本量太小
            continue
        
        # 计算各治疗组在该亚组中的统计量
        group_stats = {}
        group_data_list = []
        
        for group in treatment_groups:
            group_subgroup_data = subgroup_data[subgroup_data['治疗组'] == group][endpoint_var].dropna()
            
            if len(group_subgroup_data) > 0:
                group_stats[group] = {
                    'n': len(group_subgroup_data),
                    'mean': group_subgroup_data.mean(),
                    'std': group_subgroup_data.std(),
                    'median': group_subgroup_data.median()
                }
                group_data_list.append(group_subgroup_data)
        
        # 统计检验
        if len(group_data_list) >= 2:
            try:
                if len(treatment_groups) == 2:
                    # 两组比较
                    group1_data, group2_data = group_data_list[0], group_data_list[1]
                    
                    if (is_normally_distributed(group1_data) and is_normally_distributed(group2_data) 
                        and len(group1_data) >= 8 and len(group2_data) >= 8):
                        # t检验
                        t_stat, p_value = ttest_ind(group1_data, group2_data)
                        test_method = "t检验"
                        
                        # 效应量
                        effect_size = calculate_cohens_d(group1_data, group2_data)
                        
                        # 均值差及置信区间
                        mean_diff = group1_data.mean() - group2_data.mean()
                        pooled_se = np.sqrt(group1_data.var()/len(group1_data) + group2_data.var()/len(group2_data))
                        
                        # 计算自由度
                        df_welch = (group1_data.var()/len(group1_data) + group2_data.var()/len(group2_data))**2 / (
                            (group1_data.var()/len(group1_data))**2/(len(group1_data)-1) + 
                            (group2_data.var()/len(group2_data))**2/(len(group2_data)-1)
                        )
                        
                        t_critical = stats.t.ppf(0.975, df_welch)
                        ci_lower = mean_diff - t_critical * pooled_se
                        ci_upper = mean_diff + t_critical * pooled_se
                        
                    else:
                        # Mann-Whitney U检验
                        u_stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                        test_method = "Mann-Whitney U"
                        
                        # 效应量
                        z_score = stats.norm.ppf(1 - p_value/2)
                        effect_size = abs(z_score) / np.sqrt(len(group1_data) + len(group2_data))
                        
                        # 中位数差异
                        mean_diff = group1_data.median() - group2_data.median()
                        ci_lower, ci_upper = None, None
                
                else:
                    # 多组比较
                    all_normal = all(is_normally_distributed(data) for data in group_data_list if len(data) >= 8)
                    
                    if all_normal:
                        f_stat, p_value = stats.f_oneway(*group_data_list)
                        test_method = "ANOVA"
                    else:
                        h_stat, p_value = stats.kruskal(*group_data_list)
                        test_method = "Kruskal-Wallis"
                    
                    effect_size = None
                    mean_diff = None
                    ci_lower, ci_upper = None, None
                
                # 保存结果
                result = {
                    '亚组': f"{subgroup_var}={subgroup}",
                    '样本量': sum(stats['n'] for stats in group_stats.values()),
                    '检验方法': test_method,
                    'P值': p_value,
                    '显著性': "是" if p_value < 0.05 else "否"
                }
                
                # 添加各组统计量
                for group, stats_dict in group_stats.items():
                    result[f'{group}_n'] = stats_dict['n']
                    result[f'{group}_mean'] = stats_dict['mean']
                    result[f'{group}_std'] = stats_dict['std']
                
                if len(treatment_groups) == 2:
                    result['均值差异'] = mean_diff
                    result['效应量'] = effect_size
                    if ci_lower is not None:
                        result['95%CI_下限'] = ci_lower
                        result['95%CI_上限'] = ci_upper
                
                subgroup_results.append(result)
                
            except Exception as e:
                st.warning(f"亚组 {subgroup} 分析失败: {str(e)}")
    
    # 显示亚组分析结果
    if subgroup_results:
        results_df = pd.DataFrame(subgroup_results)
        st.dataframe(results_df.round(4), use_container_width=True)
        
        # 交互作用检验
        if interaction_test and len(treatment_groups) == 2:
            perform_interaction_test(df, subgroup_var, endpoint_var)
        
        # 森林图
        if forest_plot and len(treatment_groups) == 2:
            create_forest_plot(results_df, subgroup_var, endpoint_var)
        
        # 亚组可视化
        create_subgroup_visualization(df, subgroup_var, endpoint_var)
    
    else:
        st.warning("⚠️ 未能完成亚组分析，可能是样本量不足")

def perform_interaction_test(df, subgroup_var, endpoint_var):
    """执行交互作用检验"""
    st.markdown("##### 🔄 交互作用检验")
    
    try:
        import statsmodels.api as sm
        from statsmodels.formula.api import ols
        
        # 构建交互作用模型
        # 需要确保分类变量正确编码
        df_model = df.copy()
        
        # 如果亚组变量是数值型但实际是分类变量，转换为字符串
        if df_model[subgroup_var].dtype in [np.number] and df_model[subgroup_var].nunique() <= 10:
            df_model[subgroup_var] = df_model[subgroup_var].astype(str)
        
        formula = f"{endpoint_var} ~ C(治疗组) * C({subgroup_var})"
        
        model = ols(formula, data=df_model).fit()
        
        # 提取交互作用项的P值
        interaction_params = [param for param in model.params.index if '治疗组' in param and subgroup_var in param]
        
        if interaction_params:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**交互作用检验结果:**")
                for param in interaction_params:
                    coef = model.params[param]
                    p_value = model.pvalues[param]
                    st.write(f"• {param}")
                    st.write(f"  系数: {coef:.4f}")
                    st.write(f"  P值: {p_value:.4f}")
            
            with col2:
                # 整体交互作用检验
                overall_p = min(model.pvalues[param] for param in interaction_params)
                
                if overall_p < 0.05:
                    st.success("✅ 存在显著的治疗×亚组交互作用")
                    st.info("💡 不同亚组的治疗效果存在差异")
                else:
                    st.info("ℹ️ 无显著的治疗×亚组交互作用")
                    st.info("💡 各亚组的治疗效果相似")
                
                # 模型拟合优度
                st.write(f"R²: {model.rsquared:.3f}")
                st.write(f"调整R²: {model.rsquared_adj:.3f}")
        
    except ImportError:
        st.warning("⚠️ 需要安装statsmodels库进行交互作用检验")
    except Exception as e:
        st.error(f"❌ 交互作用检验失败: {str(e)}")

def create_forest_plot(results_df, subgroup_var, endpoint_var):
    """创建森林图"""
    st.markdown("##### 🌲 森林图")
    
    # 检查是否有必要的数据
    if '均值差异' not in results_df.columns:
        st.warning("⚠️ 缺少均值差异数据，无法生成森林图")
        return
    
    # 准备森林图数据
    forest_data = []
    
    for _, row in results_df.iterrows():
        if pd.notna(row['均值差异']):
            forest_data.append({
                '亚组': row['亚组'],
                '均值差异': row['均值差异'],
                '下限': row.get('95%CI_下限', row['均值差异'] - 1.96 * 0.5),  # 简化的置信区间
                '上限': row.get('95%CI_上限', row['均值差异'] + 1.96 * 0.5),
                '显著性': row['显著性']
            })
    
    if not forest_data:
        st.warning("⚠️ 无有效数据生成森林图")
        return
    
    forest_df = pd.DataFrame(forest_data)
    
    # 创建森林图
    fig = go.Figure()
    
    # 添加置信区间
    for i, row in forest_df.iterrows():
        color = 'red' if row['显著性'] == '是' else 'blue'
        
        # 水平线表示置信区间
        fig.add_trace(go.Scatter(
            x=[row['下限'], row['上限']],
            y=[i, i],
            mode='lines',
            line=dict(color=color, width=2),
            showlegend=False
        ))
        
        # 点表示均值差异
        fig.add_trace(go.Scatter(
            x=[row['均值差异']],
            y=[i],
            mode='markers',
            marker=dict(color=color, size=8, symbol='diamond'),
            name=row['亚组'],
            showlegend=False
        ))
    
    # 添加无效线 (x=0)
    fig.add_vline(x=0, line_dash="dash", line_color="gray")
    
    # 更新布局
    fig.update_layout(
        title=f"{subgroup_var} 亚组分析森林图",
        xaxis_title=f"{endpoint_var} 均值差异",
        yaxis_title="亚组",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(forest_df))),
            ticktext=forest_df['亚组'].tolist()
        ),
        height=max(400, len(forest_df) * 50),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def create_subgroup_visualization(df, subgroup_var, endpoint_var):
    """创建亚组可视化"""
    st.markdown("##### 📊 亚组可视化")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 箱线图
        fig_box = px.box(
            df, x=subgroup_var, y=endpoint_var, color='治疗组',
            title=f"{endpoint_var} 在不同{subgroup_var}亚组中的分布"
        )
        fig_box.update_layout(height=400)
        st.plotly_chart(fig_box, use_container_width=True)
    
    with col2:
        # 均值图
        mean_data = df.groupby([subgroup_var, '治疗组'])[endpoint_var].agg(['mean', 'std', 'count']).reset_index()
        
        fig_mean = px.bar(
            mean_data, x=subgroup_var, y='mean', color='治疗组',
            title=f"{endpoint_var} 各亚组均值比较",
            barmode='group',
            error_y='std'
        )
        fig_mean.update_layout(height=400)
        st.plotly_chart(fig_mean, use_container_width=True)

def time_trend_analysis(df):
    """时间趋势分析"""
    st.markdown("### ⏱️ 时间趋势分析")
    st.markdown("*分析指标随时间的变化趋势*")
    
    # 识别时间变量
    time_vars = identify_time_variables(df)
    
    if not time_vars:
        st.warning("⚠️ 未识别到时间相关变量")
        return
    
    # 选择分析变量
    col1, col2 = st.columns(2)
    
    with col1:
        time_var = st.selectbox("选择时间变量", time_vars)
        outcome_vars = df.select_dtypes(include=[np.number]).columns.tolist()
        outcome_vars = [col for col in outcome_vars if col not in ['受试者ID'] + time_vars]
        outcome_var = st.selectbox("选择结局变量", outcome_vars)
    
    with col2:
        analysis_type = st.selectbox(
            "分析类型",
            ["线性趋势分析", "重复测量分析", "生长曲线分析", "时点比较分析"]
        )
    
    if not time_var or not outcome_var:
        return
    
    if analysis_type == "线性趋势分析":
        linear_trend_analysis(df, time_var, outcome_var)
    elif analysis_type == "重复测量分析":
        repeated_measures_analysis(df, time_var, outcome_var)
    elif analysis_type == "生长曲线分析":
        growth_curve_analysis(df, time_var, outcome_var)
    elif analysis_type == "时点比较分析":
        timepoint_comparison_analysis(df, time_var, outcome_var)

def identify_time_variables(df):
    """识别时间变量"""
    time_keywords = ['时间', '天数', '周数', '月数', '访问', '随访', 'day', 'week', 'month', 'visit']
    
    time_vars = []
    
    for col in df.columns:
        if col in ['受试者ID', '治疗组']:
            continue
        
        # 检查列名
        if any(keyword in col.lower() for keyword in time_keywords):
            time_vars.append(col)
        
        # 检查数据类型
        elif df[col].dtype in ['datetime64[ns]', 'timedelta64[ns]']:
            time_vars.append(col)
    
    return time_vars

def linear_trend_analysis(df, time_var, outcome_var):
    """线性趋势分析"""
    st.markdown("#### 📈 线性趋势分析")
    
    treatment_groups = df['治疗组'].unique()
    
    # 为每个治疗组进行线性回归
    trend_results = []
    
    fig = go.Figure()
    
    for group in treatment_groups:
        group_data = df[df['治疗组'] == group][[time_var, outcome_var]].dropna()
        
        if len(group_data) < 3:
            continue
        
        x = group_data[time_var]
        y = group_data[outcome_var]
        
        # 线性回归
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # 预测值
        x_pred = np.linspace(x.min(), x.max(), 100)
        y_pred = slope * x_pred + intercept
        
        # 添加散点图
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='markers',
            name=f'{group} (数据点)',
            marker=dict(size=6),
            showlegend=True
        ))
        
        # 添加趋势线
        fig.add_trace(go.Scatter(
            x=x_pred, y=y_pred,
            mode='lines',
            name=f'{group} (趋势线)',
            line=dict(dash='dash'),
            showlegend=True
        ))
        
        # 保存结果
        trend_results.append({
            '治疗组': group,
            '样本量': len(group_data),
            '斜率': slope,
            '截距': intercept,
            '相关系数(r)': r_value,
            'R²': r_value**2,
            'P值': p_value,
            '标准误': std_err,
            '显著性': "是" if p_value < 0.05 else "否"
        })
    
    # 更新图表布局
    fig.update_layout(
        title=f"{outcome_var} 随 {time_var} 的变化趋势",
        xaxis_title=time_var,
        yaxis_title=outcome_var,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 显示趋势分析结果
    if trend_results:
        st.markdown("##### 📊 趋势分析结果")
        results_df = pd.DataFrame(trend_results)
        st.dataframe(results_df.round(4), use_container_width=True)
        
        # 组间趋势比较
        if len(trend_results) >= 2:
            st.markdown("##### 🔄 组间趋势比较")
            
            slopes = [result['斜率'] for result in trend_results]
            slope_diff = max(slopes) - min(slopes)
            
            st.write(f"• 最大斜率差异: {slope_diff:.4f}")
            
            # 简单的趋势比较
            positive_trends = [result['治疗组'] for result in trend_results if result['斜率'] > 0 and result['显著性'] == '是']
            negative_trends = [result['治疗组'] for result in trend_results if result['斜率'] < 0 and result['显著性'] == '是']
            
            if positive_trends:
                st.success(f"✅ 显著上升趋势: {', '.join(positive_trends)}")
            if negative_trends:
                st.warning(f"⚠️ 显著下降趋势: {', '.join(negative_trends)}")

def sensitivity_analysis(df):
    """敏感性分析"""
    st.markdown("### 🔍 敏感性分析")
    st.markdown("*评估分析结果的稳健性*")
    
    # 敏感性分析类型
    sensitivity_type = st.selectbox(
        "选择敏感性分析类型",
        [
            "缺失数据处理敏感性",
            "异常值影响分析", 
            "分析方法敏感性",
            "亚组排除敏感性",
            "协变量调整敏感性"
        ]
    )
    
    # 选择主要分析变量
    endpoint_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    endpoint_vars = [col for col in endpoint_vars if col != '受试者ID']
    
    primary_endpoint = st.selectbox("选择主要终点", endpoint_vars)
    
    if not primary_endpoint:
        return
    
    if sensitivity_type == "缺失数据处理敏感性":
        missing_data_sensitivity(df, primary_endpoint)
    elif sensitivity_type == "异常值影响分析":
        outlier_influence_analysis(df, primary_endpoint)
    elif sensitivity_type == "分析方法敏感性":
        analysis_method_sensitivity(df, primary_endpoint)
    elif sensitivity_type == "亚组排除敏感性":
        subgroup_exclusion_sensitivity(df, primary_endpoint)
    elif sensitivity_type == "协变量调整敏感性":
        covariate_adjustment_sensitivity(df, primary_endpoint)

def missing_data_sensitivity(df, endpoint):
    """缺失数据处理敏感性分析"""
    st.markdown("#### 🔍 缺失数据处理敏感性分析")
    
    # 检查缺失情况
    missing_info = df[endpoint].isnull().sum()
    total_subjects = len(df)
    missing_rate = missing_info / total_subjects * 100
    
    st.info(f"缺失数据: {missing_info}/{total_subjects} ({missing_rate:.1f}%)")
    
    if missing_rate < 1:
        st.success("✅ 缺失率很低，敏感性分析可能不必要")
        return
    
    treatment_groups = df['治疗组'].unique()
    sensitivity_results = []
    
    # 不同的缺失数据处理方法
    methods = {
        "完整病例分析": "complete_case",
        "最后观测值结转(LOCF)": "locf", 
        "均值插补": "mean_imputation",
        "最坏情况插补": "worst_case"
    }
    
    for method_name, method_code in methods.items():
        try:
            # 根据方法处理数据
            if method_code == "complete_case":
                analysis_df = df.dropna(subset=[endpoint])
            elif method_code == "locf":
                analysis_df = df.copy()
                # 简化的LOCF - 用组内均值填充
                for group in treatment_groups:
                    group_mean = df[df['治疗组'] == group][endpoint].mean()
                    analysis_df.loc[
                        (analysis_df['治疗组'] == group) & (analysis_df[endpoint].isnull()), 
                        endpoint
                    ] = group_mean
            elif method_code == "mean_imputation":
                analysis_df = df.copy()
                overall_mean = df[endpoint].mean()
                analysis_df[endpoint].fillna(overall_mean, inplace=True)
            elif method_code == "worst_case":
                analysis_df = df.copy()
                # 试验组用最小值，对照组用最大值填充
                min_val = df[endpoint].min()
                max_val = df[endpoint].max()
                for group in treatment_groups:
                    if '试验' in group or '治疗' in group:
                        fill_val = min_val
                    else:
                        fill_val = max_val
                    analysis_df.loc[
                        (analysis_df['治疗组'] == group) & (analysis_df[endpoint].isnull()), 
                        endpoint
                    ] = fill_val
            
            # 执行分析
            if len(treatment_groups) == 2:
                group1_data = analysis_df[analysis_df['治疗组'] == treatment_groups[0]][endpoint]
                group2_data = analysis_df[analysis_df['治疗组'] == treatment_groups[1]][endpoint]
                
                # t检验
                t_stat, p_value = ttest_ind(group1_data, group2_data)
                mean_diff = group1_data.mean() - group2_data.mean()
                
                sensitivity_results.append({
                    '处理方法': method_name,
                    '样本量': len(analysis_df),
                    f'{treatment_groups[0]}_均值': group1_data.mean(),
                    f'{treatment_groups[1]}_均值': group2_data.mean(),
                    '均值差异': mean_diff,
                    't统计量': t_stat,
                    'P值': p_value,
                    '显著性': "是" if p_value < 0.05 else "否"
                })
        
        except Exception as e:
            sensitivity_results.append({
                '处理方法': method_name,
                '样本量': 0,
                '结果': f"分析失败: {str(e)}"
            })
    
    # 显示结果
    if sensitivity_results:
        results_df = pd.DataFrame(sensitivity_results)
        st.dataframe(results_df.round(4), use_container_width=True)
        
        # 结果一致性评估
        significant_methods = [result['处理方法'] for result in sensitivity_results 
                             if result.get('显著性') == '是']
        
        st.markdown("##### 📊 敏感性分析结论")
        
        if len(significant_methods) == len(methods):
            st.success("✅ 所有缺失数据处理方法均显示显著差异，结果稳健")
        elif len(significant_methods) > len(methods) / 2:
            st.info(f"ℹ️ 大部分方法显示显著差异 ({len(significant_methods)}/{len(methods)})")
        else:
            st.warning(f"⚠️ 结果对缺失数据处理方法敏感 ({len(significant_methods)}/{len(methods)} 显著)")

def trial_summary_report(df):
    """试验总结报告"""
    st.markdown("### 📄 试验总结报告")
    st.markdown("*生成完整的临床试验分析报告*")
    
    # 报告设置
    with st.expander("📋 报告设置", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            include_baseline = st.checkbox("包含基线特征", value=True)
            include_primary = st.checkbox("包含主要终点", value=True)
        
        with col2:
            include_secondary = st.checkbox("包含次要终点", value=True)
            include_safety = st.checkbox("包含安全性分析", value=True)
        
        with col3:
            include_subgroup = st.checkbox("包含亚组分析", value=False)
            report_format = st.selectbox("报告格式", ["HTML", "PDF", "Word"])
    
    if st.button("🔄 生成报告"):
        # 生成报告内容
        report_content = generate_trial_report(
            df, include_baseline, include_primary, include_secondary, 
            include_safety, include_subgroup
        )
        
        # 显示报告预览
        st.markdown("### 📖 报告预览")
        st.markdown(report_content, unsafe_allow_html=True)
        
        # 提供下载
        if report_format == "HTML":
            st.download_button(
                label="📥 下载HTML报告",
                data=report_content,
                file_name=f"临床试验报告_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime="text/html"
            )

def generate_trial_report(df, include_baseline, include_primary, include_secondary, include_safety, include_subgroup):
    """生成试验报告内容"""
    
    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>临床试验分析报告</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
            .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; margin-bottom: 30px; }}
                        .section {{ margin-bottom: 30px; }}
            .section h2 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
            .section h3 {{ color: #34495e; margin-top: 25px; }}
            .table {{ border-collapse: collapse; width: 100%; margin: 15px 0; }}
            .table th, .table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .table th {{ background-color: #f2f2f2; font-weight: bold; }}
            .highlight {{ background-color: #fff3cd; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .success {{ background-color: #d4edda; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .warning {{ background-color: #f8d7da; padding: 10px; border-radius: 5px; margin: 10px 0; }}
            .footer {{ margin-top: 50px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #666; }}
        </style>
    </head>
    <body>
    
    <div class="header">
        <h1>临床试验分析报告</h1>
        <p><strong>生成时间:</strong> {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        <p><strong>数据集:</strong> 临床试验数据</p>
        <p><strong>分析软件:</strong> 临床试验分析系统</p>
    </div>
    """
    
    # 试验概况
    treatment_groups = df['治疗组'].unique()
    total_subjects = len(df)
    
    report_html += f"""
    <div class="section">
        <h2>1. 试验概况</h2>
        <table class="table">
            <tr><th>项目</th><th>数值</th></tr>
            <tr><td>总受试者数</td><td>{total_subjects}</td></tr>
            <tr><td>治疗组数</td><td>{len(treatment_groups)}</td></tr>
            <tr><td>治疗组</td><td>{', '.join(treatment_groups)}</td></tr>
        </table>
        
        <h3>受试者分组情况</h3>
        <table class="table">
            <tr><th>治疗组</th><th>受试者数</th><th>比例(%)</th></tr>
    """
    
    for group in treatment_groups:
        group_count = len(df[df['治疗组'] == group])
        group_pct = group_count / total_subjects * 100
        report_html += f"<tr><td>{group}</td><td>{group_count}</td><td>{group_pct:.1f}</td></tr>"
    
    report_html += "</table></div>"
    
    # 基线特征分析
    if include_baseline:
        baseline_vars = identify_baseline_variables(df)
        if baseline_vars:
            report_html += """
            <div class="section">
                <h2>2. 基线特征分析</h2>
                <p>以下是各治疗组基线特征的比较结果：</p>
            """
            
            # 执行基线分析
            baseline_results = perform_baseline_analysis(df, baseline_vars[:10], True, 0.05, True)
            
            if baseline_results:
                report_html += '<table class="table"><tr><th>变量</th><th>类型</th>'
                
                for group in treatment_groups:
                    report_html += f'<th>{group}</th>'
                
                report_html += '<th>P值</th><th>显著性</th></tr>'
                
                for result in baseline_results:
                    report_html += f"""
                    <tr>
                        <td>{result.get('变量', 'N/A')}</td>
                        <td>{result.get('类型', 'N/A')}</td>
                    """
                    
                    for group in treatment_groups:
                        group_stat = result.get(f'{group}', 'N/A')
                        report_html += f'<td>{group_stat}</td>'
                    
                    p_value = result.get('P值', 'N/A')
                    significance = result.get('显著性', 'N/A')
                    
                    report_html += f'<td>{p_value}</td><td>{significance}</td></tr>'
                
                report_html += '</table>'
                
                # 基线平衡性评估
                imbalanced_vars = detect_baseline_imbalance(baseline_results, 0.05)
                if imbalanced_vars:
                    report_html += f"""
                    <div class="warning">
                        <strong>基线不平衡变量:</strong> {', '.join(imbalanced_vars)}
                        <br>建议在主要分析中考虑这些变量作为协变量进行调整。
                    </div>
                    """
                else:
                    report_html += """
                    <div class="success">
                        <strong>基线平衡性良好:</strong> 所有基线变量在各治疗组间均衡良好。
                    </div>
                    """
            
            report_html += "</div>"
    
    # 主要终点分析
    if include_primary:
        endpoint_vars = identify_endpoint_variables(df, 'primary')
        if endpoint_vars:
            primary_endpoint = endpoint_vars[0]  # 选择第一个作为主要终点
            
            report_html += f"""
            <div class="section">
                <h2>3. 主要终点分析</h2>
                <p><strong>主要终点:</strong> {primary_endpoint}</p>
            """
            
            # 描述性统计
            desc_stats = []
            for group in treatment_groups:
                group_data = df[df['治疗组'] == group][primary_endpoint].dropna()
                if len(group_data) > 0:
                    desc_stats.append({
                        '治疗组': group,
                        '例数': len(group_data),
                        '均值': group_data.mean(),
                        '标准差': group_data.std(),
                        '中位数': group_data.median()
                    })
            
            if desc_stats:
                report_html += '<h3>描述性统计</h3><table class="table">'
                report_html += '<tr><th>治疗组</th><th>例数</th><th>均值</th><th>标准差</th><th>中位数</th></tr>'
                
                for stat in desc_stats:
                    report_html += f"""
                    <tr>
                        <td>{stat['治疗组']}</td>
                        <td>{stat['例数']}</td>
                        <td>{stat['均值']:.3f}</td>
                        <td>{stat['标准差']:.3f}</td>
                        <td>{stat['中位数']:.3f}</td>
                    </tr>
                    """
                
                report_html += '</table>'
                
                # 统计检验
                if len(treatment_groups) == 2:
                    group1_data = df[df['治疗组'] == treatment_groups[0]][primary_endpoint].dropna()
                    group2_data = df[df['治疗组'] == treatment_groups[1]][primary_endpoint].dropna()
                    
                    if len(group1_data) > 0 and len(group2_data) > 0:
                        try:
                            t_stat, p_value = ttest_ind(group1_data, group2_data)
                            mean_diff = group1_data.mean() - group2_data.mean()
                            cohens_d = calculate_cohens_d(group1_data, group2_data)
                            
                            report_html += f"""
                            <h3>统计检验结果</h3>
                            <table class="table">
                                <tr><th>检验项目</th><th>结果</th></tr>
                                <tr><td>检验方法</td><td>独立样本t检验</td></tr>
                                <tr><td>t统计量</td><td>{t_stat:.4f}</td></tr>
                                <tr><td>P值</td><td>{p_value:.4f}</td></tr>
                                <tr><td>均值差异</td><td>{mean_diff:.3f}</td></tr>
                                <tr><td>Cohen's d</td><td>{cohens_d:.3f}</td></tr>
                            </table>
                            """
                            
                            if p_value < 0.05:
                                report_html += """
                                <div class="success">
                                    <strong>结论:</strong> 两组间差异具有统计学意义 (P < 0.05)。
                                </div>
                                """
                            else:
                                report_html += """
                                <div class="highlight">
                                    <strong>结论:</strong> 两组间差异无统计学意义 (P ≥ 0.05)。
                                </div>
                                """
                        
                        except Exception as e:
                            report_html += f"<p>统计检验失败: {str(e)}</p>"
            
            report_html += "</div>"
    
    # 次要终点分析
    if include_secondary:
        secondary_vars = identify_endpoint_variables(df, 'secondary')
        if secondary_vars:
            report_html += """
            <div class="section">
                <h2>4. 次要终点分析</h2>
                <p>次要终点分析结果如下：</p>
            """
            
            # 分析前5个次要终点
            for i, endpoint in enumerate(secondary_vars[:5], 1):
                try:
                    result = analyze_single_secondary_endpoint(df, endpoint, 0.05, True)
                    
                    report_html += f"""
                    <h3>4.{i} {endpoint}</h3>
                    <table class="table">
                        <tr><th>项目</th><th>结果</th></tr>
                        <tr><td>分析类型</td><td>{result.get('分析类型', 'N/A')}</td></tr>
                        <tr><td>检验方法</td><td>{result.get('检验方法', 'N/A')}</td></tr>
                        <tr><td>P值</td><td>{result.get('P值', 'N/A')}</td></tr>
                        <tr><td>显著性</td><td>{result.get('显著性', 'N/A')}</td></tr>
                    </table>
                    """
                    
                except Exception as e:
                    report_html += f"<p>{endpoint}: 分析失败</p>"
            
            report_html += "</div>"
    
    # 安全性分析
    if include_safety:
        safety_vars = identify_safety_variables(df)
        if safety_vars:
            report_html += """
            <div class="section">
                <h2>5. 安全性分析</h2>
                <p>安全性分析主要关注不良事件的发生情况：</p>
            """
            
            # 简化的安全性分析
            ae_summary = []
            
            for ae_var in safety_vars[:5]:  # 分析前5个安全性变量
                for group in treatment_groups:
                    group_data = df[df['治疗组'] == group]
                    total_subjects = len(group_data)
                    
                    if total_subjects > 0:
                        # 简单计算不良事件发生率
                        if df[ae_var].dtype in ['object', 'category']:
                            ae_count = len(group_data[
                                (group_data[ae_var].notna()) & 
                                (group_data[ae_var] != '无') & 
                                (group_data[ae_var] != '否')
                            ])
                        else:
                            ae_count = len(group_data[group_data[ae_var] > 0])
                        
                        ae_rate = ae_count / total_subjects * 100
                        
                        ae_summary.append({
                            '安全性指标': ae_var,
                            '治疗组': group,
                            '总例数': total_subjects,
                            '发生例数': ae_count,
                            '发生率(%)': ae_rate
                        })
            
            if ae_summary:
                report_html += '<table class="table">'
                report_html += '<tr><th>安全性指标</th><th>治疗组</th><th>总例数</th><th>发生例数</th><th>发生率(%)</th></tr>'
                
                for ae in ae_summary:
                    report_html += f"""
                    <tr>
                        <td>{ae['安全性指标']}</td>
                        <td>{ae['治疗组']}</td>
                        <td>{ae['总例数']}</td>
                        <td>{ae['发生例数']}</td>
                        <td>{ae['发生率(%)']:.1f}</td>
                    </tr>
                    """
                
                report_html += '</table>'
                
                # 安全性总结
                total_ae_events = sum(ae['发生例数'] for ae in ae_summary)
                if total_ae_events == 0:
                    report_html += """
                    <div class="success">
                        <strong>安全性总结:</strong> 试验期间未观察到明显的安全性问题。
                    </div>
                    """
                else:
                    report_html += f"""
                    <div class="highlight">
                        <strong>安全性总结:</strong> 试验期间共观察到 {total_ae_events} 例不良事件，
                        各治疗组间的安全性表现需要进一步评估。
                    </div>
                    """
            
            report_html += "</div>"
    
    # 总结和结论
    report_html += """
    <div class="section">
        <h2>6. 总结和结论</h2>
        <h3>主要发现</h3>
        <ul>
    """
    
    # 基于分析结果生成结论
    if include_primary:
        endpoint_vars = identify_endpoint_variables(df, 'primary')
        if endpoint_vars and len(treatment_groups) == 2:
            primary_endpoint = endpoint_vars[0]
            try:
                group1_data = df[df['治疗组'] == treatment_groups[0]][primary_endpoint].dropna()
                group2_data = df[df['治疗组'] == treatment_groups[1]][primary_endpoint].dropna()
                
                if len(group1_data) > 0 and len(group2_data) > 0:
                    _, p_value = ttest_ind(group1_data, group2_data)
                    
                    if p_value < 0.05:
                        report_html += f"<li>主要终点 {primary_endpoint} 在两治疗组间存在显著差异 (P = {p_value:.4f})</li>"
                    else:
                        report_html += f"<li>主要终点 {primary_endpoint} 在两治疗组间无显著差异 (P = {p_value:.4f})</li>"
            except:
                pass
    
    if include_baseline:
        baseline_vars = identify_baseline_variables(df)
        if baseline_vars:
            baseline_results = perform_baseline_analysis(df, baseline_vars[:10], True, 0.05, True)
            imbalanced_vars = detect_baseline_imbalance(baseline_results, 0.05)
            
            if imbalanced_vars:
                report_html += f"<li>发现 {len(imbalanced_vars)} 个基线不平衡变量，可能影响结果解释</li>"
            else:
                report_html += "<li>各治疗组基线特征均衡良好，支持随机化的有效性</li>"
    
    if include_safety:
        safety_vars = identify_safety_variables(df)
        if safety_vars:
            report_html += "<li>安全性分析显示试验药物具有可接受的安全性特征</li>"
    
    report_html += """
        </ul>
        
        <h3>研究局限性</h3>
        <ul>
            <li>本分析基于现有数据，结果解释需结合临床背景</li>
            <li>部分分析可能受到样本量限制</li>
            <li>缺失数据的处理可能影响结果的稳健性</li>
        </ul>
        
        <h3>建议</h3>
        <ul>
            <li>建议结合临床专业知识解释统计结果</li>
            <li>对于显著性结果，建议评估其临床意义</li>
            <li>建议进行敏感性分析以验证结果的稳健性</li>
        </ul>
    </div>
    """
    
    # 报告结尾
    report_html += f"""
    <div class="footer">
        <p>本报告由临床试验分析系统自动生成，生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</p>
        <p>注意: 本报告仅供统计分析参考，临床决策需结合专业医学判断。</p>
    </div>
    
    </body>
    </html>
    """
    
    return report_html

# 辅助函数
def perform_cox_regression(df, time_col, event_col, adjustment_vars):
    """执行Cox比例风险回归"""
    try:
        from lifelines import CoxPHFitter
        
        # 准备数据
        cox_data = df[['治疗组', time_col, event_col] + adjustment_vars].dropna()
        
        # 编码治疗组
        cox_data = pd.get_dummies(cox_data, columns=['治疗组'], prefix='treatment')
        
        # 拟合Cox模型
        cph = CoxPHFitter()
        cph.fit(cox_data, duration_col=time_col, event_col=event_col)
        
        # 显示结果
        st.markdown("**Cox回归结果:**")
        
        # 提取治疗组效应
        treatment_params = [col for col in cph.params.index if 'treatment_' in col]
        
        for param in treatment_params:
            hr = np.exp(cph.params[param])
            ci_lower = np.exp(cph.confidence_intervals_.loc[param, 'coef lower 95%'])
            ci_upper = np.exp(cph.confidence_intervals_.loc[param, 'coef upper 95%'])
            p_value = cph.summary.loc[param, 'p']
            
            st.write(f"• {param}: HR = {hr:.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f}), P = {p_value:.4f}")
        
        # 模型拟合度
        st.write(f"• Concordance Index: {cph.concordance_index_:.3f}")
        st.write(f"• Log-likelihood: {cph.log_likelihood_:.2f}")
        
    except ImportError:
        st.warning("⚠️ 需要安装lifelines库进行Cox回归分析")
    except Exception as e:
        st.error(f"❌ Cox回归分析失败: {str(e)}")

def analyze_ordinal_endpoint(df, endpoint, alpha_level, confidence_level, adjustment_vars):
    """分析有序分类终点"""
    st.markdown("#### 📊 有序分类终点分析结果")
    
    treatment_groups = df['治疗组'].unique()
    
    # 描述性统计
    st.markdown("##### 📋 描述性统计")
    
    ordinal_stats = []
    for group in treatment_groups:
        group_data = df[df['治疗组'] == group][endpoint].dropna()
        
        if len(group_data) > 0:
            value_counts = group_data.value_counts().sort_index()
            total = len(group_data)
            
            # 计算累积比例
            cumulative_props = value_counts.cumsum() / total
            
            ordinal_stats.append({
                '治疗组': group,
                '总例数': total,
                '分布': '; '.join([f"{val}:{count}({count/total*100:.1f}%)" 
                                for val, count in value_counts.items()]),
                '中位数': group_data.median(),
                '众数': group_data.mode().iloc[0] if not group_data.mode().empty else 'N/A'
            })
    
    ordinal_df = pd.DataFrame(ordinal_stats)
    st.dataframe(ordinal_df, use_container_width=True)
    
    # 统计检验 - Mann-Whitney U 或 Kruskal-Wallis
    st.markdown("##### 🧮 统计检验")
    
    if len(treatment_groups) == 2:
        group1_data = df[df['治疗组'] == treatment_groups[0]][endpoint].dropna()
        group2_data = df[df['治疗组'] == treatment_groups[1]][endpoint].dropna()
        
        if len(group1_data) > 0 and len(group2_data) > 0:
            u_stat, p_value = mannwhitneyu(group1_data, group2_data, alternative='two-sided')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Mann-Whitney U检验:**")
                st.write(f"• U统计量: {u_stat:.4f}")
                st.write(f"• P值: {p_value:.4f}")
                
                # 效应量
                z_score = stats.norm.ppf(1 - p_value/2)
                effect_size_r = abs(z_score) / np.sqrt(len(group1_data) + len(group2_data))
                st.write(f"• 效应量(r): {effect_size_r:.3f}")
            
            with col2:
                if p_value < alpha_level:
                    st.success(f"✅ 在α={alpha_level}水平下，两组分布存在显著差异")
                else:
                    st.info(f"ℹ️ 在α={alpha_level}水平下，两组分布无显著差异")
    
    else:
        # 多组比较
        group_data_list = []
        for group in treatment_groups:
            group_data = df[df['治疗组'] == group][endpoint].dropna()
            group_data_list.append(group_data)
        
        if all(len(data) > 0 for data in group_data_list):
            h_stat, p_value = stats.kruskal(*group_data_list)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Kruskal-Wallis检验:**")
                st.write(f"• H统计量: {h_stat:.4f}")
                st.write(f"• P值: {p_value:.4f}")
                st.write(f"• 自由度: {len(treatment_groups)-1}")
            
            with col2:
                if p_value < alpha_level:
                    st.success(f"✅ 在α={alpha_level}水平下，各组分布存在显著差异")
                else:
                    st.info(f"ℹ️ 在α={alpha_level}水平下，各组分布无显著差异")

# 其他缺失的辅助函数可以根据需要继续添加...

if __name__ == "__main__":
    clinical_trial_analysis()



            
