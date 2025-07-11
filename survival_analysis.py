import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

def survival_analysis_ui():
  st.header("⏱️ 生存分析")

if st.session_state.cleaned_data is None:
  st.warning("请先导入并清理数据")
return

df = st.session_state.cleaned_data

# 分析类型选择
analysis_type = st.selectbox(
  "选择分析类型",
  ["Kaplan-Meier分析", "Cox回归分析", "生存率比较"]
)

if analysis_type == "Kaplan-Meier分析":
  kaplan_meier_analysis(df)
elif analysis_type == "Cox回归分析":
  cox_regression_analysis(df)
elif analysis_type == "生存率比较":
  survival_comparison_analysis(df)

def kaplan_meier_analysis(df):
  st.subheader("📈 Kaplan-Meier生存分析")

col1, col2 = st.columns([1, 2])

with col1:
  st.write("**变量选择**")

# 数值变量（时间变量）
numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
time_var = st.selectbox("时间变量", ["请选择"] + numeric_vars)

# 事件变量
all_vars = df.columns.tolist()
event_var = st.selectbox("事件变量", ["请选择"] + all_vars)

# 分组变量（可选）
categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
group_var = st.selectbox("分组变量（可选）", ["无分组"] + categorical_vars)

st.write("**分析选项**")
conf_level = st.number_input("置信水平", value=0.95, min_value=0.8, max_value=0.99, step=0.01)
show_risk_table = st.checkbox("显示风险表", True)
show_conf_int = st.checkbox("显示置信区间", True)

run_km = st.button("🚀 运行K-M分析", type="primary")

with col2:
  if run_km and time_var != "请选择" and event_var != "请选择":
  try:
  # 数据准备
  analysis_df = df[[time_var, event_var]].copy()
if group_var != "无分组":
  analysis_df[group_var] = df[group_var]

# 删除缺失值
analysis_df = analysis_df.dropna()

# 确保事件变量是0/1编码
if analysis_df[event_var].dtype == 'object':
  unique_vals = analysis_df[event_var].unique()
if len(unique_vals) == 2:
  analysis_df[event_var] = (analysis_df[event_var] == unique_vals[1]).astype(int)

# Kaplan-Meier分析
results = perform_kaplan_meier(analysis_df, time_var, event_var, group_var, conf_level)

# 显示结果
display_km_results(results, show_risk_table, show_conf_int)

except Exception as e:
  st.error(f"分析失败: {str(e)}")

def perform_kaplan_meier(df, time_var, event_var, group_var, conf_level):
  """执行Kaplan-Meier分析"""

results = {
  'survival_data': {},
  'median_survival': {},
  'logrank_test': None,
  'survival_table': None
}

if group_var == "无分组":
  # 单组分析
  kmf = KaplanMeierFitter(alpha=1-conf_level)
kmf.fit(df[time_var], df[event_var], label='Overall')

results['survival_data']['Overall'] = {
  'timeline': kmf.timeline,
  'survival_function': kmf.survival_function_['Overall'],
  'confidence_interval_lower': kmf.confidence_interval_['Overall_lower_' + str(conf_level)],
  'confidence_interval_upper': kmf.confidence_interval_['Overall_upper_' + str(conf_level)]
}

results['median_survival']['Overall'] = kmf.median_survival_time_
results['survival_table'] = kmf.survival_function_

else:
  # 分组分析
  groups = df[group_var].unique()

for group in groups:
  group_data = df[df[group_var] == group]

kmf = KaplanMeierFitter(alpha=1-conf_level)
kmf.fit(group_data[time_var], group_data[event_var], label=str(group))

results['survival_data'][str(group)] = {
  'timeline': kmf.timeline,
  'survival_function': kmf.survival_function_[str(group)],
  'confidence_interval_lower': kmf.confidence_interval_[str(group) + '_lower_' + str(conf_level)],
  'confidence_interval_upper': kmf.confidence_interval_[str(group) + '_upper_' + str(conf_level)]
}

results['median_survival'][str(group)] = kmf.median_survival_time_

# Log-rank检验
if len(groups) == 2:
  group1_data = df[df[group_var] == groups[0]]
group2_data = df[df[group_var] == groups[1]]

results['logrank_test'] = logrank_test(
  group1_data[time_var], group2_data[time_var],
  group1_data[event_var], group2_data[event_var]
)

return results

def display_km_results(results, show_risk_table, show_conf_int):
  """显示Kaplan-Meier结果"""

# 生存曲线图
st.write("**生存曲线**")

fig = go.Figure()

for group_name, data in results['survival_data'].items():
  # 主曲线
  fig.add_trace(go.Scatter(
    x=data['timeline'],
    y=data['survival_function'],
    mode='lines',
    name=group_name,
    line=dict(width=3)
  ))

# 置信区间
if show_conf_int:
  fig.add_trace(go.Scatter(
    x=data['timeline'],
    y=data['confidence_interval_upper'],
    mode='lines',
    line=dict(width=0),
    showlegend=False,
    hoverinfo='skip'
  ))

fig.add_trace(go.Scatter(
  x=data['timeline'],
  y=data['confidence_interval_lower'],
  mode='lines',
  line=dict(width=0),
  fill='tonexty',
  fillcolor=f'rgba(0,100,80,0.2)',
  showlegend=False,
  hoverinfo='skip'
))

fig.update_layout(
  title="Kaplan-Meier生存曲线",
  xaxis_title="时间",
  yaxis_title="生存概率",
  yaxis=dict(range=[0, 1]),
  hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)

# 中位生存时间
st.write("**中位生存时间**")
median_df = pd.DataFrame([
  {'组别': group, '中位生存时间': median_time}
  for group, median_time in results['median_survival'].items()
])
st.dataframe(median_df, use_container_width=True)

# Log-rank检验结果
if results['logrank_test'] is not None:
  st.write("**Log-rank检验**")
col1, col2, col3 = st.columns(3)

with col1:
  st.metric("检验统计量", f"{results['logrank_test'].test_statistic:.4f}")
with col2:
  st.metric("P值", f"{results['logrank_test'].p_value:.4f}")
with col3:
  significance = "显著" if results['logrank_test'].p_value < 0.05 else "不显著"
st.metric("结果", significance)

def cox_regression_analysis(df):
  st.subheader("📊 Cox回归分析")

col1, col2 = st.columns([1, 2])

with col1:
  st.write("**变量选择**")

# 时间和事件变量
numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
time_var = st.selectbox("时间变量", ["请选择"] + numeric_vars, key="cox_time")

all_vars = df.columns.tolist()
event_var = st.selectbox("事件变量", ["请选择"] + all_vars, key="cox_event")

# 协变量
available_vars = [var for var in all_vars if var not in [time_var, event_var]]
covariates = st.multiselect("协变量", available_vars)

st.write("**分析选项**")
conf_level = st.number_input("置信水平", value=0.95, min_value=0.8, max_value=0.99, step=0.01, key="cox_conf")

run_cox = st.button("🚀 运行Cox回归", type="primary")

with col2:
  if run_cox and time_var != "请选择" and event_var != "请选择" and covariates:
  try:
  # 数据准备
  analysis_vars = [time_var, event_var] + covariates
analysis_df = df[analysis_vars].copy()
analysis_df = analysis_df.dropna()

# 确保事件变量是0/1编码
if analysis_df[event_var].dtype == 'object':
  unique_vals = analysis_df[event_var].unique()
if len(unique_vals) == 2:
  analysis_df[event_var] = (analysis_df[event_var] == unique_vals[1]).astype(int)

# 处理分类变量（创建虚拟变量）
for var in covariates:
  if analysis_df[var].dtype == 'object':
  dummies = pd.get_dummies(analysis_df[var], prefix=var, drop_first=True)
analysis_df = pd.concat([analysis_df, dummies], axis=1)
analysis_df = analysis_df.drop(var, axis=1)

# Cox回归分析
cph = CoxPHFitter(alpha=1-conf_level)
cph.fit(analysis_df, duration_col=time_var, event_col=event_var)

# 显示结果
display_cox_results(cph, analysis_df, time_var, event_var)

except Exception as e:
  st.error(f"分析失败: {str(e)}")

def display_cox_results(cph, df, time_var, event_var):
  """显示Cox回归结果"""

# 回归系数表
st.write("**Cox回归系数**")
summary_df = cph.summary
st.dataframe(summary_df, use_container_width=True)

# 模型拟合统计
col1, col2, col3 = st.columns(3)

with col1:
  st.metric("对数似然", f"{cph.log_likelihood_:.2f}")
with col2:
  st.metric("AIC", f"{cph.AIC_:.2f}")
with col3:
  st.metric("Concordance", f"{cph.concordance_index_:.4f}")

# 风险比森林图
st.write("**风险比森林图**")

hazard_ratios = np.exp(cph.params_)
lower_ci = np.exp(cph.confidence_intervals_.iloc[:, 0])
upper_ci = np.exp(cph.confidence_intervals_.iloc[:, 1])

fig = go.Figure()

for i, var in enumerate(hazard_ratios.index):
  fig.add_trace(go.Scatter(
    x=[hazard_ratios[var]],
    y=[i],
    mode='markers',
    marker=dict(size=10, color='blue'),
    name=var,
    showlegend=False
  ))

# 置信区间
fig.add_shape(
  type="line",
  x0=lower_ci[var], y0=i,
  x1=upper_ci[var], y1=i,
  line=dict(color="blue", width=2)
)

# 添加参考线 HR=1
fig.add_vline(x=1, line_dash="dash", line_color="red")

fig.update_layout(
  title="风险比及95%置信区间",
  xaxis_title="风险比 (HR)",
  yaxis=dict(
    tickmode='array',
    tickvals=list(range(len(hazard_ratios))),
    ticktext=hazard_ratios.index.tolist()
  ),
  showlegend=False
)

st.plotly_chart(fig, use_container_width=True)

def survival_comparison_analysis(df):
  st.subheader("⚖️ 生存率比较")

col1, col2 = st.columns([1, 2])

with col1:
  st.write("**参数设置**")

# 变量选择
numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
time_var = st.selectbox("时间变量", ["请选择"] + numeric_vars, key="comp_time")

all_vars = df.columns.tolist()
event_var = st.selectbox("事件变量", ["请选择"] + all_vars, key="comp_event")

categorical_vars = df.select_dtypes(include=['object', 'category']).columns.tolist()
group_var = st.selectbox("分组变量", ["请选择"] + categorical_vars, key="comp_group")

# 比较时间点
time_points = st.text_input("比较时间点（用逗号分隔）", "12,24,36")

run_comparison = st.button("🚀 运行比较分析", type="primary")

with col2:
  if run_comparison and all(var != "请选择" for var in [time_var, event_var, group_var]):
  try:
  # 解析时间点
  time_points_list = [float(t.strip()) for t in time_points.split(',')]

# 数据准备
analysis_df = df[[time_var, event_var, group_var]].copy()
analysis_df = analysis_df.dropna()

# 确保事件变量是0/1编码
if analysis_df[event_var].dtype == 'object':
  unique_vals = analysis_df[event_var].unique()
if len(unique_vals) == 2:
  analysis_df[event_var] = (analysis_df[event_var] == unique_vals[1]).astype(int)

# 执行比较分析
results = perform_survival_comparison(
  analysis_df, time_var, event_var, group_var, time_points_list
)

# 显示结果
display_comparison_results(results, time_points_list)

except Exception as e:
  st.error(f"分析失败: {str(e)}")

def perform_survival_comparison(df, time_var, event_var, group_var, time_points):
  """执行生存率比较分析"""

results = {
  'survival_rates': {},
  'comparisons': []
}

groups = df[group_var].unique()

# 计算各组在不同时间点的生存率
for group in groups:
  group_data = df[df[group_var] == group]

kmf = KaplanMeierFitter()
kmf.fit(group_data[time_var], group_data[event_var])

group_rates = {}
for time_point in time_points:
  try:
  survival_rate = kmf.survival_function_at_times(time_point).values[0]
group_rates[time_point] = survival_rate
except:
  group_rates[time_point] = np.nan

results['survival_rates'][str(group)] = group_rates

# 组间比较
if len(groups) == 2:
  group1_data = df[df[group_var] == groups[0]]
group2_data = df[df[group_var] == groups[1]]

logrank_result = logrank_test(
  group1_data[time_var], group2_data[time_var],
  group1_data[event_var], group2_data[event_var]
)

results['comparisons'].append({
  'comparison': f"{groups[0]} vs {groups[1]}",
  'test_statistic': logrank_result.test_statistic,
  'p_value': logrank_result.p_value
})

return results

def display_comparison_results(results, time_points):
  """显示生存率比较结果"""

# 生存率表格
st.write("**各组生存率**")

survival_table = []
for group, rates in results['survival_rates'].items():
  row = {'组别': group}
for time_point in time_points:
  row[f'{time_point}时间点生存率'] = f"{rates[time_point]:.3f}" if not np.isnan(rates[time_point]) else "N/A"
survival_table.append(row)

survival_df = pd.DataFrame(survival_table)
st.dataframe(survival_df, use_container_width=True)

# 生存率比较图
st.write("**生存率比较图**")

fig = go.Figure()

for group, rates in results['survival_rates'].items():
  valid_times = []
valid_rates = []

for time_point in time_points:
  if not np.isnan(rates[time_point]):
  valid_times.append(time_point)
valid_rates.append(rates[time_point])

fig.add_trace(go.Scatter(
  x=valid_times,
  y=valid_rates,
  mode='lines+markers',
  name=group,
  line=dict(width=3),
  marker=dict(size=8)
))

fig.update_layout(
  title="不同时间点生存率比较",
  xaxis_title="时间",
  yaxis_title="生存率",
  yaxis=dict(range=[0, 1])
)

st.plotly_chart(fig, use_container_width=True)

# 统计检验结果
if results['comparisons']:
  st.write("**统计检验结果**")
comparison_df = pd.DataFrame(results['comparisons'])
st.dataframe(comparison_df, use_container_width=True)
