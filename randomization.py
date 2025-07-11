import streamlit as st
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

def randomization_ui():
    st.header("🎲 随机化方案")
    
    # 随机化参数设置
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("基本参数")
        total_subjects = st.number_input("总受试者数", value=100, min_value=10, max_value=10000)
        num_groups = st.number_input("组数", value=2, min_value=2, max_value=6)
        
        # 组名设置
        group_names = []
        group_ratios = []
        
        for i in range(num_groups):
            col_name, col_ratio = st.columns(2)
            with col_name:
                name = st.text_input(f"组{i+1}名称", value=f"组{i+1}", key=f"group_name_{i}")
                group_names.append(name)
            with col_ratio:
                ratio = st.number_input(f"组{i+1}比例", value=1, min_value=1, max_value=10, key=f"group_ratio_{i}")
                group_ratios.append(ratio)
    
    with col2:
        st.subheader("随机化方法")
        randomization_type = st.selectbox(
            "随机化类型",
            ["简单随机化", "区组随机化", "分层随机化", "最小化随机化"]
        )
        
        if randomization_type == "区组随机化":
            block_size = st.number_input("区组大小", value=4, min_value=2, max_value=20)
            
        elif randomization_type == "分层随机化":
            stratification_vars = st.multiselect(
                "分层变量",
                ["性别", "年龄组", "疾病严重程度", "中心"],
                default=["性别"]
            )
            
        # 随机种子
        use_seed = st.checkbox("使用固定随机种子", value=True)
        if use_seed:
            random_seed = st.number_input("随机种子", value=12345)
        else:
            random_seed = None
    
    # 生成随机化序列
    if st.button("🎯 生成随机化序列", type="primary"):
        try:
            if random_seed:
                np.random.seed(random_seed)
                random.seed(random_seed)
            
            if randomization_type == "简单随机化":
                randomization_list = generate_simple_randomization(
                    total_subjects, group_names, group_ratios
                )
            elif randomization_type == "区组随机化":
                randomization_list = generate_block_randomization(
                    total_subjects, group_names, group_ratios, block_size
                )
            elif randomization_type == "分层随机化":
                randomization_list = generate_stratified_randomization(
                    total_subjects, group_names, group_ratios, stratification_vars
                )
            else:  # 最小化随机化
                randomization_list = generate_minimization_randomization(
                    total_subjects, group_names, group_ratios
                )
            
            # 显示结果
            display_randomization_results(randomization_list, group_names)
            
        except Exception as e:
            st.error(f"随机化生成失败: {str(e)}")

def generate_simple_randomization(total_subjects, group_names, group_ratios):
    """简单随机化"""
    
    # 计算各组样本量
    total_ratio = sum(group_ratios)
    group_sizes = [int(total_subjects * ratio / total_ratio) for ratio in group_ratios]
    
    # 调整总数
    diff = total_subjects - sum(group_sizes)
    if diff > 0:
        group_sizes[0] += diff
    
    # 生成随机序列
    allocation_list = []
    for i, (name, size) in enumerate(zip(group_names, group_sizes)):
        allocation_list.extend([name] * size)
    
    random.shuffle(allocation_list)
    
    # 创建DataFrame
    df = pd.DataFrame({
        '受试者编号': range(1, total_subjects + 1),
        '分组': allocation_list,
        '随机化时间': [datetime.now() + timedelta(days=i) for i in range(total_subjects)]
    })
    
    return df

def generate_block_randomization(total_subjects, group_names, group_ratios, block_size):
    """区组随机化"""
    
    num_groups = len(group_names)
    
    # 确保区组大小是组数的倍数
    if block_size % num_groups != 0:
        block_size = num_groups * (block_size // num_groups + 1)
    
    # 计算区组内各组分配数
    group_per_block = [block_size * ratio // sum(group_ratios) for ratio in group_ratios]
    
    # 生成区组
    allocation_list = []
    remaining = total_subjects
    
    while remaining > 0:
        current_block_size = min(block_size, remaining)
        
        # 当前区组的分配
        block_allocation = []
        for i, (name, count) in enumerate(zip(group_names, group_per_block)):
            actual_count = min(count, current_block_size - len(block_allocation))
            block_allocation.extend([name] * actual_count)
        
        # 如果区组未满，随机填充
        while len(block_allocation) < current_block_size:
            block_allocation.append(random.choice(group_names))
        
        # 随机打乱区组内顺序
        random.shuffle(block_allocation)
        allocation_list.extend(block_allocation)
        
        remaining -= current_block_size
    
    # 创建DataFrame
    df = pd.DataFrame({
        '受试者编号': range(1, len(allocation_list) + 1),
        '分组': allocation_list,
        '区组': [(i // block_size) + 1 for i in range(len(allocation_list))],
        '随机化时间': [datetime.now() + timedelta(days=i) for i in range(len(allocation_list))]
    })
    
    return df

def generate_stratified_randomization(total_subjects, group_names, group_ratios, stratification_vars):
    """分层随机化"""
    
    # 生成分层变量的模拟数据
    strata_data = {}
    
    for var in stratification_vars:
        if var == "性别":
            strata_data[var] = np.random.choice(['男', '女'], total_subjects, p=[0.5, 0.5])
        elif var == "年龄组":
            strata_data[var] = np.random.choice(['<65岁', '≥65岁'], total_subjects, p=[0.6, 0.4])
        elif var == "疾病严重程度":
            strata_data[var] = np.random.choice(['轻度', '中度', '重度'], total_subjects, p=[0.3, 0.5, 0.2])
        elif var == "中心":
            strata_data[var] = np.random.choice(['中心A', '中心B', '中心C'], total_subjects, p=[0.4, 0.35, 0.25])
    
    # 创建分层组合
    df = pd.DataFrame(strata_data)
    df['受试者编号'] = range(1, total_subjects + 1)
    
    # 为每个分层进行随机化
    allocation_list = []
    
    for stratum, group in df.groupby(stratification_vars):
        stratum_size = len(group)
        stratum_allocation = generate_simple_randomization(
            stratum_size, group_names, group_ratios
        )['分组'].tolist()
        allocation_list.extend(stratum_allocation)
    
    df['分组'] = allocation_list
    df['随机化时间'] = [datetime.now() + timedelta(days=i) for i in range(total_subjects)]
    
    return df

def generate_minimization_randomization(total_subjects, group_names, group_ratios):
    """最小化随机化（简化版本）"""
    
    # 初始化组计数
    group_counts = {name: 0 for name in group_names}
    allocation_list = []
    
    for i in range(total_subjects):
        # 计算不平衡度
        min_count = min(group_counts.values())
        min_groups = [name for name, count in group_counts.items() if count == min_count]
        
        # 80%概率选择样本量最少的组，20%随机选择
        if random.random() < 0.8:
            selected_group = random.choice(min_groups)
        else:
            selected_group = random.choice(group_names)
        
        allocation_list.append(selected_group)
        group_counts[selected_group] += 1
    
    # 创建DataFrame
    df = pd.DataFrame({
        '受试者编号': range(1, total_subjects + 1),
        '分组': allocation_list,
        '随机化时间': [datetime.now() + timedelta(days=i) for i in range(total_subjects)]
    })
    
    return df

def display_randomization_results(df, group_names):
    """显示随机化结果"""
    
    st.subheader("📋 随机化结果")
    
    # 汇总统计
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("总受试者数", len(df))
    
    with col2:
        group_counts = df['分组'].value_counts()
        st.write("**各组样本量:**")
        for group in group_names:
            if group in group_counts:
                st.write(f"{group}: {group_counts[group]}")
    
    with col3:
        # 组间平衡性检验
        expected_freq = len(df) / len(group_names)
        chi2_stat = sum((group_counts[group] - expected_freq)**2 / expected_freq 
                       for group in group_names if group in group_counts)
        st.metric("平衡性(χ²)", f"{chi2_stat:.2f}")
    
    # 可视化
    col1, col2 = st.columns(2)
    
    with col1:
        # 组分布饼图
        fig = px.pie(values=group_counts.values, names=group_counts.index, 
                    title="组分布")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # 累积分配图
        df_cumsum = df.copy()
        for group in group_names:
            df_cumsum[f'{group}_cumsum'] = (df_cumsum['分组'] == group).cumsum()
        
        fig = go.Figure()
        for group in group_names:
            fig.add_trace(go.Scatter(
                x=df_cumsum['受试者编号'],
                y=df_cumsum[f'{group}_cumsum'],
                mode='lines',
                name=group
            ))
        
        fig.update_layout(
            title="累积分配趋势",
            xaxis_title="受试者编号",
            yaxis_title="累积人数"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 详细列表
    st.subheader("📊 详细分配列表")
    st.dataframe(df, use_container_width=True)
    
    # 下载按钮
    csv = df.to_csv(index=False, encoding='utf-8-sig')
    st.download_button(
        label="📥 下载随机化列表",
        data=csv,
        file_name=f"randomization_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
