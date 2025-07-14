# randomization.py
"""
随机化列表生成模块

• Simple / Block / Stratified Block
• 支持多臂试验、指定块长、指定分层变量
• 生成结果可保存到会话并下载 CSV
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

# ---------------- Session 工具 ---------------- #
def _session_dataset_key(name: str) -> str:
    return f"dataset_{name}"

def list_datasets() -> Dict[str, pd.DataFrame]:
    ds = {}
    for k, v in st.session_state.items():
        if k.startswith("dataset_") and isinstance(v, dict) and "data" in v:
            ds[v["name"]] = v["data"]
    return ds

def save_dataset(name: str, df: pd.DataFrame) -> None:
    st.session_state[_session_dataset_key(name)] = {
        "name": name,
        "data": df,
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

# ---------------- 随机化算法 ---------------- #
def simple_randomization(n: int, arms: List[str], seed: int = None) -> pd.DataFrame:
    """简单随机化"""
    if seed is not None:
        np.random.seed(seed)
    
    assignments = np.random.choice(arms, size=n)
    
    return pd.DataFrame({
        'subject_id': range(1, n + 1),
        'treatment': assignments,
        'randomization_date': datetime.now().strftime("%Y-%m-%d"),
        'method': 'Simple Randomization'
    })

def block_randomization(n: int, arms: List[str], block_size: int, seed: int = None) -> pd.DataFrame:
    """分块随机化"""
    if seed is not None:
        np.random.seed(seed)
    
    if block_size % len(arms) != 0:
        st.warning(f"块大小 {block_size} 不能被组数 {len(arms)} 整除，建议调整")
    
    assignments = []
    block_num = 1
    
    for i in range(0, n, block_size):
        remaining = min(block_size, n - i)
        
        # 创建一个块
        block = []
        per_arm = remaining // len(arms)
        extra = remaining % len(arms)
        
        for j, arm in enumerate(arms):
            count = per_arm + (1 if j < extra else 0)
            block.extend([arm] * count)
        
        # 随机打乱块内顺序
        np.random.shuffle(block)
        assignments.extend(block)
        block_num += 1
    
    return pd.DataFrame({
        'subject_id': range(1, len(assignments) + 1),
        'treatment': assignments,
        'randomization_date': datetime.now().strftime("%Y-%m-%d"),
        'method': f'Block Randomization (Block Size: {block_size})'
    })

def stratified_randomization(n: int, arms: List[str], strata_info: Dict, seed: int = None) -> pd.DataFrame:
    """分层随机化"""
    if seed is not None:
        np.random.seed(seed)
    
    all_assignments = []
    subject_id = 1
    
    for stratum, count in strata_info.items():
        # 为每个分层进行简单随机化
        stratum_assignments = np.random.choice(arms, size=count)
        
        for assignment in stratum_assignments:
            all_assignments.append({
                'subject_id': subject_id,
                'treatment': assignment,
                'stratum': stratum,
                'randomization_date': datetime.now().strftime("%Y-%m-%d"),
                'method': 'Stratified Randomization'
            })
            subject_id += 1
    
    return pd.DataFrame(all_assignments)

# ---------------- 主UI函数 ---------------- #
def randomization_ui():
    """随机化工具主界面"""
    st.title("🎲 随机化工具")
    st.markdown("生成临床试验随机化列表")
    
    # 基本参数设置
    st.header("📋 基本设置")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_subjects = st.number_input("受试者总数", min_value=1, max_value=10000, value=100)
        seed = st.number_input("随机种子（可选）", min_value=0, value=42, help="设置种子可确保结果可重现")
    
    with col2:
        # 治疗组设置
        n_arms = st.selectbox("治疗组数", options=[2, 3, 4, 5], value=2)
        
        arms = []
        for i in range(n_arms):
            arm_name = st.text_input(f"第 {i+1} 组名称", value=f"组{i+1}", key=f"arm_{i}")
            arms.append(arm_name)
    
    # 随机化方法选择
    st.header("🔄 随机化方法")
    
    method = st.selectbox(
        "选择随机化方法",
        options=["简单随机化", "分块随机化", "分层随机化"]
    )
    
    # 方法特定参数
    if method == "分块随机化":
        block_size = st.selectbox(
            "块大小",
            options=[4, 6, 8, 10, 12],
            value=4,
            help="建议选择能被组数整除的块大小"
        )
    
    elif method == "分层随机化":
        st.subheader("分层信息设置")
        
        # 简化的分层设置
        strata_names = st.text_input(
            "分层名称（用逗号分隔）",
            value="男性,女性",
            help="例如：男性,女性 或 中心A,中心B,中心C"
        ).split(",")
        
        strata_info = {}
        cols = st.columns(len(strata_names))
        
        for i, stratum in enumerate(strata_names):
            with cols[i]:
                count = st.number_input(
                    f"{stratum.strip()} 人数",
                    min_value=1,
                    value=n_subjects // len(strata_names),
                    key=f"stratum_{i}"
                )
                strata_info[stratum.strip()] = count
        
        # 检查总数
        total_strata = sum(strata_info.values())
        if total_strata != n_subjects:
            st.warning(f"分层总人数 ({total_strata}) 与设定总数 ({n_subjects}) 不符")
    
    # 生成随机化列表
    st.header("🎯 生成随机化列表")
    
    if st.button("生成随机化列表", type="primary"):
        try:
            if method == "简单随机化":
                df = simple_randomization(n_subjects, arms, seed)
            
            elif method == "分块随机化":
                df = block_randomization(n_subjects, arms, block_size, seed)
            
            elif method == "分层随机化":
                df = stratified_randomization(n_subjects, arms, strata_info, seed)
            
            # 显示结果
            st.success(f"✅ 成功生成 {len(df)} 个受试者的随机化列表")
            
            # 基本统计
            st.subheader("📊 分组统计")
            treatment_counts = df['treatment'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**各组人数：**")
                for treatment, count in treatment_counts.items():
                    st.write(f"- {treatment}: {count} 人 ({count/len(df)*100:.1f}%)")
            
            with col2:
                # 简单的柱状图
                import plotly.express as px
                fig = px.bar(
                    x=treatment_counts.index,
                    y=treatment_counts.values,
                    title="各组受试者分布",
                    labels={'x': '治疗组', 'y': '人数'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 显示数据表
            st.subheader("📋 随机化列表")
            st.dataframe(df, use_container_width=True)
            
            # 保存选项
            col1, col2 = st.columns(2)
            
            with col1:
                dataset_name = st.text_input(
                    "保存到会话（数据集名称）",
                    value=f"randomization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                
                if st.button("保存到会话"):
                    save_dataset(dataset_name, df)
                    st.success(f"✅ 已保存为数据集：{dataset_name}")
            
            with col2:
                # 下载CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 下载CSV文件",
                    data=csv,
                    file_name=f"randomization_list_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        except Exception as e:
            st.error(f"❌ 生成失败：{str(e)}")
    
    # 显示现有数据集
    st.header("💾 会话中的数据集")
    datasets = list_datasets()
    
    if datasets:
        for name, df in datasets.items():
            with st.expander(f"📊 {name} ({len(df)} 行 × {len(df.columns)} 列)"):
                st.dataframe(df.head(), use_container_width=True)
    else:
        st.info("暂无保存的数据集")

if __name__ == "__main__":
    randomization_ui()
