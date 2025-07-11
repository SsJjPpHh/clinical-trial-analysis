import streamlit as st
import pandas as pd
import numpy as np
from modules import (
    data_management, sample_size, randomization, 
    clinical_trial, survival_analysis, epidemiology, reporting
)

# 页面配置
st.set_page_config(
    page_title="临床试验统计分析系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 初始化session state
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'cleaned_data' not in st.session_state:
    st.session_state.cleaned_data = None

def main():
    st.title("🏥 临床试验统计分析系统")
    
    # 侧边栏导航
    st.sidebar.title("导航菜单")
    
    # 主菜单
    main_menu = st.sidebar.selectbox(
        "选择模块",
        ["数据管理", "研究设计", "统计分析", "报告生成"]
    )
    
    if main_menu == "数据管理":
        sub_menu = st.sidebar.selectbox(
            "数据管理子菜单",
            ["数据导入", "数据清理", "数据探索"]
        )
        
        if sub_menu == "数据导入":
            data_management.data_import_ui()
        elif sub_menu == "数据清理":
            data_management.data_cleaning_ui()
        elif sub_menu == "数据探索":
            data_management.data_exploration_ui()
            
    elif main_menu == "研究设计":
        sub_menu = st.sidebar.selectbox(
            "研究设计子菜单",
            ["样本量计算", "随机化方案"]
        )
        
        if sub_menu == "样本量计算":
            sample_size.sample_size_ui()
        elif sub_menu == "随机化方案":
            randomization.randomization_ui()
            
    elif main_menu == "统计分析":
        sub_menu = st.sidebar.selectbox(
            "统计分析子菜单",
            ["基线特征分析", "生存分析", "流行病学分析"]
        )
        
        if sub_menu == "基线特征分析":
            clinical_trial.baseline_analysis_ui()
        elif sub_menu == "生存分析":
            survival_analysis.survival_analysis_ui()
        elif sub_menu == "流行病学分析":
            epidemiology.epidemiology_ui()
            
    elif main_menu == "报告生成":
        reporting.reporting_ui()

if __name__ == "__main__":
    main()
