import streamlit as st
import pandas as pd
import numpy as np

# 修改导入语句 - 直接导入各个模块文件
from clinical_trial import clinical_trial_ui
from data_management import data_management_ui
from epidemiology import epidemiology_ui
from randomization import randomization_ui
from reporting import reporting_ui
from sample_size import sample_size_ui
from survival_analysis import survival_analysis_ui

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
    
    menu_options = {
        "🏠 首页": "home",
        "📊 数据管理": "data_management",
        "🧪 临床试验分析": "clinical_trial",
        "🦠 流行病学分析": "epidemiology",
        "🎲 随机化": "randomization",
        "📈 生存分析": "survival_analysis",
        "🧮 样本量计算": "sample_size",
        "📄 报告生成": "reporting"
    }
    
    selected = st.sidebar.selectbox("选择功能模块", list(menu_options.keys()))
    
    # 根据选择显示相应页面
    if menu_options[selected] == "home":
        show_home_page()
    elif menu_options[selected] == "data_management":
        data_management_ui()
    elif menu_options[selected] == "clinical_trial":
        clinical_trial_ui()
    elif menu_options[selected] == "epidemiology":
        epidemiology_ui()
    elif menu_options[selected] == "randomization":
        randomization_ui()
    elif menu_options[selected] == "survival_analysis":
        survival_analysis_ui()
    elif menu_options[selected] == "sample_size":
        sample_size_ui()
    elif menu_options[selected] == "reporting":
        reporting_ui()

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

if __name__ == "__main__":
    main()
