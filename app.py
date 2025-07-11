# app.py  ────────────────────────────────────────────────
"""
Streamlit 主入口
Author : H
Date   : 2025-07-11
"""

from __future__ import annotations
import streamlit as st
from typing import Callable, Dict

# ───────────── 本地模块导入 ─────────────
try:
    from data_management import data_management_ui
    from randomization import randomization_ui
    from sample_size import sample_size_ui
    from survival_analysis import survival_ui
    from reporting import reporting_ui
    from clinical_trial import clinical_trial_ui
    from epidemiology import epidemiology_ui
except Exception as e:      # 捕获所有异常，统一提示
    st.error(f"❌ 模块导入失败：{e}")
    st.stop()               # 中断执行，防止后续报错

# ───────────── 页面 & 路由定义 ─────────────
PAGES: Dict[str, Callable[[], None]] = {
    "📂 数据管理": data_management_ui,
    "🎲 随机分组": randomization_ui,
    "📏 样本量计算": sample_size_ui,
    "⏳ 生存分析": survival_ui,
    "📝 报告生成": reporting_ui,
    "🧪 临床试验分析": clinical_trial_ui,
    "🔬 流行病学分析": epidemiology_ui,
}

def main() -> None:
    st.set_page_config(
        page_title="统计分析平台",
        layout="wide",
        menu_items={
            "About": "基于 Streamlit 的临床与流行病学分析一体化工具\nAuthor: H  (2025-07-11)",
        },
    )

    # ──────────── 侧边栏导航 ────────────
    st.sidebar.title("📊 功能导航")
    selection = st.sidebar.radio("选择页面", list(PAGES.keys()))

    # ──────────── 渲染选中页面 ────────────
    page_func = PAGES[selection]
    page_func()


if __name__ == "__main__":
    main()

