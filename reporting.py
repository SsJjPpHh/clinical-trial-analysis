# reporting.py
"""
快速生成 Markdown 报告

• 选择要插入的数据集、结果表
• 自定义章节内容
• 一键导出 Markdown / HTML
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict

# -------- 会话工具 -------- #
def list_datasets() -> Dict[str, pd.DataFrame]:
    ds = {}
    for k, v in st.session_state.items():
        if k.startswith("dataset_") and isinstance(v, dict) and "data" in v:
            ds[v["name"]] = v["data"]
    return ds

# -------- UI -------- #
def reporting_ui() -> None:
    st.set_page_config("报告生成", "📝", layout="wide")
    st.markdown("# 📝 报告生成")

    st.sidebar.markdown("## 报告结构")
    title = st.sidebar.text_input("报告标题", "研究分析报告")
    author = st.sidebar.text_input("作者/团队", "")
    date_str = datetime.today().strftime("%Y-%m-%d")
    abstract = st.text_area("摘要", "")

    st.markdown("### **正文撰写**")
    intro = st.text_area("1. 背景与目的", "", height=150)
    methods = st.text_area("2. 方法", "", height=150)
    results_text = st.text_area("3. 结果（文字描述）", "", height=150)
    discussion = st.text_area("4. 讨论", "", height=150)

    # 插入结果表
    st.markdown("#### 插入数据表")
    datasets = list_datasets()
    table_names = st.multiselect("选择要附加为附录的数据集", list(datasets.keys()))
    # 预览
    for name in table_names:
        with st.expander(f"📄 {name}"):
            st.dataframe(datasets[name].head())

    if st.button("📑 生成 Markdown 报告"):
        lines = [
            f"# {title}",
            f"*{author}*  \n{date_str}",
            "",
            "## 摘要",
            abstract,
            "## 1. 背景与目的",
            intro,
            "## 2. 方法",
            methods,
            "## 3. 结果",
            results_text,
        ]
        # 自动添加表格
        for name in table_names:
            lines.append(f"### 附录：{name}")
            lines.append(datasets[name].head().to_markdown(index=False))
        lines.append("## 4. 讨论")
        lines.append(discussion)
        md_report = "\n\n".join(lines)

        st.success("报告生成成功✅")
        st.markdown("---")
        st.markdown(md_report)

        # 下载
        st.download_button("📥 下载 Markdown", md_report.encode("utf-8-sig"), "report.md", "text/markdown")
        html = st.markdown(md_report, unsafe_allow_html=True)
        st.download_button("📥 下载 HTML", html.html.encode("utf-8-sig"), "report.html", "text/html")

if __name__ == "__main__":
    reporting_ui()

