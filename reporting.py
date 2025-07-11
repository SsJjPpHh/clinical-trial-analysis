# reporting.py  ────────────────────────────────────────────────
"""
自动报告生成器（重构版）
Author : Hu Fan
Date   : 2025-07-11
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import io
from datetime import datetime
import base64

# ╭─────────────────── 数据接口 ────────────────────╮
def get_dataset() -> tuple[pd.DataFrame | None, str]:
    ds = st.session_state.get("dataset_current")
    if ds:
        return ds["data"], ds["name"]
    return None, ""


def get_rand_table() -> tuple[pd.DataFrame | None, str]:
    rt = st.session_state.get("rand_table")
    if rt:
        return rt["data"], rt["name"]
    return None, ""


# ╭─────────────────── 报告生成逻辑 ───────────────────╮
def build_markdown_report(sections: list[str]) -> str:
    """
    根据用户选择的章节，拼接 Markdown 文本
    """
    md: list[str] = [f"# 📑 试验分析报告  \n生成时间：{datetime.now():%Y-%m-%d %H:%M}"]
    df, name = get_dataset()
    if "数据概览" in sections and df is not None:
        md += [
            "## 1. 数据概览",
            f"- 数据集：**{name}**",
            f"- 行数：{len(df)}",
            f"- 列数：{len(df.columns)}",
            "",
            "```text",
            df.head().to_string(index=False),
            "```",
        ]

    if "随机表" in sections:
        rt, label = get_rand_table()
        if rt is not None:
            md += [
                "## 2. 随机分组表",
                f"方案：**{label}**  ",
                "```text",
                rt.head().to_string(index=False),
                "```",
            ]
        else:
            md.append("> ⚠️ 未检测到随机表，已跳过该章节。")

    if "自定义段落" in sections:
        md += [
            "## 3. 讨论与结论",
            "（此处可在导出的 Markdown 文件中继续补充）",
        ]

    return "\n".join(md)


def download_markdown(md: str, filename: str) -> None:
    """
    提供 Markdown 下载按钮
    """
    b64 = base64.b64encode(md.encode()).decode()  # str -> base64
    href = f'<a href="data:text/markdown;base64,{b64}" download="{filename}">⬇️ 点击下载 Markdown</a>'
    st.markdown(href, unsafe_allow_html=True)


# ╭─────────────────── UI ────────────────────────────╮
def reporting_ui() -> None:
    st.title("📝 报告生成器")
    st.markdown("选择需要包含的部分，一键生成 Markdown 报告。")

    df, _ = get_dataset()
    if df is None:
        st.warning("请先在数据管理中心导入数据。")

    st.sidebar.header("📋 章节选择")
    sections = st.sidebar.multiselect(
        "包含以下章节",
        ["数据概览", "随机表", "自定义段落"],
        default=["数据概览"],
    )

    if st.button("生成报告"):
        md = build_markdown_report(sections)
        st.success("报告已生成，可在下方预览并下载。")
        st.markdown(md)
        download_markdown(md, f"report_{datetime.now():%Y%m%d%H%M}.md")


if __name__ == "__main__":
    st.set_page_config(page_title="报告生成器", layout="wide")
    reporting_ui()
