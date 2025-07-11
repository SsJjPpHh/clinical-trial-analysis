# randomization.py  ───────────────────────────────────────────────
"""
随机分组生成器（重构版）
Author : H
Date   : 2025-07-11
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import io
from typing import List, Dict, Tuple
from datetime import datetime


# ╭────────────────────── SessionState 工具 ──────────────────────╮
def save_rand_table(df: pd.DataFrame, label: str) -> None:
    st.session_state["rand_table"] = {
        "data": df,
        "name": label,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_rand_table() -> Tuple[pd.DataFrame | None, str]:
    rt = st.session_state.get("rand_table")
    if rt:
        return rt["data"], rt["name"]
    return None, ""


def get_dataset() -> Tuple[pd.DataFrame | None, str]:
    ds = st.session_state.get("dataset_current")
    if ds:
        return ds["data"], ds["name"]
    return None, ""


# ╭──────────────────────── 生成算法实现 ──────────────────────────╮
def simple_randomization(n: int, ratio: Tuple[int, int] = (1, 1),
                         seed: int | None = None) -> pd.DataFrame:
    """
    简单随机：按给定比例随机分配
    """
    np.random.seed(seed)
    g1, g2 = ratio
    choices = ["A"] * g1 + ["B"] * g2
    grp = np.random.choice(choices, size=n, replace=True)
    return pd.DataFrame({"Subject": range(1, n + 1), "Group": grp})


def permuted_block_randomization(n: int, block_size: int,
                                 ratio: Tuple[int, int] = (1, 1),
                                 seed: int | None = None) -> pd.DataFrame:
    """
    块随机（固定块长）
    """
    np.random.seed(seed)
    g1, g2 = ratio
    per_block = ["A"] * g1 + ["B"] * g2
    # 补齐块内元素
    per_block = (per_block * (block_size // len(per_block)))[:block_size]
    blocks: List[str] = []
    while len(blocks) < n:
        blk = np.random.permutation(per_block)
        blocks.extend(blk)
    grp = blocks[:n]
    return pd.DataFrame({"Subject": range(1, n + 1), "Group": grp})


def stratified_block_randomization(df: pd.DataFrame, strata_cols: List[str],
                                   block_size: int, ratio: Tuple[int, int] = (1, 1),
                                   seed: int | None = None) -> pd.DataFrame:
    """
    分层块随机：对每个层内再做块随机
    """
    np.random.seed(seed)
    results: List[pd.DataFrame] = []
    for values, sub in df.groupby(strata_cols):
        sub = sub.copy()
        sub.sort_values(by=strata_cols, inplace=True)  # 仅保证 deterministic 行号
        rand_tbl = permuted_block_randomization(
            len(sub), block_size, ratio, seed=np.random.randint(1e9)
        )
        sub["Group"] = rand_tbl["Group"].values
        results.append(sub)
    return pd.concat(results).reset_index(drop=True)


# ╭──────────────────────────── UI 主体 ───────────────────────────╮
def randomization_ui() -> None:
    st.title("🎲 随机分组生成器")
    st.markdown(
        "*支持 简单随机 / 块随机 / 分层块随机；生成的随机表可下载，并自动保存到 SessionState*"
    )

    tab_gen, tab_preview = st.tabs(["⚙️ 生成随机表", "📑 随机表预览"])

    with tab_gen:
        scheme = st.selectbox("随机化方案", ("简单随机", "块随机", "分层块随机"))

        # 受试者数量
        n = st.number_input("受试者总数 (N)", min_value=1, step=1, value=100)

        # 随机种子
        seed = st.number_input("随机种子 (可选)", step=1, value=0)
        seed = None if seed == 0 else int(seed)

        # 组别比例
        col1, col2 = st.columns(2)
        ratio_a = col1.number_input("A 组比例", min_value=1, step=1, value=1)
        ratio_b = col2.number_input("B 组比例", min_value=1, step=1, value=1)
        ratio = (int(ratio_a), int(ratio_b))

        if scheme == "简单随机":
            if st.button("生成随机表"):
                rand_df = simple_randomization(n, ratio, seed)
                save_rand_table(rand_df, "简单随机")
                st.success("已生成")

        elif scheme == "块随机":
            block_size = st.number_input("块大小", min_value=sum(ratio), step=1,
                                         value=sum(ratio))
            if st.button("生成随机表"):
                rand_df = permuted_block_randomization(
                    n, int(block_size), ratio, seed
                )
                save_rand_table(rand_df, f"块随机(block={block_size})")
                st.success("已生成")

        else:  # stratified
            df, name = get_dataset()
            if df is None:
                st.warning("需要先在数据管理中心导入数据集以供分层。")
            else:
                st.write(f"使用数据集：**{name}**")
                strata_cols = st.multiselect("选择分层变量", df.columns)
                block_size = st.number_input("块大小", min_value=sum(ratio),
                                             step=1, value=sum(ratio))
                if strata_cols and st.button("生成随机表"):
                    rand_df = stratified_block_randomization(
                        df, strata_cols, int(block_size), ratio, seed
                    )
                    save_rand_table(rand_df, f"分层块随机(block={block_size})")
                    st.success("已生成")

    # ───────────── 预览与导出 ─────────────
    with tab_preview:
        rand_df, label = get_rand_table()
        if rand_df is None:
            st.info("尚未生成随机表")
        else:
            st.subheader(f"随机表：{label}")
            st.dataframe(rand_df.head())

            # 下载按钮
            buf = io.StringIO()
            rand_df.to_csv(buf, index=False)
            st.download_button(
                "⬇️ 下载 CSV",
                data=buf.getvalue().encode(),
                file_name=f"randomization_{datetime.now():%Y%m%d%H%M%S}.csv",
                mime="text/csv",
            )


# ╭─────────────────────────── 调试入口 ──────────────────────────╮
if __name__ == "__main__":
    st.set_page_config(page_title="随机分组生成器", layout="wide")
    randomization_ui()
