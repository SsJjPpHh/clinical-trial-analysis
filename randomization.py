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

# ---------------- 随机化核心 ---------------- #
def simple_randomization(n: int, arms: List[str], seed: int | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    alloc = rng.choice(arms, n)
    return pd.DataFrame({"id": np.arange(1, n + 1), "treatment": alloc})

def blocked_randomization(
    n: int, arms: List[str], block_size: int, seed: int | None = None
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if block_size % len(arms) != 0:
        raise ValueError("块长必须能被组数整除")
    seq = []
    while len(seq) < n:
        block = np.repeat(arms, block_size // len(arms))
        rng.shuffle(block)
        seq.extend(block)
    alloc = seq[:n]
    return pd.DataFrame({"id": np.arange(1, n + 1), "treatment": alloc})

def stratified_block_randomization(
    df: pd.DataFrame, strat_cols: List[str], arms: List[str], block_size: int, seed: int | None = None
) -> pd.DataFrame:
    """
    df 必须包含待随机化受试者，每行为 1 名受试者，strat_cols 为分层列
    """
    rng = np.random.default_rng(seed)
    if block_size % len(arms) != 0:
        raise ValueError("块长必须能被组数整除")
    alloc_list = []
    for _, sub in df.groupby(strat_cols):
        n = len(sub)
        sub_df = blocked_randomization(n, arms, block_size, seed=rng.integers(1e9))
        sub_df.index = sub.index
        alloc_list.append(sub_df)
    alloc = pd.concat(alloc_list).sort_index()
    result = df.copy()
    result["treatment"] = alloc["treatment"]
    return result.reset_index(drop=True)

# ---------------- UI ---------------- #
def randomization_ui() -> None:
    st.set_page_config("随机化生成", "🎲", layout="wide")
    st.markdown("# 🎲 随机化列表生成")

    st.sidebar.markdown("## 随机化设置")
    rand_type = st.sidebar.radio("随机化类型", ["Simple", "Blocked", "Stratified Block"])
    arms_num = st.sidebar.number_input("受试组数", min_value=2, max_value=6, value=2, step=1)
    arms_names = [f"A{i+1}" for i in range(arms_num)]
    arms_names = st.sidebar.text_input("各组名称（以逗号分隔）", ",".join(arms_names)).split(",")
    arms_names = [a.strip() for a in arms_names if a.strip()]
    seed = st.sidebar.number_input("随机种子 (可选)", value=0, step=1)

    if rand_type == "Simple":
        n = st.number_input("随机化总例数", min_value=2, value=60, step=1)
        if st.button("🚀 生成随机化表"):
            df = simple_randomization(n, arms_names, seed or None)
            st.success("生成完成！")
            st.dataframe(df.head(20))
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button("📥 下载 CSV", csv, "randomization.csv", "text/csv")
            if st.checkbox("保存到会话", value=True):
                save_dataset("randomization", df)

    elif rand_type == "Blocked":
        n = st.number_input("随机化总例数", min_value=2, value=60, step=1)
        block_size = st.number_input("块长", min_value=len(arms_names), value=len(arms_names) * 2, step=len(arms_names))
        if st.button("🚀 生成随机化表"):
            try:
                df = blocked_randomization(n, arms_names, block_size, seed or None)
                st.dataframe(df.head(20))
                csv = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("📥 下载 CSV", csv, "randomization.csv", "text/csv")
                if st.checkbox("保存到会话", value=True):
                    save_dataset("randomization", df)
            except Exception as e:
                st.error(f"生成失败: {e}")

    else:  # Stratified
        datasets = list_datasets()
        if not datasets:
            st.warning("请先在数据管理页导入待随机化人员列表")
            return
        src_name = st.selectbox("选择人员数据集", list(datasets.keys()))
        df_people = datasets[src_name]
        strat_cols = st.multiselect("选择分层变量", df_people.columns.tolist())
        block_size = st.number_input("块长", min_value=len(arms_names), value=len(arms_names) * 2, step=len(arms_names))
        if strat_cols and st.button("🚀 生成随机化表"):
            try:
                df = stratified_block_randomization(df_people, strat_cols, arms_names, block_size, seed or None)
                st.dataframe(df.head())
                csv = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("📥 下载 CSV", csv, "randomization.csv", "text/csv")
                if st.checkbox("保存到会话", value=True):
                    save_dataset("randomization", df)
            except Exception as e:
                st.error(f"生成失败: {e}")

# 入口
if __name__ == "__main__":
    randomization_ui()
