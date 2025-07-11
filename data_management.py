# data_management.py
"""
数据管理中心 (重构版)

Streamlit 入口示例：
-------------------------------------------
import streamlit as st
from data_management import data_management_ui
data_management_ui()
-------------------------------------------
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import io
import json
from datetime import datetime
from typing import Dict, List, Tuple
import plotly.express as px
import plotly.graph_objects as go


# ============ Session 数据集管理 ============ #

def _session_dataset_key(name: str) -> str:
    return f"dataset_{name}"


def save_dataset_to_session(name: str, df: pd.DataFrame) -> None:
    st.session_state[_session_dataset_key(name)] = {
        "name": name,
        "data": df,
        "upload_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def delete_dataset_from_session(name: str) -> None:
    st.session_state.pop(_session_dataset_key(name), None)


def list_datasets() -> Dict[str, pd.DataFrame]:
    ds = {}
    for k, v in st.session_state.items():
        if k.startswith("dataset_") and isinstance(v, dict) and "data" in v:
            ds[v["name"]] = v["data"]
    return ds


# ============ 文件读取工具 ============ #

@st.cache_data(show_spinner=False)
def _read_uploaded_file(file) -> pd.DataFrame | None:
    """
    根据扩展名自动解析，返回 DataFrame
    支持 csv / tsv / xlsx / json / sav / dta / sas7bdat / parquet
    """
    name = file.name.lower()
    try:
        if name.endswith(".csv"):
            return pd.read_csv(file)
        if name.endswith((".tsv", ".txt")):
            return pd.read_csv(file, sep="\t")
        if name.endswith((".xlsx", ".xls")):
            return pd.read_excel(file)
        if name.endswith(".json"):
            return pd.read_json(file)
        if name.endswith(".parquet"):
            return pd.read_parquet(file)
        if name.endswith(".sav"):
            return pd.read_spss(file)
        if name.endswith(".dta"):
            return pd.read_stata(file)
        if name.endswith((".sas7bdat", ".sas")):
            return pd.read_sas(file)
    except Exception as e:
        st.error(f"❌ 文件解析失败: {e}")
    return None


# ============ 1. 数据导入 ============ #

def data_import_section() -> None:
    st.markdown("### 📥 数据导入")

    uploaded_file = st.file_uploader(
        "选择数据文件",
        type=[
            "csv",
            "tsv",
            "txt",
            "xlsx",
            "xls",
            "json",
            "sav",
            "dta",
            "sas7bdat",
            "parquet",
        ],
        help="支持 CSV/TSV、Excel、JSON、SPSS、Stata、SAS、Parquet 等格式",
    )

    if uploaded_file is not None:
        df = _read_uploaded_file(uploaded_file)
        if df is not None:
            st.success(f"✅ 文件读取成功！共 {df.shape[0]} 行 {df.shape[1]} 列。")
            st.dataframe(df.head())
            default_name = uploaded_file.name.split(".")[0]
            new_name = st.text_input("为数据集命名", value=default_name)
            if st.button("💾 保存到会话"):
                save_dataset_to_session(new_name, df)
                st.success("已保存！")
    st.markdown("---")
    st.markdown("#### 已加载数据集")
    show_loaded_datasets()


def show_loaded_datasets() -> None:
    datasets = list_datasets()
    if not datasets:
        st.info("暂无数据集")
        return
    for name, df in datasets.items():
        with st.expander(f"📂 {name} ({df.shape[0]}×{df.shape[1]})"):
            st.dataframe(df.head())
            col1, col2 = st.columns(2)
            with col1:
                csv = df.to_csv(index=False).encode("utf-8-sig")
                st.download_button("⬇️ 下载 CSV", csv, file_name=f"{name}.csv", mime="text/csv")
            with col2:
                if st.button("🗑️ 删除", key=f"del_{name}"):
                    delete_dataset_from_session(name)
                    st.experimental_rerun()


# ============ 2. 数据探索 ============ #

def data_exploration_section() -> None:
    st.markdown("### 🔎 数据探索")
    datasets = list_datasets()
    if not datasets:
        st.info("请先在『数据导入』中加载数据")
        return

    name = st.selectbox("选择数据集", list(datasets.keys()))
    df = datasets[name]

    sub_tabs = st.tabs(["👁️ 概览", "📈 分布", "🔗 相关性"])
    # --- 概览
    with sub_tabs[0]:
        st.write("#### 基本信息")
        st.write(df.describe(include="all").T)
        st.write("#### 缺失值概览")
        miss = df.isna().sum().to_frame("缺失数")
        miss["缺失率"] = (miss["缺失数"] / len(df)).round(3)
        st.dataframe(miss)

    # --- 分布
    with sub_tabs[1]:
        col = st.selectbox("选择变量绘制分布图", df.columns)
        if pd.api.types.is_numeric_dtype(df[col]):
            fig = px.histogram(df, x=col, marginal="box", nbins=30)
        else:
            fig = px.histogram(df, x=col, color=col)
        st.plotly_chart(fig, use_container_width=True)

    # --- 相关性
    with sub_tabs[2]:
        num_cols = df.select_dtypes("number").columns
        if len(num_cols) < 2:
            st.info("数值变量不足 2 个，无法绘制相关性热图")
        else:
            corr = df[num_cols].corr()
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                origin="lower",
                title="相关系数热图",
            )
            st.plotly_chart(fig, use_container_width=True)


# ============ 3. 数据清洗 ============ #

def data_cleaning_section() -> None:
    st.markdown("### 🧹 数据清洗")

    datasets = list_datasets()
    if not datasets:
        st.info("请先加载数据")
        return
    name = st.selectbox("选择要清洗的数据集", list(datasets.keys()), key="clean_ds")
    df = datasets[name].copy()

    st.markdown("#### 缺失值处理")
    miss_cols = df.columns[df.isna().any()].tolist()
    if miss_cols:
        with st.expander(f"有缺失值的列 ({len(miss_cols)})", expanded=False):
            st.dataframe(df[miss_cols].isna().sum())

        col_sel = st.multiselect("选择要填充缺失值的列", miss_cols)
        fill_strategy = st.selectbox("填充策略", ["均值", "中位数", "众数", "常数"])
        const_val = None
        if fill_strategy == "常数":
            const_val = st.text_input("填充值", "0")
        if st.button("🩹 执行填充"):
            for c in col_sel:
                if fill_strategy == "均值":
                    df[c].fillna(df[c].mean(), inplace=True)
                elif fill_strategy == "中位数":
                    df[c].fillna(df[c].median(), inplace=True)
                elif fill_strategy == "众数":
                    df[c].fillna(df[c].mode().iloc[0], inplace=True)
                else:
                    df[c].fillna(const_val, inplace=True)
            st.success("缺失值填充完成")

    st.markdown("#### 重复值处理")
    dup_count = df.duplicated().sum()
    st.write(f"检测到 {dup_count} 行重复")
    if dup_count > 0 and st.button("🚮 删除重复行"):
        df.drop_duplicates(inplace=True)
        st.success("已删除重复行")

    st.markdown("---")
    new_name = st.text_input("为清洗后的数据集命名", value=f"{name}_clean")
    if st.button("💾 保存清洗结果"):
        save_dataset_to_session(new_name, df)
        st.success("保存成功！")


# ============ 4. 变量管理 ============ #

def variable_management_section() -> None:
    st.markdown("### 📝 变量管理")

    datasets = list_datasets()
    if not datasets:
        st.info("请先加载数据")
        return
    name = st.selectbox("选择数据集", list(datasets.keys()), key="var_ds")
    df = datasets[name].copy()

    st.markdown("#### 重命名变量")
    col1, col2 = st.columns(2)
    with col1:
        old_name = st.selectbox("原变量名", df.columns)
    with col2:
        new_name = st.text_input("新变量名")
    if st.button("✏️ 重命名"):
        if new_name:
            df.rename(columns={old_name: new_name}, inplace=True)
            st.success("已重命名")

    st.markdown("#### 变量类型转换")
    col = st.selectbox("选择变量", df.columns, key="dtype_col")
    target_type = st.selectbox("转换为", ["数值", "分类", "日期"])
    if st.button("🔄 转换类型"):
        try:
            if target_type == "数值":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif target_type == "分类":
                df[col] = df[col].astype("category")
            else:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            st.success("类型转换成功")
        except Exception as e:
            st.error(f"转换失败: {e}")

    st.markdown("---")
    new_name2 = st.text_input("保存为新数据集名", value=f"{name}_var")
    if st.button("💾 保存变量管理结果"):
        save_dataset_to_session(new_name2, df)
        st.success("保存成功！")


# ============ 5. 数据导出 ============ #

def data_export_section() -> None:
    st.markdown("### 📤 数据导出")

    datasets = list_datasets()
    if not datasets:
        st.info("请先加载数据")
        return
    name = st.selectbox("选择数据集", list(datasets.keys()), key="export_ds")
    df = datasets[name]

    fmt = st.selectbox("导出格式", ["csv", "xlsx", "json", "parquet"])
    if st.button("⬇️ 生成文件"):
        try:
            if fmt == "csv":
                data = df.to_csv(index=False).encode("utf-8-sig")
            elif fmt == "xlsx":
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                    df.to_excel(writer, index=False)
                data = buffer.getvalue()
            elif fmt == "json":
                data = df.to_json(orient="records").encode()
            else:
                buffer = io.BytesIO()
                df.to_parquet(buffer, index=False)
                data = buffer.getvalue()

            st.download_button(
                label="📥 点击下载",
                data=data,
                file_name=f"{name}.{fmt}",
                mime="application/octet-stream",
            )
        except Exception as e:
            st.error(f"导出失败: {e}")


# ============ 主 UI ============ #

def data_management_ui() -> None:
    st.set_page_config(page_title="数据管理中心", page_icon="📊", layout="wide")
    st.markdown("# 📊 数据管理中心")
    st.markdown("*专业的数据导入、清洗、探索和管理工具*")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📥 数据导入", "🔎 数据探索", "🧹 数据清洗", "📝 变量管理", "📤 数据导出"]
    )

    with tab1:
        data_import_section()
    with tab2:
        data_exploration_section()
    with tab3:
        data_cleaning_section()
    with tab4:
        variable_management_section()
    with tab5:
        data_export_section()


# ============ 入口保护 ============ #

if __name__ == "__main__":
    data_management_ui()
