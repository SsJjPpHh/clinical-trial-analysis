# data_management.py ───────────────────────────────────────────────
"""
数据管理中心（重构版）
Author : H
Date   : 2025-07-11
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from typing import Tuple, Dict, Any
from datetime import datetime

# ╭──────────────────────── 公共常量 ─────────────────────────╮
READERS: Dict[str, Any] = {
    "csv": pd.read_csv,
    "xlsx": pd.read_excel,
    "xls": pd.read_excel,
    "json": pd.read_json,
    "sav": pd.read_spss,
    "dta": pd.read_stata,
    "sas7bdat": pd.read_sas,
}


# ╭─────────────────── Cached I/O & Session Utils ──────────────────╮
@st.cache_data(show_spinner=False)
def load_file(uploaded, suffix: str) -> pd.DataFrame:
    """根据后缀读取文件"""
    read_fn = READERS[suffix]
    return read_fn(uploaded)


def set_current_dataset(df: pd.DataFrame, name: str) -> None:
    """把最新数据写入 SessionState，键名固定为 dataset_current"""
    st.session_state["dataset_current"] = {
        "name": name,
        "data": df,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_current_dataset() -> Tuple[pd.DataFrame | None, str]:
    ds = st.session_state.get("dataset_current")
    if ds:
        return ds["data"], ds["name"]
    return None, ""


# ╭───────────────────────── 标签页 1 数据导入 ──────────────────────╮
def tab_import() -> None:
    st.markdown("### 📥 数据导入")

    import_method = st.radio(
        "选择数据导入方式",
        ["📂 文件上传", "🖇️ 数据库连接", "🗂️ 示例数据", "✏️ 手动输入"],
        horizontal=True,
    )

    if import_method == "📂 文件上传":
        uploaded = st.file_uploader(
            "上传数据文件",
            type=list(READERS.keys()),
            help="支持 CSV / Excel / JSON / SAV / DTA / SAS7BDAT 等格式",
        )
        if uploaded:
            suffix = uploaded.name.split(".")[-1].lower()
            try:
                df = load_file(uploaded, suffix)
                set_current_dataset(df, uploaded.name)
                st.success(f"✅ 文件 {uploaded.name} 导入成功！")
            except Exception as e:
                st.error(f"读取失败：{e}")

    elif import_method == "🖇️ 数据库连接":
        st.info("数据库连接功能开发中，敬请期待…")

    elif import_method == "🗂️ 示例数据":
        df = px.data.tips()  # Plotly 自带示例
        set_current_dataset(df, "示例数据 tips")
        st.success("已载入示例数据 tips")

    elif import_method == "✏️ 手动输入":
        st.caption("在下方粘贴 CSV 文本：")
        txt = st.text_area("CSV 文本")
        if st.button("解析"):
            try:
                df = pd.read_csv(io.StringIO(txt))
                set_current_dataset(df, "手动输入数据")
                st.success("解析成功")
            except Exception as e:
                st.error(e)

    # 预览
    df, name = get_current_dataset()
    if df is not None:
        with st.expander(f"👀 数据预览 – {name}", expanded=False):
            st.write(df.head())


# ╭─────────────────── 标签页 2 数据探索 ──────────────────────────╮
def tab_explore() -> None:
    st.markdown("### 🔍 数据探索")
    df, _ = get_current_dataset()
    if df is None:
        st.warning("请先导入数据")
        return

    st.write("#### 1️⃣ 基本信息")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("行数", len(df))
    c2.metric("列数", len(df.columns))
    c3.metric("缺失值总数", int(df.isna().sum().sum()))
    c4.metric("重复行", int(df.duplicated().sum()))

    st.write("#### 2️⃣ 描述性统计")
    st.dataframe(df.describe(include="all").T)

    st.write("#### 3️⃣ 缺失值热图")
    if st.checkbox("显示热图"):
        fig = px.imshow(df.isna(), aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)

    st.write("#### 4️⃣ 变量分布")
    col = st.selectbox("选择列绘图", df.columns)
    if pd.api.types.is_numeric_dtype(df[col]):
        st.plotly_chart(px.histogram(df, x=col, nbins=30), use_container_width=True)
    else:
        st.plotly_chart(px.bar(df[col].value_counts().reset_index(),
                               x="index", y=col), use_container_width=True)


# ╭─────────────────── 标签页 3 数据清洗 ──────────────────────────╮
def tab_clean() -> None:
    st.markdown("### 🛠️ 数据清洗")
    df, name = get_current_dataset()
    if df is None:
        st.warning("请先导入数据")
        return

    st.write(f"当前数据集：**{name}**")

    # 缺失值处理
    st.subheader("① 缺失值处理")
    strategy = st.selectbox("选择策略", ("不处理", "删除含缺失的行", "均值填充", "中位数填充", "众数填充"))
    if st.button("执行缺失值处理"):
        if strategy == "删除含缺失的行":
            df = df.dropna()
        elif strategy in ("均值填充", "中位数填充", "众数填充"):
            for col in df.columns:
                if df[col].isna().any():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        val = (
                            df[col].mean() if strategy == "均值填充"
                            else df[col].median()
                        )
                    else:
                        val = df[col].mode().iloc[0]
                    df[col].fillna(val, inplace=True)
        set_current_dataset(df, name + " (cleaned)")
        st.success("缺失值处理完成 ✅")

    # 重复值处理
    st.subheader("② 重复行处理")
    if st.button("删除重复行"):
        df = df.drop_duplicates()
        set_current_dataset(df, name + " (cleaned)")
        st.success("重复行已删除")

    # 异常值（IQR）清理
    st.subheader("③ 异常值处理（IQR）")
    num_cols = df.select_dtypes("number").columns.tolist()
    target_col = st.selectbox("选择数值列", num_cols)
    if st.button("去除异常值"):
        q1, q3 = df[target_col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = df[target_col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        df = df[mask]
        set_current_dataset(df, name + " (cleaned)")
        st.success("异常值已删除")

    with st.expander("当前数据快照"):
        st.write(df.head())


# ╭─────────────────── 标签页 4 变量管理 ──────────────────────────╮
def tab_variables() -> None:
    st.markdown("### 📝 变量管理")
    df, name = get_current_dataset()
    if df is None:
        st.warning("请先导入数据")
        return

    st.write(f"数据集：**{name}**")

    # 字段重命名
    col_to_rename = st.selectbox("选择列重命名", df.columns)
    new_name = st.text_input("新列名")
    if st.button("重命名"):
        if new_name:
            df.rename(columns={col_to_rename: new_name}, inplace=True)
            set_current_dataset(df, name)
            st.success("已重命名")

    # 类型转换
    col_to_convert = st.selectbox("选择列转换类型", df.columns, key="convert")
    new_type = st.selectbox("目标类型", ("字符串", "分类", "整数", "浮点", "日期"))
    if st.button("转换"):
        try:
            if new_type == "字符串":
                df[col_to_convert] = df[col_to_convert].astype(str)
            elif new_type == "分类":
                df[col_to_convert] = df[col_to_convert].astype("category")
            elif new_type == "整数":
                df[col_to_convert] = pd.to_numeric(df[col_to_convert]).astype("Int64")
            elif new_type == "浮点":
                df[col_to_convert] = pd.to_numeric(df[col_to_convert]).astype(float)
            elif new_type == "日期":
                df[col_to_convert] = pd.to_datetime(df[col_to_convert])
            set_current_dataset(df, name)
            st.success("类型转换成功")
        except Exception as e:
            st.error(e)

    with st.expander("字段信息"):
        st.write(df.dtypes)


# ╭─────────────────── 标签页 5 数据导出 ──────────────────────────╮
def tab_export() -> None:
    st.markdown("### 💾 数据导出")
    df, name = get_current_dataset()
    if df is None:
        st.warning("暂无可导出的数据")
        return

    file_fmt = st.selectbox("选择格式", ("csv", "xlsx"))
    if file_fmt == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        bytes_data = buf.getvalue().encode()
    else:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False)
        bytes_data = buf.getvalue()

    st.download_button(
        "⬇️ 点击下载",
        data=bytes_data,
        file_name=f"{name}_{datetime.now():%Y%m%d%H%M%S}.{file_fmt}",
        mime="text/csv" if file_fmt == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ╭─────────────────────────── 主界面 ───────────────────────────╮
def data_management_ui() -> None:
    st.title("📊 数据管理中心")
    st.markdown("*专业的数据导入、清洗、探索和管理工具*")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📥 数据导入", "🔍 数据探索", "🛠️ 数据清洗", "📝 变量管理", "💾 数据导出"]
    )

    with tab1:
        tab_import()
    with tab2:
        tab_explore()
    with tab3:
        tab_clean()
    with tab4:
        tab_variables()
    with tab5:
        tab_export()


# ╭─────────────────────────── 调试 ────────────────────────────╮
if __name__ == "__main__":
    st.set_page_config(page_title="数据管理中心", layout="wide")
    data_management_ui()
# data_management.py ───────────────────────────────────────────────
"""
数据管理中心（重构版）
Author : Your Name
Date   : 2025-07-11
"""

from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import io
from typing import Tuple, Dict, Any
from datetime import datetime

# ╭──────────────────────── 公共常量 ─────────────────────────╮
READERS: Dict[str, Any] = {
    "csv": pd.read_csv,
    "xlsx": pd.read_excel,
    "xls": pd.read_excel,
    "json": pd.read_json,
    "sav": pd.read_spss,
    "dta": pd.read_stata,
    "sas7bdat": pd.read_sas,
}


# ╭─────────────────── Cached I/O & Session Utils ──────────────────╮
@st.cache_data(show_spinner=False)
def load_file(uploaded, suffix: str) -> pd.DataFrame:
    """根据后缀读取文件"""
    read_fn = READERS[suffix]
    return read_fn(uploaded)


def set_current_dataset(df: pd.DataFrame, name: str) -> None:
    """把最新数据写入 SessionState，键名固定为 dataset_current"""
    st.session_state["dataset_current"] = {
        "name": name,
        "data": df,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def get_current_dataset() -> Tuple[pd.DataFrame | None, str]:
    ds = st.session_state.get("dataset_current")
    if ds:
        return ds["data"], ds["name"]
    return None, ""


# ╭───────────────────────── 标签页 1 数据导入 ──────────────────────╮
def tab_import() -> None:
    st.markdown("### 📥 数据导入")

    import_method = st.radio(
        "选择数据导入方式",
        ["📂 文件上传", "🖇️ 数据库连接", "🗂️ 示例数据", "✏️ 手动输入"],
        horizontal=True,
    )

    if import_method == "📂 文件上传":
        uploaded = st.file_uploader(
            "上传数据文件",
            type=list(READERS.keys()),
            help="支持 CSV / Excel / JSON / SAV / DTA / SAS7BDAT 等格式",
        )
        if uploaded:
            suffix = uploaded.name.split(".")[-1].lower()
            try:
                df = load_file(uploaded, suffix)
                set_current_dataset(df, uploaded.name)
                st.success(f"✅ 文件 {uploaded.name} 导入成功！")
            except Exception as e:
                st.error(f"读取失败：{e}")

    elif import_method == "🖇️ 数据库连接":
        st.info("数据库连接功能开发中，敬请期待…")

    elif import_method == "🗂️ 示例数据":
        df = px.data.tips()  # Plotly 自带示例
        set_current_dataset(df, "示例数据 tips")
        st.success("已载入示例数据 tips")

    elif import_method == "✏️ 手动输入":
        st.caption("在下方粘贴 CSV 文本：")
        txt = st.text_area("CSV 文本")
        if st.button("解析"):
            try:
                df = pd.read_csv(io.StringIO(txt))
                set_current_dataset(df, "手动输入数据")
                st.success("解析成功")
            except Exception as e:
                st.error(e)

    # 预览
    df, name = get_current_dataset()
    if df is not None:
        with st.expander(f"👀 数据预览 – {name}", expanded=False):
            st.write(df.head())


# ╭─────────────────── 标签页 2 数据探索 ──────────────────────────╮
def tab_explore() -> None:
    st.markdown("### 🔍 数据探索")
    df, _ = get_current_dataset()
    if df is None:
        st.warning("请先导入数据")
        return

    st.write("#### 1️⃣ 基本信息")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("行数", len(df))
    c2.metric("列数", len(df.columns))
    c3.metric("缺失值总数", int(df.isna().sum().sum()))
    c4.metric("重复行", int(df.duplicated().sum()))

    st.write("#### 2️⃣ 描述性统计")
    st.dataframe(df.describe(include="all").T)

    st.write("#### 3️⃣ 缺失值热图")
    if st.checkbox("显示热图"):
        fig = px.imshow(df.isna(), aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig, use_container_width=True)

    st.write("#### 4️⃣ 变量分布")
    col = st.selectbox("选择列绘图", df.columns)
    if pd.api.types.is_numeric_dtype(df[col]):
        st.plotly_chart(px.histogram(df, x=col, nbins=30), use_container_width=True)
    else:
        st.plotly_chart(px.bar(df[col].value_counts().reset_index(),
                               x="index", y=col), use_container_width=True)


# ╭─────────────────── 标签页 3 数据清洗 ──────────────────────────╮
def tab_clean() -> None:
    st.markdown("### 🛠️ 数据清洗")
    df, name = get_current_dataset()
    if df is None:
        st.warning("请先导入数据")
        return

    st.write(f"当前数据集：**{name}**")

    # 缺失值处理
    st.subheader("① 缺失值处理")
    strategy = st.selectbox("选择策略", ("不处理", "删除含缺失的行", "均值填充", "中位数填充", "众数填充"))
    if st.button("执行缺失值处理"):
        if strategy == "删除含缺失的行":
            df = df.dropna()
        elif strategy in ("均值填充", "中位数填充", "众数填充"):
            for col in df.columns:
                if df[col].isna().any():
                    if pd.api.types.is_numeric_dtype(df[col]):
                        val = (
                            df[col].mean() if strategy == "均值填充"
                            else df[col].median()
                        )
                    else:
                        val = df[col].mode().iloc[0]
                    df[col].fillna(val, inplace=True)
        set_current_dataset(df, name + " (cleaned)")
        st.success("缺失值处理完成 ✅")

    # 重复值处理
    st.subheader("② 重复行处理")
    if st.button("删除重复行"):
        df = df.drop_duplicates()
        set_current_dataset(df, name + " (cleaned)")
        st.success("重复行已删除")

    # 异常值（IQR）清理
    st.subheader("③ 异常值处理（IQR）")
    num_cols = df.select_dtypes("number").columns.tolist()
    target_col = st.selectbox("选择数值列", num_cols)
    if st.button("去除异常值"):
        q1, q3 = df[target_col].quantile([0.25, 0.75])
        iqr = q3 - q1
        mask = df[target_col].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        df = df[mask]
        set_current_dataset(df, name + " (cleaned)")
        st.success("异常值已删除")

    with st.expander("当前数据快照"):
        st.write(df.head())


# ╭─────────────────── 标签页 4 变量管理 ──────────────────────────╮
def tab_variables() -> None:
    st.markdown("### 📝 变量管理")
    df, name = get_current_dataset()
    if df is None:
        st.warning("请先导入数据")
        return

    st.write(f"数据集：**{name}**")

    # 字段重命名
    col_to_rename = st.selectbox("选择列重命名", df.columns)
    new_name = st.text_input("新列名")
    if st.button("重命名"):
        if new_name:
            df.rename(columns={col_to_rename: new_name}, inplace=True)
            set_current_dataset(df, name)
            st.success("已重命名")

    # 类型转换
    col_to_convert = st.selectbox("选择列转换类型", df.columns, key="convert")
    new_type = st.selectbox("目标类型", ("字符串", "分类", "整数", "浮点", "日期"))
    if st.button("转换"):
        try:
            if new_type == "字符串":
                df[col_to_convert] = df[col_to_convert].astype(str)
            elif new_type == "分类":
                df[col_to_convert] = df[col_to_convert].astype("category")
            elif new_type == "整数":
                df[col_to_convert] = pd.to_numeric(df[col_to_convert]).astype("Int64")
            elif new_type == "浮点":
                df[col_to_convert] = pd.to_numeric(df[col_to_convert]).astype(float)
            elif new_type == "日期":
                df[col_to_convert] = pd.to_datetime(df[col_to_convert])
            set_current_dataset(df, name)
            st.success("类型转换成功")
        except Exception as e:
            st.error(e)

    with st.expander("字段信息"):
        st.write(df.dtypes)


# ╭─────────────────── 标签页 5 数据导出 ──────────────────────────╮
def tab_export() -> None:
    st.markdown("### 💾 数据导出")
    df, name = get_current_dataset()
    if df is None:
        st.warning("暂无可导出的数据")
        return

    file_fmt = st.selectbox("选择格式", ("csv", "xlsx"))
    if file_fmt == "csv":
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        bytes_data = buf.getvalue().encode()
    else:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False)
        bytes_data = buf.getvalue()

    st.download_button(
        "⬇️ 点击下载",
        data=bytes_data,
        file_name=f"{name}_{datetime.now():%Y%m%d%H%M%S}.{file_fmt}",
        mime="text/csv" if file_fmt == "csv" else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ╭─────────────────────────── 主界面 ───────────────────────────╮
def data_management_ui() -> None:
    st.title("📊 数据管理中心")
    st.markdown("*专业的数据导入、清洗、探索和管理工具*")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📥 数据导入", "🔍 数据探索", "🛠️ 数据清洗", "📝 变量管理", "💾 数据导出"]
    )

    with tab1:
        tab_import()
    with tab2:
        tab_explore()
    with tab3:
        tab_clean()
    with tab4:
        tab_variables()
    with tab5:
        tab_export()


# ╭─────────────────────────── 调试 ────────────────────────────╮
if __name__ == "__main__":
    st.set_page_config(page_title="数据管理中心", layout="wide")
    data_management_ui()
