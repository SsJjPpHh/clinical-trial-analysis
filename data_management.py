import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

def data_import_ui():
    st.header("📁 数据导入")
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "选择数据文件",
        type=['csv', 'xlsx', 'xls'],
        help="支持CSV、Excel格式文件"
    )
    
    if uploaded_file is not None:
        # 文件类型检测
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # 导入选项
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if file_extension == 'csv':
                separator = st.selectbox("分隔符", [',', ';', '\t'], index=0)
            else:
                separator = ','
                
        with col2:
            header_row = st.number_input("标题行", min_value=0, value=0)
            
        with col3:
            encoding = st.selectbox("编码", ['utf-8', 'gbk', 'gb2312'], index=0)
        
        try:
            # 读取数据
            if file_extension == 'csv':
                df = pd.read_csv(uploaded_file, sep=separator, header=header_row, encoding=encoding)
            else:
                df = pd.read_excel(uploaded_file, header=header_row)
            
            # 数据预览
            st.subheader("📊 数据预览")
            st.write(f"数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
            st.dataframe(df.head(10))
            
            # 数据信息
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 数据类型")
                dtype_df = pd.DataFrame({
                    '变量名': df.columns,
                    '数据类型': df.dtypes.astype(str),
                    '非空值数': df.count(),
                    '缺失值数': df.isnull().sum()
                })
                st.dataframe(dtype_df)
            
            with col2:
                st.subheader("📋 基本统计")
                st.dataframe(df.describe())
            
            # 导入按钮
            if st.button("✅ 导入数据", type="primary"):
                st.session_state.raw_data = df
                st.session_state.cleaned_data = df.copy()
                st.success(f"数据导入成功！共 {df.shape[0]} 行，{df.shape[1]} 列")
                
        except Exception as e:
            st.error(f"数据导入失败: {str(e)}")

def data_cleaning_ui():
    st.header("🧹 数据清理")
    
    if st.session_state.raw_data is None:
        st.warning("请先导入数据")
        return
    
    df = st.session_state.raw_data.copy()
    
    # 缺失值处理
    st.subheader("🔍 缺失值处理")
    
    # 显示缺失值情况
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
    
    if len(missing_data) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("缺失值统计:")
            missing_df = pd.DataFrame({
                '变量': missing_data.index,
                '缺失数': missing_data.values,
                '缺失率(%)': (missing_data.values / len(df) * 100).round(2)
            })
            st.dataframe(missing_df)
        
        with col2:
            # 缺失值可视化
            fig = px.bar(missing_df, x='变量', y='缺失率(%)', 
                        title="缺失值分布")
            st.plotly_chart(fig, use_container_width=True)
        
        # 缺失值处理选项
        missing_method = st.selectbox(
            "选择缺失值处理方法",
            ["保持原样", "删除含缺失值的行", "删除含缺失值的列", "用均值填充", "用中位数填充", "用众数填充"]
        )
        
        if missing_method != "保持原样":
            if st.button("应用缺失值处理"):
                if missing_method == "删除含缺失值的行":
                    df = df.dropna()
                elif missing_method == "删除含缺失值的列":
                    df = df.dropna(axis=1)
                elif missing_method == "用均值填充":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                elif missing_method == "用中位数填充":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                elif missing_method == "用众数填充":
                    for col in df.columns:
                        df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else df[col])
                
                st.success(f"缺失值处理完成！数据维度: {df.shape}")
    else:
        st.info("数据中没有缺失值")
    
    # 异常值检测
    st.subheader("🎯 异常值检测")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        selected_col = st.selectbox("选择要检测异常值的变量", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # 箱线图
            fig = px.box(df, y=selected_col, title=f"{selected_col} 箱线图")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 直方图
            fig = px.histogram(df, x=selected_col, title=f"{selected_col} 分布")
            st.plotly_chart(fig, use_container_width=True)
        
        # IQR方法检测异常值
        Q1 = df[selected_col].quantile(0.25)
        Q3 = df[selected_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
        
        st.write(f"检测到 {len(outliers)} 个异常值")
        
        if len(outliers) > 0:
            outlier_method = st.selectbox(
                "异常值处理方法",
                ["保持原样", "删除异常值", "用边界值替换"]
            )
            
            if outlier_method != "保持原样":
                if st.button("应用异常值处理"):
                    if outlier_method == "删除异常值":
                        df = df[(df[selected_col] >= lower_bound) & (df[selected_col] <= upper_bound)]
                    elif outlier_method == "用边界值替换":
                        df.loc[df[selected_col] < lower_bound, selected_col] = lower_bound
                        df.loc[df[selected_col] > upper_bound, selected_col] = upper_bound
                    
                    st.success("异常值处理完成！")
    
    # 数据类型转换
    st.subheader("🔄 数据类型转换")
    
    col1, col2 = st.columns(2)
    
    with col1:
        convert_col = st.selectbox("选择要转换的变量", df.columns)
        
    with col2:
        new_type = st.selectbox(
            "目标数据类型",
            ["int64", "float64", "object", "datetime64", "category"]
        )
    
    if st.button("转换数据类型"):
        try:
            if new_type == "datetime64":
                df[convert_col] = pd.to_datetime(df[convert_col])
            elif new_type == "category":
                df[convert_col] = df[convert_col].astype('category')
            else:
                df[convert_col] = df[convert_col].astype(new_type)
            
            st.success(f"{convert_col} 已转换为 {new_type}")
        except Exception as e:
            st.error(f"转换失败: {str(e)}")
    
    # 保存清理后的数据
    if st.button("💾 保存清理后的数据", type="primary"):
        st.session_state.cleaned_data = df
        st.success("数据清理完成并已保存！")

def data_exploration_ui():
    st.header("🔍 数据探索")
    
    if st.session_state.cleaned_data is None:
        st.warning("请先导入并清理数据")
        return
    
    df = st.session_state.cleaned_data
    
    # 数据概览
    st.subheader("📊 数据概览")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总行数", df.shape[0])
    with col2:
        st.metric("总列数", df.shape[1])
    with col3:
        st.metric("数值变量", len(df.select_dtypes(include=[np.number]).columns))
    with col4:
        st.metric("分类变量", len(df.select_dtypes(include=['object', 'category']).columns))
    
    # 变量分析
    st.subheader("📈 变量分析")
    
    analysis_type = st.selectbox(
        "选择分析类型",
        ["单变量分析", "双变量分析", "相关性分析"]
    )
    
    if analysis_type == "单变量分析":
        selected_var = st.selectbox("选择变量", df.columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            if df[selected_var].dtype in ['int64', 'float64']:
                # 数值变量
                st.write("**描述统计:**")
                st.write(df[selected_var].describe())
                
                # 直方图
                fig = px.histogram(df, x=selected_var, title=f"{selected_var} 分布")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # 分类变量
                value_counts = df[selected_var].value_counts()
                st.write("**频数统计:**")
                st.write(value_counts)
                
                # 饼图
                fig = px.pie(values=value_counts.values, names=value_counts.index, 
                           title=f"{selected_var} 分布")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if df[selected_var].dtype in ['int64', 'float64']:
                # 箱线图
                fig = px.box(df, y=selected_var, title=f"{selected_var} 箱线图")
                st.plotly_chart(fig, use_container_width=True)
            else:
                # 条形图
                fig = px.bar(x=value_counts.index, y=value_counts.values, 
                           title=f"{selected_var} 频数")
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "双变量分析":
        col1, col2 = st.columns(2)
        
        with col1:
            var1 = st.selectbox("选择变量1", df.columns, key="var1")
        with col2:
            var2 = st.selectbox("选择变量2", df.columns, key="var2")
        
        if var1 != var2:
            # 判断变量类型并选择合适的可视化
            var1_numeric = df[var1].dtype in ['int64', 'float64']
            var2_numeric = df[var2].dtype in ['int64', 'float64']
            
            if var1_numeric and var2_numeric:
                # 两个数值变量：散点图
                fig = px.scatter(df, x=var1, y=var2, title=f"{var1} vs {var2}")
                st.plotly_chart(fig, use_container_width=True)
                
                # 计算相关系数
                corr = df[var1].corr(df[var2])
                st.write(f"**相关系数:** {corr:.4f}")
                
            elif var1_numeric and not var2_numeric:
                # 数值 vs 分类：箱线图
                fig = px.box(df, x=var2, y=var1, title=f"{var1} by {var2}")
                st.plotly_chart(fig, use_container_width=True)
                
            elif not var1_numeric and var2_numeric:
                # 分类 vs 数值：箱线图
                fig = px.box(df, x=var1, y=var2, title=f"{var2} by {var1}")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                # 两个分类变量：交叉表
                crosstab = pd.crosstab(df[var1], df[var2])
                st.write("**交叉表:**")
                st.dataframe(crosstab)
                
                # 热力图
                fig = px.imshow(crosstab, title=f"{var1} vs {var2} 交叉表")
                st.plotly_chart(fig, use_container_width=True)
    
    elif analysis_type == "相关性分析":
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 1:
            # 相关性矩阵
            corr_matrix = df[numeric_cols].corr()
            
            # 热力图
            fig = px.imshow(corr_matrix, 
                          x=corr_matrix.columns, 
                          y=corr_matrix.columns,
                          color_continuous_scale='RdBu',
                          title="变量相关性矩阵")
            st.plotly_chart(fig, use_container_width=True)
            
            # 显示相关性表格
            st.subheader("相关性系数表")
            st.dataframe(corr_matrix)
        else:
            st.warning("需要至少2个数值变量进行相关性分析")
