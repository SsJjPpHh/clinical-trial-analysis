import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def data_management_ui():
    """数据管理界面 - 商业化升级版"""
    st.markdown("## 📊 数据管理中心")
    st.markdown("*专业的数据导入、清洗、探索和管理工具*")
    
    # 创建标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📤 数据导入", "🔍 数据探索", "🛠️ 数据清洗", 
        "📋 变量管理", "💾 数据导出"
    ])
    
    with tab1:
        data_import_section()
    
    with tab2:
        data_exploration_section()
    
    with tab3:
        data_cleaning_section
    
    with tab4:
        variable_management_section()
    
    with tab5:
        data_export_section()

def data_import_section():
    """数据导入部分"""
    st.markdown("### 📤 数据导入")
    
    # 导入方式选择
    import_method = st.radio(
        "选择数据导入方式",
        ["📁 文件上传", "🔗 数据库连接", "🌐 示例数据", "✏️ 手动输入"],
        horizontal=True
    )
    
    if import_method == "📁 文件上传":
        file_upload_interface()
    elif import_method == "🔗 数据库连接":
        database_connection_interface()
    elif import_method == "🌐 示例数据":
        sample_data_interface()
    elif import_method == "✏️ 手动输入":
        manual_input_interface()

def file_upload_interface():
    """文件上传界面"""
    st.markdown("#### 📁 文件上传")
    
    # 支持的文件格式信息
    with st.expander("📋 支持的文件格式", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **表格文件:**
            - 📊 Excel (.xlsx, .xls)
            - 📄 CSV (.csv)
            - 📋 TSV (.tsv)
            """)
        with col2:
            st.markdown("""
            **统计软件:**
            - 🔢 SPSS (.sav)
            - 📈 Stata (.dta)
            - 🅰️ SAS (.sas7bdat)
            """)
        with col3:
            st.markdown("""
            **其他格式:**
            - 📊 JSON (.json)
            - 🗃️ Parquet (.parquet)
            - 📝 TXT (.txt)
            """)
    
    # 文件上传器
    uploaded_files = st.file_uploader(
        "选择数据文件 (支持多文件上传)",
        type=['csv', 'xlsx', 'xls', 'sav', 'dta', 'json', 'parquet', 'txt'],
        accept_multiple_files=True,
        help="拖拽文件到此处或点击选择文件"
    )
    
    if uploaded_files:
        st.success(f"✅ 已选择 {len(uploaded_files)} 个文件")
        
        for i, uploaded_file in enumerate(uploaded_files):
            with st.expander(f"📄 {uploaded_file.name} ({uploaded_file.size:,} bytes)", expanded=True):
                process_uploaded_file(uploaded_file, i)

def process_uploaded_file(uploaded_file, index):
    """处理上传的文件"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # 根据文件类型显示不同的读取选项
        if file_extension == 'csv':
            df = process_csv_file(uploaded_file, index)
        elif file_extension in ['xlsx', 'xls']:
            df = process_excel_file(uploaded_file, index)
        elif file_extension == 'json':
            df = pd.read_json(uploaded_file)
        elif file_extension == 'parquet':
            df = pd.read_parquet(uploaded_file)
        elif file_extension == 'txt':
            df = process_text_file(uploaded_file, index)
        else:
            st.warning(f"⚠️ 暂不支持 {file_extension} 格式的高级选项")
            df = pd.read_csv(uploaded_file)
        
        if df is not None:
            # 数据质量检查
            quality_score = perform_data_quality_check(df)
            
            # 显示数据信息
            display_data_info(df, uploaded_file.name, quality_score)
            
            # 保存到session state
            dataset_key = f'dataset_{index}_{uploaded_file.name}'
            st.session_state[dataset_key] = {
                'data': df,
                'name': uploaded_file.name,
                'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'quality_score': quality_score,
                'file_size': uploaded_file.size
            }
            
            st.success(f"✅ 文件 {uploaded_file.name} 导入成功! 数据质量评分: {quality_score:.1f}/10")
            
    except Exception as e:
        st.error(f"❌ 文件读取失败: {str(e)}")
        st.info("💡 请检查文件格式、编码设置或数据完整性")

def process_csv_file(uploaded_file, index):
    """处理CSV文件"""
    st.markdown("**CSV文件读取选项:**")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        encoding = st.selectbox(
            "编码格式", 
            ['utf-8', 'gbk', 'gb2312', 'latin-1'], 
            key=f"encoding_{index}"
        )
    with col2:
        separator = st.selectbox(
            "分隔符", 
            [',', ';', '\t', '|', ' '], 
            key=f"sep_{index}"
        )
    with col3:
        header_row = st.number_input(
            "标题行", 
            min_value=0, 
            value=0, 
            key=f"header_{index}"
        )
    with col4:
        skip_rows = st.number_input(
            "跳过行数", 
            min_value=0, 
            value=0, 
            key=f"skip_{index}"
        )
    
    try:
        df = pd.read_csv(
            uploaded_file, 
            encoding=encoding, 
            sep=separator, 
            header=header_row if header_row >= 0 else None,
            skiprows=skip_rows
        )
        return df
    except Exception as e:
        st.error(f"CSV读取错误: {str(e)}")
        return None

def process_excel_file(uploaded_file, index):
    """处理Excel文件"""
    st.markdown("**Excel文件读取选项:**")
    
    # 先读取文件获取工作表名
    try:
        excel_file = pd.ExcelFile(uploaded_file)
        sheet_names = excel_file.sheet_names
        
        col1, col2, col3 = st.columns(3)
        with col1:
            sheet_name = st.selectbox(
                "选择工作表", 
                sheet_names, 
                key=f"sheet_{index}"
            )
        with col2:
            header_row = st.number_input(
                "标题行", 
                min_value=0, 
                value=0, 
                key=f"header_excel_{index}"
            )
        with col3:
            skip_rows = st.number_input(
                "跳过行数", 
                min_value=0, 
                value=0, 
                key=f"skip_excel_{index}"
            )
        
        df = pd.read_excel(
            uploaded_file, 
            sheet_name=sheet_name, 
            header=header_row if header_row >= 0 else None,
            skiprows=skip_rows
        )
        return df
        
    except Exception as e:
        st.error(f"Excel读取错误: {str(e)}")
        return None

def process_text_file(uploaded_file, index):
    """处理文本文件"""
    st.markdown("**文本文件读取选项:**")
    
    col1, col2 = st.columns(2)
    with col1:
        delimiter = st.text_input("分隔符", value="\t", key=f"txt_delim_{index}")
    with col2:
        encoding = st.selectbox("编码", ['utf-8', 'gbk', 'latin-1'], key=f"txt_enc_{index}")
    
    try:
        df = pd.read_csv(uploaded_file, sep=delimiter, encoding=encoding)
        return df
    except Exception as e:
        st.error(f"文本文件读取错误: {str(e)}")
        return None

def perform_data_quality_check(df):
    """执行数据质量检查"""
    score = 10.0
    
    # 缺失值检查 (最多扣2分)
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    if missing_ratio > 0.5:
        score -= 2
    elif missing_ratio > 0.2:
        score -= 1
    elif missing_ratio > 0.1:
        score -= 0.5
    
    # 重复行检查 (最多扣1分)
    duplicate_ratio = df.duplicated().sum() / len(df)
    if duplicate_ratio > 0.1:
        score -= 1
    elif duplicate_ratio > 0.05:
        score -= 0.5
    
    # 数据类型一致性检查 (最多扣1分)
    type_issues = 0
    for col in df.columns:
        if df[col].dtype == 'object':
            # 检查是否应该是数值型
            try:
                pd.to_numeric(df[col].dropna())
                type_issues += 1
            except:
                pass
    
    if type_issues > len(df.columns) * 0.3:
        score -= 1
    elif type_issues > len(df.columns) * 0.1:
        score -= 0.5
    
    return max(0, score)

def display_data_info(df, filename, quality_score):
    """显示数据基本信息"""
    st.markdown(f"#### 📋 数据概览 - {filename}")
    
    # 数据质量评分
    quality_color = "🟢" if quality_score >= 8 else "🟡" if quality_score >= 6 else "🔴"
    st.markdown(f"**数据质量评分: {quality_color} {quality_score:.1f}/10**")
    
    # 基本统计信息
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric("📊 总行数", f"{df.shape[0]:,}")
    with col2:
        st.metric("📋 总列数", f"{df.shape[1]:,}")
    with col3:
        missing_count = df.isnull().sum().sum()
        st.metric("❌ 缺失值", f"{missing_count:,}")
    with col4:
        duplicate_count = df.duplicated().sum()
        st.metric("🔄 重复行", f"{duplicate_count:,}")
    with col5:
        numeric_cols = df.select_dtypes(include=[np.number]).shape[1]
        st.metric("🔢 数值列", f"{numeric_cols:,}")
    with col6:
        text_cols = df.select_dtypes(include=['object']).shape[1]
        st.metric("🔤 文本列", f"{text_cols:,}")
    
    # 数据预览
    st.markdown("##### 📄 数据预览")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        preview_rows = st.slider("预览行数", 5, min(50, len(df)), 10, key=f"preview_{filename}")
    with col2:
        show_all_cols = st.checkbox("显示所有列", key=f"cols_{filename}")
    with col3:
        show_dtypes = st.checkbox("显示数据类型", key=f"dtypes_{filename}")
    
    # 显示数据预览
    preview_df = df.head(preview_rows)
    if not show_all_cols and df.shape[1] > 10:
        preview_df = preview_df.iloc[:, :10]
        st.info(f"显示前10列，共{df.shape[1]}列")
    
    st.dataframe(preview_df, use_container_width=True)
    
    # 数据类型和质量信息
    if show_dtypes or st.button(f"📊 查看详细信息", key=f"detail_{filename}"):
        with st.expander("🔍 详细数据信息", expanded=True):
            
            # 创建详细信息表
            detail_info = []
            for col in df.columns:
                col_data = df[col]
                detail_info.append({
                    '列名': col,
                    '数据类型': str(col_data.dtype),
                    '非空值数': col_data.count(),
                    '缺失值数': col_data.isnull().sum(),
                    '缺失率(%)': round(col_data.isnull().sum() / len(df) * 100, 2),
                    '唯一值数': col_data.nunique(),
                    '重复值数': len(df) - col_data.nunique(),
                    '内存使用': f"{col_data.memory_usage(deep=True) / 1024:.1f} KB"
                })
            
            detail_df = pd.DataFrame(detail_info)
            st.dataframe(detail_df, use_container_width=True)
            
            # 数据质量问题提醒
            issues = []
            if df.isnull().sum().sum() > 0:
                issues.append(f"⚠️ 发现 {df.isnull().sum().sum()} 个缺失值")
            if df.duplicated().sum() > 0:
                issues.append(f"⚠️ 发现 {df.duplicated().sum()} 个重复行")
            
            # 检查可能的数据类型问题
            type_suggestions = []
            for col in df.select_dtypes(include=['object']).columns:
                try:
                    # 尝试转换为数值
                    pd.to_numeric(df[col].dropna())
                    type_suggestions.append(f"💡 列 '{col}' 可能应该是数值类型")
                except:
                    pass
                
                # 检查日期格式
                if col.lower() in ['date', 'time', '日期', '时间'] or 'date' in col.lower():
                    type_suggestions.append(f"💡 列 '{col}' 可能是日期类型")
            
            if issues or type_suggestions:
                st.markdown("##### 🔍 数据质量建议")
                for issue in issues:
                    st.warning(issue)
                for suggestion in type_suggestions:
                    st.info(suggestion)

def database_connection_interface():
    """数据库连接界面"""
    st.markdown("#### 🔗 数据库连接")
    st.info("🚧 数据库连接功能正在开发中，敬请期待！")
    
    # 预留数据库连接选项
    db_type = st.selectbox(
        "数据库类型",
        ["MySQL", "PostgreSQL", "SQLite", "SQL Server", "Oracle"]
    )
    
    col1, col2 = st.columns(2)
    with col1:
        host = st.text_input("主机地址", placeholder="localhost")
        database = st.text_input("数据库名", placeholder="database_name")
    with col2:
        port = st.number_input("端口", value=3306)
        username = st.text_input("用户名", placeholder="username")
    
    password = st.text_input("密码", type="password")
    
    if st.button("🔌 测试连接"):
        st.warning("数据库连接功能开发中...")

def sample_data_interface():
    """示例数据界面"""
    st.markdown("#### 🌐 示例数据集")
    st.markdown("*选择内置示例数据集进行学习和测试*")
    
    # 示例数据集
    sample_datasets = {
        "🧪 临床试验数据": {
            "description": "随机对照试验示例数据，包含基线特征、疗效指标和安全性数据",
            "size": "500行 × 15列",
            "generator": generate_clinical_trial_data
        },
        "🦠 流行病学数据": {
            "description": "队列研究示例数据，包含暴露因素、协变量和结局变量",
            "size": "1000行 × 12列", 
            "generator": generate_epidemiology_data
        },
        "📊 生存分析数据": {
            "description": "生存分析示例数据，包含生存时间、删失状态和协变量",
            "size": "300行 × 8列",
            "generator": generate_survival_data
        },
        "🔬 实验室数据": {
            "description": "实验室检测结果数据，包含多个生化指标和参考范围",
            "size": "800行 × 20列",
            "generator": generate_lab_data
        }
    }
    
    for name, info in sample_datasets.items():
        with st.expander(f"{name} ({info['size']})", expanded=False):
            st.markdown(f"**描述:** {info['description']}")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**数据规模:** {info['size']}")
            with col2:
                if st.button("📥 加载数据", key=f"load_{name}"):
                    df = info['generator']()
                    dataset_key = f'dataset_sample_{name}'
                    st.session_state[dataset_key] = {
                        'data': df,
                        'name': f"示例_{name}",
                        'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'quality_score': 9.5,
                        'file_size': len(df) * len(df.columns) * 8  # 估算大小
                    }
                    st.success(f"✅ 已加载 {name}")
                    st.rerun()

def generate_clinical_trial_data():
    """生成临床试验示例数据"""
    np.random.seed(42)
    n = 500
    
    # 基线特征
    data = {
        '受试者ID': [f'S{i:04d}' for i in range(1, n+1)],
        '治疗组': np.random.choice(['试验组', '对照组'], n),
        '年龄': np.random.normal(65, 12, n).astype(int),
        '性别': np.random.choice(['男', '女'], n),
        '体重kg': np.random.normal(70, 15, n).round(1),
        '身高cm': np.random.normal(165, 10, n).round(1),
        '基线血压收缩压': np.random.normal(140, 20, n).astype(int),
        '基线血压舒张压': np.random.normal(90, 15, n).astype(int),
        '基线胆固醇': np.random.normal(5.2, 1.2, n).round(2),
        '糖尿病史': np.random.choice(['是', '否'], n, p=[0.3, 0.7]),
        '吸烟史': np.random.choice(['是', '否'], n, p=[0.4, 0.6]),
        '随访时间天': np.random.normal(180, 30, n).astype(int),
        '主要终点达成': np.random.choice(['是', '否'], n, p=[0.6, 0.4]),
        '不良事件': np.random.choice(['无', '轻度', '中度', '重度'], n, p=[0.6, 0.25, 0.1, 0.05]),
        '依从性百分比': np.random.normal(85, 15, n).round(1)
    }
    
    return pd.DataFrame(data)

def generate_epidemiology_data():
    """生成流行病学示例数据"""
    np.random.seed(42)
    n = 1000
    
    data = {
        'ID': range(1, n+1),
        '年龄': np.random.normal(45, 15, n).astype(int),
        '性别': np.random.choice(['男', '女'], n),
        '教育程度': np.random.choice(['小学', '中学', '大学', '研究生'], n, p=[0.2, 0.4, 0.3, 0.1]),
        '收入水平': np.random.choice(['低', '中', '高'], n, p=[0.3, 0.5, 0.2]),
        '吸烟状态': np.random.choice(['从不', '曾经', '现在'], n, p=[0.5, 0.3, 0.2]),
        '饮酒频率': np.random.choice(['从不', '偶尔', '经常'], n, p=[0.4, 0.4, 0.2]),
        '运动频率': np.random.choice(['从不', '偶尔', '经常'], n, p=[0.3, 0.4, 0.3]),
        'BMI': np.random.normal(24, 4, n).round(1),
        '血压mmHg': np.random.normal(120, 20, n).astype(int),
        '随访年数': np.random.uniform(1, 10, n).round(1),
        '疾病发生': np.random.choice([0, 1], n, p=[0.8, 0.2])
    }
    
    return pd.DataFrame(data)

def generate_survival_data():
    """生成生存分析示例数据"""
    np.random.seed(42)
    n = 300
    
    # 生成生存时间（指数分布）
    survival_time = np.random.exponential(20, n)
    # 生成删失时间
    censoring_time = np.random.exponential(30, n)
    # 观察时间取最小值
    observed_time = np.minimum(survival_time, censoring_time)
    # 事件发生标志
    event = (survival_time <= censoring_time).astype(int)
    
    data = {
        '患者ID': [f'P{i:03d}' for i in range(1, n+1)],
        '年龄': np.random.normal(60, 15, n).astype(int),
        '性别': np.random.choice(['男', '女'], n),
        '治疗方案': np.random.choice(['A', 'B', 'C'], n),
        '肿瘤分期': np.random.choice(['I', 'II', 'III', 'IV'], n, p=[0.2, 0.3, 0.3, 0.2]),
        '生存时间月': observed_time.round(1),
        '事件发生': event,
        '随访状态': ['事件' if e else '删失' for e in event]
    }
    
    return pd.DataFrame(data)

def generate_lab_data():
    """生成实验室数据"""
    np.random.seed(42)
    n = 800
    
    data = {
        '样本ID': [f'L{i:04d}' for i in range(1, n+1)],
        '检测日期': pd.date_range('2023-01-01', periods=n, freq='H'),
        '患者年龄': np.random.normal(50, 20, n).astype(int),
        '患者性别': np.random.choice(['男', '女'], n),
        '白细胞计数': np.random.normal(6.5, 2.0, n).round(2),
        '红细胞计数': np.random.normal(4.5, 0.5, n).round(2),
        '血红蛋白': np.random.normal(140, 20, n).round(1),
        '血小板计数': np.random.normal(250, 50, n).astype(int),
        '总胆固醇': np.random.normal(5.0, 1.0, n).round(2),
        '甘油三酯': np.random.normal(1.5, 0.8, n).round(2),
        '空腹血糖': np.random.normal(5.5, 1.5, n).round(2),
        '肌酐': np.random.normal(80, 20, n).round(1),
        '尿素氮': np.random.normal(5.0, 2.0, n).round(2),
        'ALT': np.random.normal(30, 15, n).round(1),
        'AST': np.random.normal(25, 12, n).round(1),
        '总胆红素': np.random.normal(15, 8, n).round(2),
        '白蛋白': np.random.normal(40, 5, n).round(1),
        'CRP': np.random.exponential(5, n).round(2),
        'ESR': np.random.normal(15, 10, n).astype(int),
        '检测结果': np.random.choice(['正常', '异常'], n, p=[0.7, 0.3])
    }
    
    return pd.DataFrame(data)

def manual_input_interface():
    """手动输入界面"""
    st.markdown("#### ✏️ 手动输入数据")
    
    input_method = st.radio(
        "选择输入方式",
        ["📝 表格编辑器", "📋 CSV格式输入"],
        horizontal=True
    )
    
    if input_method == "📝 表格编辑器":
        st.markdown("**使用表格编辑器创建数据:**")
        
        # 设置表格维度
        col1, col2 = st.columns(2)
        with col1:
            n_rows = st.number_input("行数", min_value=1, max_value=100, value=5)
        with col2:
            n_cols = st.number_input("列数", min_value=1, max_value=20, value=3)
        
        # 创建空数据框
        if 'manual_data' not in st.session_state:
            st.session_state.manual_data = pd.DataFrame(
                np.empty((n_rows, n_cols), dtype=object),
                columns=[f'列{i+1}' for i in range(n_cols)]
            )
        
        # 数据编辑器
        edited_df = st.data_editor(
            st.session_state.manual_data,
            use_container_width=True,
            num_rows="dynamic"
        )
        
        if st.button("💾 保存手动输入数据"):
            dataset_key = 'dataset_manual_input'
            st.session_state[dataset_key] = {
                'data': edited_df,
                'name': '手动输入数据',
                'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'quality_score': 8.0,
                'file_size': len(edited_df) * len(edited_df.columns) * 8
            }
            st.success("✅ 手动输入数据已保存!")
    
    else:
        st.markdown("**CSV格式输入 (粘贴CSV格式的数据):**")
        csv_input = st.text_area(
            "输入CSV格式数据",
            height=200,
            placeholder="列1,列2,列3\n值1,值2,值3\n值4,值5,值6"
        )
        
        if csv_input and st.button("📥 解析CSV数据"):
            try:
                df = pd.read_csv(io.StringIO(csv_input))
                dataset_key = 'dataset_csv_input'
                st.session_state[dataset_key] = {
                    'data': df,
                    'name': 'CSV输入数据',
                    'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'quality_score': 8.0,
                    'file_size': len(df) * len(df.columns) * 8
                }
                st.success("✅ CSV数据解析成功!")
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"❌ CSV解析失败: {str(e)}")

def data_exploration_section():
    """数据探索部分"""
    st.markdown("### 🔍 数据探索分析")
    st.markdown("*深入了解您的数据特征、分布和关系*")
    
    # 检查是否有数据
    datasets = get_available_datasets()
    if not datasets:
        st.warning("⚠️ 请先在 '数据导入' 标签页中导入数据")
        return
    
    # 选择数据集
    selected_dataset = st.selectbox(
        "📊 选择要探索的数据集", 
        list(datasets.keys()),
        help="选择已导入的数据集进行探索分析"
    )
    df = datasets[selected_dataset]['data']
    
    # 显示数据集基本信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 数据形状", f"{df.shape[0]} × {df.shape[1]}")
    with col2:
        st.metric("💾 内存使用", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    with col3:
        st.metric("📈 数值列", f"{df.select_dtypes(include=[np.number]).shape[1]}")
    with col4:
        st.metric("🔤 分类列", f"{df.select_dtypes(include=['object']).shape[1]}")
    
    # 探索选项
    exploration_type = st.radio(
        "🎯 选择探索类型",
        ["📊 描述性统计", "📈 数据分布", "🔗 相关性分析", "📋 交叉表分析", "🎨 数据可视化"],
        horizontal=True
    )
    
    if exploration_type == "📊 描述性统计":
        descriptive_statistics(df)
    elif exploration_type == "📈 数据分布":
        distribution_analysis(df)
    elif exploration_type == "🔗 相关性分析":
        correlation_analysis(df)
    elif exploration_type == "📋 交叉表分析":
        crosstab_analysis(df)
    elif exploration_type == "🎨 数据可视化":
        data_visualization(df)

def get_available_datasets():
    """获取可用的数据集"""
    datasets = {}
    for key, value in st.session_state.items():
        if key.startswith('dataset_') and isinstance(value, dict) and 'data' in value:
            datasets[value.get('name', key)] = value
    return datasets

def descriptive_statistics(df):
    """描述性统计分析"""
    st.markdown("#### 📊 描述性统计分析")
    
    # 数值变量统计
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) > 0:
        st.markdown("##### 🔢 数值变量统计")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_numeric = st.multiselect(
                "选择数值变量", 
                numeric_cols, 
                default=numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols,
                help="选择要分析的数值变量"
            )
        with col2:
            stat_options = st.multiselect(
                "选择统计量", 
                ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'skew', 'kurt'],
                default=['count', 'mean', 'std', 'min', '50%', 'max'],
                help="选择要计算的统计指标"
            )
        
        if selected_numeric and stat_options:
            # 计算描述性统计
            desc_stats = df[selected_numeric].describe()
            
            # 添加偏度和峰度
            if 'skew' in stat_options:
                desc_stats.loc['skew'] = df[selected_numeric].skew()
            if 'kurt' in stat_options:
                desc_stats.loc['kurt'] = df[selected_numeric].kurtosis()
            
            # 显示统计表
            st.dataframe(
                desc_stats.loc[stat_options].round(3), 
                use_container_width=True
            )
            
            # 统计解释
            with st.expander("📖 统计指标解释"):
                st.markdown("""
                - **count**: 非缺失值数量
                - **mean**: 平均值
                - **std**: 标准差
                - **min/max**: 最小值/最大值
                - **25%/50%/75%**: 四分位数
                - **skew**: 偏度 (>0右偏, <0左偏)
                - **kurt**: 峰度 (>0尖峰, <0平峰)
                """)
    
    # 分类变量统计
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if len(categorical_cols) > 0:
        st.markdown("##### 🔤 分类变量统计")
        
        selected_categorical = st.selectbox(
            "选择分类变量", 
            categorical_cols,
            help="选择要分析的分类变量"
        )
        
        if selected_categorical:
            col1, col2 = st.columns(2)
            
            with col1:
                # 频数统计
                value_counts = df[selected_categorical].value_counts()
                st.markdown("**频数统计:**")
                
                freq_df = pd.DataFrame({
                    '类别': value_counts.index,
                    '频数': value_counts.values,
                    '频率(%)': (value_counts.values / len(df) * 100).round(2)
                })
                st.dataframe(freq_df, use_container_width=True)
            
            with col2:
                # 饼图
                fig = px.pie(
                    values=value_counts.values,
                    names=value_counts.index,
                    title=f"{selected_categorical} 分布",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

def distribution_analysis(df):
    """数据分布分析"""
    st.markdown("#### 📈 数据分布分析")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("⚠️ 没有找到数值型变量")
        return
    
    # 选择变量
    col1, col2 = st.columns(2)
    with col1:
        selected_var = st.selectbox("选择变量", numeric_cols)
    with col2:
        plot_type = st.selectbox(
            "图表类型", 
            ["直方图", "密度图", "箱线图", "Q-Q图", "小提琴图"]
        )
    
    if selected_var:
        data_series = df[selected_var].dropna()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 绘制分布图
            if plot_type == "直方图":
                fig = px.histogram(
                    df, x=selected_var, 
                    title=f"{selected_var} 分布直方图",
                    marginal="box",
                    nbins=30
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot_type == "密度图":
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=data_series,
                    histnorm='probability density',
                    name='直方图',
                    opacity=0.7
                ))
                
                # 添加核密度估计
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(data_series)
                x_range = np.linspace(data_series.min(), data_series.max(), 100)
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=kde(x_range),
                    mode='lines',
                    name='密度曲线',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title=f"{selected_var} 密度分布图",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot_type == "箱线图":
                fig = px.box(
                    df, y=selected_var,
                    title=f"{selected_var} 箱线图",
                    points="outliers"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot_type == "Q-Q图":
                from scipy import stats
                fig = go.Figure()
                
                # 计算Q-Q图数据
                (osm, osr), (slope, intercept, r) = stats.probplot(data_series, dist="norm")
                
                fig.add_trace(go.Scatter(
                    x=osm, y=osr,
                    mode='markers',
                    name='观测值',
                    marker=dict(color='blue', size=6)
                ))
                
                fig.add_trace(go.Scatter(
                    x=osm, y=slope * osm + intercept,
                    mode='lines',
                    name='理论线',
                    line=dict(color='red', width=2)
                ))
                
                fig.update_layout(
                    title=f"{selected_var} Q-Q图 (正态性检验)",
                    xaxis_title="理论分位数",
                    yaxis_title="样本分位数",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
                
            elif plot_type == "小提琴图":
                fig = px.violin(
                    df, y=selected_var,
                    title=f"{selected_var} 小提琴图",
                    box=True,
                    points="outliers"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 分布统计信息
            st.markdown("**分布统计:**")
            
            stats_info = {
                "样本数": len(data_series),
                "均值": data_series.mean(),
                "中位数": data_series.median(),
                "标准差": data_series.std(),
                "偏度": data_series.skew(),
                "峰度": data_series.kurtosis(),
                "最小值": data_series.min(),
                "最大值": data_series.max(),
                "四分位距": data_series.quantile(0.75) - data_series.quantile(0.25)
            }
            
            for key, value in stats_info.items():
                if isinstance(value, (int, float)):
                    st.metric(key, f"{value:.3f}")
                else:
                    st.metric(key, value)
            
            # 正态性检验
            st.markdown("**正态性检验:**")
            from scipy.stats import shapiro, normaltest
            
            if len(data_series) <= 5000:  # Shapiro-Wilk适用于小样本
                stat, p_value = shapiro(data_series)
                test_name = "Shapiro-Wilk"
            else:
                stat, p_value = normaltest(data_series)
                test_name = "D'Agostino"
            
            st.write(f"**{test_name} 检验:**")
            st.write(f"统计量: {stat:.4f}")
            st.write(f"p值: {p_value:.4f}")
            
            if p_value > 0.05:
                st.success("✅ 接受正态分布假设")
            else:
                st.warning("⚠️ 拒绝正态分布假设")

def correlation_analysis(df):
    """相关性分析"""
    st.markdown("#### 🔗 相关性分析")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ 需要至少2个数值型变量进行相关性分析")
        return
    
    # 选择变量
    col1, col2 = st.columns(2)
    with col1:
        selected_vars = st.multiselect(
            "选择变量", 
            numeric_cols,
            default=numeric_cols[:min(8, len(numeric_cols))],
            help="选择要分析相关性的变量"
        )
    with col2:
        corr_method = st.selectbox(
            "相关系数类型",
            ["pearson", "spearman", "kendall"],
            help="Pearson: 线性相关; Spearman: 单调相关; Kendall: 秩相关"
        )
    
    if len(selected_vars) >= 2:
        # 计算相关矩阵
        corr_matrix = df[selected_vars].corr(method=corr_method)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # 相关性热力图
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title=f"相关性矩阵热力图 ({corr_method.title()})",
                color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # 显示相关矩阵数值
            st.markdown("**相关系数矩阵:**")
            st.dataframe(corr_matrix.round(3), use_container_width=True)
            
            # 找出强相关对
            st.markdown("**强相关变量对:**")
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr.append({
                            '变量1': corr_matrix.columns[i],
                            '变量2': corr_matrix.columns[j],
                            '相关系数': corr_val
                        })
            
            if strong_corr:
                strong_corr_df = pd.DataFrame(strong_corr)
                st.dataframe(strong_corr_df.round(3), use_container_width=True)
            else:
                st.info("没有发现强相关变量对 (|r| > 0.7)")
        
        # 散点图矩阵
        if st.checkbox("显示散点图矩阵") and len(selected_vars) <= 6:
            st.markdown("##### 📊 散点图矩阵")
            fig = px.scatter_matrix(
                df[selected_vars],
                title="变量间散点图矩阵"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

def crosstab_analysis(df):
    """交叉表分析"""
    st.markdown("#### 📋 交叉表分析")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(categorical_cols) < 2:
        st.warning("⚠️ 需要至少2个分类变量进行交叉表分析")
        return
    
    # 选择变量
    col1, col2 = st.columns(2)
    with col1:
        var1 = st.selectbox("选择行变量", categorical_cols)
    with col2:
        var2 = st.selectbox("选择列变量", [col for col in categorical_cols if col != var1])
    
    if var1 and var2:
        # 创建交叉表
        crosstab = pd.crosstab(df[var1], df[var2], margins=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**频数交叉表:**")
            st.dataframe(crosstab, use_container_width=True)
            
            # 百分比交叉表
            crosstab_pct = pd.crosstab(df[var1], df[var2], normalize='all') * 100
            st.markdown("**百分比交叉表:**")
            st.dataframe(crosstab_pct.round(2), use_container_width=True)
        
        with col2:
            # 堆积柱状图
            crosstab_no_margin = pd.crosstab(df[var1], df[var2])
            fig = px.bar(
                crosstab_no_margin,
                title=f"{var1} vs {var2} 分布",
                barmode='stack'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # 热力图
            fig2 = px.imshow(
                crosstab_no_margin,
                text_auto=True,
                aspect="auto",
                title="交叉表热力图"
            )
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        
        # 卡方检验
        st.markdown("##### 🧮 卡方独立性检验")
        from scipy.stats import chi2_contingency
        
        chi2, p_value, dof, expected = chi2_contingency(crosstab_no_margin)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("卡方统计量", f"{chi2:.4f}")
        with col2:
            st.metric("p值", f"{p_value:.4f}")
        with col3:
            st.metric("自由度", dof)
        with col4:
            if p_value < 0.05:
                st.success("显著相关")
            else:
                st.info("无显著相关")

def data_visualization(df):
    """数据可视化"""
    st.markdown("#### 🎨 数据可视化")
    
    # 图表类型选择
    chart_type = st.selectbox(
        "选择图表类型",
        [
            "📊 柱状图", "📈 折线图", "🔵 散点图", "📦 箱线图", 
            "🥧 饼图", "🎻 小提琴图", "🔥 热力图", "📊 直方图"
        ]
    )
    
    if chart_type == "📊 柱状图":
        create_bar_chart(df)
    elif chart_type == "📈 折线图":
        create_line_chart(df)
    elif chart_type == "🔵 散点图":
        create_scatter_plot(df)
    elif chart_type == "📦 箱线图":
        create_box_plot(df)
    elif chart_type == "🥧 饼图":
        create_pie_chart(df)
    elif chart_type == "🎻 小提琴图":
        create_violin_plot(df)
    elif chart_type == "🔥 热力图":
        create_heatmap(df)
    elif chart_type == "📊 直方图":
        create_histogram(df)

def create_bar_chart(df):
    """创建柱状图"""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_var = st.selectbox("X轴变量", categorical_cols)
    with col2:
        y_var = st.selectbox("Y轴变量", numeric_cols)
    with col3:
        color_var = st.selectbox("颜色分组", [None] + categorical_cols)
    
    if x_var and y_var:
        fig = px.bar(
            df, x=x_var, y=y_var, color=color_var,
            title=f"{y_var} by {x_var}",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

def create_line_chart(df):
    """创建折线图"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        x_var = st.selectbox("X轴变量", all_cols)
    with col2:
        y_var = st.selectbox("Y轴变量", numeric_cols)
    with col3:
        color_var = st.selectbox("颜色分组", [None] + df.select_dtypes(include=['object']).columns.tolist())
    
    if x_var and y_var:
        fig = px.line(
            df, x=x_var, y=y_var, color=color_var,
            title=f"{y_var} vs {x_var}",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

def create_scatter_plot(df):
    """创建散点图"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        x_var = st.selectbox("X轴变量", numeric_cols)
        color_var = st.selectbox("颜色变量", [None] + categorical_cols + numeric_cols)
    with col2:
        y_var = st.selectbox("Y轴变量", numeric_cols)
        size_var = st.selectbox("大小变量", [None] + numeric_cols)
    
    if x_var and y_var:
        fig = px.scatter(
            df, x=x_var, y=y_var, color=color_var, size=size_var,
            title=f"{y_var} vs {x_var}",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

def create_box_plot(df):
    """创建箱线图"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        y_var = st.selectbox("数值变量", numeric_cols)
    with col2:
        x_var = st.selectbox("分组变量", [None] + categorical_cols)
    
    if y_var:
        fig = px.box(
            df, x=x_var, y=y_var,
            title=f"{y_var} 箱线图",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

def create_pie_chart(df):
    """创建饼图"""
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        st.warning("⚠️ 没有分类变量可用于饼图")
        return
    
    selected_var = st.selectbox("选择分类变量", categorical_cols)
    
    if selected_var:
        value_counts = df[selected_var].value_counts()
        fig = px.pie(
            values=value_counts.values,
            names=value_counts.index,
            title=f"{selected_var} 分布饼图",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

def create_violin_plot(df):
    """创建小提琴图"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2 = st.columns(2)
    with col1:
        y_var = st.selectbox("数值变量", numeric_cols)
    with col2:
        x_var = st.selectbox("分组变量", [None] + categorical_cols)
    
    if y_var:
        fig = px.violin(
            df, x=x_var, y=y_var,
            title=f"{y_var} 小提琴图",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

def create_heatmap(df):
    """创建热力图"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.warning("⚠️ 需要至少2个数值变量创建热力图")
        return
    
    selected_vars = st.multiselect(
        "选择变量", 
        numeric_cols,
        default=numeric_cols[:min(10, len(numeric_cols))]
    )
    
    if len(selected_vars) >= 2:
        corr_matrix = df[selected_vars].corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="相关性热力图",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

def create_histogram(df):
    """创建直方图"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("⚠️ 没有数值变量可用于直方图")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        selected_var = st.selectbox("选择变量", numeric_cols)
    with col2:
        bins = st.slider("直方图箱数", 10, 100, 30)
    
    if selected_var:
        fig = px.histogram(
            df, x=selected_var,
            nbins=bins,
            title=f"{selected_var} 分布直方图",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

def data_cleaning_section():
    """数据清洗部分"""
    st.markdown("### 🛠️ 数据清洗")
    st.markdown("*清理和预处理您的数据，确保数据质量*")
    
    # 检查是否有数据
    datasets = get_available_datasets()
    if not datasets:
        st.warning("⚠️ 请先导入数据")
        return
    
    # 选择数据集
    selected_dataset = st.selectbox("选择数据集", list(datasets.keys()))
    df = datasets[selected_dataset]['data'].copy()
    
    # 清洗选项
    cleaning_type = st.radio(
        "选择清洗类型",
        ["🧹 缺失值处理", "🔄 重复值处理", "🎯 异常值检测", "🔧 数据类型转换", "📝 数据标准化"],
        horizontal=True
    )
    
    if cleaning_type == "🧹 缺失值处理":
        missing_value_handling(df, selected_dataset)
    elif cleaning_type == "🔄 重复值处理":
        duplicate_handling(df, selected_dataset)
    elif cleaning_type == "🎯 异常值检测":
        outlier_detection(df, selected_dataset)
    elif cleaning_type == "🔧 数据类型转换":
        data_type_conversion(df, selected_dataset)
    elif cleaning_type == "📝 数据标准化":
        data_standardization(df, selected_dataset)

def missing_value_handling(df, dataset_name):
    """缺失值处理"""
    st.markdown("#### 🧹 缺失值处理")
    
    # 缺失值概览
    missing_summary = df.isnull().sum()
    missing_summary = missing_summary[missing_summary > 0].sort_values(ascending=False)
    
    if len(missing_summary) == 0:
        st.success("✅ 数据中没有缺失值!")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**缺失值统计:**")
        missing_df = pd.DataFrame({
            '列名': missing_summary.index,
            '缺失数': missing_summary.values,
            '缺失率(%)': (missing_summary.values / len(df) * 100).round(2)
        })
        st.dataframe(missing_df, use_container_width=True)
    
    with col2:
        # 缺失值可视化
        fig = px.bar(
            missing_df, x='列名', y='缺失率(%)',
            title="缺失值分布",
            color='缺失率(%)',
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 缺失值处理选项
    st.markdown("**处理方法:**")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_columns = st.multiselect(
            "选择要处理的列",
            missing_summary.index.tolist(),
            default=missing_summary.index.tolist()
        )
    
    with col2:
        handling_method = st.selectbox(
            "处理方法",
            ["删除含缺失值的行", "删除含缺失值的列", "均值填充", "中位数填充", "众数填充", "前向填充", "后向填充", "插值填充", "自定义值填充"]
        )
    
    if selected_columns and st.button("🔧 执行处理"):
        df_cleaned = df.copy()
        
        if handling_method == "删除含缺失值的行":
            df_cleaned = df_cleaned.dropna(subset=selected_columns)
            st.success(f"✅ 已删除 {len(df) - len(df_cleaned)} 行含缺失值的数据")
            
        elif handling_method == "删除含缺失值的列":
            df_cleaned = df_cleaned.drop(columns=selected_columns)
            st.success(f"✅ 已删除 {len(selected_columns)} 列")
            
        elif handling_method == "均值填充":
            for col in selected_columns:
                if df[col].dtype in ['int64', 'float64']:
                    df_cleaned[col].fillna(df[col].mean(), inplace=True)
            st.success("✅ 已用均值填充数值列的缺失值")
            
        elif handling_method == "中位数填充":
            for col in selected_columns:
                if df[col].dtype in ['int64', 'float64']:
                    df_cleaned[col].fillna(df[col].median(), inplace=True)
            st.success("✅ 已用中位数填充数值列的缺失值")
            
        elif handling_method == "众数填充":
            for col in selected_columns:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df_cleaned[col].fillna(mode_val[0], inplace=True)
            st.success("✅ 已用众数填充缺失值")
            
        elif handling_method == "前向填充":
            df_cleaned[selected_columns] = df_cleaned[selected_columns].fillna(method='ffill')
            st.success("✅ 已执行前向填充")
            
        elif handling_method == "后向填充":
            df_cleaned[selected_columns] = df_cleaned[selected_columns].fillna(method='bfill')
            st.success("✅ 已执行后向填充")
            
        elif handling_method == "插值填充":
            for col in selected_columns:
                if df[col].dtype in ['int64', 'float64']:
                    df_cleaned[col] = df_cleaned[col].interpolate()
            st.success("✅ 已执行插值填充")
            
        elif handling_method == "自定义值填充":
            fill_value = st.text_input("输入填充值")
            if fill_value:
                df_cleaned[selected_columns] = df_cleaned[selected_columns].fillna(fill_value)
                st.success(f"✅ 已用 '{fill_value}' 填充缺失值")
        
        # 保存清洗后的数据
        if st.button("💾 保存清洗后的数据"):
            new_dataset_key = f'dataset_cleaned_{dataset_name}'
            st.session_state[new_dataset_key] = {
                'data': df_cleaned,
                'name': f'已清洗_{dataset_name}',
                'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'quality_score': perform_data_quality_check(df_cleaned),
                'file_size': len(df_cleaned) * len(df_cleaned.columns) * 8
            }
            st.success("✅ 清洗后的数据已保存!")

def duplicate_handling(df, dataset_name):
    """重复值处理"""
    st.markdown("#### 🔄 重复值处理")
    
    # 检查重复值
    duplicate_count = df.duplicated().sum()
    total_rows = len(df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总行数", total_rows)
    with col2:
        st.metric("重复行数", duplicate_count)
    with col3:
        st.metric("重复率(%)", f"{duplicate_count/total_rows*100:.2f}")
    
    if duplicate_count == 0:
        st.success("✅ 数据中没有重复行!")
        return
    
    # 显示重复行
    if st.checkbox("显示重复行"):
        duplicate_rows = df[df.duplicated(keep=False)]
        st.dataframe(duplicate_rows, use_container_width=True)
    
    # 重复值处理选项
    col1, col2 = st.columns(2)
    with col1:
        subset_cols = st.multiselect(
            "基于哪些列判断重复 (留空表示所有列)",
            df.columns.tolist()
        )
    
    with col2:
        keep_option = st.selectbox(
            "保留策略",
            ["first", "last", "False"],
            format_func=lambda x: {"first": "保留第一个", "last": "保留最后一个", "False": "全部删除"}[x]
        )
    
    if st.button("🗑️ 删除重复行"):
        df_dedup = df.copy()
        
        if subset_cols:
            df_dedup = df_dedup.drop_duplicates(subset=subset_cols, keep=keep_option if keep_option != "False" else False)
        else:
            df_dedup = df_dedup.drop_duplicates(keep=keep_option if keep_option != "False" else False)
        
        removed_count = len(df) - len(df_dedup)
        st.success(f"✅ 已删除 {removed_count} 行重复数据")
        
        # 保存去重后的数据
        if st.button("💾 保存去重后的数据"):
            new_dataset_key = f'dataset_dedup_{dataset_name}'
            st.session_state[new_dataset_key] = {
                'data': df_dedup,
                'name': f'已去重_{dataset_name}',
                'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'quality_score': perform_data_quality_check(df_dedup),
                'file_size': len(df_dedup) * len(df_dedup.columns) * 8
            }
            st.success("✅ 去重后的数据已保存!")

def outlier_detection(df, dataset_name):
    """异常值检测"""
    st.markdown("#### 🎯 异常值检测")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("⚠️ 没有数值型变量可检测异常值")
        return
    
    # 选择检测方法
    detection_method = st.selectbox(
        "异常值检测方法",
        ["IQR方法", "Z-Score方法", "改进Z-Score方法", "孤立森林"]
    )
    
    selected_cols = st.multiselect(
        "选择要检测的列",
        numeric_cols,
        default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols
    )
    
    if not selected_cols:
        return
    
    outliers_info = {}
    
    for col in selected_cols:
        data = df[col].dropna()
        
        if detection_method == "IQR方法":
            Q1 = data.quantile(0.25)
            Q3 = data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            
        elif detection_method == "Z-Score方法":
            z_scores = np.abs(stats.zscore(data))
            threshold = st.slider(f"Z-Score阈值 ({col})", 2.0, 4.0, 3.0, 0.1)
            outlier_indices = np.where(z_scores > threshold)[0]
            outliers = df.iloc[outlier_indices]
            
        elif detection_method == "改进Z-Score方法":
            median = np.median(data)
            mad = np.median(np.abs(data - median))
            modified_z_scores = 0.6745 * (data - median) / mad
            threshold = st.slider(f"改进Z-Score阈值 ({col})", 2.0, 4.0, 3.5, 0.1)
            outlier_indices = np.where(np.abs(modified_z_scores) > threshold)[0]
            outliers = df.iloc[outlier_indices]
            
        elif detection_method == "孤立森林":
            from sklearn.ensemble import IsolationForest
            isolation_forest = IsolationForest(contamination=0.1, random_state=42)
            outlier_labels = isolation_forest.fit_predict(data.values.reshape(-1, 1))
            outlier_indices = np.where(outlier_labels == -1)[0]
            outliers = df.iloc[outlier_indices]
        
        outliers_info[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(df) * 100,
            'data': outliers
        }
    
    # 显示异常值统计
    st.markdown("**异常值统计:**")
    
    outlier_summary = pd.DataFrame({
        '列名': list(outliers_info.keys()),
        '异常值数量': [info['count'] for info in outliers_info.values()],
        '异常值比例(%)': [round(info['percentage'], 2) for info in outliers_info.values()]
    })
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(outlier_summary, use_container_width=True)
    
    with col2:
        fig = px.bar(
            outlier_summary, x='列名', y='异常值比例(%)',
            title="异常值分布",
            color='异常值比例(%)',
            color_continuous_scale='Oranges'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # 显示异常值详情
    for col, info in outliers_info.items():
        if info['count'] > 0:
            with st.expander(f"📊 {col} 的异常值详情 ({info['count']} 个)"):
                st.dataframe(info['data'], use_container_width=True)
                
                # 异常值可视化
                fig = px.box(df, y=col, title=f"{col} 箱线图 (异常值标记)")
                st.plotly_chart(fig, use_container_width=True)
    
    # 异常值处理
    st.markdown("**异常值处理:**")
    handling_method = st.selectbox(
        "处理方法",
        ["删除异常值", "用中位数替换", "用均值替换", "保留异常值"]
    )
    
    if st.button("🔧 处理异常值"):
        df_processed = df.copy()
        
        for col, info in outliers_info.items():
            if info['count'] > 0:
                outlier_indices = info['data'].index
                
                if handling_method == "删除异常值":
                    df_processed = df_processed.drop(outlier_indices)
                elif handling_method == "用中位数替换":
                    median_val = df[col].median()
                    df_processed.loc[outlier_indices, col] = median_val
                elif handling_method == "用均值替换":
                    mean_val = df[col].mean()
                    df_processed.loc[outlier_indices, col] = mean_val
        
        if handling_method != "保留异常值":
            st.success(f"✅ 异常值处理完成")
            
            # 保存处理后的数据
            if st.button("💾 保存处理后的数据"):
                new_dataset_key = f'dataset_outlier_processed_{dataset_name}'
                st.session_state[new_dataset_key] = {
                    'data': df_processed,
                    'name': f'异常值已处理_{dataset_name}',
                    'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'quality_score': perform_data_quality_check(df_processed),
                    'file_size': len(df_processed) * len(df_processed.columns) * 8
                }
                st.success("✅ 处理后的数据已保存!")

def data_type_conversion(df, dataset_name):
    """数据类型转换"""
    st.markdown("#### 🔧 数据类型转换")
    
    # 显示当前数据类型
    st.markdown("**当前数据类型:**")
    
    dtype_info = pd.DataFrame({
        '列名': df.columns,
        '当前类型': [str(dtype) for dtype in df.dtypes],
        '非空值数': [df[col].count() for col in df.columns],
        '示例值': [str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else 'N/A' for col in df.columns]
    })
    
    st.dataframe(dtype_info, use_container_width=True)
    
    # 类型转换选项
    st.markdown("**类型转换:**")
    
    col1, col2 = st.columns(2)
    with col1:
        selected_col = st.selectbox("选择要转换的列", df.columns.tolist())
    with col2:
        target_type = st.selectbox(
            "目标类型",
            ["int64", "float64", "object", "datetime64", "category", "bool"]
        )
    
    if selected_col and st.button("🔄 执行转换"):
        df_converted = df.copy()
        
        try:
            if target_type == "int64":
                df_converted[selected_col] = pd.to_numeric(df_converted[selected_col], errors='coerce').astype('Int64')
            elif target_type == "float64":
                df_converted[selected_col] = pd.to_numeric(df_converted[selected_col], errors='coerce')
            elif target_type == "object":
                df_converted[selected_col] = df_converted[selected_col].astype(str)
            elif target_type == "datetime64":
                df_converted[selected_col] = pd.to_datetime(df_converted[selected_col], errors='coerce')
            elif target_type == "category":
                df_converted[selected_col] = df_converted[selected_col].astype('category')
            elif target_type == "bool":
                df_converted[selected_col] = df_converted[selected_col].astype(bool)
            
            st.success(f"✅ 列 '{selected_col}' 已转换为 {target_type} 类型")
            
            # 显示转换后的信息
            st.markdown("**转换后信息:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"转换前类型: {df[selected_col].dtype}")
                st.write(f"转换后类型: {df_converted[selected_col].dtype}")
            with col2:
                st.write(f"转换前非空值: {df[selected_col].count()}")
                st.write(f"转换后非空值: {df_converted[selected_col].count()}")
            
            # 保存转换后的数据
            if st.button("💾 保存转换后的数据"):
                new_dataset_key = f'dataset_converted_{dataset_name}'
                st.session_state[new_dataset_key] = {
                    'data': df_converted,
                    'name': f'类型已转换_{dataset_name}',
                    'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'quality_score': perform_data_quality_check(df_converted),
                    'file_size': len(df_converted) * len(df_converted.columns) * 8
                }
                st.success("✅ 转换后的数据已保存!")
                
        except Exception as e:
            st.error(f"❌ 转换失败: {str(e)}")

def data_standardization(df, dataset_name):
    """数据标准化"""
    st.markdown("#### 📝 数据标准化")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not numeric_cols:
        st.warning("⚠️ 没有数值型变量可进行标准化")
        return
    
    # 标准化方法选择
    standardization_method = st.selectbox(
        "标准化方法",
        ["Z-Score标准化", "Min-Max标准化", "Robust标准化", "单位向量标准化"]
    )
    
    selected_cols = st.multiselect(
        "选择要标准化的列",
        numeric_cols,
        default=numeric_cols
    )
    
    if selected_cols and st.button("📊 执行标准化"):
        df_standardized = df.copy()
        
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, Normalizer
        
        if standardization_method == "Z-Score标准化":
            scaler = StandardScaler()
        elif standardization_method == "Min-Max标准化":
            scaler = MinMaxScaler()
        elif standardization_method == "Robust标准化":
            scaler = RobustScaler()
        elif standardization_method == "单位向量标准化":
            scaler = Normalizer()
        
        # 执行标准化
        df_standardized[selected_cols] = scaler.fit_transform(df[selected_cols])
        
        st.success(f"✅ 已完成 {standardization_method}")
        
        # 显示标准化前后对比
        st.markdown("**标准化前后对比:**")
        
        comparison_data = []
        for col in selected_cols:
            comparison_data.append({
                '列名': col,
                '原始均值': df[col].mean(),
                '原始标准差': df[col].std(),
                '标准化后均值': df_standardized[col].mean(),
                '标准化后标准差': df_standardized[col].std()
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df.round(4), use_container_width=True)
        
        # 可视化对比
        if len(selected_cols) <= 4:
            fig = make_subplots(
                rows=2, cols=len(selected_cols),
                subplot_titles=[f'{col} (原始)' for col in selected_cols] + [f'{col} (标准化)' for col in selected_cols]
            )
            
            for i, col in enumerate(selected_cols):
                # 原始数据分布
                fig.add_trace(
                    go.Histogram(x=df[col], name=f'{col}_原始', showlegend=False),
                    row=1, col=i+1
                )
                # 标准化后分布
                fig.add_trace(
                    go.Histogram(x=df_standardized[col], name=f'{col}_标准化', showlegend=False),
                    row=2, col=i+1
                )
            
            fig.update_layout(height=500, title="标准化前后分布对比")
            st.plotly_chart(fig, use_container_width=True)
        
        # 保存标准化后的数据
        if st.button("💾 保存标准化后的数据"):
            new_dataset_key = f'dataset_standardized_{dataset_name}'
            st.session_state[new_dataset_key] = {
                'data': df_standardized,
                'name': f'已标准化_{dataset_name}',
                'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'quality_score': perform_data_quality_check(df_standardized),
                'file_size': len(df_standardized) * len(df_standardized.columns) * 8
            }
            st.success("✅ 标准化后的数据已保存!")

def variable_management_section():
    """变量管理部分"""
    st.markdown("### 📋 变量管理")
    st.markdown("*管理和编辑数据集中的变量信息*")
    
    # 检查是否有数据
    datasets = get_available_datasets()
    if not datasets:
        st.warning("⚠️ 请先导入数据")
        return
    
    # 选择数据集
    selected_dataset = st.selectbox("选择数据集", list(datasets.keys()))
    df = datasets[selected_dataset]['data']
    
    # 变量管理选项
    management_type = st.radio(
        "管理类型",
        ["📝 变量重命名", "🗂️ 变量分类", "📊 变量编码", "🔄 变量创建", "🗑️ 变量删除"],
        horizontal=True
    )
    
    if management_type == "📝 变量重命名":
        variable_renaming(df, selected_dataset)
    elif management_type == "🗂️ 变量分类":
        variable_categorization(df, selected_dataset)
    elif management_type == "📊 变量编码":
        variable_encoding(df, selected_dataset)
    elif management_type == "🔄 变量创建":
        variable_creation(df, selected_dataset)
    elif management_type == "🗑️ 变量删除":
        variable_deletion(df, selected_dataset)

def variable_renaming(df, dataset_name):
    """变量重命名"""
    st.markdown("#### 📝 变量重命名")
    
    # 显示当前变量名
    st.markdown("**当前变量列表:**")
    current_names = pd.DataFrame({
        '序号': range(1, len(df.columns) + 1),
        '当前名称': df.columns.tolist(),
        '数据类型': [str(dtype) for dtype in df.dtypes],
        '非空值数': [df[col].count() for col in df.columns]
    })
    st.dataframe(current_names, use_container_width=True)
    
    # 重命名选项
    rename_method = st.radio(
        "重命名方式",
        ["单个重命名", "批量重命名", "使用映射文件"],
        horizontal=True
    )
    
    if rename_method == "单个重命名":
        col1, col2 = st.columns(2)
        with col1:
            old_name = st.selectbox("选择要重命名的变量", df.columns.tolist())
        with col2:
            new_name = st.text_input("新变量名", value=old_name)
        
        if old_name != new_name and new_name and st.button("🔄 重命名"):
            df_renamed = df.copy()
            df_renamed = df_renamed.rename(columns={old_name: new_name})
            
            st.success(f"✅ 变量 '{old_name}' 已重命名为 '{new_name}'")
            
            # 保存重命名后的数据
            if st.button("💾 保存重命名后的数据"):
                new_dataset_key = f'dataset_renamed_{dataset_name}'
                st.session_state[new_dataset_key] = {
                    'data': df_renamed,
                    'name': f'已重命名_{dataset_name}',
                    'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'quality_score': perform_data_quality_check(df_renamed),
                    'file_size': len(df_renamed) * len(df_renamed.columns) * 8
                }
                st.success("✅ 重命名后的数据已保存!")
    
    elif rename_method == "批量重命名":
        st.markdown("**批量重命名规则:**")
        
        rule_type = st.selectbox(
            "规则类型",
            ["添加前缀", "添加后缀", "替换文本", "转换大小写"]
        )
        
        if rule_type == "添加前缀":
            prefix = st.text_input("前缀")
            if prefix and st.button("🔄 应用前缀"):
                new_columns = [f"{prefix}{col}" for col in df.columns]
                df_renamed = df.copy()
                df_renamed.columns = new_columns
                st.success(f"✅ 已为所有变量添加前缀 '{prefix}'")
                
        elif rule_type == "添加后缀":
            suffix = st.text_input("后缀")
            if suffix and st.button("🔄 应用后缀"):
                new_columns = [f"{col}{suffix}" for col in df.columns]
                df_renamed = df.copy()
                df_renamed.columns = new_columns
                st.success(f"✅ 已为所有变量添加后缀 '{suffix}'")
                
        elif rule_type == "替换文本":
            col1, col2 = st.columns(2)
            with col1:
                old_text = st.text_input("要替换的文本")
            with col2:
                new_text = st.text_input("替换为")
            
            if old_text and st.button("🔄 执行替换"):
                new_columns = [col.replace(old_text, new_text) for col in df.columns]
                df_renamed = df.copy()
                df_renamed.columns = new_columns
                st.success(f"✅ 已将 '{old_text}' 替换为 '{new_text}'")
                
        elif rule_type == "转换大小写":
            case_type = st.selectbox("转换类型", ["全部大写", "全部小写", "首字母大写"])
            
            if st.button("🔄 转换大小写"):
                if case_type == "全部大写":
                    new_columns = [col.upper() for col in df.columns]
                elif case_type == "全部小写":
                    new_columns = [col.lower() for col in df.columns]
                elif case_type == "首字母大写":
                    new_columns = [col.title() for col in df.columns]
                
                df_renamed = df.copy()
                df_renamed.columns = new_columns
                st.success(f"✅ 已转换为{case_type}")

def variable_categorization(df, dataset_name):
    """变量分类"""
    st.markdown("#### 🗂️ 变量分类")
    
    # 自动识别变量类型
    numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_vars = df.select_dtypes(include=['object']).columns.tolist()
    datetime_vars = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # 显示自动分类结果
    st.markdown("**自动变量分类:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔢 数值变量:**")
        for var in numeric_vars:
            st.write(f"• {var}")
    
    with col2:
        st.markdown("**🔤 分类变量:**")
        for var in categorical_vars:
            st.write(f"• {var}")
    
    with col3:
        st.markdown("**📅 日期变量:**")
        for var in datetime_vars:
            st.write(f"• {var}")
    
    # 手动分类调整
    st.markdown("**手动分类调整:**")
    
    selected_var = st.selectbox("选择变量", df.columns.tolist())
    
        if selected_var:
        current_type = "数值型" if selected_var in numeric_vars else "分类型" if selected_var in categorical_vars else "日期型"
        st.info(f"当前类型: {current_type}")
        
        new_type = st.selectbox(
            "重新分类为",
            ["数值型", "分类型", "日期型", "二元型", "有序分类型"]
        )
        
        if st.button("🔄 应用分类"):
            df_categorized = df.copy()
            
            try:
                if new_type == "数值型":
                    df_categorized[selected_var] = pd.to_numeric(df_categorized[selected_var], errors='coerce')
                elif new_type == "分类型":
                    df_categorized[selected_var] = df_categorized[selected_var].astype('category')
                elif new_type == "日期型":
                    df_categorized[selected_var] = pd.to_datetime(df_categorized[selected_var], errors='coerce')
                elif new_type == "二元型":
                    unique_vals = df[selected_var].unique()
                    if len(unique_vals) <= 2:
                        df_categorized[selected_var] = df_categorized[selected_var].astype('category')
                    else:
                        st.warning("⚠️ 该变量有超过2个唯一值，不适合作为二元变量")
                elif new_type == "有序分类型":
                    order = st.text_input("输入顺序 (用逗号分隔)", placeholder="低,中,高")
                    if order:
                        order_list = [x.strip() for x in order.split(',')]
                        df_categorized[selected_var] = pd.Categorical(df_categorized[selected_var], categories=order_list, ordered=True)
                
                st.success(f"✅ 变量 '{selected_var}' 已重新分类为 {new_type}")
                
            except Exception as e:
                st.error(f"❌ 分类失败: {str(e)}")

def variable_encoding(df, dataset_name):
    """变量编码"""
    st.markdown("#### 📊 变量编码")
    
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if not categorical_cols:
        st.warning("⚠️ 没有分类变量需要编码")
        return
    
    # 编码方法选择
    encoding_method = st.selectbox(
        "编码方法",
        ["标签编码", "独热编码", "目标编码", "二进制编码", "哈希编码"]
    )
    
    selected_col = st.selectbox("选择要编码的变量", categorical_cols)
    
    if selected_col:
        # 显示变量信息
        unique_values = df[selected_col].unique()
        st.info(f"唯一值数量: {len(unique_values)}")
        st.write("唯一值:", unique_values[:10].tolist() + (['...'] if len(unique_values) > 10 else []))
        
        if st.button("🔢 执行编码"):
            df_encoded = df.copy()
            
            if encoding_method == "标签编码":
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df_encoded[f'{selected_col}_encoded'] = le.fit_transform(df_encoded[selected_col].astype(str))
                st.success("✅ 标签编码完成")
                
                # 显示编码映射
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                st.write("编码映射:", mapping)
                
            elif encoding_method == "独热编码":
                encoded_df = pd.get_dummies(df_encoded[selected_col], prefix=selected_col)
                df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
                st.success("✅ 独热编码完成")
                st.info(f"生成了 {len(encoded_df.columns)} 个新变量")
                
            elif encoding_method == "目标编码":
                target_col = st.selectbox("选择目标变量", df.select_dtypes(include=[np.number]).columns.tolist())
                if target_col:
                    target_mean = df.groupby(selected_col)[target_col].mean()
                    df_encoded[f'{selected_col}_target_encoded'] = df_encoded[selected_col].map(target_mean)
                    st.success("✅ 目标编码完成")
                
            elif encoding_method == "二进制编码":
                try:
                    import category_encoders as ce
                    encoder = ce.BinaryEncoder(cols=[selected_col])
                    encoded_df = encoder.fit_transform(df_encoded[selected_col])
                    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
                    st.success("✅ 二进制编码完成")
                except ImportError:
                    st.error("❌ 需要安装 category_encoders 库")
                
            elif encoding_method == "哈希编码":
                try:
                    import category_encoders as ce
                    n_components = st.number_input("哈希维度", min_value=2, max_value=20, value=8)
                    encoder = ce.HashingEncoder(cols=[selected_col], n_components=n_components)
                    encoded_df = encoder.fit_transform(df_encoded[selected_col])
                    df_encoded = pd.concat([df_encoded, encoded_df], axis=1)
                    st.success("✅ 哈希编码完成")
                except ImportError:
                    st.error("❌ 需要安装 category_encoders 库")
            
            # 保存编码后的数据
            if st.button("💾 保存编码后的数据"):
                new_dataset_key = f'dataset_encoded_{dataset_name}'
                st.session_state[new_dataset_key] = {
                    'data': df_encoded,
                    'name': f'已编码_{dataset_name}',
                    'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'quality_score': perform_data_quality_check(df_encoded),
                    'file_size': len(df_encoded) * len(df_encoded.columns) * 8
                }
                st.success("✅ 编码后的数据已保存!")

def variable_creation(df, dataset_name):
    """变量创建"""
    st.markdown("#### 🔄 变量创建")
    
    creation_type = st.selectbox(
        "创建类型",
        ["数学运算", "条件创建", "分组统计", "时间特征", "文本处理"]
    )
    
    if creation_type == "数学运算":
        st.markdown("**数学运算创建新变量:**")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) < 2:
            st.warning("⚠️ 需要至少2个数值变量进行数学运算")
            return
        
        col1, col2, col3 = st.columns(3)
        with col1:
            var1 = st.selectbox("变量1", numeric_cols)
        with col2:
            operation = st.selectbox("运算", ["+", "-", "*", "/", "**", "log", "sqrt"])
        with col3:
            if operation in ["+", "-", "*", "/", "**"]:
                var2 = st.selectbox("变量2", numeric_cols)
            else:
                var2 = None
        
        new_var_name = st.text_input("新变量名", value=f"{var1}_{operation}_{var2}" if var2 else f"{operation}_{var1}")
        
        if st.button("➕ 创建变量"):
            df_new = df.copy()
            
            try:
                if operation == "+":
                    df_new[new_var_name] = df_new[var1] + df_new[var2]
                elif operation == "-":
                    df_new[new_var_name] = df_new[var1] - df_new[var2]
                elif operation == "*":
                    df_new[new_var_name] = df_new[var1] * df_new[var2]
                elif operation == "/":
                    df_new[new_var_name] = df_new[var1] / df_new[var2]
                elif operation == "**":
                    df_new[new_var_name] = df_new[var1] ** df_new[var2]
                elif operation == "log":
                    df_new[new_var_name] = np.log(df_new[var1])
                elif operation == "sqrt":
                    df_new[new_var_name] = np.sqrt(df_new[var1])
                
                st.success(f"✅ 新变量 '{new_var_name}' 创建成功")
                
                # 显示新变量的统计信息
                st.write("新变量统计信息:")
                st.write(df_new[new_var_name].describe())
                
            except Exception as e:
                st.error(f"❌ 创建失败: {str(e)}")
    
    elif creation_type == "条件创建":
        st.markdown("**条件创建新变量:**")
        
        condition_col = st.selectbox("条件变量", df.columns.tolist())
        condition_type = st.selectbox("条件类型", ["数值条件", "文本条件", "多条件"])
        
        new_var_name = st.text_input("新变量名", value=f"{condition_col}_category")
        
        if condition_type == "数值条件":
            threshold = st.number_input("阈值")
            operator = st.selectbox("操作符", [">", ">=", "<", "<=", "==", "!="])
            
            true_value = st.text_input("满足条件时的值", value="高")
            false_value = st.text_input("不满足条件时的值", value="低")
            
            if st.button("➕ 创建条件变量"):
                df_new = df.copy()
                
                if operator == ">":
                    condition = df_new[condition_col] > threshold
                elif operator == ">=":
                    condition = df_new[condition_col] >= threshold
                elif operator == "<":
                    condition = df_new[condition_col] < threshold
                elif operator == "<=":
                    condition = df_new[condition_col] <= threshold
                elif operator == "==":
                    condition = df_new[condition_col] == threshold
                elif operator == "!=":
                    condition = df_new[condition_col] != threshold
                
                df_new[new_var_name] = np.where(condition, true_value, false_value)
                st.success(f"✅ 条件变量 '{new_var_name}' 创建成功")
        
        elif condition_type == "文本条件":
            text_condition = st.text_input("包含文本")
            true_value = st.text_input("包含时的值", value="是")
            false_value = st.text_input("不包含时的值", value="否")
            
            if st.button("➕ 创建文本条件变量"):
                df_new = df.copy()
                condition = df_new[condition_col].astype(str).str.contains(text_condition, na=False)
                df_new[new_var_name] = np.where(condition, true_value, false_value)
                st.success(f"✅ 文本条件变量 '{new_var_name}' 创建成功")

def variable_deletion(df, dataset_name):
    """变量删除"""
    st.markdown("#### 🗑️ 变量删除")
    
    # 显示变量列表
    st.markdown("**当前变量列表:**")
    
    var_info = pd.DataFrame({
        '变量名': df.columns,
        '数据类型': [str(dtype) for dtype in df.dtypes],
        '非空值数': [df[col].count() for col in df.columns],
        '缺失值数': [df[col].isnull().sum() for col in df.columns],
        '唯一值数': [df[col].nunique() for col in df.columns]
    })
    
    st.dataframe(var_info, use_container_width=True)
    
    # 删除选项
    deletion_method = st.radio(
        "删除方式",
        ["手动选择", "按条件删除", "删除重复列"],
        horizontal=True
    )
    
    if deletion_method == "手动选择":
        selected_vars = st.multiselect(
            "选择要删除的变量",
            df.columns.tolist(),
            help="可以选择多个变量进行删除"
        )
        
        if selected_vars and st.button("🗑️ 删除选中变量"):
            df_deleted = df.drop(columns=selected_vars)
            st.success(f"✅ 已删除 {len(selected_vars)} 个变量")
            
            # 保存删除后的数据
            if st.button("💾 保存删除后的数据"):
                new_dataset_key = f'dataset_deleted_{dataset_name}'
                st.session_state[new_dataset_key] = {
                    'data': df_deleted,
                    'name': f'已删除变量_{dataset_name}',
                    'upload_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'quality_score': perform_data_quality_check(df_deleted),
                    'file_size': len(df_deleted) * len(df_deleted.columns) * 8
                }
                st.success("✅ 删除后的数据已保存!")
    
    elif deletion_method == "按条件删除":
        condition_type = st.selectbox(
            "删除条件",
            ["缺失值比例过高", "唯一值过少", "方差过小", "相关性过高"]
        )
        
        if condition_type == "缺失值比例过高":
            threshold = st.slider("缺失值比例阈值", 0.0, 1.0, 0.5, 0.1)
            missing_ratios = df.isnull().sum() / len(df)
            to_delete = missing_ratios[missing_ratios > threshold].index.tolist()
            
            if to_delete:
                st.warning(f"将删除 {len(to_delete)} 个变量: {to_delete}")
                if st.button("🗑️ 执行删除"):
                    df_deleted = df.drop(columns=to_delete)
                    st.success(f"✅ 已删除缺失值比例过高的 {len(to_delete)} 个变量")
            else:
                st.info("没有符合条件的变量需要删除")
        
        elif condition_type == "唯一值过少":
            threshold = st.number_input("最少唯一值数", min_value=1, value=2)
            unique_counts = df.nunique()
            to_delete = unique_counts[unique_counts < threshold].index.tolist()
            
            if to_delete:
                st.warning(f"将删除 {len(to_delete)} 个变量: {to_delete}")
                if st.button("🗑️ 执行删除"):
                    df_deleted = df.drop(columns=to_delete)
                    st.success(f"✅ 已删除唯一值过少的 {len(to_delete)} 个变量")
            else:
                st.info("没有符合条件的变量需要删除")
    
    elif deletion_method == "删除重复列":
        # 检测重复列
        duplicate_cols = []
        for i in range(len(df.columns)):
            for j in range(i+1, len(df.columns)):
                if df.iloc[:, i].equals(df.iloc[:, j]):
                    duplicate_cols.append(df.columns[j])
        
        if duplicate_cols:
            st.warning(f"发现 {len(duplicate_cols)} 个重复列: {duplicate_cols}")
            if st.button("🗑️ 删除重复列"):
                df_deleted = df.drop(columns=duplicate_cols)
                st.success(f"✅ 已删除 {len(duplicate_cols)} 个重复列")
        else:
            st.info("没有发现重复列")

def data_export_section():
    """数据导出部分"""
    st.markdown("### 💾 数据导出")
    st.markdown("*将处理后的数据导出为各种格式*")
    
    # 检查是否有数据
    datasets = get_available_datasets()
    if not datasets:
        st.warning("⚠️ 没有可导出的数据")
        return
    
    # 选择要导出的数据集
    selected_dataset = st.selectbox("选择要导出的数据集", list(datasets.keys()))
    df = datasets[selected_dataset]['data']
    
    # 显示数据集信息
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("数据形状", f"{df.shape[0]} × {df.shape[1]}")
    with col2:
        st.metric("内存大小", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
    with col3:
        st.metric("数值列", f"{df.select_dtypes(include=[np.number]).shape[1]}")
    with col4:
        st.metric("文本列", f"{df.select_dtypes(include=['object']).shape[1]}")
    
    # 导出选项
    export_format = st.selectbox(
        "选择导出格式",
        ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)", "Parquet (.parquet)", "统计报告 (HTML)"]
    )
    
    # 导出设置
    with st.expander("🔧 导出设置", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            include_index = st.checkbox("包含行索引", value=False)
            selected_columns = st.multiselect(
                "选择要导出的列 (留空表示全部)",
                df.columns.tolist()
            )
        
        with col2:
            if export_format == "CSV (.csv)":
                encoding = st.selectbox("编码格式", ["utf-8", "gbk", "gb2312"])
                separator = st.selectbox("分隔符", [",", ";", "\t", "|"])
            elif export_format == "Excel (.xlsx)":
                sheet_name = st.text_input("工作表名称", value="Sheet1")
    
    # 数据预览
    if st.checkbox("预览导出数据"):
        export_df = df[selected_columns] if selected_columns else df
        st.dataframe(export_df.head(10), use_container_width=True)
    
    # 执行导出
    if st.button("📥 生成下载文件"):
        export_df = df[selected_columns] if selected_columns else df
        
        try:
            if export_format == "Excel (.xlsx)":
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    export_df.to_excel(writer, sheet_name=sheet_name, index=include_index)
                output.seek(0)
                
                st.download_button(
                    label="📥 下载 Excel 文件",
                    data=output.getvalue(),
                    file_name=f"{selected_dataset}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            elif export_format == "CSV (.csv)":
                csv_data = export_df.to_csv(index=include_index, encoding=encoding, sep=separator)
                
                st.download_button(
                    label="📥 下载 CSV 文件",
                    data=csv_data,
                    file_name=f"{selected_dataset}.csv",
                    mime="text/csv"
                )
            
            elif export_format == "JSON (.json)":
                json_data = export_df.to_json(orient='records', force_ascii=False, indent=2)
                
                st.download_button(
                    label="📥 下载 JSON 文件",
                    data=json_data,
                    file_name=f"{selected_dataset}.json",
                    mime="application/json"
                )
            
            elif export_format == "Parquet (.parquet)":
                output = io.BytesIO()
                export_df.to_parquet(output, index=include_index)
                output.seek(0)
                
                st.download_button(
                    label="📥 下载 Parquet 文件",
                    data=output.getvalue(),
                    file_name=f"{selected_dataset}.parquet",
                    mime="application/octet-stream"
                )
            
            elif export_format == "统计报告 (HTML)":
                # 生成统计报告
                report_html = generate_statistical_report(export_df, selected_dataset)
                
                st.download_button(
                    label="📥 下载统计报告",
                    data=report_html,
                    file_name=f"{selected_dataset}_report.html",
                    mime="text/html"
                )
            
            st.success("✅ 文件生成成功，请点击下载按钮!")
            
        except Exception as e:
            st.error(f"❌ 导出失败: {str(e)}")

def generate_statistical_report(df, dataset_name):
    """生成统计报告HTML"""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>数据统计报告 - {dataset_name}</title>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ background-color: #f8f9fa; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9ecef; border-radius: 5px; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>数据统计报告</h1>
            <h2>数据集: {dataset_name}</h2>
            <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h3>数据概览</h3>
            <div class="metric">总行数: {df.shape[0]:,}</div>
            <div class="metric">总列数: {df.shape[1]:,}</div>
            <div class="metric">数值列: {df.select_dtypes(include=[np.number]).shape[1]}</div>
            <div class="metric">文本列: {df.select_dtypes(include=['object']).shape[1]}</div>
        </div>
        
        <div class="section">
            <h3>描述性统计</h3>
            {df.describe().to_html()}
        </div>
        
        <div class="section">
            <h3>缺失值统计</h3>
            {pd.DataFrame({'缺失值数': df.isnull().sum(), '缺失率(%)': (df.isnull().sum() / len(df) * 100).round(2)}).to_html()}
        </div>
        
        <div class="section">
            <h3>数据类型</h3>
            {pd.DataFrame({'数据类型': df.dtypes}).to_html()}
        </div>
    </body>
    </html>
    """
    
    return html_content



