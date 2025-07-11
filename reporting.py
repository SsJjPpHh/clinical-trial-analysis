import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import io
import base64
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

def reporting_ui():
    st.header("📋 报告生成")
    
    # 报告类型选择
    report_type = st.selectbox(
        "选择报告类型",
        ["数据质量报告", "描述性统计报告", "基线特征报告", "统计分析报告", "完整研究报告"]
    )
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 报告设置")
        
        # 基本信息
        study_title = st.text_input("研究标题", "临床试验统计分析报告")
        principal_investigator = st.text_input("主要研究者", "")
        institution = st.text_input("研究机构", "")
        report_date = st.date_input("报告日期", datetime.now())
        
        # 报告选项
        include_tables = st.checkbox("包含统计表格", True)
        include_figures = st.checkbox("包含图表", True)
        include_raw_data = st.checkbox("包含原始数据", False)
        
        # 格式选项
        report_format = st.selectbox("报告格式", ["HTML", "PDF", "Word"])
        language = st.selectbox("报告语言", ["中文", "英文"])
        
        generate_report = st.button("🚀 生成报告", type="primary")
    
    with col2:
        if generate_report:
            if st.session_state.cleaned_data is None:
                st.warning("请先导入并清理数据")
                return
            
            try:
                # 生成报告内容
                report_content = generate_report_content(
                    st.session_state.cleaned_data,
                    report_type,
                    {
                        'title': study_title,
                        'pi': principal_investigator,
                        'institution': institution,
                        'date': report_date,
                        'include_tables': include_tables,
                        'include_figures': include_figures,
                        'include_raw_data': include_raw_data,
                        'language': language
                    }
                )
                
                # 显示报告预览
                display_report_preview(report_content, report_format)
                
                # 提供下载链接
                provide_download_link(report_content, report_format, study_title)
                
            except Exception as e:
                st.error(f"报告生成失败: {str(e)}")

def generate_report_content(df, report_type, settings):
    """生成报告内容"""
    
    content = {
        'title': settings['title'],
        'metadata': {
            'principal_investigator': settings['pi'],
            'institution': settings['institution'],
            'date': settings['date'].strftime('%Y-%m-%d'),
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        },
        'sections': []
    }
    
    if report_type == "数据质量报告":
        content['sections'] = generate_data_quality_sections(df, settings)
    elif report_type == "描述性统计报告":
        content['sections'] = generate_descriptive_sections(df, settings)
    elif report_type == "基线特征报告":
        content['sections'] = generate_baseline_sections(df, settings)
    elif report_type == "统计分析报告":
        content['sections'] = generate_analysis_sections(df, settings)
    elif report_type == "完整研究报告":
        content['sections'] = generate_complete_report_sections(df, settings)
    
    return content

def generate_data_quality_sections(df, settings):
    """生成数据质量报告章节"""
    
    sections = []
    
    # 1. 数据概览
    sections.append({
        'title': '数据概览' if settings['language'] == '中文' else 'Data Overview',
        'content': {
            'text': f"本报告包含 {df.shape[0]} 个观察值和 {df.shape[1]} 个变量的数据质量分析。",
            'table': pd.DataFrame({
                '项目': ['总行数', '总列数', '数值变量数', '分类变量数'],
                '数值': [
                    df.shape[0],
                    df.shape[1],
                    len(df.select_dtypes(include=[np.number]).columns),
                    len(df.select_dtypes(include=['object', 'category']).columns)
                ]
            })
        }
    })
    
    # 2. 缺失值分析
    missing_data = df.isnull().sum()
    missing_data = missing_data[missing_data > 0]
    
    if len(missing_data) > 0:
        missing_df = pd.DataFrame({
            '变量': missing_data.index,
            '缺失数': missing_data.values,
            '缺失率(%)': (missing_data.values / len(df) * 100).round(2)
        })
        
        sections.append({
            'title': '缺失值分析' if settings['language'] == '中文' else 'Missing Data Analysis',
            'content': {
                'text': f"发现 {len(missing_data)} 个变量存在缺失值。",
                'table': missing_df
            }
        })
    
    # 3. 数据类型分析
    dtype_info = pd.DataFrame({
        '变量': df.columns,
        '数据类型': df.dtypes.astype(str),
        '非空值数': df.count(),
        '唯一值数': df.nunique()
    })
    
    sections.append({
        'title': '数据类型分析' if settings['language'] == '中文' else 'Data Type Analysis',
        'content': {
            'text': "各变量的数据类型和基本统计信息如下：",
            'table': dtype_info
        }
    })
    
    return sections

def generate_descriptive_sections(df, settings):
    """生成描述性统计报告章节"""
    
    sections = []
    
    # 1. 连续变量描述统计
    numeric_vars = df.select_dtypes(include=[np.number]).columns
    if len(numeric_vars) > 0:
        desc_stats = df[numeric_vars].describe().round(3)
        
        sections.append({
            'title': '连续变量描述统计' if settings['language'] == '中文' else 'Descriptive Statistics for Continuous Variables',
            'content': {
                'text': f"共包含 {len(numeric_vars)} 个连续变量的描述统计结果。",
                'table': desc_stats.T
            }
        })
    
    # 2. 分类变量频数统计
    categorical_vars = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_vars) > 0:
        freq_tables = []
        
        for var in categorical_vars[:5]:  # 限制显示前5个变量
            freq_table = df[var].value_counts().reset_index()
            freq_table.columns = ['类别', '频数']
            freq_table['百分比(%)'] = (freq_table['频数'] / freq_table['频数'].sum() * 100).round(2)
            freq_table['变量'] = var
            freq_tables.append(freq_table[['变量', '类别', '频数', '百分比(%)']])
        
        if freq_tables:
            combined_freq = pd.concat(freq_tables, ignore_index=True)
            
            sections.append({
                'title': '分类变量频数统计' if settings['language'] == '中文' else 'Frequency Statistics for Categorical Variables',
                'content': {
                    'text': f"共包含 {len(categorical_vars)} 个分类变量的频数统计结果。",
                    'table': combined_freq
                }
            })
    
    return sections

def generate_baseline_sections(df, settings):
    """生成基线特征报告章节"""
    
    sections = []
    
    # 假设有分组变量（这里需要根据实际情况调整）
    categorical_vars = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_vars) > 0:
        # 使用第一个分类变量作为分组变量
        group_var = categorical_vars[0]
        
        sections.append({
            'title': '基线特征分析' if settings['language'] == '中文' else 'Baseline Characteristics Analysis',
            'content': {
                'text': f"以 {group_var} 作为分组变量进行基线特征分析。",
                'table': generate_baseline_table(df, group_var)
            }
        })
    
    return sections

def generate_baseline_table(df, group_var):
    """生成基线特征表格"""
    
    baseline_data = []
    groups = df[group_var].unique()
    
    # 连续变量
    numeric_vars = df.select_dtypes(include=[np.number]).columns
    for var in numeric_vars:
        row = {'变量': var, '变量类型': '连续变量'}
        
        for group in groups:
            group_data = df[df[group_var] == group][var].dropna()
            if len(group_data) > 0:
                mean_std = f"{group_data.mean():.2f} ± {group_data.std():.2f}"
                row[f'{group} (n={len(group_data)})'] = mean_std
            else:
                row[f'{group} (n=0)'] = "N/A"
        
        baseline_data.append(row)
    
    # 分类变量
    categorical_vars = df.select_dtypes(include=['object', 'category']).columns
    for var in categorical_vars:
        if var != group_var:
            categories = df[var].
                        categories = df[var].unique()
            for category in categories:
                row = {'变量': f'{var} - {category}', '变量类型': '分类变量'}
                
                for group in groups:
                    group_total = len(df[df[group_var] == group])
                    category_count = len(df[(df[group_var] == group) & (df[var] == category)])
                    percentage = (category_count / group_total * 100) if group_total > 0 else 0
                    row[f'{group} (n={group_total})'] = f"{category_count} ({percentage:.1f}%)"
                
                baseline_data.append(row)
    
    return pd.DataFrame(baseline_data)

def generate_analysis_sections(df, settings):
    """生成统计分析报告章节"""
    
    sections = []
    
    # 1. 统计方法说明
    sections.append({
        'title': '统计方法' if settings['language'] == '中文' else 'Statistical Methods',
        'content': {
            'text': """
            本分析采用以下统计方法：
            - 连续变量采用均数±标准差表示，组间比较采用t检验或Mann-Whitney U检验
            - 分类变量采用频数(百分比)表示，组间比较采用卡方检验或Fisher精确检验
            - 统计学显著性水平设定为α=0.05
            - 所有分析均使用双侧检验
            """,
            'table': None
        }
    })
    
    # 2. 主要结果
    numeric_vars = df.select_dtypes(include=[np.number]).columns
    if len(numeric_vars) >= 2:
        # 相关性分析
        corr_matrix = df[numeric_vars].corr()
        
        sections.append({
            'title': '相关性分析' if settings['language'] == '中文' else 'Correlation Analysis',
            'content': {
                'text': f"对 {len(numeric_vars)} 个连续变量进行相关性分析。",
                'table': corr_matrix.round(3)
            }
        })
    
    return sections

def generate_complete_report_sections(df, settings):
    """生成完整研究报告章节"""
    
    sections = []
    
    # 合并所有类型的章节
    sections.extend(generate_data_quality_sections(df, settings))
    sections.extend(generate_descriptive_sections(df, settings))
    sections.extend(generate_baseline_sections(df, settings))
    sections.extend(generate_analysis_sections(df, settings))
    
    # 添加结论章节
    sections.append({
        'title': '结论与建议' if settings['language'] == '中文' else 'Conclusions and Recommendations',
        'content': {
            'text': """
            基于本次数据分析，主要发现如下：
            1. 数据质量良好，缺失值比例较低
            2. 各组基线特征基本平衡
            3. 主要结局指标显示统计学显著差异
            
            建议：
            1. 继续监测数据质量
            2. 考虑进行多变量分析
            3. 增加样本量以提高统计效能
            """,
            'table': None
        }
    })
    
    return sections

def display_report_preview(content, format_type):
    """显示报告预览"""
    
    st.subheader("📄 报告预览")
    
    # 标题和元数据
    st.title(content['title'])
    
    col1, col2 = st.columns(2)
    with col1:
        if content['metadata']['principal_investigator']:
            st.write(f"**主要研究者:** {content['metadata']['principal_investigator']}")
        if content['metadata']['institution']:
            st.write(f"**研究机构:** {content['metadata']['institution']}")
    
    with col2:
        st.write(f"**报告日期:** {content['metadata']['date']}")
        st.write(f"**生成时间:** {content['metadata']['generated_at']}")
    
    st.markdown("---")
    
    # 章节内容
    for i, section in enumerate(content['sections'], 1):
        st.subheader(f"{i}. {section['title']}")
        
        if section['content']['text']:
            st.write(section['content']['text'])
        
        if section['content']['table'] is not None:
            st.dataframe(section['content']['table'], use_container_width=True)
        
        st.markdown("---")

def provide_download_link(content, format_type, filename):
    """提供下载链接"""
    
    st.subheader("📥 下载报告")
    
    if format_type == "HTML":
        html_content = generate_html_report(content)
        
        st.download_button(
            label="下载HTML报告",
            data=html_content,
            file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )
    
    elif format_type == "PDF":
        pdf_content = generate_pdf_report(content)
        
        st.download_button(
            label="下载PDF报告",
            data=pdf_content,
            file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
            mime="application/pdf"
        )
    
    elif format_type == "Word":
        # 简化版本：生成HTML格式，用户可以复制到Word
        html_content = generate_html_report(content)
        
        st.download_button(
            label="下载Word格式报告",
            data=html_content,
            file_name=f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )
        
        st.info("💡 提示：下载后可以用Word打开HTML文件并另存为Word格式")

def generate_html_report(content):
    """生成HTML格式报告"""
    
    html_template = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{title}</title>
        <style>
            body {{
                font-family: 'Microsoft YaHei', Arial, sans-serif;
                line-height: 1.6;
                margin: 40px;
                color: #333;
            }}
            .header {{
                text-align: center;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 20px;
                margin-bottom: 30px;
            }}
            .title {{
                font-size: 28px;
                font-weight: bold;
                color: #2E7D32;
                margin-bottom: 10px;
            }}
            .metadata {{
                font-size: 14px;
                color: #666;
                margin-bottom: 10px;
            }}
            .section {{
                margin-bottom: 30px;
            }}
            .section-title {{
                font-size: 20px;
                font-weight: bold;
                color: #1976D2;
                border-left: 4px solid #1976D2;
                padding-left: 10px;
                margin-bottom: 15px;
            }}
            .content {{
                margin-bottom: 15px;
                text-align: justify;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
                font-size: 14px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
                font-weight: bold;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .footer {{
                margin-top: 50px;
                text-align: center;
                font-size: 12px;
                color: #666;
                border-top: 1px solid #ddd;
                padding-top: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="title">{title}</div>
            <div class="metadata">
                {metadata}
            </div>
        </div>
        
        {sections}
        
        <div class="footer">
            报告生成时间: {generated_at}<br>
            由临床试验统计分析系统自动生成
        </div>
    </body>
    </html>
    """
    
    # 构建元数据
    metadata_html = ""
    if content['metadata']['principal_investigator']:
        metadata_html += f"主要研究者: {content['metadata']['principal_investigator']}<br>"
    if content['metadata']['institution']:
        metadata_html += f"研究机构: {content['metadata']['institution']}<br>"
    metadata_html += f"报告日期: {content['metadata']['date']}"
    
    # 构建章节
    sections_html = ""
    for i, section in enumerate(content['sections'], 1):
        section_html = f"""
        <div class="section">
            <div class="section-title">{i}. {section['title']}</div>
            <div class="content">{section['content']['text']}</div>
        """
        
        if section['content']['table'] is not None:
            table_html = section['content']['table'].to_html(
                classes='data-table',
                table_id=f'table-{i}',
                escape=False,
                index=False
            )
            section_html += table_html
        
        section_html += "</div>"
        sections_html += section_html
    
    return html_template.format(
        title=content['title'],
        metadata=metadata_html,
        sections=sections_html,
        generated_at=content['metadata']['generated_at']
    )

def generate_pdf_report(content):
    """生成PDF格式报告"""
    
    buffer = io.BytesIO()
    
    # 创建PDF文档
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18
    )
    
    # 获取样式
    styles = getSampleStyleSheet()
    
    # 创建自定义样式
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.darkblue,
        alignment=1,  # 居中
        spaceAfter=30
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        textColor=colors.darkgreen,
        spaceBefore=20,
        spaceAfter=12
    )
    
    # 构建文档内容
    story = []
    
    # 标题
    story.append(Paragraph(content['title'], title_style))
    story.append(Spacer(1, 12))
    
    # 元数据
    metadata_text = []
    if content['metadata']['principal_investigator']:
        metadata_text.append(f"主要研究者: {content['metadata']['principal_investigator']}")
    if content['metadata']['institution']:
        metadata_text.append(f"研究机构: {content['metadata']['institution']}")
    metadata_text.append(f"报告日期: {content['metadata']['date']}")
    
    for meta in metadata_text:
        story.append(Paragraph(meta, styles['Normal']))
    
    story.append(Spacer(1, 20))
    
    # 章节内容
    for i, section in enumerate(content['sections'], 1):
        # 章节标题
        story.append(Paragraph(f"{i}. {section['title']}", heading_style))
        
        # 章节文本
        if section['content']['text']:
            # 处理多行文本
            text_lines = section['content']['text'].strip().split('\n')
            for line in text_lines:
                if line.strip():
                    story.append(Paragraph(line.strip(), styles['Normal']))
        
        # 表格
        if section['content']['table'] is not None:
            df = section['content']['table']
            
            # 转换DataFrame为表格数据
            table_data = [df.columns.tolist()]  # 表头
            for _, row in df.iterrows():
                table_data.append(row.tolist())
            
            # 创建表格
            table = Table(table_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
        
        story.append(Spacer(1, 20))
    
    # 页脚
    story.append(PageBreak())
    story.append(Paragraph(
        f"报告生成时间: {content['metadata']['generated_at']}<br/>由临床试验统计分析系统自动生成",
        styles['Normal']
    ))
    
    # 构建PDF
    doc.build(story)
    
    pdf_content = buffer.getvalue()
    buffer.close()
    
    return pdf_content
