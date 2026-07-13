import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from pingouin import cronbach_alpha
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram
import scipy.cluster.hierarchy as sch
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from scipy.cluster.hierarchy import linkage
import matplotlib.pyplot as plt
import base64
import requests
from dotenv import load_dotenv

load_dotenv("API.env")

st.set_page_config(page_title="数据可视化工坊",layout="wide")
st.title("数据可视化工坊")

#CSS美化
st.markdown("""
<style>
    .stAPP {
        background-color: #f8fafc;
    }
    .stExpander, .stContainer, .stTabs [role="tabpanel"] {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        border: 1px solid #eef2f6;
        margin-bottom: 12px;
    }
    h1,h2,h3 {
        color: #1f2937;
        font-weight: 600;
    }
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
        background-color: #3b82f6;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
    }
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(59,130,246,0.3);
        background-color: #2563eb;
    }
    .stAlert, .stInfo, .stWarning, .stSuccess, .stError {
        border-radius: 8px;
    }
    .stDataFrame {
        border-radius: 8px;
        overflew: hidden;
    }
    .css-1d391kg, .stSidebar {
        background-color: #ffffff;
        border-right: 1px solid #eef2f6;
    }
    [data-testid="metric-container"] {
        background: #ffffff;
        border-radius: 8px;
        padding: 12px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        border: 1px solid #eef2f6;
    }
</style>
""",unsafe_allow_html=True)

if 'df' not in st.session_state:
    st.session_state.df = None
#历史记录存储
if 'df_history' not in st.session_state:
    st.session_state.df_history = []

uploaded_file = st.file_uploader("上传数据文件(支持CSV和Excel格式)",type=['csv','xlsx','xls'])

if uploaded_file is not None:
    #读取表格文件设置
    if uploaded_file.name.endswith('csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    
    st.session_state.df = df
    st.success(f"成功加载{uploaded_file.name},共{df.shape[0]}行,{df.shape[1]}列")

    #数据预览设置
    st.subheader("数据预览")
    st.dataframe(df)

    st.caption(f"当前列名:{' , '.join(df.columns)}")

    #数据清洗功能设置
    st.subheader("数据清洗")

    def apply_clean(operation_func):
        st.session_state.df_history.append(st.session_state.df.copy())
        st.session_state.df = operation_func(st.session_state.df)
        st.rerun()

    with st.expander("缺失值处理",expanded=True):
        col1,col2 = st.columns([1,3])
        with col1:
            target_col = st.selectbox("选择你要处理的列",st.session_state.df.columns,key="missing_col")
        with col2:
            method = st.selectbox("选择处理方法",["删除该列","删除含缺失值的行","均值填充","中位数填充","众数填充","指定值填充"],key="missing_method")
        
        fill_value = None
        if method == "指定值填充":
            fill_value = st.text_input("请输入填充值",value="0",key="fill_val")
        
        #缺失值预览设置
        #统计选中列的空值数量
        null_count = df[target_col].isna().sum()
        total_rows = len(df)

        if null_count > 0:
            st.warning(f"当前选中列[{target_col}]共有{null_count}个缺失值,占总行数的{null_count/total_rows*100:.1f}%")
            if st.checkbox("点击预览含缺失值的具体行",key="show_null_rows"):
                null_rows = df[df[target_col].isna()]
                st.dataframe(null_rows,use_container_width=True)
                st.caption(f"以上是包含缺失值的{len(null_rows)}行数据,请判断是否要删除或填充")
        else:
            st.success(f"当前选中列[{target_col}]没有缺失值,数据很完整")


        if st.button("执行缺失值处理",key="btn_missing"):
            def clean_missing(d):
                d_copy = d.copy()
                if method == "删除该列":
                    return d_copy.drop(columns=[target_col])
                elif method == "删除含缺失值的行":
                    return d_copy.dropna(subset=[target_col])
                elif method == "均值填充":
                    d_copy[target_col] = d_copy[target_col].fillna(d_copy[target_col].mean())
                    return d_copy
                elif method == "中位数填充":
                    d_copy[target_col] = d_copy[target_col].fillna(d_copy[target_col].median())
                    return d_copy
                elif method == "众数填充":
                    mode_val = d_copy[target_col].mode()[0] if not d_copy[target_col].mode().empty else 0
                    d_copy[target_col] = d_copy[target_col].fillna(mode_val)
                    return d_copy
                elif method == "指定值填充":
                    try:
                        val = float(fill_value)
                    except ValueError:
                        val = fill_value
                    d_copy[target_col] = d_copy[target_col].fillna(val)
                    return d_copy
                
            apply_clean(clean_missing)
            st.success("缺失值处理完成!")

    #重复值处理
    with st.expander("重复值处理",expanded=False):
        dup_mode = st.radio("去重范围",["完全重复","按指定列判断重复"],key="dup_mode")
        dup_keys = None
        if dup_mode == "按指定列判断重复":
            dup_keys = st.multiselect("选择用于判断重复的列",df.columns,key="dup_keys")

        keep_option = st.selectbox("保留哪一条",["保留第一条","保留最后一条"],key="dup_keep")

        #预览重复行设置
        if st.checkbox("预览重复行",key="show_dup"):
            if dup_mode == "完全重复":
                dup_mask = df.duplicated(keep=False)
            else:
                if dup_keys:
                    dup_mask = df.duplicated(subset=dup_keys,keep=False)
                else:
                    dup_mask = pd.Series([False]*len(df))
            if dup_mask.any():
                dup_rows = df[dup_mask]
                st.dataframe(dup_rows,use_container_width=True)
                st.caption(f"共检测到{len(dup_rows)}行重复数据")
            else:
                st.info("没有检测到重复行")

        if st.button("执行去重",key="bin_dup"):
            def clean_dup(d):
                subset = None if dup_mode == "完全重复" else dup_keys
                keep = 'first' if keep_option == "保留第一条" else 'last'
                return d.drop_duplicates(subset=subset,keep=keep)
            
            #记录历史执行
            st.session_state.df_history.append(st.session_state.df.copy())
            st.session_state.df = clean_dup(st.session_state.df)
            st.rerun()
            st.success("重复值处理完成!")

    #异常值处理功能设置
    with st.expander("异常值处理",expanded=False):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) == 0:
            st.info(" 当前数据没有数值列,无法进行异常值检测")
        else:
            method = st.selectbox("检测方法",["Z-score(3σ原则)","IQR(箱线图法)"],key="outlier_method")
            target_col = st.selectbox("选择要检测的数据列",numeric_cols,key="outlier_col")

            #阙值可调
            if method == "Z-score(3σ原则)":
                threshold = st.slider("Z-score阙值",min_value=2.0,max_value=5.0,value=3.0,step=0.5,help="通常使用3.0,数值越小判定越严格(越多的点被判定为异常)",key="z_threshold")
            else:
                threshold = st.slider("IQR倍数",min_value=1.0,max_value=4.0,value=1.5,step=0.5,help="通常使用1.5,数值越小判定越严格",key="iqr_threshold")

            action = st.selectbox("处理方法",["删除异常行","替换为均值/中位数","截尾处理"],key="outlier_action")

            #异常值预览设置
            if st.checkbox("预览异常值所在行",key="show_outlier"):
                col_data = df[target_col]
                if method == "Z-score(3σ原则)":
                    z = np.abs((col_data - col_data.mean()) / col_data.std())
                    outlier_mask = z > threshold
                else:
                    Q1,Q3 = col_data.quantile(0.25),col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    outlier_mask = (col_data < Q1 - threshold * IQR) | (col_data > Q3 + threshold * IQR)

                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    st.warning(f"检测到{outlier_count}个异常值")
                    st.dataframe(df[outlier_mask],use_container_width=True)
                else:
                    st.success("未检测到异常值")

            #执行异常值处理
            if st.button("执行异常值处理",key="btn_outlier"):
                def clean_outlier(d):
                    d_copy = d.copy()
                    col_data = d_copy[target_col]

                    if method == "Z-score(3σ原则)":
                        z = np.abs((col_data - col_data.mean()) / col_data.std())
                        outlier_mask = z > threshold
                    else:
                        Q1,Q3 = col_data.quantile(0.25),col_data.quantile(0.75)
                        IQR = Q3 - Q1
                        outlier_mask = (col_data < Q1 - threshold * IQR) | (col_data > Q3 + threshold * IQR)

                    if action == "删除异常行":
                        return d_copy[~outlier_mask]
                    elif action == "替换为均值/中位数":
                        replace_val = d_copy[target_col].mean()
                        d_copy.loc[outlier_mask,target_col] = replace_val
                        return d_copy
                    else:
                        lower = d_copy[target_col].quantile(0.01)
                        upper = d_copy[target_col].quantile(0.99)
                        d_copy[target_col] = d_copy[target_col].clip(lower=lower,upper=upper)
                        return d_copy
                
                #记录历史并执行
                st.session_state.df_history.append(st.session_state.df.copy())
                st.session_state.df = clean_outlier(st.session_state.df)
                st.rerun()
                st.success("异常值处理完成!")

    #标准化处理设置
    with st.expander("数据标准化 / 归一化",expanded=False):
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols)  == 0:
            st.info("当前数据没有数值列,无法进行标准化处理")
        else:
            target_col = st.selectbox("选择要处理的数值列",numeric_cols,key="scale_col")
            scale_method = st.selectbox("选择方法",["Min-Max标准化","Z-score标准化","对数变换"],key="scale_method")

            #预览设置
            col_data = df[target_col]
            st.caption(f"当前统计:均值 = {col_data.mean():.4f},"
                       f"标准差 = {col_data.std():.4f},"
                       f"最小值 = {col_data.min()},"
                       f"最大值 = {col_data.max()}")
            
            #执行标准化处理
            if st.button("执行标准化处理",key="btn_scale"):
                def clean_scale(d):
                    d_copy = d.copy()
                    if scale_method == "Min-Max标准化":
                        min_val = d_copy[target_col].min()
                        max_val = d_copy[target_col].max()
                        if max_val - min_val == 0:
                            st.warning("该列所有值都相等,无法进行标准化处理")
                            return d_copy
                        d_copy[target_col] = (d_copy[target_col] - min_val) / (max_val - min_val)
                    elif scale_method == "Z-score标准化":
                        mean_val = d_copy[target_col].mean()
                        std_val = d_copy[target_col].std()
                        if std_val ==0:
                            st.warning("该列数值标准差为0,无法标准化")
                            return d_copy
                        d_copy[target_col] = (d_copy[target_col] - mean_val) / std_val
                    else:
                        min_val = d_copy[target_col].min()
                        if min_val <= 0:
                            offset = abs(min_val) + 1
                            st.info("检测到该列存在非正数(最小值为{min_val}),自动整体+{offset:.4f}后再取对数")
                            d_copy[target_col] = d_copy[target_col] + offset
                        d_copy[target_col] = np.log(d_copy[target_col])
                        st.caption("已完成对数变换(自然对数)")
                    return d_copy
                
                #记录历史并执行
                st.session_state.df_history.append(st.session_state.df.copy())
                st.session_state.df = clean_outlier(st.session_state.df)
                st.rerun()
                st.success("标准化处理完成!")

    #回退与导出功能设置
    col_back,col_export = st.columns(2)
    with col_back:
        if st.button("回退到上一步"):
            if len(st.session_state.df_history) > 0:
                st.session_state.df = st.session_state.df_history.pop()
                st.rerun()
            else:
                st.warning("没有更早的记录了")

    with col_export:
        if st.button("下载清洗后的数据(CSV)"):
            csv = st.session_state.df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("点击下载",data=csv,file_name="cleaned_data.csv",mime="text/csv")

    #聚合函数设置
    def apply_aggregation(df,x_col,y_col,agg_method):
        if agg_method == "求和":
            return df.groupby(x_col)[y_col].sum().reset_index()
        elif agg_method == "平均值":
            return df.groupby(x_col)[y_col].mean().reset_index()
        elif agg_method == "计数":
            return df.groupby(x_col)[y_col].count().reset_index()
        elif agg_method == "最大值":
            return df.groupby(x_col)[y_col].max().reset_index()
        elif agg_method == "最小值":
            return df.groupby(x_col)[y_col].min().reset_index()
        else:
            return df[[x_col,y_col]]     

    #AI结论生成   
    def generate_chart_conclusion(fig,config):
        """调用豆包大模型，对图表进行分析，生成结论"""
        #获取API key
        api_key = st.secrets.get("DOUBO_API_KEY") or os.getenv("DOUBO_API_KEY")
        if not api_key:
            return "未配置API key"
        
        #将Plotly图表导出为PNG图片的Base64编码
        try:
            img_bytes = fig.to_image(format="png",width=800,height=500)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            img_url = f"data:image/png;base64,{img_base64}"
        except Exception as e:
            return f"图表导出失败：{e}"
        
        #构建提示词
        chart_type = config.get('type','图表')
        prompt = f"""请分析这张{chart_type},用一段专业的文字（200字左右）总结最核心的数据发现。
        
        要求：
        1.指出最突出的数据特征
        2.如果有分组，对比各组差异
        3.以专业数据分析师的身份，对图表进行分析
        4.语言专业而不晦涩
        5.只输出结论，不要加前缀和后缀"""

        #构造请求
        headers = {"Content-Type":"application/json",
                   "Authorization":f"Bearer {api_key}"}
        
        payload = {
            "model":"ep-20260713181428-4pjj9",
            "messages":[
                {
                    "role":"user",
                    "content":[
                        {
                            "type":"image_url",
                            "image_url":{
                                "url":img_url
                            }
                        },
                        {
                            "type":"text",
                            "text":prompt
                        }
                    ]
                }
            ],
            "max_tokens":500,
            "temperature":0.3
        }

        #发送请求
        try:
            response = requests.post(
                "https://ark.cn-beijing.volces.com/api/v3/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            result = response.json()

            if "choices" in result and len(result["choices"]) > 0:
                return result["choices"][0]["message"]["content"]
            else:
                return f"API响应格式异常：{result}"
            
        except requests.exceptions.Timeout:
            return "请求超时"
        except requests.exceptions.ConnectionError:
            return "网络连接失败"
        except requests.exceptions.HTTPError as e:
            return f"HTTP错误：{e.response.status.code} - {e.response.text}"
        except requests.exceptions.RequestException as e:
            return f"请求异常：{e}"
        except (KeyError,IndexError) as e:
            return f"API响应格式异常：{e}"
        except Exception as e:
            return f"未知错误：{e}"


    #数据可视化模块
    st.divider()
    st.subheader("数据可视化模块")

    #初始化图表存储列表
    if 'chart_list' not in st.session_state:
        st.session_state.chart_list = []

    if st.session_state.df is not None:
        
        df_viz = st.session_state.df
        numeric_cols = df_viz.select_dtypes(include=[np.number]).columns.tolist()
        text_cols = df_viz.select_dtypes(include=['object','category']).columns.tolist()

        #可视化图表类型配置
        chart_types = {"柱状图":"比较各类别数值",
                       "条形图":"类别名较长时使用",
                       "折线图":"趋势变化",
                       "面积图":"累积趋势",
                       "直方图":"数值分布",
                       "箱线图":"分散程度+异常值",
                       "核密度图":"平滑的数值分布形态",
                       "散点图":"两变量关系",
                       "气泡图":"三变量关系",
                       "饼图":"简单占比",
                       "环形图":"美观占比",
                       "热力图":"相关性矩阵",
                       "雷达图":"多维度指标对比",
                       "瀑布图":"累积增减变化"}
        
        #配置功能设置
        with st.container():
            st.caption("配置图表参数,点击[添加到看板]可累积添加多张图表至看板")

            #图表标题自定义
            chart_title_input = st.text_input("图表自定义标题",key="global_chart_title")

            col1,col2,col3,col4,col5 = st.columns([1.2,1.2,1.2,1,0.8])

            with col1:
                chart_type = st.selectbox("图表类型",list(chart_types.keys()),key="viz_type")
                st.caption(f"{chart_types[chart_type]}")

            with col2:
                if chart_type in ["直方图","箱线图","核密度图"]:
                    x_col = st.selectbox("选择数值列",numeric_cols,key="viz_x")
                elif chart_type == "热力图":
                    x_col = st.selectbox("选择数值列(多个请用逗号分隔)",numeric_cols,key="viz_x",help="选一个字段,所有数值列会自动计算相关性")
                else:
                    x_col = st.selectbox("x轴(分类/时间)",df_viz.columns,key="viz_x")
            
            with col3:
                if chart_type in ["直方图","箱线图","核密度图"]:
                    y_col = None
                    color_col = None
                    st.info(f"分析字段:{x_col}")
                
                elif chart_type == "热力图":
                    selected_cols = st.multiselect("请选择绘制热力图的数值列",options=numeric_cols,default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols,key="heatmap_cols")
                    x_col = None
                    y_col = None
                    color_col = None
                    agg_method = None
                
                elif chart_type in ["柱状图","条形图","折线图","面积图","雷达图"]:
                    #需要聚合的图表类型
                    y_col = st.selectbox("Y轴",numeric_cols,key="viz_y")

                    #雷达图仅保留平均值，其他保留全部聚合方式
                    if chart_type == "雷达图":
                        agg_method = st.selectbox("聚合方式",["平均值"],key="agg_method_radar")
                    else:
                        agg_method = st.selectbox("聚合方式",["求和","平均值","计数","最大值","最小值"],key=f"agg_method_{chart_type}")

                    color_options = [None] + text_cols
                    color_col = st.selectbox("分组颜色(可选)",color_options,key="viz_color")

                    st.caption(f"将按[{x_col}]分组，对[{y_col}]进行[{agg_method}]")

                elif chart_type in ["饼图","环形图"]:
                    #饼图/环形图：计数or求和
                    pie_agg_method = st.radio("统计方式",["计数","求和"],horizontal=True,key="pie_agg_method")
                    if pie_agg_method.startswith("计数"):
                        y_col = None
                        st.info("按类别计数，无需选择数值列")
                    else:
                        y_col = st.selectbox("数值列(求和)",numeric_cols,key="viz_y_pie")
                    color_col = None
                    agg_method = None


                else:
                    y_col = st.selectbox("Y轴(数值)",numeric_cols,key="viz_y")
                    color_options = [None] + text_cols
                    color_col = st.selectbox("分组颜色(可选)",color_options,key="viz_color")
                    agg_method = None

            with col4:
                color_theme = st.selectbox("颜色",["默认","学术蓝","暖色系","冷色系","莫兰迪","鲜艳","柔和"],key="viz_theme")

            with col5:
                add_btn = st.button("添加到看板",key="add_chart",use_container_width=True)
                clear_btn = st.button("清空全部",key="clear_charts",use_container_width=True)
                if clear_btn:
                    st.session_state.chart_list = []
                    st.rerun()

        #处理添加逻辑
        if add_btn:

            selected_cols = []
            pie_method = None
            agg_method = None

            if chart_type in ["饼图","环形图"]:
                pie_method = pie_agg_method if 'pie_agg_method' in locals() else "求和"
            
            if chart_type in ["柱状图","条形图","折线图","面积图","雷达图"]:
                agg_method = agg_method if 'agg_method' in locals() else "求和"

            if chart_type == "热力图":
                heatmap_cols = selected_cols if 'selected_cols' in locals() else []
                if not heatmap_cols:
                    st.warning("请至少选择一列数值再绘制热力图")
                    st.stop()
                selected_cols = heatmap_cols

            config = {"type":chart_type,
                    "x":x_col,
                    "y":y_col if 'y_col' in locals() else None,
                    "color":color_col if 'color_col' in locals() else None,
                    "theme":color_theme,
                    "custom_title":chart_title_input if 'chart_title_input' in locals() else "",
                    "agg_method":agg_method,
                    "pie_method":pie_method,
                    "heatmap_cols":selected_cols}
            st.session_state.chart_list.append(config)
            st.rerun()

        #看板展示区设置
        if len(st.session_state.chart_list) == 0:
            st.info("配置好参数后,点击[添加到看板],图表会出现在这里")
        else:
            #配色映射
            theme_map = {"默认":px.colors.qualitative.Plotly,
                         "学术蓝":px.colors.sequential.Blues[2:],
                         "暖色系":px.colors.sequential.Oranges[2:],
                         "冷色系":px.colors.sequential.Greens[2:],
                         "莫兰迪":["#8DA0CB",'#FC8D62','#66C2A5','#E78AC3','#A6D854'],
                         "鲜艳":['#FF6B6B','#4ECDC4','#45B7D1','#96CEB4','#FFEAA7'],
                         "柔和":['#A8D8EA','#AA96DA','#FCBAD3','#FFFFD2','#B5EAD7']}
            
            #遍历展示所有图表
            for idx,config in enumerate(st.session_state.chart_list):
                with st.container():
                    #标题+删除按钮
                    col_title,col_del = st.columns([6,1])
                    with col_title:
                        st.markdown(f"**图表#{idx+1}** - {config['type']} | X: {config['x']}")
                        if config.get('y'):
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;Y: {config['y']}")
                    with col_del:
                        if st.button("删除",key=f"del_chart_{idx}"):
                            st.session_state.chart_list.pop(idx)
                            st.rerun()

                    #生成图表
                    try:
                        fig = None
                        if config["type"] == "柱状图":
                            agg_method = config.get("agg_method","求和")
                            if agg_method and config.get("y") is not None:
                                plot_df = apply_aggregation(df_viz,config["x"],config["y"],agg_method)
                                y_col_plot = config["y"]
                                x_col_plot = config["x"]
                                title_suffix = f"{agg_method}"
                            else:
                                plot_df = df_viz
                                y_col_plot = config.get("y")
                                x_col_plot = config.get("x")
                                title_suffix = ""

                            fig = px.bar(plot_df,x=x_col_plot,y=y_col_plot,
                                         color=config["color"] if config.get("color") else None,
                                         color_discrete_sequence=theme_map[config["theme"]],
                                         title=f"{config['y']} 按 {config['x']} 分组({title_suffix})")
                            
                        elif config["type"] == "条形图":
                            agg_method = config.get("agg_method","求和")
                            if agg_method and config.get("y") is not None:
                                plot_df = apply_aggregation(df_viz,config["x"],config["y"],agg_method)
                                y_col_plot = config["y"]
                                x_col_plot = config["x"]
                                title_suffix = f"{agg_method}"
                            else:
                                plot_df = df_viz
                                y_col_plot = config.get("y")
                                x_col_plot = config.get("x")
                                title_suffix = ""

                            fig = px.bar(plot_df,x=y_col_plot,y=x_col_plot,
                                         color=config["color"] if config.get("color") else None,
                                         color_discrete_sequence=theme_map[config["theme"]],
                                         orientation='h',
                                         title=f"{config['y']} 按 {config['x']} 分组({title_suffix})")
                            
                        elif config["type"] == "折线图":
                            agg_method = config.get("agg_method","求和")
                            if agg_method and config.get("y") is not None:
                                plot_df = apply_aggregation(df_viz,config["x"],config["y"],agg_method)
                                y_col_plot = config["y"]
                                x_col_plot = config["x"]
                                title_suffix = f"{agg_method}"
                            else:
                                plot_df = df_viz
                                y_col_plot = config.get("y")
                                x_col_plot = config.get("x")
                                title_suffix = ""

                            fig = px.line(plot_df,x=x_col_plot,y=y_col_plot,
                                         color=config["color"] if config.get("color") else None,
                                         color_discrete_sequence=theme_map[config["theme"]],
                                         title=f"{config['y']} 趋势图{title_suffix}")
                            
                        elif config["type"] == "面积图":
                            agg_method = config.get("agg_method","求和")
                            if agg_method and config.get("y") is not None:
                                plot_df = apply_aggregation(df_viz,config["x"],config["y"],agg_method)
                                y_col_plot = config["y"]
                                x_col_plot = config["x"]
                                title_suffix = f"{agg_method}"
                            else:
                                plot_df = df_viz
                                y_col_plot = config.get("y")
                                x_col_plot = config.get("x")
                                title_suffix = ""

                            fig = px.area(plot_df,x=x_col_plot,y=y_col_plot,
                                          color=config["color"] if config.get("color") else None,
                                          color_discrete_sequence=theme_map[config["theme"]],
                                          title=f"{config['y']} 累积趋势({title_suffix})")
                            
                        elif config["type"] == "直方图":
                            fig = px.histogram(df_viz,x=config["x"],
                                               color_discrete_sequence=theme_map[config["theme"]],
                                               title=f"{config['x']} 分布直方图")
                            
                        elif config["type"] == "箱线图":
                            fig = px.box(df_viz,x=config["x"],
                                         color_discrete_sequence=theme_map[config["theme"]],
                                         title=f"{config['x']} 箱线图")
                            
                        elif config["type"] == "核密度图":
                            fig = px.kde(df_viz,x=config["x"],
                                         color=config["color"] if config.get("color") else None,
                                         color_discrete_sequence=theme_map[config["theme"]],
                                         title=f"{config['x']} 核密度图")
                            
                        elif config["type"] == "散点图":
                            fig = px.scatter(df_viz,x=config["x"],y=config["y"],
                                             color=config["color"] if config.get("color") else None,
                                             color_discrete_sequence=theme_map[config["theme"]],
                                             title=f"{config['y']} 与 {config['x']} 的散点图")
                            
                        elif config["type"] == "气泡图":
                            size_options = [col for col in numeric_cols if col != config["y"]]
                            if size_options:
                                size_col = st.selectbox("选择气泡大小字段(图表#{idx+1})",size_options,key=f"bubble_size_{idx}")
                                fig = px.scatter(df_viz,x=config["x"],y=config["y"],size=size_col,
                                                 color=config["color"] if config.get("color") else None,
                                                 color_discrete_sequence=theme_map[config["theme"]],
                                                 title=f"{config['y']} 与 {config['x']} 的关系气泡图")
                            else:
                                st.warning("没有足够的数值列作为气泡大小")

                        elif config["type"] == "饼图":
                            pie_method = config.get("pie_method","求和")
                            if pie_method and pie_method.startswith("计数"):
                                #按类别计数
                                pie_data = df_viz[config["x"]].value_counts().reset_index()
                                pie_data.columns = [config["x"],"计数"]
                                fig = px.pie(pie_data,
                                             names=config["x"],
                                             values="计数",
                                             color_discrete_sequence=theme_map[config["theme"]],
                                             title=f"{config['x']}频数分布")
                            else:
                                #按数值列求和
                                if config.get("y") is None:
                                    st.warning("请选择数值列或切换计数模式")
                                    fig = None
                                else:
                                    plot_df = df_viz.groupby(config["x"])[config["y"]].sum().reset_index()
                                    fig = px.pie(plot_df,
                                                names=config["x"],
                                                values=config["y"],
                                                color_discrete_sequence=theme_map[config["theme"]],
                                                title=f"{config['y']} 占比分布饼图")
                            
                        elif config["type"] == "环形图":
                            pie_method = config.get("pie_method","求和")

                            if pie_method and pie_method.startswith("计数"):
                                pie_data = df_viz[config["x"]].value_counts().reset_index()
                                pie_data.columns = [config["x"],"计数"]
                                fig = px.pie(pie_data,
                                             names=config["x"],
                                             values="计数",
                                             color_discrete_sequence=theme_map[config["theme"]],
                                             hole=0.4,
                                             title=f"{config['x']}频数分布环形图")
                            else:
                                if config.get("y") is None:
                                    st.warning("请选择数值列或切换计数模式")
                                    fig = None
                                else:
                                    plot_df = df_viz.groupby(config["x"])[config["y"]].sum().reset_index()
                                    fig = px.pie(plot_df,
                                                names=config["x"],
                                                values=config["y"],
                                                color_discrete_sequence=theme_map[config["theme"]],
                                                hole=0.4,
                                                title=f"{config["y"]} 占比分布环形图")
                            
                        elif config["type"] == "热力图":
                            heatmap_cols = config.get("heatmap_cols",[])
                            valid_cols = [col for col in heatmap_cols if col in df_viz.columns]

                            if len(valid_cols) < 2:
                                fig = None
                                st.warning("热力图需要至少2个数值列")
                            else:
                                corr_matrix = df_viz[numeric_cols].corr()
                                fig = px.imshow(corr_matrix,
                                                text_auto='.2f',
                                                color_continuous_scale='RdBu_r',
                                                aspect="auto",
                                                title="数值列相关性矩阵图")
                            
                        elif config["type"] == "雷达图":
                            agg_method = config.get("agg_method","平均值")
                            if agg_method and config.get("y") is not None:
                                plot_df = apply_aggregation(df_viz,config["x"],config["y"],agg_method)
                                y_col_plot = config["y"]
                                x_col_plot = config["x"]
                            else:
                                plot_df = df_viz
                                y_col_plot = config.get("y")
                                x_col_plot = config.get("x")

                            color_col = config.get("color")
                            if color_col:
                                #有分组，画多条雷达                    
                                pivot_df = plot_df.pivot(index=x_col_plot,columns=color_col,values=y_col_plot)
                                fig = go.Figure()
                                for col in pivot_df.columns:
                                    fig.add_trace(go.Scatterpolar(r=pivot_df[col].values,
                                                                  theta=pivot_df.index.tolist(),
                                                                  mode='lines+markers',
                                                                  name=col,
                                                                  fill='toself'))
                                    fig.update_layout(polar=dict(radialaxis=dict(visible=True)),title=f"{y_col_plot}多维度雷达图({agg_method})")
                            else:
                                #单条雷达图
                                fig = px.line_polar(plot_df,
                                                    r=y_col_plot,
                                                    theta=x_col_plot,
                                                    line_close=True,
                                                    title=f"{y_col_plot} 雷达图({agg_method})")
                        
                        elif config["type"] == "瀑布图":
                            df_waterfall = df_viz[[config["x"],config["y"]]].copy()
                            df_waterfall["累积值"] = df_waterfall[config["y"]].cumsum()

                            fig = px.bar(df_waterfall,x=config["x"],y=config["y"],
                                         color=config["y"],
                                         color_continuous_scale='RdBu_r',
                                         title=f"{config['y']} 瀑布图")
                            #添加累积折线
                            fig.add_scatter(x=df_waterfall[config["x"]],y=df_waterfall[config["y"]],
                                            mode='lines+markers',
                                            name='累积值',
                                            line=dict(color='black',width=2))
                            fig.update_layout(coloraxis_showscale=False)

                        if fig:
                            if config.get('custom_title'):
                                fig.update_layout(title=config['custom_title'])

                            fig.update_layout(title_x=0.5,height=380,template="plotly_white",font=dict(size=11),margin=dict(l=40,r=40,t=50,b=40))
                            st.plotly_chart(fig,use_container_width=True)

                            #导出按钮设置
                            col_export1,col_export2 = st.columns(2)
                            with col_export1:
                                #导出HTML(保留交互)
                                html_bytes = fig.to_html().encode('utf-8')
                                st.download_button(label="导出HTML",
                                                   data=html_bytes,
                                                   file_name=f"chart_{idx+1}.html",
                                                   mime="text/html",
                                                   key=f"export_html{idx}")
                            with col_export2:
                                #导出PNG
                                try:
                                    png_bytes = fig.to_image(format="png",width=800,height=500)
                                    st.download_button(label="导出PNG",
                                                       data=png_bytes,
                                                       file_name=f"chart_{idx+1}.png",
                                                       mime="image/png",
                                                       key=f"export_png_{idx}")
                                except Exception as e:
                                    st.caption("需安装kaleido才能导出PNG")

                            #AI生成结论
                            with st.container():
                                if st.button(f"生成AI结论",key=f"ai_concl_{idx}"):
                                    with st.spinner("AI正在分析数据......"):
                                        conclusion = generate_chart_conclusion(fig,config)
                                        st.session_state[f"concl_{idx}"] = conclusion

                                if f"concl_{idx}" in st.session_state:
                                    st.info(f"{st.session_state[f'concl_{idx}']}")


                    except Exception as e:
                        st.error(f"图表 #{idx+1} 生成失败: {e}")
                            

                        st.divider()

    #建模分析方法设置
    st.divider()
    st.subheader("建模分析")

    if st.session_state.df is not None:
        df_model = st.session_state.df
        numeric_cols_model = df_model.select_dtypes(include=[np.number]).columns.tolist()
        all_cols_model = df_model.columns.tolist()

        tab1,tab2,tab3,tab4,tab5 = st.tabs(["回归分析","聚类分析","预测分析","相关性分析","信效度检验"])

        with tab1:
            st.caption("线性回归 / 逻辑回归")

            reg_type = st.radio("选择回归类型",["线性回归","逻辑回归"],horizontal=True,key="reg_type")
            
            if reg_type.startswith("线性"):
                target_col = st.selectbox("选择因变量（Y，连续数值）",numeric_cols_model,key="lr_target")
                feature_cols = st.multiselect("选择自变量",[col for col in numeric_cols_model if col != target_col],key="lr_features")

                if st.button("运行线性回归",key="btn_lr"):
                    if len(feature_cols) == 0:
                        st.warning("请至少选择一个自变量")
                    else:
                        x = df_model[feature_cols].values
                        y = df_model[target_col].values

                        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

                        model = LinearRegression().fit(x_train,y_train)
                        y_pred = model.predict(x_test)

                        col1,col2,col3 = st.columns(3)
                        col1.metric("R²得分",f"{r2_score(y_test,y_pred):.4f}")
                        col2.metric("均方误差(MSE)",F"{mean_squared_error(y_test,y_pred):.4f}")
                        col3.metric("样本数",f"{len(x_train)} / {len(x_test)} 测试")

                        st.dataframe(pd.DataFrame({"特征":feature_cols,
                                                  "系数":model.coef_,
                                                  "截距":[model.intercept_] + [""] * (len(feature_cols) - 1)}))
                        
                        #残差图
                        residuals = y_test - y_pred
                        fig_res = px.scatter(x=y_pred,y=residuals,
                                             labels={"x":"预测值","y":"残差"},
                                             title="残差图（应随机分布在0附近）")
                        fig_res.add_hline(y=0,line_color="red",line_dash="dash")
                        st.plotly_chart(fig_res,use_container_width=True)

            else:
                target_col = st.selectbox("选择因变量(Y，二分类0/1)",all_cols_model,key="logit_target")

                #检查目标列是否为二分类
                unique_vals = df_model[target_col].dropna().unique()
                if len(unique_vals) != 2:
                    st.warning(f"当前列有{len(unique_vals)}个唯一值,逻辑回归需要二分类(0/1)")
                else:
                    st.info(f"检测到二分类:{unique_vals.tolist()}")
                    feature_cols = st.multiselect("选择自变量(X)",[col for col in numeric_cols_model if col != target_col],key="logit_features")

                    if st.button("运行逻辑回归",key="btn_logit"):
                        if len(feature_cols) == 0:
                            st.warning("请至少选择一个自变量")
                        else:
                            x = df_model[feature_cols].values
                            y = df_model[target_col].values

                            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

                            model = LogisticRegression(max_iter=1000).fit(x_train,y_train)
                            y_pred = model.predict(x_test)

                            col1,col2 = st.columns(2)
                            col1.metric("准确率",f"{accuracy_score(y_test,y_pred):.4f}")
                            col2.metric("样本数",f"{len(x_train)}训练 / {len(x_test)}测试")

                            st.text("分类报告:")
                            st.text(classification_report(y_test,y_pred))

                            st.dataframe(pd.DataFrame({"特征":feature_cols,
                                                       "系数":model.coef_[0],
                                                       "截距":[model.intercept_[0]] + [""] * (len(feature_cols) - 1)}))
                            
        with tab2:
            st.caption("K-Means / 层次聚类")

            cluster_type = st.radio("选择聚类方法",["K-Means聚类","层次聚类"],horizontal=True,key="cluster_type")
            
            cluster_cols = st.multiselect("选择参与聚类的数值列",numeric_cols_model,default=numeric_cols_model[:3] if len(numeric_cols_model) >= 3 else numeric_cols_model,key="cluster_cols")
            
            if len(cluster_cols) < 2:
                st.warning("请至少选择2个数值列进行聚类")
            else:
                x_cluster = df_model[cluster_cols].dropna()

                st.caption(f"有效样本数：{len(x_cluster)}行（已剔除空值")

                #检查是否为空
                if len(x_cluster) < 2:
                    st.error("有效样本不足(至少需要2行完整数据)，请检查所选列是否包含太多空值")
                else:
                    x_scaled = StandardScaler().fit_transform(x_cluster)

                    if cluster_type == "K-Means聚类":
                        #K-Means
                        n_clusters = st.slider("选择聚类数(K)",2,10,3,key="kmeans_k")

                        if st.button("运行K-Means",key="btn_kmeans"):
                            kmeans = KMeans(n_clusters=n_clusters,random_state=42,n_init=10)
                            labels = kmeans.fit_predict(x_scaled)

                            if len(x_scaled) < n_clusters:
                                st.warning("样本数({len(x_scaled)})少于聚类数({n_clusters})")
                            else:
                                col1,col2 = st.columns(2)
                                col1.metric("聚类数",n_clusters)
                                col2.metric("轮廓系数",f"{silhouette_score(x_scaled,labels):.4f}")

                                #添加标签到数据
                                x_cluster_result = x_cluster.copy()
                                x_cluster_result["聚类标签"] = labels

                                #用前两个特征画散点图
                                fig = px.scatter(x_cluster_result,
                                                x=cluster_cols[0],
                                                y=cluster_cols[1],
                                                color="聚类标签",
                                                title=f"K-Means聚类结果(K={n_clusters})",
                                                color_continuous_scale='Viridis')
                                st.plotly_chart(fig,use_container_width=True)

                                #显示每类中心
                                st.dataframe(pd.DataFrame(kmeans.cluster_centers_,columns=cluster_cols).round(3))

                    else:
                        #层次聚类
                        if st.button("运行层次聚类",key="btn_hierarchical"):
                            if len(x_scaled) < 2:
                                st.warning("样本数不足")
                            else:
                                linked = linkage(x_scaled,method='ward')

                                fig_dend = plt.figure(figsize=(10,6))
                                dendrogram(linked,truncate_mode='lastp',p=20,show_leaf_counts=True)
                                plt.title("层次聚类树状图")
                                plt.xlabel("样本索引")
                                plt.ylabel("距离")
                                st.pyplot(fig_dend)

                                #截断树状图只显示前20个样本
                                if len(x_scaled) > 20:
                                    fig_trunc = plt.figure(figsize=(10,6))
                                    dendrogram(linked,truncate_mode='lastp',p=20,show_leaf_counts=True)
                                    plt.title("层次聚类树状图(显示前20个簇)")
                                    st.pyplot(fig_trunc)


        with tab3:
            st.caption("时间序列趋势预测 / 决策树预测")

            pred_type = st.radio("选择预测方法",["时间序列趋势预测","决策树回归预测"],horizontal=True,key="pred_type")

            if pred_type.startswith("时间"):
                #时间序列趋势预测
                date_col = st.selectbox("选择日期/时间列",all_cols_model,key="data_col")

                #尝试转换成日期格式
                try:
                    df_model[date_col] = pd.to_datatime(df_model[date_col])
                    st.success("日期列识别成功")
                except:
                    st.warning("无法自动识别日期格式，请确保日期列格式正确")

                future_steps = st.number_input("预测未来期数",1,30,5,key="ts_steps")

                if st.button("运行趋势预测",key="btn_ts"):
                    #按日期排序
                    df_ts = df_model[[date_col,target_col]].dropna().sort_values(date_col)
                    df_ts["时间序号"] = range(len(df_ts))
                    
                    #训练趋势模型
                    x_ts = df_ts[["时间序号"]].values
                    y_ts = df_ts[target_col].values
                    model = LinearRegression().fit(x_ts,y_ts)

                    #预测
                    last_idx = len(df_ts)
                    future_idx = np.arange(last_idx,last_idx + future_steps).reshape(-1,1)
                    future_pred = model.predict(future_idx)

                    #生成预测日期
                    last_date = df_ts[date_col].iloc[-1]
                    date_diff = (df_ts[date_col].iloc[-1] - df_ts[date_col].iloc[-2]).days
                    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=date_diff),periods=future_steps)

                    #绘图
                    fig = px.line(df_ts,x=date_col,y=target_col,
                                  mode='lines+markers',
                                  name='预测值',
                                  line=dict(color='red',dash='dash'))
                    st.plotly_chart(fig,use_container_width=True)

                    st.dataframe(pd.DataFrame({"预测日期":future_dates,
                                               "预测值":future_pred.round(2)}))
                    
                else:
                    #决策树回归预测
                    target_col = st.selectbox("选择目标数值列",numeric_cols_model,key="dt_target")
                    feature_cols = st.multiselect("选择特征列",[col for col in all_cols_model if col != target_col],key="df_features")

                    if st.button("运行决策树预测",key="btn_dt"):
                        if len(feature_cols) == 0:
                            st.warning("请至少选择一个特征")
                        else:
                            #处理分类特征
                            x_dt = df_model[feature_cols].copy()
                            for col in x_dt.columns:
                                if x_dt[col].dtype == 'object' or x_dt[col].dtype.name == 'category':
                                    x_dt[col] = x_dt[col].astype('category').cat.codes
                            x_dt = x_dt.fillna(0)
                            y_dt = df_model[target_col].fillna(0)

                            x_train,x_test,y_train,y_test = train_test_split(x_dt,y_dt,test_size=0.2,random_state=42)

                            model = DecisionTreeRegressor(max_depth=5,random_state=42)
                            model.fit(x_train,y_train)
                            y_pred = model.predict(x_test)

                            col1,col2 = st.columns(2)
                            col1.metric("R²得分",f"{r2_score(y_test,y_pred):.4f}")
                            col2.metric("样本数",f"{len(x_train)}训练 / {len(x_test)}测试")

                            #特征重要性
                            st.dataframe(pd.DataFrame({"特征":feature_cols,"重要性":model.feature_importances_}).sort_values("重要性",ascending=False).round(4))

                            #可视化决策树
                            fig_dt = plt.figure(figsize=(12,8))
                            plot_tree(model,feature_names=feature_cols,max_depth=3,filled=True,rounded=True)
                            st.pyplot(fig_dt)

        with tab4:
            st.caption("皮尔逊相关系数 / 斯皮尔曼相关系数")

            corr_type = st.radio("选择相关系数类型",["皮尔逊(线性相关)","斯皮尔曼(秩相关)"],horizontal=True,key="corr_type")

            selected_corr_cols = st.multiselect("选择参与相关性分析的数值列",numeric_cols_model,default=numeric_cols_model[:4] if len(numeric_cols_model) >= 4 else numeric_cols_model,key="corr_cols")

            if len(selected_corr_cols) < 2:
                st.warning("请至少选择2个数值列")
            else:
                if st.button("计算相关系数矩阵",key="btn_corr"):
                    if corr_type.startswith("皮尔逊"):
                        corr_matrix = df_model[selected_corr_cols].corr(method='pearson')
                    else:
                        corr_matrix = df_model[selected_corr_cols].corr(method='spearman')

                    st.dataframe(corr_matrix.round(4),use_container_width=True)

                    #热力图
                    fig_corr = px.imshow(corr_matrix,
                                         text_auto='.2f',
                                         color_continuous_scale='RdBu_r',
                                         aspect="auto",
                                         title=f"{corr_type}相关系数矩阵")
                    st.plotly_chart(fig_corr,use_container_width=True)

                    #显示强相关关系
                    strong_corr = []
                    for i in range(len(selected_corr_cols)):
                        for j in range(i+1,len(selected_corr_cols)):
                            r = corr_matrix.iloc[i,j]
                            if abs(r) > 0.6:
                                strong_corr.append({"变量1":selected_corr_cols[i],
                                                    "变量2":selected_corr_cols[j],
                                                    "相关系数":r})
                    if strong_corr:
                        st.subheader("强相关关系(|r|>0.6)")
                        st.dataframe(pd.DataFrame(strong_corr).round(4))

        with tab5:
            st.caption("Cronbach's α / KMO / Bartlett球形检验")

            test_type = st.radio("选择信效度检验方法",["Cronbach's α(内部一致性)","KMO + Bartlett球形检验(因子分析适用性)"],horizontal=True,key="test_type")

            if test_type.startswith("Cronbach"):
                #Cronbach's α
                scale_cols = st.multiselect("选择量表题项列",numeric_cols_model,key="alpha_cols")

                if len(scale_cols) < 2:
                    st.warning("请至少选择2个量表题项")
                else:
                    if st.button("计算Cronbach's α",key="btn_alpha"):
                        #删除含有空值的行
                        alpha_data = df_model[scale_cols].dropna()
                        if len(alpha_data) < 2:
                            st.warning("有效样本不足。请检查数据")
                        else:
                            alpha = cronbach_alpha(alpha_data)[0]

                            #判断信度水平
                            if alpha >= 0.9:
                                level = "优秀"
                            elif alpha >= 0.9:
                                level = "良好"
                            elif alpha >= 0.7:
                                level = "可接受"
                            elif alpha >= 0.6:
                                level = "及格"
                            else:
                                level = "不可接受"

                            col1,col2 = st.columns(2)
                            col1.metric("Cronbach's α",f"{alpha:.4f}")
                            col2.metric("信度水平",level)

                            st.caption("若α值偏低，可考虑删除与总分相关性较低的题目")
            else:
                #KMO + Bartlett球形检验
                kmo_cols = st.multiselect("选择参与检验的数值列",numeric_cols_model,key="kmo_cols")

                if len(kmo_cols) < 3:
                    st.warning("KMO检验需要至少3个变量")
                else:
                    if st.button("运行KMO + Bartlett检验",key="btn_kmo"):
                        #计算KMO
                        data_kmo = df_model[kmo_cols].dropna()
                        if len(data_kmo) < 3:
                            st.warning("有效样本不足，请检查数据")
                        else:
                            #计算相关矩阵与偏相关矩阵
                            corr_matrix = data_kmo.corr()
                            inv_corr = pd.DataFrame(np.linalg.pinv(corr_matrix.values),
                                                    index=corr_matrix.index,
                                                    columns=corr_matrix.columns)
                            
                            r2_sum = 0
                            p2_sum = 0
                            for i in range(len(kmo_cols)):
                                for j in range(len(kmo_cols)):
                                    if i != j:
                                        r2_sum += corr_matrix.iloc[i,j]**2
                                        p2_sum += inv_corr.iloc[i,j]**2

                            kmo = r2_sum / (r2_sum + p2_sum)

                            #Bartlett检验
                            n = len(data_kmo)
                            det = np.linalg.det(corr_matrix.values)
                            chi2_stat = - (n - 1 - (2*len(kmo_cols) + 5) / 6) * np.log(det)
                            df_chi2 = len(kmo_cols) * (len(kmo_cols) - 1) / 2
                            p_value = 1 - chi2.cdf(chi2_stat,df_chi2)

                            col1,col2 = st.columns(2)
                            with col1:
                                st.metric("KMO值",f"{kmo:.4f}")
                                if kmo >= 0.9:
                                    st.success("KMO > 0.9:极适合因子分析")
                                elif kmo >= 0.8:
                                    st.success("KMO > 0.8:很适合因子分析")
                                elif kmo >= 0.7:
                                    st.info("KMO > 0.7:较适合因子分析")
                                elif kmo >= 0.6:
                                    st.warning("KOM > 0.6:勉强适合因子分析")
                                else:
                                    st.error("KMO < 0.6:不适合因子分析")
                            
                            with col2:
                                st.metric("Bartlett chi2",f"{chi2_stat:.2f}")
                                st.metric("p值",f"{p_value:.4f}")
                                if p_value < 0.05:
                                    st.success("p < 0.005,变量间存在显著相关，适合因子分析")
                                else:
                                    st.warning("p >= 0.05,变量间相关性不足")



else:
    st.info("请上传一个CSV或Excel文件")





