import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# 设置页面配置
st.set_page_config(
    page_title="PI预测模型",
    page_icon="🏥",
    layout="wide"
)

# 作者和单位信息
#AUTHOR_INFO = {
#    "author": "石层层",
#    "institution": "山东药品食品职业学院"
#}

# 加载模型
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model.pkl')
        return model
    except FileNotFoundError:
        st.error("模型文件 'model.pkl' 未找到。请确保模型文件已上传。")
        return None

model = load_model()

# 特征缩写映射
feature_abbreviations = {
    "FCTI": "FCTI",
    "Age": "Age",
    "Ser": "Ser",
    "Fra": "Fra",
    "Air": "Air",
    "Com": "Com",
    "PCAT": "PCAT",
    "Braden": "Braden"
}

# 特征范围定义 - 优化步长设置
feature_ranges = {
    "FCTI": {
        "type": "numerical", 
        "min": 0, 
        "max": 40, 
        "default": 32, 
        "step": 1,  # 整数步长
        "label": "FCTI总分"
    },
    "Age": {
        "type": "numerical", 
        "min": 70, 
        "max": 98, 
        "default": 87, 
        "step": 1,  # 整数步长
        "label": "年龄（岁）"
    },
    "Ser": {
        "type": "numerical", 
        "min": 20.0, 
        "max": 60.0, 
        "default": 26.3, 
        "step": 0.1,  # 小数步长
        "label": "血清白蛋白 (g/L)"
    },
    "Fra": {
        "type": "categorical", 
        "options": [0,1,2,3,4,5,6,7,8,9,10,11,12,13], 
        "default": 9, 
        "label": "骨折类型", 
        "option_labels": {
            0: "颈椎骨折", 1: "胸椎骨折", 2: "腰椎骨折", 
            3: "股骨颈骨折", 4: "股骨粗隆间骨折", 5: "股骨干骨折", 6: "胫腓骨上段骨折",
            7: "尾骨粉碎性骨折", 8: "骶髂关节脱位", 9: "髋骨骨折", 
            10: "髌骨粉碎性骨折", 11: "髋关节内骨折", 12: "脆性骨折", 13: "其他"
        }
    },
    "Air": {
        "type": "categorical", 
        "options": [0, 1], 
        "default": 0, 
        "label": "气垫床/充气床垫", 
        "option_labels": {0: "未使用", 1: "使用"}
    },
    "Com": {
        "type": "numerical", 
        "min": 0, 
        "max": 8, 
        "default": 5, 
        "step": 1,  # 整数步长
        "label": "合并症数量"
    },
    "PCAT": {
        "type": "numerical", 
        "min": 1, 
        "max": 4, 
        "default": 2, 
        "step": 1,  # 整数步长
        "label": "PCAT总分"
    },
    "Braden": {
        "type": "numerical", 
        "min": 6, 
        "max": 23, 
        "default": 18, 
        "step": 1,  # 整数步长
        "label": "Braden总分"
    },
}

# 创建一个更稳定的背景数据集
@st.cache_resource
def create_background_data():
    """创建稳定的背景数据集"""
    background_samples = []
    
    # 创建5个不同的背景样本
    for i in range(5):
        sample = []
        for feature in feature_ranges.keys():
            prop = feature_ranges[feature]
            if prop["type"] == "numerical":
                # 使用中间值
                value = (prop["min"] + prop["max"]) / 2
                if i == 0:
                    value = prop["min"]
                elif i == 1:
                    value = prop["max"]
                elif i == 2:
                    value = prop["default"]
                elif i == 3:
                    value = prop["min"] + (prop["max"] - prop["min"]) * 0.25
                else:
                    value = prop["min"] + (prop["max"] - prop["min"]) * 0.75
                # 根据特征类型调整数值格式
                if "step" in prop and prop["step"] == 1:
                    value = int(round(value))  # 整数特征取整
                else:
                    value = round(value, 1)  # 小数特征保留1位小数
            else:
                # 对于分类变量，使用默认值
                value = prop["default"]
            sample.append(value)
        background_samples.append(sample)
    
    background_df = pd.DataFrame(background_samples, columns=feature_ranges.keys())
    background_df.columns = [feature_abbreviations[col] for col in background_df.columns]
    
    return background_df

# Streamlit 界面
st.title('联合护理模式下骨折老年卧床患者压力性损伤预测模型')

# 添加作者信息
#st.markdown(f"""
#<div style='text-align: center; color: #666; margin-top: -10px; margin-bottom: 20px;'>
#    开发单位：{AUTHOR_INFO["institution"]} | 作者：{AUTHOR_INFO["author"]}
#</div>
#""", unsafe_allow_html=True)

# 添加说明文本
st.markdown("""
本应用基于机器学习模型预测在"医院-家庭-社区"联合护理模式下的老年骨折卧床患者PI风险。
请在下方的表单中输入患者的各项指标，然后点击"开始预测"按钮。
""")

# 动态生成输入项
st.header("请输入患者各项指标:")
feature_values = []

# 创建两列布局
col1, col2 = st.columns(2)

features_list = list(feature_ranges.keys())
half_point = len(features_list) // 2

for i, feature in enumerate(features_list):
    properties = feature_ranges[feature]
    
    if i < half_point:
        with col1:
            if properties["type"] == "numerical":
                # 设置步长
                step = properties.get("step", 1)
                
                # 根据步长确定value的类型和显示格式
                if step == 1:
                    # 整数特征 - 使用%g格式自动选择整数显示
                    value = st.number_input(
                        label=f"{properties['label']}",
                        min_value=float(properties["min"]),
                        max_value=float(properties["max"]),
                        value=float(properties["default"]),
                        step=float(step),
                        format="%g",  # 使用%g格式，自动显示整数
                        help=f"范围: {properties['min']} - {properties['max']}，每次增减: {step}"
                    )
                    value = int(value)  # 转换为整数
                else:
                    # 小数特征 - 显示一位小数
                    value = st.number_input(
                        label=f"{properties['label']}",
                        min_value=float(properties["min"]),
                        max_value=float(properties["max"]),
                        value=float(properties["default"]),
                        step=float(step),
                        format="%.1f",  # 显示一位小数
                        help=f"范围: {properties['min']} - {properties['max']}，每次增减: {step}"
                    )
                    value = round(value, 1)  # 保留1位小数
                    
            elif properties["type"] == "categorical":
                option_labels = properties.get("option_labels", {k: str(k) for k in properties["options"]})
                selected_label = st.selectbox(
                    label=f"{properties['label']}",
                    options=properties["options"],
                    format_func=lambda x: option_labels[x],
                    index=properties["options"].index(properties["default"])
                )
                value = selected_label
            feature_values.append(value)
    else:
        with col2:
            if properties["type"] == "numerical":
                # 设置步长
                step = properties.get("step", 1)
                
                # 根据步长确定value的类型和显示格式
                if step == 1:
                    # 整数特征 - 使用%g格式自动选择整数显示
                    value = st.number_input(
                        label=f"{properties['label']}",
                        min_value=float(properties["min"]),
                        max_value=float(properties["max"]),
                        value=float(properties["default"]),
                        step=float(step),
                        format="%g",  # 使用%g格式，自动显示整数
                        help=f"范围: {properties['min']} - {properties['max']}，每次增减: {step}"
                    )
                    value = int(value)  # 转换为整数
                else:
                    # 小数特征 - 显示一位小数
                    value = st.number_input(
                        label=f"{properties['label']}",
                        min_value=float(properties["min"]),
                        max_value=float(properties["max"]),
                        value=float(properties["default"]),
                        step=float(step),
                        format="%.1f",  # 显示一位小数
                        help=f"范围: {properties['min']} - {properties['max']}，每次增减: {step}"
                    )
                    value = round(value, 1)  # 保留1位小数
                    
            elif properties["type"] == "categorical":
                option_labels = properties.get("option_labels", {k: str(k) for k in properties["options"]})
                selected_label = st.selectbox(
                    label=f"{properties['label']}",
                    options=properties["options"],
                    format_func=lambda x: option_labels[x],
                    index=properties["options"].index(properties["default"])
                )
                value = selected_label
            feature_values.append(value)

# 显示当前输入值预览
with st.expander("📋 当前输入值预览"):
    preview_data = []
    for i, (feature, value) in enumerate(zip(features_list, feature_values)):
        prop = feature_ranges[feature]
        if prop["type"] == "categorical" and "option_labels" in prop:
            display_value = prop["option_labels"].get(int(value), value)
        else:
            # 根据特征类型调整显示格式
            if feature in ["FCTI", "Age", "Com", "PCAT", "Braden"]:
                display_value = int(value)  # 整数特征显示整数
            elif feature == "Ser":
                display_value = round(value, 1)  # Ser显示一位小数
            else:
                display_value = value
        preview_data.append({"特征": feature_abbreviations[feature], "值": display_value})
    
    preview_df = pd.DataFrame(preview_data)
    st.dataframe(preview_df, use_container_width=True)

st.markdown("---")

# 预测与 SHAP 可视化
if model is not None and st.button("开始预测", type="primary"):
    with st.spinner('模型正在计算中，请稍候...'):
        # 创建DataFrame用于模型预测
        features_df = pd.DataFrame([feature_values], columns=features_list)

        # 模型预测
        predicted_class = model.predict(features_df)[0]
        predicted_proba = model.predict_proba(features_df)[0]

        # 提取概率 - 修复逻辑错误
        # 总是显示PI发生的概率（正类，类别1的概率）
        probability_positive = predicted_proba[1] * 100  # PI发生的概率
        probability_negative = predicted_proba[0] * 100  # 不发生PI的概率
        
        # 显示的PI发生概率
        probability = probability_positive

    # 显示预测结果
    st.subheader("预测结果")
    
    # 使用进度条和指标显示PI发生概率
    st.metric(label="PI发生概率", value=f"{probability:.2f}%")
    st.progress(min(100, int(probability)))  # 确保不超过100
    
    # 添加风险等级解读 - 基于PI发生概率
    if probability < 20:
        risk_level = "低风险"
        color = "green"
        recommendation = "建议：常规护理即可"
    elif probability < 50:
        risk_level = "中风险"
        color = "orange"
        recommendation = "建议：加强观察，增加翻身频率"
    else:
        risk_level = "高风险"
        color = "red"
        recommendation = "建议：采取强化护理措施，使用专业防压疮设备"
    
    st.markdown(f"<h4 style='color: {color};'>风险等级: {risk_level}</h4>", unsafe_allow_html=True)
    st.info(recommendation)
    
    # 预测类别解释 - 修复逻辑
    if probability_positive >= 50:  # 使用50%作为阈值
        st.warning(f"预测结果：该患者发生PI的风险较高 (概率: {probability_positive:.2f}%)")
    else:
        st.info(f"预测结果：该患者发生PI的风险较低 (概率: {probability_positive:.2f}%)")
    
    # 创建用于SHAP的DataFrame
    shap_df = pd.DataFrame([feature_values], columns=features_list)
    shap_df.columns = [feature_abbreviations[col] for col in shap_df.columns]
    
    # 计算 SHAP 值
    with st.spinner('正在生成模型解释图...'):
        try:
            # 使用预创建的背景数据集
            background_df = create_background_data()
            
            # 使用LinearExplainer
            explainer = shap.LinearExplainer(model, background_df)
            
            # 计算SHAP值
            shap_values = explainer.shap_values(shap_df)
            
            # 确保SHAP值是二维数组
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # 对于二分类逻辑回归，取正类（PI发生）的SHAP值
                shap_values_array = shap_values[1]
            elif len(shap_values.shape) == 3:
                shap_values_array = shap_values[:, :, 1]
            else:
                shap_values_array = shap_values
            
            # 获取基准值
            if isinstance(explainer.expected_value, list):
                base_value = explainer.expected_value[1]  # 正类的基准值
            else:
                base_value = explainer.expected_value
            
            # 生成 SHAP 力图
            plt.figure(figsize=(12, 4), dpi=100)
            shap.force_plot(
                base_value,
                shap_values_array[0],
                shap_df.iloc[0].values,
                feature_names=shap_df.columns.tolist(),
                matplotlib=True,
                show=False
            )
            
            # 移除中文标题
            # plt.title(f"SHAP力图 - PI发生概率: {probability:.2f}%", fontsize=12, pad=20)
            plt.tight_layout()
            
            buf_force = BytesIO()
            plt.savefig(buf_force, format="png", bbox_inches="tight", dpi=100)
            plt.close()
            
            # 生成 SHAP 瀑布图 - 使用更稳定的方法
            plt.figure(figsize=(12, 6), dpi=100)  # 增加宽度，使瀑布图更清晰
            max_display = min(8, len(shap_df.columns))
            
            # 创建Explanation对象
            exp = shap.Explanation(
                values=shap_values_array[0],
                base_values=base_value,
                data=shap_df.iloc[0].values,
                feature_names=shap_df.columns.tolist()
            )
            
            # 尝试绘制瀑布图，如果失败则使用条形图
            try:
                # 绘制瀑布图
                shap.plots.waterfall(exp, max_display=max_display, show=False)
                # 移除中文标题
                # plt.title(f"SHAP瀑布图 - PI发生概率: {probability:.2f}%", fontsize=12, pad=20)
            except Exception as e:
                st.warning(f"瀑布图生成异常，使用条形图替代: {str(e)}")
                plt.clf()  # 清除当前图形
                
                # 绘制条形图
                # 计算特征重要性
                feature_importance = np.abs(shap_values_array[0])
                sorted_idx = np.argsort(feature_importance)[-max_display:]
                
                # 创建颜色：红色表示正影响，蓝色表示负影响
                colors = ['red' if shap_values_array[0][i] > 0 else 'blue' for i in sorted_idx]
                
                plt.barh(range(len(sorted_idx)), shap_values_array[0][sorted_idx], color=colors)
                plt.yticks(range(len(sorted_idx)), [shap_df.columns[i] for i in sorted_idx])
                plt.xlabel("SHAP Value (Impact on PI Probability)")
                # 移除中文标题
                # plt.title(f"特征对PI风险的影响 - PI发生概率: {probability:.2f}%", fontsize=12, pad=20)
                
                # 添加图例
                from matplotlib.patches import Patch
                legend_elements = [Patch(facecolor='red', label='Increase PI Risk'),
                                  Patch(facecolor='blue', label='Decrease PI Risk')]
                plt.legend(handles=legend_elements, loc='lower right')
            
            plt.tight_layout()
            buf_waterfall = BytesIO()
            plt.savefig(buf_waterfall, format="png", bbox_inches="tight", dpi=100)
            plt.close()
            
            # 重置缓冲区位置
            buf_force.seek(0)
            buf_waterfall.seek(0)
            
            # 显示SHAP解释图 - 改为上下排列
            st.subheader("模型解释")
            st.markdown("以下图表显示了各个特征变量对预测结果的贡献程度：")
            
            # SHAP力图在上面
            st.markdown("#### SHAP Force Plot")
            st.image(buf_force, use_column_width=True)
            st.caption("The force plot shows how each feature pushes the model output from the base value to the final prediction")
            
            # 添加一个小分隔
            st.markdown("<br>", unsafe_allow_html=True)
            
            # SHAP瀑布图在下面
            st.markdown("#### SHAP Waterfall Plot")
            st.image(buf_waterfall, use_column_width=True)
            st.caption("The waterfall plot shows the cumulative contribution of each feature to the prediction")
            
            # 添加特征影响分析
            st.subheader("特征影响分析")
            
            # 计算每个特征的SHAP值贡献
            feature_shap = {}
            for i, feature in enumerate(shap_df.columns):
                feature_shap[feature] = shap_values_array[0][i]
            
            # 按绝对贡献值排序
            sorted_features = sorted(feature_shap.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # 显示前5个最重要的特征
            st.markdown("**对预测影响最大的特征：**")
            for feature, shap_value in sorted_features[:5]:
                direction = "增加" if shap_value > 0 else "降低"
                color = "red" if shap_value > 0 else "green"
                st.markdown(f"- **{feature}**: <span style='color:{color}'>{direction}PI风险</span> (影响值: {shap_value:.4f})", 
                           unsafe_allow_html=True)
            
            # 显示特征值
            st.subheader("当前输入的特征值")
            feature_data = []
            for i, (feature, value) in enumerate(zip(shap_df.columns, feature_values)):
                prop = feature_ranges[features_list[i]]
                if prop["type"] == "categorical" and "option_labels" in prop:
                    display_value = prop["option_labels"].get(int(value), value)
                else:
                    # 根据特征类型调整显示格式
                    if feature_abbreviations[features_list[i]] in ["FCTI", "Age", "Com", "PCAT"]:
                        display_value = int(value)  # 整数特征显示整数
                    elif feature_abbreviations[features_list[i]] == "Ser":
                        display_value = round(value, 1)  # Ser显示一位小数
                    else:
                        display_value = value
                feature_data.append({"特征": feature_abbreviations[features_list[i]], "值": display_value})
            
            feature_df = pd.DataFrame(feature_data)
            st.dataframe(feature_df, use_container_width=True)
            
            # 显示概率详情
            with st.expander("查看详细概率"):
                st.markdown(f"""
                ### 预测概率详情
                - **发生PI的概率**: {probability_positive:.2f}%
                - **不发生PI的概率**: {probability_negative:.2f}%
                - **模型预测类别**: {'发生PI' if predicted_class == 1 else '不发生PI'}
                - **决策阈值**: 50%
                - **预测置信度**: {max(probability_positive, probability_negative):.2f}%
                """)
                
        except Exception as e:
            st.error(f"生成模型解释图时出错: {str(e)}")
            st.info("""
            **解决方案：**
            1. 刷新页面并重试
            2. 确保所有输入值在合理范围内
            3. 如果问题持续，请联系开发人员
            """)

# 侧边栏信息
with st.sidebar:
    st.header("关于本应用")
    st.markdown(f"""
    
    ### 模型信息
    - **算法**: Logistic Regression (逻辑回归)
    - **训练数据**: 临床数据
    - **预测目标**: 压力性损伤(PI)风险
    - **特征数量**: 8个指标
    
    ### 特征描述
    - **FCTI总分**: 家庭照顾者能力量表总分 0 ~ 40（无量纲）
    - **Age**: 年龄 70 ~ 98（岁）
    - **Ser**: 血清白蛋白 20.0 ~ 60.0 (g/L)
    - **Fra**: 骨折类型 14类 包括颈椎骨折、胸椎骨折、腰椎骨折、股骨颈骨折、股骨粗隆间骨折、股骨干骨折、胫腓骨上段骨折、尾骨粉碎性骨折、骶髂关节脱位、髋骨骨折、髌骨粉碎性骨折、髋关节内骨折、脆性骨折等
    - **Air**: 是否使用气垫床或充气床垫
    - **Com**: 患合并症数量 0 ~ 8（个）  包括神经系统疾病、自身免疫性疾病、糖尿病、水肿、中风、下肢静脉血栓、冠心病和高血压
    - **PCAT总分**: 基层医疗质量评估量表总分 1 ~ 4（无量纲）
    - **Braden总分**: Braden量表总分 6 ~ 23（无量纲）
    """)

# 页脚
# st.markdown("---")
# st.markdown(
#     f"""
#     <div style='text-align: center; color: gray;'>
#         临床决策支持工具 • {AUTHOR_INFO["institution"]} • {AUTHOR_INFO["author"]} • 仅供参考
#     </div>
#     """, 
#     unsafe_allow_html=True
# )

# 添加SHAP图例说明
with st.expander("如何解读SHAP图"):
    st.markdown("""
    ### SHAP力图解读
    - **红色箭头**：增加PI风险的因素
    - **蓝色箭头**：降低PI风险的因素  
    - **箭头长度**：表示该因素影响程度的大小
    - **基准值**：模型在训练数据上的平均预测值
    - **输出值**：当前患者的预测概率
    
    ### SHAP瀑布图解读
    - **从上到下**：显示了每个特征如何将预测值从基准值推到最终预测值
    - **条形长度**：表示每个特征的影响大小
    - **红色条形**：正向影响（增加风险）
    - **蓝色条形**：负向影响（降低风险）
    - **底部值**：最终预测概率
    """)
