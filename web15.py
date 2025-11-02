import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle
import matplotlib.pyplot as plt
import matplotlib
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

if not hasattr(np, 'bool'):
    np.bool = bool

def setup_chinese_font():
    """设置中文字体（云端优先加载本地fonts目录内的CJK字体）"""
    try:
        import os
        import matplotlib.font_manager as fm

        # 优先尝试系统已安装字体
        chinese_fonts = [
            'WenQuanYi Zen Hei',
            'WenQuanYi Micro Hei',
            'SimHei',
            'Microsoft YaHei',
            'PingFang SC',
            'Hiragino Sans GB',
            'Noto Sans CJK SC',
            'Source Han Sans SC'
        ]

        available_fonts = [f.name for f in fm.fontManager.ttflist]
        for font in chinese_fonts:
            if font in available_fonts:
                matplotlib.rcParams['font.sans-serif'] = [font, 'DejaVu Sans', 'Arial']
                matplotlib.rcParams['font.family'] = 'sans-serif'
                print(f"使用中文字体: {font}")
                return font

        # 若系统无中文字体，尝试从./fonts 目录加载随应用打包的字体
        candidates = [
            'NotoSansSC-Regular.otf',
            'NotoSansCJKsc-Regular.otf',
            'SourceHanSansSC-Regular.otf',
            'SimHei.ttf',
            'MicrosoftYaHei.ttf'
        ]
        fonts_dir = os.path.join(os.path.dirname(__file__), 'fonts')
        if os.path.isdir(fonts_dir):
            for fname in candidates:
                fpath = os.path.join(fonts_dir, fname)
                if os.path.exists(fpath):
                    try:
                        fm.fontManager.addfont(fpath)
                        fp = fm.FontProperties(fname=fpath)
                        fam = fp.get_name()
                        matplotlib.rcParams['font.sans-serif'] = [fam, 'DejaVu Sans', 'Arial']
                        matplotlib.rcParams['font.family'] = 'sans-serif'
                        print(f"使用本地打包字体: {fam} ({fname})")
                        return fam
                    except Exception as ie:
                        print(f"加载本地字体失败 {fname}: {ie}")

        # 兜底：使用英文字体（中文将显示为方框）
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        print("未找到中文字体，使用默认英文字体")
        return None

    except Exception as e:
        print(f"字体设置失败: {e}")
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
        matplotlib.rcParams['font.family'] = 'sans-serif'
        return None

chinese_font = setup_chinese_font()
matplotlib.rcParams['axes.unicode_minus'] = False # 确保可以显示负号

# ==============================================================================
# 1. 项目名称和配置 
# ==============================================================================
st.set_page_config(
    page_title="基于XGBoost算法的早发心梗后心衰中西医结合预测模型",
    page_icon="❤️", 
    layout="wide"
)

if chinese_font:
    plt.rcParams['font.sans-serif'] = [chinese_font, 'DejaVu Sans', 'Arial']
else:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False 


global feature_names_display, feature_dict, variable_descriptions


feature_names_display = [
    'Outcome_CHD_DM',     # 糖尿病
    'Outcome_feiyan',     # 肺部感染
    'Tachyarrhythmia',    # 快速性心律失常
    'TCM',                # 中药干预
    'qizhixueyu',         # 气滞血瘀 (小写)
    'yangxu',             # 阳虚 (小写)
    'xueyushuiting',      # 血瘀水停 (小写)
    'age',                # 年龄 (小写)
    'Pulse_rate',         # 心率
    'Hb',                 # 血红蛋白
    'SCr',                # 血清肌酐
    'BUN'                 # 血尿素氮
]

# 12个特征的中文名称
feature_names_cn = [
    '糖尿病', '肺部感染', '快速性心律失常', '中药干预',
    '气滞血瘀', '阳虚', '血瘀水停',
    '年龄', '心率', '血红蛋白', '血清肌酐', '血尿素氮'
]

# 用于英文键名到中文显示名的映射
feature_dict = dict(zip(feature_names_display, feature_names_cn))

# 变量说明字典：键名已修改为模型要求的格式
variable_descriptions = {
    'Outcome_CHD_DM': '是否有糖尿病（0=无，1=有）',
    'Outcome_feiyan': '是否有肺部感染（0=无，1=有）',
    'Tachyarrhythmia': '是否有快速性心律失常（0=无，1=有）',
    'TCM': '是否有中药干预（0=无，1=有）',
    'qizhixueyu': '是否有气滞血瘀证（0=无，1=有）', 
    'yangxu': '是否有阳虚证（0=无，1=有）',          
    'xueyushuiting': '是否有血瘀水停证（0=无，1=有）', 
    'age': '年龄（岁）',                             
    'Pulse_rate': '心率（次/分）',
    'Hb': '血红蛋白（g/L）',
    'SCr': '血清肌酐（μmol/L）',
    'BUN': '血尿素氮（mmol/L）'
}

@st.cache_resource
def load_model(model_path: str = './xgb_model.pkl'):
    """加载模型文件，优先使用joblib，其次pickle"""
    try:
        try:
            model = joblib.load(model_path)
        except Exception:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

        # 尝试获取模型内部特征名
        model_feature_names = None
        if hasattr(model, 'feature_names_in_'):
            model_feature_names = list(model.feature_names_in_)
        else:
            try:
                # 针对XGBoost/LightGBM等尝试获取booster
                booster = getattr(model, 'get_booster', lambda: None)()
                if booster is not None:
                    model_feature_names = booster.feature_names
            except Exception:
                model_feature_names = None

        return model, model_feature_names
    except Exception as e:
        raise RuntimeError(f"无法加载模型，请检查文件路径和格式: {e}")


def main():
    global feature_names_display, feature_dict, variable_descriptions

    # ==============================================================================
    # 2. 侧边栏和主标题 
    # ==============================================================================
    # 侧边栏标题
    st.sidebar.title("早发心梗后心衰中西医结合预测模型")
    st.sidebar.image("https://img.freepik.com/free-vector/hospital-logo-design-vector-medical-cross_53876-136743.jpg", width=200) 

    # 添加系统说明到侧边栏
    st.sidebar.markdown("""
    # 系统说明

    ## 关于本系统
    这是一个基于XGBoost算法的**早发心肌梗死后心力衰竭**中西医结合预测系统，用于评估患者发生心衰的风险。

    ## 预测结果
    系统输出：
    - **心力衰竭**发生概率
    - 未发生**心力衰竭**概率
    - 风险分层（低/中/高）
    """)

    # 添加变量说明到侧边栏
    with st.sidebar.expander("变量说明"):
        for feature in feature_names_display:
            st.markdown(f"**{feature_dict.get(feature, feature)}**: {variable_descriptions.get(feature, '无详细说明')}")


    # 主页面标题
    st.title("基于XGBoost算法的早发心肌梗死后心力衰竭中西医结合预测模型")
    st.markdown("### 请在下方录入全部特征后进行预测")

    # 加载模型
    try:
        model, model_feature_names = load_model('./xgb_model.pkl')
        st.sidebar.success("模型加载成功！")
    except Exception as e:
        st.sidebar.error(f"模型加载失败: {e}")
        return


    # ==============================================================================
    # 3. 特征输入控件 - 移除小标题，每列 3 个 (已恢复心衰变量)
    # ==============================================================================
    st.header("患者指标录入")
    # 使用 4 列布局来容纳 12 个特征 (4x3=12)
    col1, col2, col3, col4 = st.columns(4) 
    
    # 类别变量的格式化函数
    to_cn = lambda x: "有" if x == 1 else "无"

    # --- 第 1 列 (特征 1-3) ---
    with col1:
        # 1. 糖尿病（0/1）
        outcome_chd_dm = st.selectbox("糖尿病", options=[0, 1], format_func=to_cn, index=0, key='dm') 
        # 2. 肺部感染（0/1）
        outcome_feiyan = st.selectbox("肺部感染", options=[0, 1], format_func=to_cn, index=0, key='fy') 
        # 3. 快速性心律失常（0/1）
        tachyarrhythmia = st.selectbox("快速性心律失常", options=[0, 1], format_func=to_cn, index=0, key='ta')

    # --- 第 2 列 (特征 4-6) ---
    with col2:
        # 4. 中药干预（0/1）
        tcm = st.selectbox("中药干预", options=[0, 1], format_func=to_cn, index=0, key='tcm')
        # 5. 气滞血瘀（0/1）
        qizhixueyu = st.selectbox("气滞血瘀证", options=[0, 1], format_func=to_cn, index=0, key='qzxy')
        # 6. 阳虚（0/1）
        yangxu = st.selectbox("阳虚证", options=[0, 1], format_func=to_cn, index=0, key='yx')

    # --- 第 3 列 (特征 7-9) ---
    with col3:
        # 7. 血瘀水停（0/1）
        xueyushuiting = st.selectbox("血瘀水停证", options=[0, 1], format_func=to_cn, index=0, key='xyst')
        # 8. 年龄（数值）
        age = st.number_input("年龄（岁）", value=60, step=1, min_value=18, max_value=120, key='age_val') 
        # 9. 心率（数值）
        pulse_rate = st.number_input("心率（次/分）", value=75, step=1, min_value=30, max_value=150, key='pr')

    # --- 第 4 列 (特征 10-12) ---
    with col4:
        # 10. 血红蛋白（数值）
        hb = st.number_input("血红蛋白（g/L）", value=120.0, step=1.0, key='hb')
        # 11. 血清肌酐（μmol/L）
        scr = st.number_input("血清肌酐（μmol/L）", value=80.0, step=0.1, key='scr')
        # 12. 血尿素氮（数值）
        bun = st.number_input("血尿素氮（mmol/L）", value=5.0, step=0.1, key='bun')


    # 预测按钮
    predict_button = st.button("开始预测", type="primary")

    if predict_button:
        # 根据模型的特征顺序构建输入DataFrame
        user_inputs = {
            'Outcome_CHD_DM': outcome_chd_dm,
            'Outcome_feiyan': outcome_feiyan,
            'Tachyarrhythmia': tachyarrhythmia,
            'TCM': tcm,
            'qizhixueyu': qizhixueyu,
            'yangxu': yangxu,
            'xueyushuiting': xueyushuiting,
            'age': age,
            'Pulse_rate': pulse_rate,
            'Hb': hb,
            'SCr': scr,
            'BUN': bun,
        }

        # 特征对齐逻辑
        if model_feature_names:
            # 简化特征名映射（假设模型特征名与 feature_names_display 相似）
            # 不再使用复杂的 APTT 映射，恢复为心衰模型的逻辑
            alias_to_user_key = {f: f for f in feature_names_display}
            
            resolved_values = []
            missing_features = []
            for c in model_feature_names: # 遍历模型要求的特征名
                ui_key = alias_to_user_key.get(c, c) 
                val = user_inputs.get(ui_key, user_inputs.get(c, None)) 
                if val is None:
                    missing_features.append(c)
                resolved_values.append(val)

            if missing_features:
                st.error(f"以下模型特征未在页面录入或名称不匹配：{missing_features}。\n请核对特征名（注意大小写）。")
                with st.expander("调试信息：模型与输入特征名对比"):
                    st.write("模型特征名：", model_feature_names)
                    st.write("页面输入键：", list(user_inputs.keys()))
                return

            input_df = pd.DataFrame([resolved_values], columns=model_feature_names)
        else:
            # 如果无法获取模型特征名，则使用 feature_names_display 顺序
            ordered_cols = feature_names_display
            input_df = pd.DataFrame([[user_inputs[c] for c in ordered_cols]], columns=ordered_cols)

        # 简单检查缺失
        if input_df.isnull().any().any():
            st.error("存在缺失的输入值，请完善后重试。")
            return

        # 确保 input_df 中的数据类型为数字
        for col in input_df.columns:
            try:
                input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
            except Exception:
                pass

        # 进行预测（概率）
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(input_df)[0]
                # 假设第1列为阴性（未发生），第2列为阳性（发生）
                if len(proba) == 2:
                    no_hf_prob = float(proba[0])
                    hf_prob = float(proba[1]) # 心力衰竭发生概率
                else:
                    raise ValueError("predict_proba返回的维度异常")
            else:
                # 预测失败的退路，概率近似
                if hasattr(model, 'decision_function'):
                    score = float(model.decision_function(input_df))
                    hf_prob = 1 / (1 + np.exp(-score))
                    no_hf_prob = 1 - hf_prob
                else:
                    pred = int(model.predict(input_df)[0])
                    hf_prob = float(pred)
                    no_hf_prob = 1 - hf_prob

            # 显示预测结果
            st.header("心力衰竭风险预测结果")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("未发生心力衰竭概率")
                st.progress(no_hf_prob) 
                st.write(f"{no_hf_prob:.2%}")
            with col2:
                st.subheader("心力衰竭发生概率")
                st.progress(hf_prob) 
                st.write(f"{hf_prob:.2%}")

            # 风险分层
            risk_level = "低风险" if hf_prob < 0.3 else ("中等风险" if hf_prob < 0.7 else "高风险")
            risk_color = "green" if hf_prob < 0.3 else ("orange" if hf_prob < 0.7 else "red")
            st.markdown(f"### 心力衰竭风险评估: <span style='color:{risk_color}'>{risk_level}</span>", unsafe_allow_html=True)
            
            # ====== 新增：诊疗建议 (已适配心衰模型) ======
            st.write("---")
            st.header("诊疗建议")
            
            if hf_prob < 0.3:
                st.markdown("#### 低风险")
                st.info("建议繼續維持原有的心衰二級預防和管理方案。密切監測患者症狀變化、體液平衡及腎功能，並鼓勵患者堅持中藥干預（如果適用）。")
            elif hf_prob < 0.7:
                st.markdown("#### 中等风险")
                st.warning("建議加強管理，特別是針對合併症（如糖尿病、肺部感染）和中醫證候（如氣滯血瘀、陽虛）的干預。需更頻繁地隨訪，調整利尿劑用量，並考慮優化心衰藥物（如ACEI/ARB/ARNI、β受體阻滯劑）。")
            else:
                st.markdown("#### 高风险")
                st.error("需立即進行心衰強化管理。考慮住院治療以優化藥物治療，並積極處理快速性心律失常和感染等誘因。在西醫規範治療基礎上，結合中醫辨證施治，以期改善症狀、穩定病情。")
            # ==========================

        except Exception as e:
            st.error(f"預測或結果展示失敗: {str(e)}")
            import traceback
            st.error(traceback.format_exc())

    # 版权或说明
    st.write("---")
    st.caption("© 2025 基于机器学习的早发心梗后心衰中西医结合预测模型")

if __name__ == "__main__":
    main()