import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os

# 网页标题和说明
st.set_page_config(page_title="大米色差分析系统", layout="wide")
st.title("🍚 大米色差自动检测系统")

# 配置文件，用来保存提取好的真实标准色
CONFIG_FILE = "color_standard.json"

# 读取之前保存的标准色
if 'std_colors' not in st.session_state:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            st.session_state.std_colors = json.load(f)
    else:
        st.session_state.std_colors = {"xiang": None, "pu": None}

# 计算 Delta E 色差
def calculate_color_difference(rgb1, rgb2):
    rgb1_uint8 = np.uint8([[rgb1]])
    rgb2_uint8 = np.uint8([[rgb2]])
    lab1 = cv2.cvtColor(rgb1_uint8, cv2.COLOR_RGB2LAB)[0][0]
    lab2 = cv2.cvtColor(rgb2_uint8, cv2.COLOR_RGB2LAB)[0][0]
    
    delta_L = float(lab1[0]) - float(lab2[0])
    delta_a = float(lab1[1]) - float(lab2[1])
    delta_b = float(lab1[2]) - float(lab2[2])
    return np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)

# =======================================================
# 侧边栏：定标系统
# =======================================================
with st.sidebar:
    st.header("⚙️ 系统定标")
    
    if st.session_state.std_colors["xiang"] is None or st.session_state.std_colors["pu"] is None:
        st.warning("⚠️ 尚未建立标准色，请先定标。")
        
        std_file = st.file_uploader("上传标准图片", type=["jpg", "jpeg", "png"], key="std_upload")
        if std_file is not None:
            std_img = Image.open(std_file)
            std_array = np.array(std_img)
            if std_array.shape[2] == 4:
                std_array = cv2.cvtColor(std_array, cv2.COLOR_RGBA2RGB)
                
            # 显示原始图片参考
            st.image(std_array, caption="标准原图", use_container_width=True)
            
            s_x = st.slider("定标位置 X", 0.0, 1.0, 0.5, 0.01)
            s_y = st.slider("定标位置 Y", 0.0, 1.0, 0.5, 0.01)
            
            sh, sw = std_array.shape[:2]
            sx, sy = int(sw * s_x), int(sh * s_y)
            
            # 生成带红框的预览图
            preview_std = std_array.copy()
            cv2.rectangle(preview_std, (max(0, sx-20), max(0, sy-20)), (min(sw, sx+20), min(sh, sy+20)), (255, 0, 0), 2)
            st.image(preview_std, caption="当前取样预览", use_container_width=True)
            
            s_roi = std_array[max(0, sy-20):min(sh, sy+20), max(0, sx-20):min(sw, sx+20)]
            
            if s_roi.size > 0:
                s_color = np.mean(s_roi, axis=(0, 1))[:3].tolist()
                
                col1, col2 = st.columns(2)
                if col1.button("✅ 定为【香米】"):
                    st.session_state.std_colors["xiang"] = s_color
                    with open(CONFIG_FILE, 'w') as f:
                        json.dump(st.session_state.std_colors, f)
                    st.rerun()
                    
                if col2.button("✅ 定为【普米】"):
                    st.session_state.std_colors["pu"] = s_color
                    with open(CONFIG_FILE, 'w') as f:
                        json.dump(st.session_state.std_colors, f)
                    st.rerun()
    else:
        st.success("✅ 系统已完成定标！")
        if st.button("🔄 重新定标"):
            os.remove(CONFIG_FILE)
            st.session_state.std_colors = {"xiang": None, "pu": None}
            st.rerun()

# =======================================================
# 主界面：检测部分
# =======================================================
st.header("🔍 请上传待检测样本")
test_file = st.file_uploader("上传你要测试的图片", type=["jpg", "jpeg", "png"], key="test_upload")

if test_file is not None:
    test_img = Image.open(test_file)
    test_array = np.array(test_img)
    
    if test_array.shape[2] == 4:
        test_array = cv2.cvtColor(test_array, cv2.COLOR_RGBA2RGB)
    
    # 左中右布局：原图、带框预览、结果
    col_orig, col_preview, col_result = st.columns([1, 1, 1])
    
    with col_orig:
        st.write("📷 **原图**")
        st.image(test_array, use_container_width=True)

    with col_preview:
        st.write("🎯 **取样位置**")
        t_x = st.slider("检测取样 X轴", 0.0, 1.0, 0.5, 0.01)
        t_y = st.slider("检测取样 Y轴", 0.0, 1.0, 0.5, 0.01)
        t_r = st.slider("取样框大小", 5, 100, 30)
        
        th, tw = test_array.shape[:2]
        tx, ty = int(tw * t_x), int(th * t_y)
        
        ty_min, ty_max = max(0, ty-t_r), min(th, ty+t_r)
        tx_min, tx_max = max(0, tx-t_r), min(tw, tx+t_r)
        
        t_roi = test_array[ty_min:ty_max, tx_min:tx_max]
        
        if t_roi.size > 0:
            preview = test_array.copy()
            # 画个明显的绿框
            cv2.rectangle(preview, (tx_min, ty_min), (tx_max, ty_max), (0, 255, 0), 2)
            st.image(preview, caption="看这里确认框的位置", use_container_width=True)
        else:
            st.error("取样框在图片外！")

    with col_result:
        st.write("📊 **检测报告**")
        
        if st.session_state.std_colors["xiang"] is None or st.session_state.std_colors["pu"] is None:
            st.error("❌ 系统尚未定标，请在左侧先建立标准。")
        elif t_roi.size > 0:
            t_color = np.mean(t_roi, axis=(0, 1))[:3]
            
            # 读取真实的定标数据
            xiang_std = np.array(st.session_state.std_colors["xiang"])
            pu_std = np.array(st.session_state.std_colors["pu"])
            
            # 计算色差
            diff_x = calculate_color_difference(t_color, xiang_std)
            diff_p = calculate_color_difference(t_color, pu_std)
            
            st.divider()
            if diff_x < diff_p:
                st.success("🎯 检测结论：该样本符合【香米】特征")
            else:
                st.success("🎯 检测结论：该样本符合【普米】特征")