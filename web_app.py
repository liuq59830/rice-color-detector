import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import os

st.set_page_config(page_title="大米色差智能分析系统", layout="wide")
st.title("🍚 PVDF膜/大米显色智能分析系统")
st.markdown("基于 **CIELAB 色彩空间** 与 **CIE76 \(\Delta E\) 色差算法** 的客观量化判别")

CONFIG_FILE = "color_standard.json"

if 'std_colors' not in st.session_state:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            st.session_state.std_colors = json.load(f)
    else:
        st.session_state.std_colors = {"xiang": None, "pu": None}

def rgb_to_lab(rgb):
    rgb_uint8 = np.uint8([[rgb]])
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)[0][0]
    return lab[0] * 100 / 255, lab[1] - 128, lab[2] - 128

def calculate_color_difference(rgb1, rgb2):
    rgb1_uint8 = np.uint8([[rgb1]])
    rgb2_uint8 = np.uint8([[rgb2]])
    lab1 = cv2.cvtColor(rgb1_uint8, cv2.COLOR_RGB2LAB)[0][0]
    lab2 = cv2.cvtColor(rgb2_uint8, cv2.COLOR_RGB2LAB)[0][0]
    delta_L = float(lab1[0]) - float(lab2[0])
    delta_a = float(lab1[1]) - float(lab2[1])
    delta_b = float(lab1[2]) - float(lab2[2])
    return np.sqrt(delta_L**2 + delta_a**2 + delta_b**2)

def get_circular_color(img, center_x, center_y, radius):
    h, w = img.shape[:2]
    y_min, y_max = max(0, center_y-radius), min(h, center_y+radius)
    x_min, x_max = max(0, center_x-radius), min(w, center_x+radius)
    roi = img[y_min:y_max, x_min:x_max]
    if roi.size == 0: return None
    mask = np.zeros(roi.shape[:2], dtype=np.uint8)
    cv2.circle(mask, (center_x - x_min, center_y - y_min), radius, 255, -1)
    pixels = roi[mask == 255]
    if len(pixels) == 0: return None
    return np.mean(pixels, axis=0)[:3]

# 侧边栏
with st.sidebar:
    st.header("⚙️ 标准色定标系统")
    if st.session_state.std_colors["xiang"] is None or st.session_state.std_colors["pu"] is None:
        st.warning("⚠️ 尚未建立标准色，请先上传标准图。")
        std_file = st.file_uploader("上传标准图片", type=["jpg", "jpeg", "png"], key="std_upload")
        if std_file is not None:
            std_img = Image.open(std_file)
            std_array = np.array(std_img)
            if len(std_array.shape) == 3 and std_array.shape[2] == 4:
                std_array = cv2.cvtColor(std_array, cv2.COLOR_RGBA2RGB)
                
            s_x = st.slider("定标位置 X", 0.0, 1.0, 0.5, 0.01)
            s_y = st.slider("定标位置 Y", 0.0, 1.0, 0.5, 0.01)
            s_r = st.slider("定标圆圈半径", 5, 100, 20)
            
            sh, sw = std_array.shape[:2]
            sx, sy = int(sw * s_x), int(sh * s_y)
            preview_std = std_array.copy()
            cv2.circle(preview_std, (sx, sy), s_r, (255, 0, 0), 2)
            st.image(preview_std, caption="红色圆圈为取样位置", use_container_width=True)
            
            s_color = get_circular_color(std_array, sx, sy, s_r)
            if s_color is not None:
                s_color = s_color.tolist()
                col1, col2 = st.columns(2)
                if col1.button("✅ 定为【香米】"):
                    st.session_state.std_colors["xiang"] = s_color
                    with open(CONFIG_FILE, 'w') as f: json.dump(st.session_state.std_colors, f)
                    st.rerun()
                if col2.button("✅ 定为【普米】"):
                    st.session_state.std_colors["pu"] = s_color
                    with open(CONFIG_FILE, 'w') as f: json.dump(st.session_state.std_colors, f)
                    st.rerun()
    else:
        st.success("✅ 标准色已确定！")
        if st.button("🔄 重新定标"):
            if os.path.exists(CONFIG_FILE): os.remove(CONFIG_FILE)
            st.session_state.std_colors = {"xiang": None, "pu": None}
            st.rerun()

# 主界面
test_file = st.file_uploader("📂 请上传待检测膜的显色照片", type=["jpg", "jpeg", "png"])

if test_file is not None:
    test_img = Image.open(test_file)
    test_array = np.array(test_img)
    if len(test_array.shape) == 3 and test_array.shape[2] == 4:
        test_array = cv2.cvtColor(test_array, cv2.COLOR_RGBA2RGB)
    
    col_img, col_result = st.columns([1.2, 1])
    
    with col_img:
        st.write("🖱️ **图像采样区域标定**")
        t_x = st.slider("X轴采样坐标", 0.0, 1.0, 0.5, 0.01)
        t_y = st.slider("Y轴采样坐标", 0.0, 1.0, 0.5, 0.01)
        t_r = st.slider("采样光圈半径 (px)", 5, 100, 20)
        
        th, tw = test_array.shape[:2]
        tx, ty = int(tw * t_x), int(th * t_y)
        
        preview = test_array.copy()
        cv2.circle(preview, (tx, ty), t_r, (0, 255, 0), 2)
        cv2.line(preview, (tx-t_r-10, ty), (tx+t_r+10, ty), (0, 255, 0), 1)
        cv2.line(preview, (tx, ty-t_r-10), (tx, ty+t_r+10), (0, 255, 0), 1)
        st.image(preview, caption="图 1. 颜色特征提取区域", use_container_width=True)

    with col_result:
        st.write("📊 **智能分析结果**")
        
        if st.session_state.std_colors["xiang"] is None or st.session_state.std_colors["pu"] is None:
            st.error("❌ 请先在左侧完成标准色的定标！")
        else:
            t_color = get_circular_color(test_array, tx, ty, t_r)
            if t_color is not None:
                xiang_std = np.array(st.session_state.std_colors["xiang"])
                pu_std = np.array(st.session_state.std_colors["pu"])
                
                L, a, b = rgb_to_lab(t_color)
                st.markdown(f"**1. 样本色彩特征提取 (CIELAB)**")
                st.info(f"L* (明度): {L:.1f}  |  a* (红绿轴): {a:.1f}  |  b* (黄蓝轴): {b:.1f}")
                
                diff_x = calculate_color_difference(t_color, xiang_std)
                diff_p = calculate_color_difference(t_color, pu_std)
                
                # 计算色差和判断
                st.markdown(f"**2. 色差比对与范围判定**")
                
                if diff_x < diff_p:
                    # 属于香米的判定逻辑
                    if diff_x < 2.0:
                        st.success(f"该样本与【香米】的色差极小 (\(\\Delta E\) = {diff_x:.2f})，显著小于与【普米】的差距 (\(\\Delta E\) = {diff_p:.2f})。符合高置信度【香米】标准。")
                    else:
                        st.info(f"该样本与【香米】的色差为 {diff_x:.2f}，虽然略高于基准，但明显小于与【普米】的差距 ({diff_p:.2f})。落入合理浮动范围，判定偏向【香米】。")
                    st.markdown(f"**🎯 最终结论：符合【香米】特征**")
                
                else:
                    # 属于普米的判定逻辑
                    if diff_p < 2.0:
                        st.success(f"该样本与【普米】的色差极小 (\(\\Delta E\) = {diff_p:.2f})，显著小于与【香米】的差距 (\(\\Delta E\) = {diff_x:.2f})。符合高置信度【普米】标准。")
                    else:
                        st.info(f"该样本与【普米】的色差为 {diff_p:.2f}，虽然略高于基准，但明显小于与【香米】的差距 ({diff_x:.2f})。落入合理浮动范围，判定偏向【普米】。")
                    st.markdown(f"**🎯 最终结论：符合【普米】特征**")