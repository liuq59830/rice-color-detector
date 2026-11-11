import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="大米色差智能分析系统", layout="wide")
st.title("🍚 PVDF膜/大米显色智能分析系统")
st.markdown("基于 **CIELAB 色彩空间** 与 **CIE76 \(\Delta E\) 色差算法** 的客观量化判别")

# =======================================================
# 💡 写死的标准色（记得改成你自己测出来的准确值）
# =======================================================
STANDARD_XIANGMI = np.array([238, 228, 205]) # 香米 RGB
STANDARD_PUMI = np.array([223, 211, 185])    # 普米 RGB

def rgb_to_lab(rgb):
    """将 RGB 转换为 Lab 色彩空间（展示用）"""
    rgb_uint8 = np.uint8([[rgb]])
    lab = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2LAB)[0][0]
    # OpenCV 的 Lab 范围和标准有点不一样，做个转换方便展示
    L = lab[0] * 100 / 255
    a = lab[1] - 128
    b = lab[2] - 128
    return L, a, b

def calculate_color_difference(rgb1, rgb2):
    """计算 Delta E 色差"""
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

# =======================================================
# 主界面
# =======================================================
test_file = st.file_uploader("📂 请上传待检测膜的显色照片 (支持 JPG/PNG)", type=["jpg", "jpeg", "png"])

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
        # 画个带准星的圈，更显专业
        cv2.circle(preview, (tx, ty), t_r, (0, 255, 0), 2)
        cv2.line(preview, (tx-t_r-10, ty), (tx+t_r+10, ty), (0, 255, 0), 1)
        cv2.line(preview, (tx, ty-t_r-10), (tx, ty+t_r+10), (0, 255, 0), 1)
        
        st.image(preview, caption="图 1. 颜色特征提取区域", use_container_width=True)

    with col_result:
        st.write("📊 **智能分析结果**")
        t_color = get_circular_color(test_array, tx, ty, t_r)
        
        if t_color is not None:
            # 核心技术展示：将颜色转为 Lab 空间
            L, a, b = rgb_to_lab(t_color)
            
            st.markdown(f"**1. 样本色彩特征提取 (CIELAB)**")
            st.info(f"L* (明度): {L:.1f}  |  a* (红绿轴): {a:.1f}  |  b* (黄蓝轴): {b:.1f}")
            
            # 计算色差
            diff_x = calculate_color_difference(t_color, STANDARD_XIANGMI)
            diff_p = calculate_color_difference(t_color, STANDARD_PUMI)
            
            st.markdown(f"**2. 色差比对 (\(\Delta E\))**")
            st.write(f"与【香米】标准色差 \(\Delta E\) = **{diff_x:.2f}**")
            st.write(f"与【普米】标准色差 \(\Delta E\) = **{diff_p:.2f}**")
            
            # 引入人眼感知等级评价（体现专业度）
            st.markdown(f"**3. 感知度评估 (基于 \(\Delta E\) 阈值)**")
            min_diff = min(diff_x, diff_p)
            if min_diff < 1.0:
                st.success("视觉感知等级：无明显差异 (极佳匹配)")
            elif min_diff < 2.0:
                st.success("视觉感知等级：微小差异 (高置信度匹配)")
            elif min_diff < 3.5:
                st.info("视觉感知等级：可察觉差异 (中等置信度)")
            else:
                st.warning("视觉感知等级：明显差异 (低置信度，建议重新采样)")

            st.divider()
            
            st.markdown(f"**🎯 最终判别结论**")
            if diff_x < diff_p:
                st.success(f"该膜显色结果符合【香米】特征 (置信度高出 {diff_p - diff_x:.2f})")
            else:
                st.success(f"该膜显色结果符合【普米】特征 (置信度高出 {diff_x - diff_p:.2f})")