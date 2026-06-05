"""
全局配置参数 — 控制路线A/B切换、运动提取方法、投影类型等。
"""

import os
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# ============================================================
# 核心路由选择
# ============================================================
ROUTE = "B"            # "A" = 鱼眼域直接处理, "B" = 校正后处理, "BOTH" = 两条路线都跑
MOTION_METHOD = "optical_flow"  # "frame_diff" | "optical_flow" | "background_sub" | "all"
PROJECTION = "perspective"      # Route B 目标投影: "perspective" | "cylindrical" | "equirectangular"

# ============================================================
# 数据路径
# ============================================================
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "homework2")
RGB_DIR = DATA_DIR + r"\rgb_images"
PREV_DIR = DATA_DIR + r"\previous_images"
CALIB_DIR = DATA_DIR + r"\calibration_data"
GT_DIR = DATA_DIR + r"\motion_annotation\GroudTruth"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

# ============================================================
# radial_poly 相机内参 (所有帧共享)
# ============================================================
INTRINSIC = {
    "model": "radial_poly",
    "poly_order": 4,
    "k1": 339.749,
    "k2": -31.988,
    "k3": 48.275,
    "k4": -7.201,
    "cx_offset": 3.942,
    "cy_offset": -3.093,
    "width": 1280,
    "height": 966,
    "aspect_ratio": 1.0,
}

# 推导量
K1, K2, K3, K4 = INTRINSIC["k1"], INTRINSIC["k2"], INTRINSIC["k3"], INTRINSIC["k4"]
CX = INTRINSIC["width"] / 2.0 + INTRINSIC["cx_offset"]   # 643.942
CY = INTRINSIC["height"] / 2.0 + INTRINSIC["cy_offset"]  # 479.907
IMG_W = int(INTRINSIC["width"])
IMG_H = int(INTRINSIC["height"])
R_MAX = np.sqrt(max(CX, IMG_W - CX)**2 + max(CY, IMG_H - CY)**2)  # 最大像高

# ============================================================
# 校正 (Route B) 参数
# ============================================================
UNDIST_SIZE = (900, 600)   # 校正后图像尺寸 (w, h)
UNDIST_FOCAL = 400.0       # 透视投影焦距 (px)
BALANCE = 0.5              # FOV 保留权衡 [0,1]

# ============================================================
# 运动提取参数
# ============================================================
# 帧差法
FRAME_DIFF_THRESHOLD = 15       # 灰度差阈值
FRAME_DIFF_BLUR_KSIZE = (5, 5)  # 预处理高斯模糊核

# Farneback 稠密光流
FARNEBACK_PARAMS = dict(
    pyr_scale=0.5, levels=3, winsize=15,
    iterations=3, poly_n=5, poly_sigma=1.2, flags=0
)
FLOW_MAG_THRESHOLD = 3.0        # 光流幅度绝对阈值(像素) — 百分位禁用时使用
FLOW_MAG_PERCENTILE = 97        # 自适应阈值百分位: 0=禁用, 97=取P97
EGO_MODEL = "median"            # 自运动补偿模型: "median"|"radial"|"local"
COMPENSATE_EGO_MOTION = True    # 启用自运动补偿

# ============================================================
# 特征级帧间对齐参数 (替代简单的全局中位数光流减法)
# ============================================================
USE_FEATURE_ALIGNMENT = True    # 启用 ORB+RANSAC 帧间对齐补偿自运动
ALIGNMENT_TRANSFORM = "affine"  # "affine"(推荐,4DOF更稳定) | "homography"(8DOF)
ALIGNMENT_MAX_FEATURES = 2000   # ORB 最大特征点数
ALIGNMENT_MIN_INLIERS = 10      # RANSAC 最小内点数 (不足则回退到原方案)
ALIGNMENT_RANSAC_THRESH = 3.0   # RANSAC 重投影误差阈值 (px)
FRAME_DIFF_THRESHOLD_ALIGNED = 12  # 对齐后帧差法阈值 (对齐质量好, 可设更低)

# Lucas-Kanade 稀疏光流
if cv2 is not None:
    LK_PARAMS = dict(
        winSize=(15, 15), maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
else:
    LK_PARAMS = dict(winSize=(15, 15), maxLevel=2, criteria=(3, 10, 0.03))
SHI_TOMASI_PARAMS = dict(
    maxCorners=200, qualityLevel=0.3, minDistance=7, blockSize=7
)

# 背景建模 (MOG2/KNN)
BG_HISTORY = 100         # 连续序列较短，设小
BG_VAR_THRESHOLD = 20

# ============================================================
# 后处理参数
# ============================================================
MEDIAN_BLUR_KSIZE = 5
MORPH_OPEN_KSIZE = (3, 3)
MORPH_CLOSE_KSIZE = (7, 7)
MIN_CONTOUR_AREA = 50          # 最小运动区域面积 (px²)
MIN_CONTOUR_AREA_FISHEYE = 30  # 鱼眼域边缘区域放宽阈值

# ============================================================
# 跟踪参数
# ============================================================
TRACKING_IOU_THRESHOLD = 0.3   # 匈牙利匹配 IoU 阈值
TRACKING_MAX_MISSING = 3       # 最大丢失帧数
TRACKING_MIN_HITS = 2          # 轨迹确认最小命中帧数
LK_TRACK_MIN_DISPLACEMENT = 2.0  # LK 点跟踪最小位移

# ============================================================
# 区域划分参数 (用于中心/边缘分析)
# ============================================================
REGION_RATIOS = [0.3, 0.7]     # 中心 < 0.3*r_max < 过渡 < 0.7*r_max < 边缘
REGION_NAMES = ["center", "transition", "edge"]

# ============================================================
# 运行控制
# ============================================================
SAMPLE_IDS = None     # None = 处理全部, 或 [0,1,5,10] 等
SAVE_INTERMEDIATE = True       # 是否保存中间结果可视化
VERBOSE = True                 # 打印详细日志


