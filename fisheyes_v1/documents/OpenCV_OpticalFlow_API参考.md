# OpenCV 光流法 API 参考

> 来源：https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html

## 一、Lucas-Kanade 稀疏光流

假设局部邻域内像素具有相似运动，使用 3×3 邻域通过最小二乘法求解过定方程组。

### 核心函数：calcOpticalFlowPyrLK

```python
nextPts, status, err = cv2.calcOpticalFlowPyrLK(
    prevImg,      # 前一帧灰度图
    nextImg,      # 当前帧灰度图
    prevPts,      # 前一帧中的特征点坐标 (N×1×2)
    nextPts,      # 输出：当前帧中对应点坐标
    winSize=(21, 21),     # 搜索窗口大小
    maxLevel=3,           # 金字塔层数
    criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30, 0.01),
    flags=0,
    minEigThreshold=1e-4  # 最小特征值阈值
)
```

### 特征点提取：goodFeaturesToTrack (Shi-Tomasi)

```python
corners = cv2.goodFeaturesToTrack(
    gray,
    maxCorners=100,       # 最大角点数
    qualityLevel=0.3,     # 质量阈值
    minDistance=7,        # 角点最小间距
    blockSize=7           # 角点检测窗口大小
)
```

### 完整示例

```python
import numpy as np
import cv2 as cv

cap = cv.VideoCapture('video.mp4')

# Shi-Tomasi 角点检测参数
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
# Lucas-Kanade 参数
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

colors = np.random.randint(0, 255, (100, 3))

ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 计算光流
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # 筛选成功跟踪的点
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # 绘制轨迹
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)),
                       colors[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, colors[i].tolist(), -1)

    img = cv.add(frame, mask)
    cv.imshow('LK Optical Flow', img)

    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv.destroyAllWindows()
```

---

## 二、Farneback 稠密光流

基于多项式展开运动估计，计算图像中所有像素的光流。

### 核心函数：calcOpticalFlowFarneback

```python
flow = cv2.calcOpticalFlowFarneback(
    prev,          # 前一帧灰度图
    next,          # 当前帧灰度图
    None,          # 初始光流（None = 自动初始化）
    pyr_scale=0.5, # 金字塔缩放因子
    levels=3,      # 金字塔层数
    winsize=15,    # 平均窗口大小，越大越平滑
    iterations=3,  # 每层迭代次数
    poly_n=5,      # 多项式展开邻域大小
    poly_sigma=1.2,# 多项式高斯平滑 sigma
    flags=0        # 标志（如 OPTFLOW_FARNEBACK_GAUSSIAN）
)
```

返回的 `flow` 是一个 `(h, w, 2)` 的数组，包含每个像素的 `(dx, dy)` 位移矢量。

### 光流可视化

```python
# 将光流转换为 HSV 颜色空间进行可视化
mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
hsv = np.zeros_like(frame)
hsv[..., 0] = ang * 180 / np.pi / 2        # 方向 → 色相
hsv[..., 1] = 255                           # 饱和度固定
hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)  # 幅度 → 明度
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
```

### 运动区域提取

```python
# 基于光流幅度提取运动区域
mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
_, motion_mask = cv2.threshold(mag, threshold, 255, cv2.THRESH_BINARY)
motion_mask = motion_mask.astype(np.uint8)

# 后处理
motion_mask = cv2.medianBlur(motion_mask, 5)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
```

---

## 两种光流对比

| 维度 | Lucas-Kanade (稀疏) | Farneback (稠密) |
|------|---------------------|-------------------|
| 密度 | 稀疏特征点 | 逐像素 |
| 输入要求 | 预提取角点 | 无需特征点 |
| 速度 | 快 | 较慢 |
| 金字塔 | 内置（maxLevel） | 内置（levels） |
| 输出 | 特征点坐标 | 完整 2 通道光流场 |
| 适用场景 | 特征明显的跟踪 | 运动区域分割、全局运动估计 |

## 鱼眼视频中的光流计算注意事项

1. **亮度恒定假设**：鱼眼镜头的 vignetting 效应（边缘变暗）会破坏此假设，建议先做亮度归一化
2. **空间平滑假设**：畸变导致边缘区域像素"拉伸"，平滑假设不成立，建议减小 `winsize` 或先在鱼眼域对特征点 `undistortPoints`
3. **路线 A（鱼眼域）**：先用 `cv2.fisheye.undistortPoints()` 对特征点去畸变，再计算光流矢量
4. **路线 B（校正域）**：先去畸变整张图，再正常计算光流
