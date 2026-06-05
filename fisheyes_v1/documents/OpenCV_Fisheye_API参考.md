# OpenCV Fisheye Camera Model API 参考

> 来源：https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html

## 畸变模型

鱼眼相机模型使用 4 参数畸变（k1-k4）：

```
θ_d = θ (1 + k₁ θ² + k₂ θ⁴ + k₃ θ⁶ + k₄ θ⁸)
```

像素坐标转换中包含偏斜系数 α：
```
u = f_x(x' + α·y') + c_x
v = f_y·y' + c_y
```

## 核心函数

### 1. calibrate — 相机标定

```python
ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
    objectPoints,    # 世界坐标系中的棋盘格角点
    imagePoints,     # 图像中的角点位置
    image_size,      # 图像尺寸 (w, h)
    K,               # 输出 3×3 内参矩阵
    D,               # 输出 4 元素畸变系数
    flags=0,         # 标定标志
    criteria=(cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
)
```

关键 flags：
- `cv2.fisheye.CALIB_USE_INTRINSIC_GUESS`：使用提供的 K 作为初始值
- `cv2.fisheye.CALIB_FIX_K1~K4`：固定指定畸变系数
- `cv2.fisheye.CALIB_FIX_SKEW`：固定偏斜系数
- `cv2.fisheye.CALIB_FIX_PRINCIPAL_POINT`：固定主点
- `cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC`：重新计算外参

### 2. undistortImage — 单次去畸变

```python
undistorted = cv2.fisheye.undistortImage(
    distorted,   # 输入畸变图像
    K,           # 内参矩阵
    D,           # 畸变系数
    Knew=None,   # 可选的新内参矩阵
    new_size=()  # 输出图像尺寸
)
```

注：此函数是 initUndistortRectifyMap + remap 的便捷组合。

### 3. initUndistortRectifyMap — 预计算映射表（推荐用于视频）

```python
map1, map2 = cv2.fisheye.initUndistortRectifyMap(
    K,          # 内参矩阵
    D,          # 畸变系数
    R,          # 校正旋转矩阵（单位矩阵用于单目）
    P,          # 新投影矩阵
    size,       # 图像尺寸
    cv2.CV_32FC1  # map 数据类型
)
# 然后对每帧使用
undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
```

### 4. estimateNewCameraMatrixForUndistortRectify — 估计新内参矩阵

```python
P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K,            # 原始内参矩阵
    D,            # 畸变系数
    image_size,   # 图像尺寸
    R=np.eye(3),  # 校正旋转
    balance=0.5,  # [0,1]，0=保留全部像素（多黑边），1=无黑边（丢失边缘）
    new_size=(),  # 输出尺寸
    fov_scale=1.0 # FOV 缩放因子
)
```

### 5. undistortPoints — 特征点去畸变（重要！用于鱼眼域光流）

```python
undistorted_pts = cv2.fisheye.undistortPoints(
    distorted_pts,  # 输入畸变图像中的像素坐标
    K,              # 内参矩阵
    D,              # 畸变系数
    R=np.eye(3),    # 校正旋转
    P=np.eye(3)     # 若为 identity，输出为归一化坐标
)
```

**关键说明**：如果 P 为空或单位矩阵，输出是归一化相机坐标（无量纲）。如果 P 是具体内参矩阵，输出是该图像平面上的像素坐标。此函数对于在鱼眼域直接计算光流非常有用——先对特征点去畸变再计算运动矢量。

---

## 典型使用流程

```python
# 1. 标定
ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(...)

# 2. 计算新内参
P = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, size, balance=0.5)

# 3. 预计算映射表
map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), P, size, cv2.CV_32FC1)

# 4. 逐帧去畸变
while True:
    ret, frame = cap.read()
    undistorted = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR)
    # ... 后续处理
```
