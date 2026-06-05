# OpenCV 背景减除 API 参考

> 来源：https://docs.opencv.org/4.x/d1/dc5/tutorial_background_subtraction.html

## 概述

背景减除通过将每一帧与维护的背景模型进行比较，生成前景掩膜（二值图像），突出显示运动物体。主要包括两个阶段：**背景初始化**和**背景更新**。

## 两种主要方法

### MOG2 — 高斯混合模型

基于高斯混合模型的背景/前景分割算法，内置阴影检测功能。

```python
mog2 = cv2.createBackgroundSubtractorMOG2(
    history=500,         # 用于构建背景模型的帧数（默认 500）
    varThreshold=16,     # 马氏距离平方阈值，值越高检测越保守
    detectShadows=True   # 是否检测阴影（阴影区域在掩膜中标记为灰色 127）
)
```

**适用场景**：室内监控、停车场、动态背景但光照相对稳定的场景。

### KNN — K 近邻背景减除

基于 K 近邻策略的背景/前景分割算法，同样支持阴影检测。

```python
knn = cv2.createBackgroundSubtractorKNN(
    history=500,          # 历史帧数
    dist2Threshold=400.0, # 平方距离阈值，值越高检测越保守
    detectShadows=True
)
```

**适用场景**：室外复杂光照、红外视频、精度要求高的场景。

## 核心使用模式

```python
import cv2

# 创建背景减除器
backSub = cv2.createBackgroundSubtractorMOG2()

# 打开视频
cap = cv2.VideoCapture('video.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 应用背景减除，同时更新背景模型
    fgMask = backSub.apply(frame)

    # 可选：传递学习率控制背景更新速度
    # fgMask = backSub.apply(frame, learningRate=0.05)

    # 显示结果
    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## 后处理建议

原始前景掩膜通常含有噪声，建议进行以下后处理：

```python
# 1. 中值滤波去噪
fgMask = cv2.medianBlur(fgMask, 5)

# 2. 形态学操作
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_OPEN, kernel)   # 去除小噪点
fgMask = cv2.morphologyEx(fgMask, cv2.MORPH_CLOSE, kernel)  # 填充内部孔洞

# 3. 连通域分析
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fgMask)
# 过滤面积过小的区域
min_area = 500
for i in range(1, num_labels):
    if stats[i, cv2.CC_STAT_AREA] >= min_area:
        # 保留有效区域
        pass
```

## 参数调优速查

| 问题 | 解决方案 |
|------|----------|
| 前景全是噪点 | 增大 `varThreshold` (MOG2) 或 `dist2Threshold` (KNN) |
| 运动物体被"掏空" | 添加形态学闭运算或膨胀 |
| 背景缓慢变化导致误检 | 增大 `history`，或使用 KNN |
| 阴影被误检为前景 | 设置 `detectShadows=True` |
| 鱼眼边缘检测效果差 | 调整 distance threshold，或分区域使用不同参数 |
| 需精确轮廓 | `detectShadows=False` |

## MOG2 vs KNN 对比

| 维度 | MOG2 | KNN |
|------|------|-----|
| 速度 | 较快 | 较慢 |
| 光照突变适应性 | 一般 | 好 |
| 噪声鲁棒性 | 一般 | 好 |
| 阴影处理 | 一般 | 更好 |
| 内存占用 | 较低 | 较高 |
| 默认参数稳定性 | 好 | 需要针对场景调参 |
