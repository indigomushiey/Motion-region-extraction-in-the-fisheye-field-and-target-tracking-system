# 鱼眼运动检测 — V3：利用标定参数改进光流质量

## 一、项目背景

鱼眼相机具有超大视场角，广泛应用于自动驾驶环视、机器人视觉感知、智能交通、全景监控等领域。但鱼眼镜头的非线性畸变导致传统基于透视图像的运动分析方法性能下降。

本项目在 V2 基础上，探索四条利用相机标定参数的技术路线：

- **Route B（去畸变域处理）**：将鱼眼图像映射到透视域后执行运动检测
- **Hybrid Flow** ⭐：融合透视域和鱼眼域的 Farneback 光流幅值，V3 最优
- **Residual Flow（残差光流）**：利用地面投影分离背景自运动
- **Angular Flow（角位移）**：将像素位移转为物理角位移，消除畸变尺度差异

---

## 二、V2 改进历史

V2 阶段在鱼眼域直接处理（Route A）上进行了 12 个版本的迭代优化：

| 版本 | 核心思路 | F1 (13帧) | 关键发现 |
|------|---------|-----------|---------|
| V1 | 帧差种子 + 光流扩张 (seed_expand) | 0.1455 | 基准线 |
| V2 | 自适应百分位阈值 | 0.1289 | 极端帧失效 |
| V3 | V2 + 方向一致性 | 0.1303 | 边际提升 |
| V4 | 低种子 + 紧扩张 | 0.1382 | 不如 V1 |
| ⭐ V5 | **Flow Ratio 替代绝对幅值** | 0.1465 | FP↓83%，突破方向 |
| V6 | abs + ratio 双重过滤 | 0.1302 | 不如纯 ratio |
| V7 | 空间一致性种子过滤 | 0.1301 | 作为辅助模块价值大 |
| V8 | 多尺度种子扩张 | 0.1358 | ❌ 失败，已放弃 |
| V9 | 帧自适应策略选择 | 0.1421 | 方向对，实现粗糙 |
| ⭐ V10 | 自适应 ratio + 空间种子 + 方向 | 0.1542 | ✅ 当时最优 |
| ⭐⭐ V11 | 区域自适应 ratio + V1 回退 + CCA | 0.1639 | 13帧最优 |
| ⭐⭐⭐ V12 | **+CCA 去噪 + 信号验证 + 种子面积检查** | 0.2008 | V2 最终版本 |

**V12 核心架构**：
```
diff seeds → 5-bin p95自适应参数 → flow ratio 候选区
→ 距离约束扩张 → CCA三级过滤(L1面积+L2形状+L3密度)
→ P0-2内部信号验证 → P0-3种子面积检查 → 形态学
```

已在 20 帧测试集上达到 Mean F1 = 0.2008，FP 相比 V1 减少 78.6%。

---

## 三、V3 改进动机

V2 的 V1~V12 已将参数调优空间基本挖掘完毕。核心瓶颈转向**如何利用相机标定参数从信号源头上改进光流质量**。

所有帧共享同一标定参数：
- 内参：radial_poly 模型 (k1=339.75, k2=-31.99, k3=48.28, k4=-7.21)
- 外参：相机固定安装 (前 3.75m, 高 0.66m, 俯仰≈0.6°)

---

## 四、V3 技术路线与实验结果

### 4.1 Route B — 去畸变域检测

**思路**：利用 radial_poly 标定将鱼眼图像映射至透视域，在透视域运行 V12，再将 mask 反投影回鱼眼域评估。

**结果**：

| fov_scale | Mean F1 | vs V12 | 分析 |
|-----------|---------|--------|------|
| 1.0 | 0.1687 | -0.032 | 边缘目标被压缩/裁剪 |
| 0.5 | 0.1792 | -0.022 | 边缘拉伸恶化 Farneback |

**结论**：透视图中心区域 Farneback 质量确实优于鱼眼域（小目标帧提升 +0.03~0.07），但边缘透视拉伸导致 Farneback 窗口假设失效。纯 Route B 无法同时兼顾中心和边缘。

### 4.2 Residual Flow — 残差光流

**思路**：利用外参（相机高度 0.66m）计算地面平面投影，估计路面区域的自运动光流，从原始光流中减去背景分量。

**结果**：Mean F1 = 0.2001，与 V12 持平（-0.0007）。

**结论**：Farneback 光流的路面中值仅 0.2~1.0 px/frame，而噪声主体（2~5 px）来自结构性伪影而非自运动，减去微小背景分量对 magnitude 几乎无影响。

### 4.3 ⭐ Hybrid Flow — 混合光流幅值（当前最优）

**思路**：

```
鱼眼域 Farneback → mag_fish ─────────────╮
                                           ├→ w(r)·mag_p + (1-w)·mag_f → V12 Pipeline
透视域 Farneback → mag_persp → 标量反投 ─╯
                     ↑                    ↑
              fov_scale=1.0         w(r): 中心→1.0, 边缘→0.0
```

- 中心区域（r<0.4）：完全使用透视域 magnitude，Farneback 质量更优
- 边缘区域（r>0.7）：完全使用鱼眼域 magnitude，避免透视拉伸
- 过渡带（0.4<r<0.7）：余弦平滑

**结果（20帧）**：

| 指标 | V1 | V12 | **HF (混合)** | Δ vs V12 |
|------|-----|------|------|------|
| **Mean F1** | 0.1546 | 0.2008 | **0.2040** | **+0.0032** |
| Precision | 0.1033 | 0.1574 | 0.1610 | +0.0036 |
| Recall | 0.8432 | 0.3716 | 0.3745 | +0.0029 |
| Frame wins | 4 | 6 | **10** | — |

**关键单帧**：00005 (+0.038), 00053 (+0.016), 00037 (+0.012), 00006 (+0.010)

### 4.4 Angular Flow — 角位移

**思路**：利用 radial_poly 标定将每个像素的 flow 向量转为物理角位移：

```
ang_mag(u,v) = arccos(ray(u,v) · ray(u+fx, v+fy))
```

角位移在中心和边缘有统一的物理含义（rad/frame），消除畸变引起的尺度差异。

**结果（20帧）**：

| 指标 | V12 | HF | AB |
|------|-----|-----|-----|
| Mean F1 | 0.2008 | **0.2040** | 0.1978 |

**结论**：方向正确（00000 +0.015, 00053 +0.030），但像素域阈值直接线性缩放到角位移域不够精确，需单独搜索参数。架构已就绪。

### 4.5 四条路线汇总

| 路线 | Mean F1 | vs V12 | 状态 |
|------|---------|--------|------|
| V12 (V2 基线) | 0.2008 | — | 基准 |
| Route B | 0.1792 | -0.022 | ❌ 边缘问题无法解决 |
| Residual Flow | 0.2001 | -0.001 | ≈ 持平，无效 |
| **Hybrid Flow** | **0.2040** | **+0.0032** | ✅ V3 最优 |
| Angular Flow | 0.1978 | -0.003 | ⏳ 参数待调优 |

---

## 五、处理全流程（Hybrid Flow）

```
输入: prev_f, curr_f (鱼眼灰度, 1280×966)
               │
  ┌────────────┴────────────┐
  │  鱼眼域 Farneback        │  透视域 Farneback
  │  flow_f = FB(p_f, c_f)  │  p_p = undistort(p_f)
  │  mag_f = ‖flow_f‖       │  c_p = undistort(c_f)
  │                         │  flow_p = FB(p_p, c_p)
  │                         │  mag_p = ‖flow_p‖
  │                         │  mag_pr = reproject(mag_p)
  └────────────┬────────────┘
               │
       mag = w(r)·mag_pr + (1-w)·mag_f
       w(r): 中心≈1.0, 边缘≈0.0 (余弦过渡)
               │
  ┌────────────┴──────────────────────────┐
  │           V12 Pipeline                 │
  │                                        │
  │  diff = |c_f - p_f|, p95 = P95(diff)  │
  │                                        │
  │  p95>120 or <20? ──→ V1 fallback      │
  │  else: 5-bin 自适应参数                │
  │    seed = diff > seed_th (空间过滤)    │
  │    cand = mag_ratio > ratio_th         │
  │    expand = cand (距seed ≤ expand_d)   │
  │    mask = seed | expand                │
  │                                        │
  │  CCA 三级过滤:                         │
  │    L1: area < 100px → 删除            │
  │    P0-2: 信号验证 (diff/mag vs 全局)   │
  │    P0-3: 种子面积比 < 3% → 删除        │
  │    L2+L3: 形状/密度检查                │
  │                                        │
  │  morph open + close (7×7)              │
  └────────────────────────────────────────┘
               │
        输出: mask (0/255 二值)
```

### 全流程关键参数

| 阶段 | 参数 | 值 | 说明 |
|------|------|-----|------|
| 混合 | w(r) 过渡带 | r∈[0.4, 0.7] | 余弦平滑 |
| 去畸变 | fov_scale | 1.0 | 透视图视场 |
| 候选 | ratio_thresh | 1.3~2.2 | p95_diff 5档自适应 |
| 候选 | flow_thresh | 1.5~2.0 | p95>70 时启用 |
| 种子 | seed_thresh | 14~22 | p95 分档 |
| 种子 | min_neighbors | 3~4 | 7×7 邻域像素数 |
| 扩张 | expand_dist | 4~12 | p95 分档 |
| CCA | min_area | 100px | 最小组件面积 |
| CCA | sig_diff_ratio | 1.5 | mean_diff/global_diff |
| CCA | sig_mag_ratio | 1.3 | mean_mag/global_mag |
| CCA | seed_area_ratio | 0.03 | 组件内种子占比 |
| 形态学 | morph_ksize | 7 | 开闭运算核大小 |

---

## 六、项目结构

```
fisheyes_v3/
├── core/
│   ├── calib.py              # 向量化 radial_poly 去畸变 + 反投影
│   ├── angular.py            # 像素光流转角位移 (rad/frame)
│   ├── ground.py             # 地面平面投影 + 道路 mask
│   ├── motion.py             # 透视域 V12 运动检测 (Route B)
│   ├── motion_hybrid.py       # ⭐ Hybrid Flow (V3 最优)
│   ├── motion_angular.py     # Angular Flow + Hybrid Flow
│   └── motion_residual.py    # 残差光流 (背景减法)
├── scripts/
│   ├── eval_hybrid.py         # ⭐ Hybrid Flow vs V12 vs V1
│   ├── eval_angular.py       # Angular vs Hybrid Flow vs V12
│   ├── eval_route_b.py       # Route B vs V12
│   └── eval_residual.py      # Residual Flow vs V12
├── output/evaluation/
│   ├── hybrid/per_frame/      # ⭐ Hybrid Flow0帧对比图
│   ├── angular/per_frame/    # Angular Flow 对比图
│   ├── route_b/per_frame/    # Route B 对比图
│   └── residual_flow/        # 残差光流对比图
└── readme.md                 # 本文件
```

---

## 七、运行方式

```bash
cd fisheyes_v3

# Hybrid Flow 评估（V3 最优方案）
python scripts/eval_hybrid.py

# Angular Flow 评估
python scripts/eval_angular.py

# Route B 评估
python scripts/eval_route_b.py

# 残差光流评估
python scripts/eval_residual.py
```

所有脚本自动引用 `fisheyes_v2` 的数据集和 V1/V12 基准代码。

---

## 八、后续方向

| 优先级 | 方向 | 说明 |
|--------|------|------|
| ⭐⭐ | Angular Flow 参数搜索 | 像素阈值→角位移阈值的精确映射 |
| ⭐⭐ | 径向统计归一化 | 全数据集 expected_mag(r)，替代 5-bin 分档 |
| ⭐⭐ | 时间一致性 | 130帧来自10个序列，跨帧平滑 |
| ⭐ | 混合权重+阈值联合调优 | 网格搜索 w(r) 过渡带 + ratio 阈值 |
