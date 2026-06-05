"""
鱼眼畸变影响分析模块 — 量化展示鱼眼畸变对运动提取与跟踪的影响。

对应题目要求:
  "分析鱼眼成像相较于传统透视成像的特点，
   以及鱼眼畸变对运动区域提取与目标跟踪的影响"
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from radial_poly import r_theta, dr_dtheta, theta_from_r


def analyze_distortion_characteristics(k1, k2, k3, k4, cx, cy, img_w, img_h, save_dir):
    """鱼眼畸变特征综合分析 — 生成多张分析图表。

    分析内容:
    1. 投影曲线对比 (鱼眼 vs 透视)
    2. 径向尺度压缩比热力图
    3. 同一物体在中心/边缘的外观差异
    4. 畸变对光流矢量的影响示意
    """
    print("\n[畸变分析] 生成鱼眼畸变特征分析图...")

    # ---- 1. 投影曲线对比 ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1a: r(θ) 投影曲线
    theta = np.linspace(0, np.pi / 2.05, 500)
    r_fisheye = r_theta(theta, k1, k2, k3, k4)
    r_perspective = k1 * np.tan(theta)  # 透视投影: r = f*tan(θ)

    axes[0].plot(np.degrees(theta), r_fisheye, 'b-', linewidth=2, label='Fisheye (radial_poly)')
    axes[0].plot(np.degrees(theta), r_perspective, 'r--', linewidth=2, label='Perspective (r=f·tanθ)')
    axes[0].axhline(y=np.sqrt((img_w/2)**2 + (img_h/2)**2), color='gray',
                    linestyle=':', label='Image diagonal limit')
    axes[0].set_xlabel('Incident Angle θ (degrees)', fontsize=12)
    axes[0].set_ylabel('Image Radius r (pixels)', fontsize=12)
    axes[0].set_title('Projection Curve: r(θ)', fontsize=14)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 1b: 局部尺度因子 dr/dθ
    dr_fisheye = dr_dtheta(theta, k1, k2, k3, k4)
    dr_perspective = k1 / (np.cos(theta) ** 2)

    axes[1].plot(np.degrees(theta), dr_fisheye, 'b-', linewidth=2,
                 label='Fisheye dr/dθ')
    axes[1].plot(np.degrees(theta), dr_perspective, 'r--', linewidth=2,
                 label='Perspective dr/dθ')
    axes[1].set_xlabel('Incident Angle θ (degrees)', fontsize=12)
    axes[1].set_ylabel('Scale Factor dr/dθ (px/rad)', fontsize=12)
    axes[1].set_title('Local Scale Factor: dr/dθ', fontsize=14)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle('Fisheye vs Perspective Projection Characteristics', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/projection_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 投影曲线对比 → {save_dir}/projection_curves.png")

    # ---- 2. 径向尺度压缩比热力图 ----
    y_grid, x_grid = np.meshgrid(np.arange(img_h), np.arange(img_w), indexing='ij')
    dx = x_grid - cx
    dy = y_grid - cy
    r = np.sqrt(dx**2 + dy**2)

    # 计算每个像素的局部尺度 dθ/dr (物理角度变化 / 像素)
    theta_map = np.zeros_like(r)
    mask = r > 1e-6
    for v in range(img_h):
        for u in range(img_w):
            if mask[v, u]:
                theta_map[v, u] = theta_from_r(r[v, u], k1, k2, k3, k4)

    dr_map = np.ones_like(r)
    dr_map[mask] = dr_dtheta(theta_map[mask], k1, k2, k3, k4)

    # 尺度比 = (dr/dθ 当前值) / (dr/dθ at θ=0), 即边缘相对于中心的"拉伸比"
    dr_center = dr_dtheta(0.0, k1, k2, k3, k4)  # k1 = 339.749
    scale_ratio = dr_map / (dr_center + 1e-10)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.log10(scale_ratio), cmap='RdYlBu_r', origin='upper',
                   extent=[0, img_w, img_h, 0])
    ax.plot(cx, cy, 'k+', markersize=15, markeredgewidth=2)
    ax.set_title('Radial Scale Distortion — log10(dr/dθ / dr/dθ₀)', fontsize=14)
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log10(Scale Ratio)', fontsize=12)

    # 标注三个区域的边界
    r_max_val = np.sqrt(max(cx, img_w - cx)**2 + max(cy, img_h - cy)**2)
    for ratio, color, label in [(0.3, 'white', 'Center'), (0.7, 'white', 'Edge')]:
        circle = plt.Circle((cx, cy), ratio * r_max_val, fill=False,
                            color=color, linewidth=2, linestyle='--')
        ax.add_patch(circle)
        ax.annotate(label, (cx + ratio * r_max_val * 0.7, cy),
                    color=color, fontsize=10, ha='center')

    plt.tight_layout()
    plt.savefig(f'{save_dir}/scale_distortion_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 尺度畸变热力图 → {save_dir}/scale_distortion_heatmap.png")

    # ---- 3. 关键数值统计 ----
    stats = {
        "dr_dtheta_at_center": float(dr_center),
        "dr_dtheta_at_45deg": float(dr_dtheta(np.radians(45), k1, k2, k3, k4)),
        "dr_dtheta_at_60deg": float(dr_dtheta(np.radians(60), k1, k2, k3, k4)),
        "dr_dtheta_at_80deg": float(dr_dtheta(np.radians(80), k1, k2, k3, k4)),
        "scale_ratio_at_45deg": float(dr_dtheta(np.radians(45), k1, k2, k3, k4) / dr_center),
        "scale_ratio_at_60deg": float(dr_dtheta(np.radians(60), k1, k2, k3, k4) / dr_center),
        "scale_ratio_at_80deg": float(dr_dtheta(np.radians(80), k1, k2, k3, k4) / dr_center),
        "r_at_45deg_px": float(r_theta(np.radians(45), k1, k2, k3, k4)),
        "r_at_60deg_px": float(r_theta(np.radians(60), k1, k2, k3, k4)),
        "r_at_80deg_px": float(r_theta(np.radians(80), k1, k2, k3, k4)),
        "image_diagonal_px": float(r_max_val),
    }

    print("\n  === 鱼眼畸变关键数值 ===")
    print(f"  中心尺度 dr/dθ(θ=0°):     {stats['dr_dtheta_at_center']:.1f} px/rad")
    print(f"  45° 尺度 dr/dθ(θ=45°):    {stats['dr_dtheta_at_45deg']:.1f} px/rad  (×{stats['scale_ratio_at_45deg']:.2f})")
    print(f"  60° 尺度 dr/dθ(θ=60°):    {stats['dr_dtheta_at_60deg']:.1f} px/rad  (×{stats['scale_ratio_at_60deg']:.2f})")
    print(f"  80° 尺度 dr/dθ(θ=80°):    {stats['dr_dtheta_at_80deg']:.1f} px/rad  (×{stats['scale_ratio_at_80deg']:.2f})")
    print(f"  45° 像高:                 {stats['r_at_45deg_px']:.0f} px")
    print(f"  60° 像高:                 {stats['r_at_60deg_px']:.0f} px")
    print(f"  80° 像高:                 {stats['r_at_80deg_px']:.0f} px")
    print(f"  图像对角线半长:           {stats['image_diagonal_px']:.0f} px")

    return stats


def demonstrate_edge_effect(image, k1, k2, k3, k4, cx, cy, save_dir):
    """在同一张鱼眼图中标注物体在中心vs边缘的外观差异。

    选取图像中特定区域，展示畸变如何改变物体形状。
    """
    img = image.copy()
    h, w = img.shape[:2]

    # 在中心、过渡、边缘各选一个 ROI 区域框出
    r_max = np.sqrt(max(cx, w - cx)**2 + max(cy, h - cy)**2)
    roi_size = 80

    regions = [
        ("Center\n(low distortion)", cx - roi_size//2, cy - roi_size//2, roi_size, roi_size, (0, 255, 0)),
        ("Transition\n(moderate)", cx + 280, cy + 0, roi_size, roi_size, (0, 255, 255)),
        ("Edge\n(severe distortion)", cx - 460, cy + 280, roi_size, roi_size, (0, 0, 255)),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))

    # 0: 原图 + ROI标注
    img_annotated = img.copy()
    for label, rx, ry, rw, rh, color in regions:
        cv2.rectangle(img_annotated, (int(rx), int(ry)),
                      (int(rx + rw), int(ry + rh)), color, 2)
        cv2.putText(img_annotated, label.split('\n')[0],
                    (int(rx), int(ry) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 画区域划分圆
    for ratio, color in [(0.3, (255, 255, 255)), (0.7, (200, 200, 200))]:
        cv2.circle(img_annotated, (int(cx), int(cy)), int(ratio * r_max), color, 1)

    axes[0].imshow(cv2.cvtColor(img_annotated, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Fisheye Image\nwith ROI Annotations', fontsize=11)
    axes[0].axis('off')

    # 1-3: 各 ROI 放大
    for i, (label, rx, ry, rw, rh, color) in enumerate(regions):
        rx_i, ry_i = int(rx), int(ry)
        if rx_i < 0: rx_i = 0
        if ry_i < 0: ry_i = 0
        if rx_i + rw > w: rx_i = w - rw
        if ry_i + rh > h: ry_i = h - rh

        roi = img[ry_i:ry_i + rh, rx_i:rx_i + rw]
        axes[i + 1].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))

        # 计算该位置尺度比
        r_center = np.sqrt((rx_i + rw/2 - cx)**2 + (ry_i + rh/2 - cy)**2)
        dr_center = dr_dtheta(0.0, k1, k2, k3, k4)
        if r_center > 0:
            theta_c = theta_from_r(r_center, k1, k2, k3, k4)
            dr_local = dr_dtheta(theta_c, k1, k2, k3, k4)
            scale = dr_local / dr_center
        else:
            scale = 1.0

        axes[i + 1].set_title(f'{label}\nScale factor: ×{scale:.2f}', fontsize=10, color='darkred')
        axes[i + 1].axis('off')

    plt.suptitle('Fisheye Distortion: Center vs Edge Object Appearance', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/edge_effect_demo.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  [OK] 边缘效应演示 → {save_dir}/edge_effect_demo.png")


def analyze_flow_distortion_impact(flow, cx, cy, k1, k2, k3, k4, save_dir):
    """分析鱼眼畸变对光流场的影响。

    在中心与边缘区域分别统计光流幅度, 展示畸变导致的光流放大/缩小效应。
    """
    h, w = flow.shape[:2]
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 区域划分
    r_max = np.sqrt(max(cx, w - cx)**2 + max(cy, h - cy)**2)
    y_grid, x_grid = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    r = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)

    regions = {
        "Center (r<0.3·rmax)": r < 0.3 * r_max,
        "Transition (0.3-0.7)": (r >= 0.3 * r_max) & (r < 0.7 * r_max),
        "Edge (r>=0.7·rmax)": r >= 0.7 * r_max,
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = ['steelblue', 'darkorange', 'crimson']
    for i, (name, region_mask) in enumerate(regions.items()):
        mag_region = mag[region_mask]
        if len(mag_region) > 0:
            axes[i].hist(mag_region, bins=50, color=colors[i], edgecolor='black', alpha=0.7)
            axes[i].axvline(np.median(mag_region), color='red', linestyle='--',
                            label=f'Median={np.median(mag_region):.2f}px')
            axes[i].set_title(f'{name}\n{region_mask.sum()} pixels', fontsize=11)
            axes[i].set_xlabel('Flow Magnitude (px)')
            axes[i].set_ylabel('Frequency')
            axes[i].legend()

    plt.suptitle('Optical Flow Magnitude Distribution by Region (Fisheye Domain)', fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{save_dir}/flow_region_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # 打印统计
    print("\n  === 光流分区域统计 ===")
    for name, rm in regions.items():
        mag_r = mag[rm]
        if len(mag_r) > 0:
            print(f"  {name}: mean={mag_r.mean():.2f}px, median={np.median(mag_r):.2f}px, "
                  f"std={mag_r.std():.2f}px, P95={np.percentile(mag_r, 95):.2f}px")
