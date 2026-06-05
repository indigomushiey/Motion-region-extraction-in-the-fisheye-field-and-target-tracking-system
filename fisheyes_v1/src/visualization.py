"""
可视化模块 — 结果绘图、对比图生成、光流可视化等。
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# 中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def flow_to_color(flow, max_mag=None):
    """将光流场转换为 HSV 彩色图。"""
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = 255
    if max_mag is None:
        max_mag = np.percentile(mag, 95)
    hsv[..., 2] = np.clip(mag / (max_mag + 1e-8) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def draw_bboxes(image, bboxes, color=(0, 255, 0), thickness=2, labels=None):
    """在图像上绘制边界框和标签。"""
    img = image.copy()
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        if labels:
            cv2.putText(img, str(labels[i]), (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img


def draw_tracks(image, tracks, colors=None):
    """绘制跟踪轨迹。

    Args:
        image: 背景图像
        tracks: [(track_id, (cx,cy,w,h)), ...]
        colors: 颜色列表
    """
    img = image.copy()
    if colors is None:
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
                   (255, 0, 255), (0, 255, 255), (128, 255, 0), (0, 128, 255)]

    for i, (tid, state) in enumerate(tracks):
        c = colors[tid % len(colors)]
        cx, cy, w, h = state
        x, y = int(cx - w / 2), int(cy - h / 2)
        cv2.rectangle(img, (x, y), (x + w, y + h), c, 2)
        cv2.putText(img, f"ID:{tid}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)
    return img


def draw_trails(image, trails_dict, colors=None):
    """绘制跟踪轨迹历史。

    Args:
        image: 背景图
        trails_dict: {track_id: [(cx,cy), ...]}
    """
    img = image.copy()
    if colors is None:
        colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 255, 0),
                   (255, 0, 255), (0, 255, 255)]

    for tid, trail in trails_dict.items():
        c = colors[tid % len(colors)]
        for j in range(1, len(trail)):
            pt1 = (int(trail[j - 1][0]), int(trail[j - 1][1]))
            pt2 = (int(trail[j][0]), int(trail[j][1]))
            cv2.line(img, pt1, pt2, c, 2)
        # 画当前点
        if trail:
            cx, cy = int(trail[-1][0]), int(trail[-1][1])
            cv2.circle(img, (cx, cy), 4, c, -1)
    return img


def draw_sparse_flow(image, prev_pts, next_pts, color=(0, 255, 0), thickness=1):
    """绘制稀疏光流矢量。"""
    img = image.copy()
    for p, n in zip(prev_pts, next_pts):
        p1 = (int(p[0]), int(p[1]))
        p2 = (int(n[0]), int(n[1]))
        cv2.arrowedLine(img, p1, p2, color, thickness, tipLength=0.3)
        cv2.circle(img, p2, 2, color, -1)
    return img


def create_comparison_figure(curr_img, prev_img, mask_a, mask_b, gt_mask,
                             flow_color=None, route_a_name="Route A (Fisheye)",
                             route_b_name="Route B (Rectified)", save_path=None):
    """创建路线 A vs B 对比图。

    2行布局:
    Row 1: 当前帧 | 前一帧 | 光流可视化
    Row 2: GT | Route A 结果 | Route B 结果
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Row 1
    axes[0, 0].imshow(cv2.cvtColor(curr_img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Current Frame", fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(prev_img, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Previous Frame", fontsize=12)
    axes[0, 1].axis('off')

    if flow_color is not None:
        axes[0, 2].imshow(cv2.cvtColor(flow_color, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("Optical Flow", fontsize=12)
    else:
        axes[0, 2].text(0.5, 0.5, 'N/A', ha='center', va='center',
                        transform=axes[0, 2].transAxes, fontsize=20)
        axes[0, 2].set_title("Optical Flow")
    axes[0, 2].axis('off')

    # Row 2
    axes[1, 0].imshow(gt_mask, cmap='gray')
    axes[1, 0].set_title("Ground Truth", fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(mask_a, cmap='gray')
    axes[1, 1].set_title(route_a_name, fontsize=12)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(mask_b, cmap='gray')
    axes[1, 2].set_title(route_b_name, fontsize=12)
    axes[1, 2].axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_region_analysis_figure(region_metrics_a, region_metrics_b,
                                  region_names=None, save_path=None):
    """创建分区域指标对比柱状图。"""
    if region_names is None:
        region_names = ["Center", "Transition", "Edge"]

    metrics = ["IoU", "Precision", "Recall", "F1"]
    x = np.arange(len(region_names))
    width = 0.35

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        ax = axes[i]
        vals_a = [region_metrics_a[r][metric] for r in ["center", "transition", "edge"]]
        vals_b = [region_metrics_b[r][metric] for r in ["center", "transition", "edge"]]

        bars1 = ax.bar(x - width / 2, vals_a, width, label='Route A (Fisheye)',
                        color='steelblue', edgecolor='black')
        bars2 = ax.bar(x + width / 2, vals_b, width, label='Route B (Rectified)',
                        color='darkorange', edgecolor='black')

        # 数值标注
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8)

        ax.set_title(metric, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(region_names)
        ax.set_ylim(0, max(max(vals_a), max(vals_b)) * 1.3 + 0.1)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle("Region-wise Performance Comparison (A vs B)", fontsize=16)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_distortion_analysis_figure(original, undistorted_p, undistorted_c,
                                       undistorted_e, save_path=None):
    """创建多投影校正对比图。

    Args:
        original: 原始鱼眼图像
        undistorted_p: 透视投影校正
        undistorted_c: 柱面投影校正
        undistorted_e: 等距矩形投影校正
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Fisheye", fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(cv2.cvtColor(undistorted_p, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("Perspective Projection", fontsize=12)
    axes[0, 1].axis('off')

    axes[1, 0].imshow(cv2.cvtColor(undistorted_c, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Cylindrical Projection", fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(cv2.cvtColor(undistorted_e, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("Equirectangular Projection", fontsize=12)
    axes[1, 1].axis('off')

    plt.suptitle("Distortion Correction — Multi-Projection Comparison", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_metrics_summary_figure(all_results, route_name, save_path=None):
    """创建整体评估指标汇总图 (直方图 + 箱线图)。"""
    ids = [r["id"] for r in all_results]
    ious = [r["metrics"]["IoU"] for r in all_results]
    f1s = [r["metrics"]["F1"] for r in all_results]
    precisions = [r["metrics"]["Precision"] for r in all_results]
    recalls = [r["metrics"]["Recall"] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # IoU 分布
    axes[0, 0].hist(ious, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(ious), color='red', linestyle='--', label=f'Mean={np.mean(ious):.4f}')
    axes[0, 0].set_title(f"IoU Distribution — {route_name}", fontsize=12)
    axes[0, 0].set_xlabel("IoU")
    axes[0, 0].legend()

    # F1 分布
    axes[0, 1].hist(f1s, bins=30, color='darkorange', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(np.mean(f1s), color='red', linestyle='--', label=f'Mean={np.mean(f1s):.4f}')
    axes[0, 1].set_title(f"F1 Distribution — {route_name}", fontsize=12)
    axes[0, 1].set_xlabel("F1")
    axes[0, 1].legend()

    # Top-K 和 Bottom-K
    sorted_idx = np.argsort(ious)
    k = min(10, len(ious))
    top_k = [ids[i] for i in sorted_idx[-k:][::-1]]
    bottom_k = [ids[i] for i in sorted_idx[:k]]
    top_iou = [ious[i] for i in sorted_idx[-k:][::-1]]
    bottom_iou = [ious[i] for i in sorted_idx[:k]]

    axes[1, 0].bar(range(k), top_iou, color='green', alpha=0.7)
    axes[1, 0].set_title(f"Top-{k} Best IoU", fontsize=12)
    axes[1, 0].set_xticks(range(k))
    axes[1, 0].set_xticklabels([f"{tid}" for tid in top_k], rotation=45, fontsize=7)
    axes[1, 0].set_ylabel("IoU")

    axes[1, 1].bar(range(k), bottom_iou, color='red', alpha=0.7)
    axes[1, 1].set_title(f"Top-{k} Worst IoU", fontsize=12)
    axes[1, 1].set_xticks(range(k))
    axes[1, 1].set_xticklabels([f"{tid}" for tid in bottom_k], rotation=45, fontsize=7)
    axes[1, 1].set_ylabel("IoU")

    plt.suptitle(f"Evaluation Summary — {route_name}", fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_method_comparison_figure(results_dict, metric="F1", save_path=None):
    """创建方法对比图 (帧差 vs 光流 vs 背景建模)。"""
    methods = list(results_dict.keys())
    values = [np.mean([r["metrics"][metric] for r in results_dict[m]]) for m in methods]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(methods, values, color=['steelblue', 'darkorange', 'green'],
                  edgecolor='black')

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2., v + 0.01,
                f'{v:.4f}', ha='center', va='bottom', fontsize=12)

    ax.set_title(f"Method Comparison — {metric}", fontsize=14)
    ax.set_ylabel(metric)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def create_region_overlay(image, mask, gt, cx, cy, ratios=None, r_max=None, save_path=None):
    """创建分区域叠加可视化 — 在图像上标注中心/过渡/边缘区域边界。"""
    import numpy as np, cv2, os
    import matplotlib.pyplot as plt

    if ratios is None:
        ratios = [0.3, 0.7]
    h, w = image.shape[:2]
    if r_max is None:
        r_max = np.sqrt(max(cx, w - cx)**2 + max(cy, h - cy)**2)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    base = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    mask_disp = (mask * 255).astype(np.uint8) if mask.max() <= 1 else mask
    gt_disp = (gt * 255).astype(np.uint8) if gt.max() <= 1 else gt

    color_pred = np.zeros_like(base)
    color_pred[:, :, 0] = mask_disp
    blended_pred = cv2.addWeighted(base, 0.7, color_pred, 0.3, 0)
    for ratio, color in [(ratios[0], (255, 255, 255)), (ratios[1], (200, 200, 200))]:
        cv2.circle(blended_pred, (int(cx), int(cy)), int(ratio * r_max), color, 1)
    axes[0].imshow(blended_pred)
    axes[0].set_title('Predicted Mask + Region Boundaries', fontsize=11)
    axes[0].axis('off')

    color_gt = np.zeros_like(base)
    color_gt[:, :, 1] = gt_disp
    blended_gt = cv2.addWeighted(base, 0.7, color_gt, 0.3, 0)
    for ratio, color in [(ratios[0], (255, 255, 255)), (ratios[1], (200, 200, 200))]:
        cv2.circle(blended_gt, (int(cx), int(cy)), int(ratio * r_max), color, 1)
    axes[1].imshow(blended_gt)
    axes[1].set_title('Ground Truth + Region Boundaries', fontsize=11)
    axes[1].axis('off')

    mask_bin = (mask_disp > 127).astype(np.uint8)
    gt_bin = (gt_disp > 127).astype(np.uint8)
    diff = np.zeros((h, w, 3), dtype=np.uint8)
    diff[mask_bin == 1] = [255, 0, 0]
    diff[gt_bin == 1] = [0, 255, 0]
    diff[(mask_bin == 1) & (gt_bin == 1)] = [255, 255, 255]
    for ratio, color in [(ratios[0], (128, 128, 128)), (ratios[1], (128, 128, 128))]:
        cv2.circle(diff, (int(cx), int(cy)), int(ratio * r_max), color, 1)
    axes[2].imshow(diff)
    axes[2].set_title('Difference: Red=FP  Green=FN  White=TP', fontsize=11)
    axes[2].axis('off')

    plt.suptitle('Region-based Motion Analysis', fontsize=14)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
