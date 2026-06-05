"""Evaluate seed_expand V8 (multi-scale) against baseline methods.

Generates:
  1. Per-frame TP/FP/FN error maps
  2. Summary comparison table (text)
  3. Bar chart comparing all methods (saved as PNG)
  4. Aggregated metrics report
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.motion_detection.frame_difference import (
    detect_motion_diff,
    detect_motion_flow,
    detect_motion_hybrid,
    detect_motion_adaptive,
    detect_motion_seed_expand,
    detect_motion_seed_expand_v2,
    detect_motion_seed_expand_v8,
)
from core.evaluation.metrics import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "homework2"
OUTPUT_DIR = PROJECT_ROOT / "output"
EVAL_OUT = OUTPUT_DIR / "evaluation" / "v8_multiscale"
EVAL_OUT.mkdir(parents=True, exist_ok=True)

FRAMES = ["00000", "00001", "00002", "00003", "00004", "00005",
          "00006", "00007", "00013", "00016", "00039", "00041", "00053"]

# ── method definitions ───────────────────────────────────────────
METHODS = {
    "diff":       lambda pg, cg: detect_motion_diff(pg, cg, diff_thresh=40, morph_ksize=5),
    "flow":       lambda pg, cg: detect_motion_flow(pg, cg, mag_thresh=3.0, morph_ksize=9),
    "hybrid":     lambda pg, cg: detect_motion_hybrid(pg, cg, diff_thresh=50, mag_thresh=3.0, morph_ksize=7),
    "adaptive":   lambda pg, cg: detect_motion_adaptive(pg, cg, morph_ksize=7),
    "seed_expand": lambda pg, cg: detect_motion_seed_expand(pg, cg),
    "seed_exp_v2": lambda pg, cg: detect_motion_seed_expand_v2(pg, cg),
    "seed_exp_v8": lambda pg, cg: detect_motion_seed_expand_v8(pg, cg),
}

METHOD_COLORS = {
    "diff":        "#3498db",
    "flow":        "#e74c3c",
    "hybrid":      "#95a5a6",
    "adaptive":    "#f39c12",
    "seed_expand": "#2ecc71",
    "seed_exp_v2": "#9b59b6",
    "seed_exp_v8": "#e67e22",
}


def load_pair(fid):
    c = cv2.imread(str(DATA / "rgb_images" / f"{fid}_FV.png"))
    p = cv2.imread(str(DATA / "previous_images" / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(DATA / "motion_annotation" / "GroudTruth" / f"{fid}_FV.png"),
                   cv2.IMREAD_GRAYSCALE)
    if c is None or p is None or g is None:
        raise FileNotFoundError(f"Missing data for {fid}")
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    return pg, cg, c, (g > 0)


# ══════════════════════════════════════════════════════════════════
# 1. Run all methods on all frames, collect metrics
# ══════════════════════════════════════════════════════════════════
print("=" * 70)
print("  Multi-Scale Seed Expansion (V8) — Evaluation")
print("=" * 70)

all_results = {m: {} for m in METHODS}  # method -> frame_id -> metrics_dict
all_masks = {m: {} for m in METHODS}     # method -> frame_id -> mask

for fid in FRAMES:
    print(f"\nProcessing {fid}...")
    try:
        pg, cg, orig, gt_bin = load_pair(fid)
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        continue

    for mname, func in METHODS.items():
        mask = func(pg, cg)
        metrics = compute_metrics(mask, gt_bin.astype(np.uint8) * 255)
        all_results[mname][fid] = metrics
        all_masks[mname][fid] = mask
        print(f"  {mname:>14s}  IoU={metrics['iou']:.4f}  Prec={metrics['precision']:.4f}  "
              f"Rec={metrics['recall']:.4f}  F1={metrics['f1']:.4f}  "
              f"TP={metrics['tp']:>7,}  FP={metrics['fp']:>7,}  FN={metrics['fn']:>5,}")


# ══════════════════════════════════════════════════════════════════
# 2. Text summary table
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  PER-FRAME F1 COMPARISON")
print("=" * 100)

# Header
header = f"{'Method':<16}"
for fid in FRAMES:
    header += f" {fid:>8}"
header += f" {'avg':>8}  {'min':>8}  {'max':>8}"
print(header)
print("-" * len(header))

for mname in METHODS:
    row = f"{mname:<16}"
    f1s = []
    for fid in FRAMES:
        if fid in all_results[mname]:
            f1 = all_results[mname][fid]["f1"]
            f1s.append(f1)
            row += f" {f1:>8.4f}"
        else:
            row += f" {'N/A':>8}"
    if f1s:
        row += f" {np.mean(f1s):>8.4f}  {np.min(f1s):>8.4f}  {np.max(f1s):>8.4f}"
    print(row)

# Averages across all metrics
print("\n" + "=" * 70)
print("  AGGREGATED METRICS (macro-average over 13 frames)")
print("=" * 70)
print(f"{'Method':<16} {'IoU':>8} {'Precision':>8} {'Recall':>8} {'F1':>8}  {'TP_sum':>9} {'FP_sum':>9} {'FN_sum':>7}")
print("-" * 78)

for mname in METHODS:
    metrics_list = [all_results[mname][fid] for fid in FRAMES if fid in all_results[mname]]
    if not metrics_list:
        continue
    avg = {k: np.mean([m[k] for m in metrics_list]) for k in ["iou", "precision", "recall", "f1"]}
    tp_sum = int(sum(m["tp"] for m in metrics_list))
    fp_sum = int(sum(m["fp"] for m in metrics_list))
    fn_sum = int(sum(m["fn"] for m in metrics_list))
    print(f"{mname:<16} {avg['iou']:>8.4f} {avg['precision']:>8.4f} "
          f"{avg['recall']:>8.4f} {avg['f1']:>8.4f}  "
          f"{tp_sum:>9,} {fp_sum:>9,} {fn_sum:>7,}")


# ══════════════════════════════════════════════════════════════════
# 3. Per-frame error-map visualization
# ══════════════════════════════════════════════════════════════════
print("\n\nGenerating per-frame comparison images...")

def color_error(pred, gt_bin, bg_img=None):
    """Color-code prediction: green=TP, red=FP, blue=FN"""
    pred_bin = (pred > 0)
    gt_b = (gt_bin > 0)
    tp = pred_bin & gt_b
    fp = pred_bin & ~gt_b
    fn = ~pred_bin & gt_b

    if bg_img is not None:
        gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
        out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR).astype(np.float32)
        out *= 0.35  # dim background
    else:
        out = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.float32)

    out[tp] = [0, 255, 0]     # green
    out[fp] = [0, 0, 255]     # red
    out[fn] = [255, 0, 0]     # blue
    return np.clip(out, 0, 255).astype(np.uint8)


# Focus on: diff, seed_expand (baseline), seed_exp_v8 (our method)
FOCUS_METHODS = ["diff", "seed_expand", "seed_exp_v8"]
PER_FRAME_DIR = EVAL_OUT / "per_frame"
PER_FRAME_DIR.mkdir(parents=True, exist_ok=True)

for fid in FRAMES:
    try:
        pg, cg, orig, gt_bin = load_pair(fid)
    except FileNotFoundError:
        continue

    h_img = 280
    def rs(img, hh=h_img):
        r = hh / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * r), hh))

    # Row 1: original | GT | GT overlay
    orig_rs = rs(orig)
    gt_vis = rs(cv2.cvtColor((gt_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    # GT overlay: semi-transparent green on original
    gt_overlay = orig_rs.copy()
    gt_rs = cv2.resize((gt_bin * 255).astype(np.uint8),
                        (orig_rs.shape[1], orig_rs.shape[0]))
    gt_overlay[gt_rs > 0] = (gt_overlay[gt_rs > 0] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)

    row1 = np.hstack([orig_rs, gt_vis, gt_overlay])

    def label(img, text, y=24):
        cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    label(row1, "Original")
    label(row1[:, orig_rs.shape[1]:], "Ground Truth")
    label(row1[:, 2*orig_rs.shape[1]:], "GT Overlay")

    # Row 2: error maps for focus methods
    error_maps = []
    for mname in FOCUS_METHODS:
        mask = all_masks[mname].get(fid)
        if mask is None:
            continue
        err = color_error(mask, gt_bin, orig)
        err_rs = rs(err)
        # Add metrics text
        m = all_results[mname][fid]
        txt = f"{mname}"
        txt2 = f"IoU={m['iou']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}"
        cv2.putText(err_rs, txt, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(err_rs, txt2, (6, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
        error_maps.append(err_rs)

    while len(error_maps) < 3:
        blank = np.zeros_like(orig_rs)
        error_maps.append(blank)

    row2 = np.hstack(error_maps)

    # Row 3: legend
    legend_h = 50
    legend = np.zeros((legend_h, row1.shape[1], 3), dtype=np.uint8)
    legend[:] = [35, 35, 35]
    items = [
        ("Green = TP (correct)", (0, 255, 0)),
        ("Red   = FP (false alarm)", (0, 0, 255)),
        ("Blue  = FN (missed)", (255, 0, 0)),
    ]
    seg_w = legend.shape[1] // len(items)
    for i, (txt, col) in enumerate(items):
        x0 = i * seg_w + 10
        cv2.putText(legend, txt, (x0, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    viz = np.vstack([row1, row2, legend])
    save_path = PER_FRAME_DIR / f"{fid}_comparison.png"
    cv2.imwrite(str(save_path), viz)
    print(f"  Saved: {save_path}")

print("  All per-frame images generated.")


# ══════════════════════════════════════════════════════════════════
# 4. Summary bar chart (using only OpenCV — no matplotlib dependency)
# ══════════════════════════════════════════════════════════════════
print("\nGenerating summary bar chart...")

def draw_bar_chart(methods, values, title, ylabel, filename, color_map=None):
    """Draw a bar chart using pure OpenCV."""
    n = len(methods)
    w_bar = 60
    gap = 40
    margin_left, margin_right = 120, 60
    margin_top, margin_bottom = 60, 120
    chart_w = margin_left + n * (w_bar + gap) - gap + margin_right
    chart_h = 500

    canvas = np.ones((chart_h, chart_w, 3), dtype=np.uint8) * 240

    max_val = max(values) * 1.15 if max(values) > 0 else 1.0
    plot_h = chart_h - margin_top - margin_bottom

    # Bars
    for i, (m, v) in enumerate(zip(methods, values)):
        x = margin_left + i * (w_bar + gap)
        bar_h = int(v / max_val * plot_h)
        y = margin_top + plot_h - bar_h

        color = [200, 200, 200]
        if color_map and m in color_map:
            hex_c = color_map[m].lstrip("#")
            color = [int(hex_c[4:6], 16), int(hex_c[2:4], 16), int(hex_c[0:2], 16)]

        cv2.rectangle(canvas, (x, y), (x + w_bar, margin_top + plot_h), color, -1)
        cv2.rectangle(canvas, (x, y), (x + w_bar, margin_top + plot_h), (60, 60, 60), 2)

        # Value label on top of bar
        val_txt = f"{v:.4f}"
        (tw, th), _ = cv2.getTextSize(val_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(canvas, val_txt, (x + (w_bar - tw) // 2, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1)

        # Method label below
        (tw2, th2), _ = cv2.getTextSize(m, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(canvas, m, (x + (w_bar - tw2) // 2, margin_top + plot_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 50), 1)

    # Axes
    cv2.line(canvas, (margin_left - 5, margin_top),
             (margin_left - 5, margin_top + plot_h), (50, 50, 50), 2)
    cv2.line(canvas, (margin_left - 5, margin_top + plot_h),
             (chart_w - margin_right + 10, margin_top + plot_h), (50, 50, 50), 2)

    # Y-axis ticks
    for i in range(6):
        frac = i / 5
        val = max_val * frac
        y = margin_top + plot_h - int(frac * plot_h)
        t = f"{val:.2f}"
        (tw3, th3), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(canvas, t, (margin_left - 12 - tw3, y + th3 // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)
        cv2.line(canvas, (margin_left - 5, y), (chart_w - margin_right + 10, y),
                 (210, 210, 210), 1)

    # Title
    (twt, tht), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(canvas, title, ((chart_w - twt) // 2, 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (30, 30, 30), 2)

    cv2.imwrite(str(EVAL_OUT / filename), canvas)
    print(f"  Saved: {EVAL_OUT / filename}")


# Build aggregated data for charts
method_names = list(METHODS.keys())
avg_iou = [np.mean([all_results[m][fid]["iou"] for fid in FRAMES if fid in all_results[m]]) for m in method_names]
avg_prec = [np.mean([all_results[m][fid]["precision"] for fid in FRAMES if fid in all_results[m]]) for m in method_names]
avg_rec = [np.mean([all_results[m][fid]["recall"] for fid in FRAMES if fid in all_results[m]]) for m in method_names]
avg_f1 = [np.mean([all_results[m][fid]["f1"] for fid in FRAMES if fid in all_results[m]]) for m in method_names]

draw_bar_chart(method_names, avg_f1, "Average F1 Score by Method", "F1", "summary_f1.png", METHOD_COLORS)
draw_bar_chart(method_names, avg_prec, "Average Precision by Method", "Precision", "summary_precision.png", METHOD_COLORS)
draw_bar_chart(method_names, avg_rec, "Average Recall by Method", "Recall", "summary_recall.png", METHOD_COLORS)
draw_bar_chart(method_names, avg_iou, "Average IoU by Method", "IoU", "summary_iou.png", METHOD_COLORS)


# ══════════════════════════════════════════════════════════════════
# 5. Per-frame F1 trend line chart
# ══════════════════════════════════════════════════════════════════
print("\nGenerating per-frame F1 trend chart...")

def draw_trend_chart(methods_to_plot, title, filename, color_map=None):
    """Draw a line chart showing per-frame F1 values for selected methods."""
    n_frames = len(FRAMES)
    margin_left, margin_right = 100, 50
    margin_top, margin_bottom = 50, 120
    chart_w = margin_left + n_frames * 75 + margin_right
    chart_h = 480

    canvas = np.ones((chart_h, chart_w, 3), dtype=np.uint8) * 245

    # Find global max
    all_vals = []
    for mname in methods_to_plot:
        for fid in FRAMES:
            if fid in all_results[mname]:
                all_vals.append(all_results[mname][fid]["f1"])
    max_val = max(all_vals) * 1.12 if all_vals else 1.0
    plot_h = chart_h - margin_top - margin_bottom
    plot_w = n_frames * 75

    # Grid & axes
    for i in range(6):
        frac = i / 5
        y = margin_top + plot_h - int(frac * plot_h)
        cv2.line(canvas, (margin_left, y), (margin_left + plot_w, y), (210, 210, 210), 1)
        t = f"{max_val * frac:.2f}"
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(canvas, t, (margin_left - 10 - tw, y + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

    cv2.line(canvas, (margin_left, margin_top), (margin_left, margin_top + plot_h), (50, 50, 50), 2)
    cv2.line(canvas, (margin_left, margin_top + plot_h), (margin_left + plot_w, margin_top + plot_h), (50, 50, 50), 2)

    # Frame labels
    for i, fid in enumerate(FRAMES):
        x = margin_left + i * 75 + 37
        (tw, th), _ = cv2.getTextSize(fid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(canvas, fid, (x - tw // 2, margin_top + plot_h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (60, 60, 60), 1)

    # Lines
    markers = ["o", "s", "D", "^", "v", "<", ">", "p"]
    for j, mname in enumerate(methods_to_plot):
        color = [100, 100, 100]
        if color_map and mname in color_map:
            hex_c = color_map[mname].lstrip("#")
            color = [int(hex_c[4:6], 16), int(hex_c[2:4], 16), int(hex_c[0:2], 16)]

        pts = []
        for i, fid in enumerate(FRAMES):
            if fid in all_results[mname]:
                val = all_results[mname][fid]["f1"]
                x = margin_left + i * 75 + 37
                y = margin_top + plot_h - int(val / max_val * plot_h)
                pts.append((x, y))

        # Draw lines
        for i in range(len(pts) - 1):
            cv2.line(canvas, pts[i], pts[i + 1], color, 2)

        # Draw markers
        for x, y in pts:
            cv2.circle(canvas, (x, y), 5, color, -1)
            cv2.circle(canvas, (x, y), 5, (50, 50, 50), 1)

    # Legend
    leg_x = margin_left + plot_w - 300
    leg_y = margin_top + 10
    for j, mname in enumerate(methods_to_plot):
        color = [100, 100, 100]
        if color_map and mname in color_map:
            hex_c = color_map[mname].lstrip("#")
            color = [int(hex_c[4:6], 16), int(hex_c[2:4], 16), int(hex_c[0:2], 16)]
        cv2.rectangle(canvas, (leg_x, leg_y + j * 24), (leg_x + 15, leg_y + 15 + j * 24), color, -1)
        cv2.putText(canvas, mname, (leg_x + 22, leg_y + 13 + j * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (40, 40, 40), 1)

    # Title
    (twt, tht), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(canvas, title, ((chart_w - twt) // 2, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 30, 30), 2)

    cv2.imwrite(str(EVAL_OUT / filename), canvas)
    print(f"  Saved: {EVAL_OUT / filename}")


# Plot: seed_expand (baseline) vs seed_exp_v2 vs seed_exp_v8
draw_trend_chart(["seed_expand", "seed_exp_v2", "seed_exp_v8"],
                 "Per-Frame F1: Seed-Expand Variants Comparison",
                 "trend_seed_expand_variants.png", METHOD_COLORS)

# Plot: all methods
draw_trend_chart(["diff", "flow", "seed_expand", "seed_exp_v8"],
                 "Per-Frame F1: Key Methods Comparison",
                 "trend_key_methods.png", METHOD_COLORS)


# ══════════════════════════════════════════════════════════════════
# 6. V8 vs baseline delta analysis (text)
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("  V8 vs BASELINE (seed_expand) — PER-FRAME DELTA")
print("=" * 80)
print(f"{'Frame':<8} {'ΔIoU':>8} {'ΔPrec':>8} {'ΔRecall':>8} {'ΔF1':>8}  "
      f"{'ΔTP':>8} {'ΔFP':>8} {'ΔFN':>7}")
print("-" * 72)

delta_f1s, delta_precs, delta_recs = [], [], []
V8_KEY = "seed_exp_v8"
BL_KEY = "seed_expand"

for fid in FRAMES:
    if fid not in all_results[V8_KEY] or fid not in all_results[BL_KEY]:
        continue
    v8 = all_results[V8_KEY][fid]
    bl = all_results[BL_KEY][fid]
    d_iou = v8["iou"] - bl["iou"]
    d_prec = v8["precision"] - bl["precision"]
    d_rec = v8["recall"] - bl["recall"]
    d_f1 = v8["f1"] - bl["f1"]
    d_tp = v8["tp"] - bl["tp"]
    d_fp = v8["fp"] - bl["fp"]
    d_fn = v8["fn"] - bl["fn"]
    delta_f1s.append(d_f1)
    delta_precs.append(d_prec)
    delta_recs.append(d_rec)
    print(f"{fid:<8} {d_iou:>+8.4f} {d_prec:>+8.4f} {d_rec:>+8.4f} {d_f1:>+8.4f}  "
          f"{d_tp:>+8,} {d_fp:>+8,} {d_fn:>+7,}")

print("-" * 72)
print(f"{'MEAN':<8} {np.mean([all_results[V8_KEY][fid]['iou'] - all_results[BL_KEY][fid]['iou'] for fid in FRAMES if fid in all_results[V8_KEY]]):>+8.4f} "
      f"{np.mean(delta_precs):>+8.4f} {np.mean(delta_recs):>+8.4f} {np.mean(delta_f1s):>+8.4f}")


# ══════════════════════════════════════════════════════════════════
# 7. Save combined summary image (all methods, all frames, F1 heatmap-style)
# ══════════════════════════════════════════════════════════════════
print("\nGenerating F1 heatmap summary...")

cell_w, cell_h = 80, 32
header_h = 36
left_w = 110
n_methods = len(methods_to_plot := list(METHODS.keys()))
n_frames = len(FRAMES)

heat_w = left_w + n_frames * cell_w + 20
heat_h = header_h + n_methods * cell_h + 20

heatmap = np.ones((heat_h, heat_w, 3), dtype=np.uint8) * 248

# Corner
cv2.putText(heatmap, "Method \\ Frame", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)

# Column headers (frame IDs)
for j, fid in enumerate(FRAMES):
    x = left_w + j * cell_w
    cv2.putText(heatmap, fid, (x + 8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (50, 50, 50), 1)

# Row headers + cells
for i, mname in enumerate(methods_to_plot):
    y = header_h + i * cell_h
    cv2.putText(heatmap, mname, (6, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (50, 50, 50), 1)

    for j, fid in enumerate(FRAMES):
        x = left_w + j * cell_w
        if fid in all_results[mname]:
            f1 = all_results[mname][fid]["f1"]
            # Color: low=red, mid=yellow, high=green
            if f1 < 0.02:
                r, g, b = 255, 200, 200
            elif f1 < 0.1:
                r, g, b = 255, 230, 180
            elif f1 < 0.2:
                r, g, b = 255, 255, 170
            elif f1 < 0.35:
                r, g, b = 200, 255, 160
            else:
                r, g, b = 140, 255, 140
            cv2.rectangle(heatmap, (x + 1, y + 1), (x + cell_w - 2, y + cell_h - 2), (b, g, r), -1)
            txt = f"{f1:.3f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
            cv2.putText(heatmap, txt, (x + (cell_w - tw) // 2, y + (cell_h + th) // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (40, 40, 40), 1)

cv2.imwrite(str(EVAL_OUT / "f1_heatmap.png"), heatmap)
print(f"  Saved: {EVAL_OUT / 'f1_heatmap.png'}")


# ══════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  Evaluation complete!")
print(f"  All outputs saved to: {EVAL_OUT}")
print("=" * 70)
print("\nOutput files:")
for f in sorted(EVAL_OUT.glob("*.png")):
    print(f"  {f.name}")
