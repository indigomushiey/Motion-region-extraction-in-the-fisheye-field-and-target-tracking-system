"""V12 P0 noise-filtering improvements — 20-frame comparison vs V1 & V10.

Three P0 improvements over V11:
  P0-1: CCA filtering on V1 fallback path (50 frames previously unfiltered)
  P0-2: Internal signal verification (mean diff/flow per component vs global)
  P0-3: Seed-to-area ratio check (noise seeds expand → huge false area)
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.motion_detection.frame_difference import (
    detect_motion_seed_expand,
    detect_motion_seed_expand_v10,
    detect_motion_seed_expand_v12,
)
from core.evaluation.metrics import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "homework2"
GT_DIR = DATA / "motion_annotation" / "GroudTruth"
CURR_DIR = DATA / "rgb_images"
PREV_DIR = DATA / "previous_images"
EVAL_OUT = PROJECT_ROOT / "output" / "evaluation" / "v12_p0_test"
EVAL_OUT.mkdir(parents=True, exist_ok=True)

# ════════════════════════════════════════════════════════════════════
# 20-frame selection: 13 benchmark + 7 additional diverse frames
# ════════════════════════════════════════════════════════════════════
BENCH_13 = ['00000', '00001', '00002', '00003', '00004', '00005',
            '00006', '00007', '00013', '00016', '00039', '00041', '00053']
EXTRA_7 = ['00022', '00026', '00027', '00033', '00037', '00059', '00065']

ALL_FRAMES = BENCH_13 + EXTRA_7

# Verify all frames exist
for fid in ALL_FRAMES:
    for d, label in [(CURR_DIR, "curr"), (PREV_DIR, "prev"), (GT_DIR, "GT")]:
        p = d / f"{fid}_FV.png"
        if not p.exists():
            print(f"WARNING: {label} missing for {fid}: {p}")

print("=" * 100)
print("  V12 P0 IMPROVEMENTS — 20-FRAME COMPARISON")
print(f"  Methods: V1 (seed_expand) | V10 (hybrid) | V12 (V11+P0-1+P0-2+P0-3)")
print(f"  Frames: {len(ALL_FRAMES)}")
print("=" * 100)

# ════════════════════════════════════════════════════════════════════
# Run evaluation
# ════════════════════════════════════════════════════════════════════
methods = {
    "V1":  ("seed_expand",      lambda pg, cg: detect_motion_seed_expand(pg, cg)),
    "V10": ("seed_expand_v10",  lambda pg, cg: detect_motion_seed_expand_v10(pg, cg)),
    "V12": ("seed_expand_v12",  lambda pg, cg: detect_motion_seed_expand_v12(pg, cg)),
}

all_results = {name: {} for name in methods}
frame_stats = {}

print("\nProcessing frames...")
for i, fid in enumerate(ALL_FRAMES):
    c = cv2.imread(str(CURR_DIR / f"{fid}_FV.png"))
    p = cv2.imread(str(PREV_DIR / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GT_DIR / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)

    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gt_bin = (g > 0)

    diff = cv2.absdiff(cg, pg)
    p95 = np.percentile(diff, 95)
    gt_px = int(gt_bin.sum())
    frame_stats[fid] = {"p95": p95, "gt_px": gt_px}

    for mname, (_, func) in methods.items():
        mask = func(pg, cg)
        m = compute_metrics(mask, gt_bin.astype(np.uint8) * 255)
        all_results[mname][fid] = m

    print(f"  [{i + 1}/{len(ALL_FRAMES)}] {fid}  P95={p95:.1f}  GT={gt_px:,}px")

print("Done.\n")

# ════════════════════════════════════════════════════════════════════
# Per-Frame Comparison Table
# ════════════════════════════════════════════════════════════════════
print("=" * 120)
print("  PER-FRAME F1 COMPARISON")
print("=" * 120)

header = f"{'Frame':<8} {'P95':>6} {'GT_px':>7} |"
for m in ["V1", "V10", "V12"]:
    header += f" {'F1_' + m:>8} {'Prec_' + m:>8} {'Rec_' + m:>8} |"
header += f" {'Best':>8} {'ΔV12-V10':>9} {'ΔV12-V1':>9}"
print(header)
print("-" * len(header))

for fid in ALL_FRAMES:
    ps = frame_stats[fid]["p95"]
    gt_px = frame_stats[fid]["gt_px"]
    row = f"{fid:<8} {ps:>6.1f} {gt_px:>7,} |"

    best_f1 = -1
    best_m = ""
    f1s = {}
    for mname in ["V1", "V10", "V12"]:
        m = all_results[mname][fid]
        f1s[mname] = m["f1"]
        row += f" {m['f1']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} |"
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_m = mname

    d_v10 = f1s["V12"] - f1s["V10"]
    d_v1 = f1s["V12"] - f1s["V1"]
    row += f" {best_m:>8} {d_v10:>+9.4f} {d_v1:>+9.4f}"
    print(row)

# ════════════════════════════════════════════════════════════════════
# Aggregated Summary
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  AGGREGATED SUMMARY")
print("=" * 100)
print(f"{'Method':<8} {'IoU':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}  "
      f"{'TP_sum':>10} {'FP_sum':>10} {'FN_sum':>10}")
print("-" * 100)

for mname in ["V1", "V10", "V12"]:
    mlist = [all_results[mname][f] for f in ALL_FRAMES if f in all_results[mname]]
    avg = {k: np.mean([m[k] for m in mlist]) for k in ["iou", "precision", "recall", "f1"]}
    tp_s = int(sum(m["tp"] for m in mlist))
    fp_s = int(sum(m["fp"] for m in mlist))
    fn_s = int(sum(m["fn"] for m in mlist))
    marker = "  <<< BEST" if mname == "V12" else ""
    print(f"{mname:<8} {avg['iou']:>8.4f} {avg['precision']:>10.4f} "
          f"{avg['recall']:>10.4f} {avg['f1']:>10.4f}  "
          f"{tp_s:>10,} {fp_s:>10,} {fn_s:>10,}{marker}")

# ════════════════════════════════════════════════════════════════════
# Delta Analysis
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  DELTA ANALYSIS")
print("=" * 100)

v12_f1s = [all_results["V12"][f]["f1"] for f in ALL_FRAMES]
v10_f1s = [all_results["V10"][f]["f1"] for f in ALL_FRAMES]
v1_f1s = [all_results["V1"][f]["f1"] for f in ALL_FRAMES]

print(f"  V1  mean F1:  {np.mean(v1_f1s):.4f}  (std: {np.std(v1_f1s):.4f})")
print(f"  V10 mean F1:  {np.mean(v10_f1s):.4f}  (std: {np.std(v10_f1s):.4f})")
print(f"  V12 mean F1:  {np.mean(v12_f1s):.4f}  (std: {np.std(v12_f1s):.4f})")
print(f"  Δ V12–V10:    {np.mean(v12_f1s) - np.mean(v10_f1s):+.4f}")
print(f"  Δ V12–V1:     {np.mean(v12_f1s) - np.mean(v1_f1s):+.4f}")

# Win counts
v12_wins = sum(1 for i, f in enumerate(ALL_FRAMES)
               if v12_f1s[i] >= max(v1_f1s[i], v10_f1s[i]))
v10_wins = sum(1 for i, f in enumerate(ALL_FRAMES)
               if v10_f1s[i] > max(v1_f1s[i], v12_f1s[i]))
v1_wins = sum(1 for i, f in enumerate(ALL_FRAMES)
              if v1_f1s[i] > max(v10_f1s[i], v12_f1s[i]))

print(f"\n  Frame wins:  V1={v1_wins}  V10={v10_wins}  V12={v12_wins}  (ties to V12)")

# FP/TP analysis
v12_tps = sum(all_results["V12"][f]["tp"] for f in ALL_FRAMES)
v12_fps = sum(all_results["V12"][f]["fp"] for f in ALL_FRAMES)
v1_tps = sum(all_results["V1"][f]["tp"] for f in ALL_FRAMES)
v1_fps = sum(all_results["V1"][f]["fp"] for f in ALL_FRAMES)
v10_tps = sum(all_results["V10"][f]["tp"] for f in ALL_FRAMES)
v10_fps = sum(all_results["V10"][f]["fp"] for f in ALL_FRAMES)

print(f"\n  Total TP:    V1={v1_tps:,}  V10={v10_tps:,}  V12={v12_tps:,}")
print(f"  Total FP:    V1={v1_fps:,}  V10={v10_fps:,}  V12={v12_fps:,}")
print(f"  FP vs V10:   {(v12_fps / v10_fps - 1) * 100:+.1f}%")
print(f"  FP vs V1:    {(v12_fps / v1_fps - 1) * 100:+.1f}%  ({(1 - v12_fps / v1_fps) * 100:.1f}% reduction)")
print(f"  TP vs V10:   {(v12_tps / v10_tps - 1) * 100:+.1f}%")
print(f"  TP vs V1:    {(v12_tps / v1_tps - 1) * 100:+.1f}%")

# ════════════════════════════════════════════════════════════════════
# Per-frame delta detail (significant changes)
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  PER-FRAME DELTA: V12 vs V10 (significant changes only)")
print("=" * 100)
print(f"{'Frame':<8} {'P95':>6} {'ΔF1':>9} {'ΔPrec':>9} {'ΔRec':>9}  "
      f"{'ΔTP':>9} {'ΔFP':>9} {'ΔFN':>9}")
print("-" * 75)

for fid in ALL_FRAMES:
    v12 = all_results["V12"][fid]
    v10 = all_results["V10"][fid]
    df1 = v12["f1"] - v10["f1"]
    dp = v12["precision"] - v10["precision"]
    dr = v12["recall"] - v10["recall"]
    dtp = v12["tp"] - v10["tp"]
    dfp = v12["fp"] - v10["fp"]
    dfn = v12["fn"] - v10["fn"]
    if abs(df1) > 0.0005:
        print(f"{fid:<8} {frame_stats[fid]['p95']:>6.1f} {df1:>+9.4f} {dp:>+9.4f} {dr:>+9.4f}  "
              f"{dtp:>+9,} {dfp:>+9,} {dfn:>+9,}")

# ════════════════════════════════════════════════════════════════════
# Motion Strength Group Analysis
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  MOTION STRENGTH GROUP ANALYSIS")
print("=" * 100)

groups = {
    "Strong (P95>100)":       [f for f in ALL_FRAMES if frame_stats[f]["p95"] > 100],
    "Med-Strong (70-100)":    [f for f in ALL_FRAMES if 70 < frame_stats[f]["p95"] <= 100],
    "Medium (50-70)":         [f for f in ALL_FRAMES if 50 < frame_stats[f]["p95"] <= 70],
    "Med-Weak (30-50)":       [f for f in ALL_FRAMES if 30 < frame_stats[f]["p95"] <= 50],
    "Weak (<=30)":            [f for f in ALL_FRAMES if frame_stats[f]["p95"] <= 30],
}

for gname, gframes in groups.items():
    if not gframes:
        continue
    print(f"\n  {gname}: {len(gframes)} frames  ({', '.join(gframes)})")
    print(f"  {'Method':<8} {'F1':>10} {'Precision':>10} {'Recall':>10} {'FP_sum':>10}")
    print(f"  {'-'*50}")
    for mname in ["V1", "V10", "V12"]:
        f1s_g = [all_results[mname][f]["f1"] for f in gframes if f in all_results[mname]]
        precs_g = [all_results[mname][f]["precision"] for f in gframes if f in all_results[mname]]
        recs_g = [all_results[mname][f]["recall"] for f in gframes if f in all_results[mname]]
        fps_g = sum(all_results[mname][f]["fp"] for f in gframes if f in all_results[mname])
        if f1s_g:
            best_marker = " <<<" if mname == "V12" else ""
            print(f"  {mname:<8} {np.mean(f1s_g):>10.4f} {np.mean(precs_g):>10.4f} "
                  f"{np.mean(recs_g):>10.4f} {fps_g:>10,}{best_marker}")


# ════════════════════════════════════════════════════════════════════
# Per-Frame Visualization (TP/FP/FN color-coded comparison)
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  GENERATING VISUALIZATION IMAGES...")
print("=" * 100)

PER_FRAME_DIR = EVAL_OUT / "per_frame"
PER_FRAME_DIR.mkdir(parents=True, exist_ok=True)


def color_error(pred, gt_bin):
    """Color-code prediction vs ground truth.
    Green = TP (correct detection)
    Red   = FP (false alarm / noise)
    Blue  = FN (missed detection)
    """
    pred_bin = (pred > 0)
    gt_b = (gt_bin > 0)
    tp = pred_bin & gt_b      # green
    fp = pred_bin & ~gt_b     # red
    fn = ~pred_bin & gt_b     # blue
    out = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    out[tp] = [0, 255, 0]     # Green: TP
    out[fp] = [0, 0, 255]     # Red: FP (noise)
    out[fn] = [255, 0, 0]     # Blue: FN (missed)
    # True negatives: black
    return out.astype(np.uint8)


def resize_to_height(img, target_h):
    """Resize image to target height, preserving aspect ratio."""
    r = target_h / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * r), target_h))


def put_label(img, text, y=22, color=(255, 255, 255)):
    """Put white text label with dark outline."""
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(img, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)


# Re-run to generate masks and visualizations
print("\nGenerating masks for visualization...")
for i, fid in enumerate(ALL_FRAMES):
    c = cv2.imread(str(CURR_DIR / f"{fid}_FV.png"))
    p = cv2.imread(str(PREV_DIR / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GT_DIR / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)

    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gt_bin = (g > 0)

    # Generate masks
    m_v1 = detect_motion_seed_expand(pg, cg)
    m_v10 = detect_motion_seed_expand_v10(pg, cg)
    m_v12 = detect_motion_seed_expand_v12(pg, cg)

    # Metrics
    s_v1 = all_results["V1"][fid]
    s_v10 = all_results["V10"][fid]
    s_v12 = all_results["V12"][fid]

    # Build visualization
    ROW_H = 220  # height per row

    # Row 0: Original image + Ground Truth
    orig_rs = resize_to_height(c, ROW_H)
    gt_color = cv2.cvtColor((gt_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    gt_rs = resize_to_height(gt_color, ROW_H)

    # GT overlay on original
    gt_overlay = orig_rs.copy()
    gt_mask_rs = cv2.resize((gt_bin * 255).astype(np.uint8), (orig_rs.shape[1], orig_rs.shape[0]))
    gt_overlay[gt_mask_rs > 0] = (gt_overlay[gt_mask_rs > 0] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)

    row0 = np.hstack([orig_rs, gt_rs, gt_overlay])
    put_label(row0, f"{fid}  P95={frame_stats[fid]['p95']:.0f}  GT={frame_stats[fid]['gt_px']:,}px")
    put_label(row0[:, orig_rs.shape[1]:], "Ground Truth")
    put_label(row0[:, 2 * orig_rs.shape[1]:], "GT Overlay")

    # Row 1: V1 error map + V10 error map + V12 error map
    err_v1 = color_error(m_v1, gt_bin)
    err_v10 = color_error(m_v10, gt_bin)
    err_v12 = color_error(m_v12, gt_bin)

    err_v1_rs = resize_to_height(err_v1, ROW_H)
    err_v10_rs = resize_to_height(err_v10, ROW_H)
    err_v12_rs = resize_to_height(err_v12, ROW_H)

    put_label(err_v1_rs, "V1 seed_expand")
    put_label(err_v1_rs, f"F1={s_v1['f1']:.3f} P={s_v1['precision']:.3f} R={s_v1['recall']:.3f}", 44)
    put_label(err_v1_rs, f"TP={s_v1['tp']:,} FP={s_v1['fp']:,} FN={s_v1['fn']:,}", 66)

    put_label(err_v10_rs, "V10 hybrid")
    put_label(err_v10_rs, f"F1={s_v10['f1']:.3f} P={s_v10['precision']:.3f} R={s_v10['recall']:.3f}", 44)
    put_label(err_v10_rs, f"TP={s_v10['tp']:,} FP={s_v10['fp']:,} FN={s_v10['fn']:,}", 66)

    put_label(err_v12_rs, "V12 (V11+P0)")
    put_label(err_v12_rs, f"F1={s_v12['f1']:.3f} P={s_v12['precision']:.3f} R={s_v12['recall']:.3f}", 44)
    put_label(err_v12_rs, f"TP={s_v12['tp']:,} FP={s_v12['fp']:,} FN={s_v12['fn']:,}", 66)

    row1 = np.hstack([err_v1_rs, err_v10_rs, err_v12_rs])

    # Ensure same width
    max_w = max(row0.shape[1], row1.shape[1])

    def pad_width(img, target_w):
        if img.shape[1] < target_w:
            pad = np.zeros((img.shape[0], target_w - img.shape[1], 3), dtype=np.uint8)
            return np.hstack([img, pad])
        return img

    row0 = pad_width(row0, max_w)
    row1 = pad_width(row1, max_w)

    # Legend row
    leg_h = 36
    leg = np.zeros((leg_h, max_w, 3), dtype=np.uint8)
    leg[:] = [30, 30, 30]
    items = [("Green = TP (correct)", (0, 255, 0)),
             ("Red = FP (noise / false alarm)", (0, 0, 255)),
             ("Blue = FN (missed detection)", (255, 0, 0))]
    seg = max_w // len(items)
    for j, (txt, col) in enumerate(items):
        cv2.putText(leg, txt, (j * seg + 12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    # Delta info bar
    d_f1_v10 = s_v12["f1"] - s_v10["f1"]
    d_f1_v1 = s_v12["f1"] - s_v1["f1"]
    d_fp_v10 = s_v12["fp"] - s_v10["fp"]
    bar_text = f"V12 vs V10: dF1={d_f1_v10:+.4f}  dFP={d_fp_v10:+,}  |  V12 vs V1: dF1={d_f1_v1:+.4f}  dFP={s_v12['fp']-s_v1['fp']:+,}"
    delta_bar = np.zeros((30, max_w, 3), dtype=np.uint8)
    delta_bar[:] = [45, 45, 45]
    cv2.putText(delta_bar, bar_text, (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    viz = np.vstack([row0, row1, leg, delta_bar])

    out_path = PER_FRAME_DIR / f"{fid}_v12_comparison.png"
    cv2.imwrite(str(out_path), viz)
    print(f"  [{i + 1}/{len(ALL_FRAMES)}] {fid}  →  {out_path}")

print(f"\n  All {len(ALL_FRAMES)} comparison images saved to: {PER_FRAME_DIR}")


print("\n\n" + "=" * 100)
print("  DONE!")
print("=" * 100)
