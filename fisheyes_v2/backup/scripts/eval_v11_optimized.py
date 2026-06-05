"""Comprehensive evaluation: V11 vs V10 vs V5 vs V1 on all available frames.

Pattern follows eval_v10_final.py but extended with:
  - All frames with GT (not just 13)
  - V11 added to comparison
  - Motion-strength group analysis
  - Per-frame visualization (V1/V5/V10/V11 error maps)
  - Detailed delta analysis
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.motion_detection.frame_difference import (
    detect_motion_diff,
    detect_motion_flow,
    detect_motion_seed_expand,
    detect_motion_seed_expand_v5,
    detect_motion_seed_expand_v10,
    detect_motion_seed_expand_v11,
)
from core.evaluation.metrics import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "homework2"
GT_DIR = DATA / "motion_annotation" / "GroudTruth"
CURR_DIR = DATA / "rgb_images"
PREV_DIR = DATA / "previous_images"
EVAL_OUT = PROJECT_ROOT / "output" / "evaluation" / "v11_optimized"
EVAL_OUT.mkdir(parents=True, exist_ok=True)
PER_FRAME_DIR = EVAL_OUT / "per_frame"
PER_FRAME_DIR.mkdir(parents=True, exist_ok=True)

# ── Find all frames with GT ────────────────────────────────────────
gt_files = sorted(GT_DIR.glob("*_FV.png"))
ALL_FRAMES = [f.stem.replace("_FV", "") for f in gt_files
              if (CURR_DIR / f"{f.stem.replace('_FV', '')}_FV.png").exists()
              and (PREV_DIR / f"{f.stem.replace('_FV', '')}_FV_prev.png").exists()]

print(f"Found {len(ALL_FRAMES)} frames with GT")

METHODS = {
    "diff":        lambda pg, cg: detect_motion_diff(pg, cg, diff_thresh=40, morph_ksize=5),
    "flow":        lambda pg, cg: detect_motion_flow(pg, cg, mag_thresh=3.0, morph_ksize=9),
    "V1_seed_exp": lambda pg, cg: detect_motion_seed_expand(pg, cg),
    "V5_ratio":    lambda pg, cg: detect_motion_seed_expand_v5(pg, cg),
    "V10_hybrid":  lambda pg, cg: detect_motion_seed_expand_v10(pg, cg),
    "V11":         lambda pg, cg: detect_motion_seed_expand_v11(pg, cg),
}

M_COLORS = {
    "diff": "#3498db", "flow": "#e74c3c",
    "V1_seed_exp": "#2ecc71", "V5_ratio": "#f39c12",
    "V10_hybrid": "#9b59b6", "V11": "#e67e22",
}

KEY_METHODS = ["V1_seed_exp", "V5_ratio", "V10_hybrid", "V11"]


def load_pair(fid):
    c = cv2.imread(str(CURR_DIR / f"{fid}_FV.png"))
    p = cv2.imread(str(PREV_DIR / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GT_DIR / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    return pg, cg, c, (g > 0)


# ════════════════════════════════════════════════════════════════════
# 1. Run evaluation
# ════════════════════════════════════════════════════════════════════
print("=" * 80)
print("  V11 OPTIMIZED — COMPREHENSIVE EVALUATION")
print(f"  {len(ALL_FRAMES)} frames, {len(METHODS)} methods")
print("=" * 80)

all_results = {m: {} for m in METHODS}
frame_stats = {}

for i, fid in enumerate(ALL_FRAMES):
    pg, cg, orig, gt_bin = load_pair(fid)

    diff = cv2.absdiff(cg, pg)
    p95 = np.percentile(diff, 95)
    p50 = np.percentile(diff, 50)
    gt_px = int(gt_bin.sum())
    frame_stats[fid] = {"p95": p95, "p50": p50, "gt_px": gt_px}

    for mname, func in METHODS.items():
        mask = func(pg, cg)
        m = compute_metrics(mask, gt_bin.astype(np.uint8) * 255)
        all_results[mname][fid] = m

    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{len(ALL_FRAMES)} frames done")

print("  Evaluation complete.")


# ════════════════════════════════════════════════════════════════════
# 2. Aggregated Summary
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("  AGGREGATED SUMMARY")
print("=" * 80)
print(f"{'Method':<16} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}  {'TP_sum':>9} {'FP_sum':>9} {'FN_sum':>7}")
print("-" * 80)

for mname in METHODS:
    mlist = [all_results[mname][f] for f in ALL_FRAMES if f in all_results[mname]]
    if not mlist:
        continue
    avg = {k: np.mean([m[k] for m in mlist]) for k in ["iou", "precision", "recall", "f1"]}
    tp_s = int(sum(m["tp"] for m in mlist))
    fp_s = int(sum(m["fp"] for m in mlist))
    fn_s = int(sum(m["fn"] for m in mlist))
    marker = " <<<" if mname == "V11" else ""
    print(f"{mname:<16} {avg['iou']:>8.4f} {avg['precision']:>8.4f} "
          f"{avg['recall']:>8.4f} {avg['f1']:>8.4f}  "
          f"{tp_s:>9,} {fp_s:>9,} {fn_s:>7,}{marker}")


# ════════════════════════════════════════════════════════════════════
# 3. Per-Frame F1 Table (Key Methods)
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 95)
print("  PER-FRAME F1 (Key Methods)")
print("=" * 95)
hdr = f"{'Frame':<8} {'P95':>6} {'GT_px':>7}"
for m in KEY_METHODS:
    hdr += f" {m:>10}"
hdr += f" {'Best':>8}"
print(hdr)
print("-" * len(hdr))

for fid in ALL_FRAMES:
    ps = frame_stats[fid]["p95"]
    gt_px = frame_stats[fid]["gt_px"]
    row = f"{fid:<8} {ps:>6.1f} {gt_px:>7,}"
    f1s = {}
    for m in KEY_METHODS:
        f1 = all_results[m].get(fid, {}).get("f1", 0)
        f1s[m] = f1
    best_m = max(f1s, key=f1s.get)
    for m in KEY_METHODS:
        f1 = f1s[m]
        mark = "*" if m == best_m else " "
        row += f" {mark}{f1:>9.4f}"
    row += f" {best_m:>8}"
    print(row)


# ════════════════════════════════════════════════════════════════════
# 4. V11 vs V10 Delta Analysis
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("  V11 vs V10 — DELTA ANALYSIS")
print("=" * 80)
print(f"{'Frame':<8} {'ΔF1':>8} {'ΔPrec':>8} {'ΔRecall':>8}  {'ΔTP':>8} {'ΔFP':>10} {'ΔFN':>7}")
print("-" * 58)

d_f1s, d_ps, d_rs = [], [], []
for fid in ALL_FRAMES:
    v11 = all_results["V11"].get(fid)
    v10 = all_results["V10_hybrid"].get(fid)
    if not v11 or not v10:
        continue
    df1 = v11["f1"] - v10["f1"]
    dp = v11["precision"] - v10["precision"]
    dr = v11["recall"] - v10["recall"]
    dtp = v11["tp"] - v10["tp"]
    dfp = v11["fp"] - v10["fp"]
    dfn = v11["fn"] - v10["fn"]
    d_f1s.append(df1); d_ps.append(dp); d_rs.append(dr)
    # Only print significant deltas
    if abs(df1) > 0.001:
        print(f"{fid:<8} {df1:>+8.4f} {dp:>+8.4f} {dr:>+8.4f}  "
              f"{dtp:>+8,} {dfp:>+10,} {dfn:>+7,}")

print("-" * 58)
print(f"{'MEAN':<8} {np.mean(d_f1s):>+8.4f} {np.mean(d_ps):>+8.4f} {np.mean(d_rs):>+8.4f}")
print(f"  V11 better on {sum(1 for d in d_f1s if d > 0)}/{len(d_f1s)} frames")


# ════════════════════════════════════════════════════════════════════
# 5. V11 vs V1 Delta Analysis
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("  V11 vs V1 — DELTA ANALYSIS")
print("=" * 80)
print(f"{'Frame':<8} {'ΔF1':>8} {'ΔPrec':>8} {'ΔRecall':>8}  {'ΔTP':>8} {'ΔFP':>10} {'ΔFN':>7}")
print("-" * 58)

d_f1s, d_ps, d_rs = [], [], []
for fid in ALL_FRAMES:
    v11 = all_results["V11"].get(fid)
    v1 = all_results["V1_seed_exp"].get(fid)
    if not v11 or not v1:
        continue
    df1 = v11["f1"] - v1["f1"]
    dp = v11["precision"] - v1["precision"]
    dr = v11["recall"] - v1["recall"]
    dtp = v11["tp"] - v1["tp"]
    dfp = v11["fp"] - v1["fp"]
    dfn = v11["fn"] - v1["fn"]
    d_f1s.append(df1); d_ps.append(dp); d_rs.append(dr)
    if abs(df1) > 0.001:
        print(f"{fid:<8} {df1:>+8.4f} {dp:>+8.4f} {dr:>+8.4f}  "
              f"{dtp:>+8,} {dfp:>+10,} {dfn:>+7,}")

print("-" * 58)
print(f"{'MEAN':<8} {np.mean(d_f1s):>+8.4f} {np.mean(d_ps):>+8.4f} {np.mean(d_rs):>+8.4f}")
print(f"  V11 better on {sum(1 for d in d_f1s if d > 0)}/{len(d_f1s)} frames")


# ════════════════════════════════════════════════════════════════════
# 6. Motion-Strength Group Analysis
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("  METHOD PREFERENCE BY MOTION STRENGTH")
print("=" * 80)

groups = {
    "Strong (P95>100)": [f for f in ALL_FRAMES if frame_stats.get(f, {}).get("p95", 0) > 100],
    "Med-Strong (80-100)": [f for f in ALL_FRAMES if 80 < frame_stats.get(f, {}).get("p95", 0) <= 100],
    "Medium (50-80)": [f for f in ALL_FRAMES if 50 < frame_stats.get(f, {}).get("p95", 0) <= 80],
    "Med-Weak (30-50)": [f for f in ALL_FRAMES if 30 < frame_stats.get(f, {}).get("p95", 0) <= 50],
    "Weak (P95<30)": [f for f in ALL_FRAMES if frame_stats.get(f, {}).get("p95", 0) <= 30],
}

for gname, gframes in groups.items():
    if not gframes:
        continue
    print(f"\n  {gname}: {len(gframes)} frames")
    scores = []
    for mname in KEY_METHODS:
        f1s = [all_results[mname][f]["f1"] for f in gframes if f in all_results[mname]]
        if f1s:
            avg_f1 = np.mean(f1s)
            precs = [all_results[mname][f]["precision"] for f in gframes if f in all_results[mname]]
            recalls = [all_results[mname][f]["recall"] for f in gframes if f in all_results[mname]]
            scores.append((mname, avg_f1, np.mean(precs), np.mean(recalls)))
    scores.sort(key=lambda x: -x[1])
    for sn, f1, prec, rec in scores:
        bar = "█" * int(f1 * 100)
        print(f"    {sn:>12s}  F1={f1:.4f}  Prec={prec:.4f}  Recall={rec:.4f}  {bar}")


# ════════════════════════════════════════════════════════════════════
# 7. Per-Frame Visualization (first 20 frames + extremes)
# ════════════════════════════════════════════════════════════════════
print("\n\nGenerating per-frame comparison images...")

# Select frames for visualization: first 15, plus extremes
viz_frames = ALL_FRAMES[:15].copy()
# Add strongest motion frames
by_p95 = sorted(ALL_FRAMES, key=lambda f: frame_stats[f]["p95"], reverse=True)
for f in by_p95[:3]:
    if f not in viz_frames:
        viz_frames.append(f)
# Add weakest motion frames
for f in by_p95[-3:]:
    if f not in viz_frames:
        viz_frames.append(f)

print(f"  Visualizing {len(viz_frames)} frames")


def color_error(pred, gt_bin, bg_img=None):
    pred_bin = (pred > 0)
    gt_b = (gt_bin > 0)
    tp = pred_bin & gt_b
    fp = pred_bin & ~gt_b
    fn = ~pred_bin & gt_b
    if bg_img is not None:
        gray = cv2.cvtColor(bg_img, cv2.COLOR_BGR2GRAY)
        out = (cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) * 0.35).astype(np.float32)
    else:
        out = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.float32)
    out[tp] = [0, 255, 0]
    out[fp] = [0, 0, 255]
    out[fn] = [255, 0, 0]
    return np.clip(out, 0, 255).astype(np.uint8)


V_COMPARE = ["V1_seed_exp", "V5_ratio", "V10_hybrid", "V11"]

for fid in viz_frames:
    pg, cg, orig, gt_bin = load_pair(fid)
    H = 260

    def rs(img, hh=H):
        r = hh / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * r), hh))

    orig_rs = rs(orig)
    gt_vis = rs(cv2.cvtColor((gt_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))

    # GT overlay
    gt_ov = orig_rs.copy()
    gt_rs_mask = cv2.resize((gt_bin * 255).astype(np.uint8), (orig_rs.shape[1], orig_rs.shape[0]))
    gt_ov[gt_rs_mask > 0] = (gt_ov[gt_rs_mask > 0] * 0.35 + np.array([0, 255, 0]) * 0.65).astype(np.uint8)

    row1 = np.hstack([orig_rs, gt_vis, gt_ov])

    def put(img, txt, y=22):
        cv2.putText(img, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    put(row1, f"{fid} Original")
    put(row1[:, orig_rs.shape[1]:], "Ground Truth")
    put(row1[:, 2*orig_rs.shape[1]:], "GT Overlay")

    errs = []
    for mname in V_COMPARE:
        mask = None
        # Re-run to get mask
        func = METHODS[mname]
        mask = func(pg, cg)

        err = color_error(mask, gt_bin, orig)
        err_rs = rs(err)
        m = all_results[mname][fid]
        put(err_rs, mname)
        put(err_rs, f"F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f}", 44)
        put(err_rs, f"TP={m['tp']:,} FP={m['fp']:,}", 66)
        errs.append(err_rs)

    row2 = np.hstack(errs[:2])
    row3 = np.hstack(errs[2:])

    # Ensure all rows have same width (pad if needed)
    max_w = max(row1.shape[1], row2.shape[1], row3.shape[1])
    def pad_width(img, target_w):
        if img.shape[1] < target_w:
            pad = np.zeros((img.shape[0], target_w - img.shape[1], 3), dtype=np.uint8)
            return np.hstack([img, pad])
        return img
    row1 = pad_width(row1, max_w)
    row2 = pad_width(row2, max_w)
    row3 = pad_width(row3, max_w)

    # Legend
    leg_h = 40
    leg = np.zeros((leg_h, max_w, 3), dtype=np.uint8)
    leg[:] = [35, 35, 35]
    items = [("Green=TP", (0, 255, 0)), ("Red=FP", (0, 0, 255)), ("Blue=FN", (255, 0, 0))]
    seg = max_w // len(items)
    for j, (txt, col) in enumerate(items):
        cv2.putText(leg, txt, (j * seg + 10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    viz = np.vstack([row1, row2, row3, leg])
    cv2.imwrite(str(PER_FRAME_DIR / f"{fid}_v11_comparison.png"), viz)

print(f"  Per-frame images saved to {PER_FRAME_DIR}")


# ════════════════════════════════════════════════════════════════════
# 8. Bar Charts
# ════════════════════════════════════════════════════════════════════
print("\nGenerating summary charts...")


def bar_chart(labels, values, title, fname, highlight=None):
    n = len(labels)
    bw, gap = 65, 35
    ml, mr, mt, mb = 110, 50, 55, 110
    cw = ml + n * (bw + gap) - gap + mr
    ch = 450
    cv = np.ones((ch, cw, 3), np.uint8) * 242
    mv = max(values) * 1.18 if max(values) > 0 else 1
    ph = ch - mt - mb

    for i, (l, v) in enumerate(zip(labels, values)):
        x = ml + i * (bw + gap)
        bh = int(v / mv * ph)
        y = mt + ph - bh
        is_hl = (highlight and l == highlight)
        col = (60, 140, 255) if is_hl else (180, 180, 180)
        cv2.rectangle(cv, (x, y), (x + bw, mt + ph), col, -1)
        cv2.rectangle(cv, (x, y), (x + bw, mt + ph), (50, 50, 50), 2)
        txt = f"{v:.4f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(cv, txt, (x + (bw - tw) // 2, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1)
        (tw2, th2), _ = cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(cv, l, (x + (bw - tw2) // 2, mt + ph + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 50, 50), 1)

    # Axes
    cv2.line(cv, (ml - 5, mt), (ml - 5, mt + ph), (50, 50, 50), 2)
    cv2.line(cv, (ml - 5, mt + ph), (cw - mr + 10, mt + ph), (50, 50, 50), 2)
    for i in range(6):
        frac = i / 5
        val = mv * frac
        y = mt + ph - int(frac * ph)
        t = f"{val:.2f}"
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(cv, t, (ml - 10 - tw, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1)
        cv2.line(cv, (ml - 5, y), (cw - mr + 10, y), (215, 215, 215), 1)

    (twt, tht), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(cv, title, ((cw - twt) // 2, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (25, 25, 25), 2)
    cv2.imwrite(str(EVAL_OUT / fname), cv)


labels_all = list(METHODS.keys())
f1s_all = [np.mean([all_results[m][f]["f1"] for f in ALL_FRAMES if f in all_results[m]]) for m in labels_all]
precs_all = [np.mean([all_results[m][f]["precision"] for f in ALL_FRAMES if f in all_results[m]]) for m in labels_all]
recs_all = [np.mean([all_results[m][f]["recall"] for f in ALL_FRAMES if f in all_results[m]]) for m in labels_all]

bar_chart(labels_all, f1s_all, f"Average F1 Score ({len(ALL_FRAMES)} frames)", "v11_summary_f1.png", "V11")
bar_chart(labels_all, precs_all, f"Average Precision ({len(ALL_FRAMES)} frames)", "v11_summary_precision.png", "V11")
bar_chart(labels_all, recs_all, f"Average Recall ({len(ALL_FRAMES)} frames)", "v11_summary_recall.png", "V11")


# ════════════════════════════════════════════════════════════════════
# 9. Trend Chart: Per-Frame F1
# ════════════════════════════════════════════════════════════════════
print("Generating trend chart...")


def trend_chart(show_methods, title, fname, frames=None):
    if frames is None:
        frames = ALL_FRAMES
    nf = len(frames)
    ml, mr, mt, mb = 90, 40, 45, 100
    cw = ml + nf * 42 + mr  # narrower bars for many frames
    ch = 440
    cv = np.ones((ch, cw, 3), np.uint8) * 242

    all_vals = []
    for m in show_methods:
        for f in frames:
            if f in all_results[m]:
                all_vals.append(all_results[m][f]["f1"])
    mv = max(all_vals) * 1.15 if all_vals else 1
    ph = ch - mt - mb
    pw = nf * 42

    for i in range(6):
        frac = i / 5
        y = mt + ph - int(frac * ph)
        cv2.line(cv, (ml, y), (ml + pw, y), (210, 210, 210), 1)
        t = f"{mv * frac:.2f}"
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(cv, t, (ml - 8 - tw, y + th // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70, 70, 70), 1)

    cv2.line(cv, (ml, mt), (ml, mt + ph), (50, 50, 50), 2)
    cv2.line(cv, (ml, mt + ph), (ml + pw, mt + ph), (50, 50, 50), 2)

    for i, fid in enumerate(frames):
        x = ml + i * 42 + 21
        if len(frames) <= 20 or i % 5 == 0:
            (tw, th), _ = cv2.getTextSize(fid, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)
            cv2.putText(cv, fid, (x - tw // 2, mt + ph + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 60, 60), 1)

    colors = [(46, 134, 193), (231, 76, 60), (155, 89, 182), (230, 126, 34)]
    for j, mname in enumerate(show_methods):
        col = colors[j % len(colors)]
        pts = []
        for i, fid in enumerate(frames):
            if fid in all_results.get(mname, {}):
                v = all_results[mname][fid]["f1"]
                x = ml + i * 42 + 21
                y = mt + ph - int(v / mv * ph)
                pts.append((x, y))
        for i in range(len(pts) - 1):
            cv2.line(cv, pts[i], pts[i + 1], col, 2)
        for x, y in pts:
            cv2.circle(cv, (x, y), 3, col, -1)

    # Legend
    lx, ly = ml + pw - 280, mt + 8
    for j, mname in enumerate(show_methods):
        col = colors[j % len(colors)]
        cv2.rectangle(cv, (lx, ly + j * 22), (lx + 14, ly + 14 + j * 22), col, -1)
        cv2.putText(cv, mname, (lx + 20, ly + 12 + j * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40, 40, 40), 1)

    (twt, tht), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(cv, title, ((cw - twt) // 2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (25, 25, 25), 2)
    cv2.imwrite(str(EVAL_OUT / fname), cv)


trend_chart(KEY_METHODS, f"Per-Frame F1: V1 vs V5 vs V10 vs V11 ({len(ALL_FRAMES)} frames)",
            "v11_trend_all.png")


# ════════════════════════════════════════════════════════════════════
# 10. Summary Statistics
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("  SUMMARY STATISTICS")
print("=" * 80)

v11_f1s = [all_results["V11"][f]["f1"] for f in ALL_FRAMES]
v10_f1s = [all_results["V10_hybrid"][f]["f1"] for f in ALL_FRAMES]
v1_f1s = [all_results["V1_seed_exp"][f]["f1"] for f in ALL_FRAMES]
v5_f1s = [all_results["V5_ratio"][f]["f1"] for f in ALL_FRAMES]

print(f"  V1  mean F1: {np.mean(v1_f1s):.4f}  (std: {np.std(v1_f1s):.4f})")
print(f"  V5  mean F1: {np.mean(v5_f1s):.4f}  (std: {np.std(v5_f1s):.4f})")
print(f"  V10 mean F1: {np.mean(v10_f1s):.4f}  (std: {np.std(v10_f1s):.4f})")
print(f"  V11 mean F1: {np.mean(v11_f1s):.4f}  (std: {np.std(v11_f1s):.4f})")
print(f"  Δ V11-V10:   {np.mean(v11_f1s) - np.mean(v10_f1s):+.4f}")
print(f"  Δ V11-V1:    {np.mean(v11_f1s) - np.mean(v1_f1s):+.4f}")

# Count wins
v11_wins = sum(1 for i, f in enumerate(ALL_FRAMES) if v11_f1s[i] >= max(v1_f1s[i], v5_f1s[i], v10_f1s[i]))
v10_wins = sum(1 for i, f in enumerate(ALL_FRAMES) if v10_f1s[i] > max(v1_f1s[i], v5_f1s[i], v11_f1s[i]))
v1_wins = sum(1 for i, f in enumerate(ALL_FRAMES) if v1_f1s[i] > max(v5_f1s[i], v10_f1s[i], v11_f1s[i]))
v5_wins = sum(1 for i, f in enumerate(ALL_FRAMES) if v5_f1s[i] > max(v1_f1s[i], v10_f1s[i], v11_f1s[i]))

print(f"\n  Frame wins: V1={v1_wins}  V5={v5_wins}  V10={v10_wins}  V11={v11_wins}  (ties split)")

# FP/TP analysis
v11_tps = sum(all_results["V11"][f]["tp"] for f in ALL_FRAMES)
v11_fps = sum(all_results["V11"][f]["fp"] for f in ALL_FRAMES)
v1_tps = sum(all_results["V1_seed_exp"][f]["tp"] for f in ALL_FRAMES)
v1_fps = sum(all_results["V1_seed_exp"][f]["fp"] for f in ALL_FRAMES)
v10_tps = sum(all_results["V10_hybrid"][f]["tp"] for f in ALL_FRAMES)
v10_fps = sum(all_results["V10_hybrid"][f]["fp"] for f in ALL_FRAMES)

print(f"\n  Total TP: V1={v1_tps:,}  V10={v10_tps:,}  V11={v11_tps:,}")
print(f"  Total FP: V1={v1_fps:,}  V10={v10_fps:,}  V11={v11_fps:,}")
print(f"  FP reduction V11 vs V1: {(1 - v11_fps / v1_fps) * 100:.1f}%")
print(f"  FP change V11 vs V10:   {(v11_fps / v10_fps - 1) * 100:+.1f}%")


# ════════════════════════════════════════════════════════════════════
# Done
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("  DONE! All results saved to:", EVAL_OUT)
print("=" * 80)
for f in sorted(EVAL_OUT.glob("*.png")):
    print(f"  {f.name}")
