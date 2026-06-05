"""Evaluate V10 against all other variants with full visual output."""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.motion_detection.frame_difference import (
    detect_motion_diff,
    detect_motion_flow,
    detect_motion_seed_expand,
    detect_motion_seed_expand_v2,
    detect_motion_seed_expand_v5,
    detect_motion_seed_expand_v9,
    detect_motion_seed_expand_v10,
)
from core.evaluation.metrics import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "homework2"
EVAL_OUT = PROJECT_ROOT / "output" / "evaluation" / "v10_final"
EVAL_OUT.mkdir(parents=True, exist_ok=True)
PER_FRAME_DIR = EVAL_OUT / "per_frame"
PER_FRAME_DIR.mkdir(parents=True, exist_ok=True)

FRAMES = ["00000", "00001", "00002", "00003", "00004", "00005",
          "00006", "00007", "00013", "00016", "00039", "00041", "00053"]

METHODS = {
    "diff":        lambda pg, cg: detect_motion_diff(pg, cg, diff_thresh=40, morph_ksize=5),
    "flow":        lambda pg, cg: detect_motion_flow(pg, cg, mag_thresh=3.0, morph_ksize=9),
    "V1_seed_exp": lambda pg, cg: detect_motion_seed_expand(pg, cg),
    "V5_ratio":    lambda pg, cg: detect_motion_seed_expand_v5(pg, cg),
    "V9_adaptive": lambda pg, cg: detect_motion_seed_expand_v9(pg, cg),
    "V10_hybrid":  lambda pg, cg: detect_motion_seed_expand_v10(pg, cg),
}

M_COLORS = {
    "diff": "#3498db", "flow": "#e74c3c",
    "V1_seed_exp": "#2ecc71", "V5_ratio": "#f39c12",
    "V9_adaptive": "#9b59b6", "V10_hybrid": "#e67e22",
}


def load_pair(fid):
    c = cv2.imread(str(DATA / "rgb_images" / f"{fid}_FV.png"))
    p = cv2.imread(str(DATA / "previous_images" / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(DATA / "motion_annotation" / "GroudTruth" / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    return pg, cg, c, (g > 0)


# ══════════════════════════════════════════════════════════════════
# 1. Run evaluation
# ══════════════════════════════════════════════════════════════════
print("=" * 80)
print("  V10 HYBRID — FINAL EVALUATION")
print("=" * 80)

all_results = {m: {} for m in METHODS}
all_masks = {m: {} for m in METHODS}
frame_stats = {}

for fid in FRAMES:
    print(f"\n{fid}...")
    try:
        pg, cg, orig, gt_bin = load_pair(fid)
    except FileNotFoundError:
        print(f"  SKIP")
        continue

    diff = cv2.absdiff(cg, pg)
    p95 = np.percentile(diff, 95)
    gt_px = int(gt_bin.sum())
    frame_stats[fid] = {"p95": p95, "gt_px": gt_px}

    for mname, func in METHODS.items():
        mask = func(pg, cg)
        m = compute_metrics(mask, gt_bin.astype(np.uint8) * 255)
        all_results[mname][fid] = m
        all_masks[mname][fid] = mask
        print(f"  {mname:>14s}  F1={m['f1']:.4f}  P={m['precision']:.4f}  "
              f"R={m['recall']:.4f}  TP={m['tp']:>7,}  FP={m['fp']:>7,}  FN={m['fn']:>5,}")


# ══════════════════════════════════════════════════════════════════
# 2. Summary table
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("  AGGREGATED SUMMARY")
print("=" * 80)
print(f"{'Method':<16} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}  {'TP_sum':>9} {'FP_sum':>9} {'FN_sum':>7}")
print("-" * 82)

for mname in METHODS:
    mlist = [all_results[mname][f] for f in FRAMES if f in all_results[mname]]
    if not mlist:
        continue
    avg = {k: np.mean([m[k] for m in mlist]) for k in ["iou", "precision", "recall", "f1"]}
    tp_s = int(sum(m["tp"] for m in mlist))
    fp_s = int(sum(m["fp"] for m in mlist))
    fn_s = int(sum(m["fn"] for m in mlist))
    marker = " <<<" if mname == "V10_hybrid" else ""
    print(f"{mname:<16} {avg['iou']:>8.4f} {avg['precision']:>8.4f} "
          f"{avg['recall']:>8.4f} {avg['f1']:>8.4f}  "
          f"{tp_s:>9,} {fp_s:>9,} {fn_s:>7,}{marker}")


# ══════════════════════════════════════════════════════════════════
# 3. Per-frame F1 table
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 90)
print("  PER-FRAME F1")
print("=" * 90)
hdr = f"{'Frame':<8} {'P95':>6}"
for m in METHODS:
    hdr += f" {m:>11}"
hdr += f" {'Best':>11}"
print(hdr)
print("-" * len(hdr))

for fid in FRAMES:
    ps = frame_stats.get(fid, {}).get("p95", 0)
    row = f"{fid:<8} {ps:>6.1f}"
    f1s = {}
    for m in METHODS:
        f1 = all_results[m].get(fid, {}).get("f1", 0)
        f1s[m] = f1
    best_m = max(f1s, key=f1s.get)
    for m in METHODS:
        f1 = f1s[m]
        mark = "*" if m == best_m else " "
        row += f" {mark}{f1:>10.4f}"
    row += f" {best_m:>11}"
    print(row)


# ══════════════════════════════════════════════════════════════════
# 4. V10 vs baseline delta analysis
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("  V10 vs V1 (seed_expand) — DELTA")
print("=" * 80)
print(f"{'Frame':<8} {'ΔF1':>8} {'ΔPrec':>8} {'ΔRecall':>8}  {'ΔTP':>8} {'ΔFP':>10} {'ΔFN':>7}")
print("-" * 58)

d_f1s, d_ps, d_rs = [], [], []
for fid in FRAMES:
    v10 = all_results["V10_hybrid"].get(fid)
    v1 = all_results["V1_seed_exp"].get(fid)
    if not v10 or not v1:
        continue
    df1 = v10["f1"] - v1["f1"]
    dp = v10["precision"] - v1["precision"]
    dr = v10["recall"] - v1["recall"]
    dtp = v10["tp"] - v1["tp"]
    dfp = v10["fp"] - v1["fp"]
    dfn = v10["fn"] - v1["fn"]
    d_f1s.append(df1); d_ps.append(dp); d_rs.append(dr)
    print(f"{fid:<8} {df1:>+8.4f} {dp:>+8.4f} {dr:>+8.4f}  "
          f"{dtp:>+8,} {dfp:>+10,} {dfn:>+7,}")

print("-" * 58)
print(f"{'MEAN':<8} {np.mean(d_f1s):>+8.4f} {np.mean(d_ps):>+8.4f} {np.mean(d_rs):>+8.4f}")


# ══════════════════════════════════════════════════════════════════
# 5. V10 vs V5 delta
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("  V10 vs V5 (ratio baseline) — DELTA")
print("=" * 80)
print(f"{'Frame':<8} {'ΔF1':>8} {'ΔPrec':>8} {'ΔRecall':>8}  {'ΔTP':>8} {'ΔFP':>10} {'ΔFN':>7}")
print("-" * 58)

d_f1s, d_ps, d_rs = [], [], []
for fid in FRAMES:
    v10 = all_results["V10_hybrid"].get(fid)
    v5 = all_results["V5_ratio"].get(fid)
    if not v10 or not v5:
        continue
    df1 = v10["f1"] - v5["f1"]
    dp = v10["precision"] - v5["precision"]
    dr = v10["recall"] - v5["recall"]
    dtp = v10["tp"] - v5["tp"]
    dfp = v10["fp"] - v5["fp"]
    dfn = v10["fn"] - v5["fn"]
    d_f1s.append(df1); d_ps.append(dp); d_rs.append(dr)
    print(f"{fid:<8} {df1:>+8.4f} {dp:>+8.4f} {dr:>+8.4f}  "
          f"{dtp:>+8,} {dfp:>+10,} {dfn:>+7,}")

print("-" * 58)
print(f"{'MEAN':<8} {np.mean(d_f1s):>+8.4f} {np.mean(d_ps):>+8.4f} {np.mean(d_rs):>+8.4f}")


# ══════════════════════════════════════════════════════════════════
# 6. Per-frame visualization (V10 vs V1 vs V5 vs GT)
# ══════════════════════════════════════════════════════════════════
print("\n\nGenerating per-frame comparisons...")

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


FOCUS = ["V1_seed_exp", "V5_ratio", "V10_hybrid"]

for fid in FRAMES:
    try:
        pg, cg, orig, gt_bin = load_pair(fid)
    except FileNotFoundError:
        continue

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

    put(row1, "Original")
    put(row1[:, orig_rs.shape[1]:], "Ground Truth")
    put(row1[:, 2*orig_rs.shape[1]:], "GT Overlay")

    errs = []
    for mname in FOCUS:
        mask = all_masks[mname].get(fid)
        if mask is None:
            errs.append(np.zeros_like(orig_rs))
            continue
        err = color_error(mask, gt_bin, orig)
        err_rs = rs(err)
        m = all_results[mname][fid]
        put(err_rs, mname)
        put(err_rs, f"F1={m['f1']:.3f} P={m['precision']:.3f} R={m['recall']:.3f}", 44)
        put(err_rs, f"TP={m['tp']:,} FP={m['fp']:,}", 66)
        errs.append(err_rs)

    row2 = np.hstack(errs)

    # Legend
    leg_h = 40
    leg = np.zeros((leg_h, row1.shape[1], 3), dtype=np.uint8)
    leg[:] = [35, 35, 35]
    items = [("Green=TP", (0,255,0)), ("Red=FP", (0,0,255)), ("Blue=FN", (255,0,0))]
    seg = leg.shape[1] // len(items)
    for i, (txt, col) in enumerate(items):
        cv2.putText(leg, txt, (i*seg+10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

    viz = np.vstack([row1, row2, leg])
    cv2.imwrite(str(PER_FRAME_DIR / f"{fid}_v10_comparison.png"), viz)
    print(f"  {fid}_v10_comparison.png")


# ══════════════════════════════════════════════════════════════════
# 7. Bar charts
# ══════════════════════════════════════════════════════════════════
print("\nGenerating charts...")

def bar_chart(labels, values, title, fname, highlight=None):
    n = len(labels)
    bw, gap = 70, 40
    ml, mr, mt, mb = 110, 50, 55, 110
    cw = ml + n*(bw+gap) - gap + mr
    ch = 460
    cv = np.ones((ch, cw, 3), np.uint8) * 242
    mv = max(values) * 1.18 if max(values) > 0 else 1
    ph = ch - mt - mb

    for i, (l, v) in enumerate(zip(labels, values)):
        x = ml + i*(bw+gap)
        bh = int(v/mv*ph)
        y = mt + ph - bh
        is_hl = (highlight and l == highlight)
        col = (60, 140, 255) if is_hl else (180, 180, 180)
        cv2.rectangle(cv, (x, y), (x+bw, mt+ph), col, -1)
        cv2.rectangle(cv, (x, y), (x+bw, mt+ph), (50,50,50), 2)
        txt = f"{v:.4f}"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.putText(cv, txt, (x+(bw-tw)//2, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30,30,30), 1)
        (tw2, th2), _ = cv2.getTextSize(l, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        cv2.putText(cv, l, (x+(bw-tw2)//2, mt+ph+20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50,50,50), 1)

    # Axes
    cv2.line(cv, (ml-5, mt), (ml-5, mt+ph), (50,50,50), 2)
    cv2.line(cv, (ml-5, mt+ph), (cw-mr+10, mt+ph), (50,50,50), 2)
    for i in range(6):
        frac = i/5
        val = mv*frac
        y = mt+ph-int(frac*ph)
        t = f"{val:.2f}"
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(cv, t, (ml-10-tw, y+th//2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70,70,70), 1)
        cv2.line(cv, (ml-5, y), (cw-mr+10, y), (215,215,215), 1)

    (twt, tht), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    cv2.putText(cv, title, ((cw-twt)//2, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25,25,25), 2)
    cv2.imwrite(str(EVAL_OUT / fname), cv)
    print(f"  {fname}")


labels = list(METHODS.keys())
f1s = [np.mean([all_results[m][f]["f1"] for f in FRAMES if f in all_results[m]]) for m in labels]
precs = [np.mean([all_results[m][f]["precision"] for f in FRAMES if f in all_results[m]]) for m in labels]
recs = [np.mean([all_results[m][f]["recall"] for f in FRAMES if f in all_results[m]]) for m in labels]

bar_chart(labels, f1s, "Average F1 Score by Method", "v10_summary_f1.png", "V10_hybrid")
bar_chart(labels, precs, "Average Precision by Method", "v10_summary_precision.png", "V10_hybrid")
bar_chart(labels, recs, "Average Recall by Method", "v10_summary_recall.png", "V10_hybrid")


# ══════════════════════════════════════════════════════════════════
# 8. Trend chart: per-frame F1 for key methods
# ══════════════════════════════════════════════════════════════════
print("\nTrend chart...")

def trend_chart(methods, title, fname):
    nf = len(FRAMES)
    ml, mr, mt, mb = 90, 40, 45, 100
    cw = ml + nf*75 + mr
    ch = 440
    cv = np.ones((ch, cw, 3), np.uint8) * 242

    all_vals = []
    for m in methods:
        for f in FRAMES:
            if f in all_results[m]:
                all_vals.append(all_results[m][f]["f1"])
    mv = max(all_vals) * 1.15 if all_vals else 1
    ph = ch - mt - mb
    pw = nf*75

    for i in range(6):
        frac = i/5
        y = mt+ph-int(frac*ph)
        cv2.line(cv, (ml, y), (ml+pw, y), (210,210,210), 1)
        t = f"{mv*frac:.2f}"
        (tw, th), _ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
        cv2.putText(cv, t, (ml-8-tw, y+th//2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (70,70,70), 1)

    cv2.line(cv, (ml, mt), (ml, mt+ph), (50,50,50), 2)
    cv2.line(cv, (ml, mt+ph), (ml+pw, mt+ph), (50,50,50), 2)

    for i, fid in enumerate(FRAMES):
        x = ml+i*75+37
        (tw, th), _ = cv2.getTextSize(fid, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        cv2.putText(cv, fid, (x-tw//2, mt+ph+20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (60,60,60), 1)

    colors = [(46,134,193), (231,76,60), (46,204,113), (243,156,18), (155,89,182), (230,126,34)]
    for j, mname in enumerate(methods):
        col = colors[j % len(colors)]
        pts = []
        for i, fid in enumerate(FRAMES):
            if fid in all_results.get(mname, {}):
                v = all_results[mname][fid]["f1"]
                x = ml+i*75+37
                y = mt+ph-int(v/mv*ph)
                pts.append((x, y))
        for i in range(len(pts)-1):
            cv2.line(cv, pts[i], pts[i+1], col, 2)
        for x, y in pts:
            cv2.circle(cv, (x, y), 4, col, -1)

    # Legend
    lx, ly = ml+pw-280, mt+8
    for j, mname in enumerate(methods):
        col = colors[j % len(colors)]
        cv2.rectangle(cv, (lx, ly+j*22), (lx+14, ly+14+j*22), col, -1)
        cv2.putText(cv, mname, (lx+20, ly+12+j*22), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (40,40,40), 1)

    (twt, tht), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(cv, title, ((cw-twt)//2, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (25,25,25), 2)
    cv2.imwrite(str(EVAL_OUT / fname), cv)
    print(f"  {fname}")

trend_chart(["V1_seed_exp", "V5_ratio", "V10_hybrid"],
            "Per-Frame F1: V1 vs V5 vs V10", "v10_trend.png")

trend_chart(["diff", "flow", "V1_seed_exp", "V5_ratio", "V10_hybrid"],
            "Per-Frame F1: All Key Methods", "v10_trend_all.png")

# ══════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("  DONE! All results in:", EVAL_OUT)
print("=" * 80)
for f in sorted(EVAL_OUT.glob("*.png")):
    print(f"  {f.name}")
