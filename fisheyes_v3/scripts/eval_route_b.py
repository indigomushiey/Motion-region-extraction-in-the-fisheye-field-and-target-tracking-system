"""Route B evaluation: undistort → detect → reproject → compare.

Compares three methods on 20 test frames:
  V1  – original seed_expand in fisheye domain (v2 baseline)
  V12 – V12 in fisheye domain (v2 current best)
  R-B – Route B: V12 in perspective domain, reprojected back to fisheye
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import importlib.util

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(__file__).resolve().parents[2] / "fisheyes_v2" / "fisheye_motion_tracking"

# ── Import v3 modules ────────────────────────────────────────────
sys.path.insert(0, str(PROJECT_ROOT))
from core.calib import RadialPoly
from core.motion import detect_motion_persp

# ── Import v2 modules (bypass package shadowing) ─────────────────
def _import_v2(rel_path, name):
    """Import a module from v2 by file path, avoiding package conflicts."""
    fp = V2_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, fp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_v2_metrics = _import_v2("core/evaluation/metrics.py", "v2_metrics")
_v2_fd = _import_v2("core/motion_detection/frame_difference.py", "v2_fd")

compute_metrics = _v2_metrics.compute_metrics
detect_motion_seed_expand = _v2_fd.detect_motion_seed_expand
detect_motion_seed_expand_v12 = _v2_fd.detect_motion_seed_expand_v12

# Data: use v2's dataset directly
DATA = V2_ROOT / "data" / "homework2"
GT_DIR = DATA / "motion_annotation" / "GroudTruth"
CURR_DIR = DATA / "rgb_images"
PREV_DIR = DATA / "previous_images"
CALIB_DIR = DATA / "calibration_data"

OUT_DIR = PROJECT_ROOT / "output" / "evaluation" / "route_b"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_FRAMES = ['00000', '00001', '00002', '00003', '00004', '00005',
              '00006', '00007', '00013', '00016', '00039', '00041', '00053',
              '00022', '00026', '00027', '00033', '00037', '00059', '00065']

print("=" * 100)
print("  ROUTE B: Undistort → Detect → Reproject")
print(f"  Methods: V1 (fisheye) | V12 (fisheye) | R-B (perspective → fisheye)")
print(f"  Frames: {len(ALL_FRAMES)}")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════
# Preload calibration cameras and frames
# ═══════════════════════════════════════════════════════════════════
print("\nPreloading calibration models...")
cameras = {}
for fid in ALL_FRAMES:
    json_path = CALIB_DIR / f"{fid}_FV.json"
    cameras[fid] = RadialPoly(str(json_path), fov_scale=0.5)
print(f"  Loaded {len(cameras)} cameras.")

# ═══════════════════════════════════════════════════════════════════
# Route B pipeline function
# ═══════════════════════════════════════════════════════════════════
def _route_b(pg_fish, cg_fish, camera):
    """Route B: undistort → detect in perspective → reproject to fisheye."""
    prev_und = camera.undistort(pg_fish)
    curr_und = camera.undistort(cg_fish)
    mask_persp = detect_motion_persp(prev_und, curr_und)
    mask_fish = camera.reproject_to_fisheye(mask_persp)
    return (mask_fish > 127).astype(np.uint8) * 255


# ═══════════════════════════════════════════════════════════════════
# Run evaluation
# ═══════════════════════════════════════════════════════════════════
methods = {
    "V1":  lambda pg, cg, fid: detect_motion_seed_expand(pg, cg),
    "V12": lambda pg, cg, fid: detect_motion_seed_expand_v12(pg, cg),
    "R-B": lambda pg, cg, fid: _route_b(pg, cg, cameras[fid]),
}

all_results = {name: {} for name in methods}
frame_stats = {}

print("\nProcessing frames...")
for idx, fid in enumerate(ALL_FRAMES):
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

    for mname, func in methods.items():
        mask = func(pg, cg, fid)
        m = compute_metrics(mask, gt_bin.astype(np.uint8) * 255)
        all_results[mname][fid] = m

    print(f"  [{idx + 1}/{len(ALL_FRAMES)}] {fid}  P95={p95:.0f}  GT={gt_px:,}px"
          f"  V1={all_results['V1'][fid]['f1']:.3f}"
          f"  V12={all_results['V12'][fid]['f1']:.3f}"
          f"  R-B={all_results['R-B'][fid]['f1']:.3f}")

print("Done.\n")


# ═══════════════════════════════════════════════════════════════════
# Per-frame table
# ═══════════════════════════════════════════════════════════════════
print("=" * 120)
print("  PER-FRAME F1 COMPARISON")
print("=" * 120)

header = f"{'Frame':<8} {'P95':>6} {'GT_px':>7} |"
for m in ["V1", "V12", "R-B"]:
    header += f" {'F1_' + m:>8} {'Prec_' + m:>8} {'Rec_' + m:>8} |"
header += f" {'Best':>8} {'ΔRB-V12':>9} {'ΔRB-V1':>9}"
print(header)
print("-" * len(header))

for fid in ALL_FRAMES:
    ps = frame_stats[fid]
    row = f"{fid:<8} {ps['p95']:>6.0f} {ps['gt_px']:>7,} |"
    best_f1, best_m = -1, ""
    f1s = {}
    for mname in ["V1", "V12", "R-B"]:
        m = all_results[mname][fid]
        f1s[mname] = m["f1"]
        row += f" {m['f1']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} |"
        if m["f1"] > best_f1:
            best_f1, best_m = m["f1"], mname
    d_v12 = f1s["R-B"] - f1s["V12"]
    d_v1 = f1s["R-B"] - f1s["V1"]
    row += f" {best_m:>8} {d_v12:>+9.4f} {d_v1:>+9.4f}"
    print(row)

# ═══════════════════════════════════════════════════════════════════
# Aggregated summary
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  AGGREGATED SUMMARY")
print("=" * 100)
print(f"{'Method':<8} {'IoU':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}  "
      f"{'TP_sum':>10} {'FP_sum':>10} {'FN_sum':>10}")
print("-" * 100)

for mname in ["V1", "V12", "R-B"]:
    ml = [all_results[mname][f] for f in ALL_FRAMES]
    avg = {k: np.mean([m[k] for m in ml]) for k in ["iou", "precision", "recall", "f1"]}
    tp_s = int(sum(m["tp"] for m in ml))
    fp_s = int(sum(m["fp"] for m in ml))
    fn_s = int(sum(m["fn"] for m in ml))
    marker = "  <<< BEST" if mname == "R-B" else ""
    print(f"{mname:<8} {avg['iou']:>8.4f} {avg['precision']:>10.4f} "
          f"{avg['recall']:>10.4f} {avg['f1']:>10.4f}  "
          f"{tp_s:>10,} {fp_s:>10,} {fn_s:>10,}{marker}")

# ═══════════════════════════════════════════════════════════════════
# Delta analysis
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  DELTA ANALYSIS")
print("=" * 100)
rb_f1s = [all_results["R-B"][f]["f1"] for f in ALL_FRAMES]
v12_f1s = [all_results["V12"][f]["f1"] for f in ALL_FRAMES]
v1_f1s = [all_results["V1"][f]["f1"] for f in ALL_FRAMES]
print(f"  V1  mean F1: {np.mean(v1_f1s):.4f}  (std: {np.std(v1_f1s):.4f})")
print(f"  V12 mean F1: {np.mean(v12_f1s):.4f}  (std: {np.std(v12_f1s):.4f})")
print(f"  R-B mean F1: {np.mean(rb_f1s):.4f}  (std: {np.std(rb_f1s):.4f})")
print(f"  Δ R-B–V12:   {np.mean(rb_f1s) - np.mean(v12_f1s):+.4f}")
print(f"  Δ R-B–V1:    {np.mean(rb_f1s) - np.mean(v1_f1s):+.4f}")

rb_wins = sum(1 for i, f in enumerate(ALL_FRAMES) if rb_f1s[i] >= max(v1_f1s[i], v12_f1s[i]))
v12_wins = sum(1 for i, f in enumerate(ALL_FRAMES) if v12_f1s[i] > max(v1_f1s[i], rb_f1s[i]))
v1_wins = sum(1 for i, f in enumerate(ALL_FRAMES) if v1_f1s[i] > max(v12_f1s[i], rb_f1s[i]))
print(f"\n  Frame wins: V1={v1_wins}  V12={v12_wins}  R-B={rb_wins}")

rb_tp = sum(all_results["R-B"][f]["tp"] for f in ALL_FRAMES)
rb_fp = sum(all_results["R-B"][f]["fp"] for f in ALL_FRAMES)
v12_tp = sum(all_results["V12"][f]["tp"] for f in ALL_FRAMES)
v12_fp = sum(all_results["V12"][f]["fp"] for f in ALL_FRAMES)
v1_tp = sum(all_results["V1"][f]["tp"] for f in ALL_FRAMES)
v1_fp = sum(all_results["V1"][f]["fp"] for f in ALL_FRAMES)
print(f"\n  Total TP:  V1={v1_tp:,}  V12={v12_tp:,}  R-B={rb_tp:,}")
print(f"  Total FP:  V1={v1_fp:,}  V12={v12_fp:,}  R-B={rb_fp:,}")
print(f"  FP vs V12: {(rb_fp / v12_fp - 1) * 100:+.1f}%")
print(f"  TP vs V12: {(rb_tp / v12_tp - 1) * 100:+.1f}%")

# ═══════════════════════════════════════════════════════════════════
# Per-frame delta
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  PER-FRAME DELTA: R-B vs V12 (significant changes)")
print("=" * 100)
print(f"{'Frame':<8} {'P95':>6} {'ΔF1':>9} {'ΔPrec':>9} {'ΔRec':>9}  "
      f"{'ΔTP':>9} {'ΔFP':>9} {'ΔFN':>9}")
print("-" * 80)
for fid in ALL_FRAMES:
    rb = all_results["R-B"][fid]
    v12 = all_results["V12"][fid]
    df1 = rb["f1"] - v12["f1"]
    if abs(df1) > 0.0005:
        print(f"{fid:<8} {frame_stats[fid]['p95']:>6.0f} {df1:>+9.4f} "
              f"{rb['precision']-v12['precision']:>+9.4f} "
              f"{rb['recall']-v12['recall']:>+9.4f}  "
              f"{rb['tp']-v12['tp']:>+9,} {rb['fp']-v12['fp']:>+9,} "
              f"{rb['fn']-v12['fn']:>+9,}")

# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  GENERATING VISUALIZATION...")
print("=" * 100)

PER_FRAME_DIR = OUT_DIR / "per_frame"
PER_FRAME_DIR.mkdir(parents=True, exist_ok=True)


def color_error(pred, gt_bin):
    pred_bin = (pred > 0)
    gt_b = (gt_bin > 0)
    out = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    out[pred_bin & gt_b] = [0, 255, 0]      # TP green
    out[pred_bin & ~gt_b] = [0, 0, 255]     # FP red
    out[~pred_bin & gt_b] = [255, 0, 0]     # FN blue
    return out


def resize_to_h(img, th):
    r = th / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * r), th))


def put_label(img, text, y=22, color=(255, 255, 255)):
    cv2.putText(img, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(img, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)


print("\nGenerating masks for visualization...")
for idx, fid in enumerate(ALL_FRAMES):
    c = cv2.imread(str(CURR_DIR / f"{fid}_FV.png"))
    p = cv2.imread(str(PREV_DIR / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GT_DIR / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gt_bin = (g > 0)

    m_v1 = detect_motion_seed_expand(pg, cg)
    m_v12 = detect_motion_seed_expand_v12(pg, cg)
    m_rb = _route_b(pg, cg, cameras[fid])

    s = {m: all_results[m][fid] for m in ["V1", "V12", "R-B"]}

    RH = 220
    orig_rs = resize_to_h(c, RH)
    gt_rs = resize_to_h(cv2.cvtColor((gt_bin * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), RH)
    gt_ov = orig_rs.copy()
    gm = cv2.resize((gt_bin * 255).astype(np.uint8), (orig_rs.shape[1], orig_rs.shape[0]))
    gt_ov[gm > 0] = (gt_ov[gm > 0] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)

    row0 = np.hstack([orig_rs, gt_rs, gt_ov])
    put_label(row0, f"{fid}  P95={frame_stats[fid]['p95']:.0f}  GT={frame_stats[fid]['gt_px']:,}px")

    e_v1 = resize_to_h(color_error(m_v1, gt_bin), RH)
    e_v12 = resize_to_h(color_error(m_v12, gt_bin), RH)
    e_rb = resize_to_h(color_error(m_rb, gt_bin), RH)

    for e, mname in [(e_v1, "V1"), (e_v12, "V12"), (e_rb, "R-B")]:
        put_label(e, f"{mname}  F1={s[mname]['f1']:.3f}  P={s[mname]['precision']:.3f}  R={s[mname]['recall']:.3f}")
        put_label(e, f"TP={s[mname]['tp']:,}  FP={s[mname]['fp']:,}", 42)

    row1 = np.hstack([e_v1, e_v12, e_rb])
    mw = max(row0.shape[1], row1.shape[1])
    if row0.shape[1] < mw:
        row0 = np.hstack([row0, np.zeros((row0.shape[0], mw - row0.shape[1], 3), dtype=np.uint8)])
    if row1.shape[1] < mw:
        row1 = np.hstack([row1, np.zeros((row1.shape[0], mw - row1.shape[1], 3), dtype=np.uint8)])

    leg = np.zeros((30, mw, 3), dtype=np.uint8)
    leg[:] = [30, 30, 30]
    for j, (txt, col) in enumerate([
        ("Green=TP", (0, 255, 0)), ("Red=FP", (0, 0, 255)), ("Blue=FN", (255, 0, 0))
    ]):
        cv2.putText(leg, txt, (j * mw // 3 + 10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)

    db = np.zeros((24, mw, 3), dtype=np.uint8)
    db[:] = [45, 45, 45]
    dtxt = f"R-B vs V12: dF1={s['R-B']['f1']-s['V12']['f1']:+.4f}  dFP={s['R-B']['fp']-s['V12']['fp']:+,}"
    cv2.putText(db, dtxt, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    viz = np.vstack([row0, row1, leg, db])
    out_path = PER_FRAME_DIR / f"{fid}_route_b.png"
    cv2.imwrite(str(out_path), viz)
    print(f"  [{idx+1}/{len(ALL_FRAMES)}] {fid} → {out_path}")

print(f"\n  All {len(ALL_FRAMES)} images saved to: {PER_FRAME_DIR}")
print("\n" + "=" * 100)
print("  DONE!")
print("=" * 100)
