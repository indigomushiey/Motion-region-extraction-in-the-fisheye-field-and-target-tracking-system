"""Residual Flow evaluation: ego-motion subtraction vs V12 baseline.

Compares: V1 | V12 (raw flow) | R-F (residual flow: actual - ego-motion)
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(__file__).resolve().parents[2] / "fisheyes_v2" / "fisheye_motion_tracking"

sys.path.insert(0, str(PROJECT_ROOT))
from core.ground import get_projector
from core.motion_residual import detect_motion_residual

# Import v2 modules
def _import_v2(rel_path, name):
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

# Data paths
DATA = V2_ROOT / "data" / "homework2"
GT_DIR = DATA / "motion_annotation" / "GroudTruth"
CURR_DIR = DATA / "rgb_images"
PREV_DIR = DATA / "previous_images"
OUT_DIR = PROJECT_ROOT / "output" / "evaluation" / "residual_flow"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_FRAMES = ['00000', '00001', '00002', '00003', '00004', '00005',
              '00006', '00007', '00013', '00016', '00039', '00041', '00053',
              '00022', '00026', '00027', '00033', '00037', '00059', '00065']

print("=" * 100)
print("  RESIDUAL FLOW: Ego-Motion Subtraction")
print("  Methods: V1 | V12 (raw flow) | R-F (residual flow)")
print(f"  Frames: {len(ALL_FRAMES)}")
print("=" * 100)

# Preload ground projector (once, shared by all frames)
print("\nPreloading ground projector...")
proj = get_projector()
road_pct = proj.is_road.mean() * 100
print(f"  Road pixels: {road_pct:.1f}% of image")

# ═══════════════════════════════════════════════════════════════════
methods = {
    "V1":  lambda pg, cg: detect_motion_seed_expand(pg, cg),
    "V12": lambda pg, cg: detect_motion_seed_expand_v12(pg, cg),
    "R-F": lambda pg, cg: detect_motion_residual(pg, cg),
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
        mask = func(pg, cg)
        m = compute_metrics(mask, gt_bin.astype(np.uint8) * 255)
        all_results[mname][fid] = m

    print(f"  [{idx+1}/{len(ALL_FRAMES)}] {fid}  P95={p95:.0f}  GT={gt_px:,}px"
          f"  V1={all_results['V1'][fid]['f1']:.3f}"
          f"  V12={all_results['V12'][fid]['f1']:.3f}"
          f"  R-F={all_results['R-F'][fid]['f1']:.3f}")

print("Done.\n")

# ═══════════════════════════════════════════════════════════════════
# Per-frame table
# ═══════════════════════════════════════════════════════════════════
print("=" * 120)
print("  PER-FRAME F1 COMPARISON")
print("=" * 120)
header = f"{'Frame':<8} {'P95':>6} {'GT_px':>7} |"
for m in ["V1", "V12", "R-F"]:
    header += f" {'F1_'+m:>8} {'Prec_'+m:>8} {'Rec_'+m:>8} |"
header += f" {'Best':>8} {'ΔRF-V12':>9} {'ΔRF-V1':>9}"
print(header)
print("-" * len(header))

for fid in ALL_FRAMES:
    ps = frame_stats[fid]
    row = f"{fid:<8} {ps['p95']:>6.0f} {ps['gt_px']:>7,} |"
    best_f1, best_m = -1, ""
    f1s = {}
    for mname in ["V1", "V12", "R-F"]:
        m = all_results[mname][fid]
        f1s[mname] = m["f1"]
        row += f" {m['f1']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} |"
        if m["f1"] > best_f1:
            best_f1, best_m = m["f1"], mname
    row += f" {best_m:>8} {f1s['R-F']-f1s['V12']:>+9.4f} {f1s['R-F']-f1s['V1']:>+9.4f}"
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

for mname in ["V1", "V12", "R-F"]:
    ml = [all_results[mname][f] for f in ALL_FRAMES]
    avg = {k: np.mean([m[k] for m in ml]) for k in ["iou", "precision", "recall", "f1"]}
    tp_s = int(sum(m["tp"] for m in ml))
    fp_s = int(sum(m["fp"] for m in ml))
    fn_s = int(sum(m["fn"] for m in ml))
    marker = "  <<< BEST" if mname == "R-F" else ""
    print(f"{mname:<8} {avg['iou']:>8.4f} {avg['precision']:>10.4f} "
          f"{avg['recall']:>10.4f} {avg['f1']:>10.4f}  "
          f"{tp_s:>10,} {fp_s:>10,} {fn_s:>10,}{marker}")

# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  DELTA ANALYSIS")
print("=" * 100)
rf = [all_results["R-F"][f]["f1"] for f in ALL_FRAMES]
v12 = [all_results["V12"][f]["f1"] for f in ALL_FRAMES]
v1 = [all_results["V1"][f]["f1"] for f in ALL_FRAMES]
print(f"  V1  mean F1: {np.mean(v1):.4f}")
print(f"  V12 mean F1: {np.mean(v12):.4f}")
print(f"  R-F mean F1: {np.mean(rf):.4f}")
print(f"  Δ R-F–V12:   {np.mean(rf)-np.mean(v12):+.4f}")
print(f"  Δ R-F–V1:    {np.mean(rf)-np.mean(v1):+.4f}")

rf_wins = sum(1 for i,f in enumerate(ALL_FRAMES) if rf[i]>=max(v1[i],v12[i]))
v12_wins = sum(1 for i,f in enumerate(ALL_FRAMES) if v12[i]>max(v1[i],rf[i]))
v1_wins = sum(1 for i,f in enumerate(ALL_FRAMES) if v1[i]>max(v12[i],rf[i]))
print(f"\n  Frame wins: V1={v1_wins}  V12={v12_wins}  R-F={rf_wins}")

rf_tp = sum(all_results["R-F"][f]["tp"] for f in ALL_FRAMES)
rf_fp = sum(all_results["R-F"][f]["fp"] for f in ALL_FRAMES)
v12_tp = sum(all_results["V12"][f]["tp"] for f in ALL_FRAMES)
v12_fp = sum(all_results["V12"][f]["fp"] for f in ALL_FRAMES)
print(f"\n  Total TP: V12={v12_tp:,}  R-F={rf_tp:,}  ({(rf_tp/v12_tp-1)*100:+.1f}%)")
print(f"  Total FP: V12={v12_fp:,}  R-F={rf_fp:,}  ({(rf_fp/v12_fp-1)*100:+.1f}%)")

print("\n" + "=" * 100)
print("  DONE!")
print("=" * 100)
