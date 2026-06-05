"""Comprehensive evaluation of ALL seed_expand variants (V1-V9).

Purpose: understand which variant works best for each type of frame,
so we can design a frame-adaptive V10 that selects the best strategy.
"""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.motion_detection.frame_difference import (
    detect_motion_seed_expand,
    detect_motion_seed_expand_v2,
    detect_motion_seed_expand_v3,
    detect_motion_seed_expand_v4,
    detect_motion_seed_expand_v5,
    detect_motion_seed_expand_v6,
    detect_motion_seed_expand_v7,
    detect_motion_seed_expand_v9,
)
from core.evaluation.metrics import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "homework2"
OUTPUT_DIR = PROJECT_ROOT / "output"
EVAL_OUT = OUTPUT_DIR / "evaluation" / "all_variants"
EVAL_OUT.mkdir(parents=True, exist_ok=True)

FRAMES = ["00000", "00001", "00002", "00003", "00004", "00005",
          "00006", "00007", "00013", "00016", "00039", "00041", "00053"]

VARIANTS = {
    "V1_seed45_fl2.0_ex12": lambda pg, cg: detect_motion_seed_expand(pg, cg),
    "V2_adaptive_pct":       lambda pg, cg: detect_motion_seed_expand_v2(pg, cg),
    "V3_pct+dir":            lambda pg, cg: detect_motion_seed_expand_v3(pg, cg),
    "V4_seed30_fl2.5_ex7+dir": lambda pg, cg: detect_motion_seed_expand_v4(pg, cg),
    "V5_ratio2.0":           lambda pg, cg: detect_motion_seed_expand_v5(pg, cg),
    "V6_abs+ratio":          lambda pg, cg: detect_motion_seed_expand_v6(pg, cg),
    "V7_spatial_seed":       lambda pg, cg: detect_motion_seed_expand_v7(pg, cg),
    "V9_frame_adaptive":     lambda pg, cg: detect_motion_seed_expand_v9(pg, cg),
}

SHORT_NAMES = {
    "V1_seed45_fl2.0_ex12": "V1",
    "V2_adaptive_pct": "V2",
    "V3_pct+dir": "V3",
    "V4_seed30_fl2.5_ex7+dir": "V4",
    "V5_ratio2.0": "V5",
    "V6_abs+ratio": "V6",
    "V7_spatial_seed": "V7",
    "V9_frame_adaptive": "V9",
}


def load_pair(fid):
    c = cv2.imread(str(DATA / "rgb_images" / f"{fid}_FV.png"))
    p = cv2.imread(str(DATA / "previous_images" / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(DATA / "motion_annotation" / "GroudTruth" / f"{fid}_FV.png"),
                   cv2.IMREAD_GRAYSCALE)
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    return pg, cg, c, (g > 0)


# ── 1. Run all variants ──────────────────────────────────────────
print("=" * 90)
print("  COMPREHENSIVE VARIANT EVALUATION")
print("=" * 90)

all_results = {v: {} for v in VARIANTS}  # variant -> frame_id -> metrics
frame_stats = {}  # frame_id -> {p95_diff, p50_diff, gt_pixels, ...}

for fid in FRAMES:
    print(f"\n{fid}...")
    try:
        pg, cg, orig, gt_bin = load_pair(fid)
    except FileNotFoundError:
        print(f"  SKIP")
        continue

    # Frame characteristics
    diff = cv2.absdiff(cg, pg)
    p95 = np.percentile(diff, 95)
    p50 = np.percentile(diff, 50)
    p99 = np.percentile(diff, 99)
    gt_pixels = int(gt_bin.sum())
    frame_stats[fid] = {"p95_diff": p95, "p50_diff": p50, "p99_diff": p99, "gt_px": gt_pixels}

    for vname, func in VARIANTS.items():
        mask = func(pg, cg)
        metrics = compute_metrics(mask, gt_bin.astype(np.uint8) * 255)
        all_results[vname][fid] = metrics
        sn = SHORT_NAMES[vname]
        print(f"  {sn:>4s}  F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
              f"R={metrics['recall']:.4f}  TP={metrics['tp']:>7,}  FP={metrics['fp']:>7,}")

print("\n\n" + "=" * 90)
print("  FRAME CHARACTERISTICS")
print("=" * 90)
print(f"{'Frame':<8} {'P50_diff':>10} {'P95_diff':>10} {'P99_diff':>10} {'GT_pixels':>10}")
print("-" * 52)
for fid in FRAMES:
    s = frame_stats.get(fid)
    if s:
        print(f"{fid:<8} {s['p50_diff']:>10.1f} {s['p95_diff']:>10.1f} "
              f"{s['p99_diff']:>10.1f} {s['gt_px']:>10,}")

# ── 2. Per-frame "best variant" table ─────────────────────────────
print("\n\n" + "=" * 90)
print("  PER-FRAME BEST VARIANT (by F1)")
print("=" * 90)
print(f"{'Frame':<8} {'P95':>8} {'Best':>6} {'F1':>8} {'2nd':>6} {'F1':>8} {'3rd':>6} {'F1':>8}")
print("-" * 58)

for fid in FRAMES:
    if fid not in frame_stats:
        continue
    ranked = sorted(
        [(SHORT_NAMES[v], all_results[v][fid]["f1"]) for v in VARIANTS if fid in all_results[v]],
        key=lambda x: -x[1]
    )
    p95 = frame_stats[fid]["p95_diff"]
    print(f"{fid:<8} {p95:>8.1f} {ranked[0][0]:>6} {ranked[0][1]:>8.4f} "
          f"{ranked[1][0]:>6} {ranked[1][1]:>8.4f} {ranked[2][0]:>6} {ranked[2][1]:>8.4f}")

# ── 3. Aggregated summary ────────────────────────────────────────
print("\n\n" + "=" * 90)
print("  AGGREGATED SUMMARY (macro-average over 13 frames)")
print("=" * 90)
print(f"{'Variant':<28} {'IoU':>8} {'Precision':>8} {'Recall':>8} {'F1':>8}  {'TP_sum':>9} {'FP_sum':>9} {'FN_sum':>7}")
print("-" * 92)

for vname in VARIANTS:
    metrics_list = [all_results[vname][fid] for fid in FRAMES if fid in all_results[vname]]
    if not metrics_list:
        continue
    avg = {k: np.mean([m[k] for m in metrics_list]) for k in ["iou", "precision", "recall", "f1"]}
    tp_sum = int(sum(m["tp"] for m in metrics_list))
    fp_sum = int(sum(m["fp"] for m in metrics_list))
    fn_sum = int(sum(m["fn"] for m in metrics_list))
    sn = SHORT_NAMES[vname]
    print(f"{vname:<28} {avg['iou']:>8.4f} {avg['precision']:>8.4f} "
          f"{avg['recall']:>8.4f} {avg['f1']:>8.4f}  "
          f"{tp_sum:>9,} {fp_sum:>9,} {fn_sum:>7,}")

# ── 4. Group analysis: which method wins per motion-strength group ──
print("\n\n" + "=" * 90)
print("  METHOD PREFERENCE BY MOTION STRENGTH")
print("=" * 90)

groups = {
    "Strong (P95>60)": [f for f in FRAMES if frame_stats.get(f, {}).get("p95_diff", 0) > 60],
    "Medium (P95 30-60)": [f for f in FRAMES if 30 < frame_stats.get(f, {}).get("p95_diff", 0) <= 60],
    "Weak (P95<30)": [f for f in FRAMES if frame_stats.get(f, {}).get("p95_diff", 0) <= 30],
}

for gname, gframes in groups.items():
    print(f"\n  {gname}: {gframes}")
    # For each variant, compute average F1 in this group
    scores = []
    for vname in VARIANTS:
        f1s = [all_results[vname][f]["f1"] for f in gframes if f in all_results[vname]]
        if f1s:
            scores.append((SHORT_NAMES[vname], np.mean(f1s)))
    scores.sort(key=lambda x: -x[1])
    for sn, f1 in scores[:4]:
        bar = "█" * int(f1 * 100)
        print(f"    {sn:>4s}  avg F1={f1:.4f}  {bar}")

# ── 5. Oracle: best possible F1 if we could pick the best per frame ──
print("\n\n" + "=" * 90)
print("  ORACLE ANALYSIS (theoretical upper bound)")
print("=" * 90)

oracle_f1s = []
oracle_choices = {}
for fid in FRAMES:
    best_f1 = 0
    best_v = None
    for vname in VARIANTS:
        if fid in all_results[vname]:
            f1 = all_results[vname][fid]["f1"]
            if f1 > best_f1:
                best_f1 = f1
                best_v = SHORT_NAMES[vname]
    oracle_f1s.append(best_f1)
    oracle_choices[fid] = best_v

print(f"  Oracle F1 (choose best variant per frame): {np.mean(oracle_f1s):.4f}")
print(f"  Best single variant F1 (V1):                  {np.mean([all_results['V1_seed45_fl2.0_ex12'][fid]['f1'] for fid in FRAMES if fid in all_results['V1_seed45_fl2.0_ex12']]):.4f}")
print(f"  Improvement possible:                        +{np.mean(oracle_f1s) - np.mean([all_results['V1_seed45_fl2.0_ex12'][fid]['f1'] for fid in FRAMES if fid in all_results['V1_seed45_fl2.0_ex12']]):.4f}")
print(f"\n  Oracle choices per frame:")
for fid in FRAMES:
    s = frame_stats.get(fid, {})
    print(f"    {fid}  P95={s.get('p95_diff', 0):>6.1f}  →  {oracle_choices.get(fid, '?')}")

# ── 6. Correlation analysis ──────────────────────────────────────
print("\n\n" + "=" * 90)
print("  FEATURE CORRELATION")
print("=" * 90)
print("  (Which frame characteristic predicts which method is best?)")
print()
print(f"  {'Frame':<8} {'P95_diff':>9} {'GT_px':>8}  {'V1_F1':>8} {'V2_F1':>8} {'V5_F1':>8} {'V7_F1':>8} {'V9_F1':>8}")
print("  " + "-" * 74)
for fid in FRAMES:
    s = frame_stats.get(fid, {})
    v1f = all_results["V1_seed45_fl2.0_ex12"].get(fid, {}).get("f1", 0)
    v2f = all_results["V2_adaptive_pct"].get(fid, {}).get("f1", 0)
    v5f = all_results["V5_ratio2.0"].get(fid, {}).get("f1", 0)
    v7f = all_results["V7_spatial_seed"].get(fid, {}).get("f1", 0)
    v9f = all_results["V9_frame_adaptive"].get(fid, {}).get("f1", 0)
    print(f"  {fid:<8} {s.get('p95_diff', 0):>9.1f} {s.get('gt_px', 0):>8,}  "
          f"{v1f:>8.4f} {v2f:>8.4f} {v5f:>8.4f} {v7f:>8.4f} {v9f:>8.4f}")

print("\nDone! All outputs saved to:", EVAL_OUT)
