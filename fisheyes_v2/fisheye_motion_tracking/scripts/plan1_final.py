"""方案1最终评估: 使用最佳参数对比 V1 vs V2"""

import cv2
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.motion_detection.frame_difference import (
    detect_motion_seed_expand,
    detect_motion_seed_expand_v2,
)

DATA = Path(__file__).resolve().parents[1] / "data" / "homework2"
FRAMES = ["00000", "00001", "00002", "00003", "00004", "00005",
          "00006", "00007", "00013", "00016", "00039", "00041", "00053"]


def score(pred, gt_bin):
    p = pred > 0
    tp = int((p & gt_bin).sum())
    fp = int((p & ~gt_bin).sum())
    fn = int((~p & gt_bin).sum())
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"iou": iou, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def load_pair(fid):
    c = cv2.imread(str(DATA / "rgb_images" / f"{fid}_FV.png"))
    p = cv2.imread(str(DATA / "previous_images" / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(DATA / "motion_annotation" / "GroudTruth" / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    return pg, cg, (g > 0)


# Best params from grid search
BEST_SP = 85
BEST_FPCT = 90
BEST_ED = 15

print("=" * 115)
print("  Plan 1 Final: seed_expand V1 vs V2 (best params)")
print(f"  V2 params: seed_percentile={BEST_SP}, flow_percentile={BEST_FPCT}, expand_dist={BEST_ED}")
print("=" * 115)

print(f"\n  {'frame':>6} | {'V1-F1':>8} {'V1-Prec':>8} {'V1-Rec':>8}  {'V1-FP':>9}"
      f" | {'V2-F1':>8} {'V2-Prec':>8} {'V2-Rec':>8}  {'V2-FP':>9}"
      f" | {'dF1':>8} {'FP-':>8}")
print("  " + "-" * 107)

v1_scores = []
v2_scores = []

for fid in FRAMES:
    pg, cg, gb = load_pair(fid)
    m1 = detect_motion_seed_expand(pg, cg)
    m2 = detect_motion_seed_expand_v2(pg, cg, seed_percentile=BEST_SP,
                                      flow_percentile=BEST_FPCT, expand_dist=BEST_ED)
    s1 = score(m1, gb)
    s2 = score(m2, gb)
    v1_scores.append(s1)
    v2_scores.append(s2)

    d_f1 = s2["f1"] - s1["f1"]
    fp_red = (s1["fp"] - s2["fp"]) / s1["fp"] * 100 if s1["fp"] > 0 else 0
    sym = "+" if d_f1 > 0.005 else ("-" if d_f1 < -0.005 else "~")
    print(f"  {fid:>6} | {s1['f1']:>8.4f} {s1['prec']:>8.4f} {s1['rec']:>8.4f} {s1['fp']:>9,}"
          f" | {s2['f1']:>8.4f} {s2['prec']:>8.4f} {s2['rec']:>8.4f} {s2['fp']:>9,}"
          f" | {d_f1:>+8.4f} {fp_red:>+7.1f}%  {sym}")

# Macro averages
v1_macro = {k: np.mean([s[k] for s in v1_scores]) for k in ["iou", "prec", "rec", "f1"]}
v2_macro = {k: np.mean([s[k] for s in v2_scores]) for k in ["iou", "prec", "rec", "f1"]}

# Totals
v1_tp = sum(s["tp"] for s in v1_scores)
v1_fp = sum(s["fp"] for s in v1_scores)
v1_fn = sum(s["fn"] for s in v1_scores)
v2_tp = sum(s["tp"] for s in v2_scores)
v2_fp = sum(s["fp"] for s in v2_scores)
v2_fn = sum(s["fn"] for s in v2_scores)

print("  " + "-" * 107)

# Summary block
print(f"""
  {'='*55}
    Summary
  {'='*55}
    {'Metric':<16} {'V1 (original)':>14} {'V2 (Plan 1)':>14} {'Delta':>14}
    {'-'*55}""")

for key, label in [("f1", "F1 (macro avg)"), ("prec", "Precision"), ("rec", "Recall"), ("iou", "IoU")]:
    d = v2_macro[key] - v1_macro[key]
    print(f"    {label:<16} {v1_macro[key]:>14.4f} {v2_macro[key]:>14.4f} {d:>+14.4f}")

# Micro stats
v1_micro_prec = v1_tp / (v1_tp + v1_fp) if (v1_tp + v1_fp) > 0 else 0
v1_micro_rec = v1_tp / (v1_tp + v1_fn) if (v1_tp + v1_fn) > 0 else 0
v1_micro_f1 = 2*v1_micro_prec*v1_micro_rec/(v1_micro_prec+v1_micro_rec) if (v1_micro_prec+v1_micro_rec) > 0 else 0
v2_micro_prec = v2_tp / (v2_tp + v2_fp) if (v2_tp + v2_fp) > 0 else 0
v2_micro_rec = v2_tp / (v2_tp + v2_fn) if (v2_tp + v2_fn) > 0 else 0
v2_micro_f1 = 2*v2_micro_prec*v2_micro_rec/(v2_micro_prec+v2_micro_rec) if (v2_micro_prec+v2_micro_rec) > 0 else 0

print(f"""
    --- Global (micro) ---
    {'F1 (micro avg)':<16} {v1_micro_f1:>14.4f} {v2_micro_f1:>14.4f} {v2_micro_f1-v1_micro_f1:>+14.4f}
    {'Precision':<16} {v1_micro_prec:>14.4f} {v2_micro_prec:>14.4f} {v2_micro_prec-v1_micro_prec:>+14.4f}
    {'Recall':<16} {v1_micro_rec:>14.4f} {v2_micro_rec:>14.4f} {v2_micro_rec-v1_micro_rec:>+14.4f}

    --- Global counts ---
    {'Total TP':<16} {v1_tp:>14,} {v2_tp:>14,}
    {'Total FP':<16} {v1_fp:>14,} {v2_fp:>14,}  ({(v1_fp-v2_fp)/v1_fp*100:+.1f}%)
    {'Total FN':<16} {v1_fn:>14,} {v2_fn:>14,}

  {'='*55}
    Conclusion: Plan 1 reduces FP by {(v1_fp-v2_fp)/v1_fp*100:.1f}% and improves Precision
    by {(v2_macro['prec']/v1_macro['prec']-1)*100:.1f}%, but at the cost of {(v1_macro['rec']-v2_macro['rec'])/v1_macro['rec']*100:.1f}% Recall loss.
    F1 change: {v2_macro['f1']-v1_macro['f1']:+.4f} (macro), {v2_micro_f1-v1_micro_f1:+.4f} (micro).
  {'='*55}
""")
