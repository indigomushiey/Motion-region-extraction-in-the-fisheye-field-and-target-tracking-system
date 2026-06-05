"""方案1 参数扫描: 找到最佳百分位组合"""

import cv2
import numpy as np
from pathlib import Path
import sys
import itertools
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.motion_detection.frame_difference import detect_motion_seed_expand_v2

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


# Preload all frame pairs
print("Loading frames...")
frame_pairs = [(fid,) + load_pair(fid) for fid in FRAMES]
print("Done.\n")

# Parameter grid
seed_pcts = [85, 88, 90, 92, 95]
flow_pcts = [80, 85, 88, 90, 93]
expand_dists = [8, 10, 12, 15]

# Baseline: V1 raw
def eval_v1():
    from core.motion_detection.frame_difference import detect_motion_seed_expand
    results = []
    for fid, pg, cg, gb in frame_pairs:
        m = detect_motion_seed_expand(pg, cg)
        results.append(score(m, gb))
    macro = {k: np.mean([r[k] for r in results]) for k in ["iou", "prec", "rec", "f1"]}
    tp = sum(r["tp"] for r in results)
    fp = sum(r["fp"] for r in results)
    fn = sum(r["fn"] for r in results)
    micro_f1 = 2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) > 0 else 0
    return macro["f1"], macro["prec"], macro["rec"], fp, micro_f1

v1_f1, v1_prec, v1_rec, v1_fp, v1_micro_f1 = eval_v1()
print(f"V1 Baseline: F1={v1_f1:.4f}, Prec={v1_prec:.4f}, Rec={v1_rec:.4f}, FP={v1_fp:,}, microF1={v1_micro_f1:.4f}")
print()

# Grid search
best = {"f1": 0, "params": None, "prec": 0, "rec": 0, "fp": 0}
results_grid = []

print(f"{'seed%':>7} {'flow%':>7} {'expand':>7} {'F1':>8} {'Prec':>8} {'Rec':>8} {'FP':>10} {'vsV1-F1':>10}")
print("-" * 75)

for sp, fpct, ed in itertools.product(seed_pcts, flow_pcts, expand_dists):
    all_scores = []
    for fid, pg, cg, gb in frame_pairs:
        m = detect_motion_seed_expand_v2(pg, cg,
                                          seed_percentile=sp,
                                          flow_percentile=fpct,
                                          expand_dist=ed)
        all_scores.append(score(m, gb))

    macro_f1 = np.mean([s["f1"] for s in all_scores])
    macro_prec = np.mean([s["prec"] for s in all_scores])
    macro_rec = np.mean([s["rec"] for s in all_scores])
    total_fp = sum(s["fp"] for s in all_scores)

    delta = macro_f1 - v1_f1
    marker = " <--" if macro_f1 > best["f1"] else ""
    print(f"{sp:>7} {fpct:>7} {ed:>7} {macro_f1:>8.4f} {macro_prec:>8.4f} {macro_rec:>8.4f} {total_fp:>10,} {delta:>+10.4f}{marker}")

    if macro_f1 > best["f1"]:
        best = {"f1": macro_f1, "params": (sp, fpct, ed), "prec": macro_prec, "rec": macro_rec, "fp": total_fp}

# Best result
print(f"\n{'='*60}")
print(f"Best params: seed_percentile={best['params'][0]}, flow_percentile={best['params'][1]}, expand_dist={best['params'][2]}")
print(f"Best F1={best['f1']:.4f} (V1={v1_f1:.4f}, delta={best['f1']-v1_f1:+.4f})")
print(f"Best Prec={best['prec']:.4f} (V1={v1_prec:.4f})")
print(f"Best Rec={best['rec']:.4f} (V1={v1_rec:.4f})")
print(f"Best FP={best['fp']:,} (V1={v1_fp:,}, reduction={(v1_fp-best['fp'])/v1_fp*100:+.1f}%)")
print(f"\nSuggested v2 call:")
print(f"  detect_motion_seed_expand_v2(pg, cg, seed_percentile={best['params'][0]}, flow_percentile={best['params'][1]}, expand_dist={best['params'][2]})")
