"""方案1评估: seed_expand (V1) vs seed_expand_v2 (自适应百分位阈值)"""

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

# 使用 readme_1.md 中相同的 13 帧评测集
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
    if c is None or p is None or g is None:
        raise FileNotFoundError(f"Missing data for {fid}")
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    return pg, cg, (g > 0)


if __name__ == "__main__":
    print("=" * 130)
    print("  方案1: 自适应百分位阈值 — seed_expand (V1) vs seed_expand_v2")
    print("=" * 130)

    # ====== Per-frame detailed comparison ======
    print(f"\n{'frame':>7} | {'V1-IoU':>8} {'V1-Prec':>8} {'V1-Rec':>8} {'V1-F1':>8} {'V1-TP':>8} {'V1-FP':>8} {'V1-FN':>7}"
          f" | {'V2-IoU':>8} {'V2-Prec':>8} {'V2-Rec':>8} {'V2-F1':>8} {'V2-TP':>8} {'V2-FP':>8} {'V2-FN':>7}"
          f" | {'ΔF1':>8}")
    print("-" * 130)

    v1_all = {"iou": [], "prec": [], "rec": [], "f1": [], "tp": 0, "fp": 0, "fn": 0}
    v2_all = {"iou": [], "prec": [], "rec": [], "f1": [], "tp": 0, "fp": 0, "fn": 0}

    for fid in FRAMES:
        pg, cg, gb = load_pair(fid)

        # V1: original seed_expand
        m1 = detect_motion_seed_expand(pg, cg)
        s1 = score(m1, gb)
        # V2: adaptive percentile
        m2 = detect_motion_seed_expand_v2(pg, cg)
        s2 = score(m2, gb)

        for k in ["iou", "prec", "rec", "f1"]:
            v1_all[k].append(s1[k])
            v2_all[k].append(s2[k])
        for k in ["tp", "fp", "fn"]:
            v1_all[k] += s1[k]
            v2_all[k] += s2[k]

        delta_f1 = s2["f1"] - s1["f1"]
        marker = " **" if delta_f1 > 0.005 else (" --" if delta_f1 < -0.005 else "")
        print(f"  {fid:>5} | {s1['iou']:>8.4f} {s1['prec']:>8.4f} {s1['rec']:>8.4f} {s1['f1']:>8.4f} {s1['tp']:>8} {s1['fp']:>8} {s1['fn']:>7}"
              f" | {s2['iou']:>8.4f} {s2['prec']:>8.4f} {s2['rec']:>8.4f} {s2['f1']:>8.4f} {s2['tp']:>8} {s2['fp']:>8} {s2['fn']:>7}"
              f" | {delta_f1:>+8.4f}{marker}")

    # ====== Macro-average ======
    v1_macro = {k: np.mean(v) for k, v in v1_all.items() if k in ["iou", "prec", "rec", "f1"]}
    v2_macro = {k: np.mean(v) for k, v in v2_all.items() if k in ["iou", "prec", "rec", "f1"]}

    # ====== Micro-average (global) ======
    def micro(tp, fp, fn):
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        return {"iou": iou, "prec": prec, "rec": rec, "f1": f1}

    v1_micro = micro(v1_all["tp"], v1_all["fp"], v1_all["fn"])
    v2_micro = micro(v2_all["tp"], v2_all["fp"], v2_all["fn"])

    print("-" * 130)

    # ====== Summary ======
    print(f"\n{'='*70}")
    print(f"  汇总对比")
    print(f"{'='*70}")
    print(f"  {'指标':<12} {'V1 (原始)':>12} {'V2 (方案1)':>12} {'Δ':>12} {'变化%':>10}")
    print(f"  {'-'*60}")

    for metric in ["iou", "prec", "rec", "f1"]:
        v1v = v1_macro[metric]
        v2v = v2_macro[metric]
        delta = v2v - v1v
        pct = (delta / v1v * 100) if v1v > 0 else 0
        label = {"iou": "IoU (宏平均)", "prec": "Precision", "rec": "Recall", "f1": "F1"}[metric]
        print(f"  {label:<12} {v1v:>12.4f} {v2v:>12.4f} {delta:>+12.4f} {pct:>+9.1f}%")

    print(f"\n  --- 微观平均 (全局TP/FP/FN) ---")
    for metric in ["iou", "prec", "rec", "f1"]:
        v1v = v1_micro[metric]
        v2v = v2_micro[metric]
        delta = v2v - v1v
        pct = (delta / v1v * 100) if v1v > 0 else 0
        label = {"iou": "IoU (微平均)", "prec": "Precision", "rec": "Recall", "f1": "F1"}[metric]
        print(f"  {label:<12} {v1v:>12.4f} {v2v:>12.4f} {delta:>+12.4f} {pct:>+9.1f}%")

    # ====== FP reduction analysis ======
    print(f"\n  --- FP 降低分析 ---")
    print(f"  V1 总FP: {v1_all['fp']:>10,}")
    print(f"  V2 总FP: {v2_all['fp']:>10,}")
    fp_reduction = (v1_all['fp'] - v2_all['fp']) / v1_all['fp'] * 100 if v1_all['fp'] > 0 else 0
    print(f"  FP 减少: {v1_all['fp'] - v2_all['fp']:>10,}  ({fp_reduction:+.1f}%)")

    # ====== Per-frame FP/TP ratio ======
    print(f"\n  --- 逐帧 FP/TP 比值变化 ---")
    print(f"  {'frame':>7} {'V1-FP/TP':>10} {'V2-FP/TP':>10} {'改善':>10}")
    print(f"  {'-'*40}")
    for fid in FRAMES:
        pg, cg, gb = load_pair(fid)
        m1 = detect_motion_seed_expand(pg, cg)
        m2 = detect_motion_seed_expand_v2(pg, cg)
        s1 = score(m1, gb)
        s2 = score(m2, gb)
        r1 = s1['fp'] / s1['tp'] if s1['tp'] > 0 else float('inf')
        r2 = s2['fp'] / s2['tp'] if s2['tp'] > 0 else float('inf')
        imp = f"{(r1-r2)/r1*100:+.1f}%" if r1 != float('inf') and r1 > 0 else "N/A"
        r1s = f"{r1:.1f}" if r1 != float('inf') else "inf"
        r2s = f"{r2:.1f}" if r2 != float('inf') else "inf"
        print(f"  {fid:>7} {r1s:>10} {r2s:>10} {imp:>10}")

    print(f"\n{'='*70}")
    print("  方案1评估完成!")
