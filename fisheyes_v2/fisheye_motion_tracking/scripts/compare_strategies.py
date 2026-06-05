"""Compare motion detection strategies across multiple frames to find the best approach."""

import cv2
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "homework2"

FRAMES = ["00000", "00001", "00002", "00003", "00004", "00005",
          "00006", "00007", "00013", "00016", "00039", "00041", "00053"]


def score(pred, gt_bin):
    p = pred > 0
    tp = (p & gt_bin).sum()
    fp = (p & ~gt_bin).sum()
    fn = (~p & gt_bin).sum()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return {"iou": iou, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def morph(mask, ks, iters=1):
    k = np.ones((ks, ks), np.uint8)
    for _ in range(iters):
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def load_pair(fid):
    c = cv2.imread(str(DATA / "rgb_images" / f"{fid}_FV.png"))
    p = cv2.imread(str(DATA / "previous_images" / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(DATA / "motion_annotation" / "GroudTruth" / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    cg = cv2.GaussianBlur(cg, (5, 5), 0)
    pg = cv2.GaussianBlur(pg, (5, 5), 0)
    return pg, cg, (g > 0)


def compute_flow(pg, cg):
    f = cv2.calcOpticalFlowFarneback(pg, cg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.linalg.norm(f, axis=2)
    return f, mag


# ============ strategy definitions ============

def strat_diff_fixed(pg, cg, th=40, ks=5):
    diff = cv2.absdiff(cg, pg)
    _, m = cv2.threshold(diff, th, 255, cv2.THRESH_BINARY)
    return morph(m.astype(np.uint8), ks)


def strat_diff_otsu(pg, cg, ks=7):
    diff = cv2.absdiff(cg, pg)
    # Clip extreme values for better Otsu
    p99 = np.percentile(diff, 99)
    diff_clip = np.clip(diff, 0, p99).astype(np.uint8)
    otsu, _ = cv2.threshold(diff_clip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, m = cv2.threshold(diff, otsu, 255, cv2.THRESH_BINARY)
    return morph(m, ks)


def strat_flow_fixed(pg, cg, mt=3.0, ks=9):
    _, mag = compute_flow(pg, cg)
    _, m = cv2.threshold(mag, mt, 255, cv2.THRESH_BINARY)
    return morph(m.astype(np.uint8), ks)


def strat_flow_percentile(pg, cg, pct=90, ks=9):
    """Adaptive threshold at given percentile of flow magnitude."""
    _, mag = compute_flow(pg, cg)
    th = np.percentile(mag, pct)
    _, m = cv2.threshold(mag, th, 255, cv2.THRESH_BINARY)
    return morph(m.astype(np.uint8), ks)


def strat_flow_direction(pg, cg, mt=3.0, angle_grad_max=0.5, ks=9):
    """Flow magnitude + direction consistency."""
    flow, mag = compute_flow(pg, cg)
    ang = np.arctan2(flow[..., 1], flow[..., 0])
    gx = cv2.Sobel(ang, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(ang, cv2.CV_32F, 0, 1, ksize=3)
    ang_grad = np.sqrt(gx ** 2 + gy ** 2)
    _, mag_mask = cv2.threshold(mag, mt, 255, cv2.THRESH_BINARY)
    dir_mask = (ang_grad < angle_grad_max).astype(np.uint8) * 255
    m = mag_mask.astype(np.uint8) & dir_mask
    return morph(m, ks)


def strat_hybrid_and(pg, cg, diff_th=50, flow_mt=2.5, ks=7):
    """Intersection of frame diff and flow."""
    diff = cv2.absdiff(cg, pg)
    _, dm = cv2.threshold(diff, diff_th, 255, cv2.THRESH_BINARY)
    _, mag = compute_flow(pg, cg)
    _, fm = cv2.threshold(mag, flow_mt, 255, cv2.THRESH_BINARY)
    m = dm & fm.astype(np.uint8)
    return morph(m, ks)


if __name__ == "__main__":
    strategies = [
        ("diff th=40 k5 (baseline)", lambda pg, cg: strat_diff_fixed(pg, cg, 40, 5)),
        ("diff th=50 k7", lambda pg, cg: strat_diff_fixed(pg, cg, 50, 7)),
        ("diff th=60 k7", lambda pg, cg: strat_diff_fixed(pg, cg, 60, 7)),
        ("diff Otsu k7", lambda pg, cg: strat_diff_otsu(pg, cg, 7)),
        ("flow mag>2.0 k9", lambda pg, cg: strat_flow_fixed(pg, cg, 2.0, 9)),
        ("flow mag>3.0 k9", lambda pg, cg: strat_flow_fixed(pg, cg, 3.0, 9)),
        ("flow mag>4.0 k9", lambda pg, cg: strat_flow_fixed(pg, cg, 4.0, 9)),
        ("flow p90 k9", lambda pg, cg: strat_flow_percentile(pg, cg, 90, 9)),
        ("flow p95 k9", lambda pg, cg: strat_flow_percentile(pg, cg, 95, 9)),
        ("flow mag>2.5 + dir k9", lambda pg, cg: strat_flow_direction(pg, cg, 2.5, 0.5, 9)),
        ("hybrid AND(th50,mag2.5) k7", lambda pg, cg: strat_hybrid_and(pg, cg, 50, 2.5, 7)),
    ]

    # header
    header = f"{'strategy':<35} |"
    for fid in FRAMES:
        header += f" {fid:>7} |"
    header += f" {'avg':>7}"
    print(header)
    print("-" * len(header))

    for name, func in strategies:
        row = f"{name:<35} |"
        f1_vals = []
        for fid in FRAMES:
            pg, cg, gb = load_pair(fid)
            m = func(pg, cg)
            s = score(m, gb)
            f1_vals.append(s["f1"])
            row += f" {s['f1']:>7.4f} |"
        row += f" {np.mean(f1_vals):>7.4f}"
        print(row)
