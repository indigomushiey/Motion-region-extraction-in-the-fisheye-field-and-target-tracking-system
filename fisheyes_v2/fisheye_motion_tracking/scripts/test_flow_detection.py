import cv2
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data" / "homework2"


def detect_flow_mask(prev_gray, curr_gray, mag_thresh=2.5, kernel_size=9):
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.linalg.norm(flow, axis=2)
    _, mask = cv2.threshold(mag, mag_thresh, 255, cv2.THRESH_BINARY)
    mask = mask.astype(np.uint8)
    k = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def detect_frame_diff(prev_gray, curr_gray, thresh=40, kernel_size=5):
    diff = cv2.absdiff(curr_gray, prev_gray)
    _, mask = cv2.threshold(diff, thresh, 255, cv2.THRESH_BINARY)
    k = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def score(pred, gt_bin):
    p = pred > 0
    tp = (p & gt_bin).sum()
    fp = (p & ~gt_bin).sum()
    fn = (~p & gt_bin).sum()
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return iou, prec, rec, f1, tp, fp, fn


if __name__ == "__main__":
    frames = ["00000", "00001", "00002", "00003", "00004", "00005",
              "00006", "00007", "00013", "00016", "00039", "00041", "00053"]

    print(f"{'frame':>7} | {'Baseline (diff th=40)':>32} | {'Flow (mag>2.5 k9)':>32} |")
    print(f"{'':>7} | {'IoU':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} | {'IoU':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} | {'dF1':>7}")
    print("-" * 85)

    sum_b_f1 = 0
    sum_f_f1 = 0

    for fid in frames:
        c = cv2.imread(str(DATA / "rgb_images" / f"{fid}_FV.png"))
        p = cv2.imread(str(DATA / "previous_images" / f"{fid}_FV_prev.png"))
        g = cv2.imread(str(DATA / "motion_annotation" / "GroudTruth" / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
        gb = g > 0

        cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        cg = cv2.GaussianBlur(cg, (5, 5), 0)
        pg = cv2.GaussianBlur(pg, (5, 5), 0)

        m1 = detect_frame_diff(pg, cg)
        iou1, prec1, rec1, f1_1, _, _, _ = score(m1, gb)

        m2 = detect_flow_mask(pg, cg, mag_thresh=2.5, kernel_size=9)
        iou2, prec2, rec2, f1_2, _, _, _ = score(m2, gb)

        sum_b_f1 += f1_1
        sum_f_f1 += f1_2

        print(f"{fid:>7} | {iou1:>6.3f} {prec1:>6.3f} {rec1:>6.3f} {f1_1:>6.3f} | "
              f"{iou2:>6.3f} {prec2:>6.3f} {rec2:>6.3f} {f1_2:>6.3f} | {f1_2 - f1_1:>+7.3f}")

    n = len(frames)
    print("-" * 85)
    print(f"{'avg':>7} | {'':>6} {'':>6} {'':>6} {sum_b_f1/n:>6.3f} | "
          f"{'':>6} {'':>6} {'':>6} {sum_f_f1/n:>6.3f} | {sum_f_f1/n - sum_b_f1/n:>+7.3f}")
