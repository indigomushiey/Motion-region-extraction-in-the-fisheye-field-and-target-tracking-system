import cv2
import numpy as np
from pathlib import Path

from core.fisheye.radial_poly import RadialPolyCamera
from core.evaluation.metrics import compute_metrics
from core.motion_detection.frame_difference import (
    detect_motion_diff,
    detect_motion_flow,
    detect_motion_hybrid,
    detect_motion_adaptive,
    detect_motion_seed_expand,
    detect_motion_seed_expand_v2,
    detect_motion_seed_expand_v3,
    detect_motion_seed_expand_v4,
    detect_motion_seed_expand_v5,
    detect_motion_seed_expand_v6,
    detect_motion_seed_expand_v7,
    detect_motion_seed_expand_v8,
    detect_motion_seed_expand_v9,
    detect_motion_seed_expand_v10,
    detect_motion_seed_expand_v11,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "homework2"
OUTPUT_DIR = PROJECT_ROOT / "output"
GT_DIR = DATA_DIR / "motion_annotation" / "GroudTruth"
UNDIST_DIR = OUTPUT_DIR / "undistorted"
CALIB_DIR = DATA_DIR / "calibration_data"
CURRENT_DIR = DATA_DIR / "rgb_images"
PREVIOUS_DIR = DATA_DIR / "previous_images"


def _make_region_masks(h, w, cx, cy, max_radius, center_ratio=0.4, edge_ratio=0.6):
    """Create binary masks for center and edge regions based on radial distance.

    Returns:
        center_mask: pixels where r < center_ratio * max_radius
        edge_mask:   pixels where r > edge_ratio * max_radius
    """
    yy, xx = np.ogrid[:h, :w]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    center_mask = (dist < center_ratio * max_radius).astype(np.uint8) * 255
    edge_mask = (dist > edge_ratio * max_radius).astype(np.uint8) * 255

    return center_mask, edge_mask


def evaluate_single(frame_id, method="diff", diff_thresh=40, morph_kernel=5, center_ratio=0.4, edge_ratio=0.6):
    """Run full evaluation for a single frame pair, both routes.

    Args:
        frame_id: e.g. '00004'
        diff_thresh: threshold for frame difference
        morph_kernel: morphology kernel size

    Returns:
        dict with fisheye_metrics, remap_metrics, both split by all/center/edge
    """
    # ---------- load data ----------
    curr_path = CURRENT_DIR / f"{frame_id}_FV.png"
    prev_path = PREVIOUS_DIR / f"{frame_id}_FV_prev.png"
    gt_path = GT_DIR / f"{frame_id}_FV.png"
    calib_path = CALIB_DIR / f"{frame_id}_FV.json"
    undist_curr_path = UNDIST_DIR / f"{frame_id}_FV.png"
    undist_prev_path = UNDIST_DIR / f"{frame_id}_FV_prev.png"  # may not exist

    curr = cv2.imread(str(curr_path))
    prev = cv2.imread(str(prev_path))
    gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)

    if curr is None or prev is None or gt_mask is None:
        raise FileNotFoundError(f"Missing data for {frame_id}")

    h, w = curr.shape[:2]
    cx, cy = w / 2, h / 2
    max_radius = np.sqrt((w / 2) ** 2 + (h / 2) ** 2)

    # region masks (same coordinate system for both routes)
    center_mask, edge_mask = _make_region_masks(h, w, cx, cy, max_radius, center_ratio, edge_ratio)

    # ---------- route A: fisheye domain ----------
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

    if method == "flow":
        fisheye_mask = detect_motion_flow(prev_gray, curr_gray, mag_thresh=3.0, morph_ksize=9)
    elif method == "hybrid":
        fisheye_mask = detect_motion_hybrid(prev_gray, curr_gray, diff_thresh=50, mag_thresh=3.0, morph_ksize=7)
    elif method == "seed_expand":
        fisheye_mask = detect_motion_seed_expand(prev_gray, curr_gray)
    elif method == "seed_expand_v2":
        fisheye_mask = detect_motion_seed_expand_v2(prev_gray, curr_gray)
    elif method == "seed_expand_v3":
        fisheye_mask = detect_motion_seed_expand_v3(prev_gray, curr_gray)
    elif method == "seed_expand_v4":
        fisheye_mask = detect_motion_seed_expand_v4(prev_gray, curr_gray)
    elif method == "seed_expand_v5":
        fisheye_mask = detect_motion_seed_expand_v5(prev_gray, curr_gray)
    elif method == "seed_expand_v6":
        fisheye_mask = detect_motion_seed_expand_v6(prev_gray, curr_gray)
    elif method == "seed_expand_v7":
        fisheye_mask = detect_motion_seed_expand_v7(prev_gray, curr_gray)
    elif method == "seed_expand_v8":
        fisheye_mask = detect_motion_seed_expand_v8(prev_gray, curr_gray)
    elif method == "seed_expand_v9":
        fisheye_mask = detect_motion_seed_expand_v9(prev_gray, curr_gray)
    elif method == "seed_expand_v10":
        fisheye_mask = detect_motion_seed_expand_v10(prev_gray, curr_gray)
    elif method == "seed_expand_v11":
        fisheye_mask = detect_motion_seed_expand_v11(prev_gray, curr_gray)
    else:
        fisheye_mask = detect_motion_diff(prev_gray, curr_gray, diff_thresh, morph_ksize=morph_kernel)

    # ---------- route B: remapped domain ----------
    if not undist_curr_path.exists():
        camera = RadialPolyCamera(str(calib_path))
        undist_curr = camera.undistort(curr, fov_scale=0.5)
        undist_prev = camera.undistort(prev, fov_scale=0.5)
    else:
        undist_curr = cv2.imread(str(undist_curr_path))
        # try loading undistorted previous; if not exists, undistort on the fly
        if undist_prev_path.exists():
            undist_prev = cv2.imread(str(undist_prev_path))
        else:
            camera = RadialPolyCamera(str(calib_path))
            undist_prev = camera.undistort(prev, fov_scale=0.5)

    undist_curr_gray = cv2.cvtColor(undist_curr, cv2.COLOR_BGR2GRAY)
    undist_prev_gray = cv2.cvtColor(undist_prev, cv2.COLOR_BGR2GRAY)
    undist_curr_gray = cv2.GaussianBlur(undist_curr_gray, (5, 5), 0)
    undist_prev_gray = cv2.GaussianBlur(undist_prev_gray, (5, 5), 0)

    if method == "flow":
        remap_mask = detect_motion_flow(undist_prev_gray, undist_curr_gray, mag_thresh=3.0, morph_ksize=9)
    elif method == "hybrid":
        remap_mask = detect_motion_hybrid(undist_prev_gray, undist_curr_gray, diff_thresh=50, mag_thresh=3.0, morph_ksize=7)
    elif method == "adaptive":
        remap_mask = detect_motion_adaptive(undist_prev_gray, undist_curr_gray, morph_ksize=7)
    elif method == "seed_expand":
        remap_mask = detect_motion_seed_expand(undist_prev_gray, undist_curr_gray)
    elif method == "seed_expand_v2":
        remap_mask = detect_motion_seed_expand_v2(undist_prev_gray, undist_curr_gray)
    elif method == "seed_expand_v3":
        remap_mask = detect_motion_seed_expand_v3(undist_prev_gray, undist_curr_gray)
    elif method == "seed_expand_v4":
        remap_mask = detect_motion_seed_expand_v4(undist_prev_gray, undist_curr_gray)
    elif method == "seed_expand_v5":
        remap_mask = detect_motion_seed_expand_v5(undist_prev_gray, undist_curr_gray)
    elif method == "seed_expand_v6":
        remap_mask = detect_motion_seed_expand_v6(undist_prev_gray, undist_curr_gray)
    elif method == "seed_expand_v7":
        remap_mask = detect_motion_seed_expand_v7(undist_prev_gray, undist_curr_gray)
    elif method == "seed_expand_v8":
        remap_mask = detect_motion_seed_expand_v8(undist_prev_gray, undist_curr_gray)
    elif method == "seed_expand_v9":
        remap_mask = detect_motion_seed_expand_v9(undist_prev_gray, undist_curr_gray)
    elif method == "seed_expand_v10":
        remap_mask = detect_motion_seed_expand_v10(undist_prev_gray, undist_curr_gray)
    elif method == "seed_expand_v11":
        remap_mask = detect_motion_seed_expand_v11(undist_prev_gray, undist_curr_gray)
    else:
        remap_mask = detect_motion_diff(undist_prev_gray, undist_curr_gray, diff_thresh, morph_ksize=morph_kernel)

    # ---------- compute metrics ----------
    def metrics_in_regions(pred, gt, center, edge):
        full = compute_metrics(pred, gt)
        c = compute_metrics(pred & center, gt & center)
        e = compute_metrics(pred & edge, gt & edge)
        return {"all": full, "center": c, "edge": e}

    result = {
        "frame_id": frame_id,
        "fisheye": metrics_in_regions(fisheye_mask, gt_mask, center_mask, edge_mask),
        "remap": metrics_in_regions(remap_mask, gt_mask, center_mask, edge_mask),
    }

    return result


def print_eval_report(result):
    """Print a formatted evaluation report for a single frame."""
    fid = result["frame_id"]
    print(f"\n{'='*60}")
    print(f"  Evaluation Report — Frame {fid}")
    print(f"{'='*60}")

    for label, key in [("Fisheye Domain", "fisheye"), ("Remapped Domain", "remap")]:
        m = result[key]
        print(f"\n  [{label}]")
        print(f"  {'':>8} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        for region in ["all", "center", "edge"]:
            r = m[region]
            print(f"  {region:>8} {r['iou']:>8.4f} {r['precision']:>8.4f} "
                  f"{r['recall']:>8.4f} {r['f1']:>8.4f}")

    # delta comparison
    print(f"\n  [Delta: Remap - Fisheye]")
    print(f"  {'':>8} {'ΔIoU':>8} {'ΔPrec':>8} {'ΔRecall':>8} {'ΔF1':>8}")
    for region in ["all", "center", "edge"]:
        d_iou = result["remap"][region]["iou"] - result["fisheye"][region]["iou"]
        d_prec = result["remap"][region]["precision"] - result["fisheye"][region]["precision"]
        d_rec = result["remap"][region]["recall"] - result["fisheye"][region]["recall"]
        d_f1 = result["remap"][region]["f1"] - result["fisheye"][region]["f1"]
        print(f"  {region:>8} {d_iou:>+8.4f} {d_prec:>+8.4f} {d_rec:>+8.4f} {d_f1:>+8.4f}")

    print(f"\n{'='*60}\n")


def _format_report_text(result):
    """Build the evaluation report as a string for saving to file."""
    fid = result["frame_id"]
    lines = []
    lines.append("=" * 60)
    lines.append(f"  Evaluation Report — Frame {fid}")
    lines.append("=" * 60)

    for label, key in [("Fisheye Domain", "fisheye"), ("Remapped Domain", "remap")]:
        m = result[key]
        lines.append(f"\n  [{label}]")
        lines.append(f"  {'':>8} {'IoU':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}  {'TP':>7} {'FP':>7} {'FN':>7}")
        for region in ["all", "center", "edge"]:
            r = m[region]
            lines.append(f"  {region:>8} {r['iou']:>8.4f} {r['precision']:>8.4f} "
                         f"{r['recall']:>8.4f} {r['f1']:>8.4f}  "
                         f"{r['tp']:>7} {r['fp']:>7} {r['fn']:>7}")

    lines.append(f"\n  [Delta: Remap - Fisheye]")
    lines.append(f"  {'':>8} {'ΔIoU':>8} {'ΔPrec':>8} {'ΔRecall':>8} {'ΔF1':>8}")
    for region in ["all", "center", "edge"]:
        d_iou = result["remap"][region]["iou"] - result["fisheye"][region]["iou"]
        d_prec = result["remap"][region]["precision"] - result["fisheye"][region]["precision"]
        d_rec = result["remap"][region]["recall"] - result["fisheye"][region]["recall"]
        d_f1 = result["remap"][region]["f1"] - result["fisheye"][region]["f1"]
        lines.append(f"  {region:>8} {d_iou:>+8.4f} {d_prec:>+8.4f} {d_rec:>+8.4f} {d_f1:>+8.4f}")

    lines.append(f"\n{'=' * 60}")
    return "\n".join(lines)


def save_eval_result(result, output_dir=None):
    """Save evaluation report (txt), comparison visualization (png), and raw masks.

    Subdirectories under output/:
        evaluation/eval_comparison/  — visual comparison figures
        evaluation/reports/          — text evaluation reports
        masks/                       — raw fisheye/remap motion masks
    """
    fid = result["frame_id"]
    base = Path(output_dir) if output_dir else OUTPUT_DIR

    # Subdirectories
    out_eval_comp = base / "evaluation" / "eval_comparison"
    out_reports = base / "evaluation" / "reports"
    out_masks = base / "masks"
    for d in (out_eval_comp, out_reports, out_masks):
        d.mkdir(parents=True, exist_ok=True)

    # --- 1. text report ---
    txt = _format_report_text(result)
    txt_path = out_reports / f"{fid}_eval_report.txt"
    txt_path.write_text(txt, encoding="utf-8")
    print(f"  Saved: {txt_path}")

    # --- 2. visual comparison ---
    curr = cv2.imread(str(CURRENT_DIR / f"{fid}_FV.png"))
    prev = cv2.imread(str(PREVIOUS_DIR / f"{fid}_FV_prev.png"))
    gt = cv2.imread(str(GT_DIR / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)

    # Regenerate masks (same logic as evaluate_single)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.GaussianBlur(curr_gray, (5, 5), 0)
    prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)
    f_mask = detect_motion_diff(prev_gray, curr_gray)

    # Remap route
    undist_curr_path = UNDIST_DIR / f"{fid}_FV.png"
    camera = RadialPolyCamera(str(CALIB_DIR / f"{fid}_FV.json"))
    if undist_curr_path.exists():
        undist_curr = cv2.imread(str(undist_curr_path))
    else:
        undist_curr = camera.undistort(curr, fov_scale=0.5)
    undist_prev = camera.undistort(prev, fov_scale=0.5)
    ug = cv2.cvtColor(undist_curr, cv2.COLOR_BGR2GRAY)
    up = cv2.cvtColor(undist_prev, cv2.COLOR_BGR2GRAY)
    ug = cv2.GaussianBlur(ug, (5, 5), 0)
    up = cv2.GaussianBlur(up, (5, 5), 0)
    r_mask = detect_motion_diff(up, ug)

    # Build comparison image
    h = 400
    def rs(img, hh=h):
        r = hh / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * r), hh))

    # Color-code masks: green=TP, red=FP, blue=FN
    def color_error(pred, gt_bin, base_img=None):
        pred_bin = (pred > 0)
        gt_b = (gt_bin > 0)
        tp = pred_bin & gt_b      # green — correct detection
        fp = pred_bin & ~gt_b     # red — false alarm
        fn = ~pred_bin & gt_b     # blue — missed detection
        out = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        if base_img is not None:
            gray = cv2.cvtColor(base_img, cv2.COLOR_BGR2GRAY)
            out[:] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) * 0.3
        out[tp] = [0, 255, 0]   # green
        out[fp] = [0, 0, 255]   # red
        out[fn] = [255, 0, 0]   # blue
        return out.astype(np.uint8)

    f_err = color_error(f_mask, gt)
    r_err = color_error(r_mask, gt)
    gt_bgr = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    def label(img, text):
        cv2.putText(img, text, (8, 28), font, 0.6, (255, 255, 255), 2)

    # Top row: original | undistorted | GT
    top = np.hstack([rs(curr), rs(undist_curr), rs(gt_bgr)])
    label(top, "Original"); label(top[:, rs(curr).shape[1]:], "Undistorted")
    label(top[:, 2*rs(curr).shape[1]:], "Ground Truth")

    # Bottom row: fisheye_error | remap_error | legend
    legend = np.zeros((h, rs(curr).shape[1], 3), dtype=np.uint8)
    legend[:] = [30, 30, 30]
    items = [("Green = TP (correct)", (0, 255, 0)),
             ("Red   = FP (false alarm)", (0, 0, 255)),
             ("Blue  = FN (missed)", (255, 0, 0))]
    for i, (txt, col) in enumerate(items):
        cv2.putText(legend, txt, (20, 40 + i * 35), font, 0.5, col, 2)

    row_f = rs(f_err)
    row_r = rs(r_err)
    label(row_f, "Fisheye errors"); label(row_r, "Remap errors")
    bot = np.hstack([row_f, row_r, legend])

    viz = np.vstack([top, bot])
    viz_path = out_eval_comp / f"{fid}_eval_comparison.png"
    cv2.imwrite(str(viz_path), viz)
    print(f"  Saved: {viz_path}")

    # --- 3. save raw masks ---
    cv2.imwrite(str(out_masks / f"{fid}_fisheye_mask.png"), f_mask)
    cv2.imwrite(str(out_masks / f"{fid}_remap_mask.png"), r_mask)
    print(f"  Saved: {out_masks / f'{fid}_fisheye_mask.png'}")
    print(f"  Saved: {out_masks / f'{fid}_remap_mask.png'}")


if __name__ == "__main__":
    import sys
    frames = sys.argv[1:] if len(sys.argv) > 1 else ["00000", "00001", "00002", "00003", "00004"]
    print(f"Running v2 evaluation on {len(frames)} frames: {frames}")
    for fid in frames:
        try:
            result = evaluate_single(fid, method="diff")
            print_eval_report(result)
            save_eval_result(result)
        except FileNotFoundError as e:
            print(f"  SKIP {fid}: {e}")
    print("\nDone!")
