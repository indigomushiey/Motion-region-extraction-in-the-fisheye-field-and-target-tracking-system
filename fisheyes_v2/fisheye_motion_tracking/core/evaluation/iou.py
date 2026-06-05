import numpy as np


def mask_iou(pred_mask, gt_mask):
    """Compute IoU between two binary masks.

    Args:
        pred_mask, gt_mask: (H,W) uint8, 255 or >0 for foreground
    Returns:
        iou: float in [0,1], 1.0 when both masks are empty
    """
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()

    return 1.0 if union == 0 else float(inter / union)
