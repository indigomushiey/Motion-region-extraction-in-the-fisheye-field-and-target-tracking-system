import numpy as np


def compute_metrics(pred_mask, gt_mask):
    """Compute Precision, Recall, F1-score for binary mask.

    TP = pixels where pred>0 AND gt>0
    FP = pixels where pred>0 AND gt==0
    FN = pixels where pred==0 AND gt>0

    Returns dict with iou, precision, recall, f1, tp, fp, fn
    """
    pred_bin = (pred_mask > 0).astype(np.uint8)
    gt_bin = (gt_mask > 0).astype(np.uint8)

    tp = int(np.logical_and(pred_bin, gt_bin).sum())
    fp = int(np.logical_and(pred_bin, 1 - gt_bin).sum())
    fn = int(np.logical_and(1 - pred_bin, gt_bin).sum())

    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "iou": round(iou, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }
