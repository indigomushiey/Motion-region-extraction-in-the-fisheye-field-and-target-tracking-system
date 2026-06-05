"""Fast in-process grid search for V13 — tests parameter combinations directly.

Uses V13_PARAMS module-level overrides in frame_difference.py to test
different parameter values without subprocess or file I/O overhead.
"""

import cv2, numpy as np, sys, random, json
from pathlib import Path
from datetime import datetime
from itertools import product

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.motion_detection.frame_difference import detect_motion_seed_expand_v13, V13_PARAMS
from core.evaluation.metrics import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "homework2"
GT_DIR = DATA / "motion_annotation" / "GroudTruth"
CURR_DIR = DATA / "rgb_images"
PREV_DIR = DATA / "previous_images"
OUT_DIR = PROJECT_ROOT / "output" / "evaluation" / "v13_grid_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 20 test frames
ALL_FRAMES = ['00000', '00001', '00002', '00003', '00004', '00005',
              '00006', '00007', '00013', '00016', '00039', '00041', '00053',
              '00022', '00026', '00027', '00033', '00037', '00059', '00065']

# Preload frames
print("Preloading 20 frames...")
frames = {}
for fid in ALL_FRAMES:
    c = cv2.imread(str(CURR_DIR / f"{fid}_FV.png"))
    p = cv2.imread(str(PREV_DIR / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GT_DIR / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    frames[fid] = {
        "pg": cv2.cvtColor(p, cv2.COLOR_BGR2GRAY),
        "cg": cv2.cvtColor(c, cv2.COLOR_BGR2GRAY),
        "gt": (g > 0).astype(np.uint8) * 255,
    }
print("Done.\n")


def evaluate(params):
    """Set V13_PARAMS, run on all frames, return aggregated metrics."""
    # Clear and set new params
    V13_PARAMS.clear()
    V13_PARAMS.update(params)

    results = []
    for fid in ALL_FRAMES:
        pg = frames[fid]["pg"]
        cg = frames[fid]["cg"]
        gt = frames[fid]["gt"]
        mask = detect_motion_seed_expand_v13(pg, cg)
        m = compute_metrics(mask, gt)
        results.append(m)

    avg_f1 = np.mean([r["f1"] for r in results])
    tp_s = sum(r["tp"] for r in results)
    fp_s = sum(r["fp"] for r in results)
    fn_s = sum(r["fn"] for r in results)
    prec = tp_s / (tp_s + fp_s) if (tp_s + fp_s) > 0 else 0
    rec = tp_s / (tp_s + fn_s) if (tp_s + fn_s) > 0 else 0
    return {"f1": round(avg_f1, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "tp": tp_s, "fp": fp_s, "fn": fn_s}


# ═══════════════════════════════════════════════════════════════════
# Baseline
# ═══════════════════════════════════════════════════════════════════
print("=" * 70)
print("BASELINE (all defaults)")
V13_PARAMS.clear()
baseline = evaluate({})
print(f"  F1={baseline['f1']:.4f}  P={baseline['precision']:.4f}  "
      f"R={baseline['recall']:.4f}  TP={baseline['tp']:,}  FP={baseline['fp']:,}")

# ═══════════════════════════════════════════════════════════════════
# Param search space
# ═══════════════════════════════════════════════════════════════════
best = {"params": {}, "result": baseline}
all_results = [{"params": "baseline", "result": baseline}]

# Test 1: SNR promotion threshold (20-26)
print("\n" + "=" * 70)
print("SWEEP: snr_promote threshold")
print("=" * 70)
for v in [20, 21, 22, 23, 24, 25, 26]:
    r = evaluate({"snr_promote": v})
    print(f"  snr_promote={v}: F1={r['f1']:.4f} P={r['precision']:.4f} TP={r['tp']:,} FP={r['fp']:,}")
    all_results.append({"params": f"snr_promote={v}", "result": r})
    if r["f1"] > best["result"]["f1"]:
        best = {"params": {"snr_promote": v}, "result": r}

# Test 2: SNR promotion expand delta (0-3)
print("\n" + "=" * 70)
print("SWEEP: promote_expand delta")
print("=" * 70)
for v in [0, 1, 2, 3]:
    r = evaluate({"promote_expand": v})
    print(f"  promote_expand={v}: F1={r['f1']:.4f} P={r['precision']:.4f} TP={r['tp']:,} FP={r['fp']:,}")
    all_results.append({"params": f"promote_expand={v}", "result": r})
    if r["f1"] > best["result"]["f1"]:
        best = {"params": {"promote_expand": v}, "result": r}

# Test 3: V1 fallback thresholds
print("\n" + "=" * 70)
print("SWEEP: V1 fallback thresholds (fb_upper)")
print("=" * 70)
for v in [110, 115, 120, 125, 130, 135, 140]:
    r = evaluate({"fb_upper": v})
    print(f"  fb_upper={v}: F1={r['f1']:.4f} P={r['precision']:.4f} TP={r['tp']:,} FP={r['fp']:,}")
    all_results.append({"params": f"fb_upper={v}", "result": r})
    if r["f1"] > best["result"]["f1"]:
        best = {"params": {"fb_upper": v}, "result": r}

print("\n" + "=" * 70)
print("SWEEP: V1 fallback thresholds (fb_lower)")
print("=" * 70)
for v in [15, 18, 20, 22, 25]:
    r = evaluate({"fb_lower": v})
    print(f"  fb_lower={v}: F1={r['f1']:.4f} P={r['precision']:.4f} TP={r['tp']:,} FP={r['fp']:,}")
    all_results.append({"params": f"fb_lower={v}", "result": r})
    if r["f1"] > best["result"]["f1"]:
        best = {"params": {"fb_lower": v}, "result": r}

# Test 4: edge_bonus
print("\n" + "=" * 70)
print("SWEEP: edge_bonus")
print("=" * 70)
for v in [0.25, 0.28, 0.30, 0.32, 0.35, 0.38, 0.40, 0.42, 0.45]:
    r = evaluate({"edge_bonus": v})
    print(f"  edge_bonus={v:.2f}: F1={r['f1']:.4f} P={r['precision']:.4f} TP={r['tp']:,} FP={r['fp']:,}")
    all_results.append({"params": f"edge_bonus={v}", "result": r})
    if r["f1"] > best["result"]["f1"]:
        best = {"params": {"edge_bonus": v}, "result": r}

# Test 5: Signal verification thresholds
print("\n" + "=" * 70)
print("SWEEP: sig_diff_ratio")
print("=" * 70)
for v in [1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0]:
    r = evaluate({"sig_diff_ratio": v})
    print(f"  sig_diff_ratio={v}: F1={r['f1']:.4f} P={r['precision']:.4f} TP={r['tp']:,} FP={r['fp']:,}")
    all_results.append({"params": f"sig_diff_ratio={v}", "result": r})
    if r["f1"] > best["result"]["f1"]:
        best = {"params": {"sig_diff_ratio": v}, "result": r}

print("\n" + "=" * 70)
print("SWEEP: sig_mag_ratio")
print("=" * 70)
for v in [1.1, 1.2, 1.3, 1.4, 1.5, 1.6]:
    r = evaluate({"sig_mag_ratio": v})
    print(f"  sig_mag_ratio={v}: F1={r['f1']:.4f} P={r['precision']:.4f} TP={r['tp']:,} FP={r['fp']:,}")
    all_results.append({"params": f"sig_mag_ratio={v}", "result": r})
    if r["f1"] > best["result"]["f1"]:
        best = {"params": {"sig_mag_ratio": v}, "result": r}

# Test 6: seed_area_ratio
print("\n" + "=" * 70)
print("SWEEP: seed_area_ratio")
print("=" * 70)
for v in [0.02, 0.025, 0.03, 0.04, 0.05, 0.06]:
    r = evaluate({"seed_area_ratio": v})
    print(f"  seed_area_ratio={v}: F1={r['f1']:.4f} P={r['precision']:.4f} TP={r['tp']:,} FP={r['fp']:,}")
    all_results.append({"params": f"seed_area_ratio={v}", "result": r})
    if r["f1"] > best["result"]["f1"]:
        best = {"params": {"seed_area_ratio": v}, "result": r}

# Test 7: Combination of best individual params
print("\n" + "=" * 70)
print("COMBO: Best params combined")
print("=" * 70)
combo_params = dict(best["params"])
r = evaluate(combo_params)
print(f"  {combo_params}: F1={r['f1']:.4f} P={r['precision']:.4f} TP={r['tp']:,} FP={r['fp']:,}")
all_results.append({"params": f"combo={combo_params}", "result": r})
if r["f1"] > best["result"]["f1"]:
    best = {"params": combo_params, "result": r}

# Random search around best params
print("\n" + "=" * 70)
print("RANDOM: Fine-tuning around best params")
print("=" * 70)
random.seed(42)
for i in range(30):
    p = dict(combo_params)
    for k in list(p.keys()):
        if k == "fb_upper":
            p[k] = max(100, min(150, p[k] + random.randint(-10, 10)))
        elif k == "fb_lower":
            p[k] = max(10, min(30, p[k] + random.randint(-5, 5)))
        elif k == "snr_promote":
            p[k] = max(18, min(28, p[k] + random.randint(-3, 3)))
        elif k == "promote_expand":
            p[k] = max(0, min(4, p[k] + random.randint(-1, 1)))
        elif k == "edge_bonus":
            p[k] = round(max(0.20, min(0.50, p[k] + random.uniform(-0.05, 0.05))), 2)
        elif k == "sig_diff_ratio":
            p[k] = round(max(1.0, min(2.5, p[k] + random.uniform(-0.2, 0.2))), 1)
        elif k == "sig_mag_ratio":
            p[k] = round(max(1.0, min(2.0, p[k] + random.uniform(-0.1, 0.1))), 1)
        elif k == "seed_area_ratio":
            p[k] = round(max(0.01, min(0.08, p[k] + random.uniform(-0.01, 0.01))), 3)
    r = evaluate(p)
    if r["f1"] > best["result"]["f1"]:
        print(f"  NEW BEST: {p} → F1={r['f1']:.4f} (Δ={r['f1']-baseline['f1']:+.4f})")
        best = {"params": dict(p), "result": r}
        all_results.append({"params": f"random_best={p}", "result": r})

# ═══════════════════════════════════════════════════════════════════
# Final report
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("FINAL BEST")
print("=" * 70)
print(f"  Baseline F1: {baseline['f1']:.4f}")
print(f"  Best F1:     {best['result']['f1']:.4f}  (Δ: {best['result']['f1']-baseline['f1']:+.4f})")
print(f"  Best params: {best['params']}")
print(f"\n  Baseline: P={baseline['precision']:.4f} R={baseline['recall']:.4f} "
      f"TP={baseline['tp']:,} FP={baseline['fp']:,}")
print(f"  Best:     P={best['result']['precision']:.4f} R={best['result']['recall']:.4f} "
      f"TP={best['result']['tp']:,} FP={best['result']['fp']:,}")

# Save best params
output_data = {
    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "n_frames": len(ALL_FRAMES),
    "baseline_f1": baseline["f1"],
    "best_f1": best["result"]["f1"],
    "best_params": best["params"],
    "delta": best["result"]["f1"] - baseline["f1"],
}
best_file = OUT_DIR / "best_v13_params.json"
with open(best_file, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"\n  Saved to: {best_file}")
print("\nDONE!")

# Restore defaults
V13_PARAMS.clear()
