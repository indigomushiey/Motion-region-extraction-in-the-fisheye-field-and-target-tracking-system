"""Staged grid search for V11 optimal parameters.

Precomputes diff/flow/mag/mag_ratio for all GT frames (avoids re-running
Farneback per parameter combo), then does staged search:
  1. strong_motion_thresh
  2. Per-bin params (randomized + fine-grid around top-10)
  3. Region-adaptive multipliers (center_mult, edge_mult)
  4. CCA params (min_component_area, merge_dist)
  5. Final refinement around best params
"""

import cv2
import numpy as np
from pathlib import Path
import sys
import itertools
import random
import json
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.evaluation.metrics import compute_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA = PROJECT_ROOT / "data" / "homework2"
GT_DIR = DATA / "motion_annotation" / "GroudTruth"
CURR_DIR = DATA / "rgb_images"
PREV_DIR = DATA / "previous_images"
OUT_DIR = PROJECT_ROOT / "output" / "evaluation" / "v11_grid_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PYTHON = "/c/Users/86178/python/python.exe"

# ── Find all frames with GT ────────────────────────────────────────
gt_files = sorted(GT_DIR.glob("*_FV.png"))
ALL_FRAMES = [f.stem.replace("_FV", "") for f in gt_files
              if (CURR_DIR / f"{f.stem.replace('_FV', '')}_FV.png").exists()
              and (PREV_DIR / f"{f.stem.replace('_FV', '')}_FV_prev.png").exists()]

print(f"Found {len(ALL_FRAMES)} frames with GT")
print(f"Range: {ALL_FRAMES[0]} - {ALL_FRAMES[-1]}")


# ════════════════════════════════════════════════════════════════════
# Precompute all intermediate results
# ════════════════════════════════════════════════════════════════════
print("\nPrecomputing diff/flow/mag/ratio for all frames...")

precomputed = {}

for i, fid in enumerate(ALL_FRAMES):
    c = cv2.imread(str(CURR_DIR / f"{fid}_FV.png"))
    p = cv2.imread(str(PREV_DIR / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GT_DIR / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)

    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    pg = cv2.GaussianBlur(pg, (5, 5), 0)
    cg = cv2.GaussianBlur(cg, (5, 5), 0)

    diff = cv2.absdiff(cg, pg)
    p95_diff = np.percentile(diff, 95)

    flow = cv2.calcOpticalFlowFarneback(
        pg, cg, None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag = np.linalg.norm(flow, axis=2)
    mag_mean = cv2.GaussianBlur(mag, (41, 41), 15)
    mag_ratio = mag / (mag_mean + 0.5)

    precomputed[fid] = {
        "prev_gray": pg,
        "curr_gray": cg,
        "gt_bin": (g > 0),
        "diff": diff,
        "p95_diff": p95_diff,
        "flow": flow,
        "mag": mag,
        "mag_ratio": mag_ratio,
        "h": pg.shape[0],
        "w": pg.shape[1],
    }

    if (i + 1) % 20 == 0:
        print(f"  {i + 1}/{len(ALL_FRAMES)} done")

print(f"Precomputation complete: {len(precomputed)} frames cached.")


# ════════════════════════════════════════════════════════════════════
# V11 scoring function (all parameters exposed)
# ════════════════════════════════════════════════════════════════════

def score_v11_params(
    # Frame classification
    strong_motion_thresh=120,
    # Per-bin: [very_weak, weak, medium, strong, very_strong]
    bin_thresholds=(30, 50, 70, 100),  # 4 boundaries → 5 bins
    seed_threshs=(14, 16, 18, 20, 22),
    ratio_thresh_bases=(2.2, 1.8, 1.6, 1.5, 1.3),
    expand_dists=(4, 6, 8, 10, 12),
    use_abs_flows=(False, False, False, True, True),
    flow_threshs=(0, 0, 0, 2.5, 2.0),
    use_directions=(True, True, True, False, False),
    # Region-adaptive ratio
    edge_bonus=0.35,
    # Spatial seed filtering
    min_neighbors_strong=3,
    min_neighbors_weak=4,
    # CCA post-processing
    min_component_area=50,
    merge_dist=15,
    # V1 fallback params
    fallback_seed_thresh=45,
    fallback_flow_thresh=2.0,
    fallback_expand_dist=12,
):
    """Score a parameter set across all frames. Returns mean F1 and per-frame dict."""
    results = {}
    f1s = []

    for fid, data in precomputed.items():
        pg = data["prev_gray"]
        cg = data["curr_gray"]
        gt_bin = data["gt_bin"]
        diff = data["diff"]
        p95_diff = data["p95_diff"]
        flow = data["flow"]
        mag = data["mag"]
        mag_ratio = data["mag_ratio"]

        # ── Strong-motion fallback ──────────────────────────────────
        if p95_diff > strong_motion_thresh:
            _, seed = cv2.threshold(diff, fallback_seed_thresh, 255, cv2.THRESH_BINARY)
            _, candidate = cv2.threshold(mag, fallback_flow_thresh, 255, cv2.THRESH_BINARY)
            candidate = candidate.astype(np.uint8)
            dist_t = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
            expanded = candidate.copy()
            expanded[dist_t > fallback_expand_dist] = 0
            mask = seed | expanded
            k = np.ones((7, 7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            m = compute_metrics(mask, gt_bin.astype(np.uint8) * 255)
            results[fid] = m
            f1s.append(m["f1"])
            continue

        # ── Frame-adaptive bin selection ────────────────────────────
        if p95_diff > bin_thresholds[3]:
            idx = 4
        elif p95_diff > bin_thresholds[2]:
            idx = 3
        elif p95_diff > bin_thresholds[1]:
            idx = 2
        elif p95_diff > bin_thresholds[0]:
            idx = 1
        else:
            idx = 0

        seed_thresh = seed_threshs[idx]
        ratio_thresh_base = ratio_thresh_bases[idx]
        expand_dist = expand_dists[idx]
        use_abs_flow = use_abs_flows[idx]
        use_direction = use_directions[idx]
        flow_thresh = flow_threshs[idx]

        # ── Region-adaptive ratio threshold ─────────────────────────
        h, w = pg.shape
        cx, cy = w / 2, h / 2
        yy, xx = np.ogrid[:h, :w]
        dist_map = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        max_r = np.sqrt(cx ** 2 + cy ** 2)
        norm_dist = dist_map / max_r
        radius_mult = 1.0 + edge_bonus * (norm_dist ** 2)
        ratio_thresh_map = ratio_thresh_base * radius_mult

        if use_abs_flow:
            candidate = ((mag > flow_thresh) & (mag_ratio > ratio_thresh_map)).astype(np.uint8) * 255
        else:
            candidate = (mag_ratio > ratio_thresh_map).astype(np.uint8) * 255

        # ── Spatial seed filtering ──────────────────────────────────
        _, seed_raw = cv2.threshold(diff, seed_thresh, 255, cv2.THRESH_BINARY)
        neighbor_kernel = np.ones((7, 7), np.uint8)
        neighbor_count = cv2.filter2D(
            (seed_raw > 0).astype(np.float32), -1, neighbor_kernel
        )
        min_nb = min_neighbors_strong if p95_diff > 80 else min_neighbors_weak
        seed = seed_raw.copy()
        seed[neighbor_count < min_nb] = 0
        if seed.max() == 0:
            seed = seed_raw

        # ── Distance-constrained expansion ──────────────────────────
        dist_t = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
        expanded = candidate.copy()
        expanded[dist_t > expand_dist] = 0

        # ── Direction consistency (selective) ───────────────────────
        if use_direction:
            seed_bool = (seed > 0)
            if seed_bool.sum() > 20:
                flow_angle = np.arctan2(flow[..., 1], flow[..., 0])
                seed_angle_map = np.zeros_like(flow_angle)
                seed_angle_map[seed_bool] = flow_angle[seed_bool]
                seed_weight = np.zeros_like(flow_angle)
                seed_weight[seed_bool] = 1.0

                ks = expand_dist * 2 + 1
                seed_angle_blurred = cv2.GaussianBlur(seed_angle_map, (ks, ks), expand_dist / 2.0)
                seed_weight_blurred = cv2.GaussianBlur(seed_weight, (ks, ks), expand_dist / 2.0)
                valid_weight = seed_weight_blurred > 0.02
                seed_angle_blurred[valid_weight] /= seed_weight_blurred[valid_weight]

                angle_diff = np.abs(flow_angle - seed_angle_blurred)
                angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
                direction_ok = (angle_diff < np.pi / 4) | ~valid_weight
                expanded[~direction_ok] = 0

        # ── Merge ───────────────────────────────────────────────────
        mask = seed | expanded

        # ── CCA post-processing (simplified: remove small + morphology) ──
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] < min_component_area:
                mask[labels == i] = 0

        k = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)

        m = compute_metrics(mask, gt_bin.astype(np.uint8) * 255)
        results[fid] = m
        f1s.append(m["f1"])

    mean_f1 = np.mean(f1s) if f1s else 0.0
    return mean_f1, results


# ════════════════════════════════════════════════════════════════════
# Stage 1: strong_motion_thresh
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STAGE 1: strong_motion_thresh")
print("=" * 70)

best_stage1 = {"f1": -1, "thresh": 120}
for t in [100, 110, 115, 120, 125, 130, 140]:
    f1, _ = score_v11_params(strong_motion_thresh=t)
    marker = " <--" if f1 > best_stage1["f1"] else ""
    print(f"  thresh={t:>4d}  F1={f1:.4f}{marker}")
    if f1 > best_stage1["f1"]:
        best_stage1 = {"f1": f1, "thresh": t}

print(f"\n  Best: strong_motion_thresh={best_stage1['thresh']} (F1={best_stage1['f1']:.4f})")
strong_motion_thresh = best_stage1["thresh"]


# ════════════════════════════════════════════════════════════════════
# Stage 2: Per-bin parameters (randomized search + top-K refinement)
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STAGE 2: Per-bin parameter optimization")
print("=" * 70)

# Define search space per bin
# Format: (param_name, [candidate_values])
# We sample from these for randomized search

def random_search_stage2(n_trials=300):
    """Randomized search over per-bin parameters."""
    best = {"f1": -1, "params": {}}
    results_history = []

    for trial in range(n_trials):
        # Randomly sample each bin's parameters
        seed_threshs = tuple(random.choice([
            (12, 14, 16, 18, 20),
            (14, 16, 18, 20, 22),
            (16, 18, 20, 22, 24),
            (10, 14, 18, 22, 26),
        ]))

        ratio_thresh_bases = tuple(random.choice([
            (2.0, 1.6, 1.4, 1.3, 1.2),
            (2.2, 1.8, 1.6, 1.5, 1.3),
            (2.5, 2.0, 1.8, 1.5, 1.3),
            (2.0, 1.8, 1.5, 1.4, 1.2),
        ]))

        expand_dists = tuple(random.choice([
            (3, 5, 7, 10, 12),
            (4, 6, 8, 10, 12),
            (4, 6, 8, 10, 14),
            (5, 7, 9, 11, 13),
        ]))

        # abs_flow configs
        abs_configs = [
            (False, False, False, True, True),
            (False, False, True, True, True),
            (False, True, True, True, True),
            (False, False, False, False, True),
        ]
        use_abs_flows = tuple(random.choice(abs_configs))

        flow_threshs = tuple(random.choice([
            (0.0, 0.0, 0.0, 2.5, 2.0),
            (0.0, 0.0, 0.0, 3.0, 2.5),
            (0.0, 0.0, 2.5, 2.5, 2.0),
            (0.0, 0.0, 0.0, 2.0, 1.5),
        ]))

        dir_configs = [
            (True, True, True, False, False),
            (True, True, False, False, False),
            (False, True, True, False, False),
            (True, True, True, True, False),
        ]
        use_directions = tuple(random.choice(dir_configs))

        # Bin thresholds variations
        bin_thresh_variants = [
            (30, 50, 70, 100),
            (30, 55, 80, 100),
            (35, 55, 75, 100),
            (25, 50, 75, 100),
        ]
        bin_thresholds = tuple(random.choice(bin_thresh_variants))

        f1, _ = score_v11_params(
            strong_motion_thresh=strong_motion_thresh,
            bin_thresholds=bin_thresholds,
            seed_threshs=seed_threshs,
            ratio_thresh_bases=ratio_thresh_bases,
            expand_dists=expand_dists,
            use_abs_flows=use_abs_flows,
            flow_threshs=flow_threshs,
            use_directions=use_directions,
        )

        results_history.append({
            "f1": f1,
            "bin_thresholds": bin_thresholds,
            "seed_threshs": seed_threshs,
            "ratio_thresh_bases": ratio_thresh_bases,
            "expand_dists": expand_dists,
            "use_abs_flows": use_abs_flows,
            "flow_threshs": flow_threshs,
            "use_directions": use_directions,
        })

        if f1 > best["f1"]:
            best = {"f1": f1, "params": results_history[-1].copy()}
            print(f"  Trial {trial + 1}: F1={f1:.4f}  NEW BEST")

    return best, results_history


best_stage2, history_s2 = random_search_stage2(300)
print(f"\n  Best F1: {best_stage2['f1']:.4f}")
print(f"  Params: {json.dumps({k: v for k, v in best_stage2['params'].items() if k != 'f1'}, default=str)}")

# Fine refinement around top-10
top10 = sorted(history_s2, key=lambda x: -x["f1"])[:10]
print(f"\n  --- Fine refinement around top-10 ---")

def fine_refine_stage2(top_configs):
    """For each top config, perturb each parameter bin slightly."""
    best = {"f1": -1, "params": {}}

    for base in top_configs:
        for _ in range(20):
            # Perturb each bin's ratio_thresh_base by +/- 0.1
            ratio_pert = tuple(
                max(1.0, base["ratio_thresh_bases"][i] + random.uniform(-0.2, 0.2))
                for i in range(5)
            )
            seed_pert = tuple(
                max(8, min(30, base["seed_threshs"][i] + random.randint(-2, 2)))
                for i in range(5)
            )
            expand_pert = tuple(
                max(2, min(16, base["expand_dists"][i] + random.randint(-2, 2)))
                for i in range(5)
            )

            f1, _ = score_v11_params(
                strong_motion_thresh=strong_motion_thresh,
                bin_thresholds=base["bin_thresholds"],
                seed_threshs=seed_pert,
                ratio_thresh_bases=ratio_pert,
                expand_dists=expand_pert,
                use_abs_flows=base["use_abs_flows"],
                flow_threshs=base["flow_threshs"],
                use_directions=base["use_directions"],
            )

            if f1 > best["f1"]:
                best = {
                    "f1": f1,
                    "params": {
                        "bin_thresholds": base["bin_thresholds"],
                        "seed_threshs": seed_pert,
                        "ratio_thresh_bases": ratio_pert,
                        "expand_dists": expand_pert,
                        "use_abs_flows": base["use_abs_flows"],
                        "flow_threshs": base["flow_threshs"],
                        "use_directions": base["use_directions"],
                    }
                }

    return best


best_s2_refined = fine_refine_stage2(top10)
print(f"  Refined F1: {best_s2_refined['f1']:.4f}")
if best_s2_refined["f1"] > best_stage2["f1"]:
    print("  Refinement improved results!")
    best_stage2 = best_s2_refined

# Extract best per-bin params
bp = best_stage2["params"]
bin_thresholds = bp["bin_thresholds"]
seed_threshs = bp["seed_threshs"]
ratio_thresh_bases = bp["ratio_thresh_bases"]
expand_dists = bp["expand_dists"]
use_abs_flows = bp["use_abs_flows"]
flow_threshs = bp["flow_threshs"]
use_directions = bp["use_directions"]


# ════════════════════════════════════════════════════════════════════
# Stage 3: Region-adaptive edge bonus
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STAGE 3: Region-adaptive edge bonus")
print("=" * 70)

best_stage3 = {"f1": -1, "edge_bonus": 0.35}
for edge_b in [0.0, 0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
    f1, _ = score_v11_params(
        strong_motion_thresh=strong_motion_thresh,
        bin_thresholds=bin_thresholds,
        seed_threshs=seed_threshs,
        ratio_thresh_bases=ratio_thresh_bases,
        expand_dists=expand_dists,
        use_abs_flows=use_abs_flows,
        flow_threshs=flow_threshs,
        use_directions=use_directions,
        edge_bonus=edge_b,
    )
    marker = " <--" if f1 > best_stage3["f1"] else ""
    if f1 > best_stage3["f1"]:
        best_stage3 = {"f1": f1, "edge_bonus": edge_b}
    print(f"  edge_bonus={edge_b:.2f}  F1={f1:.4f}{marker}")

print(f"\n  Best: edge_bonus={best_stage3['edge_bonus']:.2f}  (F1={best_stage3['f1']:.4f})")
edge_bonus = best_stage3["edge_bonus"]


# ════════════════════════════════════════════════════════════════════
# Stage 4: CCA min_component_area
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STAGE 4: CCA min_component_area")
print("=" * 70)

best_stage4 = {"f1": -1, "min_area": 50}
for min_area in [25, 50, 75, 100, 150, 200]:
    f1, _ = score_v11_params(
        strong_motion_thresh=strong_motion_thresh,
        bin_thresholds=bin_thresholds,
        seed_threshs=seed_threshs,
        ratio_thresh_bases=ratio_thresh_bases,
        expand_dists=expand_dists,
        use_abs_flows=use_abs_flows,
        flow_threshs=flow_threshs,
        use_directions=use_directions,
        edge_bonus=edge_bonus,
        min_component_area=min_area,
    )
    marker = " <--" if f1 > best_stage4["f1"] else ""
    if f1 > best_stage4["f1"]:
        best_stage4 = {"f1": f1, "min_area": min_area}
    print(f"  min_area={min_area:>3d}  F1={f1:.4f}{marker}")

print(f"\n  Best: min_component_area={best_stage4['min_area']}  (F1={best_stage4['f1']:.4f})")
min_component_area = best_stage4["min_area"]


# ════════════════════════════════════════════════════════════════════
# Stage 5: Final refinement
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STAGE 5: Final fine refinement around best params")
print("=" * 70)

best_final = {"f1": -1, "params": {}}

for _ in range(100):
    # Small perturbations around current best
    cf = random.uniform(0.95, 1.05)

    seed_pert = tuple(
        max(8, min(30, int(seed_threshs[i] * cf + random.randint(-1, 1))))
        for i in range(5)
    )
    ratio_pert = tuple(
        max(1.0, ratio_thresh_bases[i] * (0.9 + random.random() * 0.2))
        for i in range(5)
    )
    expand_pert = tuple(
        max(2, min(16, int(expand_dists[i] * cf + random.randint(-1, 1))))
        for i in range(5)
    )

    f1, _ = score_v11_params(
        strong_motion_thresh=strong_motion_thresh,
        bin_thresholds=bin_thresholds,
        seed_threshs=seed_pert,
        ratio_thresh_bases=ratio_pert,
        expand_dists=expand_pert,
        use_abs_flows=use_abs_flows,
        flow_threshs=flow_threshs,
        use_directions=use_directions,
        edge_bonus=edge_bonus,
        min_component_area=min_component_area,
    )

    if f1 > best_final["f1"]:
        best_final = {
            "f1": f1,
            "params": {
                "seed_threshs": seed_pert,
                "ratio_thresh_bases": ratio_pert,
                "expand_dists": expand_pert,
            }
        }
        print(f"  F1={f1:.4f}  NEW BEST")

if best_final["f1"] > best_stage4["f1"]:
    print(f"\n  Final refinement improved: {best_stage4['f1']:.4f} → {best_final['f1']:.4f}")
    seed_threshs = best_final["params"]["seed_threshs"]
    ratio_thresh_bases = best_final["params"]["ratio_thresh_bases"]
    expand_dists = best_final["params"]["expand_dists"]


# ════════════════════════════════════════════════════════════════════
# Report final best parameters
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 70)
print("  FINAL BEST PARAMETERS")
print("=" * 70)

final_params = {
    "strong_motion_thresh": strong_motion_thresh,
    "bin_thresholds": list(bin_thresholds),
    "seed_threshs": list(seed_threshs),
    "ratio_thresh_bases": [round(r, 2) for r in ratio_thresh_bases],
    "expand_dists": list(expand_dists),
    "use_abs_flows": list(use_abs_flows),
    "flow_threshs": list(flow_threshs),
    "use_directions": list(use_directions),
    "edge_bonus": round(edge_bonus, 2),
    "min_component_area": min_component_area,
}

for k, v in final_params.items():
    print(f"  {k}: {v}")

# ── Per-frame F1 with final params ─────────────────────────────────
print("\n\n" + "=" * 70)
print("  PER-FRAME F1 WITH FINAL PARAMS")
print("=" * 70)

final_f1, final_results = score_v11_params(**final_params)

# Also get V10 baseline for comparison
print("\n  Computing V10 baseline...")
from core.motion_detection.frame_difference import detect_motion_seed_expand_v10, detect_motion_seed_expand

v10_f1s = []
v1_f1s = []
for fid, data in precomputed.items():
    pg = data["prev_gray"]
    cg = data["curr_gray"]
    gt_bin = data["gt_bin"]

    m10 = detect_motion_seed_expand_v10(pg, cg)
    r10 = compute_metrics(m10, gt_bin.astype(np.uint8) * 255)
    v10_f1s.append(r10["f1"])

    m1 = detect_motion_seed_expand(pg, cg)
    r1 = compute_metrics(m1, gt_bin.astype(np.uint8) * 255)
    v1_f1s.append(r1["f1"])

v10_mean = np.mean(v10_f1s)
v1_mean = np.mean(v1_f1s)

# Per-frame display
print(f"\n{'Frame':<8} {'P95':>6} {'V1_F1':>8} {'V10_F1':>8} {'V11_F1':>8} {'Best':>6}")
print("-" * 50)

count_best = {"V1": 0, "V10": 0, "V11": 0}
for i, fid in enumerate(ALL_FRAMES):
    p95 = precomputed[fid]["p95_diff"]
    f1_v1 = v1_f1s[i]
    f1_v10 = v10_f1s[i]
    f1_v11 = final_results[fid]["f1"]

    best_method = max([("V1", f1_v1), ("V10", f1_v10), ("V11", f1_v11)], key=lambda x: x[1])
    count_best[best_method[0]] += 1

    print(f"{fid:<8} {p95:>6.1f} {f1_v1:>8.4f} {f1_v10:>8.4f} {f1_v11:>8.4f} {best_method[0]:>6}")

print("-" * 50)
print(f"{'MEAN':<8} {'':>6} {v1_mean:>8.4f} {v10_mean:>8.4f} {final_f1:>8.4f}")
print(f"\n  Wins per method: V1={count_best['V1']}  V10={count_best['V10']}  V11={count_best['V11']}")

# ── Save results ───────────────────────────────────────────────────
results_path = OUT_DIR / "grid_search_results.json"
results_data = {
    "timestamp": datetime.now().isoformat(),
    "num_frames": len(ALL_FRAMES),
    "final_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in final_params.items()},
    "v1_f1": round(v1_mean, 4),
    "v10_f1": round(v10_mean, 4),
    "v11_f1": round(final_f1, 4),
    "delta_v10": round(final_f1 - v10_mean, 4),
    "delta_v1": round(final_f1 - v1_mean, 4),
    "count_best": count_best,
    "per_frame": {
        fid: {
            "p95": precomputed[fid]["p95_diff"],
            "V1_F1": v1_f1s[i],
            "V10_F1": v10_f1s[i],
            "V11_F1": final_results[fid]["f1"],
        }
        for i, fid in enumerate(ALL_FRAMES)
    },
}

with open(results_path, "w") as f:
    json.dump(results_data, f, indent=2)

print(f"\n  Results saved to: {results_path}")
print(f"\n  Summary: V11={final_f1:.4f}  V10={v10_mean:.4f}  Δ={final_f1 - v10_mean:+.4f}")
print(f"           V11 vs V1: Δ={final_f1 - v1_mean:+.4f}")
print(f"           Wins: V11={count_best['V11']}/{len(ALL_FRAMES)} frames")
print("\nDone!")
