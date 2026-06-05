"""P1-1: Randomized parameter search for V13 optimal settings.

Precomputes diff/flow/mag/mag_ratio for all frames → fast mask evaluation
per parameter combo. Uses staged search:
  1. Broad random search (~250 combos) over the most impactful params
  2. Local refinement around top-N results
  3. Output best parameter set for hardcoding into V13
"""

import cv2
import numpy as np
from pathlib import Path
import sys
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
OUT_DIR = PROJECT_ROOT / "output" / "evaluation" / "v13_grid_search"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Find all frames with GT ────────────────────────────────────────
gt_files = sorted(GT_DIR.glob("*_FV.png"))
ALL_FRAMES = [f.stem.replace("_FV", "") for f in gt_files
              if (CURR_DIR / f"{f.stem.replace('_FV', '')}_FV.png").exists()
              and (PREV_DIR / f"{f.stem.replace('_FV', '')}_FV_prev.png").exists()]

print(f"Found {len(ALL_FRAMES)} frames with GT")
print(f"Range: {ALL_FRAMES[0]} - {ALL_FRAMES[-1]}")

# ════════════════════════════════════════════════════════════════════
# Precompute diff/flow/mag/ratio for all frames
# ════════════════════════════════════════════════════════════════════
print("\nPrecomputing diff/flow/mag/ratio for all frames...")
precomputed = {}

for i, fid in enumerate(ALL_FRAMES):
    c = cv2.imread(str(CURR_DIR / f"{fid}_FV.png"))
    p = cv2.imread(str(PREV_DIR / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GT_DIR / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)

    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    pg = cv2.GaussianBlur(pg, (5, 5), 0)
    cg = cv2.GaussianBlur(cg, (5, 5), 0)

    diff = cv2.absdiff(cg, pg)
    flow = cv2.calcOpticalFlowFarneback(pg, cg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.linalg.norm(flow, axis=2)
    mag_mean = cv2.GaussianBlur(mag, (41, 41), 15)
    mag_ratio = mag / (mag_mean + 0.5)

    # Feature extraction (same as V13)
    p50_diff = np.percentile(diff, 50)
    p90_diff = np.percentile(diff, 90)
    p95_diff = np.percentile(diff, 95)
    p99_diff = np.percentile(diff, 99)
    snr_diff = p95_diff / (p50_diff + 1e-6)

    p50_mag = np.percentile(mag, 50)
    p90_mag = np.percentile(mag, 90)
    p95_mag = np.percentile(mag, 95)
    snr_mag = p95_mag / (p50_mag + 0.1)

    diff_flat = diff.ravel()
    top5_t = np.percentile(diff_flat, 95)
    top5_sum = diff_flat[diff_flat >= top5_t].sum()
    total_sum = diff_flat.sum() + 1e-6
    diff_concentration = top5_sum / total_sum

    global_diff_mean = float(np.mean(diff))
    global_mag_mean = float(np.mean(mag))

    gt_bin = (g > 0)

    precomputed[fid] = {
        "diff": diff, "flow": flow, "mag": mag, "mag_ratio": mag_ratio,
        "p50_diff": float(p50_diff), "p95_diff": float(p95_diff),
        "p99_diff": float(p99_diff), "snr_diff": float(snr_diff),
        "p50_mag": float(p50_mag), "p95_mag": float(p95_mag),
        "snr_mag": float(snr_mag),
        "diff_concentration": float(diff_concentration),
        "global_diff_mean": global_diff_mean,
        "global_mag_mean": global_mag_mean,
        "gt_bin": gt_bin,
    }
    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{len(ALL_FRAMES)}] frames precomputed")

print("Precomputation done.\n")

# ════════════════════════════════════════════════════════════════════
# Mask generator for a given parameter set
# ════════════════════════════════════════════════════════════════════


def generate_mask_v13(precomp, params):
    """Generate V13 mask for one frame given precomputed data and params."""
    diff = precomp["diff"]
    mag = precomp["mag"]
    mag_ratio = precomp["mag_ratio"]
    h, w = diff.shape

    p95_diff = precomp["p95_diff"]
    snr_diff = precomp["snr_diff"]
    global_diff_mean = precomp["global_diff_mean"]
    global_mag_mean = precomp["global_mag_mean"]

    # ── Fallback logic ──────────────────────────────────────────
    is_genuine_strong = (p95_diff > params["fb_p95_thresh"]) and (snr_diff > params["fb_snr_thresh"])
    is_genuine_clean = (p95_diff < 25) and (snr_diff > 2.5)
    is_challenging_weak = (p95_diff < 25) and (snr_diff < 1.8)

    if is_genuine_strong or is_genuine_clean:
        if p95_diff > params["fb_p95_thresh"]:
            st = params["fb_seed_strong"]
            ft = params["fb_flow_strong"]
            ed = params["fb_expand_strong"]
        else:
            st = params["fb_seed_weak"]
            ft = params["fb_flow_weak"]
            ed = params["fb_expand_weak"]

        _, seed_fb = cv2.threshold(diff, st, 255, cv2.THRESH_BINARY)
        _, cand_fb = cv2.threshold(mag, ft, 255, cv2.THRESH_BINARY)
        cand_fb = cand_fb.astype(np.uint8)
        dist_fb = cv2.distanceTransform((seed_fb == 0).astype(np.uint8), cv2.DIST_L2, 5)
        exp_fb = cand_fb.copy()
        exp_fb[dist_fb > ed] = 0
        mask = seed_fb | exp_fb

        # CCA on fallback
        nl, lb, st_, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, nl):
            ar = st_[i, cv2.CC_STAT_AREA]
            if ar < 100:
                mask[lb == i] = 0; continue
            comp = (lb == i)
            md = np.mean(diff[comp]); mm = np.mean(mag[comp])
            if md / (global_diff_mean + 1e-6) < params["sig_diff_ratio"] or \
               mm / (global_mag_mean + 1e-6) < params["sig_mag_ratio"]:
                mask[lb == i] = 0; continue
            if ar > 200:
                si = np.sum((seed_fb > 0) & comp)
                if si / ar < params["seed_area_ratio"]:
                    mask[lb == i] = 0; continue
            cw = st_[i, cv2.CC_STAT_WIDTH]; ch = st_[i, cv2.CC_STAT_HEIGHT]
            if cw > 0 and ch > 0:
                bba = cw * ch; fr = ar / bba
                ar_ = max(cw, ch) / max(min(cw, ch), 1)
                cb = (lb == i).astype(np.uint8)
                cts, _ = cv2.findContours(cb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                sol = 1.0
                if cts:
                    hull = cv2.convexHull(np.vstack(cts))
                    ha = cv2.contourArea(hull)
                    sol = ar / ha if ha > 0 else 1.0
                if (ar_ > params["cca_ar_thresh"] and sol < params["cca_sol_thresh"]) or \
                   (100 <= ar <= params["cca_sparse_max_area"] and fr < params["cca_fill_thresh"] and sol < params["cca_sol_thresh"]):
                    mask[lb == i] = 0
        k = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        return mask

    if is_challenging_weak:
        _, seed = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        k3 = np.ones((3, 3), np.uint8)
        return cv2.morphologyEx(seed, cv2.MORPH_OPEN, k3)

    # ── 7-tier parameter selection ───────────────────────────────
    if p95_diff > 100 and snr_diff > params["snr_boundary_A"]:
        rtb = params["r_A"]; ed = params["ed_A"]; eb = params["eb_A"]
        uaf = True; ft = params["ft_A"]; st = params["st_A"]; mn = params["mn_A"]
    elif p95_diff > 100 and snr_diff <= params["snr_boundary_A"]:
        rtb = params["r_B"]; ed = params["ed_B"]; eb = params["eb_B"]
        uaf = True; ft = params["ft_B"]; st = params["st_B"]; mn = params["mn_B"]
    elif p95_diff > 75 and snr_diff > params["snr_boundary_C"]:
        rtb = params["r_C"]; ed = params["ed_C"]; eb = params["eb_C"]
        uaf = True; ft = params["ft_C"]; st = params["st_C"]; mn = params["mn_C"]
    elif p95_diff > 75 and snr_diff <= params["snr_boundary_C"]:
        rtb = params["r_D"]; ed = params["ed_D"]; eb = params["eb_D"]
        uaf = False; ft = params["ft_D"]; st = params["st_D"]; mn = params["mn_D"]
    elif p95_diff > 45 and snr_diff > params["snr_boundary_E"]:
        rtb = params["r_E"]; ed = params["ed_E"]; eb = params["eb_E"]
        uaf = False; ft = params["ft_E"]; st = params["st_E"]; mn = params["mn_E"]
    elif p95_diff > 45 and snr_diff <= params["snr_boundary_E"]:
        rtb = params["r_F"]; ed = params["ed_F"]; eb = params["eb_F"]
        uaf = False; ft = params["ft_F"]; st = params["st_F"]; mn = params["mn_F"]
    else:
        rtb = params["r_G"]; ed = params["ed_G"]; eb = params["eb_G"]
        uaf = False; ft = params["ft_G"]; st = params["st_G"]; mn = params["mn_G"]

    # ── Region-adaptive ratio ────────────────────────────────────
    cx, cy = w / 2, h / 2
    yy, xx = np.ogrid[:h, :w]
    dmap = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    nd = dmap / max_r
    rm = 1.0 + eb * (nd ** 2)
    rtm = rtb * rm

    if uaf:
        cand = ((mag > ft) & (mag_ratio > rtm)).astype(np.uint8) * 255
    else:
        cand = (mag_ratio > rtm).astype(np.uint8) * 255

    # ── Spatial seed filtering ───────────────────────────────────
    _, seed_raw = cv2.threshold(diff, st, 255, cv2.THRESH_BINARY)
    nk = np.ones((7, 7), np.uint8)
    nc = cv2.filter2D((seed_raw > 0).astype(np.float32), -1, nk)
    seed = seed_raw.copy()
    seed[nc < mn] = 0
    if seed.max() == 0:
        seed = seed_raw

    # ── Distance expansion ───────────────────────────────────────
    dist = cv2.distanceTransform((seed == 0).astype(np.uint8), cv2.DIST_L2, 5)
    expanded = cand.copy()
    expanded[dist > ed] = 0

    mask = seed | expanded

    # ── CCA post-processing ──────────────────────────────────────
    nl, lb, st_, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, nl):
        ar = st_[i, cv2.CC_STAT_AREA]
        if ar < params["cca_min_area"]:
            mask[lb == i] = 0; continue
        comp = (lb == i)
        md = np.mean(diff[comp]); mm = np.mean(mag[comp])
        if md / (global_diff_mean + 1e-6) < params["sig_diff_ratio"] or \
           mm / (global_mag_mean + 1e-6) < params["sig_mag_ratio"]:
            mask[lb == i] = 0; continue
        if ar > 200:
            si = np.sum((seed > 0) & comp)
            if si / ar < params["seed_area_ratio"]:
                mask[lb == i] = 0; continue
        cw = st_[i, cv2.CC_STAT_WIDTH]; ch = st_[i, cv2.CC_STAT_HEIGHT]
        if cw > 0 and ch > 0:
            bba = cw * ch; fr = ar / bba
            ar_ = max(cw, ch) / max(min(cw, ch), 1)
            cb = (lb == i).astype(np.uint8)
            cts, _ = cv2.findContours(cb, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            sol = 1.0
            if cts:
                hull = cv2.convexHull(np.vstack(cts))
                ha = cv2.contourArea(hull)
                sol = ar / ha if ha > 0 else 1.0
            if (ar_ > params["cca_ar_thresh"] and sol < params["cca_sol_thresh"]) or \
               (100 <= ar <= params["cca_sparse_max_area"] and fr < params["cca_fill_thresh"] and sol < params["cca_sol_thresh"]):
                mask[lb == i] = 0

    k = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def evaluate_params(params):
    """Evaluate a parameter set across all frames, return mean F1."""
    results = []
    for fid in ALL_FRAMES:
        mask = generate_mask_v13(precomputed[fid], params)
        gt_bin = precomputed[fid]["gt_bin"]
        m = compute_metrics(mask, gt_bin.astype(np.uint8) * 255)
        results.append(m)
    avg_f1 = np.mean([r["f1"] for r in results])
    avg_prec = np.mean([r["precision"] for r in results])
    avg_rec = np.mean([r["recall"] for r in results])
    tp_sum = sum(r["tp"] for r in results)
    fp_sum = sum(r["fp"] for r in results)
    fn_sum = sum(r["fn"] for r in results)
    return {"f1": avg_f1, "precision": avg_prec, "recall": avg_rec,
            "tp": tp_sum, "fp": fp_sum, "fn": fn_sum}


# ════════════════════════════════════════════════════════════════════
# Default parameters (from V13 hardcoded values)
# ════════════════════════════════════════════════════════════════════
DEFAULT = {
    # Fallback
    "fb_p95_thresh": 110, "fb_snr_thresh": 2.0,
    "fb_seed_strong": 42, "fb_flow_strong": 2.0, "fb_expand_strong": 12,
    "fb_seed_weak": 18, "fb_flow_weak": 2.0, "fb_expand_weak": 10,

    # SNR boundaries for tiers
    "snr_boundary_A": 2.2, "snr_boundary_C": 1.9, "snr_boundary_E": 1.6,

    # Tier A: strong + clear
    "r_A": 1.25, "ed_A": 12, "eb_A": 0.28, "ft_A": 1.8, "st_A": 22, "mn_A": 3,
    # Tier B: strong + noisy
    "r_B": 1.55, "ed_B": 10, "eb_B": 0.42, "ft_B": 2.2, "st_B": 22, "mn_B": 4,
    # Tier C: med-strong + clear
    "r_C": 1.45, "ed_C": 11, "eb_C": 0.30, "ft_C": 1.8, "st_C": 18, "mn_C": 3,
    # Tier D: med-strong + noisy
    "r_D": 1.80, "ed_D": 9, "eb_D": 0.40, "ft_D": 2.5, "st_D": 18, "mn_D": 4,
    # Tier E: medium + ok
    "r_E": 1.80, "ed_E": 10, "eb_E": 0.33, "ft_E": 2.5, "st_E": 16, "mn_E": 3,
    # Tier F: medium + noisy
    "r_F": 2.15, "ed_F": 7, "eb_F": 0.45, "ft_F": 3.0, "st_F": 16, "mn_F": 4,
    # Tier G: weak
    "r_G": 2.40, "ed_G": 5, "eb_G": 0.45, "ft_G": 3.0, "st_G": 14, "mn_G": 4,

    # CCA
    "cca_min_area": 100, "cca_ar_thresh": 8, "cca_sol_thresh": 0.4,
    "cca_fill_thresh": 0.12, "cca_sparse_max_area": 5000,

    # P0-2/P0-3 signal verification
    "sig_diff_ratio": 1.5, "sig_mag_ratio": 1.3, "seed_area_ratio": 0.03,
}

# ════════════════════════════════════════════════════════════════════
# Stage 1: Evaluate baseline
# ════════════════════════════════════════════════════════════════════
print("=" * 80)
print("STAGE 1: Baseline evaluation")
print("=" * 80)
baseline_r = evaluate_params(DEFAULT)
print(f"  Default V13 params: F1={baseline_r['f1']:.4f}  "
      f"P={baseline_r['precision']:.4f}  R={baseline_r['recall']:.4f}  "
      f"TP={baseline_r['tp']:,}  FP={baseline_r['fp']:,}")


# ════════════════════════════════════════════════════════════════════
# Stage 2: Randomized search over high-impact parameters
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STAGE 2: Randomized search (~200 combos)")
print("=" * 80)


def random_params():
    """Generate a random parameter set within reasonable ranges."""
    p = dict(DEFAULT)  # Start from defaults, mutate key params

    # Perturb fallback thresholds
    p["fb_p95_thresh"] = random.choice([100, 105, 110, 115, 120])
    p["fb_snr_thresh"] = round(random.uniform(1.6, 2.4), 1)

    # Perturb SNR boundaries
    p["snr_boundary_A"] = round(random.uniform(1.8, 2.6), 1)
    p["snr_boundary_C"] = round(random.uniform(1.5, 2.2), 1)
    p["snr_boundary_E"] = round(random.uniform(1.3, 1.9), 1)
    # Keep ordering
    if p["snr_boundary_C"] >= p["snr_boundary_A"]:
        p["snr_boundary_C"] = p["snr_boundary_A"] - 0.2
    if p["snr_boundary_E"] >= p["snr_boundary_C"]:
        p["snr_boundary_E"] = p["snr_boundary_C"] - 0.2

    # Perturb ratio thresholds (±20%)
    for tier in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        p[f"r_{tier}"] = round(p[f"r_{tier}"] * random.uniform(0.80, 1.20), 2)
        p[f"ed_{tier}"] = max(3, int(p[f"ed_{tier}"] * random.uniform(0.75, 1.25)))
        p[f"eb_{tier}"] = round(max(0.15, min(0.55, p[f"eb_{tier}"] * random.uniform(0.70, 1.30))), 2)

    # Perturb flow thresholds for abs-flow tiers
    for tier in ['A', 'B', 'C']:
        p[f"ft_{tier}"] = round(p[f"ft_{tier}"] * random.uniform(0.80, 1.20), 1)

    # Perturb seed thresholds
    for tier in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
        p[f"st_{tier}"] = max(10, int(p[f"st_{tier}"] * random.uniform(0.80, 1.20)))

    # Perturb P0-2/P0-3 thresholds
    p["sig_diff_ratio"] = round(random.uniform(1.2, 2.0), 1)
    p["sig_mag_ratio"] = round(random.uniform(1.1, 1.6), 1)
    p["seed_area_ratio"] = round(random.uniform(0.02, 0.06), 2)

    # Perturb CCA
    p["cca_min_area"] = random.choice([80, 100, 120, 150])
    p["cca_ar_thresh"] = random.choice([6, 8, 10])

    return p


N_RANDOM = 200
random.seed(42)
best_random = {"params": DEFAULT, "f1": baseline_r["f1"]}
all_random_results = []

for i in range(N_RANDOM):
    params = random_params()
    r = evaluate_params(params)
    all_random_results.append({"params": params, "result": r})
    if r["f1"] > best_random["f1"]:
        best_random = {"params": params, "f1": r["f1"]}
    if (i + 1) % 20 == 0:
        print(f"  [{i+1}/{N_RANDOM}]  best F1 so far: {best_random['f1']:.4f}  "
              f"(baseline={baseline_r['f1']:.4f}, Δ={best_random['f1']-baseline_r['f1']:+.4f})")

print(f"\n  Best random F1: {best_random['f1']:.4f}  "
      f"(Δ baseline: {best_random['f1']-baseline_r['f1']:+.4f})")

# ════════════════════════════════════════════════════════════════════
# Stage 3: Fine-grid around top-5 random results
# ════════════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STAGE 3: Fine refinement around top results")
print("=" * 80)

all_random_results.sort(key=lambda x: x["result"]["f1"], reverse=True)
top5 = all_random_results[:5]

best_fine = {"params": None, "f1": best_random["f1"]}

for rank, entry in enumerate(top5):
    base = entry["params"]
    base_f1 = entry["result"]["f1"]
    print(f"\n  Refining top-{rank+1} (F1={base_f1:.4f})...")

    for _ in range(20):
        p = dict(base)
        # Small perturbations on ratio thresholds and edge_bonus
        for tier in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
            if random.random() < 0.4:
                p[f"r_{tier}"] = round(p[f"r_{tier}"] * random.uniform(0.92, 1.08), 2)
            if random.random() < 0.3:
                p[f"eb_{tier}"] = round(max(0.15, min(0.55, p[f"eb_{tier}"] * random.uniform(0.92, 1.08))), 2)
        # Perturb SNR boundaries
        p["snr_boundary_A"] = round(p["snr_boundary_A"] + random.uniform(-0.15, 0.15), 1)
        p["snr_boundary_C"] = round(p["snr_boundary_C"] + random.uniform(-0.15, 0.15), 1)
        p["snr_boundary_E"] = round(p["snr_boundary_E"] + random.uniform(-0.15, 0.15), 1)

        r = evaluate_params(p)
        if r["f1"] > best_fine["f1"]:
            best_fine = {"params": p, "f1": r["f1"]}
            print(f"    → New best: F1={best_fine['f1']:.4f}  "
                  f"(Δ baseline: {best_fine['f1']-baseline_r['f1']:+.4f})")

# ════════════════════════════════════════════════════════════════════
# Final report
# ════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 80)
print("FINAL RESULTS")
print("=" * 80)

final_params = best_fine["params"] if best_fine["params"] else best_random["params"]
final_r = evaluate_params(final_params)

print(f"\n  Baseline  F1: {baseline_r['f1']:.4f}")
print(f"  Best      F1: {final_r['f1']:.4f}  (Δ: {final_r['f1']-baseline_r['f1']:+.4f})")
print(f"\n  Baseline:  P={baseline_r['precision']:.4f}  R={baseline_r['recall']:.4f}  "
      f"TP={baseline_r['tp']:,}  FP={baseline_r['fp']:,}")
print(f"  Best:       P={final_r['precision']:.4f}  R={final_r['recall']:.4f}  "
      f"TP={final_r['tp']:,}  FP={final_r['fp']:,}")

# Save best params
best_params_file = OUT_DIR / "best_v13_params.json"
ts = datetime.now().strftime("%Y-%m-%d %H:%M")
output_data = {
    "timestamp": ts,
    "n_frames": len(ALL_FRAMES),
    "baseline": {"f1": baseline_r["f1"], "precision": baseline_r["precision"],
                 "recall": baseline_r["recall"], "tp": baseline_r["tp"],
                 "fp": baseline_r["fp"], "fn": baseline_r["fn"]},
    "best": {"f1": final_r["f1"], "precision": final_r["precision"],
             "recall": final_r["recall"], "tp": final_r["tp"],
             "fp": final_r["fp"], "fn": final_r["fn"]},
    "params": final_params,
}
with open(best_params_file, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"\n  Best parameters saved to: {best_params_file}")
print("\nDONE!")
