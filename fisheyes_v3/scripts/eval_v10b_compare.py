"""50-frame: V10 (original) vs V10 (brightness-adaptive). Quick re-run V10 for baseline."""
import cv2, numpy as np, sys, importlib.util, time, gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(__file__).resolve().parents[2] / "fisheyes_v2" / "fisheye_motion_tracking"
sys.path.insert(0, str(PROJECT_ROOT))
from core.motion_hybrid import detect_motion_hybrid_v10

def _i(rp, nm):
    fp = V2_ROOT / rp; s = importlib.util.spec_from_file_location(nm, fp)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cm = _i("core/evaluation/metrics.py", "mt").compute_metrics

DATA = PROJECT_ROOT / "data" / "homework2"
GTD = DATA / "motion_annotation" / "GroudTruth"; CUR = DATA / "rgb_images"; PRV = DATA / "previous_images"
OUT = PROJECT_ROOT / "output" / "evaluation" / "v10b_compare"
OUT.mkdir(parents=True, exist_ok=True)

gt_files = sorted(GTD.glob("*_FV.png"))
all_frames = [f.stem.replace("_FV","") for f in gt_files
              if (CUR/f"{f.stem.replace('_FV','')}_FV.png").exists()
              and (PRV/f"{f.stem.replace('_FV','')}_FV_prev.png").exists()]
N = len(all_frames); SAMPLE = 50; step = N/SAMPLE
sampled = [all_frames[int(i*step)] for i in range(SAMPLE)]
if all_frames[-1] not in sampled: sampled[-1] = all_frames[-1]

print(f"  V10 brightness-adaptive — {SAMPLE} frames (before/after comparison)")

# We need OLD V10 baseline. Let's read the saved results from earlier
# and compare with new V10
old_results = {}
try:
    with open(PROJECT_ROOT / "output" / "evaluation" / "v10_compare" / "v10_compare_results.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0].isdigit():
                old_results[parts[0]] = float(parts[3])  # V10 F1 column
    print(f"  Loaded {len(old_results)} old V10 (min) results for comparison")
except:
    print("  No old results, running fresh")

results = {}; stats = {}; times = []

print(f"\n{'Frame':<8} {'P95':>6} {'GT':>8} {'Dark%':>6} {'F1':>8} {'FP':>10} {'Δold':>8}")
print("-"*70)

for idx, fid in enumerate(sampled):
    c = cv2.imread(str(CUR/f"{fid}_FV.png")); p = cv2.imread(str(PRV/f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GTD/f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY); cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gt = (g>0).astype(np.uint8)*255
    p95 = np.percentile(cv2.absdiff(cg,pg), 95)
    dark_pct = (cv2.GaussianBlur(cg.astype(np.float32),(51,51),20) < 50).mean()
    stats[fid] = {"p95":p95, "gt_px":int(gt.sum()/255), "dark":dark_pct}

    t0 = time.perf_counter(); mask = detect_motion_hybrid_v10(pg,cg)
    times.append(time.perf_counter()-t0); results[fid] = cm(mask, gt)

    r = results[fid]
    old_f1 = old_results.get(fid, 0)
    d = r["f1"] - old_f1
    print(f"{fid:<8} {p95:>6.0f} {stats[fid]['gt_px']:>8,} {dark_pct:>5.1%} {r['f1']:>8.4f} {r['fp']:>10,} {d:>+8.4f}")

    if (idx+1)%10==0:
        avg = np.mean([results[f]["f1"] for f in sampled[:idx+1]])
        print(f"  --- [{idx+1}/{SAMPLE}] avg F1={avg:.4f} ---")
    gc.collect()

f1s = [results[f]["f1"] for f in sampled]
tp_sum = sum(results[f]["tp"] for f in sampled)
fp_sum = sum(results[f]["fp"] for f in sampled)

print(f"\n{'='*80}  SUMMARY")
print(f"  V10 (adaptive): F1={np.mean(f1s):.4f}  P={tp_sum/(tp_sum+fp_sum+1):.4f}  "
      f"R={np.mean([results[f]['recall'] for f in sampled]):.4f}  TP={tp_sum:,}  FP={fp_sum:,}  {np.mean(times):.2f}s")

if old_results:
    old_f1s = [old_results[f] for f in sampled]
    print(f"  Old V10 (min):   F1={np.mean(old_f1s):.4f}")
    print(f"  Δ: {np.mean(f1s)-np.mean(old_f1s):+.4f}")
    wins = sum(1 for i,f in enumerate(sampled) if f1s[i] >= old_f1s[i])
    print(f"  Frame wins vs old: {wins}/{SAMPLE}")

# Save
with open(OUT/"v10b_results.txt","w") as f:
    for fid in sampled:
        r = results[fid]; s = stats[fid]
        old_f1 = old_results.get(fid, 0)
        f.write(f"{fid} {s['p95']:.0f} {s['gt_px']} {s['dark']:.3f} {r['f1']:.4f} {old_f1:.4f} {r['fp']}\n")
print(f"  Done!")
