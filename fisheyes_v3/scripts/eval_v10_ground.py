"""50-frame: V10 + ground plane mask evaluation."""
import sys; sys.path.insert(0,'.')
import core.motion_hybrid as mh
mh._GROUND_MASK = None  # force recompute with corrected mask

import cv2, numpy as np, importlib.util, time, gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(__file__).resolve().parents[2] / "fisheyes_v2" / "fisheye_motion_tracking"
from core.motion_hybrid import detect_motion_hybrid_v10

def _i(rp, nm):
    fp = V2_ROOT / rp; s = importlib.util.spec_from_file_location(nm, fp)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cm = _i("core/evaluation/metrics.py", "mt").compute_metrics

DATA = PROJECT_ROOT / "data" / "homework2"
GTD = DATA / "motion_annotation" / "GroudTruth"; CUR = DATA / "rgb_images"; PRV = DATA / "previous_images"
OUT = PROJECT_ROOT / "output" / "evaluation" / "v10_ground"
OUT.mkdir(parents=True, exist_ok=True)

gt_files = sorted(GTD.glob("*_FV.png"))
all_frames = [f.stem.replace("_FV","") for f in gt_files
              if (CUR/f"{f.stem.replace('_FV','')}_FV.png").exists()
              and (PRV/f"{f.stem.replace('_FV','')}_FV_prev.png").exists()]
N = len(all_frames); SAMPLE = 50; step = N/SAMPLE
sampled = [all_frames[int(i*step)] for i in range(SAMPLE)]
if all_frames[-1] not in sampled: sampled[-1] = all_frames[-1]

print(f"  V10 + Ground Plane Mask ({SAMPLE} frames)")

results = {}; stats = {}; times = []

# Load old V10 results for comparison
old = {}
try:
    with open(PROJECT_ROOT / "output" / "evaluation" / "v10b_compare" / "v10b_results.txt") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5 and parts[0].isdigit():
                old[parts[0]] = float(parts[3])
except: pass

print(f"\n{'Frame':<8} {'P95':>6} {'GT':>8} {'F1':>8} {'FP':>10} {'Δold':>8}")
print("-"*60)

for idx, fid in enumerate(sampled):
    c = cv2.imread(str(CUR/f"{fid}_FV.png")); p = cv2.imread(str(PRV/f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GTD/f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY); cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gt = (g>0).astype(np.uint8)*255
    p95 = np.percentile(cv2.absdiff(cg,pg), 95)
    stats[fid] = {"p95":p95, "gt_px":int(gt.sum()/255)}

    t0 = time.perf_counter(); mask = detect_motion_hybrid_v10(pg,cg)
    times.append(time.perf_counter()-t0); results[fid] = cm(mask, gt)

    r = results[fid]; of = old.get(fid, 0); d = r["f1"] - of
    print(f"{fid:<8} {p95:>6.0f} {stats[fid]['gt_px']:>8,} {r['f1']:>8.4f} {r['fp']:>10,} {d:>+8.4f}")

    if (idx+1)%10==0:
        avg = np.mean([results[f]["f1"] for f in sampled[:idx+1]])
        print(f"  --- [{idx+1}/{SAMPLE}] avg F1={avg:.4f} ---")
    gc.collect()

f1s = [results[f]["f1"] for f in sampled]
tp = sum(results[f]["tp"] for f in sampled)
fp = sum(results[f]["fp"] for f in sampled)
print(f"\n{'='*80}  SUMMARY")
print(f"  V10+ground: F1={np.mean(f1s):.4f}  P={tp/(tp+fp+1):.4f}  "
      f"R={np.mean([results[f]['recall'] for f in sampled]):.4f}  TP={tp:,}  FP={fp:,}  {np.mean(times):.2f}s")
if old:
    of1s = [old[f] for f in sampled]
    print(f"  Old V10:   F1={np.mean(of1s):.4f}")
    print(f"  Δ: {np.mean(f1s)-np.mean(of1s):+.4f}")
    wins = sum(1 for i,f in enumerate(sampled) if f1s[i] >= of1s[i])
    print(f"  Frame wins: {wins}/{SAMPLE}")

with open(OUT/"v10_ground_results.txt","w") as f:
    for fid in sampled:
        r = results[fid]; s = stats[fid]; of = old.get(fid, 0)
        f.write(f"{fid} {s['p95']:.0f} {s['gt_px']} {r['f1']:.4f} {of:.4f} {r['fp']}\n")
print(f"  Done!")
