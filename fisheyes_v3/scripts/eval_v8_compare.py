"""50-frame V8 comparison: V3 vs V7 vs V8 (CLAHE + dual Canny)."""
import cv2, numpy as np, sys, importlib.util, time, gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(__file__).resolve().parents[2] / "fisheyes_v2" / "fisheye_motion_tracking"
sys.path.insert(0, str(PROJECT_ROOT))
from core.motion_hybrid import detect_motion_hybrid_v7, detect_motion_hybrid_v8b

def _i(rp, nm):
    fp = V2_ROOT / rp; s = importlib.util.spec_from_file_location(nm, fp)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
cm = _i("core/evaluation/metrics.py", "mt").compute_metrics

DATA = PROJECT_ROOT / "data" / "homework2"
GTD = DATA / "motion_annotation" / "GroudTruth"; CUR = DATA / "rgb_images"; PRV = DATA / "previous_images"
OUT = PROJECT_ROOT / "output" / "evaluation" / "v8_compare"
OUT.mkdir(parents=True, exist_ok=True)

gt_files = sorted(GTD.glob("*_FV.png"))
all_frames = [f.stem.replace("_FV","") for f in gt_files
              if (CUR/f"{f.stem.replace('_FV','')}_FV.png").exists()
              and (PRV/f"{f.stem.replace('_FV','')}_FV_prev.png").exists()]
N = len(all_frames); SAMPLE = 50; step = N/SAMPLE
sampled = [all_frames[int(i*step)] for i in range(SAMPLE)]
if all_frames[-1] not in sampled: sampled[-1] = all_frames[-1]

print(f"{'='*95}\n  V8: CLAHE + Dual Canny  (V3 vs V7 vs V8, {SAMPLE} frames)\n{'='*95}")

methods = {"V3": lambda pg,cg: detect_motion_hybrid_v3(pg,cg),
           "V7": lambda pg,cg: detect_motion_hybrid_v7(pg,cg),
           "V8": lambda pg,cg: detect_motion_hybrid_v8(pg,cg)}
results = {n:{} for n in methods}; stats = {}; times = {n:[] for n in methods}

header = f"{'Frame':<8} {'P95':>6} {'GT':>8}  {'V3':>8} {'V7':>8} {'V8':>8} {'Δ8-7':>8} {'Best':>6}"
print(f"\n{header}\n{'-'*len(header)}")

for idx, fid in enumerate(sampled):
    c = cv2.imread(str(CUR/f"{fid}_FV.png")); p = cv2.imread(str(PRV/f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GTD/f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY); cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gt = (g>0).astype(np.uint8)*255
    p95 = np.percentile(cv2.absdiff(cg,pg), 95); stats[fid] = {"p95":p95, "gt_px":int(gt.sum()/255)}

    for mn, fn in methods.items():
        t0 = time.perf_counter(); mask = fn(pg,cg)
        times[mn].append(time.perf_counter()-t0); results[mn][fid] = cm(mask, gt)

    f3=results["V3"][fid]["f1"]; f7=results["V7"][fid]["f1"]; f8=results["V8"][fid]["f1"]
    d87=f8-f7; best = "V8" if f8>=max(f3,f7) else ("V7" if f7>=f3 else "V3")
    print(f"{fid:<8} {p95:>6.0f} {stats[fid]['gt_px']:>8,}  {f3:>8.4f} {f7:>8.4f} {f8:>8.4f} {d87:>+8.4f} {best:>6}")

    if (idx+1)%10==0:
        avgs = {m: np.mean([results[m][f]["f1"] for f in sampled[:idx+1]]) for m in methods}
        print(f"  --- [{idx+1}/{SAMPLE}] V3={avgs['V3']:.4f} V7={avgs['V7']:.4f} V8={avgs['V8']:.4f} ---")
    gc.collect()

# Summary
print(f"\n{'='*95}  SUMMARY")
for mn in ["V3","V7","V8"]:
    ml=[results[mn][f] for f in sampled]
    tp=sum(r["tp"] for r in ml); fp=sum(r["fp"] for r in ml)
    print(f"  {mn}: F1={np.mean([r['f1'] for r in ml]):.4f}  P={tp/(tp+fp+1):.4f}  "
          f"R={np.mean([r['recall'] for r in ml]):.4f}  TP={tp:,}  FP={fp:,}  {np.mean(times[mn]):.2f}s")

f3=[results["V3"][f]["f1"] for f in sampled]
f7=[results["V7"][f]["f1"] for f in sampled]
f8=[results["V8"][f]["f1"] for f in sampled]
print(f"  Δ V7-V3: {np.mean(f7)-np.mean(f3):+.4f}  |  Δ V8-V7: {np.mean(f8)-np.mean(f7):+.4f}  |  Δ V8-V3: {np.mean(f8)-np.mean(f3):+.4f}")
print(f"  Frame wins: V3={sum(1 for i in range(SAMPLE) if f3[i]>=max(f7[i],f8[i]))}  "
      f"V7={sum(1 for i in range(SAMPLE) if f7[i]>max(f3[i],f8[i]))}  "
      f"V8={sum(1 for i in range(SAMPLE) if f8[i]>max(f3[i],f7[i]))}")

with open(OUT/"v8_compare_results.txt","w",encoding="utf-8") as f:
    for fid in sampled:
        s=stats[fid]; r3=results["V3"][fid]; r7=results["V7"][fid]; r8=results["V8"][fid]
        f.write(f"{fid} {s['p95']:.0f} {s['gt_px']} {r3['f1']:.4f} {r7['f1']:.4f} {r8['f1']:.4f}\n")
print(f"  Results: {OUT/'v8_compare_results.txt'}")
print(f"  DONE!")
