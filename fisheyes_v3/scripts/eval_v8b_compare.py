"""50-frame V8b comparison: V7 vs V8b (CLAHE seeds only)."""
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
OUT = PROJECT_ROOT / "output" / "evaluation" / "v8b_compare"
OUT.mkdir(parents=True, exist_ok=True)

gt_files = sorted(GTD.glob("*_FV.png"))
all_frames = [f.stem.replace("_FV","") for f in gt_files
              if (CUR/f"{f.stem.replace('_FV','')}_FV.png").exists()
              and (PRV/f"{f.stem.replace('_FV','')}_FV_prev.png").exists()]
N = len(all_frames); SAMPLE = 50; step = N/SAMPLE
sampled = [all_frames[int(i*step)] for i in range(SAMPLE)]
if all_frames[-1] not in sampled: sampled[-1] = all_frames[-1]

print(f"  V8b: CLAHE seeds only (V7 vs V8b, {SAMPLE} frames)")

methods = {"V7": lambda pg,cg: detect_motion_hybrid_v7(pg,cg),
           "V8b": lambda pg,cg: detect_motion_hybrid_v8b(pg,cg)}
results = {n:{} for n in methods}; stats = {}; times = {n:[] for n in methods}

print(f"\n{'Frame':<8} {'P95':>6} {'GT_px':>8}  {'V7 F1':>8} {'V8b F1':>8} {'Δ':>9} {'Best':>6}  {'V7 FP':>10} {'V8b FP':>10}")
print("-"*95)

for idx, fid in enumerate(sampled):
    c = cv2.imread(str(CUR/f"{fid}_FV.png")); p = cv2.imread(str(PRV/f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GTD/f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY); cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gt = (g>0).astype(np.uint8)*255
    p95 = np.percentile(cv2.absdiff(cg,pg), 95); stats[fid] = {"p95":p95, "gt_px":int(gt.sum()/255)}

    for mn, fn in methods.items():
        t0 = time.perf_counter(); mask = fn(pg,cg)
        times[mn].append(time.perf_counter()-t0); results[mn][fid] = cm(mask, gt)

    r7=results["V7"][fid]; r8=results["V8b"][fid]; d=r8["f1"]-r7["f1"]
    print(f"{fid:<8} {p95:>6.0f} {stats[fid]['gt_px']:>8,}  {r7['f1']:>8.4f} {r8['f1']:>8.4f} {d:>+9.4f} {'V8b' if d>=0 else 'V7':>6}  {r7['fp']:>10,} {r8['fp']:>10,}")
    if (idx+1)%10==0:
        v7m=np.mean([results["V7"][f]["f1"] for f in sampled[:idx+1]])
        v8m=np.mean([results["V8b"][f]["f1"] for f in sampled[:idx+1]])
        print(f"  --- [{idx+1}/{SAMPLE}] V7={v7m:.4f} V8b={v8m:.4f} Δ={v8m-v7m:+.4f} ---")
    gc.collect()

f7=[results["V7"][f]["f1"] for f in sampled]; f8=[results["V8b"][f]["f1"] for f in sampled]
print(f"\n{'='*80}  SUMMARY")
for mn in ["V7","V8b"]:
    ml=[results[mn][f] for f in sampled]
    tp=sum(r["tp"] for r in ml); fp=sum(r["fp"] for r in ml)
    print(f"  {mn}: F1={np.mean([r['f1'] for r in ml]):.4f}  P={tp/(tp+fp+1):.4f}  "
          f"R={np.mean([r['recall'] for r in ml]):.4f}  TP={tp:,}  FP={fp:,}  {np.mean(times[mn]):.2f}s")
print(f"  Δ V8b-V7: {np.mean(f8)-np.mean(f7):+.4f}  |  wins: V7={sum(1 for i in range(SAMPLE) if f7[i]>f8[i])}  V8b={sum(1 for i in range(SAMPLE) if f8[i]>=f7[i])}")
ds=[(fid,f8[i]-f7[i]) for i,fid in enumerate(sampled)]; ds.sort(key=lambda x:x[1])
print(f"  Top-5 V8b: {', '.join(f'{f}({d:+.4f})' for f,d in ds[-5:][::-1])}")
print(f"  Top-5 V7:  {', '.join(f'{f}({d:+.4f})' for f,d in ds[:5])}")
print(f"  DONE!")
