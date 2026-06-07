"""50-frame V6 comparison: V3 vs V6 (edge correlation)."""
import cv2, numpy as np, sys, importlib.util, time, gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(__file__).resolve().parents[2] / "fisheyes_v2" / "fisheye_motion_tracking"
sys.path.insert(0, str(PROJECT_ROOT))
from core.motion_hybrid import detect_motion_hybrid_v3, detect_motion_hybrid_v6

def _i(rp, nm):
    fp = V2_ROOT / rp
    s = importlib.util.spec_from_file_location(nm, fp); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
cm = _i("core/evaluation/metrics.py", "mt").compute_metrics

DATA = PROJECT_ROOT / "data" / "homework2"
GTD = DATA / "motion_annotation" / "GroudTruth"; CUR = DATA / "rgb_images"; PRV = DATA / "previous_images"
OUT = PROJECT_ROOT / "output" / "evaluation" / "v6_compare"
OUT.mkdir(parents=True, exist_ok=True)
PF = OUT / "per_frame"; PF.mkdir(parents=True, exist_ok=True)

gt_files = sorted(GTD.glob("*_FV.png"))
all_frames = [f.stem.replace("_FV","") for f in gt_files
              if (CUR/f"{f.stem.replace('_FV','')}_FV.png").exists()
              and (PRV/f"{f.stem.replace('_FV','')}_FV_prev.png").exists()]
N = len(all_frames); SAMPLE = 50; step = N/SAMPLE
sampled = [all_frames[int(i*step)] for i in range(SAMPLE)]
if all_frames[-1] not in sampled: sampled[-1] = all_frames[-1]

print(f"{'='*80}\n  V6: Tiered Edge Correlation  (V3 vs V6, {SAMPLE} frames)\n{'='*80}")

methods = {"V3": lambda pg,cg: detect_motion_hybrid_v3(pg,cg),
           "V6": lambda pg,cg: detect_motion_hybrid_v6(pg,cg)}
results = {n:{} for n in methods}; stats = {}; times = {n:[] for n in methods}

print(f"\n{'Frame':<8} {'P95':>6} {'GT_px':>8}  {'V3 F1':>8} {'V6 F1':>8} {'Δ':>9} {'Best':>6}  {'V3 FP':>10} {'V6 FP':>10}")
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

    r3=results["V3"][fid]; r6=results["V6"][fid]; d=r6["f1"]-r3["f1"]
    print(f"{fid:<8} {p95:>6.0f} {stats[fid]['gt_px']:>8,}  {r3['f1']:>8.4f} {r6['f1']:>8.4f} {d:>+9.4f} {'V6' if d>=0 else 'V3':>6}  {r3['fp']:>10,} {r6['fp']:>10,}")

    if (idx+1)%10==0:
        v3m=np.mean([results["V3"][f]["f1"] for f in sampled[:idx+1]])
        v6m=np.mean([results["V6"][f]["f1"] for f in sampled[:idx+1]])
        print(f"  --- [{idx+1}/{SAMPLE}] avg: V3={v3m:.4f} V6={v6m:.4f} Δ={v6m-v3m:+.4f} ---")
    gc.collect()

v3f=[results["V3"][f]["f1"] for f in sampled]; v6f=[results["V6"][f]["f1"] for f in sampled]
print(f"\n{'='*80}  SUMMARY")
for mn in ["V3","V6"]:
    ml=[results[mn][f] for f in sampled]
    tp=sum(r["tp"] for r in ml); fp=sum(r["fp"] for r in ml)
    print(f"  {mn}: F1={np.mean([r['f1'] for r in ml]):.4f}  P={tp/(tp+fp+1):.4f}  "
          f"R={np.mean([r['recall'] for r in ml]):.4f}  TP={tp:,}  FP={fp:,}  {np.mean(times[mn]):.2f}s")
print(f"  Δ V6-V3: F1={np.mean(v6f)-np.mean(v3f):+.4f}")
print(f"  Frame wins: V3={sum(1 for i in range(SAMPLE) if v3f[i]>v6f[i])}  V6={sum(1 for i in range(SAMPLE) if v6f[i]>=v3f[i])}")

ds=[(fid,v6f[i]-v3f[i]) for i,fid in enumerate(sampled)]; ds.sort(key=lambda x:x[1])
print(f"  Top-5 V6: {', '.join(f'{f}({d:+.4f})' for f,d in ds[-5:][::-1])}")
print(f"  Top-5 V3: {', '.join(f'{f}({d:+.4f})' for f,d in ds[:5])}")

# Save text
with open(OUT/"v6_compare_results.txt","w",encoding="utf-8") as f:
    for fid in sampled:
        s=stats[fid]; r3=results["V3"][fid]; r6=results["V6"][fid]
        f.write(f"{fid} {s['p95']:.0f} {s['gt_px']} {r3['f1']:.4f} {r6['f1']:.4f} {r6['f1']-r3['f1']:+.4f} {r3['fp']} {r6['fp']}\n")
print(f"  Results: {OUT/'v6_compare_results.txt'}")

# Quick viz
def rh(i,t): r=t/i.shape[0]; return cv2.resize(i,(int(i.shape[1]*r),t))
def pl(i,t,y=22,c=(255,255,255)): cv2.putText(i,t,(6,y),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),2); cv2.putText(i,t,(5,y),cv2.FONT_HERSHEY_SIMPLEX,.4,c,1)
def ce(p,g):
    pb,gb=(p>0),(g>0); o=np.zeros((p.shape[0],p.shape[1],3),dtype=np.uint8)
    o[pb&gb]=[0,255,0]; o[pb&~gb]=[0,0,255]; o[~pb&gb]=[255,0,0]; return o

print(f"\n  Generating viz..."); RH_=200
for idx,fid in enumerate(sampled):
    c=cv2.imread(str(CUR/f"{fid}_FV.png")); g=cv2.imread(str(GTD/f"{fid}_FV.png"),cv2.IMREAD_GRAYSCALE)
    pg=cv2.cvtColor(cv2.imread(str(PRV/f"{fid}_FV_prev.png")),cv2.COLOR_BGR2GRAY)
    cg=cv2.cvtColor(c,cv2.COLOR_BGR2GRAY); gb=(g>0)
    mv3=detect_motion_hybrid_v3(pg,cg); mv6=detect_motion_hybrid_v6(pg,cg)
    r3=results["V3"][fid]; r6=results["V6"][fid]

    cs=rh(c,RH_); gto=cs.copy(); gm=rh((gb.astype(np.uint8)*255),RH_)
    gto[gm>0]=(gto[gm>0]*.4+np.array([0,255,0])*.6).astype(np.uint8)
    r0=np.hstack([cs,gto,rh(cv2.applyColorMap(cv2.absdiff(cg,pg),cv2.COLORMAP_HOT),RH_)])
    pl(r0,f"{fid} P95={stats[fid]['p95']:.0f} GT={stats[fid]['gt_px']:,}px")

    e3=rh(ce(mv3,gb),RH_); e6=rh(ce(mv6,gb),RH_)
    r1=np.hstack([e3,e6])
    pl(r1,f"V3 F1={r3['f1']:.4f} P={r3['precision']:.4f} R={r3['recall']:.4f}")
    pl(r1[:,e3.shape[1]:],f"V6 F1={r6['f1']:.4f} P={r6['precision']:.4f} R={r6['recall']:.4f}")
    pl(e3,f"TP={r3['tp']:,} FP={r3['fp']:,}",42); pl(e6,f"TP={r6['tp']:,} FP={r6['fp']:,}",42)

    p3=rh(cv2.cvtColor(mv3,cv2.COLOR_GRAY2BGR),RH_); p6=rh(cv2.cvtColor(mv6,cv2.COLOR_GRAY2BGR),RH_)
    r2=np.hstack([p3,p6])
    rows=[r0,r1,r2]; mw=max(r.shape[1] for r in rows)
    padded=[np.hstack([r,np.zeros((r.shape[0],mw-r.shape[1],3),dtype=np.uint8)]) if r.shape[1]<mw else r for r in rows]
    lg=np.zeros((24,mw,3),dtype=np.uint8); lg[:]=[30,30,30]
    for j,(t,c) in enumerate([("Green=TP",(0,255,0)),("Red=FP",(0,0,255)),("Blue=FN",(255,0,0))]):
        cv2.putText(lg,t,(10+j*mw//3,18),cv2.FONT_HERSHEY_SIMPLEX,.35,c,2)
    db=np.zeros((22,mw,3),dtype=np.uint8); db[:]=[45,45,45]
    cv2.putText(db,f"Δ V6-V3: F1={r6['f1']-r3['f1']:+.4f} TP={r6['tp']-r3['tp']:+,} FP={r6['fp']-r3['fp']:+,}",
                (10,16),cv2.FONT_HERSHEY_SIMPLEX,.35,(200,200,200),1)
    cv2.imwrite(str(PF/f"{fid}_v6compare.png"),np.vstack([padded[0],padded[1],padded[2],lg,db]))
    print(f"  [{idx+1:>2}/{SAMPLE}] {fid}")
print(f"\n  {'='*80}  DONE!")
