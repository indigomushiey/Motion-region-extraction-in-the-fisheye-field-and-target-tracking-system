"""Angular Flow evaluation: V12 vs Blend vs Angular-Blend on 20 frames."""
import cv2, numpy as np
from pathlib import Path
import sys, importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(__file__).resolve().parents[2] / "fisheyes_v2" / "fisheye_motion_tracking"
sys.path.insert(0, str(PROJECT_ROOT))

from core.motion_hybrid import detect_motion_hybrid
from core.motion_angular import detect_motion_angular

def _imp(rp, nm):
    fp = V2_ROOT / rp
    s = importlib.util.spec_from_file_location(nm, fp)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

_m = _imp("core/evaluation/metrics.py", "mt")
_f = _imp("core/motion_detection/frame_difference.py", "fd")
cm = _m.compute_metrics; dme = _f.detect_motion_seed_expand; dm12 = _f.detect_motion_seed_expand_v12

DATA = PROJECT_ROOT / "data" / "homework2"
GTD = DATA / "motion_annotation" / "GroudTruth"
CUR = DATA / "rgb_images"; PRV = DATA / "previous_images"
OUT = PROJECT_ROOT / "output" / "evaluation" / "angular"
OUT.mkdir(parents=True, exist_ok=True)

F = ['00000','00001','00002','00003','00004','00005','00006','00007','00013','00016',
     '00039','00041','00053','00022','00026','00027','00033','00037','00059','00065']

print("=" * 100)
print("  ANGULAR FLOW: V12 | HF (pixel hybrid) | AB (angular blend)")
print("=" * 100)

methods = {
    "V12": lambda pg, cg: dm12(pg, cg),
    "HF":  lambda pg, cg: detect_motion_hybrid(pg, cg),
    "AB":  lambda pg, cg: detect_motion_angular(pg, cg),
}
ar, fs = {n: {} for n in methods}, {}

for idx, fid in enumerate(F):
    c = cv2.imread(str(CUR / f"{fid}_FV.png"))
    p = cv2.imread(str(PRV / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GTD / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg, cg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gb = (g > 0)
    fs[fid] = {"p95": np.percentile(cv2.absdiff(cg, pg), 95), "gt": int(gb.sum())}
    for mn, fn in methods.items():
        ar[mn][fid] = cm(fn(pg, cg), gb.astype(np.uint8) * 255)
    print(f"  [{idx+1}/20] {fid} P95={fs[fid]['p95']:.0f}  "
          f"V12={ar['V12'][fid]['f1']:.3f}  HF={ar['HF'][fid]['f1']:.3f}  AB={ar['AB'][fid]['f1']:.3f}")

print("\n" + "=" * 120)
print("  PER-FRAME F1")
print("=" * 120)
hdr = f"{'Frame':<8} {'P95':>5} |"
for m in ["V12","HF","AB"]: hdr += f" {'F1_'+m:>8} {'P_'+m:>8} {'R_'+m:>8} |"
hdr += f" {'Best':>6} {'ΔAB-12':>9} {'ΔAB-HF':>9}"
print(hdr); print("-" * len(hdr))
for fid in F:
    ps = fs[fid]
    rw = f"{fid:<8} {ps['p95']:>5.0f} |"
    best_f, best_m = -1, ""; f1s = {}
    for mn in ["V12","HF","AB"]:
        m = ar[mn][fid]; f1s[mn] = m["f1"]
        rw += f" {m['f1']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} |"
        if m["f1"] > best_f: best_f, best_m = m["f1"], mn
    rw += f" {best_m:>6} {f1s['AB']-f1s['V12']:>+9.4f} {f1s['AB']-f1s['HF']:>+9.4f}"
    print(rw)

print("\n\n" + "=" * 100)
print("  SUMMARY")
print("=" * 100)
print(f"{'Method':<8} {'IoU':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}  "
      f"{'TP_sum':>10} {'FP_sum':>10} {'FN_sum':>10}")
print("-" * 100)
for mn in ["V12","HF","AB"]:
    ml = [ar[mn][f] for f in F]
    av = {k: np.mean([m[k] for m in ml]) for k in ["iou","precision","recall","f1"]}
    tp, fp, fn = (int(sum(m[k] for m in ml)) for k in ["tp","fp","fn"])
    mk = "  <<<" if mn == "AB" else ""
    print(f"{mn:<8} {av['iou']:>8.4f} {av['precision']:>10.4f} "
          f"{av['recall']:>10.4f} {av['f1']:>10.4f}  "
          f"{tp:>10,} {fp:>10,} {fn:>10,}{mk}")

ab_f = [ar["AB"][f]["f1"] for f in F]
bl_f = [ar["HF"][f]["f1"] for f in F]
v12_f = [ar["V12"][f]["f1"] for f in F]
print(f"\n  V12 F1: {np.mean(v12_f):.4f}  HF F1: {np.mean(bl_f):.4f}  AB F1: {np.mean(ab_f):.4f}")
print(f"  Δ AB-V12: {np.mean(ab_f)-np.mean(v12_f):+.4f}  Δ AB-BL: {np.mean(ab_f)-np.mean(bl_f):+.4f}")
ab_w = sum(1 for i,f in enumerate(F) if ab_f[i]>=max(v12_f[i],bl_f[i]))
bl_w = sum(1 for i,f in enumerate(F) if bl_f[i]>max(v12_f[i],ab_f[i]))
v12_w = sum(1 for i,f in enumerate(F) if v12_f[i]>max(bl_f[i],ab_f[i]))
print(f"  Wins: V12={v12_w}  HF={bl_w}  AB={ab_w}")

# Per-frame significant deltas
print("\n\n" + "=" * 100)
print("  AB vs V12 — Significant Changes")
print("=" * 100)
print(f"{'Frame':<8} {'ΔF1':>9} {'ΔPrec':>9} {'ΔRec':>9}  {'ΔTP':>9} {'ΔFP':>9}")
print("-" * 60)
for fid in F:
    a, v = ar["AB"][fid], ar["V12"][fid]
    df1 = a["f1"] - v["f1"]
    if abs(df1) > 0.0005:
        print(f"{fid:<8} {df1:>+9.4f} {a['precision']-v['precision']:>+9.4f} "
              f"{a['recall']-v['recall']:>+9.4f}  "
              f"{a['tp']-v['tp']:>+9,} {a['fp']-v['fp']:>+9,}")

# Vis
print("\n\nGenerating visualization...")
PF = OUT / "per_frame"; PF.mkdir(parents=True, exist_ok=True)

def ce(pred, gt):
    pb, gb = (pred>0), (gt>0)
    out = np.zeros((pred.shape[0],pred.shape[1],3), dtype=np.uint8)
    out[pb & gb]=[0,255,0]; out[pb & ~gb]=[0,0,255]; out[~pb & gb]=[255,0,0]
    return out
def rh(img, th):
    r = th/img.shape[0]; return cv2.resize(img, (int(img.shape[1]*r), th))
def pl(img, txt, y=22, c=(255,255,255)):
    cv2.putText(img, txt, (6,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)
    cv2.putText(img, txt, (5,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

for idx, fid in enumerate(F):
    c = cv2.imread(str(CUR / f"{fid}_FV.png"))
    p = cv2.imread(str(PRV / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GTD / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg, cg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gb = (g > 0)

    mv12 = dm12(pg, cg); mbl = detect_motion_hybrid(pg, cg); mab = detect_motion_angular(pg, cg)
    s = {m: ar[m][fid] for m in ["V12","HF","AB"]}

    RH_ = 220; ors = rh(c, RH_)
    gts = rh(cv2.cvtColor((gb*255).astype(np.uint8), cv2.COLOR_GRAY2BGR), RH_)
    gov = ors.copy(); gm = cv2.resize((gb*255).astype(np.uint8),(ors.shape[1],ors.shape[0]))
    gov[gm>0] = (gov[gm>0]*0.4+np.array([0,255,0])*0.6).astype(np.uint8)
    r0 = np.hstack([ors, gts, gov])
    pl(r0, f"{fid}  P95={fs[fid]['p95']:.0f}  GT={fs[fid]['gt']:,}px")

    e12, ebl, eab = rh(ce(mv12,gb), RH_), rh(ce(mbl,gb), RH_), rh(ce(mab,gb), RH_)
    for e, mn in [(e12,"V12"),(ebl,"HF"),(eab,"AB")]:
        pl(e, f"{mn}  F1={s[mn]['f1']:.3f} P={s[mn]['precision']:.3f} R={s[mn]['recall']:.3f}")
        pl(e, f"TP={s[mn]['tp']:,} FP={s[mn]['fp']:,}", 42)
    r1 = np.hstack([e12, ebl, eab])
    mw = max(r0.shape[1], r1.shape[1])
    r0 = np.hstack([r0, np.zeros((r0.shape[0],mw-r0.shape[1],3),dtype=np.uint8)]) if r0.shape[1]<mw else r0
    r1 = np.hstack([r1, np.zeros((r1.shape[0],mw-r1.shape[1],3),dtype=np.uint8)]) if r1.shape[1]<mw else r1

    lg = np.zeros((30,mw,3),dtype=np.uint8); lg[:]=[30,30,30]
    for j,(txt,col) in enumerate([("Green=TP",(0,255,0)),("Red=FP",(0,0,255)),("Blue=FN",(255,0,0))]):
        cv2.putText(lg, txt, (j*mw//3+10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)
    db = np.zeros((24,mw,3),dtype=np.uint8); db[:]=[45,45,45]
    dt = f"AB vs V12: dF1={s['AB']['f1']-s['V12']['f1']:+.4f}  AB vs HF: dF1={s['AB']['f1']-s['HF']['f1']:+.4f}"
    cv2.putText(db, dt, (10,18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    viz = np.vstack([r0, r1, lg, db])
    op = PF / f"{fid}_angular.png"; cv2.imwrite(str(op), viz)
    print(f"  [{idx+1}/20] {fid} → {op}")

print(f"\n  All: {PF}\nDONE!")
