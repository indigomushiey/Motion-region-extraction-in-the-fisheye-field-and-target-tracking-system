"""Hybrid Flow evaluation: V1 vs V12 vs Hybrid on 20 frames."""
import cv2, numpy as np
from pathlib import Path
import sys, importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(__file__).resolve().parents[2] / "fisheyes_v2" / "fisheye_motion_tracking"

sys.path.insert(0, str(PROJECT_ROOT))
from core.motion_hybrid import detect_motion_hybrid

def _import_v2(rp, nm):
    fp = V2_ROOT / rp
    spec = importlib.util.spec_from_file_location(nm, fp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_v2m = _import_v2("core/evaluation/metrics.py", "m")
_v2f = _import_v2("core/motion_detection/frame_difference.py", "fd")
compute_metrics = _v2m.compute_metrics
detect_motion_seed_expand = _v2f.detect_motion_seed_expand
detect_motion_seed_expand_v12 = _v2f.detect_motion_seed_expand_v12

DATA = V2_ROOT / "data" / "homework2"
GT_DIR = DATA / "motion_annotation" / "GroudTruth"
CURR_DIR = DATA / "rgb_images"
PREV_DIR = DATA / "previous_images"
OUT_DIR = PROJECT_ROOT / "output" / "evaluation" / "hybrid"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_FRAMES = ['00000', '00001', '00002', '00003', '00004', '00005',
              '00006', '00007', '00013', '00016', '00039', '00041', '00053',
              '00022', '00026', '00027', '00033', '00037', '00059', '00065']

print("=" * 100)
print("  MAGNITUDE HFENDING: Center=persp mag, Edge=fish mag")
print("  Methods: V1 | V12 | HF (hybrided)")
print(f"  Frames: {len(ALL_FRAMES)}")
print("=" * 100)

# ═══════════════════════════════════════════════════════════════════
methods = {
    "V1":  lambda pg, cg: detect_motion_seed_expand(pg, cg),
    "V12": lambda pg, cg: detect_motion_seed_expand_v12(pg, cg),
    "HF":  lambda pg, cg: detect_motion_hybrid(pg, cg),
}

all_results, frame_stats = {n: {} for n in methods}, {}

print("\nProcessing frames...")
for idx, fid in enumerate(ALL_FRAMES):
    c = cv2.imread(str(CURR_DIR / f"{fid}_FV.png"))
    p = cv2.imread(str(PREV_DIR / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GT_DIR / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg, cg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gt_bin = (g > 0)

    diff = cv2.absdiff(cg, pg)
    frame_stats[fid] = {"p95": np.percentile(diff, 95), "gt_px": int(gt_bin.sum())}

    for mn, fn in methods.items():
        mask = fn(pg, cg)
        all_results[mn][fid] = compute_metrics(mask, gt_bin.astype(np.uint8) * 255)

    v1f, v12f, blf = (all_results[m][fid]['f1'] for m in ["V1", "V12", "HF"])
    print(f"  [{idx+1}/{len(ALL_FRAMES)}] {fid}  P95={frame_stats[fid]['p95']:.0f}"
          f"  V1={v1f:.3f}  V12={v12f:.3f}  HF={blf:.3f}")
print("Done.\n")

# ═══════════════════════════════════════════════════════════════════
# Per-frame table
# ═══════════════════════════════════════════════════════════════════
print("=" * 120)
print("  PER-FRAME F1 COMPARISON")
print("=" * 120)
hdr = f"{'Frame':<8} {'P95':>6} {'GT_px':>7} |"
for m in ["V1", "V12", "HF"]: hdr += f" {'F1_'+m:>8} {'P_'+m:>8} {'R_'+m:>8} |"
hdr += f" {'Best':>8} {'ΔHF-12':>9} {'ΔHF-V1':>9}"
print(hdr); print("-" * len(hdr))

for fid in ALL_FRAMES:
    ps = frame_stats[fid]
    row = f"{fid:<8} {ps['p95']:>6.0f} {ps['gt_px']:>7,} |"
    best_f, best_m = -1, ""; f1s = {}
    for mn in ["V1", "V12", "HF"]:
        m = all_results[mn][fid]; f1s[mn] = m["f1"]
        row += f" {m['f1']:>8.4f} {m['precision']:>8.4f} {m['recall']:>8.4f} |"
        if m["f1"] > best_f: best_f, best_m = m["f1"], mn
    row += f" {best_m:>8} {f1s['HF']-f1s['V12']:>+9.4f} {f1s['HF']-f1s['V1']:>+9.4f}"
    print(row)

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  AGGREGATED SUMMARY")
print("=" * 100)
print(f"{'Method':<8} {'IoU':>8} {'Precision':>10} {'Recall':>10} {'F1':>10}  "
      f"{'TP_sum':>10} {'FP_sum':>10} {'FN_sum':>10}")
print("-" * 100)
for mn in ["V1", "V12", "HF"]:
    ml = [all_results[mn][f] for f in ALL_FRAMES]
    avg = {k: np.mean([m[k] for m in ml]) for k in ["iou", "precision", "recall", "f1"]}
    tp_s, fp_s, fn_s = (int(sum(m[k] for m in ml)) for k in ["tp", "fp", "fn"])
    mk = "  <<< BEST" if mn == "HF" else ""
    print(f"{mn:<8} {avg['iou']:>8.4f} {avg['precision']:>10.4f} "
          f"{avg['recall']:>10.4f} {avg['f1']:>10.4f}  "
          f"{tp_s:>10,} {fp_s:>10,} {fn_s:>10,}{mk}")

# ═══════════════════════════════════════════════════════════════════
# Deltas
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  DELTA ANALYSIS")
print("=" * 100)
bl_f = [all_results["HF"][f]["f1"] for f in ALL_FRAMES]
v12_f = [all_results["V12"][f]["f1"] for f in ALL_FRAMES]
v1_f = [all_results["V1"][f]["f1"] for f in ALL_FRAMES]
print(f"  V1  mean F1: {np.mean(v1_f):.4f}  (std: {np.std(v1_f):.4f})")
print(f"  V12 mean F1: {np.mean(v12_f):.4f}  (std: {np.std(v12_f):.4f})")
print(f"  HF  mean F1: {np.mean(bl_f):.4f}  (std: {np.std(bl_f):.4f})")
print(f"  Δ HF–V12:    {np.mean(bl_f)-np.mean(v12_f):+.4f}")
print(f"  Δ HF–V1:     {np.mean(bl_f)-np.mean(v1_f):+.4f}")

bl_w = sum(1 for i,f in enumerate(ALL_FRAMES) if bl_f[i]>=max(v1_f[i],v12_f[i]))
v12_w = sum(1 for i,f in enumerate(ALL_FRAMES) if v12_f[i]>max(v1_f[i],bl_f[i]))
v1_w = sum(1 for i,f in enumerate(ALL_FRAMES) if v1_f[i]>max(v12_f[i],bl_f[i]))
print(f"\n  Frame wins: V1={v1_w}  V12={v12_w}  HF={bl_w}")

bl_tp = sum(all_results["HF"][f]["tp"] for f in ALL_FRAMES)
bl_fp = sum(all_results["HF"][f]["fp"] for f in ALL_FRAMES)
v12_tp = sum(all_results["V12"][f]["tp"] for f in ALL_FRAMES)
v12_fp = sum(all_results["V12"][f]["fp"] for f in ALL_FRAMES)
print(f"\n  TP: V12={v12_tp:,}  HF={bl_tp:,}  ({(bl_tp/v12_tp-1)*100:+.1f}%)")
print(f"  FP: V12={v12_fp:,}  HF={bl_fp:,}  ({(bl_fp/v12_fp-1)*100:+.1f}%)")

# ═══════════════════════════════════════════════════════════════════
# Per-frame significant deltas
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  HF vs V12 — Significant Changes")
print("=" * 100)
print(f"{'Frame':<8} {'P95':>6} {'ΔF1':>9} {'ΔPrec':>9} {'ΔRec':>9}  "
      f"{'ΔTP':>9} {'ΔFP':>9} {'ΔFN':>9}")
print("-" * 80)
for fid in ALL_FRAMES:
    bl, v12 = all_results["HF"][fid], all_results["V12"][fid]
    df1 = bl["f1"] - v12["f1"]
    if abs(df1) > 0.0005:
        print(f"{fid:<8} {frame_stats[fid]['p95']:>6.0f} {df1:>+9.4f} "
              f"{bl['precision']-v12['precision']:>+9.4f} "
              f"{bl['recall']-v12['recall']:>+9.4f}  "
              f"{bl['tp']-v12['tp']:>+9,} {bl['fp']-v12['fp']:>+9,} "
              f"{bl['fn']-v12['fn']:>+9,}")

# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 100)
print("  GENERATING VISUALIZATION...")
print("=" * 100)

PF = OUT_DIR / "per_frame"; PF.mkdir(parents=True, exist_ok=True)

def ce(pred, gt):
    pb, gb = (pred>0), (gt>0)
    out = np.zeros((pred.shape[0],pred.shape[1],3), dtype=np.uint8)
    out[pb & gb] = [0,255,0]; out[pb & ~gb] = [0,0,255]; out[~pb & gb] = [255,0,0]
    return out

def rh(img, th):
    r = th/img.shape[0]; return cv2.resize(img, (int(img.shape[1]*r), th))

def pl(img, txt, y=22, c=(255,255,255)):
    cv2.putText(img, txt, (6,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 2)
    cv2.putText(img, txt, (5,y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

for idx, fid in enumerate(ALL_FRAMES):
    c = cv2.imread(str(CURR_DIR / f"{fid}_FV.png"))
    p = cv2.imread(str(PREV_DIR / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GT_DIR / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg, cg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gb = (g > 0)

    mv1 = detect_motion_seed_expand(pg, cg)
    mv12 = detect_motion_seed_expand_v12(pg, cg)
    mbl = detect_motion_hybrid(pg, cg)

    s = {m: all_results[m][fid] for m in ["V1","V12","HF"]}
    RH_ = 220; ors = rh(c, RH_)
    gts = rh(cv2.cvtColor((gb*255).astype(np.uint8), cv2.COLOR_GRAY2BGR), RH_)
    gov = ors.copy(); gm = cv2.resize((gb*255).astype(np.uint8),(ors.shape[1],ors.shape[0]))
    gov[gm>0] = (gov[gm>0]*0.4+np.array([0,255,0])*0.6).astype(np.uint8)
    r0 = np.hstack([ors, gts, gov])
    pl(r0, f"{fid}  P95={frame_stats[fid]['p95']:.0f}  GT={frame_stats[fid]['gt_px']:,}px")

    ev1, ev12, ebl = rh(ce(mv1,gb), RH_), rh(ce(mv12,gb), RH_), rh(ce(mbl,gb), RH_)
    for e, mn in [(ev1,"V1"),(ev12,"V12"),(ebl,"HF")]:
        pl(e, f"{mn}  F1={s[mn]['f1']:.3f} P={s[mn]['precision']:.3f} R={s[mn]['recall']:.3f}")
        pl(e, f"TP={s[mn]['tp']:,} FP={s[mn]['fp']:,}", 42)
    r1 = np.hstack([ev1, ev12, ebl])
    mw = max(r0.shape[1], r1.shape[1])
    for ri in [r0, r1]:
        if ri.shape[1] < mw:
            pad = np.zeros((ri.shape[0], mw-ri.shape[1], 3), dtype=np.uint8)
    r0 = np.hstack([r0, np.zeros((r0.shape[0],mw-r0.shape[1],3),dtype=np.uint8)]) if r0.shape[1]<mw else r0
    r1 = np.hstack([r1, np.zeros((r1.shape[0],mw-r1.shape[1],3),dtype=np.uint8)]) if r1.shape[1]<mw else r1

    lg = np.zeros((30,mw,3),dtype=np.uint8); lg[:]=[30,30,30]
    for j,(txt,col) in enumerate([("Green=TP",(0,255,0)),("Red=FP",(0,0,255)),("Blue=FN",(255,0,0))]):
        cv2.putText(lg, txt, (j*mw//3+10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 2)
    db = np.zeros((24,mw,3),dtype=np.uint8); db[:]=[45,45,45]
    dt = f"HF vs V12: dF1={s['HF']['f1']-s['V12']['f1']:+.4f}  dFP={s['HF']['fp']-s['V12']['fp']:+,}"
    cv2.putText(db, dt, (10,18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)

    viz = np.vstack([r0, r1, lg, db])
    op = PF / f"{fid}_hybrid.png"; cv2.imwrite(str(op), viz)
    print(f"  [{idx+1}/{len(ALL_FRAMES)}] {fid} → {op}")

print(f"\n  All images: {PF}")
print("\n" + "=" * 100)
print("  DONE!")
print("=" * 100)
