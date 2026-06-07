"""V10 final visualization: V3 vs V7 vs V10 on all 50 frames."""
import cv2, numpy as np, sys, time, gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from core.motion_hybrid import detect_motion_hybrid_v3, detect_motion_hybrid_v7, detect_motion_hybrid_v10

DATA = PROJECT_ROOT / "data" / "homework2"
CUR = DATA / "rgb_images"; PRV = DATA / "previous_images"
GTD = DATA / "motion_annotation" / "GroudTruth"
OUT = PROJECT_ROOT / "output" / "evaluation" / "v10_50frames"
OUT.mkdir(parents=True, exist_ok=True)

gt_files = sorted(GTD.glob("*_FV.png"))
all_frames = [f.stem.replace("_FV","") for f in gt_files
              if (CUR/f"{f.stem.replace('_FV','')}_FV.png").exists()
              and (PRV/f"{f.stem.replace('_FV','')}_FV_prev.png").exists()]
N = len(all_frames); SAMPLE = 50; step = N/SAMPLE
sampled = [all_frames[int(i*step)] for i in range(SAMPLE)]
if all_frames[-1] not in sampled: sampled[-1] = all_frames[-1]

def rh(img, th):
    r = th / img.shape[0]; return cv2.resize(img, (int(img.shape[1] * r), th))

def pl(img, txt, y=20, c=(255,255,255)):
    cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0,0,0), 2)
    cv2.putText(img, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, c, 1)

def ce(pred, gt):
    pb, gb = (pred>0), (gt>0)
    out = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    out[pb&gb] = [0,255,0]; out[pb&~gb] = [0,0,255]; out[~pb&gb] = [255,0,0]
    return out

def cm_quick(mask, gt):
    pb = (mask>0); gb = (gt>0)
    tp = int((pb&gb).sum()); fp = int((pb&~gb).sum()); fn = int((~pb&gb).sum())
    p = tp/(tp+fp+1e-6); r = tp/(tp+fn+1e-6)
    return {"f1":2*p*r/(p+r+1e-6), "precision":p, "recall":r, "tp":tp, "fp":fp, "fn":fn}

methods = {"V3":detect_motion_hybrid_v3, "V7":detect_motion_hybrid_v7, "V10":detect_motion_hybrid_v10}
colors = {"V3":(46,204,113), "V7":(52,152,219), "V10":(155,89,182)}

print(f"V10 50-Frame Viz: V3 vs V7 vs V10\n{SAMPLE} frames\n")
RH = 175

for idx, fid in enumerate(sampled):
    c = cv2.imread(str(CUR/f"{fid}_FV.png")); p = cv2.imread(str(PRV/f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GTD/f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY); cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gt = (g>0).astype(np.uint8)*255; gb = (g>0)
    p95 = np.percentile(cv2.absdiff(cg,pg), 95); gt_px = int(gt.sum()/255)

    masks = {}; metrics = {}
    for mn, fn in methods.items():
        masks[mn] = fn(pg, cg)
        metrics[mn] = cm_quick(masks[mn], gt)

    # Row 0: Input
    curr_s = rh(c, RH); gto = curr_s.copy()
    gm = rh(gt, RH); gto[gm>0] = (gto[gm>0]*0.4 + np.array([0,255,0])*0.6).astype(np.uint8)
    diff_h = rh(cv2.applyColorMap(cv2.absdiff(cg,pg), cv2.COLORMAP_HOT), RH)
    r0 = np.hstack([curr_s, gto, diff_h])
    pl(r0, f"{fid}  P95={p95:.0f}  GT={gt_px:,}px")

    # Row 1: Error maps V3 | V7 | V10
    errs = [rh(ce(masks[mn], gb), RH) for mn in methods]
    r1 = np.hstack(errs)
    for i, mn in enumerate(methods):
        off = i * errs[0].shape[1]; m = metrics[mn]
        pl(r1[:, off:], f"{mn} F1={m['f1']:.4f} P={m['precision']:.3f} R={m['recall']:.3f}", 18)
        pl(r1[:, off:], f"TP={m['tp']:,} FP={m['fp']:,}", 38)

    # Row 2: Pred masks
    preds = [rh(cv2.cvtColor(masks[mn], cv2.COLOR_GRAY2BGR), RH) for mn in methods]
    r2 = np.hstack(preds)

    # Assemble
    rows = [r0, r1, r2]; mw = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        pw = mw - r.shape[1]
        padded.append(np.hstack([r, np.zeros((r.shape[0], pw, 3), dtype=np.uint8)]) if pw > 0 else r)

    lg = np.zeros((24, mw, 3), dtype=np.uint8); lg[:] = [30,30,30]
    for j, (txt, col) in enumerate([("Green=TP",(0,255,0)),("Red=FP",(0,0,255)),("Blue=FN",(255,0,0))]):
        cv2.putText(lg, txt, (10+j*mw//3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 2)
    db = np.zeros((22, mw, 3), dtype=np.uint8); db[:] = [40,40,40]
    d73 = metrics["V7"]["f1"] - metrics["V3"]["f1"]
    d107 = metrics["V10"]["f1"] - metrics["V7"]["f1"]
    cv2.putText(db, f"V7-V3={d73:+.4f}  |  V10-V7={d107:+.4f}  |  V10-V3={metrics['V10']['f1']-metrics['V3']['f1']:+.4f}",
                (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (200,200,200), 1)

    cv2.imwrite(str(OUT/f"{fid}_v10.png"), np.vstack([padded[0], padded[1], padded[2], lg, db]))
    print(f"[{idx+1:>2}/{SAMPLE}] {fid}" + ("  V10 best!" if d107 >= 0 else "  V7 better"))
    gc.collect()

print(f"\nDone! {SAMPLE} images → {OUT}/")
