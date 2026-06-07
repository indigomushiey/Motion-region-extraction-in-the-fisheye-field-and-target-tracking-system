"""50-frame V5 comparison: V3 vs V5, with visualization."""
import cv2, numpy as np, sys, importlib.util, time, gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(__file__).resolve().parents[2] / "fisheyes_v2" / "fisheye_motion_tracking"
sys.path.insert(0, str(PROJECT_ROOT))
from core.motion_hybrid import detect_motion_hybrid_v3, detect_motion_hybrid_v5

def _i(rp, nm):
    fp = V2_ROOT / rp
    s = importlib.util.spec_from_file_location(nm, fp); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m

cm = _i("core/evaluation/metrics.py", "mt").compute_metrics

DATA = PROJECT_ROOT / "data" / "homework2"
GTD = DATA / "motion_annotation" / "GroudTruth"; CUR = DATA / "rgb_images"; PRV = DATA / "previous_images"
OUT = PROJECT_ROOT / "output" / "evaluation" / "v5_compare"
OUT.mkdir(parents=True, exist_ok=True)
PF = OUT / "per_frame"; PF.mkdir(parents=True, exist_ok=True)

gt_files = sorted(GTD.glob("*_FV.png"))
all_frames = [f.stem.replace("_FV", "") for f in gt_files
              if (CUR / f"{f.stem.replace('_FV', '')}_FV.png").exists()
              and (PRV / f"{f.stem.replace('_FV', '')}_FV_prev.png").exists()]
N = len(all_frames); SAMPLE = 50; step = N / SAMPLE
sampled = [all_frames[int(i * step)] for i in range(SAMPLE)]
if all_frames[-1] not in sampled: sampled[-1] = all_frames[-1]

print(f"{'='*80}")
print(f"  V5: Local Peak Detection  ({SAMPLE} frames)")
print(f"  V3 vs V5 comparison")
print(f"{'='*80}")

methods = {"V3": lambda pg, cg: detect_motion_hybrid_v3(pg, cg),
           "V5": lambda pg, cg: detect_motion_hybrid_v5(pg, cg)}
all_results = {n: {} for n in methods}; stats = {}; times = {n: [] for n in methods}

print(f"\n{'Frame':<8} {'P95':>6} {'GT_px':>8}  {'V3 F1':>8} {'V5 F1':>8} {'Δ':>9} {'Best':>6}  {'V3 FP':>10} {'V5 FP':>10}")
print("-" * 90)

for idx, fid in enumerate(sampled):
    c = cv2.imread(str(CUR / f"{fid}_FV.png")); p = cv2.imread(str(PRV / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GTD / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY); cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gt = (g > 0).astype(np.uint8) * 255
    p95 = np.percentile(cv2.absdiff(cg, pg), 95); gt_px = int(gt.sum() / 255)
    stats[fid] = {"p95": p95, "gt_px": gt_px}

    for mn, fn in methods.items():
        t0 = time.perf_counter(); mask = fn(pg, cg)
        times[mn].append(time.perf_counter() - t0)
        all_results[mn][fid] = cm(mask, gt)

    r3 = all_results["V3"][fid]; r5 = all_results["V5"][fid]
    d = r5["f1"] - r3["f1"]; best = "V5" if d >= 0 else "V3"
    print(f"{fid:<8} {p95:>6.0f} {gt_px:>8,}  {r3['f1']:>8.4f} {r5['f1']:>8.4f} {d:>+9.4f} {best:>6}  {r3['fp']:>10,} {r5['fp']:>10,}")

    if (idx + 1) % 10 == 0:
        v3m = np.mean([all_results["V3"][f]["f1"] for f in sampled[:idx+1]])
        v5m = np.mean([all_results["V5"][f]["f1"] for f in sampled[:idx+1]])
        print(f"  --- [{idx+1}/{SAMPLE}] avg: V3={v3m:.4f} V5={v5m:.4f} Δ={v5m-v3m:+.4f} ---")
    gc.collect()

# Summary
v3f = [all_results["V3"][f]["f1"] for f in sampled]
v5f = [all_results["V5"][f]["f1"] for f in sampled]

print(f"\n{'='*80}  SUMMARY")
for mn in ["V3", "V5"]:
    ml = [all_results[mn][f] for f in sampled]
    tp = sum(r["tp"] for r in ml); fp = sum(r["fp"] for r in ml)
    print(f"  {mn}: F1={np.mean([r['f1'] for r in ml]):.4f}  P={tp/(tp+fp+1):.4f}  "
          f"R={np.mean([r['recall'] for r in ml]):.4f}  TP={tp:,}  FP={fp:,}  {np.mean(times[mn]):.2f}s")
print(f"  Δ V5-V3: F1={np.mean(v5f)-np.mean(v3f):+.4f}")
print(f"  Frame wins: V3={sum(1 for i in range(SAMPLE) if v3f[i]>v5f[i])}  "
      f"V5={sum(1 for i in range(SAMPLE) if v5f[i]>=v3f[i])}")

deltas = [(fid, v5f[i]-v3f[i]) for i, fid in enumerate(sampled)]; deltas.sort(key=lambda x: x[1])
print(f"  Top-5 V5: {', '.join(f'{f}({d:+.4f})' for f,d in deltas[-5:][::-1])}")
print(f"  Top-5 V3: {', '.join(f'{f}({d:+.4f})' for f,d in deltas[:5])}")

# Viz shortcuts
def rh(img, th):
    r = th / img.shape[0]; return cv2.resize(img, (int(img.shape[1] * r), th))
def pl(img, txt, y=22, c=(255, 255, 255)):
    cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
    cv2.putText(img, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
def ce(pred, gt):
    pb, gb = (pred > 0), (gt > 0)
    out = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    out[pb & gb] = [0, 255, 0]; out[pb & ~gb] = [0, 0, 255]; out[~pb & gb] = [255, 0, 0]
    return out

print(f"\n{'='*80}  GENERATING VIZ"); RH_ = 200
for idx, fid in enumerate(sampled):
    c = cv2.imread(str(CUR / f"{fid}_FV.png")); p = cv2.imread(str(PRV / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GTD / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY); cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY); gb = (g > 0)
    mv3 = detect_motion_hybrid_v3(pg, cg); mv5 = detect_motion_hybrid_v5(pg, cg)
    r3 = all_results["V3"][fid]; r5 = all_results["V5"][fid]

    curr_s = rh(c, RH_); gt_ov = curr_s.copy()
    gm = rh((gb.astype(np.uint8) * 255), RH_)
    gt_ov[gm > 0] = (gt_ov[gm > 0] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
    r0 = np.hstack([curr_s, gt_ov, rh(cv2.applyColorMap(cv2.absdiff(cg, pg), cv2.COLORMAP_HOT), RH_)])
    pl(r0, f"{fid}  P95={stats[fid]['p95']:.0f}  GT={stats[fid]['gt_px']:,}px")

    e3 = rh(ce(mv3, gb), RH_); e5 = rh(ce(mv5, gb), RH_)
    r1 = np.hstack([e3, e5])
    pl(r1, f"V3 F1={r3['f1']:.4f} P={r3['precision']:.4f} R={r3['recall']:.4f}")
    pl(r1[:, e3.shape[1]:], f"V5 F1={r5['f1']:.4f} P={r5['precision']:.4f} R={r5['recall']:.4f}")
    pl(e3, f"TP={r3['tp']:,} FP={r3['fp']:,}", 42); pl(e5, f"TP={r5['tp']:,} FP={r5['fp']:,}", 42)

    p3 = rh(cv2.cvtColor(mv3, cv2.COLOR_GRAY2BGR), RH_); p5 = rh(cv2.cvtColor(mv5, cv2.COLOR_GRAY2BGR), RH_)
    r2 = np.hstack([p3, p5])

    rows = [r0, r1, r2]; mw = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        pad_w = mw - r.shape[1]
        padded.append(np.hstack([r, np.zeros((r.shape[0], pad_w, 3), dtype=np.uint8)]) if pad_w > 0 else r)

    lg = np.zeros((26, mw, 3), dtype=np.uint8); lg[:] = [30, 30, 30]
    for j, (txt, col) in enumerate([("Green=TP", (0, 255, 0)), ("Red=FP", (0, 0, 255)), ("Blue=FN", (255, 0, 0))]):
        cv2.putText(lg, txt, (10 + j * mw // 3, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 2)

    db = np.zeros((22, mw, 3), dtype=np.uint8); db[:] = [45, 45, 45]
    cv2.putText(db, f"Δ V5-V3: F1={r5['f1']-r3['f1']:+.4f}  TP={r5['tp']-r3['tp']:+,}  FP={r5['fp']-r3['fp']:+,}",
                (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    cv2.imwrite(str(PF / f"{fid}_v5compare.png"), np.vstack([padded[0], padded[1], padded[2], lg, db]))
    print(f"  [{idx+1:>2}/{SAMPLE}] {fid}")
print(f"\n  Images: {PF}/")

# Chart
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 2, figsize=(18, 10))
fig.suptitle(f'V5 Local Peak Detection vs V3  |  V3 F1={np.mean(v3f):.4f}  V5 F1={np.mean(v5f):.4f}  Δ={np.mean(v5f)-np.mean(v3f):+.4f}',
             fontsize=13, fontweight='bold')
x = np.arange(SAMPLE); w = 0.35
axes[0,0].bar(x-w/2, v3f, w, label='V3', color='#2ecc71', alpha=0.85)
axes[0,0].bar(x+w/2, v5f, w, label='V5', color='#9b59b6', alpha=0.85)
axes[0,0].set_title('F1 per Frame'); axes[0,0].legend(fontsize=8)
axes[0,0].set_xticks(x[::5]); axes[0,0].set_xticklabels([sampled[i] for i in range(0, SAMPLE, 5)], rotation=45, ha='right', fontsize=6)
axes[0,0].grid(axis='y', alpha=0.3)

dv = [v5f[i]-v3f[i] for i in range(SAMPLE)]
axes[0,1].bar(x, dv, 0.6, color=['#27ae60' if d>=0 else '#e74c3c' for d in dv], alpha=0.85)
axes[0,1].axhline(y=0, color='black', linewidth=0.8)
axes[0,1].set_title('Δ F1 (V5 - V3)'); axes[0,1].set_xticks(x[::5])
axes[0,1].set_xticklabels([sampled[i] for i in range(0, SAMPLE, 5)], rotation=45, ha='right', fontsize=6)
axes[0,1].grid(axis='y', alpha=0.3)

for mn, c, mk in [("V3", '#2ecc71', 'o'), ("V5", '#9b59b6', 's')]:
    pp = [all_results[mn][f]["precision"] for f in sampled]
    rr = [all_results[mn][f]["recall"] for f in sampled]
    axes[1,0].scatter(pp, rr, c=c, label=mn, alpha=0.6, s=25, marker=mk)
axes[1,0].set_xlabel('Precision'); axes[1,0].set_ylabel('Recall'); axes[1,0].set_title('Precision-Recall')
axes[1,0].legend(fontsize=8); axes[1,0].grid(alpha=0.3)

p95s = [stats[f]["p95"] for f in sampled]
axes[1,1].scatter(p95s, v3f, c='#2ecc71', label='V3', alpha=0.4, s=20)
axes[1,1].scatter(p95s, v5f, c='#9b59b6', label='V5', alpha=0.5, s=20)
axes[1,1].set_xlabel('P95 Diff'); axes[1,1].set_ylabel('F1'); axes[1,1].set_title('F1 vs Difficulty')
axes[1,1].legend(fontsize=8); axes[1,1].grid(alpha=0.3)
plt.tight_layout()
fig.savefig(str(OUT / "v5_compare_chart.png"), dpi=150, bbox_inches='tight'); plt.close(fig)
print(f"  Chart: {OUT / 'v5_compare_chart.png'}")

# Save text
with open(OUT / "v5_compare_results.txt", "w", encoding="utf-8") as f:
    f.write(f"{'Frame':<8} {'P95':>6} {'GT_px':>8}  {'V3_F1':>8} {'V5_F1':>8} {'Δ':>9} {'Best':>6}  {'V3_FP':>10} {'V5_FP':>10}\n")
    for fid in sampled:
        s = stats[fid]; r3=all_results["V3"][fid]; r5=all_results["V5"][fid]
        d = r5["f1"]-r3["f1"]
        f.write(f"{fid:<8} {s['p95']:>6.0f} {s['gt_px']:>8,}  {r3['f1']:>8.4f} {r5['f1']:>8.4f} {d:>+9.4f} {'V5' if d>=0 else 'V3':>6}  {r3['fp']:>10,} {r5['fp']:>10,}\n")
print(f"\n{'='*80}  DONE!")
