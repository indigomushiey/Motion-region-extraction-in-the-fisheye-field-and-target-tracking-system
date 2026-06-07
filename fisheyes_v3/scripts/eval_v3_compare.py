"""50-frame V3 comparison: V12 vs HFv2 vs HFv3, with per-frame visualization."""
import cv2, numpy as np, sys, importlib.util, time, gc
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(__file__).resolve().parents[2] / "fisheyes_v2" / "fisheye_motion_tracking"
sys.path.insert(0, str(PROJECT_ROOT))
from core.motion_hybrid import detect_motion_hybrid, detect_motion_hybrid_v2, detect_motion_hybrid_v3

def _i(rp, nm):
    fp = V2_ROOT / rp
    s = importlib.util.spec_from_file_location(nm, fp)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

cm = _i("core/evaluation/metrics.py", "mt").compute_metrics
dm12 = _i("core/motion_detection/frame_difference.py", "fd").detect_motion_seed_expand_v12

DATA = PROJECT_ROOT / "data" / "homework2"
GTD = DATA / "motion_annotation" / "GroudTruth"
CUR = DATA / "rgb_images"; PRV = DATA / "previous_images"
OUT = PROJECT_ROOT / "output" / "evaluation" / "v3_compare"
OUT.mkdir(parents=True, exist_ok=True)
PF = OUT / "per_frame"; PF.mkdir(parents=True, exist_ok=True)

# ── Sample 50 frames evenly ──
gt_files = sorted(GTD.glob("*_FV.png"))
all_frames = [f.stem.replace("_FV", "") for f in gt_files
              if (CUR / f"{f.stem.replace('_FV', '')}_FV.png").exists()
              and (PRV / f"{f.stem.replace('_FV', '')}_FV_prev.png").exists()]
N = len(all_frames); SAMPLE = 50
step = N / SAMPLE
sampled = [all_frames[int(i * step)] for i in range(SAMPLE)]
if all_frames[-1] not in sampled:
    sampled[-1] = all_frames[-1]

print(f"{'='*90}")
print(f"  V3 COMPARISON: V12 vs HFv2 vs HFv3  ({SAMPLE} frames)")
print(f"  Improvements: structure separation + gradient-flow alignment")
print(f"{'='*90}")

methods = {
    "V12": dm12,
    "HFv2": lambda pg, cg: detect_motion_hybrid_v2(pg, cg),
    "V3":  lambda pg, cg: detect_motion_hybrid_v3(pg, cg),
}

all_results = {n: {} for n in methods}
frame_stats = {}
times = {n: [] for n in methods}

print(f"\n{'Frame':<8} {'P95':>6} {'GT_px':>8}  {'V12 F1':>8} {'HFv2 F1':>8} {'V3 F1':>8} {'ΔV3-12':>9} {'ΔV3-2':>9} {'Best':>6}")
print("-" * 90)

for idx, fid in enumerate(sampled):
    c = cv2.imread(str(CUR / f"{fid}_FV.png"))
    p = cv2.imread(str(PRV / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GTD / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gt = (g > 0).astype(np.uint8) * 255

    p95 = np.percentile(cv2.absdiff(cg, pg), 95)
    gt_px = int(gt.sum() / 255)
    frame_stats[fid] = {"p95": p95, "gt_px": gt_px}

    for mn, fn in methods.items():
        t0 = time.perf_counter()
        mask = fn(pg, cg)
        times[mn].append(time.perf_counter() - t0)
        all_results[mn][fid] = cm(mask, gt)

    v12f = all_results["V12"][fid]["f1"]
    v2f  = all_results["HFv2"][fid]["f1"]
    v3f  = all_results["V3"][fid]["f1"]
    d32 = v3f - v2f
    d31 = v3f - v12f
    best = max([("V12", v12f), ("HFv2", v2f), ("V3", v3f)], key=lambda x: x[1])[0]
    print(f"{fid:<8} {p95:>6.0f} {gt_px:>8,}  {v12f:>8.4f} {v2f:>8.4f} {v3f:>8.4f} {d31:>+9.4f} {d32:>+9.4f} {best:>6}")

    if (idx + 1) % 10 == 0:
        v12_m = np.mean([all_results["V12"][f]["f1"] for f in sampled[:idx+1]])
        v2_m  = np.mean([all_results["HFv2"][f]["f1"] for f in sampled[:idx+1]])
        v3_m  = np.mean([all_results["V3"][f]["f1"] for f in sampled[:idx+1]])
        print(f"  --- [{idx+1}/{SAMPLE}] running avg: V12={v12_m:.4f}  HFv2={v2_m:.4f}  V3={v3_m:.4f} ---")
    gc.collect()  # prevent memory leak from V12 CCA loops

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  SUMMARY")
print(f"{'='*90}")
print(f"{'Method':<8} {'F1':>10} {'Precision':>10} {'Recall':>10} {'TP':>12} {'FP':>12} {'Time/f':>8}")
print("-" * 80)

v12_f1s = [all_results["V12"][f]["f1"] for f in sampled]
v2_f1s  = [all_results["HFv2"][f]["f1"] for f in sampled]
v3_f1s  = [all_results["V3"][f]["f1"] for f in sampled]

for mn in ["V12", "HFv2", "V3"]:
    ml = [all_results[mn][f] for f in sampled]
    tp_s = sum(r["tp"] for r in ml)
    fp_s = sum(r["fp"] for r in ml)
    fn_s = sum(r["fn"] for r in ml)
    avg_f1 = np.mean([r["f1"] for r in ml])
    avg_p  = tp_s / (tp_s + fp_s + 1)
    avg_r  = np.mean([r["recall"] for r in ml])
    avg_t  = np.mean(times[mn])
    print(f"{mn:<8} {avg_f1:>10.4f} {avg_p:>10.4f} {avg_r:>10.4f} {tp_s:>12,} {fp_s:>12,} {avg_t:>7.2f}s")

# Deltas
print(f"\n  Δ V3-V12:  F1={np.mean(v3_f1s)-np.mean(v12_f1s):+.4f}")
print(f"  Δ V3-HFv2: F1={np.mean(v3_f1s)-np.mean(v2_f1s):+.4f}")

v12_win = sum(1 for i in range(SAMPLE) if v12_f1s[i] >= max(v2_f1s[i], v3_f1s[i]))
v2_win  = sum(1 for i in range(SAMPLE) if v2_f1s[i] > max(v12_f1s[i], v3_f1s[i]))
v3_win  = sum(1 for i in range(SAMPLE) if v3_f1s[i] > max(v12_f1s[i], v2_f1s[i]))
print(f"  Frame wins: V12={v12_win}  HFv2={v2_win}  V3={v3_win}")

# Top/bottom deltas
deltas_v3v2 = [(fid, v3_f1s[i] - v2_f1s[i]) for i, fid in enumerate(sampled)]
deltas_v3v2.sort(key=lambda x: x[1])
print(f"\n  Top-5 V3 wins:  {', '.join(f'{f}({d:+.4f})' for f,d in deltas_v3v2[-5:][::-1])}")
print(f"  Top-5 HFv2 wins: {', '.join(f'{f}({d:+.4f})' for f,d in deltas_v3v2[:5])}")

# ═══════════════════════════════════════════════════════════════════
# Save text results
# ═══════════════════════════════════════════════════════════════════
with open(OUT / "v3_compare_results.txt", "w", encoding="utf-8") as f:
    f.write(f"{'Frame':<8} {'P95':>6} {'GT_px':>8}  {'V12_F1':>8} {'HFv2_F1':>8} {'V3_F1':>8} {'ΔV3-V12':>9} {'ΔV3-HFv2':>9} {'Best':>6}\n")
    f.write("-" * 90 + "\n")
    for fid in sampled:
        s = frame_stats[fid]
        v12f = all_results["V12"][fid]["f1"]
        v2f  = all_results["HFv2"][fid]["f1"]
        v3f  = all_results["V3"][fid]["f1"]
        best = max([("V12", v12f), ("HFv2", v2f), ("V3", v3f)], key=lambda x: x[1])[0]
        f.write(f"{fid:<8} {s['p95']:>6.0f} {s['gt_px']:>8,}  {v12f:>8.4f} {v2f:>8.4f} {v3f:>8.4f} {v3f-v12f:>+9.4f} {v3f-v2f:>+9.4f} {best:>6}\n")
    f.write(f"\nV12:  F1={np.mean(v12_f1s):.4f} TP={sum(r['tp'] for r in [all_results['V12'][f] for f in sampled])} FP={sum(r['fp'] for r in [all_results['V12'][f] for f in sampled])}\n")
    f.write(f"HFv2: F1={np.mean(v2_f1s):.4f} TP={sum(r['tp'] for r in [all_results['HFv2'][f] for f in sampled])} FP={sum(r['fp'] for r in [all_results['HFv2'][f] for f in sampled])}\n")
    f.write(f"V3:   F1={np.mean(v3_f1s):.4f} TP={sum(r['tp'] for r in [all_results['V3'][f] for f in sampled])} FP={sum(r['fp'] for r in [all_results['V3'][f] for f in sampled])}\n")
print(f"Results: {OUT / 'v3_compare_results.txt'}")

# ═══════════════════════════════════════════════════════════════════
# Visualization helpers
# ═══════════════════════════════════════════════════════════════════
def color_error(pred, gt):
    pb, gb = (pred > 0), (gt > 0)
    out = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    out[pb & gb] = [0, 255, 0]; out[pb & ~gb] = [0, 0, 255]; out[~pb & gb] = [255, 0, 0]
    return out

def rh(img, th):
    r = th / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * r), th))

def pl(img, txt, y=22, c=(255, 255, 255)):
    cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2)
    cv2.putText(img, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

# ═══════════════════════════════════════════════════════════════════
# Generate comparison images
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  GENERATING VISUALIZATIONS...")
print(f"{'='*90}")

RH_ = 200
for idx, fid in enumerate(sampled):
    c = cv2.imread(str(CUR / f"{fid}_FV.png"))
    p = cv2.imread(str(PRV / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GTD / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gb = (g > 0)

    # Recompute masks
    mv12 = dm12(pg, cg)
    mv2  = detect_motion_hybrid_v2(pg, cg)
    mv3  = detect_motion_hybrid_v3(pg, cg)

    s = frame_stats[fid]
    r12 = all_results["V12"][fid]
    rv2 = all_results["HFv2"][fid]
    rv3 = all_results["V3"][fid]

    # ── Row 0: Input ──
    curr_s = rh(c, RH_)
    gt_ov = curr_s.copy()
    gm = rh((gb.astype(np.uint8) * 255), RH_)
    gt_ov[gm > 0] = (gt_ov[gm > 0] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
    diff_map = cv2.absdiff(cg, pg)
    diff_h = rh(cv2.applyColorMap(diff_map, cv2.COLORMAP_HOT), RH_)
    r0 = np.hstack([curr_s, gt_ov, diff_h])
    pl(r0, f"{fid}  P95={s['p95']:.0f}  GT={s['gt_px']:,}px")
    pl(r0[:, curr_s.shape[1]:], "GT Overlay")
    pl(r0[:, curr_s.shape[1]*2:], "Frame Diff")

    # ── Row 1: Error maps ──
    e12 = rh(color_error(mv12, gb), RH_)
    e2  = rh(color_error(mv2, gb), RH_)
    e3  = rh(color_error(mv3, gb), RH_)
    r1 = np.hstack([e12, e2, e3])
    pl(r1, f"V12  F1={r12['f1']:.4f}  P={r12['precision']:.4f}  R={r12['recall']:.4f}")
    pl(r1[:, e12.shape[1]:], f"HFv2 F1={rv2['f1']:.4f}  P={rv2['precision']:.4f}  R={rv2['recall']:.4f}")
    pl(r1[:, e12.shape[1]*2:], f"V3   F1={rv3['f1']:.4f}  P={rv3['precision']:.4f}  R={rv3['recall']:.4f}")
    pl(e12, f"TP={r12['tp']:,} FP={r12['fp']:,}", 42)
    pl(e2,  f"TP={rv2['tp']:,} FP={rv2['fp']:,}", 42)
    pl(e3,  f"TP={rv3['tp']:,} FP={rv3['fp']:,}", 42)

    # ── Row 2: Masks ──
    p12 = rh(cv2.cvtColor(mv12, cv2.COLOR_GRAY2BGR), RH_)
    p2  = rh(cv2.cvtColor(mv2, cv2.COLOR_GRAY2BGR), RH_)
    p3  = rh(cv2.cvtColor(mv3, cv2.COLOR_GRAY2BGR), RH_)
    r2 = np.hstack([p12, p2, p3])
    pl(r2, "V12 Pred", 20)
    pl(r2[:, p12.shape[1]:], "HFv2 Pred", 20)
    pl(r2[:, p12.shape[1]*2:], "V3 Pred", 20)

    # ── Assemble ──
    rows = [r0, r1, r2]
    mw = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        if r.shape[1] < mw:
            pad = np.zeros((r.shape[0], mw - r.shape[1], 3), dtype=np.uint8)
            padded.append(np.hstack([r, pad]))
        else:
            padded.append(r)

    # Legend
    lg = np.zeros((26, mw, 3), dtype=np.uint8); lg[:] = [30, 30, 30]
    legends = [("Green=TP", (0, 255, 0)), ("Red=FP", (0, 0, 255)), ("Blue=FN", (255, 0, 0))]
    for j, (txt, col) in enumerate(legends):
        cv2.putText(lg, txt, (10 + j * mw // 3, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 2)

    # Delta bar
    db = np.zeros((22, mw, 3), dtype=np.uint8); db[:] = [45, 45, 45]
    dt = (f"Δ V3-V12: F1={rv3['f1']-r12['f1']:+.4f} TP={rv3['tp']-r12['tp']:+,} FP={rv3['fp']-r12['fp']:+,}  |  "
          f"Δ V3-HFv2: F1={rv3['f1']-rv2['f1']:+.4f} TP={rv3['tp']-rv2['tp']:+,} FP={rv3['fp']-rv2['fp']:+,}")
    cv2.putText(db, dt, (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    viz = np.vstack([padded[0], padded[1], padded[2], lg, db])
    op = PF / f"{fid}_v3compare.png"
    cv2.imwrite(str(op), viz)
    print(f"  [{idx+1:>2}/{SAMPLE}] {fid} → {op.name}")

# ═══════════════════════════════════════════════════════════════════
# Summary chart
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*90}")
print("  GENERATING SUMMARY CHART...")
print(f"{'='*90}")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(22, 12))
fig.suptitle(f'V3 Comparison: V12 vs HFv2 vs V3  |  '
             f'Mean F1 — V12={np.mean(v12_f1s):.4f}  HFv2={np.mean(v2_f1s):.4f}  V3={np.mean(v3_f1s):.4f}  '
             f'ΔV3-HFv2={np.mean(v3_f1s)-np.mean(v2_f1s):+.4f}',
             fontsize=13, fontweight='bold')

x = np.arange(SAMPLE)

# F1 per frame
ax = axes[0, 0]
w = 0.25
ax.bar(x - w, v12_f1s, w, label='V12', color='#3498db', alpha=0.85)
ax.bar(x, v2_f1s, w, label='HFv2', color='#e67e22', alpha=0.85)
ax.bar(x + w, v3_f1s, w, label='V3', color='#2ecc71', alpha=0.85)
ax.set_xlabel('Frame'); ax.set_ylabel('F1'); ax.set_title('F1 per Frame')
ax.legend(fontsize=8); ax.set_xticks(x[::5])
ax.set_xticklabels([sampled[i] for i in range(0, SAMPLE, 5)], rotation=45, ha='right', fontsize=6)
ax.grid(axis='y', alpha=0.3)

# Delta V3 vs HFv2
ax = axes[0, 1]
d_v3v2 = [v3_f1s[i] - v2_f1s[i] for i in range(SAMPLE)]
colors = ['#27ae60' if d >= 0 else '#e74c3c' for d in d_v3v2]
ax.bar(x, d_v3v2, 0.6, color=colors, alpha=0.85)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Frame'); ax.set_ylabel('Δ F1 (V3 - HFv2)'); ax.set_title('V3 Improvement over HFv2')
ax.set_xticks(x[::5])
ax.set_xticklabels([sampled[i] for i in range(0, SAMPLE, 5)], rotation=45, ha='right', fontsize=6)
ax.grid(axis='y', alpha=0.3)

# Delta V3 vs V12
ax = axes[0, 2]
d_v3v12 = [v3_f1s[i] - v12_f1s[i] for i in range(SAMPLE)]
colors = ['#27ae60' if d >= 0 else '#e74c3c' for d in d_v3v12]
ax.bar(x, d_v3v12, 0.6, color=colors, alpha=0.85)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Frame'); ax.set_ylabel('Δ F1 (V3 - V12)'); ax.set_title('V3 Improvement over V12')
ax.set_xticks(x[::5])
ax.set_xticklabels([sampled[i] for i in range(0, SAMPLE, 5)], rotation=45, ha='right', fontsize=6)
ax.grid(axis='y', alpha=0.3)

# Precision-Recall scatter
ax = axes[1, 0]
for mn, c, m in [("V12", '#3498db', 'o'), ("HFv2", '#e67e22', 's'), ("V3", '#2ecc71', '^')]:
    pp = [all_results[mn][f]["precision"] for f in sampled]
    rr = [all_results[mn][f]["recall"] for f in sampled]
    ax.scatter(pp, rr, c=c, label=mn, alpha=0.6, s=25, marker=m)
ax.set_xlabel('Precision'); ax.set_ylabel('Recall'); ax.set_title('Precision-Recall')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# F1 vs P95 (difficulty)
ax = axes[1, 1]
p95s = [frame_stats[f]["p95"] for f in sampled]
ax.scatter(p95s, v12_f1s, c='#3498db', label='V12', alpha=0.4, s=20)
ax.scatter(p95s, v2_f1s, c='#e67e22', label='HFv2', alpha=0.4, s=20)
ax.scatter(p95s, v3_f1s, c='#2ecc71', label='V3', alpha=0.5, s=20)
ax.set_xlabel('P95 Diff'); ax.set_ylabel('F1'); ax.set_title('F1 vs Difficulty')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# FP comparison bar
ax = axes[1, 2]
totals = {
    "V12": sum(all_results["V12"][f]["fp"] for f in sampled),
    "HFv2": sum(all_results["HFv2"][f]["fp"] for f in sampled),
    "V3": sum(all_results["V3"][f]["fp"] for f in sampled),
}
bars = ax.bar(totals.keys(), totals.values(), color=['#3498db', '#e67e22', '#2ecc71'], alpha=0.85)
ax.set_ylabel('Total FP pixels'); ax.set_title('Total False Positives')
for bar, val in zip(bars, totals.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + val*0.01,
            f'{val:,}', ha='center', fontsize=9, fontweight='bold')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
chart_path = OUT / "v3_compare_chart.png"
fig.savefig(str(chart_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Chart: {chart_path}")

print(f"\n{'='*90}")
print(f"  DONE!  50 frames × 3 methods  |  Output: {OUT}")
print(f"  Per-frame images: {PF}/")
print(f"{'='*90}")
