"""50-frame evaluation: V12 vs HF, with per-frame comparison visualization."""
import cv2, numpy as np, sys, importlib.util, textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(__file__).resolve().parents[2] / "fisheyes_v2" / "fisheye_motion_tracking"
sys.path.insert(0, str(PROJECT_ROOT))
from core.motion_hybrid import detect_motion_hybrid

def _i(rp, nm):
    fp = V2_ROOT / rp
    s = importlib.util.spec_from_file_location(nm, fp)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

cm = _i("core/evaluation/metrics.py", "mt").compute_metrics
dm12 = _i("core/motion_detection/frame_difference.py", "fd").detect_motion_seed_expand_v12

DATA = PROJECT_ROOT / "data" / "homework2"
GTD = DATA / "motion_annotation" / "GroudTruth"
CUR = DATA / "rgb_images"; PRV = DATA / "previous_images"
OUT = PROJECT_ROOT / "output" / "evaluation" / "eval50"
OUT.mkdir(parents=True, exist_ok=True)
PF = OUT / "per_frame"; PF.mkdir(parents=True, exist_ok=True)

# ── Find all valid frames, then sample 50 evenly ──
gt_files = sorted(GTD.glob("*_FV.png"))
all_frames = [f.stem.replace("_FV", "") for f in gt_files
              if (CUR / f"{f.stem.replace('_FV', '')}_FV.png").exists()
              and (PRV / f"{f.stem.replace('_FV', '')}_FV_prev.png").exists()]
N = len(all_frames)
SAMPLE = 50
# Evenly sample: take every N/SAMPLE-th frame
step = N / SAMPLE
sampled = [all_frames[int(i * step)] for i in range(SAMPLE)]
# Ensure last frame included if not already
if all_frames[-1] not in sampled:
    sampled[-1] = all_frames[-1]

print(f"50-Frame Evaluation: {SAMPLE} frames sampled from {N} total")
print(f"Frames: {sampled[0]} → {sampled[-1]}")
print(f"{'Frame':<8} {'P95':>6} {'GT_px':>8}  {'V12 F1':>8} {'HF F1':>8} {'Δ':>9} {'Best':>6}")
print("-" * 65)

results_v12, results_hf = [], []
stats = {}

for idx, fid in enumerate(sampled):
    c = cv2.imread(str(CUR / f"{fid}_FV.png"))
    p = cv2.imread(str(PRV / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GTD / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg, cg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gt = (g > 0).astype(np.uint8) * 255

    p95 = np.percentile(cv2.absdiff(cg, pg), 95)
    gt_px = int(gt.sum() / 255)

    m12 = dm12(pg, cg)
    mhf = detect_motion_hybrid(pg, cg)
    r12 = cm(m12, gt); rhf = cm(mhf, gt)
    results_v12.append(r12); results_hf.append(rhf)

    d = rhf["f1"] - r12["f1"]
    best = "HF" if d >= 0 else "V12"
    print(f"{fid:<8} {p95:>6.0f} {gt_px:>8,}  {r12['f1']:>8.4f} {rhf['f1']:>8.4f} {d:>+9.4f} {best:>6}")
    stats[fid] = {"p95": p95, "gt_px": gt_px, "v12": r12, "hf": rhf}

    if (idx + 1) % 10 == 0:
        v12_avg = np.mean([r["f1"] for r in results_v12])
        hf_avg = np.mean([r["f1"] for r in results_hf])
        print(f"  --- [{idx+1}/{SAMPLE}] running avg: V12={v12_avg:.4f} HF={hf_avg:.4f} Δ={hf_avg-v12_avg:+.4f} ---")

# ═══════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════
v12_f1s = [r["f1"] for r in results_v12]
hf_f1s = [r["f1"] for r in results_hf]
v12_tp = sum(r["tp"] for r in results_v12)
v12_fp = sum(r["fp"] for r in results_v12)
hf_tp = sum(r["tp"] for r in results_hf)
hf_fp = sum(r["fp"] for r in results_hf)

print("\n" + "=" * 70)
print("50-FRAME SUMMARY (V12 vs Hybrid Flow)")
print("=" * 70)
print(f"  V12: F1={np.mean(v12_f1s):.4f}  P={v12_tp/(v12_tp+v12_fp+1):.4f}  R={np.mean([r['recall'] for r in results_v12]):.4f}  TP={v12_tp:,}  FP={v12_fp:,}")
print(f"  HF:  F1={np.mean(hf_f1s):.4f}  P={hf_tp/(hf_tp+hf_fp+1):.4f}  R={np.mean([r['recall'] for r in results_hf]):.4f}  TP={hf_tp:,}  FP={hf_fp:,}")
print(f"  Δ HF-V12: F1={np.mean(hf_f1s)-np.mean(v12_f1s):+.4f}  TP={(hf_tp/(v12_tp+1)-1)*100:+.1f}%  FP={(hf_fp/(v12_fp+1)-1)*100:+.1f}%")

hf_wins = sum(1 for i in range(SAMPLE) if hf_f1s[i] >= v12_f1s[i])
print(f"  Frame wins: V12={SAMPLE-hf_wins}  HF={hf_wins}")

# Top/bottom frames
deltas = [(fid, hf_f1s[i] - v12_f1s[i]) for i, fid in enumerate(sampled)]
deltas.sort(key=lambda x: x[1])
print(f"\n  Top-5 HF wins:  {', '.join(f'{f}({d:+.4f})' for f,d in deltas[-5:][::-1])}")
print(f"  Top-5 V12 wins: {', '.join(f'{f}({d:+.4f})' for f,d in deltas[:5])}")

# ═══════════════════════════════════════════════════════════════════
# Save per-frame results to text file
# ═══════════════════════════════════════════════════════════════════
with open(OUT / "eval50_results.txt", "w", encoding="utf-8") as f:
    f.write(f"{'Frame':<8} {'P95':>6} {'GT_px':>8}  {'V12_F1':>8} {'HF_F1':>8} {'Delta':>9} {'Best':>6}\n")
    f.write("-" * 65 + "\n")
    for fid in sampled:
        s = stats[fid]
        d = s["hf"]["f1"] - s["v12"]["f1"]
        f.write(f"{fid:<8} {s['p95']:>6.0f} {s['gt_px']:>8,}  {s['v12']['f1']:>8.4f} {s['hf']['f1']:>8.4f} {d:>+9.4f} {'HF' if d>=0 else 'V12':>6}\n")
    f.write(f"\nV12: F1={np.mean(v12_f1s):.4f} TP={v12_tp} FP={v12_fp}\n")
    f.write(f"HF:  F1={np.mean(hf_f1s):.4f} TP={hf_tp} FP={hf_fp}\n")
    f.write(f"Δ:   F1={np.mean(hf_f1s)-np.mean(v12_f1s):+.4f}\n")
print(f"\nResults saved: {OUT / 'eval50_results.txt'}")

# ═══════════════════════════════════════════════════════════════════
# Visualization helpers
# ═══════════════════════════════════════════════════════════════════
def color_error(pred, gt):
    """TP=green, FP=red, FN=blue, TN=black"""
    pb, gb = (pred > 0), (gt > 0)
    out = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    out[pb & gb] = [0, 255, 0]      # TP: green
    out[pb & ~gb] = [0, 0, 255]     # FP: red
    out[~pb & gb] = [255, 0, 0]     # FN: blue
    return out

def color_pred(pred):
    """Predicted mask overlaid as cyan on black bg"""
    out = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    out[pred > 0] = [255, 255, 0]  # cyan/yellow
    return out

def rh(img, th):
    """Resize keeping aspect ratio to height=th"""
    r = th / img.shape[0]
    return cv2.resize(img, (int(img.shape[1] * r), th))

def pl(img, txt, y=22, c=(255, 255, 255)):
    """Put label with shadow"""
    cv2.putText(img, txt, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)
    cv2.putText(img, txt, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)

# ═══════════════════════════════════════════════════════════════════
# Generate comparison images
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  GENERATING COMPARISON VISUALIZATIONS...")
print(f"{'='*80}")

RH_ = 220  # row height for each strip

for idx, fid in enumerate(sampled):
    c = cv2.imread(str(CUR / f"{fid}_FV.png"))
    p = cv2.imread(str(PRV / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GTD / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gb = (g > 0)

    # Recompute masks if needed (they're already computed above)
    m12 = dm12(pg, cg)
    mhf = detect_motion_hybrid(pg, cg)

    s = stats[fid]
    diff_map = cv2.absdiff(cg, pg)

    # ── Row 0: Input images ──
    prev_s = rh(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), RH_)
    prev_s = cv2.cvtColor(prev_s, cv2.COLOR_GRAY2BGR)
    curr_s = rh(c, RH_)
    gt_s = rh(cv2.cvtColor((gb.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR), RH_)
    # GT overlay on current
    curr_gt = curr_s.copy()
    gm = rh((gb.astype(np.uint8) * 255), RH_)
    curr_gt[gm > 0] = (curr_gt[gm > 0] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)

    # diff heatmap
    diff_v = rh(diff_map, RH_)
    diff_h = cv2.applyColorMap(diff_v, cv2.COLORMAP_HOT)

    r0 = np.hstack([prev_s, curr_s, gt_s, curr_gt, diff_h])
    pl(r0, f"{fid}  |  Prev", 20)
    pl(r0[:, prev_s.shape[1]:], f"Curr", 20)
    pl(r0[:, prev_s.shape[1]*2:], f"GT Mask", 20)
    pl(r0[:, prev_s.shape[1]*3:], f"GT Overlay", 20)
    pl(r0[:, prev_s.shape[1]*4:], f"Diff (P95={s['p95']:.0f})", 20)

    # ── Row 1: Error maps ──
    ev12 = rh(color_error(m12, gb), RH_)
    ehf  = rh(color_error(mhf, gb), RH_)
    # Predicted masks
    pv12 = rh(color_pred(m12), RH_)
    phf  = rh(color_pred(mhf), RH_)

    r1 = np.hstack([ev12, ehf, pv12, phf])
    pl(r1, f"V12  F1={s['v12']['f1']:.4f}  P={s['v12']['precision']:.4f}  R={s['v12']['recall']:.4f}", 20)
    pl(r1[:, ev12.shape[1]:], f"HF   F1={s['hf']['f1']:.4f}  P={s['hf']['precision']:.4f}  R={s['hf']['recall']:.4f}", 20)
    pl(r1[:, ev12.shape[1]*2:], f"V12 Pred", 20)
    pl(r1[:, ev12.shape[1]*3:], f"HF Pred", 20)
    # Add TP/FP counts
    pl(ev12, f"TP={s['v12']['tp']:,}  FP={s['v12']['fp']:,}  FN={s['v12']['fn']:,}", 42)
    pl(ehf, f"TP={s['hf']['tp']:,}  FP={s['hf']['fp']:,}  FN={s['hf']['fn']:,}", 42)

    # ── Row 2: Magnitude maps (if available from HF) ──
    # Extract mag from hybrid computation (simplified: show diff)
    diff_norm = cv2.normalize(diff_map, None, 0, 255, cv2.NORM_MINMAX)
    diff_norm_s = rh(cv2.applyColorMap(diff_norm.astype(np.uint8), cv2.COLORMAP_JET), RH_)
    # V12 mask overlay on diff
    diff_v12 = diff_norm_s.copy()
    m12_s = rh(m12, RH_)
    diff_v12[m12_s > 0] = (diff_v12[m12_s > 0] * 0.5 + np.array([0, 255, 255]) * 0.5).astype(np.uint8)
    # HF mask overlay on diff
    diff_hf = diff_norm_s.copy()
    mhf_s = rh(mhf, RH_)
    diff_hf[mhf_s > 0] = (diff_hf[mhf_s > 0] * 0.5 + np.array([0, 255, 255]) * 0.5).astype(np.uint8)

    r2 = np.hstack([diff_norm_s, diff_v12, diff_hf])
    pl(r2, f"Frame Diff (normalized)", 20)
    pl(r2[:, diff_norm_s.shape[1]:], f"V12 Mask on Diff", 20)
    pl(r2[:, diff_norm_s.shape[1]*2:], f"HF Mask on Diff", 20)

    # ── Legend bar ──
    # Pad all rows to same width
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
    lg = np.zeros((28, mw, 3), dtype=np.uint8); lg[:] = [30, 30, 30]
    legends = [("Green=TP (correct detection)", (0, 255, 0)),
               ("Red=FP (false alarm)", (0, 0, 255)),
               ("Blue=FN (missed)", (255, 0, 0))]
    for j, (txt, col) in enumerate(legends):
        cv2.putText(lg, txt, (10 + j * mw // 3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 2)

    # Delta bar
    db = np.zeros((22, mw, 3), dtype=np.uint8); db[:] = [45, 45, 45]
    d_f1 = s['hf']['f1'] - s['v12']['f1']
    d_fp = s['hf']['fp'] - s['v12']['fp']
    d_tp = s['hf']['tp'] - s['v12']['tp']
    dt = f"Δ HF-V12:  F1={d_f1:+.4f}  TP={d_tp:+,}  FP={d_fp:+,}  |  GT={s['gt_px']:,}px  P95={s['p95']:.0f}"
    cv2.putText(db, dt, (10, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    viz = np.vstack([padded[0], padded[1], padded[2], lg, db])
    op = PF / f"{fid}_comparison.png"
    cv2.imwrite(str(op), viz)
    print(f"  [{idx+1:>2}/{SAMPLE}] {fid} → {op.name}")

print(f"\n  All comparison images saved to: {PF}")

# ═══════════════════════════════════════════════════════════════════
# Summary chart image
# ═══════════════════════════════════════════════════════════════════
print(f"\n{'='*80}")
print("  GENERATING SUMMARY CHART...")
print(f"{'='*80}")

# Bar chart of F1 per frame
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle(f'50-Frame Evaluation: V12 vs Hybrid Flow\nMean F1 — V12={np.mean(v12_f1s):.4f}  HF={np.mean(hf_f1s):.4f}  Δ={np.mean(hf_f1s)-np.mean(v12_f1s):+.4f}',
             fontsize=14, fontweight='bold')

# F1 per frame
ax = axes[0, 0]
x = np.arange(SAMPLE)
ax.bar(x - 0.2, v12_f1s, 0.35, label='V12', color='#3498db', alpha=0.85)
ax.bar(x + 0.2, hf_f1s, 0.35, label='HF', color='#e74c3c', alpha=0.85)
ax.set_xlabel('Frame index'); ax.set_ylabel('F1 Score'); ax.set_title('F1 Score per Frame')
ax.legend(); ax.set_xticks(x[::5]); ax.set_xticklabels([sampled[i] for i in range(0, SAMPLE, 5)], rotation=45, ha='right', fontsize=7)
ax.grid(axis='y', alpha=0.3)

# Delta F1
ax = axes[0, 1]
delta_f1s = [hf_f1s[i] - v12_f1s[i] for i in range(SAMPLE)]
colors = ['#27ae60' if d >= 0 else '#e74c3c' for d in delta_f1s]
ax.bar(x, delta_f1s, 0.6, color=colors, alpha=0.85)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xlabel('Frame index'); ax.set_ylabel('Δ F1 (HF - V12)'); ax.set_title('F1 Improvement per Frame')
ax.set_xticks(x[::5]); ax.set_xticklabels([sampled[i] for i in range(0, SAMPLE, 5)], rotation=45, ha='right', fontsize=7)
ax.grid(axis='y', alpha=0.3)

# Precision vs Recall scatter
ax = axes[1, 0]
v12_p = [r["precision"] for r in results_v12]; v12_r = [r["recall"] for r in results_v12]
hf_p = [r["precision"] for r in results_hf]; hf_r = [r["recall"] for r in results_hf]
ax.scatter(v12_p, v12_r, c='#3498db', label='V12', alpha=0.6, s=30)
ax.scatter(hf_p, hf_r, c='#e74c3c', label='HF', alpha=0.6, s=30)
ax.set_xlabel('Precision'); ax.set_ylabel('Recall'); ax.set_title('Precision-Recall per Frame')
ax.legend(); ax.grid(alpha=0.3)

# F1 vs P95 (contextual difficulty)
ax = axes[1, 1]
p95s = [stats[fid]["p95"] for fid in sampled]
ax.scatter(p95s, v12_f1s, c='#3498db', label='V12', alpha=0.5, s=25)
ax.scatter(p95s, hf_f1s, c='#e74c3c', label='HF', alpha=0.5, s=25)
ax.set_xlabel('P95 Diff'); ax.set_ylabel('F1 Score'); ax.set_title('F1 vs Frame Difficulty (P95)')
ax.legend(); ax.grid(alpha=0.3)

plt.tight_layout()
chart_path = OUT / "eval50_summary_chart.png"
fig.savefig(str(chart_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Summary chart saved: {chart_path}")

print(f"\n{'='*80}")
print("  DONE! All outputs in:", OUT)
print(f"{'='*80}")
