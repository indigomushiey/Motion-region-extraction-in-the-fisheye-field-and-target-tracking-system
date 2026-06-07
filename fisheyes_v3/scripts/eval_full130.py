"""Full 130-frame evaluation: V12 vs HF, per-frame + summary."""
import cv2, numpy as np, sys, importlib.util
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
OUT = PROJECT_ROOT / "output" / "evaluation" / "full130"
OUT.mkdir(parents=True, exist_ok=True)

# Find all frames with GT
gt_files = sorted(GTD.glob("*_FV.png"))
ALL = [f.stem.replace("_FV", "") for f in gt_files
       if (CUR / f"{f.stem.replace('_FV', '')}_FV.png").exists()
       and (PRV / f"{f.stem.replace('_FV', '')}_FV_prev.png").exists()]

print(f"Full evaluation: {len(ALL)} frames")
print(f"{'Frame':<8} {'P95':>6} {'GT_px':>8}  {'V12 F1':>8} {'HF F1':>8} {'Δ':>9} {'Best':>6}")
print("-" * 65)

results_v12, results_hf = [], []
stats = {}

for idx, fid in enumerate(ALL):
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

    if (idx + 1) % 30 == 0:
        v12_avg = np.mean([r["f1"] for r in results_v12])
        hf_avg = np.mean([r["f1"] for r in results_hf])
        print(f"  --- [{idx+1}/{len(ALL)}] running avg: V12={v12_avg:.4f} HF={hf_avg:.4f} Δ={hf_avg-v12_avg:+.4f} ---")

# Summary
v12_f1s = [r["f1"] for r in results_v12]
hf_f1s = [r["f1"] for r in results_hf]
v12_tp = sum(r["tp"] for r in results_v12)
v12_fp = sum(r["fp"] for r in results_v12)
hf_tp = sum(r["tp"] for r in results_hf)
hf_fp = sum(r["fp"] for r in results_hf)

print("\n" + "=" * 70)
print("FULL 130-FRAME SUMMARY")
print("=" * 70)
print(f"  V12: F1={np.mean(v12_f1s):.4f}  P={v12_tp/(v12_tp+v12_fp):.4f}  R={np.mean([r['recall'] for r in results_v12]):.4f}  TP={v12_tp:,}  FP={v12_fp:,}")
print(f"  HF:  F1={np.mean(hf_f1s):.4f}  P={hf_tp/(hf_tp+hf_fp):.4f}  R={np.mean([r['recall'] for r in results_hf]):.4f}  TP={hf_tp:,}  FP={hf_fp:,}")
print(f"  Δ HF-V12: F1={np.mean(hf_f1s)-np.mean(v12_f1s):+.4f}  TP={(hf_tp/v12_tp-1)*100:+.1f}%  FP={(hf_fp/v12_fp-1)*100:+.1f}%")

hf_wins = sum(1 for i in range(len(ALL)) if hf_f1s[i] >= v12_f1s[i])
print(f"  Frame wins: V12={len(ALL)-hf_wins}  HF={hf_wins}")

# Save to file
with open(OUT / "full130_results.txt", "w") as f:
    f.write(f"Frame  P95  GT_px  V12_F1  HF_F1  Delta  Best\n")
    for fid in ALL:
        s = stats[fid]
        d = s["hf"]["f1"] - s["v12"]["f1"]
        f.write(f"{fid}  {s['p95']:.0f}  {s['gt_px']}  {s['v12']['f1']:.4f}  {s['hf']['f1']:.4f}  {d:+.4f}  {'HF' if d>=0 else 'V12'}\n")
    f.write(f"\nV12: F1={np.mean(v12_f1s):.4f} TP={v12_tp} FP={v12_fp}\n")
    f.write(f"HF:  F1={np.mean(hf_f1s):.4f} TP={hf_tp} FP={hf_fp}\n")
    f.write(f"Δ:   F1={np.mean(hf_f1s)-np.mean(v12_f1s):+.4f}\n")

print(f"\nResults saved: {OUT / 'full130_results.txt'}")
print("DONE!")
