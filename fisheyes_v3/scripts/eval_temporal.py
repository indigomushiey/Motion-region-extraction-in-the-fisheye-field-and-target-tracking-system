"""Temporal consistency test: HF vs HF-Temporal on 20 frames + 1 sequence."""
import cv2, numpy as np
from pathlib import Path
import sys, importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V2_ROOT = Path(__file__).resolve().parents[2] / "fisheyes_v2" / "fisheye_motion_tracking"
sys.path.insert(0, str(PROJECT_ROOT))
from core.motion_hybrid import detect_motion_hybrid, temporal_smooth

def _imp(rp, nm):
    fp = V2_ROOT / rp
    s = importlib.util.spec_from_file_location(nm, fp); m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m); return m
_m = _imp("core/evaluation/metrics.py", "mt")
_f = _imp("core/motion_detection/frame_difference.py", "fd")
cm = _m.compute_metrics; dm12 = _f.detect_motion_seed_expand_v12

DATA = PROJECT_ROOT / "data" / "homework2"
GTD = DATA / "motion_annotation" / "GroudTruth"
CUR = DATA / "rgb_images"; PRV = DATA / "previous_images"
OUT = PROJECT_ROOT / "output" / "evaluation" / "temporal"
OUT.mkdir(parents=True, exist_ok=True)

# Test 1: 20-frame benchmark (same as before, temporal on consecutive groups)
F20 = ['00000','00001','00002','00003','00004','00005','00006','00007','00013','00016',
       '00039','00041','00053','00022','00026','00027','00033','00037','00059','00065']

# Test 2: Full continuous sequence (00000-00012, 13 consecutive frames)
F_SEQ = [f'{i:05d}' for i in range(13)]  # 00000-00012

print("=" * 100)
print("  TEMPORAL CONSISTENCY: HF vs HF-Temporal")
print("=" * 100)

for test_name, ALL_FRAMES in [("20-frame set", F20), ("Sequence 00000-00012", F_SEQ)]:
    print(f"\n{'='*100}")
    print(f"  {test_name} ({len(ALL_FRAMES)} frames)")
    print(f"{'='*100}")

    methods = {
        "V12":  lambda pg, cg: dm12(pg, cg),
        "HF":   lambda pg, cg: detect_motion_hybrid(pg, cg),
        "HF-T": lambda pg, cg: None,  # handled specially below
    }
    all_r, fs = {n: {} for n in methods}, {}

    # Track temporal state
    prev_mask_hf = None
    prev_fid_hf = None

    for idx, fid in enumerate(ALL_FRAMES):
        c = cv2.imread(str(CUR / f"{fid}_FV.png"))
        p = cv2.imread(str(PRV / f"{fid}_FV_prev.png"))
        g = cv2.imread(str(GTD / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
        pg, cg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        gb = (g > 0)
        fs[fid] = {"p95": np.percentile(cv2.absdiff(cg, pg), 95), "gt": int(gb.sum())}

        # V12
        m_v12 = dm12(pg, cg)
        all_r["V12"][fid] = cm(m_v12, gb.astype(np.uint8)*255)

        # HF (raw, no temporal)
        m_hf = detect_motion_hybrid(pg, cg)
        all_r["HF"][fid] = cm(m_hf, gb.astype(np.uint8)*255)

        # HF-T (with temporal smoothing on consecutive frames)
        is_consecutive = (prev_fid_hf is not None and
                          int(fid) == int(prev_fid_hf) + 1)
        smooth_float, m_hft = temporal_smooth(
            m_hf, prev_mask_hf if is_consecutive else None, alpha=0.7)
        all_r["HF-T"][fid] = cm(m_hft, gb.astype(np.uint8)*255)

        prev_mask_hf = smooth_float
        prev_fid_hf = fid

        print(f"  [{idx+1}/{len(ALL_FRAMES)}] {fid} P95={fs[fid]['p95']:.0f}"
              f"  V12={all_r['V12'][fid]['f1']:.3f}  HF={all_r['HF'][fid]['f1']:.3f}"
              f"  HF-T={all_r['HF-T'][fid]['f1']:.3f}"
              + (" [temp]" if is_consecutive else ""))

    # Summary
    print(f"\n  {'Method':<8} {'F1':>10} {'Precision':>10} {'Recall':>10}  "
          f"{'TP_sum':>10} {'FP_sum':>10}")
    print(f"  {'-'*55}")
    for mn in ["V12","HF","HF-T"]:
        ml = [all_r[mn][f] for f in ALL_FRAMES]
        av = {k: np.mean([m[k] for m in ml]) for k in ["f1","precision","recall"]}
        tp, fp = (int(sum(m[k] for m in ml)) for k in ["tp","fp"])
        mk = " <<<" if mn == "HF-T" else ""
        print(f"  {mn:<8} {av['f1']:>10.4f} {av['precision']:>10.4f} "
              f"{av['recall']:>10.4f}  {tp:>10,} {fp:>10,}{mk}")

    hft = [all_r["HF-T"][f]["f1"] for f in ALL_FRAMES]
    hf = [all_r["HF"][f]["f1"] for f in ALL_FRAMES]
    v12 = [all_r["V12"][f]["f1"] for f in ALL_FRAMES]
    print(f"\n  Mean F1: V12={np.mean(v12):.4f}  HF={np.mean(hf):.4f}  HF-T={np.mean(hft):.4f}")
    print(f"  Δ HF-T vs HF: {np.mean(hft)-np.mean(hf):+.4f}")

    hft_w = sum(1 for i,f in enumerate(ALL_FRAMES) if hft[i]>=max(v12[i],hf[i]))
    hf_w = sum(1 for i,f in enumerate(ALL_FRAMES) if hf[i]>max(v12[i],hft[i]))
    print(f"  Wins: V12={sum(1 for i,f in enumerate(ALL_FRAMES) if v12[i]>max(hf[i],hft[i]))}"
          f"  HF={hf_w}  HF-T={hft_w}")

print("\n" + "=" * 100)
print("  DONE!")
print("=" * 100)
