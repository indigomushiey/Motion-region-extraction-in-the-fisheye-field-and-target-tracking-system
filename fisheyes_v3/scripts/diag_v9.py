"""V9 diagnostic visualization: edge movement, seed filter, contrast norm."""
import cv2, numpy as np, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from core.motion_hybrid import detect_motion_hybrid_v7, detect_motion_hybrid_v9
from core.calib import RadialPoly

DATA = PROJECT_ROOT / "data" / "homework2"
CUR = DATA / "rgb_images"; PRV = DATA / "previous_images"
GTD = DATA / "motion_annotation" / "GroudTruth"
CALIB_DIR = DATA / "calibration_data"
OUT = PROJECT_ROOT / "output" / "evaluation" / "v9_diag"
OUT.mkdir(parents=True, exist_ok=True)

FIDS = ["00000", "00005", "00015", "00026", "00039", "00075", "00293"]

def rh(img, th):
    r = th / img.shape[0]; return cv2.resize(img, (int(img.shape[1] * r), th))

def pl(img, txt, y=20, c=(255,255,255)):
    cv2.putText(img, txt, (6,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 2)
    cv2.putText(img, txt, (5,y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)

def ce(pred, gt):
    pb, gb = (pred > 0), (gt > 0)
    out = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    out[pb & gb] = [0, 255, 0]; out[pb & ~gb] = [0, 0, 255]; out[~pb & gb] = [255, 0, 0]
    return out

for fid in FIDS:
    print(f"Processing {fid}...")
    c = cv2.imread(str(CUR / f"{fid}_FV.png"))
    p = cv2.imread(str(PRV / f"{fid}_FV_prev.png"))
    g = cv2.imread(str(GTD / f"{fid}_FV.png"), cv2.IMREAD_GRAYSCALE)
    pg = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    cg = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    gb = (g > 0)
    img_h, img_w = pg.shape

    # ── Compute V9 internals ──────────────────────────────────
    pg_f = cv2.GaussianBlur(pg, (5, 5), 0)
    cg_f = cv2.GaussianBlur(cg, (5, 5), 0)
    pg_struct = cv2.bilateralFilter(pg_f, 9, 50, 10)
    cg_struct = cv2.bilateralFilter(cg_f, 9, 50, 10)

    # Brightness & contrast norm
    local_brightness = cv2.GaussianBlur(cg_struct.astype(np.float32), (21, 21), 7)
    grad_x = cv2.Sobel(cg_struct, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(cg_struct, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2) + 1e-8
    grad_mag_norm = grad_mag / (local_brightness + 0.5)

    # Edge differencing
    gm_valid = grad_mag[grad_mag > 1.0]
    gm_median = float(np.median(gm_valid)) if len(gm_valid) > 100 else 20.0
    lo, hi = max(15, gm_median*0.4), max(40, gm_median*1.2)
    edges_prev = cv2.Canny(pg_struct, lo, hi)
    edges_curr = cv2.Canny(cg_struct, lo, hi)
    dist_prev = cv2.distanceTransform((edges_prev==0).astype(np.uint8), cv2.DIST_L2, 3)
    dist_curr = cv2.distanceTransform((edges_curr==0).astype(np.uint8), cv2.DIST_L2, 3)

    moved = np.zeros((img_h, img_w), dtype=np.float32)
    moved[(edges_curr>0) & (dist_prev>3)] = 1.0
    disappeared = np.zeros((img_h, img_w), dtype=np.float32)
    disappeared[(edges_prev>0) & (dist_curr>3)] = 1.0
    edge_move_map = cv2.GaussianBlur(moved + disappeared, (31, 31), 9)
    if edge_move_map.max() > 0.001: edge_move_map /= edge_move_map.max()

    static_edge = np.zeros((img_h, img_w), dtype=np.float32)
    static_edge[(edges_curr>0) & (dist_prev<2)] = 1.0
    static_density = cv2.GaussianBlur(static_edge, (31, 31), 9)
    if static_density.max() > 0.001: static_density /= static_density.max()

    # Flow for seed direction filter
    flow_f = cv2.calcOpticalFlowFarneback(pg_struct, cg_struct, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow_angle = np.arctan2(flow_f[..., 1], flow_f[..., 0])
    cos_a = np.cos(flow_angle); sin_a = np.sin(flow_angle)
    cos_local = cv2.blur(cos_a, (7, 7)); sin_local = cv2.blur(sin_a, (7, 7))
    R = np.sqrt(cos_local**2 + sin_local**2)  # direction consistency

    # Seeds (V9)
    diff = cv2.absdiff(cg_struct, pg_struct)
    p95 = np.percentile(diff, 95)
    if p95 > 100: st = 22
    elif p95 > 70: st = 20
    elif p95 > 50: st = 18
    elif p95 > 30: st = 16
    else: st = 14
    _, seed_raw = cv2.threshold(diff, st, 255, cv2.THRESH_BINARY)
    nk = np.ones((7,7), np.uint8)
    nc = cv2.filter2D((seed_raw>0).astype(np.float32), -1, nk)
    mn = 3 if p95 > 80 else 4
    seed = seed_raw.copy(); seed[nc < mn] = 0
    if seed.max() == 0: seed = seed_raw

    # Apply direction filter
    seed_before = seed.copy()
    seed_dir_ok = R > 0.55
    seed_dir_killed = (seed > 0) & ~seed_dir_ok
    seed[seed_dir_killed] = 0
    if seed.max() == 0: seed = seed_before
    seed_after = seed

    # Full masks
    mv7 = detect_motion_hybrid_v7(pg, cg)
    mv9 = detect_motion_hybrid_v9(pg, cg)

    # ── Build visualization ──
    RH = 180
    # Row 0: Input
    curr_s = rh(c, RH)
    gt_ov = curr_s.copy(); gm = rh((gb.astype(np.uint8)*255), RH)
    gt_ov[gm>0] = (gt_ov[gm>0]*0.4 + np.array([0,255,0])*0.6).astype(np.uint8)
    r0 = np.hstack([curr_s, gt_ov])

    # Row 1: Edge movement diagnostic
    edge_viz = rh(cv2.applyColorMap((edge_move_map*255).astype(np.uint8), cv2.COLORMAP_JET), RH)
    static_viz = rh(cv2.applyColorMap((static_density*255).astype(np.uint8), cv2.COLORMAP_HOT), RH)
    bright_viz = rh(cv2.applyColorMap((local_brightness/255*255).clip(0,255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), RH)
    contrast_viz = rh(cv2.applyColorMap((grad_mag_norm*255/0.2).clip(0,255).astype(np.uint8), cv2.COLORMAP_MAGMA), RH)
    r1 = np.hstack([edge_viz, static_viz, bright_viz, contrast_viz])
    pl(r1, "Edge Movement (moved=red)", 18)
    pl(r1[:, edge_viz.shape[1]:], "Static Edges (hot)", 18)
    pl(r1[:, edge_viz.shape[1]*2:], "Local Brightness", 18)
    pl(r1[:, edge_viz.shape[1]*3:], "Contrast Norm (grad/bright)", 18)

    # Row 2: Seed direction filter
    seed_before_viz = rh(cv2.cvtColor(seed_before, cv2.COLOR_GRAY2BGR), RH)
    seed_after_viz = rh(cv2.cvtColor(seed_after, cv2.COLOR_GRAY2BGR), RH)
    # Show killed seeds in red
    seed_before_color = rh(cv2.cvtColor(seed_before, cv2.COLOR_GRAY2BGR), RH)
    sk = rh(seed_dir_killed.astype(np.uint8)*255, RH)
    seed_before_color[sk > 0] = [0, 0, 255]  # red = killed by direction filter
    R_viz = rh(cv2.applyColorMap((R*255).clip(0,255).astype(np.uint8), cv2.COLORMAP_VIRIDIS), RH)
    r2 = np.hstack([seed_before_color, seed_after_viz, R_viz])
    pl(r2, f"Seeds BEFORE (red=killed by dir filter) n={seed_before.sum()//255}", 18)
    pl(r2[:, seed_before_viz.shape[1]:], f"Seeds AFTER n={seed_after.sum()//255}", 18)
    pl(r2[:, seed_before_viz.shape[1]*2:], "Flow Direction Consistency (R)", 18)

    # Row 3: V7 vs V9 error maps
    e7 = rh(ce(mv7, gb), RH); e9 = rh(ce(mv9, gb), RH)
    r3 = np.hstack([e7, e9])

    # Assemble
    rows = [r0, r1, r2, r3]
    mw = max(r.shape[1] for r in rows)
    padded = []
    for r in rows:
        pad = np.zeros((r.shape[0], mw-r.shape[1], 3), dtype=np.uint8) if r.shape[1] < mw else np.zeros((r.shape[0],0,3),dtype=np.uint8)
        padded.append(np.hstack([r, pad]) if pad.shape[1]>0 else r)

    # Labels on row 3
    pl(padded[3], "V7 Error (Green=TP Red=FP Blue=FN)", 18)
    pl(padded[3][:, padded[3].shape[1]//2:], "V9 Error", 18)

    # Legend
    lg = np.zeros((24, mw, 3), dtype=np.uint8); lg[:] = [30,30,30]
    texts = [
        ("Edge Movement: red=edges that changed position between frames", (255,180,100)),
        ("Static Edges: white=edges that stayed (buildings)", (200,200,200)),
        ("Brightness: yellow=bright, dark=dark regions", (150,150,150)),
        ("Contrast Norm: yellow=high normalized contrast (includes dark objects)", (200,100,200))]
    for j, (txt, col) in enumerate(texts[:4]):
        x = 10 + j * (mw//4) if mw >= 800 else 10
        cv2.putText(lg, txt, (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.3, col, 1)

    viz = np.vstack(padded + [lg])
    op = OUT / f"{fid}_v9diag.png"
    cv2.imwrite(str(op), viz)
    print(f"  → {op.name}")

print(f"\nDone! Output: {OUT}/")
for f in sorted(OUT.glob("*.png")):
    print(f"  {f.name}")
