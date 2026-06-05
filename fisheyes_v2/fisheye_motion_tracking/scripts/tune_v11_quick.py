"""Quick parameter sweep for V11 on the 13-frame benchmark."""
import sys
sys.path.insert(0, r'D:\Project\ipmv\fisheyes_v2\fisheye_motion_tracking')
import cv2, numpy as np
from pathlib import Path

DATA = Path(r'D:\Project\ipmv\fisheyes_v2\fisheye_motion_tracking\data\homework2')
BENCH = ['00000','00001','00002','00003','00004','00005','00006','00007',
         '00013','00016','00039','00041','00053']
from core.evaluation.metrics import compute_metrics

# Precompute everything
print('Precomputing...')
pre = {}
for fid in BENCH:
    c = cv2.imread(str(DATA / 'rgb_images' / f'{fid}_FV.png'))
    p = cv2.imread(str(DATA / 'previous_images' / f'{fid}_FV_prev.png'))
    g = cv2.imread(str(DATA / 'motion_annotation' / 'GroudTruth' / f'{fid}_FV.png'), cv2.IMREAD_GRAYSCALE)
    pg = cv2.GaussianBlur(cv2.cvtColor(p, cv2.COLOR_BGR2GRAY), (5,5), 0)
    cg = cv2.GaussianBlur(cv2.cvtColor(c, cv2.COLOR_BGR2GRAY), (5,5), 0)
    gt_bin = (g > 0)
    diff = cv2.absdiff(cg, pg)
    p95 = np.percentile(diff, 95)
    flow = cv2.calcOpticalFlowFarneback(pg, cg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag = np.linalg.norm(flow, axis=2)
    mag_mean = cv2.GaussianBlur(mag, (41,41), 15)
    mag_ratio = mag / (mag_mean + 0.5)
    h, w = pg.shape
    cx, cy = w/2, h/2
    yy, xx = np.ogrid[:h, :w]
    norm_dist = np.sqrt((xx-cx)**2 + (yy-cy)**2) / np.sqrt(cx**2+cy**2)
    pre[fid] = {'pg':pg,'cg':cg,'gt':gt_bin,'diff':diff,'p95':p95,
                'flow':flow,'mag':mag,'mag_ratio':mag_ratio,'nd':norm_dist}
    print(f'  {fid} blur_P95={p95:.1f}')
print(f'{len(pre)} frames ready\n')

def score(edge_bonus=0.35, min_area=50,
          seeds=(14,16,18,20,22), ratios=(2.2,1.8,1.6,1.5,1.3),
          expands=(4,6,8,10,12), use_abs=(False,False,False,True,True),
          flow_ts=(0,0,0,2.5,2.0), use_dir=(True,True,True,False,False)):
    f1s = []
    for fid, data in pre.items():
        p95 = data['p95']; diff = data['diff']; mag = data['mag']
        mag_ratio = data['mag_ratio']; flow = data['flow']; gt_bin = data['gt']
        nd = data['nd']

        if p95 > 120 or p95 < 20:
            st, ft, ed = (45, 2.0, 12) if p95 > 120 else (20, 3.5, 6)
            _, seed = cv2.threshold(diff, st, 255, cv2.THRESH_BINARY)
            _, cand = cv2.threshold(mag, ft, 255, cv2.THRESH_BINARY)
            cand = cand.astype(np.uint8)
            dt = cv2.distanceTransform((seed==0).astype(np.uint8), cv2.DIST_L2, 5)
            exp = cand.copy(); exp[dt > ed] = 0
            mask = seed | exp
            k = np.ones((7,7), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
            f1s.append(compute_metrics(mask, gt_bin.astype(np.uint8)*255)['f1'])
            continue

        if p95 > 100: i = 4
        elif p95 > 70: i = 3
        elif p95 > 50: i = 2
        elif p95 > 30: i = 1
        else: i = 0

        st, rtb, ed = seeds[i], ratios[i], expands[i]
        ua, uft, ud = use_abs[i], flow_ts[i], use_dir[i]

        rm = 1.0 + edge_bonus * (nd ** 2)
        rtm = rtb * rm
        cand = ((mag > uft) & (mag_ratio > rtm)).astype(np.uint8) * 255 if ua else (mag_ratio > rtm).astype(np.uint8) * 255

        _, seed_raw = cv2.threshold(diff, st, 255, cv2.THRESH_BINARY)
        nk = np.ones((7,7), np.uint8)
        nc = cv2.filter2D((seed_raw>0).astype(np.float32), -1, nk)
        mn = 3 if p95 > 80 else 4
        seed = seed_raw.copy(); seed[nc < mn] = 0
        if seed.max() == 0: seed = seed_raw

        dt = cv2.distanceTransform((seed==0).astype(np.uint8), cv2.DIST_L2, 5)
        exp = cand.copy(); exp[dt > ed] = 0

        if ud:
            sb = (seed > 0)
            if sb.sum() > 20:
                fa = np.arctan2(flow[...,1], flow[...,0])
                sam = np.zeros_like(fa); sam[sb] = fa[sb]
                sw = np.zeros_like(fa); sw[sb] = 1.0
                ks = ed*2+1
                sab = cv2.GaussianBlur(sam, (ks,ks), ed/2.0)
                swb = cv2.GaussianBlur(sw, (ks,ks), ed/2.0)
                vw = swb > 0.02; sab[vw] /= swb[vw]
                ad = np.abs(fa - sab); ad = np.minimum(ad, 2*np.pi - ad)
                exp[~(ad < np.pi/4) & vw] = 0

        mask = seed | exp
        nl, lbs, sts, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for j in range(1, nl):
            if sts[j, cv2.CC_STAT_AREA] < min_area:
                mask[lbs == j] = 0
        k = np.ones((7,7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        f1s.append(compute_metrics(mask, gt_bin.astype(np.uint8)*255)['f1'])
    return np.mean(f1s)

base_f1 = score()
print(f'Baseline V11 F1: {base_f1:.4f}\n')

# Stage 1: ratio_thresh per bin
print('=== STAGE 1: ratio_thresh ===')
best_r = {'f1': base_f1, 'ratios': (2.2,1.8,1.6,1.5,1.3)}
for r0 in [2.0, 2.2, 2.4, 2.6]:
 for r1 in [1.6, 1.8, 2.0, 2.2]:
  for r2 in [1.4, 1.6, 1.8, 2.0]:
   for r3 in [1.3, 1.5, 1.7, 1.9]:
    for r4 in [1.1, 1.3, 1.5]:
     f1 = score(ratios=(r0,r1,r2,r3,r4))
     if f1 > best_r['f1'] + 0.0001:
         best_r = {'f1':f1, 'ratios':(r0,r1,r2,r3,r4)}
         print(f'  ({r0},{r1},{r2},{r3},{r4}) F1={f1:.4f} +{f1-base_f1:.4f}')
print(f'Best: {best_r["ratios"]} F1={best_r["f1"]:.4f}\n')

# Stage 2: expand_dist
print('=== STAGE 2: expand_dist ===')
best_e = {'f1': best_r['f1'], 'expands': (4,6,8,10,12)}
for e0 in [3,4,5,6]:
 for e1 in [4,6,8,10]:
  for e2 in [6,8,10,12]:
   for e3 in [8,10,12,14]:
    for e4 in [10,12,14,16]:
     f1 = score(ratios=best_r['ratios'], expands=(e0,e1,e2,e3,e4))
     if f1 > best_e['f1'] + 0.0001:
         best_e = {'f1':f1, 'expands':(e0,e1,e2,e3,e4)}
         print(f'  ({e0},{e1},{e2},{e3},{e4}) F1={f1:.4f} +{f1-best_r["f1"]:.4f}')
print(f'Best: {best_e["expands"]} F1={best_e["f1"]:.4f}\n')

# Stage 3: edge_bonus
print('=== STAGE 3: edge_bonus ===')
best_eb = {'f1': best_e['f1'], 'eb': 0.35}
for eb in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6]:
    f1 = score(ratios=best_r['ratios'], expands=best_e['expands'], edge_bonus=eb)
    if f1 > best_eb['f1'] + 0.0001:
        best_eb = {'f1':f1, 'eb':eb}
        print(f'  edge_bonus={eb:.2f} F1={f1:.4f} +{f1-best_e["f1"]:.4f}')
print(f'Best: edge_bonus={best_eb["eb"]} F1={best_eb["f1"]:.4f}\n')

# Stage 4: seed_thresh
print('=== STAGE 4: seed_thresh ===')
best_s = {'f1': best_eb['f1'], 'seeds': (14,16,18,20,22)}
for s0 in [10,12,14,16,18]:
 for s1 in [12,14,16,18,20]:
  for s2 in [14,16,18,20,22]:
   for s3 in [16,18,20,22,24]:
    for s4 in [18,20,22,24,26]:
     f1 = score(ratios=best_r['ratios'], expands=best_e['expands'],
                edge_bonus=best_eb['eb'], seeds=(s0,s1,s2,s3,s4))
     if f1 > best_s['f1'] + 0.0001:
         best_s = {'f1':f1, 'seeds':(s0,s1,s2,s3,s4)}
         print(f'  ({s0},{s1},{s2},{s3},{s4}) F1={f1:.4f} +{f1-best_eb["f1"]:.4f}')
print(f'Best: {best_s["seeds"]} F1={best_s["f1"]:.4f}\n')

# Final
print('='*60)
print(f'FINAL BEST PARAMETERS')
print(f'Baseline:  {base_f1:.4f}')
print(f'Optimized: {best_s["f1"]:.4f} (+{best_s["f1"]-base_f1:.4f})')
print(f'ratios:    {best_r["ratios"]}')
print(f'expands:   {best_e["expands"]}')
print(f'edge_bonus:{best_eb["eb"]}')
print(f'seeds:     {best_s["seeds"]}')
