"""Study C: identification of common-cause vs item dispersion, and system quantities.
Usage: python3 studyC.py <worker_id> <n_workers> <reps>
"""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, time, json, itertools
sys.path.insert(0, f'{SP_ROOT}/scripts')
import numpy as np
from scipy.optimize import brentq
import ccffit2 as cf, ccfmodel as cm

OUT = f'{SP_ROOT}/work/studyC'
os.makedirs(OUT, exist_ok=True)
wid, nw, reps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

AP = 2.0; G = 3; SIG_C = 0.3
cells = list(itertools.product([2, 5, 10, 30], [60, 120], [0.05, 0.2], [0.5, 2.0, np.inf]))
CHI90 = 2.705543

def sysq(mu, ap, ac):
    R1 = cm.series_group(mu[:2], ap, ac)          # 2-input product, one month
    R3 = cm.series_group(mu[:3] if len(mu) >= 3 else mu, ap, ac) ** 3   # 3-input product, h = 3 months
    D2 = cm.allfail_group(mu[:2], ap, ac)          # dual-sourced input: both sources lost in a month
    return R1, R3, D2

def run_rep(cell, rep):
    I, T, mub, ac = cell
    seed = abs(hash((I, T, mub, float(ac), rep))) % (2**32)
    rng = np.random.default_rng(seed)
    mus = []
    for g in range(G):
        z = rng.normal(0, SIG_C, I); z -= z.mean()
        mus.append(mub * np.exp(z))
    data = cm.simulate(mus, AP, ac, T, rng)
    r = cf.fit('S1', data)
    ri = cf.fit('IND', data)
    idx = r.idx; jt = idx['th'][0]
    lt_hat = r.x[jt]; nll_min = r.fun
    # profile likelihood for log theta_c
    def prof(lt):
        rr = cf.fit('S1', data, th0=r.x, fixed={jt: lt})
        return 2 * (rr.fun - nll_min) - CHI90
    lo_b, hi_b = cf.LOG_TH_MIN, cf.LOG_TH_MAX
    try:
        lo = lo_b if prof(lo_b) <= 0 else brentq(prof, lo_b, lt_hat, xtol=1e-2)
    except Exception:
        lo = np.nan
    try:
        hi = hi_b if prof(hi_b) <= 0 else brentq(prof, lt_hat, hi_b, xtol=1e-2)
    except Exception:
        hi = np.nan
    # LR test of theta_c = 0 (boundary) using IND (nested at theta_c -> 0)
    lr = 2 * (ri.fun - r.fun)
    mu1_hat = np.exp(r.x[idx['mu']][:I]); ap_hat = np.exp(r.x[idx['ap']]); ac_hat = np.exp(-lt_hat)
    mu1_ind = np.exp(ri.x[ri.idx['mu']][:I]); ap_ind = np.exp(ri.x[ri.idx['ap']])
    tR1, tR12, tD2 = sysq(mus[0], AP, ac)
    sR1, sR12, sD2 = sysq(mu1_hat, ap_hat, ac_hat)
    iR1, iR12, iD2 = sysq(mu1_ind, ap_ind, np.inf)
    th_true = 0.0 if not np.isfinite(ac) else 1 / ac
    lt_true = np.log(th_true) if th_true > 0 else -np.inf
    cover = (lo <= max(lt_true, cf.LOG_TH_MIN) <= hi) if np.isfinite(lo) and np.isfinite(hi) else np.nan
    return dict(I=I, T=T, mub=mub, ac=float(ac), rep=rep, ac_hat=ac_hat, lt_hat=lt_hat, lo=lo, hi=hi,
                cover90=cover, ap_hat=ap_hat, ap_ind=ap_ind, lr=lr, nit=int(r.nit), conv=bool(r.success),
                tR1=tR1, sR1=sR1, iR1=iR1, tR12=tR12, sR12=sR12, iR12=iR12, tD2=tD2, sD2=sD2, iD2=iD2,
                nevents=int(sum(d.sum() for d in data)))

mycells = [c for k, c in enumerate(cells) if k % nw == wid]
fn = f'{OUT}/C_w{wid}.jsonl'
done = set()
if os.path.exists(fn):
    for line in open(fn):
        d = json.loads(line); done.add((d['I'], d['T'], d['mub'], d['ac'], d['rep']))
t0 = time.time()
with open(fn, 'a') as fh:
    for cell in mycells:
        for rep in range(reps):
            key = (cell[0], cell[1], cell[2], float(cell[3]), rep)
            if key in done:
                continue
            try:
                d = run_rep(cell, rep)
            except Exception as e:
                d = dict(I=cell[0], T=cell[1], mub=cell[2], ac=float(cell[3]), rep=rep, error=str(e)[:200])
            fh.write(json.dumps(d, default=float) + '\n'); fh.flush()
        print(wid, cell, round(time.time() - t0, 1), flush=True)
print('DONE', wid, flush=True)
