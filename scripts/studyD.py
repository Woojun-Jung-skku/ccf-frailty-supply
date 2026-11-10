"""Study D: system-level monitoring vs unions of item alarms under common-cause frailty.
Usage: python3 studyD.py <worker_id> <n_workers> <reps>
"""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, time, json, itertools
sys.path.insert(0, f'{SP_ROOT}/scripts')
import numpy as np
from scipy.special import gammaincc
from scipy.signal import lfilter
import ccffit2 as cf, ccfmodel as cm

OUT = f'{SP_ROOT}/work/studyD'
os.makedirs(OUT, exist_ok=True)
wid, nw, reps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])

AP = 2.0; MUB = 0.1; SIG_C = 0.3
T_CAL, T_MON, T0 = 36, 120, 24          # calibration months, monitoring months, change at monitoring month 24
DELTA, RHO, PRIOR_W = 0.9, 1.5, 6.0     # discount, level-shift factor, prior weight (months of baseline exposure)
ARL0 = 24.0; B = 400
cells = list(itertools.product([5, 10, 30], [0.5, 2.0, np.inf]))


def qstat(N, mu):
    """N: (..., T, k) counts; mu: (..., 1, k) or (k,) baseline means -> q = P(r > RHO) per month"""
    T = N.shape[-2]
    t = np.arange(1, T + 1)[:, None]
    ct = DELTA ** t * PRIOR_W + (1 - DELTA ** t) / (1 - DELTA)          # (T,1)
    b = mu * ct
    a = lfilter([1.0], [1.0, -DELTA], N, axis=-2) + DELTA ** t * PRIOR_W * mu
    return gammaincc(a, RHO * b)


def eb_baseline(Ncal, kappa=12.0):
    """empirical-Bayes (gamma-Poisson) shrinkage of stream means toward the group mean.
    Ncal: (..., Tc, k) -> (..., 1, k)"""
    Tc = Ncal.shape[-2]
    tot = Ncal.sum(-2, keepdims=True)
    gbar = tot.mean(-1, keepdims=True) / Tc
    gbar = np.maximum(gbar, 0.5 / (Tc * Ncal.shape[-1]))
    return (tot + kappa * gbar) / (Tc + kappa)


def pers_min(q):
    """min(q_{t-1}, q_t): persistent (2-of-2) alarm iff >= threshold; first month cannot alarm"""
    m = np.minimum(q[..., 1:, :], q[..., :-1, :])
    pad = np.zeros(q.shape[:-2] + (1, q.shape[-1]))
    return np.concatenate([pad, m], axis=-2)


def calib(stat, target=ARL0):
    """threshold = (1 - 1/target) quantile of the persistent statistic over all path-months"""
    return float(np.quantile(stat.ravel(), 1 - 1 / target, method='higher'))


def sim_paths(mu, ap, ac, T, n, rng, mult=None):
    """n paths x T x k counts from S-1; mult: (T, k) intensity multipliers"""
    k = len(mu)
    eta = rng.gamma(ac, 1 / ac, size=(n, T, 1)) if np.isfinite(ac) else np.ones((n, T, 1))
    xi = rng.gamma(ap, 1 / ap, size=(n, T, k)) if np.isfinite(ap) else np.ones((n, T, k))
    lam = mu[None, None, :] * xi * eta
    if mult is not None:
        lam = lam * mult[None]
    return rng.poisson(lam)


def channels(N, mu_hat):
    """mu_hat: (..., 1, k). returns persistent statistics: per-item (...,T,k) and system (...,T,1)"""
    qi = qstat(N, mu_hat)
    qs = qstat(N.sum(-1, keepdims=True), mu_hat.sum(-1, keepdims=True))
    return pers_min(qi), pers_min(qs)


def run_rep(cell, rep):
    I, ac = cell
    seed = abs(hash(('D', I, float(ac), rep))) % (2**32)
    rng = np.random.default_rng(seed)
    z = rng.normal(0, SIG_C, I); z -= z.mean()
    mu = MUB * np.exp(z)
    # calibration window (truth) -> model fits used only to generate bootstrap data
    Ncal = sim_paths(mu, AP, ac, T_CAL, 1, rng)[0]
    r1 = cf.fit('S1', [Ncal]); ri = cf.fit('IND', [Ncal])
    mu1 = np.exp(r1.x[r1.idx['mu']]); ap1 = np.exp(r1.x[r1.idx['ap']]); ac1 = np.exp(-r1.x[r1.idx['th'][0]])
    mui = np.exp(ri.x[ri.idx['mu']]); api = np.exp(ri.x[ri.idx['ap']])
    # parametric bootstrap of the whole procedure (baseline estimation + monitoring)
    Pb1 = sim_paths(mu1, ap1, ac1, T_CAL + T_MON, B, rng)
    Pbi = sim_paths(mui, api, np.inf, T_CAL + T_MON, B, rng)
    mi1, ms1 = channels(Pb1[:, T_CAL:], eb_baseline(Pb1[:, :T_CAL]))
    mii, msi = channels(Pbi[:, T_CAL:], eb_baseline(Pbi[:, :T_CAL]))
    th = {}
    th['item_each'] = np.array([calib(mii[..., j]) for j in range(I)])     # each item to 24 months (item model)
    th['union_ind'] = calib(mii.max(-1))                                  # union, independence calibration
    th['union_dep'] = calib(mi1.max(-1))                                  # union, fitted S-1 calibration
    th['sys_dep'] = calib(ms1[..., 0])                                    # system channel, fitted S-1
    th['sys_ind'] = calib(msi[..., 0])                                    # system channel, independence
    base = eb_baseline(Ncal[None])                                        # truth baselines (1,1,k)
    out = dict(I=I, ac=float(ac), rep=rep, ac_hat=ac1, ap_hat=ap1, ap_ind=api)
    m_common = np.ones((T_MON, I)); m_common[T0:, :] = 2.0
    m_single = np.ones((T_MON, I)); m_single[T0:, 0] = 3.0
    for sname, mult in [('stable', None), ('common', m_common), ('single', m_single)]:
        N = sim_paths(mu, AP, ac, T_MON, 1, rng, mult)
        mi, ms = channels(N, base)
        mi, ms = mi[0], ms[0][:, 0]
        alarms = {
            'item_each_union': (mi >= th['item_each'][None, :]).any(-1),
            'union_ind': mi.max(-1) >= th['union_ind'],
            'union_dep': mi.max(-1) >= th['union_dep'],
            'sys_dep': ms >= th['sys_dep'],
            'sys_ind': ms >= th['sys_ind'],
        }
        for k, a in alarms.items():
            if sname == 'stable':
                out[f'{k}_frac'] = float(a.mean())
            else:
                post = np.nonzero(a[T0:])[0]
                out[f'{k}_{sname}_delay'] = float(post[0]) if len(post) else np.nan
    return out


mycells = [c for k, c in enumerate(cells) if k % nw == wid]
fn = f'{OUT}/D_w{wid}.jsonl'
done = set()
if os.path.exists(fn):
    for line in open(fn):
        d = json.loads(line); done.add((d['I'], d['ac'], d['rep']))
t0 = time.time()
with open(fn, 'a') as fh:
    for cell in mycells:
        for rep in range(reps):
            if (cell[0], float(cell[1]), rep) in done:
                continue
            try:
                d = run_rep(cell, rep)
            except Exception as e:
                d = dict(I=cell[0], ac=float(cell[1]), rep=rep, error=str(e)[:300])
            fh.write(json.dumps(d, default=float) + '\n'); fh.flush()
        print(wid, cell, round(time.time() - t0, 1), flush=True)
print('DONE', wid, flush=True)
