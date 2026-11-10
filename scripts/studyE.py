"""Study E: can the data tell a common-cause frailty from a binomial-failure-rate (BFR) common shock,
and what does the wrong choice cost?  Usage: python3 studyE.py <worker_id> <n_workers> <reps>
"""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, time, json, itertools
sys.path.insert(0, f'{SP_ROOT}/scripts')
import numpy as np
from scipy import stats
import ccffit2 as cf, ccfmodel as cm

OUT = f'{SP_ROOT}/work/studyE'
os.makedirs(OUT, exist_ok=True)
wid, nw, reps = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
AP = 2.0; G = 3; MUB = 0.1; SIG_C = 0.3; THC = 1.0; P_HIT = 0.5
cells = list(itertools.product(['frailty', 'bfr', 'indep'], [3, 8], [60, 120]))


def sim(mech, mus, T, rng):
    out = []
    for mu in mus:
        m = len(mu)
        if mech == 'frailty':
            out.append(cm.simulate([mu], AP, 1 / THC, T, rng)[0])
        elif mech == 'indep':
            out.append(cm.simulate([mu], AP, np.inf, T, rng)[0])
        else:
            w = THC * np.mean(mu) ** 2 / P_HIT ** 2            # matches the pairwise excess joint probability
            mu_ind = np.maximum(mu - w * P_HIT, 1e-4)          # keeps marginal means equal
            base = cm.simulate([mu_ind], AP, np.inf, T, rng)[0]
            shock = rng.random((T, 1)) < w
            hit = (rng.random((T, m)) < P_HIT) & shock
            out.append(base + hit.astype(int))
    return out


def joint_pair(model, r, mu_scale=1.0):
    """predicted probability that streams 0 and 1 of group 0 are both interrupted in a month"""
    idx = r.idx
    mu = np.exp(r.x[idx['mu']])[:2] * mu_scale
    ap = np.exp(r.x[idx['ap']]) if 'ap' in idx else np.inf
    if model == 'S1':
        return cm.allfail_group(mu, ap, np.exp(-r.x[idx['th'][0]]))
    if model == 'IND':
        p = 1 - (1 + mu / ap) ** (-ap)
        return float(p[0] * p[1])
    if model == 'BFR':
        w = 1 / (1 + np.exp(-r.x[idx['w']])); p = 1 / (1 + np.exp(-r.x[idx['p']]))
        q = (1 + mu / ap) ** (-ap)                    # P(no independent event)
        p_no = (1 - w) * (1 - q[0]) * (1 - q[1]) + w * (1 - q[0] * (1 - p)) * (1 - q[1] * (1 - p))
        return float(p_no)


def truth_joint(mech, mu_group, mu_scale=1.0):
    """true P(streams 0 and 1 both interrupted); mu_scale multiplies each stream's own intensity
    (for BFR the external common-shock process is left unchanged)."""
    mu_group = np.asarray(mu_group, float)
    mu = mu_group[:2] * mu_scale
    if mech == 'frailty':
        return cm.allfail_group(mu, AP, 1 / THC)
    if mech == 'indep':
        p = 1 - (1 + mu / AP) ** (-AP); return float(p[0] * p[1])
    w = THC * np.mean(mu_group) ** 2 / P_HIT ** 2
    mu_ind = np.maximum(mu_group[:2] - w * P_HIT, 1e-4) * mu_scale
    q = (1 + mu_ind / AP) ** (-AP)
    return float((1 - w) * (1 - q[0]) * (1 - q[1]) + w * (1 - q[0] * (1 - P_HIT)) * (1 - q[1] * (1 - P_HIT)))


def run_rep(cell, rep):
    mech, m, T = cell
    rng = np.random.default_rng(abs(hash(('E', mech, m, T, rep))) % (2**32))
    mus = []
    for g in range(G):
        z = rng.normal(0, SIG_C, m); z -= z.mean(); mus.append(MUB * np.exp(z))
    data = sim(mech, mus, T, rng)
    fits = {k: cf.fit(k, data) for k in ['S1', 'BFR', 'IND']}
    npar = {k: fits[k].idx['k'] for k in fits}
    n_obs = G * T
    aic = {k: 2 * fits[k].fun + 2 * npar[k] for k in fits}
    bic = {k: 2 * fits[k].fun + np.log(n_obs) * npar[k] for k in fits}
    out = dict(mech=mech, m=m, T=T, rep=rep,
               sel_aic=min(aic, key=aic.get), sel_bic=min(bic, key=bic.get),
               **{f'nll_{k}': fits[k].fun for k in fits})
    # consequence: joint loss now and after halving every stream's base rate
    for k in fits:
        out[f'j1_{k}'] = joint_pair(k, fits[k]); out[f'jh_{k}'] = joint_pair(k, fits[k], 0.5)
    out['j1_true'] = truth_joint(mech, mus[0]); out['jh_true'] = truth_joint(mech, mus[0], 0.5)
    return out


mycells = [c for k, c in enumerate(cells) if k % nw == wid]
fn = f'{OUT}/E_w{wid}.jsonl'
done = set()
if os.path.exists(fn):
    for line in open(fn):
        d = json.loads(line); done.add((d['mech'], d['m'], d['T'], d['rep']))
t0 = time.time()
with open(fn, 'a') as fh:
    for cell in mycells:
        for rep in range(reps):
            if (cell[0], cell[1], cell[2], rep) in done:
                continue
            try:
                d = run_rep(cell, rep)
            except Exception as e:
                d = dict(mech=cell[0], m=cell[1], T=cell[2], rep=rep, error=str(e)[:300])
            fh.write(json.dumps(d, default=float) + '\n'); fh.flush()
        print(wid, cell, round(time.time() - t0, 1), flush=True)
print('DONE', wid, flush=True)
