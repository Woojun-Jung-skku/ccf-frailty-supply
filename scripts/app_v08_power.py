"""What the case-study panel can detect (Section 5.2 and Supplementary S8, Table S8).

(a) Pair rule. For the binary-onset likelihood the score for theta_c at independence is, per group-period,
    (1/2) sum_{i != j} u_i u_j - (1/2) sum_i mu_i u_i   with   u_i = mu_i (x_i - p_i) / p_i,   p_i = 1 - exp(-mu_i),
    and the second sum lies in the span of the scores of the margins. The information about theta_c that remains after
    the margins are estimated is therefore the binary form of Lemma 1,
    (1/2) sum_{i != j} s_i s_j   with   s_i = Var u_i = mu_i^2 (1 - p_i) / p_i,
    summed over the months in which both streams are active. It is evaluated at the fitted margins of the main model,
    with the Lunar New Year multiplier on the China group, and gives a standard error of theta_c at independence and a
    local approximation to the power of the 5% boundary test.

(b) Simulated power. Panels are generated under the main model with a common cause of strength alpha_c in every
    origin group: margins at their fitted values, the activity pattern of the data, the Lunar New Year multiplier on
    the China group. Each panel is refitted and each origin is tested against independence at the 5% level of the
    boundary test (likelihood ratio above 2.71).

usage:  app_v08_power.py sim <alpha_c|inf> <n_rep> <seed>    appends one record per panel to work/app/app_v08_power.jsonl
        app_v08_power.py summary                              writes work/app/app_v08_power.json
The archived records were produced by
        sim 2 200 101;  sim 2 200 102;  sim 1 150 103;  sim 5 150 104;  sim inf 150 105;  sim 0.5 100 106
"""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json
sys.path.insert(0, f'{SP_ROOT}/scripts')
import numpy as np
from scipy import stats
import app_model2 as am
from app_model2 import Model2, GROUPS, multi, origins, H1, W, S1_STARTS
import ccffit2 as cf

CUT = float(stats.chi2.ppf(0.90, 1))          # 2.71: 5% level of the boundary test
R = json.load(open(f'{W}/app_v07_main.json'))
beta = R['fits']['beta']['hat']
ac_hat = {o: R['alpha_c'][o]['ac_hat'] for o in multi}
live = lambda a: np.isfinite(a) and a < 5e3


def fitted_hazard(o, sids):
    """-log P(no onset) of each stream in an ordinary month under the fitted model"""
    mu = np.array([R['mu_S1'][o][s] for s in sids])
    a = ac_hat[o]
    return a * np.log1p(mu / a) if live(a) else mu


def simulate(ac_true, n_rep, seed):
    rng = np.random.default_rng(seed)
    out = open(f'{W}/app_v08_power.jsonl', 'a')
    for rep in range(n_rep):
        G = {}
        for o in origins.index:
            sids, X = GROUPS[o]
            if o not in multi:
                G[o] = (sids, X); continue
            m = fitted_hazard(o, sids)
            mu_a = ac_true * np.expm1(m / ac_true) if live(ac_true) else m      # means that keep the margins under ac_true
            eta = rng.gamma(ac_true, 1 / ac_true, size=X.shape[0]) if live(ac_true) else np.ones(X.shape[0])
            h = np.where((H1 == 1) & (o == 'China'), np.exp(beta), 1.0)
            p = 1 - np.exp(-mu_a[None, :] * (eta * h)[:, None])
            Xs = (rng.random(X.shape) < p).astype(int); Xs[X < 0] = -1
            G[o] = (sids, Xs)
        M1 = Model2('S1', groups=G, H=H1, scope='China'); r1 = M1.fit(starts=S1_STARTS); x1 = r1.x; nll1 = float(r1.fun)
        rec = dict(ac_true=(ac_true if live(ac_true) else 'inf'), seed=seed, rep=rep)
        for o in multi:
            j = M1.gp[o]
            rec[o] = dict(ac_hat=M1.alpha_c(x1, o), lr=float(2 * (M1.fit(x0=x1, fixed={j: cf.LOG_TH_MIN}).fun - nll1)))
        out.write(json.dumps(rec) + '\n'); out.flush()


def pair_information():
    out = {}
    for o in multi:
        sids, X = GROUPS[o]
        act = (X >= 0).astype(float)
        m = fitted_hazard(o, sids)
        h = np.where((H1 == 1) & (o == 'China'), np.exp(beta), 1.0)
        mu = np.outer(h, m)                                  # the model at independence: Poisson means mu_i exp(beta H_t)
        p = -np.expm1(-mu)
        s = mu ** 2 * (1 - p) / p * act
        info = 0.5 * float(((s.sum(1)) ** 2 - (s ** 2).sum(1)).sum())
        se = info ** -0.5
        out[o] = dict(information=info, se_theta=se,
                      power_local={str(a): float(stats.norm.sf(stats.norm.ppf(0.95) - (1 / a) / se)) for a in (0.5, 1, 2, 5)})
    return out


def summary():
    recs = [json.loads(l) for l in open(f'{W}/app_v08_power.jsonl')]
    OUT = dict(cutoff=CUT, pair_rule=pair_information(), simulated={})
    for r in recs:
        r['ac_true'] = float(r['ac_true'])
    for a in sorted({r['ac_true'] for r in recs}):
        rr = [r for r in recs if r['ac_true'] == a]
        d = dict(n=len(rr), seeds=sorted({r['seed'] for r in rr}))
        for o in multi:
            lr = np.array([r[o]['lr'] for r in rr]); ah = np.array([r[o]['ac_hat'] for r in rr])
            d[o] = dict(reject=float((lr > CUT).mean()), median_ac_hat=float(np.median(ah)), share_at_boundary=float((ah > 5e3).mean()))
        d['any_origin'] = float(np.mean([any(r[o]['lr'] > CUT for o in multi) for r in rr]))
        OUT['simulated']['inf' if not np.isfinite(a) else f'{a:g}'] = d
    json.dump(OUT, open(f'{W}/app_v08_power.json', 'w'), indent=1)
    for o in multi:
        e = OUT['pair_rule'][o]
        print(f"{o}: information {e['information']:.2f}, s.e. of theta_c {e['se_theta']:.3f}, local power {({k: round(v, 2) for k, v in e['power_local'].items()})}")
    for a, d in OUT['simulated'].items():
        print(f"alpha_c = {a} (n = {d['n']}): reject " + ', '.join(f"{o} {d[o]['reject']:.3f}" for o in multi) + f"; any origin {d['any_origin']:.3f}")


if __name__ == '__main__':
    if len(sys.argv) >= 5 and sys.argv[1] == 'sim':
        simulate(float(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]))
    elif len(sys.argv) == 2 and sys.argv[1] == 'summary':
        summary()
    else:
        sys.exit(__doc__)
