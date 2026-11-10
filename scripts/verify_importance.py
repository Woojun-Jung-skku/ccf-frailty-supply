"""Numerical checks of Proposition 3 (importance decomposition): exact identity, pairwise truncation,
common-cause importance and re-sourcing priority; data for Fig. 2(c). Output: work/verify_importance.json"""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json, itertools, time
sys.path.insert(0, f'{SP_ROOT}/scripts')
import numpy as np, ccfmodel as cm

SP = f'{SP_ROOT}'
R = {}
ap = 2.0


def full_identity(h, groups, mu, acs):
    """R by the exact identity: sum over subset choices (empty or |A_g|>=2) in each group."""
    ids = [i for g in groups for i in g]
    p = {}; kap = []
    for g, ac in zip(groups, acs):
        pg, _ = cm.margins_and_pair_cov([mu[i] for i in g], ap, ac)
        p.update(dict(zip(g, pg)))
        k = {}
        for r in range(2, len(g) + 1):
            for A in itertools.combinations(range(len(g)), r):
                k[tuple(g[a] for a in A)] = cm.comoment([mu[i] for i in g], ap, ac, A)
        kap.append(k)
    h0 = h(p); tot = 0.0; by_order = {}
    for combo in itertools.product(*[[()] + list(k.keys()) for k in kap]):
        A = [i for part in combo for i in part]
        if not A:
            continue
        coef = h.mixed_partial(p, A)
        km = 1.0
        for gi, part in enumerate(combo):
            if part:
                km *= kap[gi][part]
        tot += coef * km
        by_order[len(A)] = by_order.get(len(A), 0.0) + coef * km
    return h0, h0 + tot, by_order, p


# 1. BOM example (Table C.1 / S1): streams 0,1 in group A (ac=1), 2,3 in group B (ac=3)
mu = {0: 0.15, 1: 0.10, 2: 0.08, 3: 0.12}
struct = lambda W: ((0 in W) or (2 in W)) and ((1 in W) or (3 in W))
h = cm.ReliabilityPolynomial([0, 1, 2, 3], structure=struct)
for name, groups in [('different_groups', [[0, 1], [2, 3]]), ('same_group', [[0, 2], [1, 3]])]:
    acs = [1.0, 3.0]
    exact = cm.system_reliability(struct, groups, mu, ap, acs, 160)
    h0, ident, by_order, p = full_identity(h, groups, mu, acs)
    R[f'bom_{name}'] = dict(exact=exact, margins_only=h0, identity=ident, pairwise=h0 + by_order.get(2, 0.0),
                            by_order={str(k): v for k, v in by_order.items()})

# 2. a 12-stream fab-input system: 6 inputs x 2 sources, three origins; events: no input lost / fewer than two lost
rng = np.random.default_rng(3)
n = 12; ids = list(range(n))
mu12 = {i: float(x) for i, x in enumerate(rng.uniform(0.02, 0.12, n))}
inputs = {k: [2 * k, 2 * k + 1] for k in range(6)}
shares = {i: 0.5 for i in ids}                       # an input is lost only when both sources are interrupted
groups = [[0, 2, 4, 6, 8], [1, 3, 10], [5, 7, 9, 11]]
acs = [0.5, 2.0, 1.0]
events = {'no_input_lost': dict(k_fail=1), 'fewer_than_two_lost': dict(k_fail=2)}
R['system12'] = dict(mu=mu12, groups=groups, acs=acs)
for ev, kw in events.items():
    hh = cm.ReliabilityPolynomial(ids, inputs=inputs, shares=shares, thr=0.99, **kw)
    def st(W, kf=kw['k_fail']):
        lost = sum(1 for k, (a, b) in inputs.items() if a not in W and b not in W)
        return lost < kf
    exact = cm.system_reliability(st, groups, mu12, ap, acs, 160)
    p = {}; covs = []
    for g, ac in zip(groups, acs):
        pg, C = cm.margins_and_pair_cov([mu12[i] for i in g], ap, ac); p.update(dict(zip(g, pg))); covs.append(C)
    imp = cm.importance_pairwise(hh, p, groups, covs)
    out = dict(exact=exact, margins_only=hh(p), pairwise=hh(p) + imp['total'], per_group_pairwise=imp['per_group'],
               priority=imp['priority'])
    # exact common-cause importance per group: replace the group's frailty by independence with the same margins
    out['per_group_exact'] = []
    for gi, (g, ac) in enumerate(zip(groups, acs)):
        mu_adj = dict(mu12)
        for i in g:
            mu_adj[i] = ap * (p[i] ** (-1 / ap) - 1)          # item model mean with the same margin
        acs_g = list(acs); acs_g[gi] = np.inf
        out['per_group_exact'].append(exact - cm.system_reliability(st, groups, mu_adj, ap, acs_g, 160))
    R['system12'][ev] = out

# 3. Fig. 2(c): common-cause importance of ONE group of 6 streams (mu = 0.05) for three structures, vs alpha_c,
#    exact and pairwise; structures: series (all 6 deliver), 2-out-of-6 failure (fewer than 2 interrupted), parallel (not all interrupted)
m = 6; mu6 = {i: 0.05 for i in range(m)}
grid = [0.1, 0.14, 0.2, 0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0]
fig2c = {}
for name, kf in [('series', 1), ('fewer_than_two', 2), ('parallel', m)]:
    st = lambda W, kf=kf: (m - len(W)) < kf
    hh = cm.ReliabilityPolynomial(list(range(m)), structure=st)
    ex, pw, h0s = [], [], []
    for ac in grid:
        p, C = cm.margins_and_pair_cov([0.05] * m, ap, ac)
        pd_ = dict(zip(range(m), p))
        h0 = hh(pd_)
        exact = cm.system_reliability(st, [list(range(m))], mu6, ap, [ac], 160)
        imp = cm.importance_pairwise(hh, pd_, [list(range(m))], [C])
        ex.append(exact - h0); pw.append(imp['total']); h0s.append(h0)
    fig2c[name] = dict(ac=grid, exact=ex, pairwise=pw, h0=h0s)
R['fig2c'] = fig2c
json.dump(R, open(f'{SP}/work/verify_importance.json', 'w'), indent=1, default=float)
for k in ['bom_different_groups', 'bom_same_group']:
    print(k, {kk: (round(v, 6) if isinstance(v, float) else v) for kk, v in R[k].items() if kk != 'by_order'}, R[k]['by_order'])
for ev in events:
    o = R['system12'][ev]
    print(ev, 'exact', round(o['exact'], 6), 'margins', round(o['margins_only'], 6), 'pairwise', round(o['pairwise'], 6),
          'I_g exact', [round(v, 6) for v in o['per_group_exact']], 'I_g pairwise', [round(v, 6) for v in o['per_group_pairwise']])
for name, d in fig2c.items():
    print(name, 'exact', [round(v, 5) for v in d['exact']][:6], 'pairwise', [round(v, 5) for v in d['pairwise']][:6])
