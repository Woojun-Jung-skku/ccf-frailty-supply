"""Numerical verification of every closed form used in the paper (Monte Carlo + adaptive quadrature)."""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, sys, time
import numpy as np
from scipy import integrate, special, stats
sys.path.insert(0, f'{SP_ROOT}/scripts')
import ccfmodel as cm

rng = np.random.default_rng(20261030)
R = {}

def gpdf(e, a):
    return stats.gamma.pdf(e, a, scale=1 / a)

# 1. quadrature accuracy against adaptive quad
acc = []
for ac in [0.3, 0.5, 1, 2, 5, 20, 80]:
    for ap in [0.5, 2, np.inf]:
        mu = np.array([0.05, 0.2, 0.5])
        f = lambda e: np.prod(cm.q_cond(mu, np.atleast_1d(e), ap), axis=0)[0]
        ref = integrate.quad(lambda e: f(e) * gpdf(e, ac), 0, np.inf, limit=500, epsabs=1e-13)[0]
        val = cm.series_group(mu, ap, ac)
        acc.append(abs(val - ref))
R['quad_max_abs_err_series'] = max(acc)

# 2. moments (Prop 1 i) by MC
mu = np.array([0.2, 0.3, 0.1, 0.4]); ap = 2.0; ac = 2.0; T = 2_000_000
N = cm.simulate([mu], ap, ac, T, rng)[0]
th_var = mu + mu**2 * (1/ap + 1/ac + 1/(ap*ac))
R['var_theory'] = th_var.round(4).tolist()
R['var_mc'] = N.var(axis=0).round(4).tolist()
R['cov01_theory'] = float(mu[0]*mu[1]/ac)
R['cov01_mc'] = float(np.cov(N[:, 0], N[:, 1])[0, 1])

# 3. system reliability: series, Jensen/fixed-ap bound, association bound
ser_th = cm.series_group(mu, ap, ac)
ser_mc = float(np.mean(N.sum(axis=1) == 0))
marg = np.array([cm.sri_marginal(m, ap, ac) for m in mu])
R['series_theory'] = ser_th; R['series_mc'] = ser_mc
R['series_product_of_marginals'] = float(np.prod(marg))
R['series_at_eta1_(item_model_same_ap)'] = float(np.prod((1 + mu/ap)**(-ap)))
# S-0 closed form
N0 = cm.simulate([mu], np.inf, ac, T, rng)[0]
R['S0_series_closed'] = float((1 + mu.sum()/ac)**(-ac)); R['S0_series_mc'] = float(np.mean(N0.sum(axis=1) == 0))
R['indep_poisson_series'] = float(np.exp(-mu.sum()))

# 4. parallel (all fail) closed form S-0 and S-1
for tag, a_p, NN in [('S0', np.inf, N0), ('S1', ap, N)]:
    R[f'{tag}_allfail_theory'] = cm.allfail_group(mu[:2], a_p, ac)
    R[f'{tag}_allfail_mc'] = float(np.mean((NN[:, 0] > 0) & (NN[:, 1] > 0)))

# 5. general coherent structure: series-parallel BOM across two groups
#    streams 0,1 in group A; 2,3 in group B. item X needs (0 or 2); item Y needs (1 or 3)
groups = [[0, 1], [2, 3]]
mu_s = {0: 0.15, 1: 0.10, 2: 0.08, 3: 0.12}
struct = lambda W: ((0 in W) or (2 in W)) and ((1 in W) or (3 in W))
acs = [1.0, 3.0]
rel_th = cm.system_reliability(struct, groups, mu_s, ap, acs)
NA = cm.simulate([[mu_s[0], mu_s[1]]], ap, acs[0], T, rng)[0]
NB = cm.simulate([[mu_s[2], mu_s[3]]], ap, acs[1], T, rng)[0]
x = np.column_stack([NA[:, 0] == 0, NA[:, 1] == 0, NB[:, 0] == 0, NB[:, 1] == 0])
rel_mc = float(np.mean((x[:, 0] | x[:, 2]) & (x[:, 1] | x[:, 3])))
R['bom_theory'] = rel_th; R['bom_mc'] = rel_mc
# alternative design: sources of each item in different groups vs same group
groups2 = [[0, 2], [1, 3]]  # now item X's two sources share a group
rel_same = cm.system_reliability(struct, groups2, mu_s, ap, acs)
R['bom_sources_same_group'] = rel_same

# 6. monotonicity in ac at fixed ap (series decreasing)
grid = [0.25, 0.5, 1, 2, 5, 20, np.inf]
R['series_vs_ac'] = [round(cm.series_group(mu, ap, a), 5) for a in grid]
R['S0_series_vs_ac'] = [round(cm.series_group(mu, np.inf, a), 5) for a in grid]

# 7. multiplicity pmf and alpha-factor asymptotics
m, mu1, ac1 = 10, 0.05, 2.0
pmf0 = cm.multiplicity_pmf(m, mu1, np.inf, ac1)
NN = cm.simulate([[mu1]*m], np.inf, ac1, 4_000_000, rng)[0]
K = (NN > 0).sum(axis=1)
R['mult_S0_theory'] = pmf0[:5].round(5).tolist()
R['mult_S0_mc'] = [round(float(np.mean(K == k)), 5) for k in range(5)]
pmf1 = cm.multiplicity_pmf(m, mu1, 2.0, ac1)
NN1 = cm.simulate([[mu1]*m], 2.0, ac1, 4_000_000, rng)[0]
K1 = (NN1 > 0).sum(axis=1)
R['mult_S1_theory'] = pmf1[:5].round(5).tolist()
R['mult_S1_mc'] = [round(float(np.mean(K1 == k)), 5) for k in range(5)]
# amplification relative to independence (same marginals) at small mu
amp = []
for mu_small in [1e-2, 1e-3, 1e-4]:
    p = cm.multiplicity_pmf(m, mu_small, np.inf, ac1)
    pind = 1 - cm.sri_marginal(mu_small, np.inf, ac1)
    ind = np.array([special.comb(m, k) * pind**k * (1 - pind)**(m - k) for k in range(m + 1)])
    amp.append((p[2:5] / ind[2:5]).round(4).tolist())
R['amp_k2_k4_vs_indep_small_mu'] = amp
R['amp_theory_k2_k4'] = [round(np.prod([1 + j/ac1 for j in range(k)]), 4) for k in [2, 3, 4]]

# 8. joint-interruption fraction beta_J = 1/(1+ac) at small mu, both S-0 and S-1
bj = {}
for a in [0.5, 2, 10]:
    for a_p in [np.inf, 2.0]:
        b, ratio, *_ = cm.beta_joint(1e-3, 2e-3, a_p, a)
        bj[f'ac={a},ap={a_p}'] = (round(b, 4), round(ratio, 4), round(1/(1+a), 4), round(1+1/a, 4))
R['beta_J_check(b,ratio,theory_b,theory_ratio)'] = bj

# 9. Fisher information at the boundary: MC variance of scores under Poisson
I = 10; mu_eq = np.full(I, 0.1)
Ip, sc, sp = cm.fisher_boundary(mu_eq)
Y = rng.poisson(mu_eq, size=(2_000_000, I))
s_p = 0.5 * np.sum((Y - mu_eq)**2 - Y, axis=1)
s_c = 0.5 * ((Y.sum(1) - mu_eq.sum())**2 - Y.sum(1))
covm = np.cov(np.vstack([s_p, s_c]))
R['info_theory'] = Ip.round(4).tolist(); R['info_mc'] = covm.round(4).tolist()
ratios = {}
for Ii in [2, 5, 10, 30]:
    _, c_, p_ = cm.fisher_boundary(np.full(Ii, 0.05))
    ratios[Ii] = (round(c_ / p_, 4), round(8 / (Ii*(Ii-1)*0.05**2), 1))
R['schur_ratio_and_periods_SE0.5_mu0.05'] = ratios

# 10. union of item alarms: any-shock rule run length equals series MTBF
ac_u, ap_u, Iu = 2.0, 2.0, 10
mu_u = np.full(Iu, 0.02)
Rser = cm.series_group(mu_u, ap_u, ac_u)
NU = cm.simulate([mu_u], ap_u, ac_u, 2_000_000, rng)[0]
R['union_anyshock_ARL_theory'] = 1 / (1 - Rser)
R['union_anyshock_ARL_mc'] = 1 / float(np.mean(NU.sum(1) > 0))
R['union_anyshock_ARL_indep_same_marg'] = 1 / (1 - np.prod([cm.sri_marginal(x, ap_u, ac_u) for x in mu_u]))

json.dump(R, open(f'{SP_ROOT}/work/verify_theory.json', 'w'), indent=1, default=float)
for k, v in R.items():
    print(k, '=', v)
