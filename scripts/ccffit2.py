"""Fast stacked-group marginal likelihoods (equal number of streams per group, patterns padded).

Groups g = 1..G, each with m streams (pad with dummy streams if needed: counts 0, fixed log mu).
Data: U (G x P x m) unique count patterns (padded rows have C = 0), C (G x P) multiplicities.
Parameters (unconstrained vector th):
  log_mu (G*m) | log_ap (S1, IND, BFR) | log_thc (S1, S0; 1 pooled or G per group) | logit_w, logit_p (BFR)
"""
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, polygamma, logsumexp
from scipy.optimize import minimize
from functools import lru_cache

jax.config.update('jax_enable_x64', True)

HZ = 0.04
ZGRID = jnp.asarray(np.arange(-30.0, 8.0 + 1e-9, HZ))
LOG_TH_MIN, LOG_TH_MAX = np.log(1e-4), np.log(50.0)
LOG_AP_MIN, LOG_AP_MAX = np.log(0.05), np.log(1e4)


def eta_logw(ac):
    sig = jnp.sqrt(polygamma(1, ac))
    s = sig * ZGRID
    logp = ac * jnp.log(ac) - gammaln(ac) + ac * s - ac * jnp.exp(s)
    return s, logp + jnp.log(sig) + jnp.log(HZ)


def ll_S1_group(U, C, log_mu, log_ap, log_thc):
    a = jnp.exp(log_ap)
    ac = jnp.exp(-log_thc)
    s, logw = eta_logw(ac)                                   # Z
    mu = jnp.exp(log_mu)                                     # m
    L = jnp.log(a + mu[:, None] * jnp.exp(s)[None, :])       # m x Z
    n = U.astype(jnp.float64)                                # P x m
    npl = jnp.sum(n, axis=1)
    cst = jnp.sum(gammaln(n + a) - gammaln(a) - gammaln(n + 1.0), axis=1) + n.shape[1] * a * jnp.log(a) + n @ log_mu
    lp = cst[:, None] - a * jnp.sum(L, axis=0)[None, :] - n @ L + npl[:, None] * s[None, :]
    ll = logsumexp(lp + logw[None, :], axis=1)
    return jnp.sum(C * ll)


def ll_S0_group(U, C, log_mu, log_thc):
    ac = jnp.exp(-log_thc)
    mu = jnp.exp(log_mu)
    M = jnp.sum(mu)
    n = U.astype(jnp.float64)
    tot = jnp.sum(n, axis=1)
    ll = (gammaln(ac + tot) - gammaln(ac) - jnp.sum(gammaln(n + 1.0), axis=1)
          + ac * (jnp.log(ac) - jnp.log(ac + M)) + n @ log_mu - tot * jnp.log(ac + M))
    return jnp.sum(C * ll)


def nb_logpmf(n, m, a):
    return (gammaln(n + a) - gammaln(a) - gammaln(n + 1.0)
            + a * (jnp.log(a) - jnp.log(a + m)) + n * (jnp.log(m) - jnp.log(a + m)))


def ll_IND_group(U, C, log_mu, log_ap):
    a = jnp.exp(log_ap)
    n = U.astype(jnp.float64)
    lp = jnp.sum(nb_logpmf(n, jnp.exp(log_mu)[None, :], a), axis=1)
    return jnp.sum(C * lp)


def ll_BFR_group(U, C, log_mu, log_ap, logit_w, logit_p):
    a = jnp.exp(log_ap)
    w = jax.nn.sigmoid(logit_w)
    p = jax.nn.sigmoid(logit_p)
    n = U.astype(jnp.float64)
    mu = jnp.exp(log_mu)[None, :]
    l0 = nb_logpmf(n, mu, a)
    l1 = jnp.where(n >= 1, nb_logpmf(jnp.maximum(n - 1.0, 0.0), mu, a), -jnp.inf)
    hit = jnp.logaddexp(jnp.log1p(-p) + l0, jnp.log(p) + l1)
    ll = jnp.logaddexp(jnp.log1p(-w) + jnp.sum(l0, axis=1), jnp.log(w) + jnp.sum(hit, axis=1))
    return jnp.sum(C * ll)


def layout(model, G, m, per_group_ac=False):
    k = G * m
    idx = {'mu': np.arange(k)}
    if model in ('S1', 'IND', 'BFR'):
        idx['ap'] = k; k += 1
    if model in ('S1', 'S0'):
        nth = G if per_group_ac else 1
        idx['th'] = np.arange(k, k + nth); k += nth
    if model == 'BFR':
        idx['w'] = k; idx['p'] = k + 1; k += 2
    idx['k'] = k
    return idx


@lru_cache(maxsize=None)
def compiled(model, G, m, per_group_ac=False):
    idx = layout(model, G, m, per_group_ac)

    def negll(th, U, C):
        lmu = th[idx['mu']].reshape(G, m)
        if model == 'S1':
            th_c = th[idx['th']] if per_group_ac else jnp.repeat(th[idx['th']], G)
            v = jax.vmap(ll_S1_group, in_axes=(0, 0, 0, None, 0))(U, C, lmu, th[idx['ap']], th_c)
        elif model == 'S0':
            th_c = th[idx['th']] if per_group_ac else jnp.repeat(th[idx['th']], G)
            v = jax.vmap(ll_S0_group, in_axes=(0, 0, 0, 0))(U, C, lmu, th_c)
        elif model == 'IND':
            v = jax.vmap(ll_IND_group, in_axes=(0, 0, 0, None))(U, C, lmu, th[idx['ap']])
        elif model == 'BFR':
            v = jax.vmap(ll_BFR_group, in_axes=(0, 0, 0, None, None, None))(
                U, C, lmu, th[idx['ap']], th[idx['w']], th[idx['p']])
        return -jnp.sum(v)

    f = jax.jit(negll)
    g = jax.jit(jax.grad(negll))
    h = jax.jit(jax.hessian(negll))
    return idx, f, g, h


def pack(groups, bucket=None):
    """groups: list of (T x m) arrays with equal m -> padded U, C"""
    Us, Cs = [], []
    for X in groups:
        u, c = np.unique(np.asarray(X, int), axis=0, return_counts=True)
        Us.append(u); Cs.append(c)
    P = max(len(c) for c in Cs)
    if bucket is None:
        bucket = int(2 ** np.ceil(np.log2(max(P, 8))))
    G, m = len(groups), groups[0].shape[1]
    U = np.zeros((G, bucket, m), int); C = np.zeros((G, bucket))
    for g in range(G):
        U[g, :len(Cs[g])] = Us[g]; C[g, :len(Cs[g])] = Cs[g]
    return jnp.asarray(U), jnp.asarray(C)


def fit(model, groups, per_group_ac=False, th0=None, fixed=None, dummy=None):
    G, m = len(groups), groups[0].shape[1]
    idx, f, g, h = compiled(model, G, m, per_group_ac)
    U, C = pack(groups)
    k = idx['k']
    x0 = np.zeros(k)
    means = np.concatenate([np.maximum(np.asarray(X).mean(0), 0.5 / X.shape[0]) for X in groups])
    x0[idx['mu']] = np.log(means)
    bnds = [(-14.0, 3.0)] * (G * m)
    if 'ap' in idx:
        x0[idx['ap']] = np.log(2.0); bnds.append((LOG_AP_MIN, LOG_AP_MAX))
    if 'th' in idx:
        x0[idx['th']] = np.log(0.5); bnds += [(LOG_TH_MIN, LOG_TH_MAX)] * len(idx['th'])
    if 'w' in idx:
        x0[idx['w']] = -3.0; x0[idx['p']] = 0.0; bnds += [(-14.0, 6.0), (-10.0, 10.0)]
    if th0 is not None:
        x0 = np.array(th0, float)
    if dummy is not None:  # indices of dummy streams: fix log mu
        for j in dummy:
            x0[j] = -30.0; bnds[j] = (-30.0, -30.0)
    if fixed:
        for j, v in fixed.items():
            x0[j] = v; bnds[j] = (v, v)
    fun = lambda x: float(f(jnp.asarray(x), U, C))
    jac = lambda x: np.asarray(g(jnp.asarray(x), U, C))
    r = minimize(fun, x0, jac=jac, method='L-BFGS-B', bounds=bnds,
                 options=dict(maxiter=3000, ftol=1e-13, gtol=1e-8))
    r.idx = idx
    r.hessian = lambda x: np.asarray(h(jnp.asarray(x), U, C))
    return r
