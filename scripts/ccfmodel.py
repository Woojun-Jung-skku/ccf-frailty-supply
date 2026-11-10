"""Shared-frailty common-cause model for multi-stream supply systems: numerical core.

Model (per period, one common-cause group g with streams i in g):
    N_i | lam_i ~ Poisson(lam_i),  lam_i = mu_i * xi_i * eta,
    xi_i ~ Gamma(ap, ap)  (item frailty; ap = alpha'),
    eta  ~ Gamma(ac, ac)  (common-cause frailty; ac = alpha_c),
all independent; groups independent; periods independent.
ap = inf -> S-0 (pure common cause, negative multinomial); ac = inf -> independence.
"""
import numpy as np
from scipy import special, stats, integrate
from functools import lru_cache
from itertools import combinations

INF = np.inf


# ---------------------------------------------------------------- quadrature
@lru_cache(maxsize=256)
def _genlag(n, a_key):
    a = float(a_key)
    x, w = special.roots_genlaguerre(n, a - 1.0)
    eta = x / a
    w = w / special.gamma(a) if a < 150 else np.exp(np.log(w) - special.gammaln(a))
    return eta, w


def eta_nodes(ac, n=120):
    """Nodes/weights such that E[f(eta)] ~= sum w f(eta), eta ~ Gamma(ac, ac).
    Generalized Gauss-Laguerre for moderate ac; Gauss-Hermite on log-scale for large ac."""
    if not np.isfinite(ac):
        return np.array([1.0]), np.array([1.0])
    if ac < 0.25:
        # log-scale trapezoid on the exact log-gamma density (stable for small shapes)
        z = np.arange(-30.0, 8.0 + 1e-9, 0.02)
        sd = np.sqrt(special.polygamma(1, ac))
        s = sd * z
        logp = ac * np.log(ac) - special.gammaln(ac) + ac * s - ac * np.exp(s)
        w = np.exp(logp) * sd * 0.02
        return np.exp(s), w / w.sum()
    if ac <= 60:
        eta, w = _genlag(n, round(float(ac), 12))
        return eta, w
    # large shape: eta approx lognormal-ish; use Gauss-Hermite in log space on the exact density
    z, wz = np.polynomial.hermite_e.hermegauss(n)
    sd = np.sqrt(special.polygamma(1, ac))
    s = sd * z  # log eta, mode ~ 0
    logp = ac * np.log(ac) - special.gammaln(ac) + ac * s - ac * np.exp(s)
    logphi = -0.5 * z**2 - 0.5 * np.log(2 * np.pi)
    w = wz / np.sqrt(2 * np.pi) * np.exp(logp + np.log(sd) - logphi)
    return np.exp(s), w


def E_eta(f, ac, n=120):
    eta, w = eta_nodes(ac, n)
    return np.sum(w * f(eta), axis=-1)


# ------------------------------------------------- conditional reliabilities
def q_cond(mu, eta, ap):
    """P(N=0 | eta) for stream with mean mu, item dispersion ap (integrated over xi)."""
    mu = np.asarray(mu, float)[..., None]
    eta = np.asarray(eta, float)
    if not np.isfinite(ap):
        return np.exp(-mu * eta)
    return (1.0 + mu * eta / ap) ** (-ap)


def lt_gamma(s, ac):
    """Laplace transform of Gamma(ac, ac): E exp(-s eta)."""
    s = np.asarray(s, float)
    if not np.isfinite(ac):
        return np.exp(-s)
    return (1.0 + s / ac) ** (-ac)


def sri_marginal(mu, ap, ac, n=120):
    """Marginal stream reliability P(N_i = 0)."""
    return E_eta(lambda e: q_cond(mu, e, ap), ac, n)


# ------------------------------------------------------------ system level
def series_group(mu, ap, ac, n=120):
    """P(all streams of one group deliver)."""
    mu = np.asarray(mu, float)
    if not np.isfinite(ap):
        return float(lt_gamma(mu.sum(), ac))
    return float(E_eta(lambda e: np.prod(q_cond(mu, e, ap), axis=0), ac, n))


def allfail_group(mu, ap, ac, n=120):
    """P(all streams of one group interrupted) (parallel unreliability)."""
    mu = np.asarray(mu, float)
    if not np.isfinite(ap):
        tot = 0.0
        m = len(mu)
        for k in range(m + 1):
            for B in combinations(range(m), k):
                tot += (-1) ** k * float(lt_gamma(mu[list(B)].sum(), ac))
        return tot
    return float(E_eta(lambda e: np.prod(1 - q_cond(mu, e, ap), axis=0), ac, n))


def working_set_probs(mu, ap, ac, n=120):
    """Probabilities of every working set W of a group (dict: frozenset -> prob)."""
    mu = np.asarray(mu, float)
    m = len(mu)
    eta, w = eta_nodes(ac, n)
    q = q_cond(mu, eta, ap)  # m x nodes
    out = {}
    for k in range(m + 1):
        for W in combinations(range(m), k):
            mask = np.zeros(m, bool)
            mask[list(W)] = True
            val = np.prod(np.where(mask[:, None], q, 1 - q), axis=0)
            out[frozenset(W)] = float(np.sum(w * val))
    return out


def system_reliability(structure, groups, mu, ap, ac, n=120):
    """Exact one-period reliability of a coherent system under shared frailty.
    structure(working_set:set of stream ids)->bool; groups: list of lists of stream ids;
    mu: dict/array stream->mean; ac: scalar or per-group list."""
    acs = ac if np.ndim(ac) else [ac] * len(groups)
    per = []
    for g, ids in enumerate(groups):
        p = working_set_probs([mu[i] for i in ids], ap, acs[g], n)
        per.append([(frozenset(ids[j] for j in W), pr) for W, pr in p.items()])
    # combine (exact enumeration; fine for small systems)
    tot = 0.0

    def rec(g, cur, pr):
        nonlocal tot
        if pr == 0.0:
            return
        if g == len(per):
            if structure(cur):
                tot += pr
            return
        for W, p in per[g]:
            rec(g + 1, cur | W, pr * p)

    rec(0, frozenset(), 1.0)
    return tot


# --------------------------------------------------------- multiplicity / CCF
def multiplicity_pmf(m, mu, ap, ac, n=160):
    """P(K = k), k = 0..m: number of interrupted streams among m exchangeable streams."""
    ks = np.arange(m + 1)
    if not np.isfinite(ap):
        out = []
        for k in ks:
            s = 0.0
            for j in range(k + 1):
                s += (-1) ** j * special.comb(k, j) * float(lt_gamma(mu * (m - k + j), ac))
            out.append(special.comb(m, k) * s)
        return np.array(out)
    eta, w = eta_nodes(ac, n)
    q = q_cond(np.array([mu]), eta, ap)[0]
    return np.array([special.comb(m, k) * np.sum(w * (1 - q) ** k * q ** (m - k)) for k in ks])


def alpha_factors(pmf):
    pmf = np.asarray(pmf)
    return pmf[1:] / pmf[1:].sum()


def beta_joint(mu1, mu2, ap, ac, n=160):
    """Common-cause fraction of joint interruptions: (P12 - P1 P2)/P12, and P12/(P1P2)."""
    p1 = 1 - sri_marginal(mu1, ap, ac, n)
    p2 = 1 - sri_marginal(mu2, ap, ac, n)
    p12 = allfail_group([mu1, mu2], ap, ac, n)
    return (p12 - p1 * p2) / p12, p12 / (p1 * p2), p12, p1, p2


# ------------------------------------------------------------- information
def fisher_boundary(mu):
    """Expected Fisher information for (theta', theta_c) at theta=0, one group-period."""
    mu = np.asarray(mu, float)
    A = 0.5 * np.sum(mu ** 2)
    M = mu.sum()
    I = np.array([[A, A], [A, 0.5 * M ** 2]])
    schur_c = I[1, 1] - I[0, 1] ** 2 / I[0, 0]
    schur_p = I[0, 0] - I[0, 1] ** 2 / I[1, 1]
    return I, schur_c, schur_p


# ------------------------------------------------------------- simulation
def simulate(mu_groups, ap, ac, T, rng):
    """Simulate counts: returns list of arrays (T x m_g) per group."""
    out = []
    acs = ac if np.ndim(ac) else [ac] * len(mu_groups)
    for g, mu in enumerate(mu_groups):
        mu = np.asarray(mu, float)
        m = len(mu)
        a = acs[g]
        eta = rng.gamma(a, 1 / a, size=(T, 1)) if np.isfinite(a) else np.ones((T, 1))
        xi = rng.gamma(ap, 1 / ap, size=(T, m)) if np.isfinite(ap) else np.ones((T, m))
        out.append(rng.poisson(mu[None, :] * xi * eta))
    return out


# ======================================================================================
# Importance decomposition (paper S, Proposition 3): R = h(p) + sum_{|A|>=2} d^A h(p) * kappa_A,
# kappa_A = E prod_{i in A} (q_i(eta) - p_i) within a group (co-moments factorise across groups).
# ======================================================================================
import itertools as _it


def margins_and_pair_cov(mu, ap, ac, n=160):
    """margins p_i = E q_i(eta) and pairwise covariances c_ij = Cov(q_i, q_j) of the conditional reliabilities
    of the streams of ONE group. mu: 1-D array. Returns (p, C) with C symmetric, zero diagonal."""
    mu = np.asarray(mu, float)
    eta, w = eta_nodes(ac, n)
    q = q_cond(mu, eta, ap)                              # m x n
    p = q @ w
    d = q - p[:, None]
    C = (d * w) @ d.T
    np.fill_diagonal(C, 0.0)
    return p, C


def comoment(mu, ap, ac, idx, n=160):
    """kappa_A for the subset idx (positions in mu) of one group."""
    mu = np.asarray(mu, float)
    eta, w = eta_nodes(ac, n)
    q = q_cond(mu, eta, ap)
    p = q @ w
    prod = np.ones_like(eta)
    for i in idx:
        prod = prod * (q[i] - p[i])
    return float(np.sum(w * prod))


class ReliabilityPolynomial:
    """h(p) = E phi(x) for independent x_i ~ Bernoulli(p_i). Two evaluators:
    - 'enumerate': structure(working_set) -> bool, exact enumeration over 2^I states (small systems);
    - 'inputs': a product that is 'up' when fewer than k_fail inputs are disrupted, input disrupted when the
      interrupted streams among its sources carry more than `thr` of its share (the application's structure)."""

    def __init__(self, ids, structure=None, inputs=None, shares=None, k_fail=2, thr=0.5):
        self.ids = list(ids)
        self.structure = structure
        self.inputs = inputs          # dict input -> list of stream ids
        self.shares = shares          # dict stream id -> share of its input
        self.k_fail, self.thr = k_fail, thr

    def __call__(self, p):
        if self.structure is not None:
            tot = 0.0
            for bits in _it.product([0, 1], repeat=len(self.ids)):
                W = frozenset(i for i, b in zip(self.ids, bits) if b)
                if self.structure(W):
                    pr = 1.0
                    for i, b in zip(self.ids, bits):
                        pr *= p[i] if b else 1 - p[i]
                    tot += pr
            return tot
        # inputs: P(input disrupted) by enumeration over its sources, then a Poisson-binomial over inputs
        pd_list = []
        for inp, srcs in self.inputs.items():
            pdis = 0.0
            for bits in _it.product([0, 1], repeat=len(srcs)):       # bit 1 = interrupted
                w = sum(self.shares[s] for s, b in zip(srcs, bits) if b)
                if w > self.thr:
                    pr = 1.0
                    for s, b in zip(srcs, bits):
                        pr *= (1 - p[s]) if b else p[s]
                    pdis += pr
            pd_list.append(pdis)
        # distribution of the number of disrupted inputs
        dist = np.zeros(len(pd_list) + 1); dist[0] = 1.0
        for pdis in pd_list:
            dist = np.concatenate([dist * (1 - pdis), [0.0]]) + np.concatenate([[0.0], dist * pdis])
        return float(dist[:self.k_fail].sum())

    def mixed_partial(self, p, A):
        """d^{|A|} h / prod_{i in A} dp_i at p (multilinear: inclusion-exclusion over 0/1 settings of A)."""
        tot = 0.0
        for bits in _it.product([0, 1], repeat=len(A)):
            q = dict(p)
            for i, b in zip(A, bits):
                q[i] = float(b)
            tot += (-1) ** (len(A) - sum(bits)) * self(q)
        return tot

    def jri(self, p, i, j):
        return self.mixed_partial(p, [i, j])


def importance_pairwise(h, p, groups, covs):
    """first-order (pairwise) common-cause terms. groups: list of lists of ids; covs: list of matrices c_ij
    aligned with groups. Returns dict(total, per_group, per_pair, priority) where priority[i] is the first-order
    gain in h from moving stream i to a group of its own (margins fixed)."""
    per_group, per_pair, pri = [], {}, {}
    for g, C in zip(groups, covs):
        s = 0.0
        for a, b in _it.combinations(range(len(g)), 2):
            i, j = g[a], g[b]
            term = h.jri(p, i, j) * C[a, b]
            per_pair[(i, j)] = term
            s += term
        per_group.append(s)
        for a, i in enumerate(g):
            pri[i] = -sum(per_pair[tuple(sorted((i, j), key=lambda z: g.index(z)))] for j in g if j != i)
    return dict(total=float(sum(per_group)), per_group=per_group, per_pair=per_pair, priority=pri)
