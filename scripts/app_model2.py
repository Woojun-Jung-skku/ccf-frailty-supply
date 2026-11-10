"""Application model, generalised for paper S v0.7.
Binary-onset likelihood of the common-cause frailty model (S-0 form: item dispersion at its Poisson limit, fixed at
alpha' = 1e4 where its estimate sits), with
 - an optional multiplicative calendar covariate on the stream intensities: mu_it = mu_i * exp(beta * H_t),
   applied to the origins in `scope` ('China' -> the China group only, 'all' -> every stream);
 - arbitrary event matrices (for supply-side-only onsets and for bootstrap replicates);
 - margins-only fits with the dispersions fixed.
Groups (origins) with >= MIN_STREAMS streams carry their own alpha_c; the other origins are independent streams.
"""
import os as _os
SP_ROOT = _os.environ.get('PAPER_S_ROOT') or _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json
sys.path.insert(0, f'{SP_ROOT}/scripts')
import numpy as np, pandas as pd
import jax, jax.numpy as jnp
from jax.scipy.special import logsumexp
from scipy.optimize import minimize, brentq
from scipy import stats
import ccffit2 as cf
jax.config.update('jax_enable_x64', True)

W = f'{SP_ROOT}/work/app'
AP_FIXED = 1e4
MIN_STREAMS = 5
CUT90 = float(stats.chi2.ppf(0.90, 1))      # 2.7055
CUT95 = float(stats.chi2.ppf(0.95, 1))
CUT965 = float(stats.chi2.ppf(0.965, 1))

E = pd.read_csv(f'{W}/stream_events.csv', parse_dates=['period'])
E['sid'] = E.code.astype(str) + '_' + E.origin
periods = sorted(E.period.unique())
T = len(periods)
origins = E.groupby('origin').sid.nunique().sort_values(ascending=False)
multi = [o for o in origins.index if origins[o] >= MIN_STREAMS]
MMAX = int(origins.max())


def group_matrix(o, E_=E):
    sub = E_[E_.origin == o]
    sids = sorted(sub.sid.unique())
    X = sub.pivot(index='period', columns='sid', values='event').reindex(periods)[sids].fillna(-1).astype(int).values
    return sids, X


GROUPS = {o: group_matrix(o) for o in origins.index}

# ---------------------------------------------------------------- calendar: Lunar New Year month (first day of the lunar year)
LNY = {2015: '2015-02-19', 2016: '2016-02-08', 2017: '2017-01-28', 2018: '2018-02-16', 2019: '2019-02-05',
       2020: '2020-01-25', 2021: '2021-02-12', 2022: '2022-02-01', 2023: '2023-01-22', 2024: '2024-02-10',
       2025: '2025-01-29'}
_pidx = pd.DatetimeIndex(periods)
H1 = np.zeros(T, int)        # the month that contains Lunar New Year's Day
for y, d in LNY.items():
    m = pd.Timestamp(d).to_period('M').start_time
    H1[_pidx.get_loc(m)] = 1
H2 = H1.copy()               # that month and the next
for t in np.where(H1 == 1)[0]:
    if t + 1 < T:
        H2[t + 1] = 1


# ---------------------------------------------------------------- likelihood pieces
def pack(X, m=MMAX):
    Xp = np.full((X.shape[0], m), -1, int); Xp[:, :X.shape[1]] = X
    u, c = np.unique(Xp, axis=0, return_counts=True)
    return jnp.asarray(u), jnp.asarray(c, dtype=jnp.float64)


def logq(log_mu, s, log_ap):
    a = jnp.exp(log_ap)
    return -a * jnp.log1p(jnp.exp(log_mu)[:, None] * jnp.exp(s)[None, :] / a)


def log1mexp(x):
    x = jnp.minimum(x, -1e-12)
    small = x > -0.693
    xs = jnp.where(small, x, -1.0)
    xl = jnp.where(small, -1.0, x)
    v = jnp.where(small, jnp.log(-jnp.expm1(xs)), jnp.log1p(-jnp.exp(xl)))
    return jnp.maximum(v, -700.0)


def ll_S1b(U, C, log_mu, log_ap, log_thc):
    ac = jnp.exp(-log_thc)
    s, logw = cf.eta_logw(ac)
    lq = logq(log_mu, s, log_ap)
    l1 = log1mexp(lq)
    x1 = (U == 1).astype(jnp.float64); x0 = (U == 0).astype(jnp.float64)
    lp = x0 @ lq + x1 @ l1
    return jnp.sum(C * logsumexp(lp + logw[None, :], axis=1))


def ll_INDb(U, C, log_mu, log_ap):
    lq = logq(log_mu, jnp.zeros(1), log_ap)[:, 0]
    x1 = (U == 1).astype(jnp.float64); x0 = (U == 0).astype(jnp.float64)
    return jnp.sum(C * (x0 @ lq + x1 @ log1mexp(lq)))


def ll_BFRb(U, C, log_mu, log_ap, lw, lp_):
    """binomial failure rate: a common shock with probability w per period hits each stream with probability p."""
    w = jax.nn.sigmoid(lw); p = jax.nn.sigmoid(lp_)
    lq = logq(log_mu, jnp.zeros(1), log_ap)[:, 0]
    lq_s = lq + jnp.log1p(-p)
    x1 = (U == 1).astype(jnp.float64); x0 = (U == 0).astype(jnp.float64)
    a0 = x0 @ lq + x1 @ log1mexp(lq)
    a1 = x0 @ lq_s + x1 @ log1mexp(lq_s)
    return jnp.sum(C * jnp.logaddexp(jnp.log1p(-w) + a0, jnp.log(w) + a1))


class Model2:
    """kind 'S1' (frailty per common group), 'BFR' (one shock process per common group) or 'IND'.
    groups: {origin: (sids, X)}. H: 0/1 vector over periods or None.
    scope: None, 'all', or an origin name (covariate applies to that origin's streams only)."""

    def __init__(self, kind, groups=GROUPS, H=None, scope=None, common=multi, ap_fixed=AP_FIXED):
        self.kind = kind; self.common = list(common) if kind != 'IND' else []
        self.order = list(origins.index)
        self.groups = groups
        self.nstream = {o: len(groups[o][0]) for o in self.order}
        self.H = None if H is None else np.asarray(H, int)
        self.scope = scope
        self.covar = {o: (self.H is not None and (scope == 'all' or scope == o)) for o in self.order}
        self.data = {}
        for o in self.order:
            X = groups[o][1]
            if self.covar[o]:
                self.data[o] = (pack(X[self.H == 0]), pack(X[self.H == 1]))
            else:
                self.data[o] = (pack(X),)
        k = 0; self.mu_slot = {}
        for o in self.order:
            self.mu_slot[o] = np.arange(k, k + MMAX); k += MMAX
        self.i_beta = None
        if any(self.covar.values()):
            self.i_beta = k; k += 1
        self.gp = {}
        for o in self.common:
            if kind == 'BFR':
                self.gp[o] = (k, k + 1); k += 2
            else:
                self.gp[o] = k; k += 1
        self.k = k
        self.log_ap = float(np.log(ap_fixed))
        data = self.data; covar = self.covar; mu_slot = self.mu_slot; gp = self.gp; i_beta = self.i_beta
        log_ap = self.log_ap; order = self.order

        def negll(th):
            tot = 0.0
            for o in order:
                lmu = th[mu_slot[o]]
                packs = data[o]
                if covar[o]:
                    sets = [(packs[0], lmu), (packs[1], lmu + th[i_beta])]
                else:
                    sets = [(packs[0], lmu)]
                for (U, C), lm in sets:
                    if o in gp and kind == 'S1':
                        tot += ll_S1b(U, C, lm, log_ap, th[gp[o]])
                    elif o in gp and kind == 'BFR':
                        tot += ll_BFRb(U, C, lm, log_ap, th[gp[o][0]], th[gp[o][1]])
                    else:
                        tot += ll_INDb(U, C, lm, log_ap)
            return -tot
        self.f = jax.jit(negll); self.g = jax.jit(jax.grad(negll))

    def start(self, th_c=0.5, beta=0.0, w=0.05, p=0.5):
        x = np.zeros(self.k); b = []
        for o in self.order:
            X = self.groups[o][1]
            for j in range(MMAX):
                if j < X.shape[1]:
                    col = X[:, j]; act = col >= 0
                    pr = max(col[act].mean(), 0.3 / max(act.sum(), 1))
                    x[self.mu_slot[o][j]] = np.log(-np.log1p(-min(pr, 0.95))); b.append((-12, 3))
                else:
                    x[self.mu_slot[o][j]] = -30.0; b.append((-30.0, -30.0))
        if self.i_beta is not None:
            x[self.i_beta] = beta; b.append((-6.0, 6.0))
        for o in self.common:
            if self.kind == 'BFR':
                x[self.gp[o][0]] = np.log(w / (1 - w)); x[self.gp[o][1]] = np.log(p / (1 - p)); b += [(-14, 6), (-10, 10)]
            else:
                x[self.gp[o]] = np.log(th_c); b.append((cf.LOG_TH_MIN, cf.LOG_TH_MAX))
        return x, b

    def fit(self, x0=None, fixed=None, starts=None):
        best = None
        cand = [x0] if x0 is not None else (starts or [None])
        for st in cand:
            if isinstance(st, np.ndarray):
                x, b = self.start(); x = st.copy()
            elif isinstance(st, dict):
                x, b = self.start(**st)
            else:
                x, b = self.start()
            if fixed:
                for j, v in fixed.items():
                    x[j] = v; b[j] = (v, v)
            r = minimize(lambda z: float(self.f(jnp.asarray(z))), x, jac=lambda z: np.asarray(self.g(jnp.asarray(z))),
                         method='L-BFGS-B', bounds=b, options=dict(maxiter=5000, ftol=1e-14, gtol=1e-9))
            if best is None or r.fun < best.fun:
                best = r
        self.res = best
        return best

    def mu(self, x, o):
        return np.exp(x[self.mu_slot[o][:self.nstream[o]]])

    def alpha_c(self, x, o):
        return float(np.exp(-x[self.gp[o]]))

    def npar(self):
        return self.k - sum(MMAX - self.nstream[o] for o in self.order)


S1_STARTS = [dict(th_c=t) for t in (0.02, 0.3, 1.5)]
BFR_STARTS = [dict(w=w, p=p) for w in (0.01, 0.05, 0.2) for p in (0.15, 0.5, 0.9)]


def profile_interval(model, j, x_hat, nll_hat, cut=CUT90, lo=None, hi=None, grid=None):
    """profile-likelihood interval for parameter j; returns (lo, hi, curve) on the parameter's own scale (log theta or beta).
    lo/hi: bounds of the parameter. Boundary values are returned when the cut-off is not reached."""
    b = model.start()[1][j]
    lo = b[0] if lo is None else lo; hi = b[1] if hi is None else hi
    def pfun(v):
        return 2 * (model.fit(x0=x_hat, fixed={j: v}).fun - nll_hat)
    f = lambda v: pfun(v) - cut
    v_hat = x_hat[j]
    L = lo if f(lo) <= 0 else brentq(f, lo, v_hat, xtol=1e-3)
    U = hi if f(hi) <= 0 else brentq(f, v_hat, hi, xtol=1e-3)
    curve = None
    if grid is not None:
        curve = [(float(v), float(pfun(v))) for v in grid]
    return float(L), float(U), curve


def summarise_S1(model, x, nll, nll_ind, prof=True, cut=CUT90):
    """per-origin alpha_c estimates, intervals and LR against independence (same covariate)."""
    out = {}
    for o in model.common:
        j = model.gp[o]
        d = dict(ac_hat=model.alpha_c(x, o))
        if prof:
            L, U, _ = profile_interval(model, j, x, nll, cut=cut)
            d['ac_lo'] = float(np.exp(-U)); d['ac_hi'] = float(np.exp(-L)) if L > cf.LOG_TH_MIN + 1e-6 else float('inf')
            # LR of the frailty for this origin alone: fix this origin at the boundary, others free
            d['lr_boundary'] = float(2 * (model.fit(x0=x, fixed={j: cf.LOG_TH_MIN}).fun - nll))
        out[o] = d
    out['lr_all'] = float(2 * (nll_ind - nll)) if nll_ind is not None else None
    return out
