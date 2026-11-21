"""
plot_chains.py -- MCMC diagnostics and posterior plots for the closure-model chains.

    python plot_chains.py [chain_prefix]  # everything, saved to chains/plots/ (CPL chains are detected by their w0 column)
    python plot_chains.py traces acf      # only the named parts (see PARTS below)

Parts
-----
traces       trace plots (+ -log posterior), burn-in marked
acf          autocorrelation functions, integrated autocorrelation time (IAT), ESS, split R-hat
iat          IAT estimate vs chain length, with the N/50 reliability line
marginals    1-D posteriors with prior ranges and the LCDM-limit values marked; prior pile-up check
degeneracy   2-D posteriors of the key correlations (H0 vs f_X, omega_m, r_d; q0 vs j0, H0)
chi2         chi2 per likelihood: trace and histogram
functions    q(z), j(z), w_DE(z), Omega_m(z), H(z)/(1+z), rho_DE(z); and f_X(z), Omega_m(z), H/H_LCDM to z ~ 1100
residuals    best-sample DESI pulls, binned SN residuals, Planck pulls (needs the data files)
corner       getdist triangle plot (needs getdist)
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import closure_cosmo as cs
from constants import MODEL, WONG_PALETTE, RD_PLANCK
from chain_utils import (is_fixed_noplanck, read_chains, expand, mean_acf, autocorr_function,
                         integrated_autocorr_time, split_rhat, sn_dataset)
from load_data import load_desi, load_sn, planck_distance_priors
from chain_utils import run_prefix, infer_model, apply_run_settings
from calculate_distances import luminosity_distance
from bao import bao_model_vector

# =========================================================
# Configuration
# =========================================================
FILE_NAME = run_prefix('pantheon', MODEL)      # overridden by the chain prefix given on the command line
YAML_DIR = 'runs'                             # make_runs.py writes <prefix>.yaml here
YAML_FILE = None                              # None -> YAML_DIR/<chain prefix>.yaml; or give a yaml as 2nd command-line argument
OUT_DIR = 'chains/plots'
BURN = 0.3                                    # fraction of each chain discarded as burn-in
SAMPLED = ["H0", "Om0", "ombh2", "q0", "j0"]  # edit for fixed-parameter runs
DERIVED = ["rd", "fX_star", "wm"]       # 'wm' = Om0 h^2 is added below; others come from the chain
Z_LOW_MAX = 4.0                               # upper redshift of the low-z function plots
Z_DATA_MAX = 2.33                             # highest redshift with SN/BAO data (DESI Lya); beyond it nothing constrains H(z)
N_POST = 400                                  # posterior draws for the redshift-dependent bands
LCDM_REF = dict(H0=67.4, Om0=0.315, j0=1.0, ombh2=0.02237)   # LCDM-limit point
USE_TEX = False                               # True: use the usetex options from constants.py (needs LaTeX)

LABELS = {"H0": r"$H_0$", "Om0": r"$\Omega_{m0}$", "j0": r"$j_0$",
          "ombh2": r"$\omega_b$", "w0": r"$w_0$", "wa": r"$w_a$", "q0": r"$q_0$", "rd": r"$r_d$ [Mpc]",
          "fX_star": r"$f_X(z_*)$", "wm": r"$\omega_m$"}
LEGACY_CPL = False                            # set in main(): chain stores w0, wa under the names q0, j0
PARTS = ("traces", "acf", "iat", "marginals", "degeneracy", "chi2", "functions", "residuals", "corner")

if USE_TEX:
    from constants import PLOT_TEXT_OPTIONS
    plt.rcParams.update(PLOT_TEXT_OPTIONS)
else:
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})


def lab(p):
    return LABELS.get(p, p)


def save(fig, name, tag=None):
    """Save to OUT_DIR/<name>_<tag>.pdf; tag defaults to the chain prefix so datasets/models never overwrite each other."""
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}_{tag or FILE_NAME}.pdf")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def add_derived(df):
    df = df.copy()
    df["wm"] = df["Om0"] * (df["H0"] / 100.0) ** 2
    return df


def available(df, names):
    return [n for n in dict.fromkeys(names) if n in df.columns]


def lcdm_value(p):
    """Value of parameter p at the LCDM-limit point (None if undefined)."""
    r = LCDM_REF
    return {"H0": r["H0"], "Om0": r["Om0"], "j0": 1.0, "ombh2": r["ombh2"],
            "q0": 1.5 * r["Om0"] - 1.0, "w0": -1.0, "wa": 0.0, "fX_star": 0.0, "wm": r["Om0"] * (r["H0"] / 100) ** 2,
            "rd": RD_PLANCK}.get(p)


def read_priors():
    try:
        import yaml
        path = YAML_FILE or os.path.join(YAML_DIR, f"{FILE_NAME}.yaml")
        with open(path) as f:
            params = yaml.safe_load(f)["params"]
        return {k: (v["prior"]["min"], v["prior"]["max"]) for k, v in params.items()
                if isinstance(v, dict) and isinstance(v.get("prior"), dict) and "min" in v["prior"]}
    except Exception as e:                                  # noqa
        print("could not read priors from yaml:", repr(e))
        return {}


def wquantiles(x, w, qs):
    return np.quantile(x, qs, method="inverted_cdf", weights=w)


# =========================================================
# a) Trace plots
# =========================================================
def plot_traces(raw):
    cols = available(raw[0], SAMPLED + ["minuslogpost"])
    fig, axes = plt.subplots(len(cols), 1, figsize=(10, 1.8 * len(cols)), sharex=True)
    for ax, c in zip(np.atleast_1d(axes), cols):
        for i, ch in enumerate(raw):
            ax.plot(np.cumsum(ch["weight"].values), ch[c].values, lw=0.5,
                    color=WONG_PALETTE[i % len(WONG_PALETTE)], label=f"chain {i + 1}")
        ax.set_ylabel(lab(c) if c != "minuslogpost" else r"$-\ln P$")
        ax.axvline(BURN * max(ch["weight"].sum() for ch in raw), color="k", ls="--", lw=0.8)
        if c == "minuslogpost":                              # zoom on the converged part of the posterior
            lo = min(ch[c].quantile(0.001) for ch in raw)
            ax.set_ylim(lo - 1, lo + 30)
    axes[0].legend(ncol=len(raw), fontsize=8, loc="upper right")
    axes[-1].set_xlabel("sampler step (cumulative weight)   [dashed: end of burn-in]")
    save(fig, "traces")


# =========================================================
# b) Autocorrelation, IAT, ESS, R-hat
# =========================================================
def _sequences(chains, p):
    return [expand(ch, p) for ch in chains]


def convergence_table(chains, params):
    rows = []
    for p in params:
        seqs = _sequences(chains, p)
        acf = mean_acf(seqs)
        tau = integrated_autocorr_time(acf)
        N = sum(len(s) for s in seqs)
        rows.append(dict(param=p, N_steps=N, IAT=tau, ESS=N / tau, N_over_50tau=N / (50 * tau),
                         split_Rhat=split_rhat(seqs)))
    tab = pd.DataFrame(rows).set_index("param")
    print("\nCONVERGENCE (IAT in sampler steps; want R-hat < 1.01 and N/(50 tau) > 1)")
    print(tab.round(3).to_string())
    os.makedirs(OUT_DIR, exist_ok=True)
    tab.to_csv(os.path.join(OUT_DIR, f"convergence_table_{FILE_NAME}.csv"))
    return tab


def plot_acf(chains, params, tab):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    lag_max = int(min(min(len(s) for s in _sequences(chains, params[0])) - 1,
                      max(20, 8 * tab["IAT"].max())))
    for i, p in enumerate(params):
        acf = mean_acf(_sequences(chains, p))[:lag_max]
        axes[0].plot(acf, color=WONG_PALETTE[i % len(WONG_PALETTE)], label=f"{lab(p)}  ($\\tau$={tab.loc[p, 'IAT']:.0f})")
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_xlabel("lag (sampler steps)")
    axes[0].set_ylabel("autocorrelation")
    axes[0].legend(fontsize=8)
    axes[0].set_title("Autocorrelation function (averaged over chains)")
    axes[1].bar([lab(p) for p in params], tab["IAT"], color=WONG_PALETTE[:len(params)])
    axes[1].set_ylabel("integrated autocorrelation time")
    axes[1].set_title("IAT (Sokal window, c = 5)")
    save(fig, "autocorrelation")


def plot_iat_convergence(chains, params):
    n_min = min(len(s) for s in _sequences(chains, params[0]))
    Ns = np.unique(np.geomspace(max(200, n_min // 100), n_min, 12).astype(int))
    fig, axes = plt.subplots(1, len(params), figsize=(3.2 * len(params), 3.2), sharex=True)
    for ax, p in zip(np.atleast_1d(axes), params):
        seqs = _sequences(chains, p)
        taus = [integrated_autocorr_time(mean_acf([s[:n] for s in seqs])) for n in Ns]
        ax.loglog(Ns, taus, "o-", color=WONG_PALETTE[0])
        ax.loglog(Ns, Ns / 50.0, "k--", lw=1, label=r"$N/50$")
        ax.set_title(lab(p))
        ax.set_xlabel("N steps (per chain)")
    np.atleast_1d(axes)[0].set_ylabel(r"$\hat\tau$")
    np.atleast_1d(axes)[0].legend()
    fig.suptitle(r"IAT estimate is trustworthy only where it lies below the dashed $N/50$ line", y=1.03)
    save(fig, "iat_vs_length")


# =========================================================
# Marginals, degeneracies, chi2
# =========================================================
def plot_marginals(df):
    priors = read_priors()
    w = df["weight"].values
    params = available(df, SAMPLED + DERIVED)
    ncol = 4
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.8 * nrow))
    axes = np.atleast_1d(axes).ravel()
    print("\nPRIOR PILE-UP CHECK (weight within 3% of the prior range of an edge)")
    for ax, p in zip(axes, params):
        ax.hist(df[p].values, bins=60, weights=w, density=True, color=WONG_PALETTE[5], alpha=0.8)
        lo, med, hi = wquantiles(df[p].values, w, [0.16, 0.5, 0.84])
        ax.axvline(med, color="k", lw=1)
        ax.axvspan(lo, hi, color="k", alpha=0.12)
        ref = lcdm_value(p)
        if ref is not None:
            ax.axvline(ref, color=WONG_PALETTE[1], ls="--", lw=1.2)
        if p in priors:
            a, b = priors[p]
            ax.axvline(a, color="r", lw=0.8)
            ax.axvline(b, color="r", lw=0.8)
            span = b - a
            near = w[(df[p] < a + 0.03 * span) | (df[p] > b - 0.03 * span)].sum() / w.sum()
            if near > 0.02:
                print(f"  WARNING {p}: {100 * near:.1f}% of the posterior sits at a prior edge -> widen the prior")
            mean = np.average(df[p].values, weights=w)
            sd = np.sqrt(np.average((df[p].values - mean) ** 2, weights=w))
            if sd > 0.8 * span / np.sqrt(12.0):                  # a flat prior has sd = span / sqrt(12)
                print(f"  WARNING {p}: posterior sd {sd:.3g} is {100 * sd / (span / np.sqrt(12.0)):.0f}% of the flat-prior sd -> "
                      f"the data do not constrain {p} (posterior = prior)")
        pmin, pmax = float(df[p].min()), float(df[p].max())
        pad = 0.1 * (pmax - pmin if pmax > pmin else 1.0)
        ax.set_xlim(pmin - pad, pmax + pad)                       # wide priors must not squash the posterior
        ax.set_title(f"{lab(p)} = {med:.4g}$^{{+{hi - med:.2g}}}_{{-{med - lo:.2g}}}$", fontsize=10)
        ax.set_yticks([])
    for ax in axes[len(params):]:
        ax.axis("off")
    fig.suptitle("1-D posteriors (grey: 68%; orange dashed: LCDM-limit value; red: prior range)", y=1.02)
    fig.tight_layout()
    save(fig, "marginals")


def plot_degeneracies(df):
    pairs = [("fX_star", "H0"), ("wm", "H0"), ("rd", "H0"), ("q0", "j0"), ("j0", "H0"), ("q0", "H0"),
             ("w0", "wa"), ("w0", "H0"), ("wa", "H0")]
    pairs = [(a, b) for a, b in pairs if a in df.columns and b in df.columns and df[a].std() > 0 and df[b].std() > 0]
    pairs = pairs[:6] if MODEL != "cpl" else [pr for pr in pairs if "w0" in pr or "wa" in pr or pr[0] in ("fX_star", "wm", "rd")][:6]
    ncol = 3
    nrow = int(np.ceil(len(pairs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.6 * nrow))
    axes = np.atleast_1d(axes).ravel()
    w = df["weight"].values
    for ax, (a, b) in zip(axes, pairs):
        hb = ax.hexbin(df[a], df[b], C=w, reduce_C_function=np.sum, gridsize=45, cmap="viridis", mincnt=1)
        ra, rb = lcdm_value(a), lcdm_value(b)
        if ra is not None:
            ax.axvline(ra, color="w", ls="--", lw=0.8)
        if rb is not None:
            ax.axhline(rb, color="w", ls="--", lw=0.8)
        rho = np.corrcoef(df[a], df[b])[0, 1]
        ax.set_xlabel(lab(a)); ax.set_ylabel(lab(b)); ax.set_title(f"corr = {rho:+.2f}", fontsize=10)
    for ax in axes[len(pairs):]:
        ax.axis("off")
    fig.suptitle("Key degeneracies (white dashed: LCDM-limit point)", y=1.01)
    fig.tight_layout()
    save(fig, "degeneracies")


def plot_chi2(chains):
    cols = [c for c in chains[0].columns if c.startswith("chi2__")] + ["chi2"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    df = pd.concat(chains, ignore_index=True)
    for i, c in enumerate(cols):
        name = c.split(".")[-1].replace("Likelihood", "").replace("CPL", "") if c != "chi2" else "total"
        for ch in chains:
            axes[0].plot(np.cumsum(ch["weight"].values), ch[c].values, lw=0.3, alpha=0.6,
                         color=WONG_PALETTE[i % len(WONG_PALETTE)], label=name if ch is chains[0] else None)
        axes[1].hist(df[c], bins=60, weights=df["weight"], histtype="step", color=WONG_PALETTE[i % len(WONG_PALETTE)],
                     label=f"{name}: min {df[c].min():.1f}")
    axes[0].set_yscale("log"); axes[0].set_xlabel("sampler step"); axes[0].set_ylabel(r"$\chi^2$"); axes[0].legend(fontsize=8)
    axes[1].set_xscale("log"); axes[1].set_xlabel(r"$\chi^2$"); axes[1].legend(fontsize=8)
    save(fig, "chi2_components")


# =========================================================
# c) Functions of redshift with uncertainty bands
# =========================================================
def _d1_d2(L, h):
    """5-point first and second derivatives of L sampled at x + h*(-2..2)."""
    d1 = (L[0] - 8 * L[1] + 8 * L[3] - L[4]) / (12 * h)
    d2 = (-L[0] + 16 * L[1] - 30 * L[2] + 16 * L[3] - L[4]) / (12 * h * h)
    return d1, d2


def curves(theta, model, z, h=1e-3):
    """
    H(z), H(z)/(1+z), q(z), j(z), Omega_m(z), w_DE(z), rho_DE(z)/rho_DE(0) from the closure H(z) (no radiation).
      NB  d[H/(1+z)]/dz = H q/(1+z)^2, so H/(1+z) = da/dt has its MINIMUM exactly where q = 0 (the
      deceleration -> acceleration transition, z_t = 0.63 in LCDM).  H(z) itself is monotonic unless rho_DE falls fast.
      q = 1/2 dlnE^2/dx - 1,  j = q + 2q^2 + 1/2 d^2lnE^2/dx^2,  x = ln(1+z)
      effective DE: rho_DE = E^2 - Om0 (1+z)^3,  w = -1 + 1/3 dln(rho_DE)/dx   (nan where rho_DE <= 0)
    """
    H0, q0, j0, Om0 = theta
    x = np.log1p(np.asarray(z, dtype=float))
    xs = x[None, :] + h * np.arange(-2, 3)[:, None]
    Hs = []
    for xx in xs:
        r = cs.background_model(theta, np.expm1(xx), model)
        if r is None:
            return None
        Hs.append(r[0])
    E2 = (np.array(Hs) / H0) ** 2
    d1, d2 = _d1_d2(np.log(E2), h)
    q = 0.5 * d1 - 1.0
    j = q + 2 * q ** 2 + 0.5 * d2
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = E2 - Om0 * np.exp(3 * xs)
        dr, _ = _d1_d2(np.log(rho), h)
    return dict(Hz=np.sqrt(E2[2]) * H0, H=np.sqrt(E2[2]) * H0 / (1 + z), q=q, j=j, Om=Om0 * np.exp(3 * x) / E2[2],
                w=-1 + dr / 3.0, rho=rho[2] / (1 - Om0))


def _theta_from_row(r):
    """[H0, p1, p2, Om0]: (q0, j0) for the closure models, (w0, wa) for CPL."""
    if MODEL == "cpl":
        return [r["H0"], r["w0"], r["wa"], r["Om0"]]
    return [r["H0"], r["q0"], r["j0"], r["Om0"]]


def _theta_lcdm():
    """The LCDM point in the parametrisation of the current model."""
    if MODEL == "cpl":
        return [LCDM_REF["H0"], -1.0, 0.0, LCDM_REF["Om0"]]
    return [LCDM_REF["H0"], 1.5 * LCDM_REF["Om0"] - 1.0, 1.0, LCDM_REF["Om0"]]


def _bands(samples, ax, z, color, label=None):
    arr = np.array(samples)
    lo95, lo68, med, hi68, hi95 = np.nanpercentile(arr, [2.5, 16, 50, 84, 97.5], axis=0)
    ax.fill_between(z, lo95, hi95, color=color, alpha=0.18, lw=0)
    ax.fill_between(z, lo68, hi68, color=color, alpha=0.38, lw=0)
    ax.plot(z, med, color=color, lw=1.8, label=label)


def plot_functions(df, rng=np.random.default_rng(0)):
    w = df["weight"].values
    idx = rng.choice(len(df), size=N_POST, p=w / w.sum())
    rows = [df.iloc[i] for i in idx]
    th_ref = _theta_lcdm()

    # ---- low redshift
    z = np.linspace(0.0, Z_LOW_MAX, 151)
    store = {k: [] for k in ("Hz", "H", "q", "j", "Om", "w", "rho")}
    for r in rows:
        c = curves(_theta_from_row(r), MODEL, z)
        if c is not None:
            for k in store:
                store[k].append(c[k])
    ref = curves(th_ref, MODEL, z)
    nbad_w = sum(np.isnan(s).any() for s in store["w"])
    if nbad_w:
        print(f"\nNOTE: rho_DE <= 0 somewhere in z<{z.max():.0f} for {nbad_w}/{len(store['w'])} draws -> w_DE undefined there (nan, ignored in bands)")
    # ---- physical bound: in a flat universe Omega_m(z) <= 1  <=>  effective dark density rho_DE >= 0
    Om_arr = np.array(store["Om"])
    over = (Om_arr > 1 + 1e-9).any(axis=1)
    zc = [z[np.argmax(o > 1 + 1e-9)] for o in Om_arr[over]]
    print(f"\nOmega_m(z) > 1 somewhere in z<{Z_LOW_MAX:g} for {over.sum()}/{len(over)} draws"
          + (f" (first crossing: median z = {np.median(zc):.2f}); i.e. rho_DE < 0 there" if over.any() else ""))
    titles = dict(Hz=r"$H(z)$ [km/s/Mpc]", H=r"$H(z)/(1+z)=\dot a$  (minimum at $q=0$)", q=r"$q(z)$", j=r"$j(z)$",
                  Om=r"$\Omega_m(z)$", w=r"$w_{\rm DE}(z)$", rho=r"$\rho_{\rm DE}(z)/\rho_{\rm DE}(0)$")
    fig, axes = plt.subplots(2, 4, figsize=(17, 7))
    keys = ["Hz", "H", "q", "j", "Om", "w", "rho"]
    for ax, k in zip(axes.ravel()[:7], keys):
        _bands(store[k], ax, z, WONG_PALETTE[5], label="posterior (68%, 95%)")
        ax.plot(z, ref[k], "k--", lw=1.2, label=r"LCDM-limit point")
        ax.set_title(titles[k], fontsize=10); ax.set_xlim(0, max(z))
        if Z_DATA_MAX < max(z):
            ax.axvspan(Z_DATA_MAX, max(z), color="0.85", alpha=0.6, zorder=0, label="beyond SN/BAO data" if k == "Hz" else None)
        if cs.Z_MATCH is not None:
            ax.axvline(cs.Z_MATCH, color="g", ls=":", lw=1.0)
        if k == "Om":
            ax.axhline(1.0, color="r", ls=":", lw=1.0)                 # physical upper bound (flat, rho_DE >= 0)
    for ax in axes[1]:
        ax.set_xlabel("$z$")
    # transition redshift z_t where q crosses zero
    def first_zero(q):
        s = np.where(np.sign(q[:-1]) != np.sign(q[1:]))[0]
        if not len(s):
            return np.nan
        i = s[0]                                   # linear interpolation of the zero crossing (not the grid point)
        return z[i] - q[i] * (z[i + 1] - z[i]) / (q[i + 1] - q[i])
    zt = np.array([first_zero(q) for q in store["q"]])
    axes[1, 3].hist(zt[~np.isnan(zt)], bins=30, color=WONG_PALETTE[5], alpha=0.8)
    axes[1, 3].axvline(first_zero(ref["q"]), color="k", ls="--")
    axes[1, 3].set_title(r"transition redshift $z_t$  ($q=0$)", fontsize=10)
    print(f"z_t (q=0): median {np.nanmedian(zt):.3f}, LCDM-limit point {first_zero(ref['q']):.3f}; "
          f"{np.isnan(zt).sum()}/{len(zt)} draws have no crossing in z<{z.max():.0f}")
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].set_ylim(-2.5, 0.5)
    fig.tight_layout()
    save(fig, "functions_lowz")

    # ---- out to recombination (log axis): matter fraction, f_X(z), H/H_LCDM
    zh = np.geomspace(0.01, 1200.0, 160)
    Om_s, fx_s, hr_s = [], [], []
    for r in rows:
        th = _theta_from_row(r)
        try:
            c = curves(th, MODEL, zh)
            fx = cs.dark_fraction(zh, th, MODEL)
            hr = cs.H_total(zh, th, MODEL) / cs.H_total(zh, th_ref, MODEL) - 1.0
        except cs.CosmoError:
            continue
        if c is not None:
            Om_s.append(c["Om"]); fx_s.append(fx); hr_s.append(hr)
    refc = curves(th_ref, MODEL, zh)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, s, t, refy in zip(axes, [Om_s, fx_s, hr_s],
                              [r"$\Omega_m(z)$ (closure, no radiation)", r"$f_X(z)$: non-(matter+radiation) fraction",
                               r"$H(z)/H_{\Lambda{\rm CDM}}(z)-1$"],
                              [refc["Om"], np.zeros_like(zh), np.zeros_like(zh)]):
        _bands(s, ax, zh, WONG_PALETTE[5])
        ax.plot(zh, refy, "k--", lw=1.0)
        ax.set_xscale("log"); ax.set_title(t); ax.set_xlabel("$z$")
        ax.axvline(1090, color="r", lw=0.8, ls=":")
    axes[1].axhline(0, color="k", lw=0.5)
    axes[0].axhline(1.0, color="r", ls=":", lw=1.0)                    # Omega_m <= 1 in a flat universe with rho_DE >= 0
    for ax in axes:
        ax.axvline(Z_DATA_MAX, color="0.5", ls="--", lw=0.8)             # last redshift with SN/BAO data
        if cs.Z_MATCH is not None:
            ax.axvline(cs.Z_MATCH, color="g", ls=":", lw=1.0)
    if Om_s:
        print(f"Omega_m(z*) > 1 for {sum(o[-1] > 1 + 1e-9 for o in Om_s)}/{len(Om_s)} draws")
    axes[1].set_ylim(-0.4, 0.4)
    axes[0].set_ylim(0, 1.3)
    fig.suptitle(r"Behaviour out to recombination (red dotted: $z_*$, grey dashed: last SN/BAO redshift). The compressed CMB priors assume $f_X(z_*)\approx 0$.", y=1.02)
    fig.tight_layout()
    save(fig, "functions_highz")


# =========================================================
# Data vs best sample
# =========================================================
def plot_residuals(df):
    best = df.loc[df["chi2"].idxmin()]
    theta = _theta_from_row(best)
    noplanck = "ombh2" not in best.index                    # --no-planck chain: r_d is a sampled parameter
    if noplanck:
        wb, q, rd = None, None, float(best["rd"])
    else:
        wb = best["ombh2"]
        q = cs.cmb_quantities(theta, wb, MODEL)
        rd = cs.calibrate(q, dict(rd=RD_PLANCK))["rd"]
    fig, axes = plt.subplots(1, 2 if noplanck else 3, figsize=(10 if noplanck else 15, 4))

    # DESI pulls
    z, means, cov, _, kinds = load_desi()
    vec = bao_model_vector(theta, z, kinds, MODEL, rd)
    pull = (vec - means) / np.sqrt(np.diag(cov))
    for i, k in enumerate(sorted(set(kinds))):
        m = np.array([kk == k for kk in kinds])
        axes[0].plot(z[m], pull[m], "o", color=WONG_PALETTE[i], label=k)
    axes[0].axhline(0, color="k", lw=0.5); axes[0].set_xlabel("$z$"); axes[0].set_ylabel("(model - data)/$\\sigma$")
    axes[0].set_title("DESI BAO pulls"); axes[0].legend(fontsize=8)

    # SN binned residuals
    z_hd, z_hel, mu, scov, _ = load_sn(sn_dataset(df.columns))     # Pantheon+ or DES-SN5YR, from the chain
    DL = luminosity_distance(theta, z_hd, z_hel, MODEL)
    r = 5 * np.log10(DL) + 25 - mu
    ones = np.ones(len(r))
    wv = np.linalg.solve(scov, ones)
    r -= (wv @ r) / (wv @ ones)                         # best-fit magnitude offset
    sig = np.sqrt(np.diag(scov))
    edges = np.quantile(z_hd, np.linspace(0, 1, 21))
    which = np.clip(np.digitize(z_hd, edges[1:-1]), 0, 19)
    zb = [z_hd[which == b].mean() for b in range(20)]
    rb = [np.average(r[which == b], weights=1 / sig[which == b] ** 2) for b in range(20)]
    eb = [1 / np.sqrt(np.sum(1 / sig[which == b] ** 2)) for b in range(20)]
    axes[1].errorbar(zb, rb, eb, fmt="o", color=WONG_PALETTE[5])
    axes[1].axhline(0, color="k", lw=0.5); axes[1].set_xscale("log")
    axes[1].set_xlabel("$z_{HD}$"); axes[1].set_ylabel("$\\mu_{\\rm model}-\\mu_{\\rm obs}$ [mag]")
    axes[1].set_title(("DES-SN5YR" if sn_dataset(df.columns) == "desy5" else "Pantheon+") + " (20 bins, diagonal errors only)")

    if noplanck:
        save(fig, "data_vs_bestfit")
        print(f"best sample: {dict(zip(['H0', 'q0', 'j0', 'Om0'], np.round(theta, 5)))}, r_d={rd:.2f} (sampled, no CMB)")
        return
    # Planck pulls
    mean, pcov, fid = planck_distance_priors()
    c = cs.calibrate(q, fid)
    pred = np.array([c["R"], c["lA"], wb])
    pp = (pred - mean) / np.sqrt(np.diag(pcov))
    axes[2].bar(["R", r"$\ell_A$", r"$\omega_b$"], pp, color=WONG_PALETTE[:3])
    axes[2].axhline(0, color="k", lw=0.5); axes[2].set_ylabel("(model - prior mean)/$\\sigma$")
    axes[2].set_title(f"Planck distance priors (f_X(z*) = {q['fX_star']:+.3f})")
    save(fig, "data_vs_bestfit")
    print(f"best sample: {dict(zip(['H0', 'q0', 'j0', 'Om0'], np.round(theta, 5)))}, ombh2={wb:.5f}, "
          f"r_d={rd:.2f}, f_X(z*)={q['fX_star']:+.4f}")


# =========================================================
# getdist corner (optional)
# =========================================================
def plot_corner():
    try:
        from getdist import loadMCSamples, plots
    except ModuleNotFoundError:
        print("getdist not installed; skipping corner plot")
        return
    names = SAMPLED + (["fX_star"] if "fX_star" in read_chains(FILE_NAME)[0].columns else [])
    s = loadMCSamples(f"chains/{FILE_NAME}", settings={"ignore_rows": BURN})
    g = plots.get_subplot_plotter()
    rename = {"w0": "q0", "wa": "j0"} if LEGACY_CPL else {}     # getdist sees the names stored in the chain files
    if LEGACY_CPL:
        print("NOTE: legacy chain: the corner plot axes q_0, j_0 are really w_0, w_a")
    g.triangle_plot(s, [rename.get(n, n) for n in names], filled=True, title_limit=1,
                    markers={rename.get(n, n): lcdm_value(n) for n in names if lcdm_value(n) is not None})
    g.export(os.path.join(OUT_DIR, f"corner_{FILE_NAME}.pdf"))
    print("saved", os.path.join(OUT_DIR, f"corner_{FILE_NAME}.pdf"))


# =========================================================
# Main
# =========================================================
def main(parts):
    apply_run_settings(FILE_NAME)                 # Z_MATCH of this run, from runs/<prefix>.yaml
    global MODEL, SAMPLED, LEGACY_CPL
    raw = read_chains(FILE_NAME, burn=0.0)
    cols = set(raw[0].columns)
    MODEL = infer_model(FILE_NAME, cols, fallback=MODEL)      # from the prefix / columns, not from constants.MODEL
    print(f"chain {FILE_NAME}: model = {MODEL}")
    if MODEL == "cpl":                                       # flat w0waCDM chain
        if "w0" not in cols and {"q0", "j0"} <= cols:       # legacy: closure classes with the q0, j0 slots = w0, wa
            print("NOTE: legacy CPL chain (columns q0, j0) -> treating q0 as w0 and j0 as wa")
            LEGACY_CPL = True
            for c in raw:
                c["w0"], c["wa"] = c["q0"], c["j0"]
        SAMPLED = ["H0", "Om0", "ombh2", "w0", "wa"]
    elif "j0" not in cols:                                   # LCDM-limit baseline: j0 = 1 fixed
        print("NOTE: no j0 column -> LCDM-limit baseline run")
        for c in raw:
            c["j0"] = 1.0
        SAMPLED = ["H0", "Om0", "ombh2"]
    else:
        SAMPLED = ["H0", "Om0", "ombh2", "q0", "j0"]
    if "ombh2" not in cols and "rd" in cols:                 # make_runs.py --no-planck: r_d sampled, no CMB likelihood
        print("NOTE: no-Planck chain (r_d sampled, no omega_b)")
        SAMPLED = ["rd" if p == "ombh2" else p for p in SAMPLED]
        if is_fixed_noplanck(raw[0]):                        # H0, Om0 fixed: only r_d, q0, j0 are sampled
            SAMPLED = [p for p in SAMPLED if p not in ("H0", "Om0")]
            print("NOTE: H0 and Om0 are fixed in this run")
    chains = [c.iloc[int(BURN * len(c)):].reset_index(drop=True) for c in raw]
    chains = [add_derived(c) for c in chains]
    raw = [add_derived(c) for c in raw]
    df = pd.concat(chains, ignore_index=True)
    print(f"{len(chains)} chains, {len(df)} rows after burn-in (total weight {df['weight'].sum():.0f})")
    params = available(df, SAMPLED)

    if "traces" in parts:
        plot_traces(raw)
    if {"acf", "iat"} & set(parts):
        tab = convergence_table(chains, params)
        if "acf" in parts:
            plot_acf(chains, params, tab)
        if "iat" in parts:
            plot_iat_convergence(chains, params)
    if "marginals" in parts:
        plot_marginals(df)
    if "degeneracy" in parts:
        plot_degeneracies(df)
    if "chi2" in parts:
        plot_chi2(chains)
    if "functions" in parts:
        plot_functions(df)
    if "residuals" in parts:
        try:
            plot_residuals(df)
        except FileNotFoundError as e:
            print("residual plots skipped (data file not found):", e)
    if "corner" in parts:
        plot_corner()


if __name__ == "__main__":
    others = [a for a in sys.argv[1:] if a not in PARTS]      # python plot_chains.py [chain_prefix [yaml]] [parts ...]
    if others:
        FILE_NAME = others[0]
    if len(others) > 1:
        YAML_FILE = others[1]                                  # optional explicit yaml; default runs/<prefix>.yaml
    wanted = tuple(a for a in sys.argv[1:] if a in PARTS) or PARTS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main(wanted)