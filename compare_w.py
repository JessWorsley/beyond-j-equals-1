"""
compare_w.py -- w_DE(z) and rho_DE(z) of the closure models against flat w0waCDM (CPL), and against DESI.
 
    python compare_w.py
 
Edit RUNS / EXTERNAL below.  For every run the plotted bands are 68% and 95% of the posterior draws.
 
Closure models : w_DE(z) = -1 + 1/3 dln(rho_DE)/dx,  rho_DE = E^2 - Om0 (1+z)^3   (numerical, validated in tests)
CPL            : w(z) = w0 + wa z/(1+z),  rho_DE/rho_DE0 = (1+z)^{3(1+w0+wa)} exp(-3 wa z/(1+z))
DESI           : read a public getdist chain (EXTERNAL) and treat it exactly like the CPL chain.
Note: w_DE of a closure model is an EFFECTIVE quantity (everything that is not Om0 (1+z)^3), and can have
rho_DE <= 0, where w is undefined; draws with that property are dropped from the w bands (count is printed).
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
 
from chain_utils import read_chains, apply_run_settings, run_prefix
from constants import WONG_PALETTE
from plot_chains import curves, save                 # numerical q, j, w, rho from H(z)
import plot_chains
 
plot_chains.OUT_DIR = 'chains/plots'
BURN = 0.3
N_POST = 600
Z = np.linspace(0.0, 2.5, 126)                       # DESI plots w(z) over roughly this range
 
SN = sys.argv[1] if len(sys.argv) > 1 else "pantheon"      # python compare_w.py [pantheon|desy5]
RUNS = [   # name, chain prefix (same convention as make_runs.py), kind, closure model
    dict(name="Model 1", prefix=run_prefix(SN, "model_1"), kind="closure", model="model_1"),
    dict(name="Model 2", prefix=run_prefix(SN, "model_2"), kind="closure", model="model_2"),
    dict(name="Model 3", prefix=run_prefix(SN, "model_3"), kind="closure", model="model_3"),
    dict(name="CPL (this pipeline)", prefix=run_prefix(SN, "cpl"), kind="cpl"),
]
# Public DESI chains (getdist format; root = path without extension).  Parameter names must be set to the
# names used in that chain's .paramnames file (often 'w' and 'wa').  Leave empty to skip.
EXTERNAL = [
    # dict(name="DESI DR2 + CMB + Pantheon+", root="path/to/chain_root", w0="w", wa="wa"),
]
# Published central values: drawn as a single curve, no band.  FILL FROM THE PAPER you compare against.
EXTERNAL_POINTS = [
    # dict(name="DESI DR2 BAO+CMB+Pantheon+ (central)", w0=None, wa=None),
]
COLORS = WONG_PALETTE
 
 
def draws_from_chain(df, n, rng):
    """Draw rows from a chain with probability proportional to their weight.

    Parameters
    ----------
    df : pandas.DataFrame
        Chain samples with a ``weight`` column.
    n : int
        Number of draws.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    draws : pandas.DataFrame
        The drawn rows.
    """
    w = df["weight"].values
    return df.iloc[rng.choice(len(df), size=n, p=w / w.sum())].reset_index(drop=True)
 
 
def cpl_curves(w0, wa, z):
    """Compute w(z) and rho_DE(z) for CPL samples.

    Parameters
    ----------
    w0, wa : numpy.ndarray
        CPL parameters, one per sample.
    z : numpy.ndarray
        Redshifts.

    Returns
    -------
    w : numpy.ndarray
        w0 + wa z/(1+z), shape (n_samples, n_z).
    rho : numpy.ndarray
        rho_DE(z) / rho_DE(0), same shape.
    """
    w = w0[:, None] + wa[:, None] * (z / (1 + z))[None, :]
    rho = (1 + z)[None, :] ** (3 * (1 + w0 + wa))[:, None] * np.exp(-3 * wa[:, None] * (z / (1 + z))[None, :])
    return w, rho
 
 
def closure_curves(draws, model, z):
    """Compute w_DE(z) and rho_DE(z) for closure-model draws.

    The number of draws with rho_DE <= 0 somewhere (where w is undefined) is
    printed.

    Parameters
    ----------
    draws : pandas.DataFrame
        Posterior draws with columns H0, q0, j0 and Om0.
    model : str
        Closure model.
    z : numpy.ndarray
        Redshifts.

    Returns
    -------
    w : numpy.ndarray
        w_DE(z), one row per valid draw (NaN where rho_DE <= 0).
    rho : numpy.ndarray
        rho_DE(z) / rho_DE(0), one row per valid draw.
    """
    W, RHO, bad = [], [], 0
    for _, r in draws.iterrows():
        c = curves([r["H0"], r["q0"], r["j0"], r["Om0"]], model, z)
        if c is None:
            continue
        if np.isnan(c["w"]).any():
            bad += 1
        W.append(c["w"]); RHO.append(c["rho"])
    if bad:
        print(f"  {model}: rho_DE <= 0 somewhere for {bad}/{len(W)} draws (w undefined there, ignored in bands)")
    return np.array(W), np.array(RHO)
 
 
def collect(rng):
    """Compute the w and rho curves of every run in RUNS and EXTERNAL.

    Runs without chain files, and external chains that cannot be read, are
    skipped.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random number generator for the posterior draws.

    Returns
    -------
    curves : dict
        Run name to ``dict(w=..., rho=..., kind=...)``.
    """
    out = {}
    for run in RUNS:
        apply_run_settings(run["prefix"])
        try:
            chains = read_chains(run["prefix"], BURN)
        except FileNotFoundError:
            print("skipping (no chain files):", run["name"]); continue
        df_all = pd.concat(chains, ignore_index=True)
        if run["kind"] == "cpl" and "w0" not in df_all.columns and {"q0", "j0"} <= set(df_all.columns):
            print(f"NOTE: {run['name']}: legacy CPL chain (columns q0, j0) -> treating q0 as w0 and j0 as wa")
            df_all["w0"], df_all["wa"] = df_all["q0"], df_all["j0"]
        d = draws_from_chain(df_all, N_POST, rng)
        if run["kind"] == "cpl":
            w, rho = cpl_curves(d["w0"].values, d["wa"].values, Z)
        else:
            w, rho = closure_curves(d, run["model"], Z)
        out[run["name"]] = dict(w=w, rho=rho, kind=run["kind"])
    for ext in EXTERNAL:
        try:
            from getdist import loadMCSamples
            s = loadMCSamples(ext["root"])
            p = s.getParams()
            idx = rng.choice(len(s.weights), size=N_POST, p=s.weights / s.weights.sum())
            w, rho = cpl_curves(np.asarray(getattr(p, ext["w0"]))[idx], np.asarray(getattr(p, ext["wa"]))[idx], Z)
            out[ext["name"]] = dict(w=w, rho=rho, kind="external")
        except Exception as e:                                      # noqa
            print("skipping external chain", ext["name"], "->", repr(e))
    return out
 
 
def band(ax, y, color, label, hatch=None):
    """Draw the 68% and 95% bands and the median of a set of curves on ``Z``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    y : numpy.ndarray
        Curves, one row per draw (NaNs ignored).
    color : str
        Colour.
    label : str
        Legend label of the median.
    hatch : str, optional
        Unused.
    """
    lo95, lo68, med, hi68, hi95 = np.nanpercentile(y, [2.5, 16, 50, 84, 97.5], axis=0)
    ax.fill_between(Z, lo95, hi95, color=color, alpha=0.15, lw=0)
    ax.fill_between(Z, lo68, hi68, color=color, alpha=0.35, lw=0)
    ax.plot(Z, med, color=color, lw=1.8, label=label)
 
 
def main():
    """Plot w_DE(z) and rho_DE(z) of the closure models against CPL and save the figure."""
    rng = np.random.default_rng(1)
    res = collect(rng)
    cpl_names = [n for n, v in res.items() if v["kind"] in ("cpl", "external")]
    closure_names = [n for n, v in res.items() if v["kind"] == "closure"]
    if not res:
        print("nothing to plot"); return
 
    fig, axes = plt.subplots(2, max(len(closure_names), 1) + 1, figsize=(4.2 * (len(closure_names) + 1), 7),
                             sharex=True, squeeze=False)
    for j, name in enumerate(closure_names):
        for i, key in enumerate(("w", "rho")):
            ax = axes[i, j]
            band(ax, res[name][key], COLORS[5], name)
            for k, c in enumerate(cpl_names):
                band(ax, res[c][key], COLORS[1 + k % 4], c)
            ax.axhline(-1 if key == "w" else 1, color="k", ls="--", lw=0.9)
            ax.set_title(name)
            if key == "w":
                ax.set_ylim(-2.5, 0.5)
    # last column: medians of everything
    for name, v in res.items():
        for i, key in enumerate(("w", "rho")):
            axes[i, -1].plot(Z, np.nanmedian(v[key], axis=0), lw=1.8, label=name)
    for ext in EXTERNAL_POINTS:
        if ext.get("w0") is not None:
            axes[0, -1].plot(Z, ext["w0"] + ext["wa"] * Z / (1 + Z), "k:", lw=2, label=ext["name"])
    axes[0, -1].axhline(-1, color="k", ls="--", lw=0.9); axes[1, -1].axhline(1, color="k", ls="--", lw=0.9)
    axes[0, -1].set_ylim(-2.5, 0.5); axes[0, -1].set_title("posterior medians")
    axes[0, -1].legend(fontsize=7)
    axes[0, 0].set_ylabel(r"$w_{\rm DE}(z)$"); axes[1, 0].set_ylabel(r"$\rho_{\rm DE}(z)/\rho_{\rm DE}(0)$")
    for ax in axes[1]:
        ax.set_xlabel("$z$")
    fig.suptitle("Effective dark energy: closure models vs CPL (68% / 95% bands; dashed: LCDM)", y=1.0)
    fig.tight_layout()
    save(fig, "w_de_comparison", tag=SN)
 
 
if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()