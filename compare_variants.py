"""
compare_variants.py -- one closure model under different assumptions: with Planck, without Planck, positive dark density.

    python compare_variants.py [sn] [model]           default: pantheon model_3
    python compare_variants.py desy5 model_2

Variants (chain prefixes from make_runs.py / run_prefix):
    base       desi_<sn>_planck_<model>               DESI + SN + Planck distance priors
    no Planck  desi_<sn>_planck_<model>_noplanck      DESI + SN, r_d sampled          (make_runs.py --no-planck)
    pos. dark  desi_<sn>_planck_<model>_posdark       as base, rho_DE >= 0 for z <= z* (make_runs.py --positive-dark --tag _posdark)
Edit VARIANTS below for other tags.  Best-fit chi^2 comes from the minimizer output (<prefix>_min) when it exists,
otherwise from the lowest chain sample (marked '*', biased high).

What can be compared (the script prints this too):
  * DESI and SN chi^2 are the same data in every variant, so they are directly comparable.
  * Total chi^2, AIC and BIC only compare variants with the SAME data: base vs pos. dark (same k, so Delta AIC = Delta chi^2).
    The no-Planck run has fewer data (no Planck) and cannot be put in one AIC/BIC ranking with the others.
"""
import os
import sys
import numpy as np
import pandas as pd

import closure_cosmo as cs
from chain_utils import read_chains, run_prefix, infer_model
from model_comparison import best_chi2
import positivity_check as pc

BURN = 0.3
N_DRAW = 400
Z_DATA_MAX = 2.5
OUT_DIR = "chains/plots"

SN = sys.argv[1] if len(sys.argv) > 1 else "pantheon"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "model_3"
# (label, chain prefix, Z_MATCH used for that chain or None).  Z_MATCH is NOT stored in the chains, so it is set here for each
# variant (overriding constants.py) whenever H(z) is recomputed from the samples (plots, rho_DE < 0 probabilities).
VARIANTS = [("base (with Planck)", run_prefix(SN, MODEL), None),
            ("no Planck", run_prefix(SN, MODEL, "_noplanck"), None),
            ("positive dark", run_prefix(SN, MODEL, "_posdark"), None),
            ("z match at 4", run_prefix(SN, MODEL, "_zm4"), 4.0)]
# If pipeline.yaml exists it defines the variants instead (labels, prefixes, Z_MATCH): edit it there.
try:
    from variants_config import compare_list
    _from_manifest = compare_list(SN, MODEL)
    if _from_manifest:
        VARIANTS = _from_manifest
        print("variants from pipeline.yaml:", [v[0] for v in VARIANTS])
except Exception as _e:                                          # noqa: manifest missing or unreadable -> the list above
    print("note: not using pipeline.yaml:", repr(_e))


def short(col):
    """Shorten a chi2 column name to its dataset name.

    Parameters
    ----------
    col : str
        Column such as ``'chi2__likelihoods.DESIFreeRdLikelihood'``.

    Returns
    -------
    name : str
        E.g. ``'DESI'``, ``'Pantheon'``, ``'DESY5'`` or ``'Planck'``.
    """
    b = col.split(".")[-1].replace("Likelihood", "").replace("CPL", "")
    return "DESI" if b.startswith("DESI") else b


def wq(x, w, qs=(0.16, 0.5, 0.84)):
    """Compute weighted quantiles (inverted-CDF method).

    Parameters
    ----------
    x : array_like
        Samples.
    w : array_like
        Sample weights.
    qs : sequence of float, optional
        Quantiles.  Default is (0.16, 0.5, 0.84).

    Returns
    -------
    q : numpy.ndarray
        The quantiles.
    """
    return np.quantile(x, qs, method="inverted_cdf", weights=w)


def load(prefix):
    """Read and concatenate a run's chains after burn-in.

    Parameters
    ----------
    prefix : str
        Chain prefix.

    Returns
    -------
    df : pandas.DataFrame or None
        All chains, or None if there are no chain files.
    """
    try:
        df = pd.concat(read_chains(prefix, BURN), ignore_index=True)
    except FileNotFoundError:
        return None
    return df


def set_zmatch(z_match):
    """Set closure_cosmo's Z_MATCH before H(z) is recomputed from a chain.

    A negative frozen density is always allowed here, because chains only
    contain points the run already accepted.

    Parameters
    ----------
    z_match : float or None
        Z_MATCH the chain was run with.
    """
    cs.Z_MATCH = z_match
    cs.Z_MATCH_ALLOW_NEGATIVE = True


def one(name, prefix, z_match=None):
    """Summarise one variant: best-fit chi^2, posteriors and dark-density positivity.

    The best-fit chi^2 comes from the minimizer if it exists and is not worse
    than the best chain sample; otherwise the best chain sample is used.

    Parameters
    ----------
    name : str
        Variant label.
    prefix : str
        Chain prefix.
    z_match : float or None, optional
        Z_MATCH the chain was run with.

    Returns
    -------
    result : tuple or None
        ``(row, df, model, z_match)``: the table row (dict), the chains, the
        model and `z_match`; None if there are no chains.
    """
    set_zmatch(z_match)
    df = load(prefix)
    if df is None:
        print(f"skipping '{name}': no chains for {prefix}")
        return None
    chi, from_chain = best_chi2(prefix)
    if not from_chain and "chi2" in df.columns and df["chi2"].min() < chi["chi2"] - 1e-6:
        # any evaluated point is an upper bound on chi2_min: keep the lowest of minimiser and chain
        r = df.loc[df["chi2"].idxmin()]
        print(f"note: '{name}': minimiser chi2 {chi['chi2']:.2f} is above the best chain sample {r['chi2']:.2f}; using the chain sample (marked +)")
        chi, from_chain = {c: r[c] for c in df.columns if c.startswith("chi2")}, "chain"
    parts = {short(k): v for k, v in chi.items() if k.startswith("chi2__")}
    sampled = [p for p in ("H0", "Om0", "ombh2", "q0", "j0", "w0", "wa") if p in df.columns]
    if "ombh2" not in df.columns and "rd" in df.columns:
        sampled.append("rd")
    row = dict(variant=name + ("*" if from_chain is True else "+" if from_chain == "chain" else ""), prefix=prefix, k=len(sampled) + 1, chi2=chi["chi2"],
               chi2_DESI=parts.get("DESI", np.nan), chi2_SN=parts.get(SN.capitalize() if SN != "desy5" else "DESY5", np.nan),
               chi2_Planck=parts.get("Planck", np.nan))
    if np.isnan(row["chi2_SN"]):                               # whichever SN column is there
        sn_keys = [k for k in parts if k not in ("DESI", "Planck")]
        row["chi2_SN"] = parts[sn_keys[0]] if sn_keys else np.nan
    if "chi2" in df.columns:
        row["chain_min_chi2"] = float(df["chi2"].min())          # lowest chain sample: an upper bound on the true minimum
    w = df["weight"].values
    for p in ("H0", "Om0", "q0", "j0", "rd"):
        if p in df.columns:
            lo, med, hi = wq(df[p].values, w)
            row[p] = f"{med:.3f} (+{hi - med:.3f} -{med - lo:.3f})"
    if "fX_star" in df.columns:
        lo, med, hi = wq(df["fX_star"].values, w)
        row["fX_star"] = f"{med:+.3f} (+{hi - med:.3f} -{med - lo:.3f})"
    # positivity of the dark density, from the posterior itself (post-processing, closure models only)
    model = infer_model(prefix, df.columns, fallback=MODEL)
    if model != "cpl":
        rng = np.random.default_rng(0)
        idx = rng.choice(len(df), size=min(3000, len(df)), p=w / w.sum())
        sub = df.iloc[idx].reset_index(drop=True)
        sub["weight"] = 1.0
        r = pc.analyse(sub, model, Z_DATA_MAX)
        row["P(rho_DE<0 inside data)"] = r["post"]["neg_in_data"]
        row["P(rho_DE<0 only beyond)"] = r["post"]["neg_only_beyond"]
        if model == "model_3" and z_match is None:           # the asymptotic bound is meaningless if the closure stops at z_match
            from bound_check import bound_margin
            m = bound_margin(df["q0"].values, df["j0"].values, df["Om0"].values)
            row["bound margin (median)"] = float(wq(m, w, (0.5,))[0])
    return row, df, model, z_match


def draws(df, model, n, rng, z_match=None):
    """Compute rho_DE(z) and Omega_m(z) for random posterior draws.

    Parameters
    ----------
    df : pandas.DataFrame
        Chain samples with a ``weight`` column.
    model : str
        Closure model.
    n : int
        Number of draws (drawn with probability proportional to the weight).
    rng : numpy.random.Generator
        Random number generator.
    z_match : float or None, optional
        Z_MATCH the chain was run with.

    Returns
    -------
    z : numpy.ndarray
        Redshifts from 0 to 6.
    rho : numpy.ndarray
        rho_DE(z) / rho_DE(0), one row per valid draw.
    om : numpy.ndarray
        Omega_m(z), one row per valid draw.
    """
    set_zmatch(z_match)
    w = df["weight"].values
    sub = df.iloc[rng.choice(len(df), size=n, p=w / w.sum())]
    z = np.linspace(0.0, 6.0, 241)
    RHO, OM = [], []
    for _, r in sub.iterrows():
        res = cs.background_model([r["H0"], r["q0"], r["j0"], r["Om0"]], z, model)
        if res is None:
            continue
        E2 = (res[0] / r["H0"]) ** 2
        RHO.append((E2 - r["Om0"] * (1 + z) ** 3) / (1 - r["Om0"]))
        OM.append(r["Om0"] * (1 + z) ** 3 / E2)
    return z, np.array(RHO), np.array(OM)


def plot(results):
    """Plot rho_DE(z) and Omega_m(z) bands (68% and 95%) for every variant.

    Saves ``OUT_DIR/variants_<MODEL>_<SN>.pdf``.

    Parameters
    ----------
    results : list of tuple
        ``(name, df, model, z_match)`` per variant.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(2)
    colours = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    for i, (name, df, model, zm) in enumerate(results):
        z, rho, om = draws(df, model, N_DRAW, rng, zm)
        for ax, y in zip(axes, (rho, om)):
            lo95, lo68, med, hi68, hi95 = np.percentile(y, [2.5, 16, 50, 84, 97.5], axis=0)
            ax.fill_between(z, lo95, hi95, color=colours[i], alpha=0.12, lw=0)
            ax.fill_between(z, lo68, hi68, color=colours[i], alpha=0.3, lw=0)
            ax.plot(z, med, color=colours[i], lw=1.8, label=name)
    axes[0].axhline(0, color="k", lw=0.8); axes[0].axhline(1, color="k", ls="--", lw=0.8)
    axes[1].axhline(1, color="k", ls="--", lw=0.8)
    axes[0].axvline(Z_DATA_MAX, color="grey", ls=":", lw=1)
    axes[1].axvline(Z_DATA_MAX, color="grey", ls=":", lw=1)
    axes[0].set_ylabel(r"$\rho_{\rm DE}(z)/\rho_{\rm DE}(0)$"); axes[1].set_ylabel(r"$\Omega_m(z)$")
    axes[0].set_ylim(-3, 6); axes[1].set_ylim(0, 3)
    for ax in axes:
        ax.set_xlabel("$z$")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{MODEL}, {SN}: effective dark density (68% / 95%; dotted: end of the data range)", y=1.0)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"variants_{MODEL}_{SN}.pdf")
    fig.savefig(path, bbox_inches="tight")
    print("saved", path)


def main():
    """Print the variant comparison tables, save them as CSV and make the plot."""
    rows, results = [], []
    for name, prefix, zm in VARIANTS:
        out = one(name, prefix, zm)
        if out is None:
            continue
        row, df, model, zm = out
        rows.append(row)
        results.append((name, df, model, zm))
    if not rows:
        print("no chains found"); return
    tab = pd.DataFrame(rows).set_index("variant")
    base_i = next((i for i in tab.index if i.startswith("base")), tab.index[0])

    pd.set_option("display.width", 200, "display.max_columns", 30)
    print(f"\nBEST-FIT chi^2 PER DATASET   ({MODEL}, {SN};  * = lowest chain sample, no minimiser run;"
          f"  + = chain sample, lower than the minimiser's)")
    c = tab[["k", "chi2_DESI", "chi2_SN", "chi2_Planck", "chi2"]].copy()
    c["DESI+SN"] = c["chi2_DESI"] + c["chi2_SN"]
    c["d(DESI+SN) vs base"] = c["DESI+SN"] - c.loc[base_i, "DESI+SN"]
    if "chain_min_chi2" in tab.columns:
        c["chain min chi2"] = tab["chain_min_chi2"]
    print(c.round(2).to_string())
    if "chain_min_chi2" in tab.columns:
        for i in tab.index:
            gap = tab.loc[i, "chi2"] - tab.loc[i, "chain_min_chi2"]
            if gap > 0.5 and not i.endswith("*"):
                print(f"  WARNING: '{i}': the minimiser chi2 is {gap:.1f} HIGHER than the best chain sample, so the "
                      f"minimiser did not converge (typical for a hard cut like positive_dark).  Do not use it: "
                      f"re-run with more starts (--best-of), or start it from the best chain sample.")

    same = [i for i in tab.index if not np.isnan(tab.loc[i, "chi2_Planck"])]
    if len(same) > 1:
        print("\nSAME-DATA COMPARISON (variants that include Planck; Delta = variant - base)")
        N = None
        try:
            from model_comparison import n_data
            N = n_data(SN)
        except Exception:                                              # noqa
            pass
        for i in same:
            if i == base_i:
                continue
            d = tab.loc[i, "chi2"] - tab.loc[base_i, "chi2"]
            dk = tab.loc[i, "k"] - tab.loc[base_i, "k"]
            msg = f"  {i:22s} d chi2 = {d:+.2f}   d k = {dk:+d}   d AIC = {d + 2 * dk:+.2f}"
            if N:
                msg += f"   d BIC = {d + dk * np.log(N):+.2f}"
            print(msg)
            if "chain_min_chi2" in tab.columns:
                dc = tab.loc[i, "chain_min_chi2"] - tab.loc[base_i, "chain_min_chi2"]
                print(f"  {'':22s} same, from the best CHAIN samples (upper bound, use if a minimiser failed): d chi2 = {dc:+.2f}")
        print("  (positive d chi2 = this assumption fits worse than the base.  Extra constraints add no parameters.)")
    if "no Planck" in " ".join(tab.index):
        print("\nNote: the no-Planck run has no Planck data; its total chi^2, AIC and BIC are not comparable with the others.\n"
              "      Compare DESI+SN above, and the parameters below.")

    cols = [x for x in ("H0", "rd", "Om0", "q0", "j0", "fX_star") if x in tab.columns]
    print("\nPOSTERIOR MEDIANS (68%)")
    print(tab[cols].to_string())
    cols = [x for x in ("P(rho_DE<0 inside data)", "P(rho_DE<0 only beyond)", "bound margin (median)") if x in tab.columns]
    if cols:
        print(f"\nNEGATIVE DARK DENSITY (data range = z <= {Z_DATA_MAX}; bound margin = j0 - bound, model 3 only)")
        print(tab[cols].round(3).to_string())

    os.makedirs("chains", exist_ok=True)
    csv = f"chains/variants_{MODEL}_{SN}.csv"
    tab.to_csv(csv)
    print("\nsaved", csv)
    plot(results)


if __name__ == "__main__":
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()