"""
bound_check.py -- is the model-3 asymptotic bound violated where rho_DE < 0 at high redshift?

    python bound_check.py desi_pantheon_planck_model_3

Model 3:  E^2(x) = [ (1-2q0)^2 e^{beta x} + 2 e^{3x} (j0 - q0 (2 q0 + 1)) ] / D ,   x = ln(1+z),
          D = 2 j0 - 6 q0 + 1,   beta = 2 (j0 - 1) / (2 q0 - 1),   E^2(0) = 1.
Write c = 2 (j0 - q0 (2 q0 + 1)) / D  (so the e^{beta x} coefficient is 1 - c = (1-2q0)^2 / D).
For D > 0 (and q0 < 1/2) one has beta < 3, so as z -> infinity E^2 -> c e^{3x} and
        rho_DE / (1+z)^3 = E^2 / (1+z)^3 - Om0  ->  c - Om0 .
rho_DE >= 0 at high z therefore needs  c >= Om0  <=>  (1-2q0)^2 / D <= 1 - Om0  <=>
        j0 >= 3 q0 - 1/2 + (1-2q0)^2 / (2 (1 - Om0))                    (the bound).
Violation <=> Omega_m(z -> infinity) = Om0 / c > 1, i.e. rho_DE is negative from some redshift onwards.

For every posterior sample this script reports the bound margin  j0 - bound  and the redshift z_cross where rho_DE first
becomes negative (searched up to Z_FAR), and cross-tabulates (weighted):
    A  bound satisfied, rho_DE >= 0 up to z*            B  bound violated, rho_DE < 0 at some z <= z*
    C  bound violated, rho_DE >= 0 up to z*             D  bound satisfied, rho_DE < 0 at some z <= z* (transient)
C means the violation is real but only sets in beyond the recombination epoch, where nothing constrains the closure.
"""
import sys
import numpy as np
import pandas as pd

import closure_cosmo as cs
from chain_utils import read_chains, infer_model, apply_run_settings

Z_STAR_MAX = 1100.0
Z_FAR = 1.0e6
BURN = 0.3
TOL = 1e-9


def bound_margin(q0, j0, Om0):
    """Compute the margin of the model-3 asymptotic positivity bound.

    margin = j0 - [3 q0 - 1/2 + (1 - 2 q0)^2 / (2 (1 - Om0))].

    Parameters
    ----------
    q0, j0, Om0 : float or array_like
        Deceleration, jerk and matter fraction today.

    Returns
    -------
    margin : float or numpy.ndarray
        Negative where the bound is violated.
    """
    return j0 - (3.0 * q0 - 0.5 + (1.0 - 2.0 * q0) ** 2 / (2.0 * (1.0 - Om0)))


def z_cross(theta, z):
    """Find the first redshift at which rho_DE becomes negative (model 3).

    rho_DE = h^2 - Om0 (1+z)^3.

    Parameters
    ----------
    theta : sequence of float
        ``[H0, q0, j0, Om0]``.
    z : numpy.ndarray
        Increasing redshift grid to search.

    Returns
    -------
    zc : float
        First grid redshift with rho_DE < 0, ``np.inf`` if there is none, or
        ``np.nan`` if the background is invalid.
    """
    r = cs.background_model(theta, z, "model_3")
    if r is None:
        return np.nan
    rho = (r[0] / theta[0]) ** 2 - theta[3] * (1.0 + z) ** 3
    neg = np.nonzero(rho < -TOL)[0]
    return z[neg[0]] if neg.size else np.inf


def analyse(df):
    """Classify posterior samples by the bound and by where rho_DE turns negative.

    Parameters
    ----------
    df : pandas.DataFrame
        Model-3 chain samples with columns H0, q0, j0, Om0 and weight.

    Returns
    -------
    res : dict
        Weighted fractions in classes A-D (see the module docstring), invalid
        backgrounds, D < 0 and bound violations; quantiles of z_cross for
        violated samples (``zc_quantiles``) and of the margin
        (``margin_quantiles``).
    """
    z = np.expm1(np.linspace(0.0, np.log1p(Z_FAR), 800))
    q0, j0, Om0 = df["q0"].values, df["j0"].values, df["Om0"].values
    D = 2 * j0 - 6 * q0 + 1
    marg = bound_margin(q0, j0, Om0)
    zc = np.array([z_cross(t, z) for t in df[["H0", "q0", "j0", "Om0"]].values])
    w = df["weight"].values.astype(float)
    w = w / w.sum()
    ok = ~np.isnan(zc)
    viol = marg < 0
    neg_star = ok & (zc <= Z_STAR_MAX)
    cls = {"A  bound ok, rho>=0 to z*": ~viol & ~neg_star & ok, "B  bound violated, rho<0 at z<=z*": viol & neg_star,
           "C  bound violated, rho>=0 to z*": viol & ~neg_star & ok, "D  bound ok, rho<0 at z<=z* (transient)": ~viol & neg_star}
    res = {k: w[m].sum() for k, m in cls.items()}
    res["invalid background"] = w[~ok].sum()
    res["D<0 (denominator)"] = w[D < 0].sum()
    res["violated (all)"] = w[viol].sum()
    # z_cross quantiles among violated samples with a crossing found inside the search range
    sel = viol & np.isfinite(zc)
    res["zc_quantiles"] = (np.quantile(zc[sel], [0.05, 0.5, 0.95], method="inverted_cdf", weights=w[sel])
                           if sel.sum() else None)
    res["no crossing up to Z_FAR (violated)"] = w[viol & np.isinf(zc)].sum()
    res["margin_quantiles"] = np.quantile(marg, [0.05, 0.5, 0.95], method="inverted_cdf", weights=w)
    return res


def report(prefix):
    """Run ``analyse`` on a chain and print the results.

    Parameters
    ----------
    prefix : str
        Chain prefix of a model-3 run.

    Returns
    -------
    res : dict or None
        Output of ``analyse``, or None if the chain is not model 3.
    """
    apply_run_settings(prefix)
    df = pd.concat(read_chains(prefix, BURN), ignore_index=True)
    model = infer_model(prefix, df.columns, fallback="model_3")
    if model != "model_3":
        print(f"{prefix}: this bound is for model_3 (chain looks like {model}); nothing to do"); return None
    r = analyse(df)
    print(f"\n{prefix}: asymptotic positivity bound  j0 >= 3q0 - 1/2 + (1-2q0)^2 / (2(1-Om0))")
    print(f"  margin  j0 - bound  (5%, 50%, 95%): {np.round(r['margin_quantiles'], 4)}   (negative = violated)")
    print(f"  posterior probability that the bound is violated      : {r['violated (all)']:.3f}")
    print(f"  posterior probability that D = 2j0 - 6q0 + 1 < 0      : {r['D<0 (denominator)']:.3f}")
    for k in ("A  bound ok, rho>=0 to z*", "B  bound violated, rho<0 at z<=z*", "C  bound violated, rho>=0 to z*",
              "D  bound ok, rho<0 at z<=z* (transient)", "invalid background"):
        print(f"  {k:46s}: {r[k]:.3f}")
    if r["zc_quantiles"] is not None:
        print(f"  z where rho_DE first < 0, violated samples (5%, 50%, 95%): {np.round(r['zc_quantiles'], 1)}")
    print(f"  violated samples with no crossing up to z = {Z_FAR:.0e}: {r['no crossing up to Z_FAR (violated)']:.3f}")
    return r


if __name__ == "__main__":
    report(sys.argv[1])