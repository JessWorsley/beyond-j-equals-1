"""
positivity_check.py -- does the DATA prefer rho_DE >= 0?  (post-processing of an UNCONSTRAINED closure chain)

    python positivity_check.py desi_desy5_planck_model_2 [z_data_max]

"positive_dark" is a hard cut C = { rho_DE(z) = h^2(z) - Om0 (1+z)^3 >= 0 for all z <= z* }.  It is a theoretical
assumption, not something the data imposed, so we ask how well the data support it, using the unconstrained chain only:

 1. P_post(C)   posterior probability of C (weighted fraction of the unconstrained samples inside C).
 2. P_prior(C)  prior probability of C (uniform draws from the prior box; only draws with a valid background count).
 3. Bayes factor  B = Z(with cut) / Z(without cut) = P_post(C) / P_prior(C)       (exact for a hard cut on the prior)
        B > 1: the data favour C more than the prior alone did;  B < 1: the data disfavour C.
 4. Where does rho_DE go negative?  inside the data range (z <= z_data_max, default 2.5) or only beyond it, where
    only the functional form of the closure (not data) fixes H(z).
 5. Best chi2 (2 x min of minuslogpost over samples) inside and outside C.  A sample-based UPPER bound on chi2_min:
    use the minimiser runs (make_runs.py --minimize, with and without --positive-dark) for the final numbers.

Not for CPL (rho_DE > 0 by construction when Om0 < 1).  The prior box is make_runs.closure_params().
"""
import sys
import numpy as np
import pandas as pd

import closure_cosmo as cs
from chain_utils import read_chains, infer_model, apply_run_settings
from make_runs import closure_params

Z_STAR_MAX = 1100.0           # closure is checked up to roughly the recombination redshift
N_GRID = 400
N_PRIOR = 20000
BURN = 0.3
TOL = 1e-6


def grid(z_data_max):
    """Build the log-spaced redshift grid up to Z_STAR_MAX.

    Parameters
    ----------
    z_data_max : float
        Upper end of the data range.

    Returns
    -------
    z : numpy.ndarray
        Redshifts.
    in_data : numpy.ndarray of bool
        True where z <= `z_data_max`.
    """
    z = np.expm1(np.linspace(0.0, np.log1p(Z_STAR_MAX), N_GRID))
    return z, z <= z_data_max


def classify(theta, model, z, in_data):
    """Find the minimum of rho_DE inside and beyond the data range for one point.

    rho_DE = h^2 - Om0 (1+z)^3.

    Parameters
    ----------
    theta : sequence of float
        ``[H0, q0, j0, Om0]``.
    model : str
        Closure model.
    z : numpy.ndarray
        Redshift grid.
    in_data : numpy.ndarray of bool
        Mask of the data range.

    Returns
    -------
    valid : bool
        Whether the background is valid.
    min_in : float
        Minimum of rho_DE for z <= z_data_max (NaN if invalid).
    min_out : float
        Minimum of rho_DE beyond (NaN if invalid).
    """
    try:
        r = cs.background_model(theta, z, model)
    except Exception:                                     # noqa
        r = None
    if r is None:
        return False, np.nan, np.nan
    H, _ = r
    rho = (H / theta[0]) ** 2 - theta[3] * (1.0 + z) ** 3
    return True, rho[in_data].min(), rho[~in_data].min()


def analyse(df, model, z_data_max=2.5, seed=0):
    """Compute the posterior and prior probability of rho_DE >= 0, and the Bayes factor.

    Parameters
    ----------
    df : pandas.DataFrame
        Unconstrained closure chain with columns H0, q0, j0, Om0 and weight.
    model : str
        Closure model.
    z_data_max : float, optional
        Upper end of the data range.  Default is 2.5.
    seed : int, optional
        Seed of the prior draws.  Default is 0.

    Returns
    -------
    out : dict
        ``post`` (posterior fractions: in C, negative inside the data range,
        negative only beyond), ``prior_C``, ``n_valid_prior``, ``n_eff``;
        ``B``, ``lnB`` and ``lnB_err`` when both probabilities are positive;
        best chi^2 inside and outside C and per-likelihood chi^2 at those
        samples when the chain has chi^2 columns.
    """
    z, in_data = grid(z_data_max)
    w = df["weight"].values.astype(float)
    th = df[["H0", "q0", "j0", "Om0"]].values
    c = np.array([classify(t, model, z, in_data) for t in th])
    ok, rin, rout = c[:, 0].astype(bool), c[:, 1], c[:, 2]
    neg_in = ok & (rin < -TOL)
    neg_out = ok & ~neg_in & (rout < -TOL)
    inC = ok & ~neg_in & ~neg_out
    W = w.sum()
    post = {"C": w[inC].sum() / W, "neg_in_data": w[neg_in].sum() / W, "neg_only_beyond": w[neg_out].sum() / W}

    # prior probability of C among valid prior draws
    rng = np.random.default_rng(seed)
    p = closure_params()
    box = {k: (p[k]["prior"]["min"], p[k]["prior"]["max"]) for k in ("H0", "q0", "j0", "Om0")}
    draws = np.column_stack([rng.uniform(*box[k], N_PRIOR) for k in ("H0", "q0", "j0", "Om0")])
    cp = np.array([classify(t, model, z, in_data) for t in draws])
    okp = cp[:, 0].astype(bool)
    inCp = okp & (cp[:, 1] >= -TOL) & (cp[:, 2] >= -TOL)
    prior_C = inCp.sum() / max(okp.sum(), 1)
    n_eff = 1.0 / np.sum((w / W) ** 2)                      # crude: effective number of posterior samples
    se_post = np.sqrt(max(post["C"] * (1 - post["C"]), 1e-12) / n_eff)
    se_prior = np.sqrt(max(prior_C * (1 - prior_C), 1e-12) / max(okp.sum(), 1))

    out = dict(post=post, prior_C=prior_C, n_valid_prior=int(okp.sum()), n_eff=n_eff)
    if post["C"] > 0 and prior_C > 0:
        out["B"] = post["C"] / prior_C
        out["lnB"] = np.log(out["B"])
        out["lnB_err"] = np.hypot(se_post / post["C"], se_prior / prior_C)
    # Cobaya's derived `chi2` column = -2 log-likelihood (no prior constant); fall back to 2*minuslogpost (offset by a constant)
    m = df["chi2"].values if "chi2" in df else (2 * df["minuslogpost"].values if "minuslogpost" in df else None)
    if m is not None:
        out["chi2_best_in_C"] = m[inC].min() if inC.any() else np.nan
        out["chi2_best_outside_C"] = m[~inC & ok].min() if (~inC & ok).any() else np.nan
        # which dataset drives the difference?  chi2 per likelihood at the best sample inside / outside C
        parts = [c for c in df.columns if c.startswith("chi2__")]
        if parts and inC.any() and (~inC & ok).any():
            i_in = np.flatnonzero(inC)[np.argmin(m[inC])]
            i_out = np.flatnonzero(~inC & ok)[np.argmin(m[~inC & ok])]
            out["chi2_parts"] = {c.split(".")[-1]: (df[c].values[i_in], df[c].values[i_out]) for c in parts}
    return out


def report(prefix, z_data_max=2.5):
    """Run ``analyse`` on a chain and print the results.

    Parameters
    ----------
    prefix : str
        Chain prefix.
    z_data_max : float, optional
        Upper end of the data range.  Default is 2.5.

    Returns
    -------
    res : dict or None
        Output of ``analyse``, or None for a CPL chain.
    """
    apply_run_settings(prefix)                       # Z_MATCH of the run (from runs/<prefix>.yaml)
    chains = read_chains(prefix, BURN)
    df = pd.concat(chains, ignore_index=True)
    model = infer_model(prefix, df.columns, fallback="model_1")
    if model == "cpl":
        print("CPL: rho_DE > 0 by construction (Om0 < 1); nothing to check"); return None
    r = analyse(df, model, z_data_max)
    p = r["post"]
    print(f"\n{prefix}  ({model}),  data range z <= {z_data_max}")
    print(f"  posterior probability of rho_DE >= 0 for all z <= z*      : {p['C']:.3f}")
    print(f"  posterior prob. rho_DE < 0 INSIDE the data range           : {p['neg_in_data']:.3f}")
    print(f"  posterior prob. rho_DE < 0 only BEYOND z = {z_data_max}            : {p['neg_only_beyond']:.3f}")
    print(f"  prior probability of rho_DE >= 0 (valid draws: {r['n_valid_prior']})       : {r['prior_C']:.3f}")
    if "B" in r:
        print(f"  Bayes factor  Z(rho>=0) / Z(unconstrained) = {r['B']:.2f}   (ln B = {r['lnB']:.2f} +- {r['lnB_err']:.2f}; "
              f"N_eff ~ {r['n_eff']:.0f})")
    else:
        print("  Bayes factor: no posterior mass (or no prior mass) in the rho_DE >= 0 region")
    if "chi2_best_in_C" in r:
        print(f"  best sample chi2: inside C {r['chi2_best_in_C']:.2f}   outside C {r['chi2_best_outside_C']:.2f}"
              f"   (upper bounds on chi2_min; use the minimiser runs for the final numbers)")
    if "chi2_parts" in r:
        print("  chi2 per likelihood at the best sample   [inside C | outside C | outside - inside]")
        for k, (a, b) in r["chi2_parts"].items():
            print(f"     {k:24s} {a:9.2f} {b:9.2f} {b - a:+8.2f}")
    return r


if __name__ == "__main__":
    report(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 2.5)