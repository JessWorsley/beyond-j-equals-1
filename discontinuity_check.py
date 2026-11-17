"""
discontinuity_check.py -- how big is the kink at Z_MATCH?   python discontinuity_check.py <prefix of a z-match run>

With Z_MATCH = z_m the closure is used for z <= z_m and E^2 = Om0 (1+z)^3 + rho_m beyond, rho_m = rho_DE(z_m) frozen.
E^2 (hence H, distances and every likelihood) is continuous at z_m; the first derivative of E^2 jumps by -rho'_left:
    rho'_left = dE^2_closure/dz - 3 Om0 (1+z_m)^2        (right side: rho' = 0)
so  w_DE jumps from w_left = -1 + (1+z) rho'_left / (3 rho_m)  to -1,  and the deceleration parameter jumps by
    Delta q = q_right - q_left = -(1+z_m) rho'_left / (2 E^2(z_m)).
What matters for the data is how much dark energy there is where this happens, Omega_DE(z_m) = rho_m / E^2(z_m), and Delta q.
Reports weighted posterior medians and 16/84 percent quantiles per sample.
"""
import sys
import numpy as np
import pandas as pd

import closure_cosmo as cs
from chain_utils import read_chains, infer_model, apply_run_settings

BURN = 0.3


def wq(x, w, qs=(0.16, 0.5, 0.84)):
    """Compute weighted quantiles of the finite values only.

    Parameters
    ----------
    x : numpy.ndarray
        Samples (may contain NaN).
    w : numpy.ndarray
        Sample weights.
    qs : sequence of float, optional
        Quantiles.  Default is (0.16, 0.5, 0.84).

    Returns
    -------
    q : list of float
        The quantiles, or NaNs if no value is finite.
    """
    m = np.isfinite(x)
    if not m.any():
        return [np.nan] * len(qs)
    return list(np.quantile(x[m], qs, method="inverted_cdf", weights=w[m]))


def kink(q0, j0, Om0, model, zm, dz=1e-5):
    """Measure the kink in E^2 at Z_MATCH for one parameter point.

    Parameters
    ----------
    q0, j0, Om0 : float
        Closure parameters.
    model : str
        Closure model.
    zm : float
        Matching redshift z_m.
    dz : float, optional
        Step of the one-sided derivative.  Default is 1e-5.

    Returns
    -------
    values : tuple of float
        ``(Omega_DE(z_m), rho'_left, Delta q, w_left, E2_m)``: dark-energy
        fraction at z_m, left derivative of rho_DE, jump of q, w_DE just below
        z_m, and E^2(z_m).  NaNs if the closure fails.
    """
    z = np.array([zm - dz, zm])
    E2 = cs._E2_closure(z, q0, j0, model, Om0)
    if E2 is None:
        return (np.nan,) * 5
    rho_m = E2[1] - Om0 * (1 + zm) ** 3
    dE2 = (E2[1] - E2[0]) / dz
    rho_p = dE2 - 3 * Om0 * (1 + zm) ** 2
    dq = -(1 + zm) * rho_p / (2 * E2[1])
    w_left = -1 + (1 + zm) * rho_p / (3 * rho_m) if abs(rho_m) > 1e-12 else np.nan
    return rho_m / E2[1], rho_p, dq, w_left, E2[1]


def analyse(df, model, zm):
    """Apply ``kink`` to every sample of a chain.

    Parameters
    ----------
    df : pandas.DataFrame
        Chain samples with columns q0, j0, Om0 and weight.
    model : str
        Closure model.
    zm : float
        Matching redshift.

    Returns
    -------
    w : numpy.ndarray
        Normalised weights.
    arr : numpy.ndarray
        Output of ``kink``, shape (n_samples, 5).
    """
    w = df["weight"].values.astype(float)
    w = w / w.sum()
    arr = np.array([kink(q, j, o, model, zm) for q, j, o in df[["q0", "j0", "Om0"]].values])
    return w, arr


def main(prefix):
    """Print the size of the kink at Z_MATCH for a z-match chain.

    Parameters
    ----------
    prefix : str
        Chain prefix; its yaml must contain ``z_match``.
    """
    apply_run_settings(prefix)
    zm = cs.Z_MATCH
    if zm is None:
        sys.exit(f"{prefix}: no z_match in the yaml; run this on a z-match chain")
    df = pd.concat(read_chains(prefix, BURN), ignore_index=True)
    model = infer_model(prefix, df.columns, fallback="model_3")
    w, a = analyse(df, model, zm)
    names = ["Omega_DE(z_m)", "rho'_left", "Delta q  (q_right - q_left)", "w_DE just below z_m (right side: -1)"]
    print(f"{prefix}: Z_MATCH = {zm}, model {model}, {len(df)} samples")
    for i, n in enumerate(names):
        lo, med, hi = wq(a[:, i], w)
        print(f"  {n:40s} {med:+.4f}  (16-84%: {lo:+.4f} .. {hi:+.4f})")
    print(f"  fraction of the posterior with rho_DE(z_m) < 0: {w[a[:, 0] < 0].sum():.3f}")


if __name__ == "__main__":
    main(sys.argv[1])