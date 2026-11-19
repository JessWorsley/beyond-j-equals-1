"""
high_z_check.py -- do the closure models approach matter domination at high redshift?

    python high_z_check.py                          # illustrative point (q0, j0, Om0) = (-0.47, 0.77, 0.31) for all three models
    python high_z_check.py --j0 0.9
    python high_z_check.py prefix1 prefix2 ...      # use the posterior medians of the given chains (any models)

Asymptotics (x = ln(1+z), no radiation, E^2 = h^2, rho_DE = E^2 - Om0 (1+z)^3, Omega_m = Om0 (1+z)^3 / E^2):
  model_1   E^2 ~ (1+z)^p ,  p = (2 + j0 + 3 q0) / (1 + q0)                 p = 3  <=>  j0 = 1
  model_2   E^2 ~ (1+z)^n ,  n = (3 + sqrt(8 j0 + 1)) / 2                   n = 3  <=>  j0 = 1
  model_3   E^2 ~ c (1+z)^3, c = 2 (j0 - q0 (2 q0 + 1)) / (2 j0 - 6 q0 + 1)   (needs D > 0, q0 < 1/2)
            matter scaling always, but Omega_m(inf) = Om0 / c, which is 1 only if c = Om0.
For j0 < 1 the exponent of models 1 and 2 is below 3: E^2 / (1+z)^3 -> 0, so rho_DE -> -Om0 (1+z)^3 and Omega_m grows without bound.
Writes high_z_check.png (Omega_m(z), log axes) and prints Omega_m at several redshifts and the redshift where rho_DE < 0.
"""
import argparse
import numpy as np
import closure_cosmo as cs

ZS = [10, 100, 1100, 1e4, 1e6]


def asymptotic(model, q0, j0, Om0):
    """(exponent n of E^2 ~ (1+z)^n, coefficient of (1+z)^3 if n = 3 else None)."""
    if model == "model_1":
        return (2 + j0 + 3 * q0) / (1 + q0), None
    if model == "model_2":
        return (3 + np.sqrt(8 * j0 + 1)) / 2, None
    D = 2 * j0 - 6 * q0 + 1
    return 3.0, 2 * (j0 - q0 * (2 * q0 + 1)) / D


def omega_m(model, th, z):
    E2 = cs._E2_closure(z, th[1], th[2], model, th[3])
    return None if E2 is None else th[3] * (1 + z) ** 3 / E2


def z_cross(model, th):
    z = np.expm1(np.linspace(0, np.log1p(1e8), 20000))
    E2 = cs._E2_closure(z, th[1], th[2], model, th[3])
    if E2 is None:
        return None
    m = th[3] * (1 + z) ** 3
    neg = np.nonzero(E2 - m < -1e-6 * m)[0]          # relative tolerance: roundoff against (1+z)^3 is ~1e-16
    return z[neg[0]] if neg.size else np.inf


def medians(prefix):
    import pandas as pd
    from chain_utils import read_chains, infer_model
    df = pd.concat(read_chains(prefix, 0.3), ignore_index=True)
    model = infer_model(prefix, df.columns, fallback="model_3")
    w = df["weight"].values.astype(float)
    med = [float(np.quantile(df[c].values, 0.5, method="inverted_cdf", weights=w)) for c in ("H0", "q0", "j0", "Om0")]
    return model, med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prefixes", nargs="*")
    ap.add_argument("--q0", type=float, default=-0.47)
    ap.add_argument("--j0", type=float, default=0.77)
    ap.add_argument("--Om0", type=float, default=0.31)
    a = ap.parse_args()
    cases = [(m, [67.7, a.q0, a.j0, a.Om0]) for m in ("model_1", "model_2", "model_3")]
    if a.prefixes:
        cases = [medians(p) for p in a.prefixes]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    zg = np.expm1(np.linspace(0, np.log1p(1e6), 600))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    print(f"{'model':8s} {'q0':>7s} {'j0':>6s} {'Om0':>6s} {'n_inf':>6s}  " + "  ".join(f"Om(z={z:g})".rjust(11) for z in ZS) + "   z(rho_DE<0)")
    for model, th in cases:
        n, c = asymptotic(model, *th[1:])
        om = [omega_m(model, th, np.array([float(z)])) for z in ZS]
        zc = z_cross(model, th)
        print(f"{model:8s} {th[1]:7.3f} {th[2]:6.3f} {th[3]:6.3f} {n:6.3f}  "
              + "  ".join(("n/a" if o is None else f"{o[0]:.4g}").rjust(11) for o in om)
              + f"   {'n/a' if zc is None else ('never' if np.isinf(zc) else f'{zc:.3g}')}"
              + (f"   [Om0/c = {th[3] / c:.4g}]" if c else ""))
        y = omega_m(model, th, zg)
        if y is not None:
            ax.loglog(1 + zg, y, label=f"{model}  (n={n:.2f})")
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.axvline(1 + 1090, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("1+z")
    ax.set_ylabel(r"$\Omega_m(z)=\Omega_{m0}(1+z)^3/E^2$")
    ax.legend()
    fig.tight_layout()
    fig.savefig("high_z_check.png", dpi=150)
    print("wrote high_z_check.png")


if __name__ == "__main__":
    main()