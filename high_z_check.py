"""
high_z_check.py -- do the closure models approach matter domination at high redshift?

    python high_z_check.py                          # illustrative point (q0, j0, Om0) = (-0.47, 0.77, 0.31) for all three models
    python high_z_check.py --j0 0.9
    python high_z_check.py pantheon                 # models 1-3 of desi_pantheon_planck (posterior medians of the chains)
    python high_z_check.py pantheon desy5           # + desi_desy5_planck: first dataset solid, second dashed, ...
    python high_z_check.py pantheon --tag _zm3pos   # chains with a tag (the closure itself is plotted, Z_MATCH is ignored)
    python high_z_check.py prefix1 prefix2 ...      # explicit chain prefixes (any models), all solid

Asymptotics (x = ln(1+z), no radiation, E^2 = h^2, rho_DE = E^2 - Om0 (1+z)^3, Omega_m = Om0 (1+z)^3 / E^2):
  model_1   E^2 ~ (1+z)^p ,  p = (2 + j0 + 3 q0) / (1 + q0)                 p = 3  <=>  j0 = 1
  model_2   E^2 ~ (1+z)^n ,  n = (3 + sqrt(8 j0 + 1)) / 2                   n = 3  <=>  j0 = 1
  model_3   E^2 ~ c (1+z)^3, c = 2 (j0 - q0 (2 q0 + 1)) / (2 j0 - 6 q0 + 1)   (needs D > 0, q0 < 1/2)
            matter scaling always, but Omega_m(inf) = Om0 / c, which is 1 only if c = Om0.
For j0 < 1 the exponent of models 1 and 2 is below 3: E^2 / (1+z)^3 -> 0, so rho_DE -> -Om0 (1+z)^3 and Omega_m grows without bound.
Colour = model, line style = dataset; the dashed vertical line is the highest SN/BAO redshift (DESI Lya, 2.33), the dotted one z*.
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


SN_NAMES = ("pantheon", "desy5")
LINESTYLES = ["-", "--", ":", "-."]
COLOURS = {"model_1": "C0", "model_2": "C1", "model_3": "C2"}
Z_DATA_MAX = 2.33          # highest redshift with SN/BAO data (DESI Lya); same value as in plot_chains.py


def build_cases(args, tag, q0, j0, Om0):
    """[(label, model, theta, linestyle)]"""
    sn_args = [x for x in args if x in SN_NAMES]
    other = [x for x in args if x not in SN_NAMES]
    cases = []
    if not args:
        cases = [(m, m, [67.7, q0, j0, Om0], "-") for m in COLOURS]
    from chain_utils import run_prefix
    for k, sn in enumerate(sn_args):
        for m in COLOURS:
            p = run_prefix(sn, m, tag)
            try:
                model, th = medians(p)
            except Exception as e:                      # missing chains etc.: report and go on
                print(f"  skip {p}: {type(e).__name__}: {e}")
                continue
            cases.append((f"{m}, {sn}", model, th, LINESTYLES[k % len(LINESTYLES)]))
    for p in other:
        model, th = medians(p)
        cases.append((p, model, th, "-"))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("args", nargs="*", help="SN dataset names (pantheon, desy5) and/or chain prefixes")
    ap.add_argument("--tag", default="", help="chain tag for the dataset names, e.g. _zm3pos")
    ap.add_argument("--q0", type=float, default=-0.47)
    ap.add_argument("--j0", type=float, default=0.77)
    ap.add_argument("--Om0", type=float, default=0.31)
    a = ap.parse_args()
    cases = build_cases(a.args, a.tag, a.q0, a.j0, a.Om0)
    if not cases:
        raise SystemExit("nothing to plot")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    zg = np.expm1(np.linspace(0, np.log1p(1e6), 600))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    w = max(len(c[0]) for c in cases)
    print(f"{'run':{w}s} {'q0':>7s} {'j0':>6s} {'Om0':>6s} {'n_inf':>6s}  " + "  ".join(f"Om(z={z:g})".rjust(11) for z in ZS) + "   z(rho_DE<0)")
    for label, model, th, ls in cases:
        n, c = asymptotic(model, *th[1:])
        om = [omega_m(model, th, np.array([float(z)])) for z in ZS]
        zc = z_cross(model, th)
        print(f"{label:{w}s} {th[1]:7.3f} {th[2]:6.3f} {th[3]:6.3f} {n:6.3f}  "
              + "  ".join(("n/a" if o is None else f"{o[0]:.4g}").rjust(11) for o in om)
              + f"   {'n/a' if zc is None else ('never' if np.isinf(zc) else f'{zc:.3g}')}"
              + (f"   [Om0/c = {th[3] / c:.4g}]" if c else ""))
        y = omega_m(model, th, zg)
        if y is not None:
            ax.loglog(1 + zg, y, color=COLOURS.get(model, "k"), ls=ls)
    ax.axhline(1, color="k", lw=0.8, ls=":")
    ax.axvline(1 + Z_DATA_MAX, color="0.4", lw=1.0, ls="--")
    ax.axvline(1 + 1090, color="0.7", lw=0.8, ls=":")
    ax.set_xlabel("1+z")
    ax.set_ylabel(r"$\Omega_m(z)=\Omega_{m0}(1+z)^3/E^2$")
    models_h = [Line2D([], [], color=c, label=m) for m, c in COLOURS.items()]
    models_h += [Line2D([], [], color="0.4", ls="--", label=f"last SN/BAO data (z={Z_DATA_MAX})"), Line2D([], [], color="0.7", ls=":", label="z*")]
    leg = ax.legend(handles=models_h, loc="upper left", fontsize=8)
    sn_args = [x for x in a.args if x in SN_NAMES]
    if len(sn_args) > 1:
        ax.add_artist(leg)
        ax.legend(handles=[Line2D([], [], color="k", ls=LINESTYLES[k % 4], label=f"desi_{sn}_planck") for k, sn in enumerate(sn_args)],
                  loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig("high_z_check.png", dpi=150)
    print("wrote high_z_check.png")


if __name__ == "__main__":
    main()