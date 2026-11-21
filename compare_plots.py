"""
compare_plots.py -- comparison figures with the models (and variants) on the same axes, driven by pipeline.yaml.

    python compare_plots.py                                 # every sn / model of the manifest, variants = `plot_variants`
    python compare_plots.py --sn pantheon --variants base zm3pos
    python compare_plots.py --only jq ww                    # only some figures (names below)

Figures (per supernova sample, files <outdir>/<sn>_<name>.pdf and .png).  Colour = model, line style = variant, black dotted = LCDM.
  jq           j versus q: the closure relation of each model with epsilon from its best-fitting variant (statefinder plane)
  w_lowz       w_DE(z), z <= 2.5                         rho_lowz   rho_DE(z)/rho_DE(0), z <= 2.5
  H_z8         H(z) and H/H_LCDM - 1, z <= 8              H_full     the same out to z_*
  rho_full     rho_DE(z)/rho_DE(0) out to z_* (symlog)
  h_lowz q_lowz j_lowz   h = H/H0, q, j for z <= 2.5 with the difference to LCDM underneath
  omega_lowz omega_full  Omega_m(z) (solid) and Omega_DE(z) = 1 - Omega_m (dashed), one panel per model, colour = variant
  ww_quintessence ww_tachyon   present-day tracks in the w-w' plane (w' = dw/dln a) with the freezing quintessence / tachyon region
With a single variant the 68% bands are drawn too (--bands forces them).  The no-Planck runs appear only in h, q, j:
Om0 (hence rho_DE, w_DE, Omega_m) and H0 are not constrained without Planck.  Needs the chains and runs/*.yaml.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import closure_cosmo as cs
import variants_config as vc
from chain_utils import read_chains, infer_model, apply_run_settings
from derived import derived_samples

plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11})

Z_DATA_MAX = 2.33                      # highest SN/BAO redshift (DESI Lya)
Z_LOW, Z_MID, Z_STAR = 2.5, 8.0, 1090.0
BURN, N_POST, H_STEP = 0.3, 300, 1e-3
COL = {"model_1": "#D55E00", "model_2": "#E69F00", "model_3": "#009E73"}
NAME = {"model_1": "Model I", "model_2": "Model II", "model_3": "Model III"}
LS = ["-", "--", "-.", ":"]
VAR_COL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#999999"]
GRIDS = {"low": np.linspace(0.0, Z_LOW, 120), "mid": np.linspace(0.0, Z_MID, 200),
         "full": np.expm1(np.linspace(0.0, np.log1p(Z_STAR), 300))}
NEEDS_OM0 = {"H", "rho", "Om", "OmDE", "w", "wp"}          # not available without Planck


# --------------------------------------------------------------------------------------------------- curves
def _d1_d2(L, h):
    d1 = (L[0] - 8 * L[1] + 8 * L[3] - L[4]) / (12 * h)
    d2 = (-L[0] + 16 * L[1] - 30 * L[2] + 16 * L[3] - L[4]) / (12 * h * h)
    return d1, d2


def curves(theta, model, z, h=H_STEP):
    """E=H/H0, H, q, j, rho_DE/rho_DE(0), Omega_m, Omega_DE, w_DE, w' = dw/dln a  on the grid z (5-point derivatives in x = ln(1+z));
    None if the background is invalid.  Derivative-based quantities are NaN within 3 steps of Z_MATCH (the kink)."""
    H0, q0, j0, Om0 = theta
    x = np.log1p(np.asarray(z, dtype=float))
    xs = x[None, :] + h * np.arange(-2, 3)[:, None]
    E2 = []
    for xx in xs:
        r = cs.background_model([1.0, q0, j0, Om0], np.expm1(xx), model)          # H0 = 1: returns E
        if r is None:
            return None
        E2.append(r[0] ** 2)
    E2 = np.array(E2)
    d1, d2 = _d1_d2(np.log(E2), h)
    q = 0.5 * d1 - 1.0
    j = q + 2 * q ** 2 + 0.5 * d2
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = E2 - Om0 * np.exp(3 * xs)
        lr = np.log(np.where(rho > 0, rho, np.nan))
        dr, d2r = _d1_d2(lr, h)
    w = -1 + dr / 3.0
    wp = -d2r / 3.0                                         # dw/dln a = -dw/dx
    if cs.Z_MATCH is not None:
        bad = np.abs(x - np.log1p(cs.Z_MATCH)) < 3 * h
        for a in (q, j, w, wp):
            a[bad] = np.nan
    E = np.sqrt(E2[2])
    Om = Om0 * np.exp(3 * x) / E2[2]
    return dict(E=E, H=H0 * E, q=q, j=j, rho=rho[2] / (1 - Om0), Om=Om, OmDE=1 - Om, w=w, wp=wp)


def lcdm_curves(H0, Om0, z):
    z = np.asarray(z, dtype=float)
    E2 = Om0 * (1 + z) ** 3 + 1 - Om0
    Om = Om0 * (1 + z) ** 3 / E2
    return dict(E=np.sqrt(E2), H=H0 * np.sqrt(E2), q=1.5 * Om - 1, j=np.ones_like(z), rho=np.ones_like(z), Om=Om, OmDE=1 - Om,
                w=-np.ones_like(z), wp=np.zeros_like(z))


# --------------------------------------------------------------------------------------------------- runs
class Run:
    def __init__(self, r, rng):
        self.variant, self.label, self.model, self.prefix, self.sn = r["variant"], r["label"], r["model"], r["prefix"], r["sn"]
        self.z_match = r["z_match"]
        self.df = None
        try:
            chains = read_chains(self.prefix, BURN)
            self.df = pd.concat(chains, ignore_index=True)
        except Exception as e:                                    # noqa
            print(f"  skip {self.prefix}: {type(e).__name__}: {e}")
            return
        self.model = infer_model(self.prefix, self.df.columns, fallback=self.model)
        self.planck = "ombh2" in self.df.columns
        w = self.df["weight"].values.astype(float)
        idx = rng.choice(len(self.df), size=N_POST, p=w / w.sum())
        self.sub = self.df.iloc[idx].reset_index(drop=True)
        self.w = w / w.sum()
        self._cache = {}

    def _set_closure(self):
        try:
            apply_run_settings(self.prefix)
        except Exception:                                         # noqa
            cs.Z_MATCH = self.z_match

    def samples(self, grid):
        """dict quantity -> array (n_draws, n_z) on the named grid."""
        if grid not in self._cache:
            self._set_closure()
            z, out, bad = GRIDS[grid], {}, 0
            for _, r in self.sub.iterrows():
                th = [r["H0"] if "H0" in r else 1.0, r["q0"], r["j0"], r["Om0"]]
                c = curves(th, self.model, z)
                if c is None:
                    bad += 1
                    continue
                for k, v in c.items():
                    out.setdefault(k, []).append(v)
            self._cache[grid] = {k: np.array(v) for k, v in out.items()}
            if bad:
                print(f"  {self.prefix}: {bad} of {len(self.sub)} draws had no valid background on grid '{grid}'")
        return self._cache[grid]

    def pct(self, grid, key):
        arr = self.samples(grid)[key]
        return np.nanpercentile(arr, [2.5, 16, 50, 84, 97.5], axis=0)

    def eps(self):
        v = derived_samples(self.df, self.model, with_om0=False)["eps"]
        return v, self.w


def wq(x, w, qs):
    m = np.isfinite(x)
    return np.quantile(x[m], qs, method="inverted_cdf", weights=w[m])


def lcdm_reference(sn, rng):
    """(H0, Om0) of the LCDM-limit benchmark chain (posterior medians); a fixed fallback if it is missing."""
    try:
        df = pd.concat(read_chains(vc.run_prefix(sn, "lcdm"), BURN), ignore_index=True)
        w = df["weight"].values.astype(float)
        return float(wq(df["H0"].values, w / w.sum(), [0.5])[0]), float(wq(df["Om0"].values, w / w.sum(), [0.5])[0])
    except Exception as e:                                        # noqa
        print(f"  LCDM benchmark chain not found ({type(e).__name__}); using H0 = 67.7, Om0 = 0.31 as the LCDM reference")
        return 67.7, 0.31


# --------------------------------------------------------------------------------------------------- drawing
def style(run, variants):
    return COL.get(run.model, "k"), LS[variants.index(run.variant) % len(LS)]


def legends(ax, runs, variants, labels, loc_models="best"):
    models = sorted({r.model for r in runs})
    h1 = [Line2D([], [], color=COL[m], label=NAME[m]) for m in models] + [Line2D([], [], color="k", ls=":", label=r"$\Lambda$CDM")]
    leg = ax.legend(handles=h1, loc=loc_models, fontsize=8, framealpha=0.85)
    if len(variants) > 1:
        ax.add_artist(leg)
        ax.legend(handles=[Line2D([], [], color="0.3", ls=LS[i % len(LS)], label=labels[v]) for i, v in enumerate(variants)],
                  loc="lower right" if loc_models != "lower right" else "upper left", fontsize=7, framealpha=0.85)


def xaxis(ax, grid):
    if grid == "full":
        ax.set_xscale("log")
        ax.set_xlabel("$1+z$")
    else:
        ax.set_xlabel("$z$")


def xvals(grid):
    z = GRIDS[grid]
    return 1 + z if grid == "full" else z


def mark_data(ax, grid):
    if grid == "full":
        ax.axvline(1 + Z_DATA_MAX, color="0.5", lw=0.8, ls="--")
        ax.axvline(1 + Z_STAR, color="0.7", lw=0.8, ls=":")
    elif Z_DATA_MAX < GRIDS[grid][-1]:
        ax.axvline(Z_DATA_MAX, color="0.5", lw=0.8, ls="--")


def line_figure(runs, variants, labels, lcdm, key, grid, ylabel, bands, delta=None, yscale=None, ylim=None, fname=None, ratio=False):
    """One figure: key(z) of every run; optional lower panel with the difference to LCDM (delta label) or ratio-1 in percent."""
    x = xvals(grid)
    two = delta is not None
    fig, axes = plt.subplots(2 if two else 1, 1, figsize=(6.4, 5.4 if two else 4.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.3]} if two else None)
    ax = axes[0] if two else axes
    ref = lcdm_curves(*lcdm, GRIDS[grid])
    for r in runs:
        if key not in r.samples(grid):
            continue
        c, ls = style(r, variants)
        lo95, lo68, med, hi68, hi95 = r.pct(grid, key)
        if bands:
            ax.fill_between(x, lo68, hi68, color=c, alpha=0.25, lw=0)
        ax.plot(x, med, color=c, ls=ls, lw=1.6)
        if two:
            if ratio:
                f = lambda a: 100 * (a / ref[key] - 1)
            else:
                f = lambda a: a - ref[key]
            if bands:
                axes[1].fill_between(x, f(lo68), f(hi68), color=c, alpha=0.25, lw=0)
            axes[1].plot(x, f(med), color=c, ls=ls, lw=1.6)
    ax.plot(x, ref[key], color="k", ls=":", lw=1.2)
    ax.set_ylabel(ylabel)
    if yscale:
        ax.set_yscale(yscale, **({"linthresh": 1.0} if yscale == "symlog" else {}))
    if ylim:
        ax.set_ylim(*ylim)
    mark_data(ax, grid)
    legends(ax, runs, variants, labels)
    if two:
        axes[1].axhline(0, color="k", ls=":", lw=1.0)
        axes[1].set_ylabel(delta)
        mark_data(axes[1], grid)
        xaxis(axes[1], grid)
        if yscale == "log":
            axes[1].set_yscale("symlog", linthresh=1.0)
    else:
        xaxis(ax, grid)
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.06)
    return fig


def omega_figure(runs, variants, labels, lcdm, grid):
    models = sorted({r.model for r in runs})
    fig, axes = plt.subplots(1, len(models), figsize=(4.6 * len(models), 3.9), sharey=True, squeeze=False)
    axes = axes[0]
    x = xvals(grid)
    ref = lcdm_curves(*lcdm, GRIDS[grid])
    for ax, m in zip(axes, models):
        for r in [r for r in runs if r.model == m and r.planck]:
            c = VAR_COL[variants.index(r.variant) % len(VAR_COL)]
            ax.plot(x, r.pct(grid, "Om")[2], color=c, lw=1.5)
            ax.plot(x, r.pct(grid, "OmDE")[2], color=c, lw=1.5, ls="--")
        ax.plot(x, ref["Om"], "k:", lw=1.0)
        ax.plot(x, ref["OmDE"], "k:", lw=1.0)
        ax.axhline(1, color="0.6", lw=0.6)
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_title(NAME[m])
        mark_data(ax, grid)
        xaxis(ax, grid)
        ax.set_ylim(-1.0, 3.0) if grid == "full" else ax.set_ylim(-0.2, 1.2)
    axes[0].set_ylabel(r"$\Omega_m$ (solid), $\Omega_{\rm DE}$ (dashed)")
    axes[-1].legend(handles=[Line2D([], [], color=VAR_COL[i % len(VAR_COL)], label=labels[v]) for i, v in enumerate(variants)] +
                    [Line2D([], [], color="k", ls=":", label=r"$\Lambda$CDM")], fontsize=7, loc="center right")
    fig.tight_layout()
    return fig


def jq_figure(sn, runs, variants, labels, best):
    """j(q) = closure relation with epsilon of each model's best-fitting variant; the (q0, j0) point lies on it."""
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    q = np.linspace(-1.0, 0.5, 100)
    rel = {"model_1": lambda e: 1 + 3 * e * (q + 1), "model_2": lambda e: 1 + e + 0 * q, "model_3": lambda e: 1 + 3 * e * (q - 0.5)}
    ax.plot(q, np.ones_like(q), "k:", lw=1.2, label=r"$\Lambda$CDM ($\epsilon=0$)")
    for m in ("model_1", "model_2", "model_3"):
        r = best.get(m)
        if r is None:
            continue
        e, w = r.eps()
        lo, med, hi = wq(e, w, [0.16, 0.5, 0.84])
        ax.fill_between(q, rel[m](lo), rel[m](hi), color=COL[m], alpha=0.2, lw=0)
        ax.plot(q, rel[m](med), color=COL[m], lw=1.6, label=f"{NAME[m]} ({labels[r.variant]}, $\\epsilon={med:.3f}$)")
        ax.annotate("", xy=(-0.55, rel[m](med)[np.argmin(abs(q + 0.55))]), xytext=(-0.3, rel[m](med)[np.argmin(abs(q + 0.3))]),
                    arrowprops=dict(arrowstyle="->", color=COL[m]))
        qw, jw = r.df["q0"].values, r.df["j0"].values
        qm, qlo, qhi = (float(wq(qw, r.w, [p])[0]) for p in (0.5, 0.16, 0.84))
        jm, jlo, jhi = (float(wq(jw, r.w, [p])[0]) for p in (0.5, 0.16, 0.84))
        ax.errorbar([qm], [jm], xerr=[[qm - qlo], [qhi - qm]], yerr=[[jm - jlo], [jhi - jm]], fmt="o", ms=4, color=COL[m], capsize=2)
    ax.set_xlabel("$q$")
    ax.set_ylabel("$j$")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    return fig


def ww_figure(runs, variants, labels, kind):
    from matplotlib.patches import Ellipse
    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    planck = [r for r in runs if r.planck]
    allw, allp = [], []
    for r in planck:
        c, ls = style(r, variants)
        S = r.samples("low")
        w, wp = np.nanmedian(S["w"], axis=0), np.nanmedian(S["wp"], axis=0)
        ax.plot(w, wp, color=c, ls=ls, lw=1.6)
        ax.plot(w[0], wp[0], "o", color=c, ms=4)
        k = int(np.argmin(abs(GRIDS["low"] - 1.2)))
        if np.isfinite(w[k]) and np.isfinite(wp[k]):                    # arrow: direction of cosmic time (decreasing z)
            ax.annotate("", xy=(w[k - 5], wp[k - 5]), xytext=(w[k], wp[k]), arrowprops=dict(arrowstyle="->", color=c))
        p0 = np.column_stack([S["w"][:, 0], S["wp"][:, 0]])
        p0 = p0[np.all(np.isfinite(p0), axis=1)]
        if len(p0) > 20:
            mu, cov = p0.mean(0), np.cov(p0.T)
            ev, evec = np.linalg.eigh(cov)
            ang = np.degrees(np.arctan2(evec[1, 1], evec[0, 1]))
            ax.add_patch(Ellipse(mu, 2 * np.sqrt(2.30 * ev[1]), 2 * np.sqrt(2.30 * ev[0]), angle=ang, fc=c, ec=c, alpha=0.18, lw=0.8))
        allw += list(w[np.isfinite(w)])
        allp += list(wp[np.isfinite(wp)])
    ax.plot([-1], [0], "ko", ms=5)
    wg = np.linspace(-1.0, 0.0, 200)
    if kind == "quintessence":      # freezing quintessence (Caldwell & Linder 2005): 3w(1+w) < w' < 0.2 w(1+w)
        ax.fill_between(wg, 3 * wg * (1 + wg), 0.2 * wg * (1 + wg), color="#aab8d8", alpha=0.5, lw=0, label="freezing quintessence bound")
    else:                           # freezing tachyon: -3(1+w) < w' < 0   (read off the paper figure; verify)
        ax.fill_between(wg, -3 * (1 + wg), 0.0, color="#aab8d8", alpha=0.5, lw=0, label="freezing tachyon bound")
    if allw:
        pad_w, pad_p = 0.1 * (np.ptp(allw) + 0.05), 0.1 * (np.ptp(allp) + 0.05)
        ax.set_xlim(min(min(allw), -1) - pad_w, max(allw) + pad_w)
        ax.set_ylim(min(allp + [0]) - pad_p, max(allp + [0]) + pad_p)
    ax.set_xlabel(r"$w_{\rm DE}$")
    ax.set_ylabel(r"$w_{\rm DE}'=\mathrm{d}w_{\rm DE}/\mathrm{d}\ln a$")
    h = [Line2D([], [], color=COL[m], label=NAME[m]) for m in sorted({r.model for r in planck})] + \
        [matplotlib.patches.Patch(color="#aab8d8", alpha=0.5, label=f"freezing {kind} bound"), Line2D([], [], color="k", marker="o", ls="", label=r"$\Lambda$CDM")]
    leg = ax.legend(handles=h, fontsize=7, loc="upper right")
    if len(variants) > 1:
        ax.add_artist(leg)
        ax.legend(handles=[Line2D([], [], color="0.3", ls=LS[i % len(LS)], label=labels[v]) for i, v in enumerate(variants)], fontsize=7, loc="lower left")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------------------- main
def best_variant(sn, model, runs):
    """The run of `model` whose csv row has the lowest chi2 among variants with Planck; the first run if there is no csv."""
    cand = [r for r in runs if r.model == model and r.planck]
    path = f"chains/variants_{model}_{sn}.csv"
    if os.path.exists(path) and cand:
        t = pd.read_csv(path, index_col=0)
        t = t[t["chi2_Planck"].notna()]
        t.index = [i.rstrip("+*").strip() for i in t.index]
        known = {r.label: r for r in cand}
        t = t[[i in known for i in t.index]]
        if len(t):
            return known[t["chi2"].idxmin()]
    return cand[0] if cand else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=vc.MANIFEST)
    ap.add_argument("--sn", nargs="+", default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--variants", nargs="+", default=None, help="variant names or labels (default: plot_variants of the manifest)")
    ap.add_argument("--only", nargs="+", default=None, help="figure names (see the docstring)")
    ap.add_argument("--bands", action="store_true", help="68%% bands also with several variants")
    ap.add_argument("--outdir", default="chains/plots/compare")
    a = ap.parse_args()
    cfg = vc.load(a.manifest)
    sns, models = a.sn or cfg["sn"], a.models or cfg["models"]
    want = a.variants or cfg.get("plot_variants") or [v["name"] for v in cfg["variants"]]
    os.makedirs(a.outdir, exist_ok=True)
    rng = np.random.default_rng(0)
    everything = {"jq", "w_lowz", "rho_lowz", "H_z8", "H_full", "rho_full", "h_lowz", "q_lowz", "j_lowz", "omega_lowz", "omega_full",
                  "ww_quintessence", "ww_tachyon"}
    todo = set(a.only) if a.only else everything

    for sn in sns:
        print(f"== {sn}")
        rows = [r for r in vc.runs(cfg) if r["sn"] == sn and r["model"] in models and r["variant"] != "benchmarks"
                and (r["variant"] in want or r["label"] in want)]
        runs = [Run(r, rng) for r in rows]
        runs = [r for r in runs if r.df is not None]
        if not runs:
            continue
        variants = [v for v in dict.fromkeys(r.variant for r in runs)]
        labels = {r.variant: r.label for r in runs}
        lcdm = lcdm_reference(sn, rng)
        planck_runs = [r for r in runs if r.planck]
        bands = a.bands or len(variants) == 1

        def save(fig, name):
            for ext in ("pdf", "png"):
                fig.savefig(f"{a.outdir}/{sn}_{name}.{ext}", dpi=150)
            plt.close(fig)
            print("  saved", f"{a.outdir}/{sn}_{name}")

        if "jq" in todo:
            best = {m: best_variant(sn, m, runs) for m in models}
            if any(best.values()):
                save(jq_figure(sn, runs, variants, labels, best), "jq")
        spec = [("w_lowz", "w", "low", r"$w_{\rm DE}(z)$", None, None, (-4, 1)),
                ("rho_lowz", "rho", "low", r"$\rho_{\rm DE}(z)/\rho_{\rm DE}(0)$", None, None, None),
                ("rho_full", "rho", "full", r"$\rho_{\rm DE}(z)/\rho_{\rm DE}(0)$", None, "symlog", None)]
        for name, key, grid, yl, delta, ys, yli in spec:
            if name in todo and planck_runs:
                save(line_figure(planck_runs, variants, labels, lcdm, key, grid, yl, bands, delta, ys, yli), name)
        for name, grid in (("H_z8", "mid"), ("H_full", "full")):
            if name in todo and planck_runs:
                save(line_figure(planck_runs, variants, labels, lcdm, "H", grid, r"$H(z)$ [km s$^{-1}$ Mpc$^{-1}$]", bands,
                                 delta=r"$H/H_{\Lambda{\rm CDM}}-1$ [%]", yscale="log" if grid == "full" else None, ratio=True), name)
        for name, key, yl, dl in (("h_lowz", "E", r"$h(z)=H/H_0$", r"$\Delta h$"), ("q_lowz", "q", r"$q(z)$", r"$\Delta q$"),
                                  ("j_lowz", "j", r"$j(z)$", r"$\Delta j$")):
            if name in todo:
                save(line_figure(runs, variants, labels, lcdm, key, "low", yl, bands, delta=dl), name)
        for name, grid in (("omega_lowz", "low"), ("omega_full", "full")):
            if name in todo and planck_runs:
                save(omega_figure(planck_runs, variants, labels, lcdm, grid), name)
        for kind in ("quintessence", "tachyon"):
            if f"ww_{kind}" in todo and planck_runs:
                save(ww_figure(planck_runs, variants, labels, kind), f"ww_{kind}")


if __name__ == "__main__":
    main()