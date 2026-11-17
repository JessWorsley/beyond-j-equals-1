"""
make_runs.py -- write the Cobaya yaml file for every dataset / model combination.

    python make_runs.py                                  # all: pantheon+desy5, model_1/2/3, cpl, lcdm
    python make_runs.py --sn desy5 --models model_2 cpl
    python make_runs.py --tag _posdark --positive-dark   # extra assumption -> separate chain names
    python make_runs.py --fx-sigma 0.02 --tag _fx02

Files go to runs/ as <chain prefix>.yaml (and <chain prefix>_min.yaml for the minimizer with --minimize).  The chain
prefix follows chain_utils.run_prefix, which every analysis script uses, so you can analyse a run just by name:
    cobaya-run -f runs/desi_desy5_planck_model_2.yaml
    python analyse_chains.py desi_desy5_planck_model_2
    python plot_chains.py   desi_desy5_planck_model_2

The model is passed to the closure likelihoods through their `model:` option, so constants.MODEL does not need to be
edited.  (If your Cobaya rejects that option, delete the `model:` lines and set MODEL in constants.py instead.)
Z_MATCH is written into the yaml (--z-match 3.0 [--z-match-negative], or from constants.py); the radiation switches are NOT: put them in --tag.
For the whole set of runs and analyses see pipeline.py / pipeline.yaml.
"""
import argparse
import os
import yaml

from chain_utils import run_prefix

CLOSURE = ("model_1", "model_2", "model_3")


def closure_params():
    """Return the Cobaya ``params`` block of the closure-model runs.

    Returns
    -------
    params : dict
        Priors, reference distributions, proposal widths and LaTeX labels for
        H0, Om0, q0, j0 and ombh2.
    """
    return {
        "H0":    {"prior": {"min": 60, "max": 75}, "ref": {"dist": "norm", "loc": 67.5, "scale": 0.5}, "proposal": 0.3, "latex": "H_0"},
        "Om0":   {"prior": {"min": 0.2, "max": 0.4}, "ref": {"dist": "norm", "loc": 0.31, "scale": 0.005}, "proposal": 0.005, "latex": "\\Omega_{m0}"},
        "q0":    {"prior": {"min": -1.2, "max": -0.2}, "ref": {"dist": "norm", "loc": -0.535, "scale": 0.005}, "proposal": 0.005, "latex": "q_0"},
        "j0":    {"prior": {"min": 0.0, "max": 1.5}, "ref": {"dist": "norm", "loc": 1.0, "scale": 0.005}, "proposal": 0.005, "latex": "j_0"},
        "ombh2": {"prior": {"min": 0.02, "max": 0.025}, "ref": {"dist": "norm", "loc": 0.02237, "scale": 0.0001}, "proposal": 0.0001, "latex": "\\omega_bh^2"},
    }


def lcdm_params():
    """Return the Cobaya ``params`` block of the LCDM-limit runs.

    Returns
    -------
    params : dict
        H0, Om0 and ombh2 as in ``closure_params``, j0 fixed to 1 and
        q0 = 1.5 Om0 - 1 derived.
    """
    p = closure_params()
    return {"H0": p["H0"], "Om0": p["Om0"], "ombh2": p["ombh2"],
            "j0": 1.0,                                                         # fixed
            "q0": {"value": "lambda Om0: 1.5*Om0 - 1", "latex": "q_0"}}        # LCDM limit of the closures


def cpl_params():
    """Return the Cobaya ``params`` block of the flat w0waCDM (CPL) runs.

    Returns
    -------
    params : dict
        Priors, reference distributions, proposal widths and LaTeX labels for
        H0, Om0, w0, wa and ombh2.
    """
    return {
        "H0":    {"prior": {"min": 50, "max": 90}, "ref": {"dist": "norm", "loc": 67.5, "scale": 0.5}, "proposal": 0.5, "latex": "H_0"},
        "Om0":   {"prior": {"min": 0.1, "max": 0.6}, "ref": {"dist": "norm", "loc": 0.31, "scale": 0.01}, "proposal": 0.01, "latex": "\\Omega_{m0}"},
        "w0":    {"prior": {"min": -3, "max": 1}, "ref": {"dist": "norm", "loc": -0.9, "scale": 0.05}, "proposal": 0.05, "latex": "w_0"},
        "wa":    {"prior": {"min": -3, "max": 2}, "ref": {"dist": "norm", "loc": -0.2, "scale": 0.1}, "proposal": 0.1, "latex": "w_a"},
        "ombh2": {"prior": {"min": 0.02, "max": 0.025}, "ref": {"dist": "norm", "loc": 0.02237, "scale": 0.0001}, "proposal": 0.0001, "latex": "\\omega_bh^2"},
    }


_ZM = None          # set from --z-match / --z-match-negative; None = take the values from constants.py


def zmatch_options():
    """Return the late-time-closure options written into every likelihood block.

    They are always written, also as None, so the yaml alone defines the run
    and a later edit of constants.py cannot change an existing yaml.

    Returns
    -------
    opts : dict
        ``z_match`` and ``z_match_allow_negative``: from ``--z-match`` if it was
        given, otherwise from constants.py at generation time.
    """
    if _ZM is not None:
        return dict(_ZM)
    import constants
    return {"z_match": getattr(constants, "Z_MATCH", None),
            "z_match_allow_negative": bool(getattr(constants, "Z_MATCH_ALLOW_NEGATIVE", False))}


def effective_tag(tag, rd_fixed=None, no_planck=False):
    """Return the tag actually used in the chain prefix.

    make_runs.py appends ``_rdfix`` and ``_noplanck`` itself unless the tag
    already contains them.

    Parameters
    ----------
    tag : str
        Tag given with ``--tag``.
    rd_fixed : float or None, optional
        Value of ``--rd-fixed``.
    no_planck : bool, optional
        Whether ``--no-planck`` is set.

    Returns
    -------
    tag : str
        The tag with the automatic suffixes.
    """
    if rd_fixed is not None and "rdfix" not in tag:
        tag += "_rdfix"
    if no_planck and "noplanck" not in tag:
        tag += "_noplanck"
    return tag


def build(sn, model, tag, planck_opts, minimize, rd_fixed=None, best_of=None, no_planck=False):
    """Build the Cobaya input of one run.

    Parameters
    ----------
    sn : {'pantheon', 'desy5'}
        SN dataset.
    model : str
        ``'model_1'``, ``'model_2'``, ``'model_3'``, ``'cpl'`` or ``'lcdm'``.
    tag : str
        Suffix of the chain prefix (see ``chain_utils.run_prefix``).
    planck_opts : dict
        Extra PlanckLikelihood options (``positive_dark``, ``fx_sigma``,
        ``fx_max``).
    minimize : bool
        Write a minimizer run (output ``chains/<prefix>_min``) instead of an
        MCMC run.
    rd_fixed : float or None, optional
        Fix r_d in the BAO likelihood (Mpc), use a 2-D (R, lA) CMB prior and
        fix ombh2.  Default is None.
    best_of : int or None, optional
        Number of minimizer starting points (Cobaya ``best_of``).
    no_planck : bool, optional
        DESI + SN only, with r_d sampled and no CMB likelihood.  Default is
        False.

    Returns
    -------
    prefix : str
        Chain prefix.
    info : dict
        Cobaya input (likelihood, params, prior, sampler, output).
    """
    cpl = model == "cpl"
    sfx = "CPL" if cpl else ""
    if no_planck:                                  # DESI + SN only, r_d a free parameter, no CMB at all
        sn_cls = {"pantheon": "PantheonLikelihood", "desy5": "DESY5Likelihood"}[sn] + sfx
        opt = {"python_path": ".", **zmatch_options()}
        if not cpl:
            opt["model"] = "model_1" if model == "lcdm" else model
        params = {"cpl": cpl_params, "lcdm": lcdm_params}.get(model, closure_params)()
        params.pop("ombh2")
        params["rd"] = {"prior": {"min": 120.0, "max": 170.0}, "ref": {"dist": "norm", "loc": 147.0, "scale": 1.0},
                        "proposal": 1.0, "latex": "r_d"}
        out = {"likelihood": {f"likelihoods.DESIFreeRdLikelihood{sfx}": dict(opt), f"likelihoods.{sn_cls}": dict(opt)},
               "params": params}
        if cpl:
            out["prior"] = {"early_matter_domination": "lambda w0, wa: 0 if w0 + wa < 0 else -float('inf')"}
        prefix = run_prefix(sn, model, tag)
        if minimize:
            out["sampler"] = {"minimize": {"method": "scipy", **({"best_of": best_of} if best_of else {})}}
            out["output"] = f"chains/{prefix}_min"
        else:
            out["sampler"] = {"mcmc": {"max_tries": 10000, "Rminus1_stop": 0.01}}
            out["output"] = f"chains/{prefix}"
        return prefix, out
    sn_cls = {"pantheon": "PantheonLikelihood", "desy5": "DESY5Likelihood"}[sn] + sfx
    opt = {"python_path": ".", **zmatch_options()}
    if not cpl:
        opt["model"] = "model_1" if model == "lcdm" else model       # LCDM limit: any closure model gives the same H(z)
    desi_opt, planck_extra = dict(opt), dict(planck_opts)
    if rd_fixed is not None:                                          # DESI-style cross-check: r_d fixed, 2D (R, lA) prior
        desi_opt["rd_fixed"] = rd_fixed
        planck_extra["use_omb"] = False
    likelihood = {f"likelihoods.DESILikelihood{sfx}": desi_opt,
                  f"likelihoods.{sn_cls}": dict(opt),
                  f"likelihoods.PlanckLikelihood{sfx}": {**opt, **planck_extra}}
    params = {"cpl": cpl_params, "lcdm": lcdm_params}.get(model, closure_params)()
    if rd_fixed is not None:
        params["ombh2"] = 0.02237                                     # fixed: only enters r_s(z*) in l_A
    params["rd"] = {"latex": "r_d"}
    params["fX_star"] = {"latex": "f_X(z_*)"}
    out = {"likelihood": likelihood, "params": params}
    if cpl:
        out["prior"] = {"early_matter_domination": "lambda w0, wa: 0 if w0 + wa < 0 else -float('inf')"}   # as in DESI
    prefix = run_prefix(sn, model, tag)
    if minimize:
        out["sampler"] = {"minimize": {"method": "scipy", **({"best_of": best_of} if best_of else {})}}
        out["output"] = f"chains/{prefix}_min"
    else:
        out["sampler"] = {"mcmc": {"max_tries": 10000, "Rminus1_stop": 0.01}}
        out["output"] = f"chains/{prefix}"
    return prefix, out


def main():
    """Parse the command line and write the yaml files (see the module docstring)."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sn", nargs="+", default=["pantheon", "desy5"], choices=["pantheon", "desy5"])
    ap.add_argument("--models", nargs="+", default=list(CLOSURE) + ["cpl", "lcdm"], choices=list(CLOSURE) + ["cpl", "lcdm"])
    ap.add_argument("--tag", default="", help="suffix for the chain prefix, e.g. _zm3 or _posdark")
    ap.add_argument("--positive-dark", action="store_true", help="PlanckLikelihood option: reject rho_DE < 0 for z <= z*")
    ap.add_argument("--fx-sigma", type=float, default=None, help="PlanckLikelihood option: Gaussian penalty on f_X(z*)")
    ap.add_argument("--fx-max", type=float, default=None, help="PlanckLikelihood option: hard cut |f_X(z*)| < value")
    ap.add_argument("--minimize", action="store_true", help="also write <prefix>_min.yaml (minimizer) for the chi2 minimum")
    ap.add_argument("--rd-fixed", type=float, default=None,
                    help="cross-check: fix r_d [Mpc] in the BAO likelihood (e.g. 147.2), use a 2D (R, lA) CMB prior, fix omega_b; tag _rdfix")
    ap.add_argument("--best-of", type=int, default=None,
                    help="minimizer: number of random starting points (Cobaya `best_of`, default 2); use ~8 if the minima disagree")
    ap.add_argument("--no-planck", action="store_true",
                    help="DESI BAO + SN only, r_d sampled (flat 120-170 Mpc), no CMB likelihood; tag _noplanck")
    ap.add_argument("--z-match", default="constants",
                    help="late-time closure: a redshift (e.g. 3.0), or 'none' for the full closure; default 'constants' = take "
                         "Z_MATCH from constants.py.  Written into the yaml, so constants.py need not be edited")
    ap.add_argument("--z-match-negative", action="store_true",
                    help="with --z-match: allow a negative frozen dark density beyond z_match (sign-switching)")
    ap.add_argument("--outdir", default="runs")
    a = ap.parse_args()
    global _ZM
    if a.z_match != "constants":
        _ZM = {"z_match": None if a.z_match.lower() in ("none", "null") else float(a.z_match),
               "z_match_allow_negative": bool(a.z_match_negative)}
    a.tag = effective_tag(a.tag, a.rd_fixed, a.no_planck)

    zm = zmatch_options()
    if zm["z_match"] is not None:
        print(f"NOTE: Z_MATCH = {zm['z_match']} (allow negative: {zm['z_match_allow_negative']}) is written into these yaml files"
              f" ({'from --z-match' if _ZM is not None else 'from constants.py'}).")
        if "zm" not in a.tag:
            print("  WARNING: the tag does not contain 'zm', so these runs would overwrite/mix with normal chains of the "
                  "same name.  Use e.g. --tag _zm3 (and _zm3pos when negative densities are not allowed).")
    elif "zm" in a.tag:
        print("  WARNING: the tag contains 'zm' but Z_MATCH is None: these are NOT z-match runs.")

    planck_opts = {}
    if a.positive_dark:
        planck_opts["positive_dark"] = True
    if a.fx_sigma is not None:
        planck_opts["fx_sigma"] = a.fx_sigma
    if a.fx_max is not None:
        planck_opts["fx_max"] = a.fx_max

    os.makedirs(a.outdir, exist_ok=True)
    cmds = []
    for sn in a.sn:
        for model in a.models:
            for mini in ([False, True] if a.minimize else [False]):
                prefix, cfg = build(sn, model, a.tag, planck_opts, mini, a.rd_fixed, a.best_of, a.no_planck)
                path = os.path.join(a.outdir, f"{prefix}{'_min' if mini else ''}.yaml")
                with open(path, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)
                cmds.append(f"cobaya-run -f {path}")
    print(f"wrote {len(cmds)} yaml files to {a.outdir}/.  Run them with e.g.:")
    for c in cmds:
        print("  ", c)
    print("\nthen analyse with:  python analyse_chains.py <prefix>   and   python plot_chains.py <prefix>")


if __name__ == "__main__":
    main()