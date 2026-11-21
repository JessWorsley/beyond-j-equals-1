"""Shared helpers for reading Cobaya chains and basic MCMC statistics (no Cobaya / getdist needed)."""
import glob
import re
import numpy as np
import pandas as pd


# --no-planck runs (make_runs.py): H0 and Om0 are FIXED (they do not enter the likelihood except through H0*r_d) and r_d is sampled.
# Cobaya does not write fixed parameters to the chain, so read_chains puts them back as constant columns.
NOPLANCK_FIXED = {"H0": 67.7, "Om0": 0.31}


def is_fixed_noplanck(df):
    """True for a no-Planck chain with H0 and Om0 fixed (constant columns)."""
    return "ombh2" not in df.columns and "rd" in df.columns and "H0" in df.columns and float(df["H0"].max() - df["H0"].min()) == 0.0


def read_chains(file_name, burn=0.0, folder="chains"):
    """List of DataFrames, one per chain file, with the first `burn` fraction of each dropped."""
    paths = sorted(glob.glob(f"{folder}/{file_name}.[0-9]*.txt"))
    if not paths:
        raise FileNotFoundError(f"no chain files matching {folder}/{file_name}.N.txt")
    out = []
    for p in paths:
        with open(p) as f:
            cols = f.readline().lstrip("#").split()
        d = pd.read_csv(p, sep=r"\s+", comment="#", names=cols)
        if "rd" in d.columns and "ombh2" not in d.columns and "q0" in d.columns:      # no-Planck chain with fixed H0, Om0
            for k, v in NOPLANCK_FIXED.items():
                if k not in d.columns:
                    d[k] = v
        out.append(d.iloc[int(burn * len(d)):].reset_index(drop=True))
    return out


def weighted_quantiles(values, weights, qs):
    return np.quantile(values, qs, method="inverted_cdf", weights=weights)


def expand(chain, column):
    """Unweighted sequence of the sampler's steps: each row repeated `weight` times."""
    reps = np.rint(chain["weight"].values).astype(int)
    return np.repeat(chain[column].values, reps)


def autocorr_function(x):
    """Normalised autocorrelation function of a 1-D series (FFT based)."""
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf / acf[0]


def integrated_autocorr_time(acf, c=5.0):
    """Sokal automatic windowing: smallest M with M >= c * tau(M)."""
    taus = 2.0 * np.cumsum(acf) - 1.0
    ok = np.arange(len(taus)) >= c * taus
    m = np.argmax(ok) if ok.any() else len(taus) - 1
    return taus[m]


def mean_acf(sequences):
    """ACF averaged over chains, weighted by chain length (common length = shortest)."""
    L = min(len(s) for s in sequences)
    return np.mean([autocorr_function(s[:L]) for s in sequences], axis=0)


def split_rhat(sequences):
    """Split Gelman-Rubin R-hat: every chain is cut in two so a single chain also gives a diagnostic."""
    parts = []
    for s in sequences:
        h = len(s) // 2
        parts += [s[:h], s[h:2 * h]]
    L = min(len(p) for p in parts)
    parts = np.array([p[:L] for p in parts])
    W = parts.var(axis=1, ddof=1).mean()
    B = L * parts.mean(axis=1).var(ddof=1)
    return np.sqrt(((L - 1) / L * W + B / L) / W)


def chi2_columns(columns):
    """
    Map each dataset to its chi2 column, whatever the likelihood class is called
    (PantheonLikelihood, DESY5LikelihoodCPL, ...).  Returns e.g. {'DESI': col, 'Pantheon': col, 'Planck': col}.
    """
    out = {}
    for c in columns:
        if not c.startswith("chi2__"):
            continue
        base = c.split(".")[-1].replace("Likelihood", "").replace("CPL", "")
        if base.startswith("DESI"):
            base = "DESI"                    # also DESIFreeRd (the --no-planck runs)
        out[base] = c
    return out


def sn_dataset(columns):
    """'pantheon' or 'desy5', from which SN chi2 column is present in the chain (default pantheon)."""
    names = chi2_columns(columns)
    return "desy5" if "DESY5" in names else "pantheon"


def apply_run_settings(prefix, runs_dir="runs"):
    """Set closure_cosmo's Z_MATCH / Z_MATCH_ALLOW_NEGATIVE from runs/<prefix>.yaml, so a chain is analysed with the late-time
    closure it was run with (make_runs.py writes them into the yaml).  Old yaml files without the options, or no yaml at all,
    leave constants.py in charge.  Returns the z_match that is now active."""
    import os
    import yaml
    import closure_cosmo as cs
    path = os.path.join(runs_dir, f"{prefix}.yaml")
    try:
        with open(path) as f:
            like = yaml.safe_load(f).get("likelihood", {})
    except (OSError, yaml.YAMLError, AttributeError):
        return cs.Z_MATCH
    for block in like.values():
        if isinstance(block, dict) and "z_match" in block:
            cs.Z_MATCH = None if block["z_match"] is None else float(block["z_match"])
            cs.Z_MATCH_ALLOW_NEGATIVE = bool(block.get("z_match_allow_negative", False))
            print(f"run settings from {path}: Z_MATCH = {cs.Z_MATCH}")
            break
    return cs.Z_MATCH


def run_prefix(sn, model, tag=""):
    """
    Chain prefix (= yaml `output` without 'chains/') for a dataset/model combination.  Used by make_runs.py,
    plot_chains.py, analyse_chains.py, compare_w.py and model_comparison.py, so the names always agree.
        closure : desi_<sn>_planck_model_<k>        e.g. desi_pantheon_planck_model_3, desi_desy5_planck_model_1
        cpl     : cpl_run (pantheon), cpl_<sn>_run  e.g. cpl_desy5_run
        lcdm    : lcdm_limit_baseline (pantheon), lcdm_limit_<sn>
    `tag` is appended (e.g. '_zm3', '_posdark') to keep runs with different extra assumptions apart.
    """
    if model == "cpl":
        base = "cpl_run" if sn == "pantheon" else f"cpl_{sn}_run"
    elif model == "lcdm":
        base = "lcdm_limit_baseline" if sn == "pantheon" else f"lcdm_limit_{sn}"
    else:
        base = f"desi_{sn}_planck_{model}"
    return base + tag


def infer_model(prefix, columns, fallback=None):
    """Closure model ('model_1/2/3') or 'cpl' for a chain, from its prefix and columns (fallback: e.g. constants.MODEL)."""
    m = re.search(r"model_[123]", prefix)
    if m:
        return m.group(0)
    if "w0" in columns or prefix.startswith("cpl"):
        return "cpl"
    if prefix.startswith("lcdm"):
        return "model_1"          # the LCDM limit has the same H(z) in every closure model
    return fallback