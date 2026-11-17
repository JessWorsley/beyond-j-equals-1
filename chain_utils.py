"""Shared helpers for reading Cobaya chains and basic MCMC statistics (no Cobaya / getdist needed)."""
import glob
import re
import numpy as np
import pandas as pd


def read_chains(file_name, burn=0.0, folder="chains"):
    """Read the chain files of a Cobaya run.

    Parameters
    ----------
    file_name : str
        Chain prefix; files ``<folder>/<file_name>.<N>.txt`` are read.
    burn : float, optional
        Fraction of each chain dropped as burn-in.  Default is 0.
    folder : str, optional
        Directory of the chains.  Default is ``'chains'``.

    Returns
    -------
    chains : list of pandas.DataFrame
        One frame per chain file, burn-in removed.

    Raises
    ------
    FileNotFoundError
        If no chain files match.
    """
    paths = sorted(glob.glob(f"{folder}/{file_name}.[0-9]*.txt"))
    if not paths:
        raise FileNotFoundError(f"no chain files matching {folder}/{file_name}.N.txt")
    out = []
    for p in paths:
        with open(p) as f:
            cols = f.readline().lstrip("#").split()
        d = pd.read_csv(p, sep=r"\s+", comment="#", names=cols)
        out.append(d.iloc[int(burn * len(d)):].reset_index(drop=True))
    return out


def weighted_quantiles(values, weights, qs):
    """Compute weighted quantiles (inverted-CDF method).

    Parameters
    ----------
    values : array_like
        Samples.
    weights : array_like
        Sample weights.
    qs : array_like
        Quantiles in [0, 1].

    Returns
    -------
    q : numpy.ndarray
        The quantiles.
    """
    return np.quantile(values, qs, method="inverted_cdf", weights=weights)


def expand(chain, column):
    """Return the unweighted sequence of sampler steps for one column.

    Each row is repeated ``weight`` times, which restores the time series the
    autocorrelation estimates need.

    Parameters
    ----------
    chain : pandas.DataFrame
        One chain, with a ``weight`` column.
    column : str
        Column to expand.

    Returns
    -------
    x : numpy.ndarray
        The expanded series.
    """
    reps = np.rint(chain["weight"].values).astype(int)
    return np.repeat(chain[column].values, reps)


def autocorr_function(x):
    """Compute the normalised autocorrelation function of a 1-D series (FFT based).

    Parameters
    ----------
    x : array_like
        Series.

    Returns
    -------
    acf : numpy.ndarray
        ACF at lags 0 to len(x) - 1, with ``acf[0] == 1``.
    """
    x = np.asarray(x, dtype=float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    return acf / acf[0]


def integrated_autocorr_time(acf, c=5.0):
    """Estimate the integrated autocorrelation time with Sokal's automatic window.

    tau(M) = 1 + 2 sum_{t=1}^{M} acf(t); the window M is the smallest with
    M >= c tau(M).

    Parameters
    ----------
    acf : numpy.ndarray
        Normalised autocorrelation function.
    c : float, optional
        Window constant.  Default is 5.

    Returns
    -------
    tau : float
        IAT in sampler steps.
    """
    taus = 2.0 * np.cumsum(acf) - 1.0
    ok = np.arange(len(taus)) >= c * taus
    m = np.argmax(ok) if ok.any() else len(taus) - 1
    return taus[m]


def mean_acf(sequences):
    """Average the autocorrelation function over chains.

    Every chain is truncated to the length of the shortest one first.

    Parameters
    ----------
    sequences : list of array_like
        One series per chain.

    Returns
    -------
    acf : numpy.ndarray
        Mean ACF.
    """
    L = min(len(s) for s in sequences)
    return np.mean([autocorr_function(s[:L]) for s in sequences], axis=0)


def split_rhat(sequences):
    """Compute the split Gelman-Rubin R-hat.

    Every chain is cut in two halves, so a single chain also gives a
    diagnostic.

    Parameters
    ----------
    sequences : list of array_like
        One series per chain.

    Returns
    -------
    rhat : float
        Split R-hat (values near 1 mean converged; aim for < 1.01).
    """
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
    """Map each dataset to its chi2 column, whatever the likelihood class is called.

    Works for the closure and CPL class names (PantheonLikelihood,
    DESY5LikelihoodCPL, ...).  DESIFreeRd columns map to ``'DESI'``.

    Parameters
    ----------
    columns : iterable of str
        Chain column names.

    Returns
    -------
    names : dict
        Dataset name to column, e.g. ``{'DESI': col, 'Pantheon': col,
        'Planck': col}``.
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
    """Detect the SN dataset of a chain from its chi2 columns.

    Parameters
    ----------
    columns : iterable of str
        Chain column names.

    Returns
    -------
    sn : {'pantheon', 'desy5'}
        ``'desy5'`` if a DESY5 chi2 column is present, otherwise
        ``'pantheon'``.
    """
    names = chi2_columns(columns)
    return "desy5" if "DESY5" in names else "pantheon"


def apply_run_settings(prefix, runs_dir="runs"):
    """Set closure_cosmo's Z_MATCH options from the run's yaml file.

    This makes sure a chain is analysed with the late-time closure it was run
    with (make_runs.py writes the options into the yaml).  Old yaml files
    without the options, or a missing yaml, leave constants.py in charge.

    Parameters
    ----------
    prefix : str
        Chain prefix; ``<runs_dir>/<prefix>.yaml`` is read.
    runs_dir : str, optional
        Directory of the yaml files.  Default is ``'runs'``.

    Returns
    -------
    z_match : float or None
        The Z_MATCH now active in closure_cosmo.
    """
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
    """Return the chain prefix for a dataset/model combination.

    The prefix is the yaml ``output`` without ``'chains/'``.  make_runs.py,
    plot_chains.py, analyse_chains.py, compare_w.py and model_comparison.py all
    use this function, so the names always agree.

    Parameters
    ----------
    sn : {'pantheon', 'desy5'}
        SN dataset.
    model : str
        ``'model_1'``, ``'model_2'``, ``'model_3'``, ``'cpl'`` or ``'lcdm'``.
    tag : str, optional
        Suffix that keeps runs with extra assumptions apart, e.g. ``'_zm3'``
        or ``'_posdark'``.  Default is ``''``.

    Returns
    -------
    prefix : str
        One of::

            closure : desi_<sn>_planck_model_<k>    (e.g. desi_desy5_planck_model_1)
            cpl     : cpl_run (pantheon), cpl_<sn>_run
            lcdm    : lcdm_limit_baseline (pantheon), lcdm_limit_<sn>

        followed by `tag`.
    """
    if model == "cpl":
        base = "cpl_run" if sn == "pantheon" else f"cpl_{sn}_run"
    elif model == "lcdm":
        base = "lcdm_limit_baseline" if sn == "pantheon" else f"lcdm_limit_{sn}"
    else:
        base = f"desi_{sn}_planck_{model}"
    return base + tag


def infer_model(prefix, columns, fallback=None):
    """Work out a chain's model from its prefix and columns.

    Parameters
    ----------
    prefix : str
        Chain prefix.
    columns : iterable of str
        Chain column names.
    fallback : str, optional
        Returned if nothing matches (e.g. ``constants.MODEL``).

    Returns
    -------
    model : str or None
        ``'model_1'``, ``'model_2'``, ``'model_3'`` or ``'cpl'``.  LCDM-limit
        chains give ``'model_1'`` (every closure model has the same H(z)
        there).
    """
    m = re.search(r"model_[123]", prefix)
    if m:
        return m.group(0)
    if "w0" in columns or prefix.startswith("cpl"):
        return "cpl"
    if prefix.startswith("lcdm"):
        return "model_1"          # the LCDM limit has the same H(z) in every closure model
    return fallback