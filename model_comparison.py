# """
# model_comparison.py -- chi^2_min, AIC, AICc, BIC and Akaike weights for several runs.

#     python model_comparison.py

# For each run the best-fit chi^2 is taken from the Cobaya `minimize` output (<prefix>.minimum.txt) if it
# exists, otherwise from the lowest-chi^2 sample in the chains (marked '*': biased high, so Delta-AIC/BIC
# are only indicative).  Run the minimizer (see minimize.yaml) with its own output prefix and give it as `minimum=` in RUNS.

#     AIC  = chi2 + 2k                 AICc = AIC + 2k(k+1)/(N-k-1)        BIC = chi2 + k ln N
#     ln B_(i,ref) ~ -(BIC_i - BIC_ref)/2   (crude Bayes-factor estimate; use nested sampling for evidences)
# k counts the sampled parameters plus 1 for the analytically marginalised SN offset M (it cancels in differences
# between models with the same data).  N = number of DESI numbers + SN + 3 Planck numbers.
# """
# import os
# import numpy as np
# import pandas as pd

# from chain_utils import read_chains

# BURN = 0.3
# REFERENCE = "LCDM-limit"                  # model the Delta's are measured from (name below); None = best AIC
# COUNT_M = True
# N_DATA = None                             # None: computed from the data loaders
# # name, chain prefix (= yaml `output` without 'chains/'), number of SAMPLED parameters, optional sn="pantheon"|"desy5"
# # (runs are only compared within the same SN dataset),
# # optional `minimum` = output prefix of the minimizer run (use a DIFFERENT prefix from the chains: `-f` deletes old output)
# RUNS = [
#     dict(name="LCDM-limit", prefix="lcdm_limit_baseline",       k=3),   # H0, Om0, ombh2   (j0 = 1, q0 = 1.5 Om0 - 1)
#     dict(name="Model 1",    prefix="desi_pantheon_planck_model_1", k=5),   # H0, Om0, q0, j0, ombh2
#     dict(name="Model 2",    prefix="desi_pantheon_planck_model_2",                k=5),
#     dict(name="Model 3",    prefix="desi_pantheon_planck_model_3",                k=5),
#     dict(name="CPL",        prefix="cpl_run",                   k=5),   # H0, Om0, w0, wa, ombh2
# ]


# def n_data(sn="pantheon"):
#     if N_DATA is not None:
#         return N_DATA
#     from load_data import load_desi, load_sn
#     return len(load_desi()[1]) + len(load_sn(sn)[0]) + 3


# def read_minimum(prefix, folder="chains"):
#     """chi2 (total and per likelihood) from <prefix>.minimum.txt, or None if unavailable/unparseable."""
#     path = f"{folder}/{prefix}.minimum.txt"
#     if not os.path.exists(path):
#         return None
#     try:
#         lines = [l for l in open(path).read().splitlines() if l.strip()]
#         head = next(l for l in lines if l.startswith("#")).lstrip("#").split()
#         vals = [l for l in lines if not l.startswith("#")][-1].split()
#         row = dict(zip(head, map(float, vals)))
#         return {k: v for k, v in row.items() if k.startswith("chi2")}
#     except Exception as e:                                              # noqa
#         print(f"  could not parse {path}: {e!r}")
#         return None


# def best_chi2(prefix, minimum=None):
#     m = read_minimum(minimum or prefix)
#     if m is not None and "chi2" in m:
#         return m, False
#     chains = read_chains(prefix, BURN)
#     df = pd.concat(chains, ignore_index=True)
#     row = df.loc[df["chi2"].idxmin()]
#     return {c: row[c] for c in df.columns if c.startswith("chi2")}, True


# def compare(runs=RUNS, N=None):
#     N = N or n_data()
#     rows = []
#     for r in runs:
#         try:
#             chi, from_chain = best_chi2(r["prefix"], r.get("minimum"))
#         except FileNotFoundError:
#             print("skipping (no chains / minimum):", r["name"]); continue
#         k = r["k"] + (1 if COUNT_M else 0)
#         c2 = chi["chi2"]
#         row = dict(model=r["name"] + ("*" if from_chain else ""), k=k, chi2=c2,
#                    AIC=c2 + 2 * k, AICc=c2 + 2 * k + 2 * k * (k + 1) / (N - k - 1), BIC=c2 + k * np.log(N))
#         for key, v in chi.items():
#             if key.startswith("chi2__"):
#                 row[key.split(".")[-1].replace("Likelihood", "").replace("CPL", "")] = v
#         rows.append(row)
#     tab = pd.DataFrame(rows).set_index("model")
#     ref = next((m for m in tab.index if m.rstrip("*") == REFERENCE), None) or tab["AIC"].idxmin()
#     for crit in ("AIC", "AICc", "BIC"):
#         tab["d" + crit] = tab[crit] - tab.loc[ref, crit]
#     w = np.exp(-0.5 * (tab["AIC"] - tab["AIC"].min()))
#     tab["Akaike_w"] = w / w.sum()
#     tab["lnB_vs_ref(BIC)"] = -0.5 * tab["dBIC"]
#     tab["dchi2"] = tab["chi2"] - tab.loc[ref, "chi2"]
#     return tab, ref, N


# def main():
#     # Runs are compared only within one supernova dataset: chi^2 values of different SN samples are not comparable.
#     by_sn = {}
#     for r in RUNS:
#         by_sn.setdefault(r.get("sn", "pantheon"), []).append(r)
#     pd.set_option("display.width", 200)
#     os.makedirs("chains/plots", exist_ok=True)
#     for sn, runs in by_sn.items():
#         tab, ref, N = compare(runs, n_data(sn))
#         show = ["k", "chi2", "dchi2", "dAIC", "dAICc", "dBIC", "Akaike_w", "lnB_vs_ref(BIC)"]
#         like = [c for c in tab.columns if c in ("DESI", "Pantheon", "DESY5", "Planck")]
#         print(f"\n=== DESI + {sn} + Planck:  N = {N} data points, reference model: {ref}"
#               "   (Delta = model - reference; negative favours the model)")
#         print(tab[show + like].round(2).to_string())
#         if any(i.endswith("*") for i in tab.index):
#             print("\n* chi2 from the lowest chain sample (not a true minimum). Run the minimizer before quoting these.")
#         tab.to_csv(f"chains/plots/model_comparison_{sn}.csv")
#         print(f"saved chains/plots/model_comparison_{sn}.csv")
#     print("""
# Rough guide (conventions, not laws):
#   Delta AIC : 0-2 no real preference, 4-7 clearly less support, >10 essentially none (Burnham & Anderson)
#   Delta BIC : 2-6 positive, 6-10 strong, >10 very strong evidence (Kass & Raftery); BIC penalises k more than AIC
#   Models with the same k differ by exactly Delta chi2 in AIC and BIC.  Compare with LCDM (smaller k) to see the Occam cost.""")


# if __name__ == "__main__":
#     main()

"""
model_comparison.py -- chi^2_min, AIC, AICc, BIC and Akaike weights for several runs.

    python model_comparison.py

For each run the best-fit chi^2 is taken from the Cobaya `minimize` output (<prefix>.minimum.txt) if it
exists, otherwise from the lowest-chi^2 sample in the chains (marked '*': biased high, so Delta-AIC/BIC
are only indicative).  Run the minimizer (see minimize.yaml) with its own output prefix and give it as `minimum=` in RUNS.

    AIC  = chi2 + 2k                 AICc = AIC + 2k(k+1)/(N-k-1)        BIC = chi2 + k ln N
    ln B_(i,ref) ~ -(BIC_i - BIC_ref)/2   (crude Bayes-factor estimate; use nested sampling for evidences)
k counts the sampled parameters plus 1 for the analytically marginalised SN offset M (it cancels in differences
between models with the same data).  N = number of DESI numbers + SN + 3 Planck numbers.
"""
import os
import numpy as np
import pandas as pd

from chain_utils import read_chains, run_prefix

BURN = 0.3
REFERENCE = "LCDM-limit"                  # model the Delta's are measured from (name below); None = best AIC
COUNT_M = True
N_DATA = None                             # None: computed from the data loaders
# name, chain prefix (= yaml `output` without 'chains/'), number of SAMPLED parameters, optional sn="pantheon"|"desy5"
# (runs are only compared within the same SN dataset),
# optional `minimum` = output prefix of the minimizer run (use a DIFFERENT prefix from the chains: `-f` deletes old output)
SN_DATASETS = ["pantheon", "desy5"]                 # runs are only compared within one SN dataset
_MODELS = [("LCDM-limit", "lcdm", 3),                 # H0, Om0, ombh2   (j0 = 1, q0 = 1.5 Om0 - 1)
           ("Model 1", "model_1", 5), ("Model 2", "model_2", 5), ("Model 3", "model_3", 5),   # H0, Om0, q0, j0, ombh2
           ("CPL", "cpl", 5)]                         # H0, Om0, w0, wa, ombh2
RUNS = [dict(name=name, prefix=run_prefix(sn, model), k=k, sn=sn) for sn in SN_DATASETS for name, model, k in _MODELS]
# Chains that do not exist are skipped.  A minimizer run with prefix <prefix>_min (make_runs.py --minimize) is used
# automatically for the chi^2 minimum.


def n_data(sn="pantheon"):
    """Return the number of data points for one SN dataset.

    Parameters
    ----------
    sn : str, optional
        SN dataset.  Default is ``'pantheon'``.

    Returns
    -------
    N : int
        DESI measurements + SNe after cuts + 3 Planck numbers, or ``N_DATA``
        if that is set.
    """
    if N_DATA is not None:
        return N_DATA
    from load_data import load_desi, load_sn
    return len(load_desi()[1]) + len(load_sn(sn)[0]) + 3


def read_minimum(prefix, folder="chains"):
    """Read chi^2 values from a Cobaya minimizer output.

    Parameters
    ----------
    prefix : str
        Output prefix; ``<folder>/<prefix>.minimum.txt`` is read.
    folder : str, optional
        Directory.  Default is ``'chains'``.

    Returns
    -------
    chi2 : dict or None
        Total and per-likelihood chi^2 (every column starting with ``chi2``),
        or None if the file is missing or cannot be parsed.
    """
    path = f"{folder}/{prefix}.minimum.txt"
    if not os.path.exists(path):
        return None
    try:
        lines = [l for l in open(path).read().splitlines() if l.strip()]
        head = next(l for l in lines if l.startswith("#")).lstrip("#").split()
        vals = [l for l in lines if not l.startswith("#")][-1].split()
        row = dict(zip(head, map(float, vals)))
        return {k: v for k, v in row.items() if k.startswith("chi2")}
    except Exception as e:                                              # noqa
        print(f"  could not parse {path}: {e!r}")
        return None


def best_chi2(prefix, minimum=None):
    """Return the best-fit chi^2 of a run.

    The minimizer output is tried first (``<minimum>``, else
    ``<prefix>_min``, else ``<prefix>``); otherwise the lowest-chi^2 chain
    sample is used.

    Parameters
    ----------
    prefix : str
        Chain prefix.
    minimum : str, optional
        Output prefix of the minimizer run, if different.

    Returns
    -------
    chi2 : dict
        Total and per-likelihood chi^2.
    from_chain : bool
        True if the values come from a chain sample (biased high).

    Raises
    ------
    FileNotFoundError
        If there is no minimum and no chain.
    """
    m = read_minimum(minimum or prefix + "_min") or read_minimum(prefix)
    if m is not None and "chi2" in m:
        return m, False
    chains = read_chains(prefix, BURN)
    df = pd.concat(chains, ignore_index=True)
    row = df.loc[df["chi2"].idxmin()]
    return {c: row[c] for c in df.columns if c.startswith("chi2")}, True


def compare(runs=RUNS, N=None):
    """Build the model-comparison table for runs on the same data.

    Parameters
    ----------
    runs : list of dict, optional
        Runs with keys ``name``, ``prefix``, ``k`` (number of sampled
        parameters) and optionally ``minimum``.  Default is ``RUNS``.
    N : int, optional
        Number of data points.  Default is ``n_data()``.

    Returns
    -------
    tab : pandas.DataFrame or None
        k, chi2, AIC, AICc, BIC, their differences from the reference, Akaike
        weights, the BIC Bayes-factor estimate and per-likelihood chi^2.  None
        if no run has chains.
    ref : str or None
        Reference model (``REFERENCE`` if present, else the lowest AIC).
    N : int
        Number of data points used.
    """
    N = N or n_data()
    rows = []
    for r in runs:
        try:
            chi, from_chain = best_chi2(r["prefix"], r.get("minimum"))
        except FileNotFoundError:
            print("skipping (no chains / minimum):", r["name"]); continue
        k = r["k"] + (1 if COUNT_M else 0)
        c2 = chi["chi2"]
        row = dict(model=r["name"] + ("*" if from_chain else ""), k=k, chi2=c2,
                   AIC=c2 + 2 * k, AICc=c2 + 2 * k + 2 * k * (k + 1) / (N - k - 1), BIC=c2 + k * np.log(N))
        for key, v in chi.items():
            if key.startswith("chi2__"):
                row[key.split(".")[-1].replace("Likelihood", "").replace("CPL", "")] = v
        rows.append(row)
    if not rows:
        return None, None, N
    tab = pd.DataFrame(rows).set_index("model")
    ref = next((m for m in tab.index if m.rstrip("*") == REFERENCE), None) or tab["AIC"].idxmin()
    for crit in ("AIC", "AICc", "BIC"):
        tab["d" + crit] = tab[crit] - tab.loc[ref, crit]
    w = np.exp(-0.5 * (tab["AIC"] - tab["AIC"].min()))
    tab["Akaike_w"] = w / w.sum()
    tab["lnB_vs_ref(BIC)"] = -0.5 * tab["dBIC"]
    tab["dchi2"] = tab["chi2"] - tab.loc[ref, "chi2"]
    return tab, ref, N


def main():
    """Print and save the model-comparison table for each SN dataset."""
    # Runs are compared only within one supernova dataset: chi^2 values of different SN samples are not comparable.
    by_sn = {}
    for r in RUNS:
        by_sn.setdefault(r.get("sn", "pantheon"), []).append(r)
    pd.set_option("display.width", 200)
    os.makedirs("chains/plots", exist_ok=True)
    for sn, runs in by_sn.items():
        tab, ref, N = compare(runs, n_data(sn))
        if tab is None:
            print(f"\n=== DESI + {sn} + Planck: no chains found, skipping")
            continue
        show = ["k", "chi2", "dchi2", "dAIC", "dAICc", "dBIC", "Akaike_w", "lnB_vs_ref(BIC)"]
        like = [c for c in tab.columns if c in ("DESI", "Pantheon", "DESY5", "Planck")]
        print(f"\n=== DESI + {sn} + Planck:  N = {N} data points, reference model: {ref}"
              "   (Delta = model - reference; negative favours the model)")
        print(tab[show + like].round(2).to_string())
        if any(i.endswith("*") for i in tab.index):
            print("\n* chi2 from the lowest chain sample (not a true minimum). Run the minimizer before quoting these.")
        tab.to_csv(f"chains/plots/model_comparison_{sn}.csv")
        print(f"saved chains/plots/model_comparison_{sn}.csv")
    print("""
Rough guide (conventions, not laws):
  Delta AIC : 0-2 no real preference, 4-7 clearly less support, >10 essentially none (Burnham & Anderson)
  Delta BIC : 2-6 positive, 6-10 strong, >10 very strong evidence (Kass & Raftery); BIC penalises k more than AIC
  Models with the same k differ by exactly Delta chi2 in AIC and BIC.  Compare with LCDM (smaller k) to see the Occam cost.""")


if __name__ == "__main__":
    main()