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
import sys
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
    if N_DATA is not None:
        return N_DATA
    from load_data import load_desi, load_sn
    return len(load_desi()[1]) + len(load_sn(sn)[0]) + 3


def read_minimum(prefix, folder="chains"):
    """chi2 (total and per likelihood) from <prefix>.minimum.txt, or None if unavailable/unparseable."""
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
    m = read_minimum(minimum or prefix + "_min") or read_minimum(prefix)
    if m is not None and "chi2" in m:
        return m, False
    chains = read_chains(prefix, BURN)
    df = pd.concat(chains, ignore_index=True)
    row = df.loc[df["chi2"].idxmin()]
    return {c: row[c] for c in df.columns if c.startswith("chi2")}, True


def compare(runs=RUNS, N=None):
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


def _tag_from_argv():
    """--tag _zm3pos : use the tagged chains for the closure models (LCDM-limit and CPL have no Z_MATCH and keep their own)."""
    a = sys.argv[1:]
    return a[a.index("--tag") + 1] if "--tag" in a else ""


def main():
    # Runs are compared only within one supernova dataset: chi^2 values of different SN samples are not comparable.
    tag = _tag_from_argv()
    global RUNS
    if tag:
        RUNS = [dict(name=name + f" [{tag}]" if model.startswith("model") else name, prefix=run_prefix(sn, model, tag if model.startswith("model") else ""),
                     k=k, sn=sn) for sn in SN_DATASETS for name, model, k in _MODELS]
    print(f"model_comparison: closure-model chains with tag {tag!r}" if tag else "model_comparison: untagged (base) chains")
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
        tab.to_csv(f"chains/plots/model_comparison_{sn}{tag}.csv")
        print(f"saved chains/plots/model_comparison_{sn}{tag}.csv")
    print("""
Rough guide (conventions, not laws):
  Delta AIC : 0-2 no real preference, 4-7 clearly less support, >10 essentially none (Burnham & Anderson)
  Delta BIC : 2-6 positive, 6-10 strong, >10 very strong evidence (Kass & Raftery); BIC penalises k more than AIC
  Models with the same k differ by exactly Delta chi2 in AIC and BIC.  Compare with LCDM (smaller k) to see the Occam cost.""")


if __name__ == "__main__":
    main()