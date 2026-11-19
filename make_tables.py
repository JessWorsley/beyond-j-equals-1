"""
make_tables.py -- LaTeX tables (booktabs) from the csv files written by compare_variants.py and model_comparison.py.

    python make_tables.py                                   # pantheon + desy5, models 1-3, every variant found
    python make_tables.py --sn pantheon --models model_3
    python make_tables.py --variants "base (with Planck)" "z match at 3 (positive)" "no Planck"
    python make_tables.py --comparison-tags "" _zm3pos      # model_comparison csv tags ("" = base chains)
    python make_tables.py --out latex/tables.tex

Reads   chains/variants_<model>_<sn>.csv            (compare_variants.py)
        chains/plots/model_comparison_<sn><tag>.csv (model_comparison.py [--tag T])
Writes  one .tex file with, per supernova sample:  fit quality of the variants, parameter constraints, negative-dark-density
probabilities, and the model comparison against LCDM/CPL.  Needs \\usepackage{booktabs} in the paper.  Missing csv files are skipped.
A dagger marks a chi^2 taken from the lowest chain sample (an upper bound on chi^2_min, minimiser not converged or not run).
"""
import argparse
import math
import os
import re

import numpy as np
import pandas as pd

SN_NAME = {"pantheon": r"Pantheon+", "desy5": r"DES-SN5YR"}
MODEL_NAME = {"model_1": "Model 1", "model_2": "Model 2", "model_3": "Model 3"}
PM = re.compile(r"^\s*([+-]?\d+\.?\d*)\s*\(\s*\+([\d.]+)\s+-([\d.]+)\s*\)\s*$")


# ------------------------------------------------------------------------------------------------------- formatting
def esc(s):
    return s.replace("\\", " ").replace("&", r"\&").replace("%", r"\%").replace("_", r"\_").replace("#", r"\#")


def label(idx):
    """csv index -> LaTeX label; a trailing + or * (chain-sample chi^2) becomes a dagger."""
    dag = idx.endswith(("+", "*"))
    s = idx.rstrip("+*").strip()
    m = re.fullmatch(r"z match at ([\d.]+)( \(positive\))?", s)
    if m:
        s = rf"$z_m={m.group(1)}$" + (" (positive)" if m.group(2) else "")
    else:
        s = esc(s)
    return s + (r"$^\dagger$" if dag else "")


def num(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "--"
    t = f"{x:.{nd}f}"
    return f"${t}$" if t.startswith("-") else t               # proper minus sign


def pm(cell, nd=None):
    """'67.387 (+0.634 -0.662)' -> $67.39^{+0.63}_{-0.66}$ with two significant digits on the smaller error."""
    if not isinstance(cell, str):
        return "--"
    m = PM.match(cell)
    if not m:
        return "--"
    v, up, lo = (float(m.group(i)) for i in (1, 2, 3))
    errs = [e for e in (up, lo) if e > 0]
    d = max(0, 1 - int(math.floor(math.log10(min(errs))))) if errs else 3          # two significant digits of the smaller error
    while nd is None and d > 0 and all(abs(round(e, d - 1) - e) < 0.03 * e for e in errs):
        d -= 1                                                                    # drop a trailing 0 (0.0060 -> 0.006)
    d = min(d, 4) if nd is None else nd
    return f"${v:.{d}f}^{{+{up:.{d}f}}}_{{-{lo:.{d}f}}}$"


def prob(x):
    return "--" if not np.isfinite(x) else (r"$<0.001$" if 0 < x < 0.001 else f"{x:.3f}")


def table(header, rows, caption, tag, colspec=None, blocks=None):
    """rows: list of lists of strings; blocks: {row index: block title} inserts a midrule + a title line before that row."""
    ncol = len(header)
    colspec = colspec or ("l" + "c" * (ncol - 1))
    out = ["\\begin{table}[ht]", "\\centering", f"\\caption{{{caption}}}", f"\\label{{{tag}}}", "\\small",
           f"\\begin{{tabular}}{{{colspec}}}", "\\toprule", " & ".join(header) + r" \\", "\\midrule"]
    for i, r in enumerate(rows):
        if blocks and i in blocks:
            if i:
                out.append("\\midrule")
            out.append(f"\\multicolumn{{{ncol}}}{{l}}{{\\emph{{{blocks[i]}}}}} \\\\")
        out.append(" & ".join(r) + r" \\")
    out += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------- tables
def keep(df, variants):
    if not variants:
        return df
    sel = [i for i in df.index if any(v == i.rstrip("+*").strip() or v.lower() in i.lower() for v in variants)]
    return df.loc[sel]


def load_variants(sn, model, variants):
    path = f"chains/variants_{model}_{sn}.csv"
    if not os.path.exists(path):
        print("  skip (not found):", path)
        return None
    return keep(pd.read_csv(path, index_col=0), variants)


def tables_for_sn(sn, models, variants):
    data = {m: load_variants(sn, m, variants) for m in models}
    data = {m: d for m, d in data.items() if d is not None and len(d)}
    if not data:
        return ""
    sn_txt = SN_NAME.get(sn, sn)
    out = []

    # 1 -- fit quality
    rows, blocks = [], {}
    for m, d in data.items():
        blocks[len(rows)] = MODEL_NAME.get(m, m)
        base = d["chi2"].iloc[0]
        has_planck = d["chi2_Planck"].notna()
        for i, r in d.iterrows():
            comparable = bool(has_planck[i])
            rows.append([label(i), num(r["chi2_DESI"]), num(r["chi2_SN"]), num(r["chi2_Planck"]), num(r["chi2"]),
                         num(r["chi2"] - base) if comparable else "--"])
    out.append(table(["Variant", r"$\chi^2_{\rm DESI}$", r"$\chi^2_{\rm SN}$", r"$\chi^2_{\rm Planck}$", r"$\chi^2_{\rm tot}$",
                      r"$\Delta\chi^2$"], rows,
                     f"Best-fit $\\chi^2$ of the analysis variants, DESI DR2 BAO + {sn_txt} (+ Planck distance priors). "
                     r"$\Delta\chi^2$ is relative to the first variant of each block; all variants have the same number of "
                     r"parameters, so it equals $\Delta$AIC and $\Delta$BIC. The no-Planck fit has no Planck data and is not "
                     r"comparable in total $\chi^2$. $^\dagger$: lowest chain sample (upper bound on $\chi^2_{\min}$).",
                     f"tab:variants_{sn}", blocks=blocks))

    # 2 -- parameters
    cols = [("H0", r"$H_0$"), ("Om0", r"$\Omega_{m0}$"), ("q0", r"$q_0$"), ("j0", r"$j_0$"), ("rd", r"$r_d$\,[Mpc]")]
    rows, blocks = [], {}
    for m, d in data.items():
        blocks[len(rows)] = MODEL_NAME.get(m, m)
        for i, r in d.iterrows():
            rows.append([label(i)] + [pm(r[c]) if c in d.columns else "--" for c, _ in cols])
    out.append(table(["Variant"] + [t for _, t in cols], rows,
                     f"Posterior medians and 68\\% intervals, DESI DR2 BAO + {sn_txt} (+ Planck distance priors). "
                     r"In the no-Planck fits $H_0$ and $r_d$ are degenerate and only $H_0 r_d$ is constrained.",
                     f"tab:params_{sn}", blocks=blocks))

    # 3 -- negative dark density
    rows, blocks = [], {}
    for m, d in data.items():
        blocks[len(rows)] = MODEL_NAME.get(m, m)
        for i, r in d.iterrows():
            fx = pm(r["fX_star"], nd=3) if "fX_star" in d.columns else "--"
            rows.append([label(i), prob(r["P(rho_DE<0 inside data)"]), prob(r["P(rho_DE<0 only beyond)"]), fx])
    out.append(table(["Variant", r"$P(\tilde\rho_{\rm DE}<0,\;z\le2.5)$", r"$P(\tilde\rho_{\rm DE}<0,\;z>2.5\text{ only})$",
                      r"$f_X(z_*)$"], rows,
                     f"Posterior probability of a negative effective dark density inside the data range and only beyond it, "
                     f"and the dark fraction at recombination ({sn_txt}).",
                     f"tab:negdark_{sn}", blocks=blocks))
    return "\n".join(out)


def comparison_table(sn, tag):
    path = f"chains/plots/model_comparison_{sn}{tag}.csv"
    if not os.path.exists(path):
        print("  skip (not found):", path)
        return ""
    d = pd.read_csv(path, index_col=0)
    like = [c for c in ("DESI", "Pantheon", "DESY5", "Planck") if c in d.columns]
    rows = []
    for i, r in d.iterrows():
        name = re.sub(r"\s*\[.*\]", "", i).rstrip("*").strip()
        name = esc(name) + (r"$^\dagger$" if i.endswith("*") else "")
        rows.append([name, f"{int(r['k'])}", num(r["chi2"]), num(r["dchi2"]), num(r["dAIC"]), num(r["dBIC"]),
                     num(r["Akaike_w"])] + [num(r[c]) for c in like])
    names = {"DESI": r"$\chi^2_{\rm DESI}$", "Pantheon": r"$\chi^2_{\rm SN}$", "DESY5": r"$\chi^2_{\rm SN}$", "Planck": r"$\chi^2_{\rm Planck}$"}
    extra = f", closure models with the late-time closure (tag \\texttt{{{esc(tag)}}})" if tag else ", closure models without a matching redshift"
    return table(["Model", "$k$", r"$\chi^2$", r"$\Delta\chi^2$", r"$\Delta$AIC", r"$\Delta$BIC", "$w_{\\rm AIC}$"] + [names[c] for c in like],
                 rows,
                 f"Model comparison relative to the $\\Lambda$CDM limit, DESI DR2 BAO + {SN_NAME.get(sn, sn)} + Planck distance priors{extra}. "
                 r"$k$ includes the marginalised magnitude offset. $^\dagger$: lowest chain sample.",
                 f"tab:modelcmp_{sn}{tag}", colspec="l" + "c" * (6 + len(like)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sn", nargs="+", default=["pantheon", "desy5"])
    ap.add_argument("--models", nargs="+", default=["model_1", "model_2", "model_3"])
    ap.add_argument("--variants", nargs="*", default=None, help="only these variants (label or part of it)")
    ap.add_argument("--comparison-tags", nargs="*", default=["", "_zm3pos"])
    ap.add_argument("--out", default="latex/tables.tex")
    a = ap.parse_args()
    parts = ["% generated by make_tables.py -- needs \\usepackage{booktabs}", ""]
    for sn in a.sn:
        print(sn)
        parts.append(tables_for_sn(sn, a.models, a.variants))
        for tag in a.comparison_tags:
            parts.append(comparison_table(sn, tag))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(p for p in parts if p))
    print("wrote", a.out)


if __name__ == "__main__":
    main()