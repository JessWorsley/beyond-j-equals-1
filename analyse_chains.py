import sys
import numpy as np
import pandas as pd

from constants import MODEL
from inferred import calculate_inferred_parameters
from load_data import load_desi, load_sn
from chain_utils import read_chains, chi2_columns, sn_dataset, run_prefix, infer_model

# usage: python analyse_chains.py <chain_prefix>   e.g.  desi_desy5_planck_model_2,  cpl_run,  lcdm_limit_baseline
# The model (model_1/2/3 or cpl) and the SN dataset are read from the prefix and the chain, not from constants.MODEL.
file_name = sys.argv[1] if len(sys.argv) > 1 else run_prefix('pantheon', MODEL)
plot_name = f'{file_name}_corner'
BURN = 0.3                                    # fraction of each chain discarded as burn-in

# ---------------------------------------------------------------- read all chains, drop burn-in
frames = read_chains(file_name, BURN)
df = pd.concat(frames, ignore_index=True)
model = infer_model(file_name, df.columns, fallback=MODEL)
if model == 'cpl':                            # flat w0waCDM: run with cpl.yaml (parameters named w0, wa)
    sampled = ["H0", "Om0", "ombh2", "w0", "wa"]
    reported = sampled
elif 'j0' in df.columns:                      # closure models: q0 and j0 are sampled directly
    sampled = ["H0", "Om0", "ombh2", "q0", "j0"]
    reported = sampled
else:                                         # LCDM-limit baseline (lcdm yaml): j0 = 1 fixed, q0 = 1.5 Om0 - 1 derived
    print("NOTE: no j0 column -> LCDM-limit baseline run (3 sampled parameters)")
    sampled = ["H0", "Om0", "ombh2"]
    reported = sampled + ["q0"]
noplanck = "ombh2" not in df.columns and "rd" in df.columns     # make_runs.py --no-planck: r_d sampled, no CMB
if noplanck:
    print("NOTE: no-Planck chain (r_d sampled, no omega_b, no CMB likelihood)")
    sampled = ["rd" if p == "ombh2" else p for p in sampled]
    reported = ["rd" if p == "ombh2" else p for p in reported]
extra = [e for e in ["rd", "fX_star"] if e not in reported]    # derived outputs (if present in the chain)

legacy_cpl = False
if model == 'cpl' and 'w0' not in df.columns and {'q0', 'j0'} <= set(df.columns):
    # chain made with the closure likelihood classes: the q0 slot held w0 and the j0 slot held wa
    print("NOTE: legacy CPL chain (columns q0, j0) -> treating q0 as w0 and j0 as wa")
    df['w0'], df['wa'] = df['q0'], df['j0']
    legacy_cpl = True
w = df["weight"].values
print(f"MODEL {model[-1] if model.startswith('model') else model}:  {len(frames)} chains, {len(df)} rows after burn-in, total weight {w.sum():.0f}")


results = {}                                  # name -> (lo, med, hi): 16th, 50th, 84th weighted percentiles


def summary(name, values):
    """Print the weighted 16th/50th/84th percentiles of a quantity and store them.

    Uses the global weights ``w``; the result goes into the global
    ``results[name]``.

    Parameters
    ----------
    name : str
        Name of the quantity.
    values : array_like
        One value per chain row.
    """
    lo, med, hi = np.quantile(values, [0.16, 0.50, 0.84], method='inverted_cdf', weights=w)
    results[name] = (lo, med, hi)
    print(f"{name}: {med:.5f} +{hi - med:.5f} -{med - lo:.5f}")


print("\nWEIGHTED PARAMETER RESULTS")
for p in reported + [e for e in extra if e in df.columns]:
    summary(p, df[p].values)
if model == 'cpl':
    summary("w0+wa", (df['w0'] + df['wa']).values)                     # DESI requires w0 + wa < 0
    summary("w(z=0.5)", (df['w0'] + df['wa'] * 0.5 / 1.5).values)
    summary("w(z=1)", (df['w0'] + df['wa'] * 0.5).values)
elif 'j0' in df.columns:
    for key, val in calculate_inferred_parameters(df['q0'], df['j0'], df['Om0'], model).items():
        summary(key, np.asarray(val))

# one row, three columns per parameter: <name>_lo, <name>_med, <name>_hi  (16th / 50th / 84th percentiles)
csv_path = f"chains/{file_name}_results.csv"
pd.DataFrame({f"{k}_{s}": [v[i]] for k, v in results.items() for i, s in enumerate(("lo", "med", "hi"))}
             ).to_csv(csv_path, index=False, float_format="%.8g")
print("saved", csv_path)

# ---------------------------------------------------------------- chi squared
N_desi = len(load_desi()[1])                  # number of measurements, not redshift bins
sn_name = sn_dataset(df.columns)               # 'pantheon' or 'desy5', from the chain's chi2 columns
sn_label = 'DESY5' if sn_name == 'desy5' else 'Pantheon'
N_sn = len(load_sn(sn_name)[0])               # after the data cuts
N_planck = 0 if noplanck else 3               # R, lA, omega_b
N_data = N_desi + N_sn + N_planck
N_free = len(sampled) + 1                     # + 1 for the analytically marginalised SN offset M
dof = N_data - N_free

best = df.loc[df["chi2"].idxmin()]
print("\nCHI SQUARED RESULTS (best sample; run `minimize` for a proper best fit)")
print(f"Best-fit chi2: {best['chi2']:.3f}   dof: {dof}   reduced: {best['chi2'] / dof:.4f}")
chi_cols = chi2_columns(df.columns)           # works for the closure and the *CPL likelihood class names
for like, n in (("DESI", N_desi), (sn_label, N_sn)) + (() if noplanck else (("Planck", N_planck),)):
    print(f"  {like:9s} chi2 = {best[chi_cols[like]]:9.3f}   (N = {n})")
print("Best-fit parameters:")
print(best[reported + [e for e in extra if e in df.columns]])

# ---------------------------------------------------------------- plotting
from getdist import loadMCSamples, plots
samples = loadMCSamples(f"chains/{file_name}", settings={'ignore_rows': BURN})
g = plots.get_subplot_plotter()
plot_names = ["H0", "Om0", "ombh2", "q0", "j0"] if legacy_cpl else sampled      # legacy: getdist sees q0, j0
g.triangle_plot(samples, plot_names + (["fX_star"] if "fX_star" in df.columns else []),
                filled=True, title_limit=1)
g.export(f"chains/{plot_name}.pdf")