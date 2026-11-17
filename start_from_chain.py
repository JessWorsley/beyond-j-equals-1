"""
start_from_chain.py -- make a minimizer yaml start at the best sample of an existing MCMC chain.

    python start_from_chain.py desi_pantheon_planck_model_3_posdark

Reads chains/<prefix>.N.txt, takes the sample with the lowest chi2, and writes runs/<prefix>_min_start.yaml: a copy of
runs/<prefix>_min.yaml in which every sampled parameter has  ref = Normal(best value, 0.1 x posterior std).
The output prefix is unchanged (chains/<prefix>_min), so compare_variants.py and model_comparison.py pick the result up.

    cobaya-run -f runs/<prefix>_min_start.yaml

Why: a hard cut such as positive_dark makes the likelihood a wall with the optimum on its edge, and a minimizer started
from random points can stall far from it (e.g. chi2 = 1440 when the chain already contains a valid sample at 1420).
"""
import sys
import numpy as np
import pandas as pd
import yaml

from chain_utils import read_chains

BURN = 0.3
SCALE = 0.1                                   # start spread, in units of the posterior std


def main(prefix, runs_dir="runs"):
    """Write a minimizer yaml that starts at the best sample of an MCMC chain.

    Every sampled parameter gets ``ref = Normal(best value, SCALE x
    posterior std)``.  The output is ``<runs_dir>/<prefix>_min_start.yaml``.

    Parameters
    ----------
    prefix : str
        Chain prefix; ``<runs_dir>/<prefix>_min.yaml`` must exist.
    runs_dir : str, optional
        Directory of the yaml files.  Default is ``'runs'``.
    """
    df = pd.concat(read_chains(prefix, BURN), ignore_index=True)
    col = "chi2" if "chi2" in df.columns else "minuslogpost"
    best = df.loc[df[col].idxmin()]
    src = f"{runs_dir}/{prefix}_min.yaml"
    with open(src) as f:
        info = yaml.safe_load(f)
    w = df["weight"].values
    changed = []
    for name, spec in info["params"].items():
        if not isinstance(spec, dict) or "prior" not in spec or name not in df.columns:
            continue
        mean = np.average(df[name], weights=w)
        std = float(np.sqrt(np.average((df[name] - mean) ** 2, weights=w)))
        spec["ref"] = {"dist": "norm", "loc": float(best[name]), "scale": max(SCALE * std, 1e-8)}
        changed.append(f"{name} = {best[name]:.5g}")
    dst = f"{runs_dir}/{prefix}_min_start.yaml"
    with open(dst, "w") as f:
        yaml.safe_dump(info, f, sort_keys=False)
    print(f"best chain sample ({col} = {best[col]:.3f}):  " + ", ".join(changed))
    print(f"wrote {dst}   (output prefix: {info.get('output')})")
    print(f"run:  cobaya-run -f {dst}")


if __name__ == "__main__":
    main(sys.argv[1])