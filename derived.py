"""
derived.py -- epsilon, w_DE(z=0) and w'_DE(z=0) per posterior sample, from the formulas in inferred.py.

  eps    closure constant of each model:  j = 1 + 3 eps (q+1) [model 1],  j = 1 + eps [model 2],  j = 1 + 3 eps (q-1/2) [model 3]
         (eps = 0 is LCDM; it depends on q0 and j0 only, so it is also defined without Planck)
  wDE0   w_DE(z=0) = (1-2 q0) / (3 (Om0-1))
  wDE0p  w'_DE(z=0) = dw_DE/dln a  at z=0, the convention of the Caldwell-Linder w-w' plane.
         NB inferred.py's `wDE0p` is dw_DE/dln(1+z) at z=0 (checked numerically against the closure H(z)), i.e. the NEGATIVE of this;
         WPRIME_SIGN = -1 converts.  Set it to +1 to use inferred.py's sign.
wDE0 and wDE0p need Om0, which only the Planck-included runs constrain (see README: Om0 does not enter H(z)/H0).
"""
import numpy as np

from inferred import calculate_inferred_parameters

WPRIME_SIGN = -1.0


def derived_samples(df, model, with_om0=True):
    """dict name -> array over the rows of df.  wDE0 / wDE0p only if with_om0 (a constrained Om0)."""
    q0, j0, Om0 = df["q0"].values, df["j0"].values, df["Om0"].values
    with np.errstate(invalid="ignore", divide="ignore"):
        d = calculate_inferred_parameters(q0, j0, Om0, model)
    out = {"eps": np.asarray(d["eps"], dtype=float)}
    if with_om0:
        out["wDE0"] = np.asarray(d["wDE0"], dtype=float)
        out["wDE0p"] = WPRIME_SIGN * np.asarray(d["wDE0p"], dtype=float)
    return out