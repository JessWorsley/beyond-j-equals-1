"""
End-to-end check of likelihoods.py WITHOUT your data files or Cobaya:
mock data are generated from the LCDM-limit model, so each chi2 must be ~0 at the truth
and grow when parameters move.  (Uses a stub Likelihood base class if cobaya is not installed.)
"""
import sys, types
import numpy as np

try:
    import cobaya  # noqa
except ModuleNotFoundError:
    mod = types.ModuleType("cobaya"); lk = types.ModuleType("cobaya.likelihood")
    class Likelihood:
        """Stand-in for ``cobaya.likelihood.Likelihood`` when Cobaya is not installed."""
        def __init__(self):
            """Call ``initialize``, as Cobaya does."""
            self.initialize()
    lk.Likelihood = Likelihood; mod.likelihood = lk
    sys.modules.update({"cobaya": mod, "cobaya.likelihood": lk})

import closure_cosmo as cs
import likelihoods as L
from calculate_distances import luminosity_distance, bao_distances
from bao import bao_model_vector
from constants import RD_PLANCK

# The test must not depend on constants.MODEL (which may be 'cpl' while you analyse a CPL run):
# force the closure classes onto a closure model before they are instantiated.
TEST_MODEL = L.MODEL if L.MODEL in cs.MODELS else "model_1"
for _c in (L.PantheonLikelihood, L.DESILikelihood, L.PlanckLikelihood):
    _c.model = TEST_MODEL
print(f"(testing closure model {TEST_MODEL}; constants.MODEL = {L.MODEL})")

H0, wb, wm = 67.36, 0.02237, 0.1430
Om = wm / (H0 / 100) ** 2
truth = dict(H0=H0, q0=1.5 * Om - 1, j0=1.0, Om0=Om)
theta = [truth[k] for k in ("H0", "q0", "j0", "Om0")]
rng = np.random.default_rng(1)

# ---- mock Pantheon (with an arbitrary magnitude offset that must be marginalised away)
n = 80
z_hd = np.sort(rng.uniform(0.012, 2.2, n)); z_hel = z_hd * 1.0002
DL = luminosity_distance(theta, z_hd, z_hel, TEST_MODEL)
sig = 0.1; cov = np.diag(np.full(n, sig ** 2)) + 0.01 ** 2          # include a correlated piece
from load_data import marginalise_over_M
cov, icov = marginalise_over_M(n, None, cov, reshape=False)
mu_obs = 5 * np.log10(DL) + 25 + 0.37
L.load_pantheon = lambda: (z_hd, z_hel, mu_obs, cov, icov)

# ---- mock DESI
zb = np.array([0.295, 0.51, 0.51, 0.93, 0.93, 1.32, 1.32, 2.33])
kinds = ["DV_over_rs", "DM_over_rs", "DH_over_rs", "DM_over_rs", "DH_over_rs",
         "DM_over_rs", "DH_over_rs", "DV_over_rs"]
q = cs.cmb_quantities(theta, wb, TEST_MODEL)
rd = cs.calibrate(q, dict(rd=RD_PLANCK))["rd"]
means = bao_model_vector(theta, zb, kinds, TEST_MODEL, rd)
dcov = np.diag((0.01 * means) ** 2)
L.load_desi = lambda: (zb, means, dcov, np.linalg.inv(dcov), kinds)

sn, desi, planck = L.PantheonLikelihood(), L.DESILikelihood(), L.PlanckLikelihood()
ok = True
def check(name, cond, info=""):
    """Print PASS or FAIL for one test and record a failure in the global ``ok``.

    Parameters
    ----------
    name : str
        Test description.
    cond : bool
        Whether the test passed.
    info : str, optional
        Extra text printed after the name.
    """
    global ok; ok &= bool(cond); print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")

d = {}
l_sn = sn.logp(**truth); l_de = desi.logp(**truth, ombh2=wb, _derived=d); l_pl = planck.logp(**truth, ombh2=wb, _derived=d)
check("Pantheon chi2 ~ 0 at truth (offset marginalised)", abs(l_sn) < 1e-6, f"({l_sn:.2e})")
check("DESI chi2 ~ 0 at truth", abs(l_de) < 1e-6, f"({l_de:.2e})")
check("Planck chi2 ~ 0 at fiducial (only |d omega_b| ~ 2e-5)", abs(l_pl) < 0.05, f"({l_pl:.2e})")
check("derived rd = 147.09 and fX_star ~ 0", abs(d['rd'] - RD_PLANCK) < 1e-6 and abs(d['fX_star']) < 1e-6)

moved = dict(truth, q0=truth['q0'] + 0.05)
check("likelihoods get worse when q0 moves", sn.logp(**moved) < l_sn - 1 and
      desi.logp(**moved, ombh2=wb) < l_de - 1 and planck.logp(**moved, ombh2=wb) < l_pl - 10)
# a point where the background is singular/undefined for THIS model (the singular point differs per model)
bad = dict(truth, **{"model_1": dict(q0=-1.0), "model_2": dict(j0=-1.0), "model_3": dict(q0=0.5)}[TEST_MODEL])
check("invalid point returns -inf (no exception)",
      sn.logp(**bad) == -np.inf and desi.logp(**bad, ombh2=wb) == -np.inf and planck.logp(**bad, ombh2=wb) == -np.inf)
d2 = {}
desi.logp(**bad, ombh2=wb, _derived=d2)
check("derived rd defined (nan) on rejected point", 'rd' in d2)

# ---- positive_dark option: reject effective negative dark-sector density (q0 below the LCDM ridge 1.5 Om0 - 1, at j0 = 1) but keep the truth
planck.positive_dark = True
check("positive_dark keeps the LCDM-limit point", planck.logp(**truth, ombh2=wb) > -1)
check("positive_dark rejects q0 below the ridge (rho_DE < 0 at high z)", planck.logp(**dict(truth, q0=truth['q0'] - 0.03), ombh2=wb) == -np.inf)
check("positive_dark accepts q0 above the ridge", np.isfinite(planck.logp(**dict(truth, q0=truth['q0'] + 0.03), ombh2=wb)))
planck.positive_dark = False

# ---- CPL classes: (w0, wa) = (-1, 0) must give exactly the same numbers as the closure LCDM limit
snc, desic, planckc = L.PantheonLikelihoodCPL(), L.DESILikelihoodCPL(), L.PlanckLikelihoodCPL()
cp = dict(H0=H0, w0=-1.0, wa=0.0, Om0=Om)
check("CPL(-1,0) reproduces closure LCDM limit in all three likelihoods",
      abs(snc.logp(**cp) - l_sn) < 1e-8 and abs(desic.logp(**cp, ombh2=wb) - l_de) < 1e-8
      and abs(planckc.logp(**cp, ombh2=wb) - l_pl) < 1e-8)
check("CPL gets worse when w0 moves", snc.logp(**dict(cp, w0=-0.8)) < -1 and desic.logp(**dict(cp, w0=-0.8), ombh2=wb) < -1)
# ---- DES-SN5YR classes: same model code, so with the same (mock) data they must match the Pantheon classes exactly
L.load_desy5 = (lambda: (z_hd, z_hel, mu_obs, cov, icov))
desy, desyc = L.DESY5Likelihood(), L.DESY5LikelihoodCPL()
check("DESY5Likelihood matches PantheonLikelihood on identical data", abs(desy.logp(**truth) - l_sn) < 1e-10)
check("DESY5LikelihoodCPL matches PantheonLikelihoodCPL", abs(desyc.logp(**cp) - snc.logp(**cp)) < 1e-10)

# ---- chain helpers: chi2 columns are found whatever the likelihood class is called
from chain_utils import chi2_columns, sn_dataset
closure_cols = ["chi2", "chi2__likelihoods.DESILikelihood", "chi2__likelihoods.PantheonLikelihood", "chi2__likelihoods.PlanckLikelihood"]
cpl_desy_cols = ["chi2", "chi2__likelihoods.DESILikelihoodCPL", "chi2__likelihoods.DESY5LikelihoodCPL", "chi2__likelihoods.PlanckLikelihoodCPL"]
check("chi2_columns: closure class names", set(chi2_columns(closure_cols)) == {"DESI", "Pantheon", "Planck"})
check("chi2_columns: CPL + DESY5 class names", set(chi2_columns(cpl_desy_cols)) == {"DESI", "DESY5", "Planck"})
check("sn_dataset detects the SN sample", sn_dataset(closure_cols) == "pantheon" and sn_dataset(cpl_desy_cols) == "desy5")

print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")