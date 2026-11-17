"""
Diagnose the Planck compressed-prior likelihood.  Run from your project folder:   python debug_planck.py
Paste the whole output back if something looks wrong.
"""
import numpy as np
import constants, closure_cosmo as cs, load_data

print("modules in use (should all be in your project folder):")
for m in (constants, cs, load_data):
    print("  ", m.__name__, "->", m.__file__)
print("MODEL =", constants.MODEL, "| RADIATION late/early =", constants.INCLUDE_RADIATION_LATE, constants.INCLUDE_RADIATION_EARLY)
print("FIDUCIAL =", constants.FIDUCIAL, "| RD_PLANCK =", constants.RD_PLANCK, "| Z_RAD_MAX =", constants.Z_RAD_MAX)

mean, cov, fid = load_data.planck_distance_priors()
print("\nplanck_distance_priors():")
print("   mean [R, lA, omega_b] =", mean)
print("   sigma                 =", np.sqrt(np.diag(cov)))
print("   fid (calibration refs)=", fid)
print("   cov eigenvalues       =", np.linalg.eigvalsh(cov), "(all must be > 0)")

H0, wb, wm = 67.36, 0.02237, 0.1430
Om = wm / (H0 / 100) ** 2
theta = [H0, 1.5 * Om - 1, 1.0, Om]
print("\ntest point (the LCDM-limit fiducial): theta =", np.round(theta, 5), " omega_b =", wb)

if constants.MODEL not in cs.MODELS:
    print(f"\nNOTE: constants.MODEL = {constants.MODEL!r} is not a closure model; diagnosing with 'model_1'. "
          "(Closure likelihood classes now refuse to run with this setting.)")
MODEL = constants.MODEL if constants.MODEL in cs.MODELS else "model_1"
for model in ("model_1", MODEL):
    q = cs.cmb_quantities(theta, wb, model)
    print(f"\nraw cmb_quantities [{model}]:", {k: round(float(v), 4) for k, v in q.items()})
print("\nfiducial_code_values():", {k: round(float(v), 4) for k, v in cs.fiducial_code_values().items()})
q = cs.cmb_quantities(theta, wb, MODEL)
c = cs.calibrate(q, fid)
print("calibrated [R, lA, rd]  :", {k: round(float(v), 4) for k, v in c.items()})
d = np.array([c["R"], c["lA"], wb]) - mean
inv = np.linalg.inv(cov)
print("d = pred - mean         :", d)
print("chi2 (full)             :", d @ inv @ d)
print("chi2 per-term (diag)    :", (d / np.sqrt(np.diag(cov))) ** 2)