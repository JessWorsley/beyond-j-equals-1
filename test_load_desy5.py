"""Check load_desy5 on synthetic files in the DES format (no real data needed).   python test_load_desy5.py"""
import os, tempfile
import numpy as np, pandas as pd
from load_data import load_desy5, marginalise_over_M

rng = np.random.default_rng(3)
n = 60
A = rng.normal(size=(n, n)); cov_true = A @ A.T * 1e-3 + np.diag(rng.uniform(0.01, 0.03, n))   # SPD
inv_true = np.linalg.inv(cov_true)
z = np.sort(rng.uniform(0.02, 1.1, n))
mu_err = np.full(n, 0.1); bad = [5, 17, 40]; mu_err[bad] = 2.0                                   # outliers to be dropped
df = pd.DataFrame(dict(CID=np.arange(n), zHD=z, zHEL=z * 1.0002, MU=38 + 5 * np.log10(z), MUERR=mu_err))

tmp = tempfile.mkdtemp()
csv, npz = os.path.join(tmp, "hd.csv"), os.path.join(tmp, "inv.npz")
df.to_csv(csv, index=False)
np.savez(npz, cov=inv_true[np.triu_indices(n)], nsn=np.array([n]))

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

# 1. no outliers dropped: marginalised inverse must equal the direct calculation from the full covariance
zh, zl, mu, cov, icov = load_desy5(csv, npz, mu_err_max=10.0)
cov_ref, icov_ref = marginalise_over_M(n, None, cov_true, reshape=False)
check("full sample: covariance recovered from the stored inverse", np.allclose(cov, cov_ref, rtol=1e-8, atol=1e-12))
check("full sample: marginalised inverse covariance", np.allclose(icov, icov_ref, rtol=1e-6, atol=1e-9))
check("marginalised inverse removes the offset direction (C^-1 . 1 = 0)", np.allclose(icov @ np.ones(n), 0, atol=1e-8))

# 2. outliers dropped: the sub-block must be taken from the COVARIANCE
zh, zl, mu, cov, icov = load_desy5(csv, npz)
keep = np.ones(n, bool); keep[bad] = False
cov_sub, icov_sub = marginalise_over_M(keep.sum(), None, cov_true[np.ix_(keep, keep)], reshape=False)
check("outliers dropped: correct (covariance sub-block) treatment", np.allclose(icov, icov_sub, rtol=1e-6, atol=1e-9) and len(mu) == keep.sum())
wrong = np.linalg.inv(inv_true[np.ix_(keep, keep)])          # what sub-selecting the INVERSE would give
check("sub-selecting the inverse would give a different covariance (why the order matters)",
      not np.allclose(wrong, cov_true[np.ix_(keep, keep)], rtol=1e-3), f"(max rel diff {np.max(np.abs(wrong / cov_true[np.ix_(keep, keep)] - 1)):.2f})")
check("zHEL and zHD both returned", np.allclose(zl, z[keep] * 1.0002) and np.allclose(zh, z[keep]))

# 3. size mismatch is caught
np.savez(npz, cov=inv_true[np.triu_indices(n)], nsn=np.array([n + 1]))
try:
    load_desy5(csv, npz); check("size mismatch raises", False)
except ValueError:
    check("size mismatch raises", True)
print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")