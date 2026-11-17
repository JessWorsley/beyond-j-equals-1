"""Run with:  python test_cosmology.py"""
import numpy as np
from scipy.integrate import quad
import closure_cosmo as cs

H0, wb, wm = 67.36, 0.02237, 0.1430            # Planck-2018-like LCDM point
h = H0 / 100
Om = wm / h ** 2
q0 = 1.5 * Om - 1.0                             # LCDM limit of the closures
j0 = 1.0
theta = [H0, q0, j0, Om]
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
    global ok
    ok &= bool(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name} {info}")


# 1. LCDM limit: all three closures must reproduce Om(1+z)^3 + 1 - Om
z = np.array([0, 0.3, 1, 5, 50, 1100.0])
lcdm = H0 * np.sqrt(Om * (1 + z) ** 3 + 1 - Om)
for m in cs.MODELS:
    Hm, Om_z = cs.background_model(theta, z, m)
    check(f"{m} -> LCDM limit", np.allclose(Hm, lcdm, rtol=1e-10))

# 2. Om(z) against the paper's hand-derived expressions
zz = np.array([0.0, 0.5, 2.0, 10.0])
q0b, j0b, Om0b = -0.45, 1.1, 0.3
th = [70.0, q0b, j0b, Om0b]
s = np.sqrt(8 * j0b + 1)
p = (j0b + 3 * q0b + 2) / (q0b + 1)
Om1_paper = Om0b * (zz + 1) ** 3 * (j0b + 3 * q0b + 2) / (
    (q0b + 1) ** 2 * (2 * (zz + 1) ** p + (j0b - q0b * (2 * q0b + 1)) / (q0b + 1) ** 2))
Om2_paper = Om0b * (zz + 1) ** (0.5 * (s + 3)) / (
    (s + 4 * q0b + 1) * ((zz + 1) ** s - 1) / (2 * s) + 1)
beta = 2 * (j0b - 1) / (2 * q0b - 1)
Om3_paper = -Om0b * (2 * j0b - 6 * q0b + 1) / (
    -(1 - 2 * q0b) ** 2 * (zz + 1) ** (beta - 3) - 2 * j0b + 2 * q0b * (2 * q0b + 1))
for m, ref in zip(cs.MODELS, (Om1_paper, Om2_paper, Om3_paper)):
    got = cs.background_model(th, zz, m)[1]
    check(f"{m} Om(z) matches paper", np.allclose(got, ref, rtol=1e-10))

# 3. Compare with an independent LCDM calculation using scipy.quad
Or = cs.omega_r_density(h)
E = lambda zq: np.sqrt(Om * (1 + zq) ** 3 + Or * (1 + zq) ** 4 + (1 - Om))
Eb = lambda zq: np.sqrt(Om * (1 + zq) ** 3 + Or * (1 + zq) ** 4 + (1 - Om) )
zs, zd = cs.z_star(wb, wm), cs.z_drag(wb, wm)
DM_ref = cs.C_KMS / H0 * quad(lambda zq: 1 / E(zq), 0, zs, epsabs=0, epsrel=1e-12)[0]
Rb = lambda zq: 31500 * wb * (cs.TCMB / 2.7) ** -4 / (1 + zq)
rs_ref = lambda zl: cs.C_KMS / H0 * quad(
    lambda x: np.exp(x) / np.sqrt(3 * (1 + Rb(np.expm1(x)))) / E(np.expm1(x)),
    np.log1p(zl), np.log1p(cs.Z_RAD_MAX), epsabs=0, epsrel=1e-12, limit=500, points=[np.log(3000.)])[0]
for m in cs.MODELS:
    q = cs.cmb_quantities(theta, wb, m)
    check(f"{m} D_M(z*) vs quad", abs(q["DM_star"] / DM_ref - 1) < 1e-6,
          f"(rel diff {q['DM_star']/DM_ref-1:.1e})")
    check(f"{m} r_s(z*) vs quad", abs(q["rs_star"] / rs_ref(zs) - 1) < 1e-6,
          f"(rel diff {q['rs_star']/rs_ref(zs)-1:.1e})")
    check(f"{m} r_d vs quad", abs(q["rd"] / rs_ref(zd) - 1) < 1e-6,
          f"(rel diff {q['rd']/rs_ref(zd)-1:.1e})")

# 4. Absolute accuracy against Planck-2018 numbers (approximate reference values)
q = cs.cmb_quantities(theta, wb, "model_1")
print("\nRaw LCDM-limit output at the Planck-like point:")
for k in ("R", "lA", "rd", "zstar", "zdrag", "fX_star", "fX_min"):
    print(f"  {k:8s} = {q[k]:.5f}")
check("R ~ 1.750 (0.1%)", abs(q["R"] / 1.750 - 1) < 1e-3)
# Isolate integral accuracy from fitting-formula accuracy: evaluate at the redshifts Planck 2018
# itself quotes for base LCDM (z* = 1089.92, z_drag = 1059.94; Planck 2018 VI, Table 2, which were
# computed with CAMB's recombination history).  Any mismatch here is then NOT due to the
# Hu-Sugiyama / Eisenstein-Hu fits.
th_ = theta
rs_d = cs.r_sound(1059.94, th_, "model_1", wb)
rs_s = cs.r_sound(1089.92, th_, "model_1", wb)
DM_s = cs.D_M(1089.92, th_, "model_1")
check("r_s(z_drag=1059.94) ~ 147.09 Mpc (0.2%)", abs(rs_d / 147.09 - 1) < 2e-3, f"({rs_d:.2f})")
check("pi*D_M/r_s at z*=1089.92 ~ pi/theta* (0.1%)",
      abs(np.pi * DM_s / rs_s / (np.pi / 1.04110e-2) - 1) < 1e-3, f"({np.pi*DM_s/rs_s:.2f})")
print("  NOTE: raw rd and lA differ from Planck because Hu-Sugiyama z* and Eisenstein-Hu z_d are")
print("        not accurate enough; this is removed by calibrate() below.")
cal = cs.calibrate(q, dict(rd=147.09, lA=np.pi / 1.04110e-2))
check("calibrate() reproduces reference at the fiducial point",
      abs(cal["rd"] - 147.09) < 1e-9 and abs(cal["lA"] - np.pi / 1.04110e-2) < 1e-9)
check("f_X(z*) ~ 0 in LCDM limit", abs(q["fX_star"]) < 1e-6)

# 5. A non-LCDM point: j0 != 1 should show a sizeable dark fraction at z*
th2 = [H0, -0.55, 0.8, 0.31]
for m in cs.MODELS:
    try:
        q2 = cs.cmb_quantities(th2, wb, m)
        print(f"{m}: j0=0.8  R={q2['R']:.3f} lA={q2['lA']:.2f} rd={q2['rd']:.2f} fX*={q2['fX_star']:.3f}")
    except cs.CosmoError as e:
        print(f"{m}: j0=0.8 rejected ({e})")


# ---------------------------------------------------------------------------
# 6. Distances used by the SN and BAO likelihoods (late-time setting, match quad)
# ---------------------------------------------------------------------------
from calculate_distances import bao_distances, luminosity_distance
Or_late = cs.radiation_density(h, cs.INCLUDE_RADIATION_LATE)       # 0 by default: SN/BAO use the closure alone
Ez = lambda zq: np.sqrt(Om * (1 + zq) ** 3 + Or_late * (1 + zq) ** 4 + (1 - Om))
chi_ref = lambda zq: cs.C_KMS / H0 * quad(lambda x: 1 / Ez(x), 0, zq, epsrel=1e-12)[0]
zb = np.array([0.295, 0.51, 0.93, 1.32, 2.33])
for m in cs.MODELS:
    DM, DH, DV = bao_distances(theta, zb, m)
    DM_ref = np.array([chi_ref(x) for x in zb])
    DH_ref = cs.C_KMS / (H0 * Ez(zb))
    DV_ref = (zb * DM_ref ** 2 * DH_ref) ** (1 / 3)
    check(f"{m} bao_distances vs quad",
          np.allclose(DM, DM_ref, rtol=1e-5) and np.allclose(DH, DH_ref, rtol=1e-5)
          and np.allclose(DV, DV_ref, rtol=1e-5))
zh = np.array([0.012, 0.1, 0.5, 1.0, 2.2])
zhel = zh * 1.0003
DL = luminosity_distance(theta, zh, zhel, "model_1")
DL_ref = (1 + zhel) * np.array([chi_ref(x) for x in zh])
check("luminosity_distance = (1+z_hel) chi(z_hd) vs quad", np.allclose(DL, DL_ref, rtol=1e-5))


# ---------------------------------------------------------------------------
# 7. Do the closure H(z) formulas really obey the closure relations they are named for?
#    q and j are recovered NUMERICALLY from H(z) (5-point finite differences in x = ln(1+z)):
#       q = 1/2 dlnE^2/dx - 1,   j = q + 2 q^2 + 1/2 d^2 lnE^2/dx^2
# ---------------------------------------------------------------------------
def q_j_numeric(th, model, z, h=1e-3):
    """Compute q(z) and j(z) of a model by finite differences of ln E^2 in x = ln(1+z).

    Parameters
    ----------
    th : sequence of float
        ``[H0, q0, j0, Om0]``.
    model : str
        Closure model.
    z : numpy.ndarray
        Redshifts.
    h : float, optional
        Step in x.  Default is 1e-3.

    Returns
    -------
    q, j : numpy.ndarray
        Deceleration and jerk parameters.
    """
    xs = np.log1p(z)[None, :] + h * np.arange(-2, 3)[:, None]
    L = np.array([np.log(cs.background_model(th, np.expm1(x), model)[0] ** 2 / th[0] ** 2) for x in xs])
    d1 = (L[0] - 8 * L[1] + 8 * L[3] - L[4]) / (12 * h)
    d2 = (-L[0] + 16 * L[1] - 30 * L[2] + 16 * L[3] - L[4]) / (12 * h * h)
    q = 0.5 * d1 - 1
    return q, q + 2 * q ** 2 + 0.5 * d2

zt = np.array([0.0, 0.3, 1.0, 3.0, 10.0])
th = [70.0, -0.45, 1.1, 0.3]
q0t, j0t, Om0t = th[1], th[2], th[3]
eps1 = (j0t - 1) / (3 * (q0t + 1))
eps3 = 2 * (1 - j0t) / (3 * (1 - 2 * q0t))
for m in cs.MODELS:
    qz, jz = q_j_numeric(th, m, zt)
    check(f"{m}: q(0)=q0 and j(0)=j0", abs(qz[0] - q0t) < 1e-6 and abs(jz[0] - j0t) < 1e-6)
    if m == "model_1":
        rel = jz - (1 + 3 * eps1 * (qz + 1))            # j = 1 + 3 eps (q + 1)
    elif m == "model_2":
        rel = jz - j0t                                    # j = const
    else:
        rel = jz - (1 + 3 * eps3 * (qz - 0.5))          # j = 1 + 3 eps (q - 1/2)
    check(f"{m}: closure relation holds at all z", np.max(np.abs(rel)) < 1e-5, f"(max {np.max(np.abs(rel)):.1e})")

# effective dark energy: rho_DE = E^2 - Om0 (1+z)^3,  w = -1 + 1/3 dln(rho_DE)/dx
def w_de_numeric(th, model, z, h=1e-3):
    """Compute the effective w_DE(z) = -1 + 1/3 dln(rho_DE)/dx by finite differences.

    Parameters
    ----------
    th : sequence of float
        ``[H0, q0, j0, Om0]``.
    model : str
        Closure model.
    z : numpy.ndarray
        Redshifts.
    h : float, optional
        Step in x = ln(1+z).  Default is 1e-3.

    Returns
    -------
    w : numpy.ndarray
        w_DE(z).
    """
    xs = np.log1p(z)[None, :] + h * np.arange(-2, 3)[:, None]
    L = np.array([np.log(cs.background_model(th, np.expm1(x), model)[0] ** 2 / th[0] ** 2
                         - th[3] * np.exp(3 * x)) for x in xs])
    return -1 + (L[0] - 8 * L[1] + 8 * L[3] - L[4]) / (12 * h) / 3

for m in cs.MODELS:
    w = w_de_numeric(th, m, zt)
    check(f"{m}: w_DE(0) = (1-2q0)/(3(Om0-1))", abs(w[0] - (1 - 2 * q0t) / (3 * (Om0t - 1))) < 1e-6)
c1_3 = 2 * (j0t - q0t - 2 * q0t ** 2) / (1 - 6 * q0t + 2 * j0t)
w3_paper = -(1 - c1_3) * (1 - eps3) / ((1 - c1_3) + (c1_3 - Om0t) * (1 + zt) ** (3 - 3 * eps3))
check("model_3: w_DE(z) matches the paper's closed form", np.allclose(w_de_numeric(th, "model_3", zt), w3_paper, atol=1e-5))


# ---------------------------------------------------------------------------
# 8. Radiation: closure applied to the non-radiation sector, radiation added separately
#    E^2 = E_X^2 + R,  R = Or e^{4x},  f_r = R/E^2.  Derived relations (see notes):
#       (i)  j = 1 + 1/2 sum_i n_i (n_i - 3) f_i      => LCDM + radiation has j = 1 + 2 f_r  (not 1!)
#       (ii) q = q_X + f_r (1 - q_X)
#            j = q + 2 q^2 + (1 - f_r)(j_X - q_X - 2 q_X^2) + 2 f_r (1 - q)(1 - q_X)
# ---------------------------------------------------------------------------
def qj_from_L(Lf, x, h=1e-3):
    """Compute q and j from a function L(x) = ln E^2 by finite differences.

    Parameters
    ----------
    Lf : callable
        ln E^2 as a function of x = ln(1+z).
    x : numpy.ndarray
        Points to evaluate at.
    h : float, optional
        Step.  Default is 1e-3.

    Returns
    -------
    q, j : numpy.ndarray
        Deceleration and jerk parameters.
    """
    L = np.array([Lf(x + k * h) for k in range(-2, 3)])
    d1 = (L[0] - 8 * L[1] + 8 * L[3] - L[4]) / (12 * h)
    d2 = (-L[0] + 16 * L[1] - 30 * L[2] + 16 * L[3] - L[4]) / (12 * h * h)
    q = 0.5 * d1 - 1
    return q, q + 2 * q ** 2 + 0.5 * d2

Or8 = cs.omega_r_density(0.7)
x8 = np.log1p(np.array([0.0, 1.0, 100.0, 1090.0, 3000.0]))
E2_lcdm = lambda x: 0.3 * np.exp(3 * x) + 0.7 + Or8 * np.exp(4 * x)
_, j_num = qj_from_L(lambda x: np.log(E2_lcdm(x)), x8)
f_r = Or8 * np.exp(4 * x8) / E2_lcdm(x8)
check("LCDM + radiation has j = 1 + 2 f_r", np.allclose(j_num, 1 + 2 * f_r, atol=1e-6), f"(j(z*) = {j_num[3]:.3f})")

th8 = [70.0, -0.45, 1.1, 0.3]
for m in cs.MODELS:
    Ecl2 = lambda x: (cs.background_model(th8, np.expm1(x), m)[0] / th8[0]) ** 2
    Etot2 = lambda x: Ecl2(x) + Or8 * np.exp(4 * x)
    qX, jX = qj_from_L(lambda x: np.log(Ecl2(x)), x8)
    qT, jT = qj_from_L(lambda x: np.log(Etot2(x)), x8)
    fr = Or8 * np.exp(4 * x8) / Etot2(x8)
    q_formula = qX + fr * (1 - qX)
    j_formula = qT + 2 * qT ** 2 + (1 - fr) * (jX - qX - 2 * qX ** 2) + 2 * fr * (1 - qT) * (1 - qX)
    check(f"{m}: q_total from q_X and f_r", np.allclose(qT, q_formula, atol=1e-6))
    check(f"{m}: j_total from (q_X, j_X, f_r)", np.allclose(jT, j_formula, atol=1e-5),
          f"(max err {np.max(np.abs(jT - j_formula)):.1e})")

# ---------------------------------------------------------------------------
# 9. Flat w0wa (CPL)
# ---------------------------------------------------------------------------
thc = [70.0, -0.8, -0.6, 0.31]                            # H0, w0, wa, Om0
zc = np.array([0.0, 0.3, 1.0, 3.0, 10.0])
Hc, Omc = cs.background_model(thc, zc, "cpl")
check("cpl: E(0) = 1", abs(Hc[0] / thc[0] - 1) < 1e-12)
check("cpl: (w0, wa) = (-1, 0) is the LCDM limit",
      np.allclose(cs.background_model([70.0, -1.0, 0.0, Om], zc, "cpl")[0],
                  70.0 * np.sqrt(Om * (1 + zc) ** 3 + 1 - Om), rtol=1e-12))
w_cpl = w_de_numeric(thc, "cpl", zc)
check("cpl: numerical w_DE(z) = w0 + wa z/(1+z)", np.allclose(w_cpl, thc[1] + thc[2] * zc / (1 + zc), atol=1e-6))

# ---------------------------------------------------------------------------
# 10. Optional late-time closure (Z_MATCH): closure for z <= Z_MATCH, matter + frozen dark density beyond
# ---------------------------------------------------------------------------
zs_ = np.array([0.0, 0.5, 2.0, 3.0, 3.0001, 10.0, 1100.0])
for m in cs.MODELS:
    thm = [H0, 1.5 * Om - 1 - 0.005, 1.0, Om]               # dq = -0.005: unstitched Omega_m(z) > 1 beyond z ~ 4.9
    cs.Z_MATCH = None
    H_free, Om_free = cs.background_model(thm, zs_, m)
    cs.Z_MATCH = 3.0
    H_st, Om_st = cs.background_model(thm, zs_, m)
    lcdm_st = cs.background_model(theta, np.array([0.0, 2.0, 50.0, 1100.0]), m)[0]
    cs.Z_MATCH = None
    check(f"{m}: Z_MATCH leaves z <= Z_MATCH unchanged", np.allclose(H_st[:4], H_free[:4], rtol=1e-12))
    check(f"{m}: Z_MATCH continuous at Z_MATCH", abs(H_st[4] / H_st[3] - 1) < 1e-3)
    check(f"{m}: unstitched Omega_m > 1 at z=1100 but stitched <= 1", Om_free[-1] > 1 and Om_st[-1] <= 1 + 1e-12,
          f"({Om_free[-1]:.3f} -> {Om_st[-1]:.3f})")
    check(f"{m}: LCDM limit unchanged by Z_MATCH",
          np.allclose(lcdm_st, H0 * np.sqrt(Om * (1 + np.array([0.0, 2.0, 50.0, 1100.0])) ** 3 + 1 - Om), rtol=1e-10))
cs.Z_MATCH = 3.0
check("Z_MATCH rejects rho_DE(Z_MATCH) < 0",
      cs.background_model([H0, 1.5 * Om - 1 - 0.02, 1.0, Om], np.array([0.0, 1.0]), "model_3") is None)
cs.Z_MATCH = None

print("\nALL PASSED" if ok else "\nSOME CHECKS FAILED")