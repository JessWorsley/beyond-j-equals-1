"""
closure_cosmo.py -- the ONLY place the closure background is defined.  No Cobaya here.

Contents
--------
* background_model / get_H_grid : closure H(z) for model_1, model_2, model_3
  (Chakraborty, Louw et al. 2025), h^2(z) written directly from the paper.
  get_H_grid is the radiation-free late-time H(z) for the SN and BAO likelihoods.
* H_total       : closure H(z) (+ radiation: off for SN/BAO, on for the CMB integrals), H^2 = H_cl^2 + H0^2 Or (1+z)^4
* D_M, r_sound  : Gauss-Legendre integrals in x = ln(1+z)
* z_star, z_drag: Hu & Sugiyama / Eisenstein & Hu fitting formulae
* cmb_quantities: R, lA, r_d, z*, z_d, dark-sector fraction, all in one call (cached)

Parameter vector everywhere: theta = [H0, q0, j0, Om0]; ombh2 passed separately.
"""
import numpy as np
from functools import lru_cache
from numpy.polynomial.legendre import leggauss

from constants import C_KMS, TCMB, NEFF, EXP_THRESHOLD, Z_RAD_MAX, FIDUCIAL, INCLUDE_RADIATION_LATE, INCLUDE_RADIATION_EARLY

import constants as _constants
# Optional "late-time closure": use the closure only for z <= Z_MATCH and continue with Om0 (1+z)^3 plus the dark
# density frozen at its Z_MATCH value beyond.  None (default) = extrapolate the closure to all z.
Z_MATCH = getattr(_constants, "Z_MATCH", None)
# With Z_MATCH set: allow rho_DE(Z_MATCH) < 0 (a sign-switching dark density: negative, frozen, beyond Z_MATCH)?
Z_MATCH_ALLOW_NEGATIVE = getattr(_constants, "Z_MATCH_ALLOW_NEGATIVE", False)

MODELS = ("model_1", "model_2", "model_3")     # closure models
ALL_MODELS = MODELS + ("cpl",)                   # + flat w0wa (CPL) dark energy for comparison

# theta = [H0, p1, p2, Om0] everywhere.  (p1, p2) = (q0, j0) for the closure models and (w0, wa) for 'cpl'.


class CosmoError(Exception):
    """Raised when a parameter point gives an unphysical or non-finite background."""


# --------------------------------------------------------------------------
# Closure backgrounds
# --------------------------------------------------------------------------
def _E2_closure(z, q0, j0, model, Om0=None):
    """Compute h^2(z) = H^2/H0^2 for one model, written directly from the paper.

    Parameters
    ----------
    z : array_like
        Redshifts.
    q0, j0 : float
        Deceleration and jerk parameters today (closure models), or w0 and wa
        for ``model='cpl'``.
    model : str
        One of ``closure_cosmo.ALL_MODELS``.
    Om0 : float, optional
        Matter fraction today.  Only used by ``'cpl'``.

    Returns
    -------
    E2 : numpy.ndarray or None
        h^2 at each z, or None if the parameters hit a singular point of the
        closed-form solution or an exponent would overflow (see
        ``constants.EXP_THRESHOLD``).

    Raises
    ------
    ValueError
        If `model` is not a known model.
    """
    z = np.asarray(z, dtype=float)
    lg = np.log1p(z)
    zmax_log = np.max(lg) if lg.size else 0.0

    if model == "model_1":
        if abs(1.0 + q0) < 1e-12:
            return None
        p = (2.0 + j0 + 3.0 * q0) / (1.0 + q0)
        N = (2.0 + j0 + 3.0 * q0) / (1.0 + q0) ** 2
        A = (j0 - q0 * (1.0 + 2.0 * q0)) / (1.0 + q0) ** 2
        if abs(N) < 1e-12 or abs(p) * zmax_log > EXP_THRESHOLD:
            return None
        return (2.0 * np.exp(p * lg) + A) / N

    if model == "model_2":
        if 8.0 * j0 + 1.0 <= 0.0:
            return None
        s = np.sqrt(8.0 * j0 + 1.0)
        if s * zmax_log > EXP_THRESHOLD:
            return None
        return np.exp(0.5 * (3.0 - s) * lg) * (
            (s + 4.0 * q0 + 1.0) * (np.exp(s * lg) - 1.0) / (2.0 * s) + 1.0
        )

    if model == "model_3":
        den = 2.0 * j0 - 6.0 * q0 + 1.0
        if abs(den) < 1e-3 or abs(2.0 * q0 - 1.0) < 1e-3:
            return None
        beta = 2.0 * (j0 - 1.0) / (2.0 * q0 - 1.0)
        if max(abs(beta), 3.0) * zmax_log > EXP_THRESHOLD:
            return None
        return ((1.0 - 2.0 * q0) ** 2 * np.exp(beta * lg)
                + 2.0 * np.exp(3.0 * lg) * (j0 - q0 * (2.0 * q0 + 1.0))) / den

    if model == "cpl":
        w0, wa = q0, j0
        n = 3.0 * (1.0 + w0 + wa)
        if abs(n) * zmax_log > EXP_THRESHOLD:
            return None
        # flat CPL: rho_DE(z)/rho_DE(0) = (1+z)^{3(1+w0+wa)} exp(-3 wa z/(1+z))
        return Om0 * np.exp(3.0 * lg) + (1.0 - Om0) * np.exp(n * lg - 3.0 * wa * z / (1.0 + z))

    raise ValueError(f"No model named {model!r}. Use one of {ALL_MODELS}.")


def background_model(theta, z_eval, model):
    """Compute H(z) and Omega_m(z) for a closure model or CPL (no radiation).

    Omega_m(z) = Om0 (1+z)^3 H0^2 / H^2 for all three closure models (checked
    against the paper's expressions in test_closure_cosmo.py).

    Parameters
    ----------
    theta : sequence of float
        ``[H0, p1, p2, Om0]``: (p1, p2) = (q0, j0) for the closure models and
        (w0, wa) for ``'cpl'``.
    z_eval : array_like
        Redshifts.
    model : str
        One of ``closure_cosmo.ALL_MODELS``.

    Returns
    -------
    H : numpy.ndarray
        H(z) in km/s/Mpc.
    Om : numpy.ndarray
        Omega_m(z).

    None is returned instead of ``(H, Om)`` for non-finite input, a failed
    closed form, or H^2 <= 0 anywhere.

    Notes
    -----
    If ``Z_MATCH`` is set (closure models only), then for z > Z_MATCH::

        E^2 = Om0 (1+z)^3 + rho_DE(Z_MATCH),
        rho_DE(Z_MATCH) = E_closure^2(Z_MATCH) - Om0 (1+Z_MATCH)^3,

    i.e. matter plus a frozen (w = -1) dark density, continuous at Z_MATCH.  A
    point with rho_DE(Z_MATCH) < 0 is rejected unless ``Z_MATCH_ALLOW_NEGATIVE``
    is True; the frozen density is then negative, so Omega_m is slightly above
    1 at Z_MATCH and tends to 1 beyond.
    """
    H0, q0, j0, Om0 = theta
    if not all(np.isfinite([H0, q0, j0, Om0])):
        return None
    z = np.asarray(z_eval, dtype=float)
    if Z_MATCH is not None and model in MODELS:
        E2c = _E2_closure(np.minimum(z, Z_MATCH), q0, j0, model, Om0)
        E2m = _E2_closure(np.array([float(Z_MATCH)]), q0, j0, model, Om0)
        if E2c is None or E2m is None:
            return None
        rho_m = E2m[0] - Om0 * (1.0 + Z_MATCH) ** 3
        if not np.isfinite(rho_m) or (rho_m < 0.0 and not Z_MATCH_ALLOW_NEGATIVE):
            return None
        E2 = np.where(z > Z_MATCH, Om0 * (1.0 + z) ** 3 + rho_m, E2c)
    else:
        E2 = _E2_closure(z, q0, j0, model, Om0)
    if E2 is None or not np.all(np.isfinite(E2)) or np.any(E2 <= 0):
        return None
    H = H0 * np.sqrt(E2)
    Om = Om0 * (1.0 + z) ** 3 / E2
    if not np.all(np.isfinite(Om)):
        return None
    return H, Om


# --------------------------------------------------------------------------
# Radiation, fitting formulae
# --------------------------------------------------------------------------
def omega_r_density(h):
    """Compute the radiation density parameter today.

    Parameters
    ----------
    h : float
        Dimensionless Hubble parameter, H0 / (100 km/s/Mpc).

    Returns
    -------
    Or : float
        Omega_r of photons plus massless-equivalent neutrinos (``NEFF``), as a
        fraction of the critical density today.
    """
    return 2.469e-5 * (TCMB / 2.7255) ** 4 * (1.0 + 0.2271 * NEFF) / h ** 2


def radiation_density(h, on):
    """Return the Omega_r used in the numerics.

    Parameters
    ----------
    h : float
        Dimensionless Hubble parameter, H0 / (100 km/s/Mpc).
    on : bool
        Whether radiation is included.

    Returns
    -------
    Or : float
        ``omega_r_density(h)`` if `on`, otherwise 0.
    """
    return omega_r_density(h) if on else 0.0


def z_star(wb, wm):
    """Compute the photon decoupling redshift z* (Hu & Sugiyama 1996 fit).

    Parameters
    ----------
    wb : float
        Physical baryon density, omega_b = Omega_b h^2.
    wm : float
        Physical matter density, omega_m = Omega_m h^2.

    Returns
    -------
    zs : float
        z*.  About 2 higher than CAMB's value; see ``calibrate``.
    """
    g1 = 0.0783 * wb ** -0.238 / (1.0 + 39.5 * wb ** 0.763)
    g2 = 0.560 / (1.0 + 21.1 * wb ** 1.81)
    return 1048.0 * (1.0 + 0.00124 * wb ** -0.738) * (1.0 + g1 * wm ** g2)


def z_drag(wb, wm):
    """Compute the baryon drag redshift z_d (Eisenstein & Hu 1998 fit).

    Parameters
    ----------
    wb : float
        Physical baryon density, omega_b = Omega_b h^2.
    wm : float
        Physical matter density, omega_m = Omega_m h^2.

    Returns
    -------
    zd : float
        z_d.  About 1021, well below CAMB's ~1060; see ``calibrate``.
    """
    b1 = 0.313 * wm ** -0.419 * (1.0 + 0.607 * wm ** 0.674)
    b2 = 0.238 * wm ** 0.223
    return 1291.0 * wm ** 0.251 / (1.0 + 0.659 * wm ** 0.828) * (1.0 + b1 * wb ** b2)


# --------------------------------------------------------------------------
# Total expansion rate and integrals
# --------------------------------------------------------------------------
def H_total(z, theta, model, radiation=INCLUDE_RADIATION_EARLY):
    """Compute the closure H(z), optionally with radiation added.

    H^2 = H_closure^2 + H0^2 Omega_r (1+z)^4.

    Parameters
    ----------
    z : array_like
        Redshifts.
    theta : sequence of float
        ``[H0, p1, p2, Om0]`` (see ``background_model``).
    model : str
        One of ``closure_cosmo.ALL_MODELS``.
    radiation : bool, optional
        Add radiation.  Default is ``constants.INCLUDE_RADIATION_EARLY``.

    Returns
    -------
    H : numpy.ndarray
        H(z) in km/s/Mpc.

    Raises
    ------
    CosmoError
        If ``background_model`` fails for this point.
    """
    H0 = theta[0]
    res = background_model(theta, z, model)
    if res is None:
        raise CosmoError("closure background failed")
    Or = radiation_density(H0 / 100.0, radiation)
    return np.sqrt(res[0] ** 2 + H0 ** 2 * Or * (1.0 + np.asarray(z)) ** 4)


def get_H_grid(theta, z_max, model, n_grid=2000, radiation=INCLUDE_RADIATION_LATE):
    """Tabulate H(z) on a linear grid for the SN and BAO likelihoods.

    Parameters
    ----------
    theta : sequence of float
        ``[H0, p1, p2, Om0]`` (see ``background_model``).
    z_max : float
        Upper end of the grid.
    model : str
        One of ``closure_cosmo.ALL_MODELS``.
    n_grid : int, optional
        Number of grid points.  Default is 2000.
    radiation : bool, optional
        Add radiation.  Default is ``constants.INCLUDE_RADIATION_LATE``.

    Returns
    -------
    grid : tuple of numpy.ndarray or None
        ``(z_grid, H_grid)`` with z_grid from 0 to `z_max`, or None if the
        background is invalid.
    """
    z_grid = np.linspace(0.0, z_max, n_grid)
    try:
        return z_grid, H_total(z_grid, theta, model, radiation)
    except CosmoError:
        return None


_xg, _wg = leggauss(200)


def _gl(f, a, b):
    """Integrate `f` from `a` to `b` with 200-point Gauss-Legendre quadrature.

    Parameters
    ----------
    f : callable
        Vectorised integrand.
    a, b : float
        Integration limits.

    Returns
    -------
    I : float
        Value of the integral.
    """
    x = 0.5 * (b - a) * _xg + 0.5 * (b + a)
    return 0.5 * (b - a) * np.sum(_wg * f(x))


def D_M(z, theta, model):
    """Compute the flat comoving distance to redshift z.

    The integral is done in x = ln(1+z), with radiation as in ``H_total``'s
    default (``INCLUDE_RADIATION_EARLY``).

    Parameters
    ----------
    z : float
        Redshift.
    theta : sequence of float
        ``[H0, p1, p2, Om0]`` (see ``background_model``).
    model : str
        One of ``closure_cosmo.ALL_MODELS``.

    Returns
    -------
    DM : float
        D_M(z) in Mpc.

    Raises
    ------
    CosmoError
        If the background fails at a quadrature node.
    """
    f = lambda x: np.exp(x) * C_KMS / H_total(np.expm1(x), theta, model)
    return _gl(f, 0.0, np.log1p(z))


def r_sound(z_low, theta, model, ombh2, z_max=Z_RAD_MAX):
    """Compute the comoving sound horizon r_s(z_low).

    r_s = integral from z_low to z_max of c_s / H dz, with
    c_s = c / sqrt(3 (1 + R_b)) and R_b = 31500 omega_b (T_CMB/2.7)^-4 / (1+z).
    The integral is done in x = ln(1+z) and split at z ~ 3000 (around
    matter-radiation equality) for accuracy.

    Parameters
    ----------
    z_low : float
        Lower limit, e.g. z* or z_d.
    theta : sequence of float
        ``[H0, p1, p2, Om0]`` (see ``background_model``).
    model : str
        One of ``closure_cosmo.ALL_MODELS``.
    ombh2 : float
        Physical baryon density omega_b.
    z_max : float, optional
        Upper limit standing in for infinity.  Default is
        ``constants.Z_RAD_MAX``.

    Returns
    -------
    rs : float
        r_s(z_low) in Mpc.

    Raises
    ------
    CosmoError
        If the background fails at a quadrature node.
    """
    def f(x):
        """Return the integrand (1+z) c_s / H at x = ln(1+z), in Mpc."""
        z = np.expm1(x)
        Rb = 31500.0 * ombh2 * (TCMB / 2.7) ** -4 / (1.0 + z)
        return np.exp(x) * C_KMS / np.sqrt(3.0 * (1.0 + Rb)) / H_total(z, theta, model)
    a, b = np.log1p(z_low), np.log1p(z_max)
    # split at z ~ 3000 (around equality) for accuracy
    m = np.clip(np.log(3000.0), a + 1e-6, b - 1e-6)
    return _gl(f, a, m) + _gl(f, m, b)


def dark_fraction(z, theta, model, radiation=INCLUDE_RADIATION_EARLY):
    """Compute f_X(z), the fraction of the energy budget that is neither matter nor radiation.

    f_X = 1 - H0^2 (Om0 (1+z)^3 + Omega_r (1+z)^4) / H^2.  The compressed
    Planck priors are only valid if f_X(z*) ~ 0.

    Parameters
    ----------
    z : array_like
        Redshifts.
    theta : sequence of float
        ``[H0, p1, p2, Om0]`` (see ``background_model``).
    model : str
        One of ``closure_cosmo.ALL_MODELS``.
    radiation : bool, optional
        Include radiation.  Default is ``constants.INCLUDE_RADIATION_EARLY``.

    Returns
    -------
    fX : numpy.ndarray
        f_X(z).

    Raises
    ------
    CosmoError
        If the background fails.
    """
    H0, q0, j0, Om0 = theta
    z = np.asarray(z, dtype=float)
    Or = radiation_density(H0 / 100.0, radiation)
    H2 = H_total(z, theta, model, radiation) ** 2
    return 1.0 - H0 ** 2 * (Om0 * (1 + z) ** 3 + Or * (1 + z) ** 4) / H2


# --------------------------------------------------------------------------
# Everything the Planck / DESI likelihoods need, in one cached call
# --------------------------------------------------------------------------
@lru_cache(maxsize=16)
def _cmb_cached(theta, ombh2, model):
    """Compute the CMB and drag-epoch quantities (cached worker for ``cmb_quantities``).

    Parameters
    ----------
    theta : tuple of float
        ``(H0, p1, p2, Om0)``; must be hashable for the cache.
    ombh2 : float
        Physical baryon density omega_b.
    model : str
        One of ``closure_cosmo.ALL_MODELS``.

    Returns
    -------
    out : dict
        Uncalibrated values: ``R`` (shift parameter), ``lA`` (acoustic scale),
        ``rd`` (r_s(z_d) in Mpc), ``zstar``, ``zdrag``, ``DM_star`` (Mpc),
        ``rs_star`` (Mpc), ``fX_star`` (f_X(z*)) and ``fX_min`` (minimum of
        f_X over 0 <= z <= z*).

    Raises
    ------
    CosmoError
        If omega_b is not in (0, omega_m), if H^2 is not positive on a dense
        grid up to ``Z_RAD_MAX``, or if any output is non-finite.
    """
    H0, q0, j0, Om0 = theta
    wm = Om0 * (H0 / 100.0) ** 2
    if not (0.0 < ombh2 < wm):
        raise CosmoError("omega_b >= omega_m")

    zs, zd = z_star(ombh2, wm), z_drag(ombh2, wm)

    # H^2 must stay positive on the whole range used (closure is monotone-ish,
    # but check on a dense log grid rather than only at quadrature nodes)
    zchk = np.expm1(np.linspace(0.0, np.log1p(Z_RAD_MAX), 600))
    H_total(zchk, theta, model)

    DM = D_M(zs, theta, model)
    rs_star = r_sound(zs, theta, model, ombh2)
    rd = r_sound(zd, theta, model, ombh2)
    R = np.sqrt(Om0) * H0 * DM / C_KMS
    lA = np.pi * DM / rs_star

    zg = np.linspace(0.0, zs, 200)
    fx = dark_fraction(zg, theta, model)

    out = dict(R=R, lA=lA, rd=rd, zstar=zs, zdrag=zd, DM_star=DM, rs_star=rs_star,
               fX_star=float(fx[-1]), fX_min=float(fx.min()))
    if not all(np.isfinite(v) for v in out.values()):
        raise CosmoError("non-finite CMB quantity")
    return out


def cmb_quantities(theta, ombh2, model):
    """Compute R, lA, r_d, z*, z_d and f_X, cached on the exact parameter point.

    The DESI and Planck likelihoods both call this at the same point, so the
    cache avoids doing the integrals twice.

    Parameters
    ----------
    theta : sequence of float
        ``[H0, p1, p2, Om0]`` (see ``background_model``).
    ombh2 : float
        Physical baryon density omega_b.
    model : str
        One of ``closure_cosmo.ALL_MODELS``.

    Returns
    -------
    out : dict
        See ``_cmb_cached``.  Values are uncalibrated; pass them through
        ``calibrate`` before comparing with data.

    Raises
    ------
    CosmoError
        See ``_cmb_cached``.
    """
    return _cmb_cached(tuple(float(t) for t in theta), float(ombh2), model)


# --------------------------------------------------------------------------
# Calibration against a fiducial LCDM point
# --------------------------------------------------------------------------
# The fitting formulae are not accurate enough in absolute terms:
#   * Hu-Sugiyama z* is ~2 higher than the CAMB value  -> lA biased ~ +0.1-0.2 %
#   * Eisenstein-Hu z_d (~1021) is far from CAMB's z_drag (~1060) -> r_d ~ +2.4 %
# Both are almost independent of the model, so we compute everything with the same
# code at a fiducial LCDM point and rescale:  X_pred = X_ref * X_code(model) / X_code(fid).


@lru_cache(maxsize=4)
def fiducial_code_values(H0=FIDUCIAL["H0"], ombh2=FIDUCIAL["ombh2"], ommh2=FIDUCIAL["ommh2"]):
    """Compute R, lA and r_d with this code at the fiducial flat-LCDM point.

    The LCDM point is evaluated through model_1 with j0 = 1 and
    q0 = 1.5 Om0 - 1.  The result is cached.

    Parameters
    ----------
    H0 : float, optional
        Hubble constant in km/s/Mpc.  Default is ``FIDUCIAL["H0"]``.
    ombh2 : float, optional
        Physical baryon density.  Default is ``FIDUCIAL["ombh2"]``.
    ommh2 : float, optional
        Physical matter density.  Default is ``FIDUCIAL["ommh2"]``.

    Returns
    -------
    fid : dict
        Keys ``R``, ``lA`` and ``rd``.
    """
    Om = ommh2 / (H0 / 100.0) ** 2
    q = cmb_quantities([H0, 1.5 * Om - 1.0, 1.0, Om], ombh2, "model_1")
    return dict(R=q["R"], lA=q["lA"], rd=q["rd"])


def calibrate(q, ref):
    """Rescale R, lA and r_d so the code reproduces reference values at the fiducial point.

    X_pred = X_ref * X_code(model) / X_code(fiducial), which removes the
    (almost model-independent) error of the z* and z_d fitting formulae.

    Parameters
    ----------
    q : dict
        Output of ``cmb_quantities``.
    ref : dict
        Reference values at the fiducial point, under any of the keys ``'R'``,
        ``'lA'`` and ``'rd'`` (e.g. ``rd=147.09`` from Planck; R and lA = the
        means of the distance priors used).  None values are ignored.

    Returns
    -------
    out : dict
        ``R``, ``lA`` and ``rd``, rescaled where a reference value is given and
        copied from `q` otherwise.
    """
    fid = fiducial_code_values()
    out = {k: q[k] for k in ("R", "lA", "rd")}
    for k, v in ref.items():
        if v is not None:
            out[k] = q[k] * v / fid[k]
    return out