"""Distances from H(z).  All H(z) comes from closure_cosmo (radiation included)."""
import numpy as np
from scipy.integrate import cumulative_trapezoid

from constants import C_KMS
from closure_cosmo import get_H_grid


def comoving_distance_from_H(z_grid, H_grid):
    """Compute the flat-universe comoving distance chi(z) on a grid (trapezoid rule).

    Parameters
    ----------
    z_grid : numpy.ndarray
        Increasing redshifts; ``z_grid[0]`` must be 0.
    H_grid : numpy.ndarray
        H(z) in km/s/Mpc on `z_grid`.

    Returns
    -------
    chi : numpy.ndarray
        chi(z) in Mpc on `z_grid`.
    """
    return C_KMS * cumulative_trapezoid(1.0 / H_grid, z_grid, initial=0.0)


def luminosity_distance(theta, z_hd, z_hel, model, n_grid=2000):
    """Compute the SN luminosity distance D_L = (1 + z_hel) chi(z_hd).

    This is the convention of the Pantheon+ cosmology analysis.

    Parameters
    ----------
    theta : sequence of float
        ``[H0, p1, p2, Om0]`` (see ``closure_cosmo.background_model``).
    z_hd : array_like
        Hubble-diagram redshifts.
    z_hel : array_like
        Heliocentric redshifts.
    model : str
        One of ``closure_cosmo.ALL_MODELS``.
    n_grid : int, optional
        Number of points in the H(z) grid.  Default is 2000.

    Returns
    -------
    DL : numpy.ndarray or None
        D_L in Mpc, or None for an invalid background.
    """
    z_hd = np.asarray(z_hd, dtype=float)
    res = get_H_grid(theta, np.max(z_hd), model, n_grid)
    if res is None:
        return None
    z_grid, H_grid = res
    chi = np.interp(z_hd, z_grid, comoving_distance_from_H(z_grid, H_grid))
    DL = (1.0 + np.asarray(z_hel, dtype=float)) * chi
    if not np.all(np.isfinite(DL)) or np.any(DL <= 0):
        return None
    return DL


def bao_distances(theta, z_bao, model, n_grid=2000):
    """Compute D_M, D_H and D_V at the BAO redshifts.

    Parameters
    ----------
    theta : sequence of float
        ``[H0, p1, p2, Om0]`` (see ``closure_cosmo.background_model``).
    z_bao : array_like
        BAO redshifts.
    model : str
        One of ``closure_cosmo.ALL_MODELS``.
    n_grid : int, optional
        Number of points in the H(z) grid.  Default is 2000.

    Returns
    -------
    distances : tuple of numpy.ndarray or None
        ``(DM, DH, DV)`` in Mpc, or None for an invalid background.
    """
    z_bao = np.asarray(z_bao, dtype=float)
    res = get_H_grid(theta, np.max(z_bao), model, n_grid)
    if res is None:
        return None
    z_grid, H_grid = res
    DM_grid = comoving_distance_from_H(z_grid, H_grid)       # flat: D_M = chi
    DH_grid = C_KMS / H_grid
    DV_grid = (z_grid * DM_grid ** 2 * DH_grid) ** (1.0 / 3.0)
    out = tuple(np.interp(z_bao, z_grid, g) for g in (DM_grid, DH_grid, DV_grid))
    if not all(np.all(np.isfinite(o)) for o in out):
        return None
    return out
