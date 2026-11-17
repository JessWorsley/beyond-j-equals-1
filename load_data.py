import numpy as np
import pandas as pd

from constants import Z_MIN_PANTHEON


def marginalise_over_M(n, means, cov, reshape=True):
    """Marginalise analytically over an additive magnitude offset.

    C_marg^-1 = C^-1 - (C^-1 1)(C^-1 1)^T / (1^T C^-1 1), which removes the
    direction of a constant shift in all magnitudes.

    Parameters
    ----------
    n : int
        Number of supernovae.
    means : array_like
        Data vector.  Unused; kept for the call signature.
    cov : numpy.ndarray
        Covariance matrix, flattened (if `reshape`) or n x n.
    reshape : bool, optional
        Reshape `cov` to (n, n) first.  Default is True.

    Returns
    -------
    cov : numpy.ndarray
        The symmetrised covariance.
    inv_cov_marg : numpy.ndarray
        The symmetrised, offset-marginalised inverse covariance.
    """
    if reshape:
        cov = cov.reshape((n, n))
    cov = 0.5 * (cov + cov.T)
    Cinv = np.linalg.inv(cov)
    ones = np.ones(n)
    v = Cinv @ ones
    inv_cov_marg = Cinv - np.outer(v, v) / (ones @ v)
    return cov, 0.5 * (inv_cov_marg + inv_cov_marg.T)


# =========================================================
# DESI BAO
# =========================================================
def load_desi():
    """Load the DESI DR2 Gaussian BAO data vector and covariance.

    Returns
    -------
    z : numpy.ndarray
        Effective redshift of each measurement.
    means : numpy.ndarray
        Measured values.
    cov : numpy.ndarray
        Covariance matrix.
    inv_cov : numpy.ndarray
        Inverse covariance matrix.
    kinds : list of str
        Quantity of each measurement: ``'DM_over_rs'``, ``'DH_over_rs'`` or
        ``'DV_over_rs'``.
    """
    cov_file = 'cobaya_packages/data/bao_data/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_cov.txt'
    mean_file = 'cobaya_packages/data/bao_data/desi_bao_dr2/desi_gaussian_bao_ALL_GCcomb_mean.txt'

    cov = np.loadtxt(cov_file)
    data = pd.read_csv(mean_file, sep=' ')
    data.dropna(axis=1, how='all', inplace=True)
    data.rename(columns={'#': 'z', '[z]': 'value', '[value': 'quantity'}, inplace=True)
    z, means, kinds = data['z'].values, data['value'].values, list(data['quantity'])
    return z, means, cov, np.linalg.inv(cov), kinds


# =========================================================
# Pantheon+
# =========================================================
def load_pantheon(z_min=Z_MIN_PANTHEON):
    """Load Pantheon+ with the statistical + systematic covariance.

    The redshift cut removes the Cepheid-calibrator SNe, whose distances are
    not Hubble-flow distances.

    Parameters
    ----------
    z_min : float or None, optional
        Keep only SNe with zHD > `z_min`.  0 or None keeps everything.
        Default is ``constants.Z_MIN_PANTHEON``.

    Returns
    -------
    z_hd : numpy.ndarray
        Hubble-diagram (CMB-frame, peculiar-velocity-corrected) redshifts.
    z_hel : numpy.ndarray
        Heliocentric redshifts.
    m_b_corr : numpy.ndarray
        Corrected apparent magnitudes.
    cov : numpy.ndarray
        Covariance of the selected SNe.
    inv_cov : numpy.ndarray
        Inverse covariance with the absolute magnitude M marginalised.
    """
    dat = pd.read_csv('cobaya_packages/data/sn_data/PantheonPlus/Pantheon+SH0ES.dat',
                      delimiter=' ', header=0, usecols=['zHD', 'zHEL', 'm_b_corr'])
    z_hd, z_hel, mb = dat['zHD'].values, dat['zHEL'].values, dat['m_b_corr'].values

    with open('cobaya_packages/data/sn_data/PantheonPlus/Pantheon+SH0ES_STAT+SYS.cov') as f:
        f.readline()                                  # first line: matrix size
        cov_flat = np.loadtxt(f)
    n = int(np.sqrt(cov_flat.size))
    cov_full = cov_flat.reshape((n, n))

    mask = np.ones(n, dtype=bool) if not z_min else (z_hd > z_min)
    cov, inv_cov = marginalise_over_M(mask.sum(), mb[mask], cov_full[np.ix_(mask, mask)], reshape=False)
    return z_hd[mask], z_hel[mask], mb[mask], cov, inv_cov


# =========================================================
# DES-SN5YR (Dovekie recalibration)
# =========================================================
def load_desy5(csv_path='cobaya_packages/data/sn_data/DES-Dovekie/DES-Dovekie_HD.csv', inv_cov_path='cobaya_packages/data/sn_data/DES-Dovekie/covtot_inv_000.npz',
               mu_err_max=1.0, z_min=None):
    """Load the DES-SN5YR (Dovekie) Hubble diagram and covariance.

    The output has the same layout as ``load_pantheon``, with M marginalised
    analytically.

    Parameters
    ----------
    csv_path : str, optional
        Hubble-diagram file with columns zHD, MU, MUERR (and zHEL if
        available).
    inv_cov_path : str, optional
        npz file with the inverse covariance (upper triangle, flattened, key
        ``'cov'``; size under key ``'nsn'``).
    mu_err_max : float, optional
        Drop SNe with a non-finite MUERR or MUERR >= `mu_err_max`.  Default is
        1.0.
    z_min : float or None, optional
        If given, keep only SNe with zHD > `z_min`.  Default is None.

    Returns
    -------
    z_hd : numpy.ndarray
        Hubble-diagram redshifts.
    z_hel : numpy.ndarray
        Heliocentric redshifts (zHD if the file has no zHEL column).
    mu : numpy.ndarray
        Distance moduli.
    cov : numpy.ndarray
        Covariance of the selected SNe.
    inv_cov : numpy.ndarray
        Inverse covariance with M marginalised.

    Raises
    ------
    ValueError
        If the covariance size does not match the number of SNe in the file.

    Notes
    -----
    SNe must be dropped from the COVARIANCE: invert the full inverse, take
    the sub-block, then marginalise over M.  Taking the sub-block of the
    INVERSE covariance (as an earlier loader did) would condition on the
    removed SNe instead of marginalising over them.
    """
    dat = pd.read_csv(csv_path, sep=',', header=0)
    z_hd, mu, mu_err = dat['zHD'].values, dat['MU'].values, dat['MUERR'].values
    if 'zHEL' in dat.columns:
        z_hel = dat['zHEL'].values
    else:
        print("load_desy5: no zHEL column; using zHD in the (1 + z_hel) factor")
        z_hel = z_hd.copy()

    f = np.load(inv_cov_path)
    n = int(f['nsn'][0])
    if n != len(dat):
        raise ValueError(f"DES covariance is {n}x{n} but the Hubble diagram has {len(dat)} supernovae")
    inv_full = np.zeros((n, n))
    inv_full[np.triu_indices(n)] = f['cov']
    inv_full += np.triu(inv_full, 1).T                       # mirror the upper triangle
    cov_full = np.linalg.inv(inv_full)
    cov_full = 0.5 * (cov_full + cov_full.T)

    mask = np.isfinite(mu_err) & (mu_err < mu_err_max) & np.isfinite(mu)
    if z_min:
        mask &= z_hd > z_min
    if (~mask).any():
        print(f"load_desy5: dropped {(~mask).sum()} of {n} supernovae")
    cov, inv_cov = marginalise_over_M(mask.sum(), mu[mask], cov_full[np.ix_(mask, mask)], reshape=False)
    return z_hd[mask], z_hel[mask], mu[mask], cov, inv_cov


def load_sn(name):
    """Load a supernova dataset by name.

    Parameters
    ----------
    name : {'pantheon', 'desy5'}
        Dataset (case-insensitive).

    Returns
    -------
    data : tuple
        ``(z_hd, z_hel, mu, cov, inv_cov)``; see ``load_pantheon``.

    Raises
    ------
    ValueError
        If `name` is not a known dataset.
    """
    loaders = {"pantheon": load_pantheon, "desy5": load_desy5}
    if name.lower() not in loaders:
        raise ValueError(f"unknown SN dataset {name!r}; use one of {list(loaders)}")
    return loaders[name.lower()]()


# =========================================================
# Planck compressed distance priors
# =========================================================
def planck_distance_priors():
    """Return the compressed Planck distance priors.

    Values are from the wCDM column of Chen, Huang & Wang 2018
    (arXiv:1808.05724), in the order [R, lA, omega_b].

    Returns
    -------
    mean : numpy.ndarray
        Prior means.
    cov : numpy.ndarray
        Covariance, built from the sigmas and correlation matrix.
    fid : dict
        Reference R and lA at the calibration fiducial point (see
        ``closure_cosmo.calibrate``).
    """
    mean = np.array([1.7493, 301.462, 0.02239])
    sigma = np.array([0.0047, 0.090, 0.00015])
    corr = np.array([[1.0, 0.47, -0.66],
                     [0.47, 1.0, -0.34],
                     [-0.66, -0.34, 1.0]])
    cov = corr * np.outer(sigma, sigma)
    # Reference values of R and lA at the calibration fiducial point (see constants.FIDUCIAL)
    fid = dict(R=mean[0], lA=mean[1])
    return mean, cov, fid