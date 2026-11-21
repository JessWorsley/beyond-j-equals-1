# =========================================================
# Model (used by ALL likelihoods)
# =========================================================
MODEL = 'model_1'

# =========================================================
# Physical constants (single definition of each)
# =========================================================
C_KMS = 299792.458          # km/s
TCMB = 2.7255               # K
NEFF = 3.046                # matches the compressed Planck priors (CLASS default is 3.044)
EXP_THRESHOLD = 700.0       # exp(700) ~ 1e304, near float max
Z_RAD_MAX = 1.0e8           # upper limit of the sound-horizon integral

# Radiation is NOT part of the closure equations, so it is handled separately:
#  * LATE  (SN and BAO distances, z < ~2.4): radiation changes distances by <~0.1%; leave it out.
#  * EARLY (r_s, r_d, D_M(z*), f_X): the compressed Planck priors are defined for a universe WITH photons
#    and neutrinos.  Without radiation r_s(z*) is 253 instead of 144 Mpc and d ln(lA)/d ln(omega_m) even
#    changes sign, so this cannot be switched off for the CMB constraint (see the derivation notes).
INCLUDE_RADIATION_LATE = True
INCLUDE_RADIATION_EARLY = True

# Aliases so older plotting scripts keep working
SPEED_OF_LIGHT_KM_S = C_KMS
T_cmb = TCMB

# =========================================================
# CMB calibration (see closure_cosmo.calibrate)
# =========================================================
FIDUCIAL = dict(H0=67.36, ombh2=0.02237, ommh2=0.1430)   # Planck 2018 TT,TE,EE+lowE+lensing
RD_PLANCK = 147.09          # Mpc, Planck 2018 r_drag at the fiducial point (check against the paper)

# =========================================================
# Data selection
# =========================================================
Z_MIN_PANTHEON = 0.01       # drop the low-z / Cepheid-calibrator SNe (zHD <= 0.01)

Z_MATCH = None
# Z_MATCH_ALLOW_NEGATIVE = False

# =========================================================
# Reference values (CLASS/Planck settings, plotting)
# =========================================================
omega_b = 0.02238280
N_ur = 3.044
omega_cdm = 0.1201075
SOUND_HORIZON_DRAG_EPOCH_MPC = 147.2
DRAG_EPOCH_REDSHIFT = 1060
SOUND_HORIZON_RECOMBINATION_EPOCH_MPC = 144.4
RECOMBINATION_REDSHIFT = 1089.92
OMEGA_M0 = 0.3


def __getattr__(name):
    """Build module attributes lazily (PEP 562).

    ``constants.LCDM`` is created on first access, so astropy is only imported
    when it is actually used.

    Parameters
    ----------
    name : str
        Attribute name.

    Returns
    -------
    cosmo : astropy.cosmology.FlatLambdaCDM
        Flat LCDM with H0 = 70 km/s/Mpc and Om0 = 0.3, for ``name == 'LCDM'``.

    Raises
    ------
    AttributeError
        For any other name.
    """
    # `const.LCDM` still works, but astropy is only imported when it is actually used
    if name == "LCDM":
        from astropy.cosmology import FlatLambdaCDM
        import astropy.units as u
        return FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Om0=0.3)
    raise AttributeError(name)

# =========================================================
# Plotting
# =========================================================
IBM_PALETTE = ["#648FFF", '#785EF0', '#DC267F', '#FE6100', '#FFB000']
WONG_PALETTE = ['#CC79A7', '#D55E00', '#E69F00', '#009E73', '#56B4E9', '#0072B2']

PLOT_TEXT_OPTIONS = {
    'text.usetex': True,
    'font.family': 'serif',
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 18,
}