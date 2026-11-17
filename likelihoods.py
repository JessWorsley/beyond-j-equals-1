import numpy as np
from cobaya.likelihood import Likelihood

import closure_cosmo as cs
from calculate_distances import luminosity_distance
from bao import bao_model_vector
from load_data import load_desi, load_pantheon, load_desy5, planck_distance_priors
from constants import MODEL, RD_PLANCK

# Every likelihood has a `_logp(theta)` that works for any model in closure_cosmo.ALL_MODELS, with
# theta = [H0, p1, p2, Om0] ((p1, p2) = (q0, j0) for the closure models, (w0, wa) for 'cpl').
# The classes below only differ in the names of the sampled parameters:
#   PantheonLikelihood / DESILikelihood / PlanckLikelihood          -> closure models (q0, j0)
#   PantheonLikelihoodCPL / DESILikelihoodCPL / PlanckLikelihoodCPL -> flat w0wa (w0, wa)


def _check_model(model, allowed, cls):
    """Refuse a model that does not match the sampled parameter names.

    Without this check, q0 and j0 would silently be read as w0 and wa (or the
    other way round).

    Parameters
    ----------
    model : str
        Model requested for the likelihood.
    allowed : tuple of str
        Models valid for the class.
    cls : str
        Class name, used in the error message.

    Raises
    ------
    ValueError
        If `model` is not in `allowed`.
    """
    if model not in allowed:
        raise ValueError(
            f"{cls}: model={model!r} is not valid for this class (allowed: {allowed}).\n"
            f"  * closure classes (q0, j0) need MODEL in constants.py to be one of {cs.MODELS};\n"
            f"  * for flat w0wa use the *CPL classes (cpl.yaml), which fix model='cpl' themselves.\n"
            f"  Otherwise q0, j0 would silently be read as w0, wa (or vice versa).")


_FROM_CONSTANTS = "constants"


def _apply_zmatch(obj):
    """Set closure_cosmo's late-time closure from a likelihood's yaml options.

    make_runs.py writes ``z_match`` and ``z_match_allow_negative`` into every
    likelihood block of a yaml, so the yaml (not whatever constants.py says
    when the run starts) fixes the late-time closure.  Old yaml files without
    these options leave the values from constants.py in place.

    Parameters
    ----------
    obj : cobaya.likelihood.Likelihood
        Likelihood instance carrying the ``z_match`` and
        ``z_match_allow_negative`` attributes.
    """
    zm = getattr(obj, "z_match", _FROM_CONSTANTS)
    if isinstance(zm, str):                                   # "constants": leave closure_cosmo as imported
        return
    cs.Z_MATCH = None if zm is None else float(zm)
    cs.Z_MATCH_ALLOW_NEGATIVE = bool(getattr(obj, "z_match_allow_negative", False))


class PantheonLikelihood(Likelihood):
    """Pantheon+ supernovae, absolute magnitude marginalised analytically.

    Samples H0, q0, j0 and Om0.  The SN sample is cut at
    zHD > ``constants.Z_MIN_PANTHEON``.

    Attributes
    ----------
    model : str
        Closure model (``'model_1'``, ``'model_2'`` or ``'model_3'``); set with
        the yaml ``model:`` option.  Default is ``constants.MODEL``.
    z_match : float, None or str
        Late-time closure redshift, None for the full closure, or
        ``"constants"`` to keep the value from constants.py (see
        ``_apply_zmatch``).
    z_match_allow_negative : bool
        Allow a negative frozen dark density beyond `z_match`.
    """

    input_params = ["H0", "q0", "j0", "Om0"]
    model = MODEL
    z_match = _FROM_CONSTANTS                                 # set by make_runs.py: None or a redshift (see _apply_zmatch)
    z_match_allow_negative = False
    _allowed = cs.MODELS

    def initialize(self):
        """Check the model, apply the z_match options and load the Pantheon+ data."""
        _check_model(self.model, self._allowed, type(self).__name__)
        _apply_zmatch(self)
        self.z_hd, self.z_hel, self.mu, self.cov, self.inv_cov = load_pantheon()

    def _logp(self, theta):
        """Compute the SN log-likelihood for one parameter vector.

        Parameters
        ----------
        theta : sequence of float
            ``[H0, p1, p2, Om0]`` (see ``closure_cosmo.background_model``).

        Returns
        -------
        logp : float
            -chi^2 / 2 with the magnitude offset marginalised, or -inf for an
            invalid background.
        """
        DL = luminosity_distance(theta, self.z_hd, self.z_hel, self.model)
        if DL is None:
            return -np.inf
        res = 5.0 * np.log10(DL) + 25.0 - self.mu
        chi2 = res @ self.inv_cov @ res
        return -0.5 * chi2 if np.isfinite(chi2) else -np.inf

    def logp(self, H0, q0, j0, Om0, _derived=None):
        """Return the log-likelihood at the sampled point (called by Cobaya).

        Parameters
        ----------
        H0, q0, j0, Om0 : float
            Sampled parameters.
        _derived : dict, optional
            Unused; there are no derived parameters.

        Returns
        -------
        logp : float
            See ``_logp``.
        """
        return self._logp([H0, q0, j0, Om0])


class DESY5Likelihood(PantheonLikelihood):
    """DES-SN5YR (Dovekie) supernovae in place of Pantheon+.

    The model code is the same as ``PantheonLikelihood``; only the data differ.
    """

    def initialize(self):
        """Check the model, apply the z_match options and load the DES-SN5YR data."""
        _check_model(self.model, self._allowed, type(self).__name__)
        _apply_zmatch(self)
        self.z_hd, self.z_hel, self.mu, self.cov, self.inv_cov = load_desy5()


class DESILikelihood(Likelihood):
    """DESI DR2 BAO, with r_d computed from the model.

    Samples H0, q0, j0, Om0 and ombh2, and outputs the derived parameter
    ``rd`` (calibrated to ``constants.RD_PLANCK`` at the fiducial point).

    Attributes
    ----------
    model : str
        Closure model; see ``PantheonLikelihood``.
    rd_fixed : float or None
        Use this r_d in Mpc (e.g. 147.2) instead of the model's, as in a
        DESI-style cross-check.  None (default) computes it.
    z_match, z_match_allow_negative
        Late-time closure options; see ``PantheonLikelihood``.
    """

    input_params = ["H0", "q0", "j0", "Om0", "ombh2"]
    output_params = ["rd"]
    model = MODEL
    rd_fixed = None        # e.g. 147.2: use this r_d [Mpc] instead of the model's (DESI-style cross-check); None = computed
    z_match = _FROM_CONSTANTS
    z_match_allow_negative = False
    _allowed = cs.MODELS

    def initialize(self):
        """Check the model, apply the z_match options and load the DESI BAO data."""
        _check_model(self.model, self._allowed, type(self).__name__)
        _apply_zmatch(self)
        self.z, self.means, self.cov, self.inv_cov, self.kinds = load_desi()

    def _logp(self, theta, ombh2, _derived):
        """Compute the BAO log-likelihood for one parameter vector.

        Parameters
        ----------
        theta : sequence of float
            ``[H0, p1, p2, Om0]`` (see ``closure_cosmo.background_model``).
        ombh2 : float
            Physical baryon density, used for r_d.
        _derived : dict or None
            If given, ``_derived["rd"]`` is set to r_d in Mpc (NaN if the point is
            rejected).

        Returns
        -------
        logp : float
            -chi^2 / 2, or -inf for an invalid point.
        """
        if _derived is not None:
            _derived["rd"] = np.nan                       # always defined, even on early return
        try:
            q = cs.cmb_quantities(theta, ombh2, self.model)           # cached, shared with Planck
            rd = float(self.rd_fixed) if self.rd_fixed is not None else cs.calibrate(q, dict(rd=RD_PLANCK))["rd"]
            vec = bao_model_vector(theta, self.z, self.kinds, self.model, rd)
            if vec is None:
                return -np.inf
            res = vec - self.means
            chi2 = res @ self.inv_cov @ res
        except (cs.CosmoError, ArithmeticError):
            return -np.inf
        if _derived is not None:
            _derived["rd"] = rd
        return -0.5 * chi2 if np.isfinite(chi2) else -np.inf

    def logp(self, H0, q0, j0, Om0, ombh2, _derived=None):
        """Return the log-likelihood at the sampled point (called by Cobaya).

        Parameters
        ----------
        H0, q0, j0, Om0, ombh2 : float
            Sampled parameters.
        _derived : dict, optional
            Filled with the derived ``rd``.

        Returns
        -------
        logp : float
            See ``_logp``.
        """
        return self._logp([H0, q0, j0, Om0], ombh2, _derived)


class DESIFreeRdLikelihood(DESILikelihood):
    """DESI BAO with r_d as a sampled parameter.

    There is no CMB and no early-universe extrapolation of the closure, so use
    this class without Planck (``make_runs.py --no-planck``).
    """

    input_params = ["H0", "q0", "j0", "Om0", "rd"]
    output_params = []

    def _chi2_logp(self, theta, rd):
        """Compute the BAO log-likelihood for a given r_d.

        Parameters
        ----------
        theta : sequence of float
            ``[H0, p1, p2, Om0]`` (see ``closure_cosmo.background_model``).
        rd : float
            Sound horizon at the drag epoch in Mpc.

        Returns
        -------
        logp : float
            -chi^2 / 2, or -inf for an invalid point.
        """
        try:
            vec = bao_model_vector(theta, self.z, self.kinds, self.model, rd)
            if vec is None:
                return -np.inf
            res = vec - self.means
            chi2 = res @ self.inv_cov @ res
        except (cs.CosmoError, ArithmeticError):
            return -np.inf
        return -0.5 * chi2 if np.isfinite(chi2) else -np.inf

    def logp(self, H0, q0, j0, Om0, rd, _derived=None):
        """Return the log-likelihood at the sampled point (called by Cobaya).

        Parameters
        ----------
        H0, q0, j0, Om0, rd : float
            Sampled parameters.
        _derived : dict, optional
            Unused; there are no derived parameters.

        Returns
        -------
        logp : float
            See ``_chi2_logp``.
        """
        return self._chi2_logp([H0, q0, j0, Om0], rd)


class PlanckLikelihood(Likelihood):
    """Compressed Planck distance priors: a Gaussian on [R, lA, omega_b].

    The mean and covariance come from ``load_data.planck_distance_priors``.
    Outputs the derived parameter ``fX_star`` = f_X(z*).

    Attributes
    ----------
    model : str
        Closure model; see ``PantheonLikelihood``.
    fx_max : float or None
        Hard cut: reject |f_X(z*)| > `fx_max` (e.g. 0.05).  None = no cut.
    positive_dark : bool
        Reject points where the effective dark density
        rho_DE = E^2 - Om0 (1+z)^3 is negative anywhere for z <= z*.
    fx_sigma : float or None
        Soft alternative to `fx_max`: add a Gaussian penalty
        -0.5 (f_X(z*) / fx_sigma)^2 to this likelihood.  None = off.
    calibrate_cmb : bool
        Rescale R and lA so the code reproduces the prior means at the
        fiducial point (see ``closure_cosmo.calibrate``).
    use_omb : bool
        False: use a 2-D Gaussian on (R, lA) only, marginalising omega_b.
    z_match, z_match_allow_negative
        Late-time closure options; see ``PantheonLikelihood``.
    debug_print : bool
        Print the reason a point is rejected.
    """

    input_params = ["H0", "q0", "j0", "Om0", "ombh2"]
    output_params = ["fX_star"]
    model = MODEL
    fx_max = None          # hard cut: e.g. 0.05 rejects |f_X(z*)| > 0.05; None = no cut
    positive_dark = False  # True: reject points where the effective dark density rho_DE = E^2 - Om0(1+z)^3 < 0 for z <= z*
    fx_sigma = None        # soft alternative: Gaussian penalty -0.5 (f_X/fx_sigma)^2 (e.g. 0.02); None = off
    calibrate_cmb = True   # rescale R, lA so the code reproduces the paper's values at the fiducial point
    use_omb = True         # False: 2D Gaussian on (R, lA) only (marginalising omega_b), as in the DESI-style cross-check
    z_match = _FROM_CONSTANTS
    z_match_allow_negative = False
    debug_print = False
    _allowed = cs.MODELS

    def initialize(self):
        """Check the model, apply the z_match options and set up the distance-prior Gaussian."""
        _check_model(self.model, self._allowed, type(self).__name__)
        _apply_zmatch(self)
        self.mean, self.cov, self.fid = planck_distance_priors()
        if not self.use_omb:
            self.mean, self.cov = self.mean[:2], self.cov[:2, :2]
        self.inv_cov = np.linalg.inv(self.cov)

    def _logp(self, theta, ombh2, _derived):
        """Compute the distance-prior log-likelihood for one parameter vector.

        Parameters
        ----------
        theta : sequence of float
            ``[H0, p1, p2, Om0]`` (see ``closure_cosmo.background_model``).
        ombh2 : float
            Physical baryon density omega_b.
        _derived : dict or None
            If given, ``_derived["fX_star"]`` is set to f_X(z*) (NaN if the
            background is invalid).

        Returns
        -------
        logp : float
            -chi^2 / 2 (plus the `fx_sigma` penalty if set), or -inf for an
            invalid point or one removed by `fx_max` or `positive_dark`.
        """
        if _derived is not None:
            _derived["fX_star"] = np.nan
        try:
            q = cs.cmb_quantities(theta, ombh2, self.model)
        except (cs.CosmoError, ArithmeticError) as e:
            if self.debug_print:
                print("Planck: point rejected:", theta, ombh2, repr(e))
            return -np.inf
        if _derived is not None:
            _derived["fX_star"] = q["fX_star"]
        if self.fx_max is not None and abs(q["fX_star"]) > self.fx_max:
            return -np.inf
        if self.positive_dark and q["fX_min"] < -1e-6:
            return -np.inf
        c = cs.calibrate(q, self.fid) if self.calibrate_cmb else q
        d = np.array([c["R"], c["lA"], ombh2][:len(self.mean)]) - self.mean
        logp = -0.5 * d @ self.inv_cov @ d
        if self.fx_sigma is not None:                  # NB: this adds a prior on f_X to the 'Planck' chi2
            logp -= 0.5 * (q["fX_star"] / self.fx_sigma) ** 2
        return logp

    def logp(self, H0, q0, j0, Om0, ombh2, _derived=None):
        """Return the log-likelihood at the sampled point (called by Cobaya).

        Parameters
        ----------
        H0, q0, j0, Om0, ombh2 : float
            Sampled parameters.
        _derived : dict, optional
            Filled with the derived ``fX_star``.

        Returns
        -------
        logp : float
            See ``_logp``.
        """
        return self._logp([H0, q0, j0, Om0], ombh2, _derived)


# ---------------------------------------------------------------- flat w0wa (CPL) versions
class PantheonLikelihoodCPL(PantheonLikelihood):
    """Pantheon+ supernovae for flat w0waCDM (CPL): samples H0, w0, wa and Om0."""

    input_params = ["H0", "w0", "wa", "Om0"]
    model = "cpl"
    _allowed = ("cpl",)

    def logp(self, H0, w0, wa, Om0, _derived=None):
        """Return the log-likelihood at the sampled point (called by Cobaya).

        Parameters
        ----------
        H0, w0, wa, Om0 : float
            Sampled parameters.
        _derived : dict, optional
            Unused; there are no derived parameters.

        Returns
        -------
        logp : float
            See ``PantheonLikelihood._logp``.
        """
        return self._logp([H0, w0, wa, Om0])


class DESILikelihoodCPL(DESILikelihood):
    """DESI BAO for flat w0waCDM (CPL): samples H0, w0, wa, Om0 and ombh2."""

    input_params = ["H0", "w0", "wa", "Om0", "ombh2"]
    model = "cpl"
    _allowed = ("cpl",)

    def logp(self, H0, w0, wa, Om0, ombh2, _derived=None):
        """Return the log-likelihood at the sampled point (called by Cobaya).

        Parameters
        ----------
        H0, w0, wa, Om0, ombh2 : float
            Sampled parameters.
        _derived : dict, optional
            Filled with the derived ``rd``.

        Returns
        -------
        logp : float
            See ``DESILikelihood._logp``.
        """
        return self._logp([H0, w0, wa, Om0], ombh2, _derived)


class DESIFreeRdLikelihoodCPL(DESIFreeRdLikelihood):
    """DESI BAO with a sampled r_d for flat w0waCDM (CPL): samples H0, w0, wa, Om0 and rd."""

    input_params = ["H0", "w0", "wa", "Om0", "rd"]
    model = "cpl"
    _allowed = ("cpl",)

    def logp(self, H0, w0, wa, Om0, rd, _derived=None):
        """Return the log-likelihood at the sampled point (called by Cobaya).

        Parameters
        ----------
        H0, w0, wa, Om0, rd : float
            Sampled parameters.
        _derived : dict, optional
            Unused; there are no derived parameters.

        Returns
        -------
        logp : float
            See ``DESIFreeRdLikelihood._chi2_logp``.
        """
        return self._chi2_logp([H0, w0, wa, Om0], rd)


class PlanckLikelihoodCPL(PlanckLikelihood):
    """Compressed Planck distance priors for flat w0waCDM (CPL): samples H0, w0, wa, Om0 and ombh2."""

    input_params = ["H0", "w0", "wa", "Om0", "ombh2"]
    model = "cpl"
    _allowed = ("cpl",)

    def logp(self, H0, w0, wa, Om0, ombh2, _derived=None):
        """Return the log-likelihood at the sampled point (called by Cobaya).

        Parameters
        ----------
        H0, w0, wa, Om0, ombh2 : float
            Sampled parameters.
        _derived : dict, optional
            Filled with the derived ``fX_star``.

        Returns
        -------
        logp : float
            See ``PlanckLikelihood._logp``.
        """
        return self._logp([H0, w0, wa, Om0], ombh2, _derived)


class DESY5LikelihoodCPL(PantheonLikelihoodCPL):
    """DES-SN5YR (Dovekie) supernovae for flat w0waCDM (CPL): samples H0, w0, wa and Om0."""

    def initialize(self):
        """Check the model, apply the z_match options and load the DES-SN5YR data."""
        _check_model(self.model, self._allowed, type(self).__name__)
        _apply_zmatch(self)
        self.z_hd, self.z_hel, self.mu, self.cov, self.inv_cov = load_desy5()