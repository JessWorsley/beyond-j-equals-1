import numpy as np
from calculate_distances import bao_distances

_KIND_INDEX = {"DM_over_rs": 0, "DH_over_rs": 1, "DV_over_rs": 2}


def bao_model_vector(theta, z_bao, kinds, model, rd):
    """Compute the DESI BAO theory vector in the same order as the data vector.

    Parameters
    ----------
    theta : sequence of float
        ``[H0, p1, p2, Om0]`` (see ``closure_cosmo.background_model``).
    z_bao : array_like
        Redshift of each measurement.
    kinds : list of str
        Quantity of each measurement: ``'DM_over_rs'``, ``'DH_over_rs'`` or
        ``'DV_over_rs'``.
    model : str
        One of ``closure_cosmo.ALL_MODELS``.
    rd : float
        Sound horizon at the drag epoch in Mpc.

    Returns
    -------
    vec : numpy.ndarray or None
        D_X / r_d for each measurement, or None for an invalid background.

    Raises
    ------
    ValueError
        If a quantity in `kinds` is not recognised.
    """
    result = bao_distances(theta, z_bao, model)       # (DM, DH, DV)
    if result is None:
        return None
    vec = np.empty(len(z_bao))
    for i, kind in enumerate(kinds):
        if kind not in _KIND_INDEX:
            raise ValueError(f"Unknown DESI BAO quantity: {kind}")
        vec[i] = result[_KIND_INDEX[kind]][i] / rd
    return vec if np.all(np.isfinite(vec)) else None
