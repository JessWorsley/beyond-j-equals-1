def calculate_inferred_parameters(q0, j0, Om0, model):
    """Compute the paper's derived constants from (q0, j0, Om0).

    Parameters
    ----------
    q0, j0, Om0 : float or array_like
        Deceleration, jerk and matter fraction today.
    model : str
        ``'model_1'``, ``'model_2'``, ``'model_3'`` or ``'cpl'``.

    Returns
    -------
    out : dict or None
        ``eps`` (closure parameter), ``c1`` and ``c2`` (integration constants
        of the paper), ``wDE0`` (w_DE at z = 0) and ``wDE0p`` (derivative of
        w_DE at z = 0).  None for ``'cpl'``.

    Raises
    ------
    ValueError
        If `model` is not known.
    """
    if model == 'model_1':
        eps = (j0 - 1) / (3 * (q0 + 1))
        c1 = (j0 - 2 * q0 ** 2 - q0) / (q0 + 1) ** 2
        c2 = Om0 * (j0 + 3 * q0 + 2) / (q0 + 1) ** 2
    elif model == 'model_2':
        s = (8 * j0 + 1) ** 0.5
        eps = j0 - 1
        c1 = (4 * s * q0 + 8 * j0 + s + 1) / (2 * (8 * j0 + 1))      # was incomplete before
        c2 = Om0
    elif model == 'model_3':
        eps = 2 * (1 - j0) / (3 * (1 - 2 * q0))
        c1 = 2 * (j0 - q0 - 2 * q0 ** 2) / (1 - 6 * q0 + 2 * j0)
        c2 = Om0
    elif model == 'cpl':
        return None
    else:
        raise ValueError(f"No model named {model}. Use 'model_1', 'model_2' or 'model_3'.")

    wDE0 = (1 - 2 * q0) / (3 * (Om0 - 1))
    wDE0p = -(2 * j0 * (Om0 - 1) + 2 * q0 * (2 * q0 - 3 * Om0 + 1) + Om0) / (3 * (Om0 - 1) ** 2)
    return {'eps': eps, 'c1': c1, 'c2': c2, 'wDE0': wDE0, 'wDE0p': wDE0p}
