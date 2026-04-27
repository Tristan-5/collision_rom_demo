def llns_noise_amplitude(step_size: float, p_forward: float, dt: float) -> float:
    """
    Convert collision statistics into LLNS stochastic forcing amplitude.

    Returns noise amplitude scaled by the simulation timestep.
    """ 