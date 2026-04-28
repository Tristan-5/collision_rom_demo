def llns_noise_amplitude(step_size: float, p_forward: float, dt: float) -> float:
    """
    Convert collision statistics into LLNS stochastic forcing amplitude.

    Returns noise amplitude scaled by the simulation timestep.
    """ 
sigma2 = collision_step_variance(step_size, p_forward)
    """
    Continuous-time scaling: variance per unit time (Langevin-style)
    """
    noise_amplitude = (sigma2 / dt) ** 0.5

    return noise_amplitude