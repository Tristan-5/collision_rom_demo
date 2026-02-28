import numpy as np

def compute_rom_parameters(velocity: np.ndarray) -> tuple[float, float]:
    """
    Compute ROM parameters from a velocity signal.
    
    Returns
    ------
    mean_v : float
        (mean of the velocity signal)
    var_v : float
        Sample variance of the velocity signal (ddof=1).
    """
    velocity=np.asarray(velocity)
    if velocity.ndim !=1:
        raise ValueError("velocity must be in an one-dimensional array")
        
    mean_v = float(np.mean(velocity))
    var_v  = float(np.var(velocity, ddof=1))
    return mean_v, var_v

def rom_prediction_physics(N: int, step_variance: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict variance growth using collision-derived step variance.

    For a random walk:
        Var[v_N] = N * sigma^2

    Therefore:
        Var[v(t)] = sigma^2 * t

    Parameters
    ------
    N: int
        Number of collision events.
    step_variance: float
        Variance contributed by a single collision step (derived analytically from collision statistics).

    Returns
    ------
    t_rom: ndarray
        Time index (1..N).
    predicted_variance: ndarray
        Linearly growing variance from microscopic physics.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer")
    if step_variance < 0:
        raise ValueError("step_variance must be non-negative")

    t_rom = np.arange(1, N+1)

    predicted_variance = step_variance * t_rom

    return t_rom, predicted_variance
