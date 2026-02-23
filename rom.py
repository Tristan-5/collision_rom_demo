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

def rom_prediction(N: int, var_v: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Predict a simple variance growth assuming linear-in-time scaling consistent with a random-walk process.
   
    Parameters
    ------
    N : int
        Number of collision events.
    var_v : float
        Final sample variance used to infer effective step variance.

    Returns
    ------
    t_rom : ndarray
        Time index (1..N).
    predicted_variance : ndarray
        Linearly scaled variance prediction.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer")
        
    t_rom = np.arange(1, N + 1)
    
    # For a random-walk process, variance grows linearly with time.
    predicted_variance = var_v * (t_rom / N)
    
    return t_rom, predicted_variance

