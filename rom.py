import numpy as np

def compute_rom_parameters(velocity):
    """
    Compute simple ROM parameters from velocity signal.
    Returns mean and variance (scalars).
    """
    velocity=np.asarray(velocity)
    if velocity.ndim !=1:
        raise ValueError("velocity must be in an one-dimensional array")
        
    mean_v = float(np.mean(velocity))
    var_v  = float(np.var(velocity, ddof=1))
    return mean_v, var_v

def rom_prediction(N, mean_v, var_v):
    """
    Predict a simple variance growth curve for demonstration.
    Returns t (1..N) and predicted variance array.
    """
    if N <= 0:
        raise ValueError("N must be a positive integer")
        
    t = np.arange(1, N + 1)
    predicted_variance = var_v * (t / t[-1])
    return t, predicted_variance



