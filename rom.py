import numpy as np

def compute_rom_parameters(velocity: np.ndarray) -> tuple[float, float]:
    """
    Compute ROM parameters from a velocity signal.
    Returns
    mean_v : float
        (mean of the velocity signal)
    var_v : float
        (variance of the velocity signal)
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
        
    t_rom = np.arange(1, N + 1)
    
    # Linear scaling ensures the predicted variance matches var_v at the final time
    predicted_variance = var_v * (t_rom / t_rom[-1])
    
    return t_rom, predicted_variance





