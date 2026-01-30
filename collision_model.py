import numpy as np

def generate_collision_steps(
    N: int,
    step_size: float,
    p_forward: float,
) -> np.ndarray:
    """
    Generate discrete collision steps for random walk
    """
    return np.random.choice(
        [-step_size, step_size],
        size=N
        p=[1 - p_forward, p_forward],
)

def generate_collision_velocity(N=10000, step_size=1.0, p_forward=0.5, seed=None)-> np.ndarray:
    """
    Generates velocity fluctuations using a binomial collision-based random walk.
    
    The parameter p_forward controls the probability of a positive velocity increment. Values different from 0.5 introduce a statistical drift in the resulting random walk.

    Returns a 1D numpy array of cumulative velocity steps.
    """
    if not (0.0 <= p_forward <= 1.0):
        raise ValueError("p_forward must be between 0 and 1")
    # This implements a biased binomial random walk in velocity space
    if seed is not None:
        np.random.seed(seed)
    steps = np.random.choice(
        [-step_size, step_size],
        size=N,
        p=[1 - p_forward, p_forward]
    )
    velocity = np.cumsum(steps)
    return velocity





