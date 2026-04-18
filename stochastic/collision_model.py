import numpy as np

def generate_collision_steps(
    rng: np.random.Generator,
    N: int,
    step_size: float,
    p_forward: float,
) -> np.ndarray:
    """
    Generate discrete collision steps for random walk

    Returns
    ------
    ndarray
        Array of length N containing values +- step_size.
    """
    if not (0.0 <= p_forward <= 1):
        raise ValueError("p_forward must be between 0 and 1")

    if N <= 0:
        raise ValueError("N must be a positive integer")
    
    return rng.choice(
        np.array([-step_size, step_size], dtype=float),
        size=N,
        p=[1 - p_forward, p_forward],
)

def generate_collision_velocity(
    N: int =10000, 
    step_size: float =1.0, 
    p_forward: float =0.5, 
    seed: int | None =None,
)-> np.ndarray:
    """
    Generate velocity fluctuations using a binomial collision-based random walk.

    The parameter p_forward controls the probability of a positive
    velocity increment. Values different from 0.5 introduce statistical drift.

    Returns
    -------
    ndarray
        1D array of cumulative velocity.
    """
    rng = np.random.default_rng(seed)
    
    steps = generate_collision_steps(rng, N, step_size, p_forward)
    velocity = np.cumsum(steps)
    return velocity

def step_variance(step_size: float, p_forward: float) -> float:
    """
    Compute the variance contributed by a single collision step.
    Derived analytically from binomial statistics:
    Var(step)=4*p_forward*(1-p_forward)*step_size^2
    This is the microscopic variance that parameterizes the reduced-order stochastic model.

    Parameters
    ------
    step_size: float
        Magnitude of the velocity increment.
    p_forward: float
        Probability of a positive step.

    Returns
    ------
    float
        Variance per collision step.
    """
    if not (0.0 <= p_forward <= 1):
        raise ValueError("p_forward must be between 0 and 1")
    return 4.0 * p_forward * (1.0 - p_forward) * (step_size ** 2)
