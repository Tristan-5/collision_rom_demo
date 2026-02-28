import numpy as np
import argparse
import os

from collision_model import generate_collision_velocity, step_variance
from rom import compute_rom_parameters, rom_prediction_physics
from plots import plot_velocity, plot_variance_comparison

DEFAULT_N = 5000
DEFAULT_STEP_SIZE = 1.0
DEFAULT_P_FORWARD = 0.55
DEFAULT_START_VAR = 10
DEFAULT_M = 200

def ensemble_variance_series(
    N: int,
    M: int,
    step_size: float,
    p_forward: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ensemble variance across M independent realizations.

    Returns
    -------
    t : ndarray
        Time index (1..N)
    var_ensemble : ndarray
        Variance across realizations at each time step.
    """
    all_velocities = []

    for m in range(M):
        v = generate_collision_velocity(
            N=N,
            step_size=step_size,
            p_forward=p_forward,
            seed=None,
        )
        all_velocities.append(v)

    V = np.array(all_velocities)  # shape (M, N)

    var_ensemble = np.var(V, axis=0, ddof=1)
    t = np.arange(1, N + 1)

    return t, var_ensemble

def main(
    N=DEFAULT_N,
    M=DEFAULT_M,
    step_size=DEFAULT_STEP_SIZE,
    p_forward=DEFAULT_P_FORWARD,
    seed=None,
    savefig=True,
):
    """
    Run a collision-based random walk simulation and compare
    empirical variance growth against a ROM prediction.

    This script is a statistical demonstration rather than a
    calibrated physical turbulence model.
    """
    if savefig:
        os.makedirs("figures", exist_ok=True)

    velocity = generate_collision_velocity(
        N=N,
        step_size=step_size,
        p_forward=p_forward,
        seed=seed,
    )
    plot_velocity(
        velocity,
        savepath="figures/velocity.png" if savefig else None,
    )

    t_emp, empirical_var = ensemble_variance_series(
        N=N,
        M=M,
        step_size=step_size,
        p_forward=p_forward,
    )
    
    mean_v, var_v = compute_rom_parameters(velocity)

    sigma2 = step_variance(step_size, p_forward)

    t_rom, predicted_var = rom_prediction_physics(N, sigma2)
    
    plot_variance_comparison(
        t_emp,
        empirical_var,
        t_emp,
        predicted_var,
        savepath="figures/variance_compare.png" if savefig else None,
    )

    print("ROM parameters:")
    print(f"  ensemble realizations (M) = {M}")
    print(f"  physics-derived step variance = {sigma2:.4f}")
    print(f"  predicted final variance (theory) = {sigma2 * N:.4f}")
    print(f"  empirical final variance (ensemble) = {empirical_var[-1]:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collision-ROM demo")
    parser.add_argument("--M", type=int, default=DEFAULT_M)
    parser.add_argument("--N", type=int, default=DEFAULT_N)
    parser.add_argument("--step", type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--p", type=float, default=DEFAULT_P_FORWARD)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--no-save",
        dest="save",
        action="store_false",
        help="Do not save figures",
    )

    args = parser.parse_args()

    main(
        N=args.N,
        M=args.M,
        step_size=args.step,
        p_forward=args.p,
        seed=args.seed,
        savefig=args.save,
    )
