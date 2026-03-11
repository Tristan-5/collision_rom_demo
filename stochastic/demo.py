import numpy as np
import argparse
import os

from collision_model import generate_collision_velocity, step_variance
from rom import compute_rom_parameters, rom_prediction_physics
from plots import plot_velocity, plot_variance_comparison, plot_loglog_variance, plot_convergence

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

def convergence_study(
    N: int,
    M_values: list[int],
    step_size: float,
    p_forward: float,
    trials: int = 20,
) -> tuple[np.ndarray, np.ndarray]:

    sigma2 = step_variance(step_size, p_forward)
    true_var = sigma2 * N

    errors = []

    for M in M_values:

        trial_errors = []

        for _ in range(trials):

            _, var_emp = ensemble_variance_series(
                N=N,
                M=M,
                step_size=step_size,
                p_forward=p_forward,
            )

            rel_error = abs(var_emp[-1] - true_var) / true_var
            trial_errors.append(rel_error)

        errors.append(np.mean(trial_errors))

    return np.array(M_values), np.array(errors)

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

    plot_loglog_variance(
        t_emp,
        empirical_var,
        savepath="figures/variance_loglog.png" if savefig else None,
    )

    M_values = [20, 50, 100, 200, 400]
    M_vals, errors = convergence_study(
        N=N,
        M_values=M_values,
        step_size=step_size,
        p_forward=p_forward,
    )

    plot_convergence(
        M_vals,
        errors,
        savepath="figures/convergence.png" if savefig else None,
    )
    
    print("ROM parameters:")
    print(f"  ensemble realizations (M) = {M}")
    print(f"  physics-derived step variance = {sigma2:.4f}")
    print(f"  predicted final variance (theory) = {sigma2 * N:.4f}")
    print(f"  empirical final variance (ensemble) = {empirical_var[-1]:.4f}")

    rel_error = abs(empirical_var[-1] - sigma2 * N) / (sigma2 * N)
    
    print(f"  relative error = {100 * rel_error:.3f}%")
    
    slope = np.polyfit(t_emp, empirical_var, 1)[0]
    D_empirical = slope / 2.0
    D_theory = sigma2 / 2.0
    
    print(f"  theoretical diffusion coefficient D = {D_theory:.4f}")
    print(f"  empirical diffusion coefficient D = {D_empirical:.4f}")

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
