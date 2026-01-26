import numpy as np
import argparse
from collision_model import generate_collision_velocity
from rom import compute_rom_parameters, rom_prediction
from plots import plot_velocity, plot_variance_comparison
import os

DEFAULT_N = 5000
DEFAULT_STEP_SIZE = 1.0
DEFAULT_P_FORWARD = 0.55
DEFAULT_START_VAR = 10

def empirical_variance_series(
    velocity: np.ndarray,
    start: int = DEFAULT_START_VAR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    compute the empirical variance of the velocity signal as a function of sample size.

    The variance index i is computed using teh prefix velocity[:i], mimiking cumulative sampling in time.
    """
    empirical = [np.var(velocity[:i]) for i in range(start, len(velocity)+1)]
    t = np.arange(start, len(velocity)+1)
    return t, np.array(empirical)

def main(
    N=DEFAULT_N,
    step_size=DEFAULT_STEP_SIZE,
    p_forward=DEFAULT_P_FORWARD,
    seed=None,
    savefig=True,
):
    velocity = generate_collision_velocity(N=N, step_size=step_size, p_forward=p_forward, seed=seed)
    if savefig:
        plot_velocity(velocity, savepath="figures/velocity.png")
    else:
        plot_velocity(velocity, savepath=None)

    t_emp, empirical_var = empirical_variance_series(velocity, start=DEFAULT_START_VAR)
    assert len(t_emp) == len(empirical_var), "Empirical variance time series length mismatch"
    
    mean_v, var_v = compute_rom_parameters(velocity)
    t_rom, predicted_var = rom_prediction(len(t_emp), mean_v, var_v)

    if savefig:
        plot_variance_comparison(t_emp, empirical_var, t_emp, predicted_var, savepath="figures/variance_compare.png")
    else:
        plot_variance_comparison(t_emp, empirical_var, t_emp, predicted_var, savepath=None)

    print("ROM parameters:")
    print(f"  mean = {mean_v:.4f}, variance = {var_v:.4f}")
    print("Tip: re-run with different p_forward to see predictable changes in variance scaling.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collision-ROM demo")
    parser.add_argument("--N", type=int, default=5000)
    parser.add_argument("--step", type=float, default=1.0)
    parser.add_argument("--p", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-save", dest="save", action="store_false", help="Do not save figures")
    args = parser.parse_args()
    main(N=args.N, step_size=args.step, p_forward=args.p, seed=args.seed, savefig=args.save)





