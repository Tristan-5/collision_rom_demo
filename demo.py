import numpy as np
import argparse
import os

from collision_model import generate_collision_velocity
from rom import compute_rom_parameters, rom_prediction
from plots import plot_velocity, plot_variance_comparison

DEFAULT_N = 5000
DEFAULT_STEP_SIZE = 1.0
DEFAULT_P_FORWARD = 0.55
DEFAULT_START_VAR = 10

def empirical_variance_series(
    velocity: np.ndarray,
    start: int = DEFAULT_START_VAR,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the empirical variance of the velocity signal as a function of sample size.

    The variance at index i is computed using the prefix velocity[:i],
    mimicking cumulative sampling in time.
    """
    velocity = np.asarray(velocity)

    if start < 2:
        raise ValueError("start must be >= 2")
    if start > velocity.size:
        raise ValueError("start must not exceed velocity length")

    empirical = [
        np.var(velocity[:i], ddof=1)
        for i in range(start, velocity.size + 1)
    ]
    t = np.arange(start, velocity.size +1)
    return t, np.array(empirical)

def main(
    N=DEFAULT_N,
    step_size=DEFAULT_STEP_SIZE,
    p_forward=DEFAULT_P_FORWARD,
    start_var=DEFAULT_START_VAR,
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

    t_emp, empirical_var = empirical_variance_series(
        velocity,
        start=start_var,
    )

    if len(t_emp) != len(empirical_var):
        raise RuntimeError("Empirical variance time series length mismatch")

    if empirical_var[-1] <= 0:
        raise RuntimeError("Final empirical variance must be positive")

    mean_v, var_v = compute_rom_parameters(velocity)

    t_rom, predicted_var = rom_prediction(len(t_emp), var_v)

    plot_variance_comparison(
        t_emp,
        empirical_var,
        t_emp,
        predicted_var,
        savepath="figures/variance_compare.png" if savefig else None,
    )

    print("ROM parameters:")
    print(f"  mean = {mean_v:.4f}, variance = {var_v:.4f}")
    print(
        "Tip: re-run with different p_forward to observe "
        "predictable changes in variance scaling with collision bias."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collision-ROM demo")
    parser.add_argument("--start-var", type=int, default=DEFAULT_START_VAR)
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
        step_size=args.step,
        p_forward=args.p,
        start_var=args.start_var,
        seed=args.seed,
        savefig=args.save,
    )




