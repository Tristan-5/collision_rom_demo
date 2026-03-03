import matplotlib.pyplot as plt
import numpy as np
import os

DEFAULT_FIG_DPI = 150

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)

def plot_velocity(
    velocity: np.ndarray, 
    savepath: str | None = None
) -> None:
    plt.figure(figsize=(8, 3.5))
    plt.plot(velocity, alpha=0.8)
    plt.xlabel("Collision events")
    plt.ylabel("Cumulative velocity")
    plt.title("Collision-generated velocity fluctuations")
    plt.tight_layout()
    if savepath:
        ensure_dir(os.path.dirname(savepath))
        plt.savefig(savepath, dpi=DEFAULT_FIG_DPI)
        print(f"Saved {savepath}")
    plt.show()
    plt.close()

def plot_variance_comparison(
    t_emp: np.ndarray,
    empirical_var: np.ndarray,
    t_rom: np.ndarray,
    predicted_var: np.ndarray,
    savepath: str | None=None,
) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(t_emp, empirical_var, label="Empirical variance")
    plt.plot(t_rom, predicted_var, linestyle="--", label="ROM prediction")
    plt.xlabel("Collision events")
    plt.ylabel("Velocity variance")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.title("ROM vs empirical variance growth")
    plt.tight_layout()
    if savepath:
        ensure_dir(os.path.dirname(savepath))
        plt.savefig(savepath, dpi=DEFAULT_FIG_DPI)
        print(f"Saved {savepath}")
    plt.show()
    plt.close()

def plot_loglog_variance(
    t: np.ndarray,
    variance: np.ndarray,
    savepath: str | None = None,
) -> None:

    plt.figure(figsize=(6, 4))
    plt.loglog(t, variance)
    plt.xlabel("Collision events")
    plt.ylabel("Velocity variance")
    plt.title("Log–log variance scaling")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if savepath:
        ensure_dir(os.path.dirname(savepath))
        plt.savefig(savepath, dpi=DEFAULT_FIG_DPI)
        print(f"Saved {savepath}")

    plt.show()
    plt.close()

def plot_convergence(
    M_values: np.ndarray,
    errors: np.ndarray,
    savepath: str | None = None,
) -> None:

    plt.figure(figsize=(6, 4))
    plt.loglog(M_values, errors)
    ref = errors[0] * (M_values / M_values[0])**(-0.5)
    plt.loglog(M_values, ref, linestyle="--", label="M^{-1/2} reference")
    plt.legend()
    plt.xlabel("Ensemble size (M)")
    plt.ylabel("Relative error")
    plt.title("Ensemble convergence study")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    if savepath:
        ensure_dir(os.path.dirname(savepath))
        plt.savefig(savepath, dpi=DEFAULT_FIG_DPI)
        print(f"Saved {savepath}")

    plt.show()
    plt.close()

