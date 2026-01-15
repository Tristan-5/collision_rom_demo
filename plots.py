import matplotlib.pyplot as plt
import numpy
import os


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_velocity(velocity, savepath=None):
    plt.figure(figsize=(8,3.5))
    plt.plot(velocity, alpha=0.8)
    plt.xlabel("Collision events")
    plt.ylabel("Cumulative velocity")
    plt.title("Collision-generated velocity fluctuations")
    plt.tight_layout()
    if savepath:
        ensure_dir(os.path.dirname(savepath))
        plt.savefig(savepath, dpi=150)
        print(f"Saved {savepath}")
    plt.show()
    plt.close()

def plot_variance_comparison(t_emp, empirical_var, t_rom, predicted_var, savepath=None):
    plt.figure(figsize=(6,4))
    plt.plot(t_emp, empirical_var, label="Empirical variance")
    plt.plot(t_rom, predicted_var, linestyle="--", label="ROM prediction")
    plt.xlabel("Collision events")
    plt.ylabel("Variance")
    plt.legend()
    plt.title("ROM vs empirical variance growth")
    plt.tight_layout()
    if savepath:
        ensure_dir(os.path.dirname(savepath))
        plt.savefig(savepath, dpi=150)
        print(f"Saved {savepath}")
    plt.show()
    plt.close()



