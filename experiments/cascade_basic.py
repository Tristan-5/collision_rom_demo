import numpy as np
import matplotlib.pyplot as plt

class MinimalShellModel:
    def __init__(self, N=10, k0=1.0, lam=2.0, nu=1e-3, forcing=0.1):
        self.N = N
        self.k = k0 * lam**np.arange(N)

        # small nonzero initial condition
        self.u = 0.01 * (np.random.rand(N) + 1j * np.random.rand(N))

        self.nu = nu
        self.f = np.zeros(N, dtype=complex)
        self.f[0] = forcing

    def rhs(self):
        du = np.zeros_like(self.u)

        for n in range(self.N):
            u_n = self.u[n]
            u_np1 = self.u[n+1] if n+1 < self.N else 0
            u_np2 = self.u[n+2] if n+2 < self.N else 0
            u_nm1 = self.u[n-1] if n-1 >= 0 else 0

            du[n] = self.k[n] * (u_np1 * u_np2 - 0.25 * u_nm1 * u_np1)
            du[n] -= self.nu * self.k[n]**2 * u_n
            du[n] += self.f[n]

        return du

    def step(self, dt):
        self.u += dt * self.rhs()

        # numerical stability: clip extremes
        self.u = np.clip(self.u.real, -1e2, 1e2) + 1j*np.clip(self.u.imag, -1e2, 1e2)

    def run(self, tmax, dt):
        nsteps = int(tmax / dt)
        traj = np.zeros((nsteps, self.N), dtype=complex)

        for i in range(nsteps):
            self.step(dt)
            traj[i] = self.u

        return traj


def plot_energy_spectrum(u):
    energy = np.abs(u) ** 2
    plt.loglog(energy)
    plt.xlabel("Shell index")
    plt.ylabel("Energy |u|^2")
    plt.title("Cascade prototype energy spectrum (experimental)")
    plt.show()


def plot_total_energy(traj):
    energy = np.sum(np.abs(traj) ** 2, axis=1)
    plt.plot(energy)
    plt.xlabel("Timestep")
    plt.ylabel("Total energy")
    plt.title("Total energy vs time")
    plt.show()


def main():
    model = MinimalShellModel(N=10, nu=1e-3, forcing=0.1)

    traj = model.run(tmax=50, dt=0.001)

    plot_energy_spectrum(traj[-1])
    plot_total_energy(traj)


if __name__ == "__main__":
    main()
