import time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from .particles import Particles
from numba import jit, njit, prange, set_num_threads

"""
The N-Body Simulator class is responsible for simulating the motion of N bodies



"""


class NBodySimulator:

    def __init__(self, particles: Particles):
        self.particles = particles
        self.time = particles.time
        self.setup()
        return

    def setup(
        self,
        G=1,
        rsoft=0.01,
        method="RK4",
        io_freq=10,
        io_header="nbody",
        io_screen=True,
        visualization=False,
    ):
        """
        The simulation enviroments.

        G: graivtational constant
        rsoft: float, a soften length
        method: euler, RK2, RK4, kick(kick-drift-kick)
        io_freq: int, the frequency to outupt data.
                        io_freq <=0 for no output.
        io_header: the output header
        io_screen: print message on screen or not.
        visualization: on the fly visualization or not.
        """
        self.G = G
        self.rsoft = rsoft
        self.method = method
        self.io_header = io_header
        self.io_screen = io_screen
        self.visualization = visualization
        if io_freq <= 0:
            io_freq = np.inf
        self.io_freq = io_freq
        return

    def evolve(self, dt: float, tf: float, ti=0.0):
        """
        evolving

        dt: the time step
        ti: initial time
        tf: ending

        """

        t = ti
        self.particles.time = ti
        nt = int(np.ceil((tf - ti) / dt)) + 1  # 向上取整
        particles = self.particles
        particles.a = self._calculate_a(particles.N, particles.m, particles.r)

        for n in range(nt):
            if tf < t + dt:
                dt = tf - t
                # last step or wrong dt

            # check IO
            if (n % self.io_freq) == 0:
                self.particles.time = t
                # print info to screen
                if self.io_screen:
                    print("Time: ", t, "dt: ", dt)

                # check output directroy
                folder = "data_" + self.io_header
                Path(folder).mkdir(parents=True, exist_ok=True)

                # zfill(width): fill string to the width with 0 in front of it.
                fn = folder + "/" + self.io_header + "_" + str(n).zfill(6) + ".dat"
                self.particles.output(fn)

                # visualization
                if self.visualization:
                    particles.draw()

            particles = self._advance_particles(dt, particles)

            t += dt

        return

    def _advance_particles(self, dt, particles):

        method = self.method
        if method == "euler":
            particles = self._advance_particles_Euler(dt, particles)
        elif method == "RK2":
            particles = self._advance_particles_RK2(dt, particles)
        elif method == "RK4":
            particles = self._advance_particles_RK4(dt, particles)
        elif method == "kick":
            particles = self._advance_particles_kick(dt, particles)
        else:
            print("method must be euler, RK2, RK4, kick")
            raise (ValueError)

        return particles

    def _advance_particles_Euler(self, dt, particles):

        N = particles.N
        ms = particles.m
        # last moment data
        rs = particles.r
        vs = particles.v
        accs = self._calculate_a(N, ms, rs)

        # record this moment
        rs += vs * dt
        vs += accs * dt
        accs = self._calculate_a(N, ms, rs)
        particles.set_particles(rs, vs, accs)

        return particles

    def _advance_particles_RK2(self, dt, particles):

        N = particles.N
        ms = particles.m
        rs = particles.r
        # k1
        vs = particles.v
        accs = self._calculate_a(N, ms, rs)

        # k2
        rs2 = rs + vs * dt
        vs2 = vs + accs * dt
        accs2 = self._calculate_a(N, ms, rs2)

        rsn = rs + 0.5 * dt * (vs + vs2)
        vsn = vs + 0.5 * dt * (accs + accs2)
        accsn = self._calculate_a(N, ms, rsn)

        particles.set_particles(rsn, vsn, accsn)

        return particles

    def _advance_particles_RK4(self, dt, particles):

        N = particles.N
        ms = particles.m

        # k1
        rs = particles.r
        vs = particles.v
        accs = self._calculate_a(N, ms, rs)
        # k2
        rs2 = rs + 0.5 * dt * vs
        vs2 = vs + 0.5 * dt * accs
        accs2 = self._calculate_a(N, ms, rs2)
        # k3
        rs3 = rs + 0.5 * dt * vs2
        vs3 = vs + 0.5 * dt * accs2
        accs3 = self._calculate_a(N, ms, rs3)
        # k4
        rs4 = rs + dt * vs3
        vs4 = vs + dt * accs3
        accs4 = self._calculate_a(N, ms, rs4)

        rsn = rs + dt / 6 * (vs + 2 * vs2 + 2 * vs3 + vs4)
        vsn = vs + dt / 6 * (accs + 2 * accs2 + 2 * accs3 + accs4)
        accsn = self._calculate_a(N, ms, rsn)
        particles.set_particles(rsn, vsn, accsn)

        return particles

    def _advance_particles_kick(self, dt, particles):

        N = particles.N
        ms = particles.m
        rs = particles.r
        vs = particles.v
        accs = self._calculate_a(N, ms, rs)

        vs_half = vs + 0.5 * dt * accs
        rsn = rs + dt * vs_half
        accsn = self._calculate_a(N, ms, rsn)
        vsn = vs_half + 0.5 * dt * accsn
        particles.set_particles(rsn, vsn, accsn)

        return particles

    def _calculate_a(self, N, m, x):
        """
        Calculate the acceleration of the particles
        """
        G = self.G
        rsoft = self.rsoft

        # invoke the kernel for acceleration calculation
        accs = _calculate_a_kernel(N, m, x, rsoft, G)

        return accs


@njit(parallel=True)
def _calculate_a_kernel(N, m, rr, rsoft, G):
    """
    When use numba, simpler data is, faster progrom runs.
    Use more variables with smaller dim than less variables with larger dim.
    """
    x = rr[:, 0].copy()
    y = rr[:, 1].copy()
    z = rr[:, 2].copy()
    mass = m[:, 0].copy()

    acc = np.empty((N, 3), rr.dtype)

    for i in prange(N):
        ax = 0
        ay = 0
        az = 0
        for j in prange(N):
            if j != i:
                dx = x[j] - x[i]
                dy = y[j] - y[i]
                dz = z[j] - z[i]

                tmp = dx**2 + dy**2 + dz**2 + rsoft**2
                factor = mass[j] / (tmp * np.sqrt(tmp))

                ax += dx * factor
                ay += dy * factor
                az += dz * factor

        acc[i, 0] = G * ax
        acc[i, 1] = G * ay
        acc[i, 2] = G * az

    """
    Fij = (
        -G
        * np.outer(m[:, 0], m[:, 0])[:, :, np.newaxis]
        * rij
        / r[:, :, np.newaxis] ** 3
    )
    acc = np.sum(Fij, axis=1) / m"""

    return acc


if __name__ == "__main__":

    pass
