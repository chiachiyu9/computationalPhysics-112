import numpy as np
import matplotlib.pyplot as plt


class Particles:
    """
    Particle class to store particle properties
    N: number of particles
    r,v,a are position, velocity, acceleration in 3D.
    """

    def __init__(self, N):  # initial state when use this class
        self.N = N
        self._time = 0.0
        self._m = np.ones((N, 1), dtype=np.float16)
        self._r = np.zeros((N, 3))
        self._v = np.zeros((N, 3))
        self._a = np.zeros((N, 3))
        self._tags = np.arange(1, N + 1)
        return

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, t0: float):
        self._time = t0
        return

    @property
    def m(self):
        return self._m

    @m.setter
    def m(self, m0: np.ndarray):
        if m0.shape != (self.N, 1):  # (N,1)
            raise ValueError("Shape of masses must be (N,1)")
        self._m = m0
        return

    @property
    def r(self):
        return self._r

    @r.setter
    def r(self, r0: np.ndarray):
        if r0.shape != (self.N, 3):
            raise ValueError("Shape of positions must be (N,3)")
        self._r = r0
        return

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, v0: np.ndarray):
        if v0.shape != (self.N, 3):
            raise ValueError("Shape of velocities must be (N,3)")
        self._v = v0
        return

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, a0: np.ndarray):
        if a0.shape != (self.N, 3):
            raise ValueError("Shape of accelerations must be (N,3)")
        self._a = a0
        return

    @property
    def tags(self):
        return self._tags

    @tags.setter
    def tags(self, tag: np.ndarray):
        if tag.shape != np.arange(self.N).shape:
            raise ValueError("Number of tags must be equal to N")
        self._tags = tag
        return

    def add_particles(self, m, r, v, a):
        """
        Add N particles to the N-body simulation at once
        """
        self.N += m.shape[0]
        self.masses = np.vstack((self.masses, m))
        self.r = np.vstack((self.r, r))
        self.v = np.vstack((self.v, v))
        self.a = np.vstack((self.a, a))
        self.tags = np.arange(self.N)
        return

    def set_particles(self, r, v, a):
        """
        Set particle properties for the N-body simulation
        """
        self.r = r
        self.v = v
        self.a = a
        return

    def output(self, filename):
        """
        Output a txt file with particles properties.
        """
        t = self.time
        header = """
                columns are tag, mass, x, y, z, vx, vy, vz, ax, ay, az
                """
        header += "Time=%f\n" % (t)
        # N rows, ? columns.
        # np.reshape(-1,n): -1 can altomatically calculate numbers of rows.
        p = np.hstack(
            (
                self._tags.reshape(-1, 1),
                self._m,
                self._r,
                self._v,
                self._a,
            )
        )
        np.savetxt(filename, p, delimiter=",", header=header, fmt="%.6f")
        return

    def draw(self, dim=2):
        fig = plt.figure()
        if dim == 2:
            ax = fig.add_subplot(111)
            ax.scatter(self.r[:, 0], self.r[:, 1])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
        elif dim == 3:
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(self.r[:, 0], self.r[:, 1], self.r[:, 2])
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        else:
            print("Check dimension.")
            return

        ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()
        return fig, ax


"""
if __name__ == "__main__":
    p = Particles(3)
    tags = p.tags
    print(tags)
    # test error, should raise ValueError
    # p.tags = []
"""
