import numpy as np


class OctTreeNode:
    def __init__(self, center, size):
        """
        center: the center of the mother node
        size: length of cubic
        """
        self._center = np.array(center)
        self.size = size
        self.particles = []
        self.children = [None] * 8  # 八個子節點

    def contains(self, particle):
        """
        check the particle is or not in the cubic
        particle: (tags of particle, particle position)
        """
        x, y, z = self._center
        hw = self.size / 2  # 節點半邊長的一半
        return (
            (x - hw) <= particle[1][0] <= x + hw
            and (y - hw) <= particle[1][1] <= y + hw
            and (z - hw) <= particle[1][2] <= z + hw
        )

    def subdivide(self):
        """
        將節點分成八個子節點
        """
        x, y, z = self._center
        hw = self.size / 2  # length of new node
        qh = self.size / 4

        self.children[0] = OctTreeNode((x - qh, y - qh, z - qh), hw)
        self.children[1] = OctTreeNode((x - qh, y - qh, z + qh), hw)
        self.children[2] = OctTreeNode((x - qh, y + qh, z - qh), hw)
        self.children[3] = OctTreeNode((x - qh, y + qh, z + qh), hw)
        self.children[4] = OctTreeNode((x + qh, y - qh, z - qh), hw)
        self.children[5] = OctTreeNode((x + qh, y - qh, z + qh), hw)
        self.children[6] = OctTreeNode((x + qh, y + qh, z - qh), hw)
        self.children[7] = OctTreeNode((x + qh, y + qh, z + qh), hw)

    def insert(self, particle):
        """
        插入粒子到八叉樹中
        """
        if not self.contains(particle):
            # 粒子不在範圍中
            return False

        if len(self.particles) == 0 and all(child is None for child in self.children):
            # 如果節點是空的，直接插入
            self.particles.append(particle)
            return True

        if len(self.particles) == 1 and all(child is None for child in self.children):
            # 如果節點只包含一個粒子，並且沒有子節點，則分裂節點
            self.subdivide()
            current_particle = self.particles[0]
            self.particles.clear()
            # 把粒子分類
            for child in self.children:
                child.insert(current_particle)
            self.insert(particle)
            return True

        # 否則插入到子節點中
        for child in self.children:
            if child.insert(particle):
                return True

        return False

    @property
    def center(self):
        return self._center


if __name__ == "__main__":
    root = OctTreeNode(center=(0, 0, 0), size=1.0)
    particles = [(1, np.array([0.5, 0.5, 0.5])), (2, np.array([-0.3, -0.3, -0.3]))]

    for particle in particles:
        root.insert(particle)

    print(f"Center of root node: {root.center}")
    print(f"Size of root node: {root.size}")
    print(f"Particles in root node: {root.particles}")

    # Check if a particle is contained within the root node
    particle_pos = np.array([0.2, 0.2, 0.2])
    print(
        f"Is particle at position {particle_pos} within the root node? {root.contains(particle_pos)}"
    )
