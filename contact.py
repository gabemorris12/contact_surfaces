import numpy as np


class MeshBody:
    def __init__(self, points, cells_dict, velocity=np.float64([0, 0, 0])):
        self.points, self.cells_dict, self.velocity = points, cells_dict, velocity
        self.surfaces, self.surface_count = [], np.zeros(self._surface_count(), dtype=np.int32)

        self.nodes = [Node(i, p, velocity) for i, p in enumerate(points)]

        self._construct_elements()

    def __contains__(self, item):
        bool_list = [item == surf for surf in self.surfaces]
        if any(bool_list):
            return bool_list.index(True)
        else:
            return None

    def _surface_count(self):
        element_count = 0
        for key, value in self.cells_dict.items():
            if key == 'hexahedron':
                n_surfaces = 6
            else:
                raise NotImplementedError(f'Element type "{key}" is not implemented.')
            element_count += n_surfaces*len(value)
        return element_count

    def _construct_elements(self):
        for key, value in self.cells_dict.items():
            if key == 'hexahedron':
                elements = [self._construct_hexahedron(row) for row in value]

    def _construct_hexahedron(self, connectivity):
        face1 = list(connectivity[:4])
        face2 = list(connectivity[4:])

        v1 = self.points[face1[1]] - self.points[face1[0]]
        v2 = self.points[face1[2]] - self.points[face1[1]]
        v3 = self.points[face2[0]] - self.points[face1[0]]
        v4 = self.points[face2[1]] - self.points[face2[0]]
        v5 = self.points[face2[2]] - self.points[face2[0]]

        if np.dot(np.cross(v1, v2), v3) > 0:
            others = face1[1:]
            others.reverse()
            face1 = [face1[0]] + others

        if np.dot(np.cross(v4, v5), v3) < 0:
            others = face2[1:0]
            others.reverse()
            face2 = [face2[0]] + others


class Node:
    def __init__(self, label, pos, vel):
        self.label, self.pos, self.vel = label, pos, vel

    def __repr__(self):
        return f'Node({self.label}, {self.pos}, {self.vel})'


class Surface:
    def __init__(self, label, nodes):
        self.label, self.nodes = label, nodes

    def __eq__(self, other):
        return sorted(self.nodes) == sorted(other.nodes)


class Element:
    def __init__(self, label, element_type, connectivity):
        self.label, self.type, self.connectivity = label, element_type, connectivity
        self.surfaces = []
