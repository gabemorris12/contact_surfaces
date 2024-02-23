import numpy as np


class MeshBody:
    def __init__(self, points, cells_dict, velocity=np.float64([0, 0, 0])):
        self.points, self.cells_dict, self.velocity = points, cells_dict, velocity
        self.surfaces, self.surface_count = [], np.zeros(self._surface_count(), dtype=np.int32)
        self.surface_dict = dict()
        self.elements = []

        self.nodes = np.array([Node(i, p, velocity) for i, p in enumerate(points)], dtype=object)

        self._construct_elements()
        self.surface_count = self.surface_count[self.surface_count > 0]

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
                elements = [self._construct_hexahedron(row, key, i) for i, row in enumerate(value)]
                self.elements.extend(elements)

    def _construct_hexahedron(self, con, elem_type, label):
        face1 = self.nodes[con[:4]]
        face1 = Surface(0, list(face1))
        face2 = self.nodes[con[4:]]
        face2 = Surface(0, list(face2))
        face3 = Surface(0, [face1[0], face2[0], face2[1], face1[1]])
        face4 = Surface(0, [face1[1], face2[1], face2[2], face1[2]])
        face5 = Surface(0, [face1[2], face2[2], face2[3], face1[3]])
        face6 = Surface(0, [face1[3], face2[3], face2[0], face1[0]])

        surfs = [face1, face2, face3, face4, face5, face6]

        for i, surf in enumerate(surfs):
            nodes = tuple(sorted([node.label for node in surf.nodes]))
            existing_surf = self.surface_dict.get(nodes)
            if existing_surf is not None:
                surfs[i] = existing_surf
                self.surface_count[existing_surf.label] += 1
            else:
                n = len(self.surfaces)
                surf.label = n
                self.surface_count[n] += 1
                self.surfaces.append(surf)
                self.surface_dict.update({nodes: surf})

        return Element(label, elem_type, self.nodes[con], surfs)

    def __repr__(self):
        return f'MeshBody(elements={len(self.elements)}, surfaces={len(self.surfaces)}, nodes={len(self.nodes)})'


class Node:
    def __init__(self, label, pos, vel):
        self.label, self.pos, self.vel = label, pos, vel

    def __sub__(self, other):
        return self.pos - other.pos

    def __repr__(self):
        return f'Node({self.label}, {self.pos}, {self.vel})'


class Surface:
    def __init__(self, label, nodes):
        self.label, self.nodes = label, nodes
        self.vecs = []
        self.points = np.array([node.pos for node in self.nodes])
        self.vel_points = np.array([node.vel for node in self.nodes])

        shifted_points = np.roll(self.points, -1, axis=0)
        vecs = shifted_points - self.points

        # noinspection PyUnreachableCode
        self.dir = np.cross(vecs[0], vecs[1])

    def reverse_dir(self):
        temp = self.nodes[1:]
        self.nodes = [self.nodes[0]] + list(reversed(temp))
        self.points = np.array([node.pos for node in self.nodes])
        self.dir = -self.dir

    def capture_box(self, dt):
        # added = [node.pos + dt*node.vel for node in self.nodes]
        # subtracted = [node.pos - dt*node.vel for node in self.nodes]
        # added = [node.pos + dt*max(abs(node.vel)) for node in self.nodes]
        # subtracted = [node.pos - dt*max(abs(node.vel)) for node in self.nodes]
        added = self.points + dt*np.max(np.abs(self.vel_points))
        subtracted = self.points - dt*np.max(np.abs(self.vel_points))
        bounds = np.concatenate((added, subtracted))

        xc_max, yc_max, zc_max = np.amax(bounds, axis=0)
        xc_min, yc_min, zc_min = np.amin(bounds, axis=0)
        return xc_max, yc_max, zc_max, xc_min, yc_min, zc_min

    def __eq__(self, other):
        return sorted([node.label for node in self.nodes]) == sorted([node.label for node in other.nodes])

    def __getitem__(self, item):
        return self.nodes[item]

    def __repr__(self):
        return f'Surface({self.label}, {[node.label for node in self.nodes]})'


class Element:
    def __init__(self, label, element_type, connectivity, surfaces):
        self.label, self.type, self.connectivity = label, element_type, connectivity
        self.surfaces = surfaces

        points = np.array([node.pos for node in connectivity])
        self.centroid = np.sum(points, axis=0)/len(points)
        self.outify_surfaces()

    def outify_surfaces(self):
        for surf in self.surfaces:
            internal_vec = self.centroid - surf[0].pos
            if np.dot(internal_vec, surf.dir) > 0:
                surf.reverse_dir()

    def __repr__(self):
        return f'Element({self.label}, {self.type!r}, {[node.label for node in self.connectivity]}, {[surf.label for surf in self.surfaces]})'


class GlobalMesh:
    def __init__(self, *MeshBodies, bs=0.1):
        self.mesh_bodies = MeshBodies
        self.elements = np.concatenate([mesh.elements for mesh in self.mesh_bodies])
        self.surfaces = np.concatenate([mesh.surfaces for mesh in self.mesh_bodies])
        self.nodes = np.concatenate([mesh.nodes for mesh in self.mesh_bodies])
        self.surface_count = np.concatenate([mesh.surface_count for mesh in self.mesh_bodies])
        self.points = np.concatenate([mesh.points for mesh in self.mesh_bodies])

        self.bs = bs

        e_start = len(self.mesh_bodies[0].elements)
        s_start = len(self.mesh_bodies[0].surfaces)
        n_start = len(self.mesh_bodies[0].nodes)

        for i in range(e_start, len(self.elements)):
            # noinspection PyUnresolvedReferences
            self.elements[i].label = i

        for i in range(s_start, len(self.surfaces)):
            # noinspection PyUnresolvedReferences
            self.surfaces[i].label = i

        for i in range(n_start, len(self.nodes)):
            # noinspection PyUnresolvedReferences
            self.nodes[i].label = i

        self.x_max, self.y_max, self.z_max = None, None, None
        self.x_min, self.y_min, self.z_min = None, None, None

        self.Sx = None
        self.Sy = None
        self.Sz = None

        self.nb = None
        self.n, _ = self.points.shape

        self.nbox, self.lbox, self.npoint, self.nsort = None, None, None, None

        self.sort()

    def sort(self):
        self.x_max, self.y_max, self.z_max = np.amax(self.points, axis=0)
        self.x_min, self.y_min, self.z_min = np.amin(self.points, axis=0)

        self.Sx = int((self.x_max - self.x_min)/self.bs) + 1
        self.Sy = int((self.y_max - self.y_min)/self.bs) + 1
        self.Sz = int((self.z_max - self.z_min)/self.bs) + 1

        self.nb = self.Sx*self.Sy*self.Sz

        self.nbox = np.zeros(self.nb, dtype=int)
        self.lbox = np.zeros(self.n, dtype=int)
        self.npoint = np.zeros(self.nb, dtype=int)
        self.nsort = np.zeros(self.n, dtype=int)

        for i, p in enumerate(self.points):
            x, y, z = p
            Si_x = int((x - self.x_min)/self.bs) + 1
            Si_y = int((y - self.y_min)/self.bs) + 1
            Si_z = int((z - self.z_min)/self.bs) + 1
            self.lbox[i] = (Si_z - 1)*self.Sx*self.Sy + (Si_y - 1)*self.Sx + Si_x - 1
            self.nbox[self.lbox[i]] += 1

        for j in range(1, self.nb):
            self.npoint[j] = self.npoint[j - 1] + self.nbox[j - 1]

        self.nbox[:] = 0

        for i in range(self.n):
            self.nsort[self.nbox[self.lbox[i]] + self.npoint[self.lbox[i]]] = i
            self.nbox[self.lbox[i]] += 1

    def surface_nodes(self, surface_id, dt):
        surf = self.surfaces[surface_id]

        (xc_max, yc_max, zc_max,
         xc_min, yc_min, zc_min) = surf.capture_box(dt)

        ibox_min = min(self.Sx, int((xc_min - self.x_min)/self.bs) + 1)
        jbox_min = min(self.Sy, int((yc_min - self.y_min)/self.bs) + 1)
        kbox_min = min(self.Sz, int((zc_min - self.z_min)/self.bs) + 1)
        ibox_max = min(self.Sx, int((xc_max - self.x_min)/self.bs) + 1)
        jbox_max = min(self.Sy, int((yc_max - self.y_min)/self.bs) + 1)
        kbox_max = min(self.Sz, int((zc_max - self.z_min)/self.bs) + 1)

        buckets, nodes = [], []
        for i in range(ibox_min, ibox_max + 1):
            for j in range(jbox_min, jbox_max + 1):
                for k in range(kbox_min, kbox_max + 1):
                    buckets.append((k - 1)*self.Sx*self.Sy + (j - 1)*self.Sx + i - 1)

        for bucket_id in buckets:
            nodes.extend(self.bucket_search(bucket_id))

        # Remove nodes in master surface
        # TODO: This needs to only include nodes that are attached to an external surface
        nodes = np.setdiff1d(nodes, [node.label for node in surf.nodes])

        return buckets, nodes

    def bucket_search(self, bucket_id):
        return self.nsort[self.npoint[bucket_id]:self.npoint[bucket_id] + self.nbox[bucket_id]]

    def contact_check(self, surface_id, node_id, dt, tol=1e-15):
        # Find the time it takes for the slave node to reach the master surface
        surf = self.surfaces[surface_id]
        slave = self.nodes[node_id]

        x1, y1, z1 = surf.nodes[0].pos
        x2, y2, z2 = surf.nodes[1].pos
        x3, y3, z3 = surf.nodes[2].pos
        x1_dot, y1_dot, z1_dot = surf.nodes[0].vel
        x2_dot, y2_dot, z2_dot = surf.nodes[1].vel
        x3_dot, y3_dot, z3_dot = surf.nodes[2].vel

        xs, ys, zs = slave.pos
        xs_dot, ys_dot, zs_dot = slave.vel

        b0 = -x1*y2*z3 + x1*y2*zs + x1*y3*z2 - x1*y3*zs - x1*ys*z2 + x1*ys*z3 + x2*y1*z3 - x2*y1*zs - x2*y3*z1 + x2*y3*zs + x2*ys*z1 - x2*ys*z3 - x3*y1*z2 + x3*y1*zs + x3*y2*z1 - x3*y2*zs - x3*ys*z1 + x3*ys*z2 + xs*y1*z2 - xs*y1*z3 - xs*y2*z1 + xs*y2*z3 + xs*y3*z1 - xs*y3*z2
        b1 = -x1_dot*y2*z3 + x1_dot*y2*zs + x1_dot*y3*z2 - x1_dot*y3*zs - x1_dot*ys*z2 + x1_dot*ys*z3 + x2_dot*y1*z3 - x2_dot*y1*zs - x2_dot*y3*z1 + x2_dot*y3*zs + x2_dot*ys*z1 - x2_dot*ys*z3 - x3_dot*y1*z2 + x3_dot*y1*zs + x3_dot*y2*z1 - x3_dot*y2*zs - x3_dot*ys*z1 + x3_dot*ys*z2 + xs_dot*y1*z2 - xs_dot*y1*z3 - xs_dot*y2*z1 + xs_dot*y2*z3 + xs_dot*y3*z1 - xs_dot*y3*z2 + y1_dot*x2*z3 - y1_dot*x2*zs - y1_dot*x3*z2 + y1_dot*x3*zs + y1_dot*xs*z2 - y1_dot*xs*z3 - y2_dot*x1*z3 + y2_dot*x1*zs + y2_dot*x3*z1 - y2_dot*x3*zs - y2_dot*xs*z1 + y2_dot*xs*z3 + y3_dot*x1*z2 - y3_dot*x1*zs - y3_dot*x2*z1 + y3_dot*x2*zs + y3_dot*xs*z1 - y3_dot*xs*z2 - ys_dot*x1*z2 + ys_dot*x1*z3 + ys_dot*x2*z1 - ys_dot*x2*z3 - ys_dot*x3*z1 + ys_dot*x3*z2 - z1_dot*x2*y3 + z1_dot*x2*ys + z1_dot*x3*y2 - z1_dot*x3*ys - z1_dot*xs*y2 + z1_dot*xs*y3 + z2_dot*x1*y3 - z2_dot*x1*ys - z2_dot*x3*y1 + z2_dot*x3*ys + z2_dot*xs*y1 - z2_dot*xs*y3 - z3_dot*x1*y2 + z3_dot*x1*ys + z3_dot*x2*y1 - z3_dot*x2*ys - z3_dot*xs*y1 + z3_dot*xs*y2 + zs_dot*x1*y2 - zs_dot*x1*y3 - zs_dot*x2*y1 + zs_dot*x2*y3 + zs_dot*x3*y1 - zs_dot*x3*y2
        b2 = -x1_dot*y2_dot*z3 + x1_dot*y2_dot*zs + x1_dot*y3_dot*z2 - x1_dot*y3_dot*zs - x1_dot*ys_dot*z2 + x1_dot*ys_dot*z3 + x1_dot*z2_dot*y3 - x1_dot*z2_dot*ys - x1_dot*z3_dot*y2 + x1_dot*z3_dot*ys + x1_dot*zs_dot*y2 - x1_dot*zs_dot*y3 + x2_dot*y1_dot*z3 - x2_dot*y1_dot*zs - x2_dot*y3_dot*z1 + x2_dot*y3_dot*zs + x2_dot*ys_dot*z1 - x2_dot*ys_dot*z3 - x2_dot*z1_dot*y3 + x2_dot*z1_dot*ys + x2_dot*z3_dot*y1 - x2_dot*z3_dot*ys - x2_dot*zs_dot*y1 + x2_dot*zs_dot*y3 - x3_dot*y1_dot*z2 + x3_dot*y1_dot*zs + x3_dot*y2_dot*z1 - x3_dot*y2_dot*zs - x3_dot*ys_dot*z1 + x3_dot*ys_dot*z2 + x3_dot*z1_dot*y2 - x3_dot*z1_dot*ys - x3_dot*z2_dot*y1 + x3_dot*z2_dot*ys + x3_dot*zs_dot*y1 - x3_dot*zs_dot*y2 + xs_dot*y1_dot*z2 - xs_dot*y1_dot*z3 - xs_dot*y2_dot*z1 + xs_dot*y2_dot*z3 + xs_dot*y3_dot*z1 - xs_dot*y3_dot*z2 - xs_dot*z1_dot*y2 + xs_dot*z1_dot*y3 + xs_dot*z2_dot*y1 - xs_dot*z2_dot*y3 - xs_dot*z3_dot*y1 + xs_dot*z3_dot*y2 - y1_dot*z2_dot*x3 + y1_dot*z2_dot*xs + y1_dot*z3_dot*x2 - y1_dot*z3_dot*xs - y1_dot*zs_dot*x2 + y1_dot*zs_dot*x3 + y2_dot*z1_dot*x3 - y2_dot*z1_dot*xs - y2_dot*z3_dot*x1 + y2_dot*z3_dot*xs + y2_dot*zs_dot*x1 - y2_dot*zs_dot*x3 - y3_dot*z1_dot*x2 + y3_dot*z1_dot*xs + y3_dot*z2_dot*x1 - y3_dot*z2_dot*xs - y3_dot*zs_dot*x1 + y3_dot*zs_dot*x2 + ys_dot*z1_dot*x2 - ys_dot*z1_dot*x3 - ys_dot*z2_dot*x1 + ys_dot*z2_dot*x3 + ys_dot*z3_dot*x1 - ys_dot*z3_dot*x2
        b3 = -x1_dot*y2_dot*z3_dot + x1_dot*y2_dot*zs_dot + x1_dot*y3_dot*z2_dot - x1_dot*y3_dot*zs_dot - x1_dot*ys_dot*z2_dot + x1_dot*ys_dot*z3_dot + x2_dot*y1_dot*z3_dot - x2_dot*y1_dot*zs_dot - x2_dot*y3_dot*z1_dot + x2_dot*y3_dot*zs_dot + x2_dot*ys_dot*z1_dot - x2_dot*ys_dot*z3_dot - x3_dot*y1_dot*z2_dot + x3_dot*y1_dot*zs_dot + x3_dot*y2_dot*z1_dot - x3_dot*y2_dot*zs_dot - x3_dot*ys_dot*z1_dot + x3_dot*ys_dot*z2_dot + xs_dot*y1_dot*z2_dot - xs_dot*y1_dot*z3_dot - xs_dot*y2_dot*z1_dot + xs_dot*y2_dot*z3_dot + xs_dot*y3_dot*z1_dot - xs_dot*y3_dot*z2_dot

        del_tc = np.roots([b3, b2, b1, b0])
        del_tc = del_tc[np.imag(del_tc) == 0]
        del_tc = del_tc[np.logical_and(del_tc >= 0, del_tc <= dt)]

        if del_tc.size == 0:
            return False, None, []
        else:
            del_tc = del_tc[0]

        contact_point = slave.pos + slave.vel*del_tc
        areas = []
        plane_points = surf.points + surf.vel_points*del_tc
        shifted = np.roll(plane_points, -1, axis=0)
        for p2, p1 in zip(plane_points, shifted):
            vec1 = p1 - p2
            vec2 = contact_point - p1
            # noinspection PyUnreachableCode
            cross = np.cross(vec1, vec2)

            if np.dot(cross, surf.dir) >= 0:
                s = 1
            else:
                s = -1

            area = s*1/2*np.linalg.norm(cross)
            area = 0. if -tol <= area <= tol else area

            areas.append(area)

        return all(np.array(areas) >= 0), del_tc, areas
