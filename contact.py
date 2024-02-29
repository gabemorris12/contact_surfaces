import numpy as np


class MeshBody:
    def __init__(self, points: np.ndarray, cells_dict: dict, velocity=np.float64([0, 0, 0])):
        """
        :param points: numpy.array; Nodal coordinates. The order determines the node labels starting at zero.
        :param cells_dict: dict; Cell information as constructed from meshio.Mesh.cells_dict.
        :param velocity: numpy.array; Velocity to be added to all the nodes as constructed.

        Additional Instance Variables
        -----------------------------
        surfaces: list; A list of surface objects (corresponds to their label)
        surface_count: numpy.array; An array consisting of the total count per surface. Where two elements meet, the
                       same surface object will be used. A surface with a count of one means that it is an external
                       surface. The index aligns with that of 'surfaces'.
        surface_dict: dict; A dictionary of surface objects. The keys are tuples consisting of node ids in numerical
                      order. This is useful for getting the surface from the known nodes.
        nodes: numpy.array; An array of node objects
        """
        self.points, self.cells_dict, self.velocity = points, cells_dict, velocity
        self.surfaces, self.surface_count = [], np.zeros(self._surface_count(), dtype=np.int32)
        self.surface_dict = dict()
        self.elements = []

        # noinspection PyTypeChecker
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
    def __init__(self, label: int, pos: np.ndarray, vel: np.ndarray):
        """
        :param label: int; The node id.
        :param pos: numpy.array; The coordinates of the node.
        :param vel: numpy.array; The vector velocity of the node.
        """
        self.label, self.pos, self.vel = label, pos, vel

    def __sub__(self, other):
        return self.pos - other.pos

    def __repr__(self):
        return f'Node({self.label}, {self.pos}, {self.vel})'


class Surface:
    def __init__(self, label: int, nodes: list):
        """
        :param label: int; The surface id.
        :param nodes: list; The list of node objects that define the surface.

        Additional Instance Variables
        -----------------------------
        points: numpy.array; An array of coordinate points corresponding to 'nodes'.
        vel_points: numpy.array; An array of velocity for each node in 'nodes'.
        dir: numpy.array; The direction of the surface. This should be outward for the external surfaces.
        """
        self.label, self.nodes = label, nodes
        self.points = np.array([node.pos for node in self.nodes])
        self.vel_points = np.array([node.vel for node in self.nodes])

        shifted_points = np.roll(self.points, -1, axis=0)
        vecs = shifted_points - self.points

        # noinspection PyUnreachableCode
        self.dir = np.cross(vecs[0], vecs[1])

    def reverse_dir(self):
        """
        Reverses the direction of the surface.
        """
        temp = self.nodes[1:]
        self.nodes = [self.nodes[0]] + list(reversed(temp))
        self.points = np.array([node.pos for node in self.nodes])
        self.dir = -self.dir

    def capture_box(self, vx_max, vy_max, vz_max, dt):
        """
        The capture box is used to determine which buckets penetrate the surface. The nodes in these buckets are
        considered for contact. The capture box is constructed based off the future position of the surface in the next
        time step. The nodes of the surface move with at a distance of [vx_max, vy_max, vz_max]*dt where the maximum
        values are determined from the velocity of all nodes. This will ensure that no nodes are missed. Refer to Figure
        11 in the Sandia Paper.

        :param vx_max: float; The max velocity in the x direction.
        :param vy_max: float; The max velocity in the y direction.
        :param vz_max: float; The max velocity in the z direction.
        :param dt: float; The time step of the analysis.
        :return: tuple; The bounding box parameters - xc_max, yc_max, zc_max, xc_min, yc_min, and zc_min.
        """
        # added = [node.pos + dt*node.vel for node in self.nodes]
        # subtracted = [node.pos - dt*node.vel for node in self.nodes]
        # added = [node.pos + dt*max(abs(node.vel)) for node in self.nodes]
        # subtracted = [node.pos - dt*max(abs(node.vel)) for node in self.nodes]
        vels = np.full(self.points.shape, np.float64([vx_max, vy_max, vz_max]))
        added = self.points + dt*vels
        subtracted = self.points - dt*vels
        bounds = np.concatenate((added, subtracted))

        xc_max, yc_max, zc_max = np.amax(bounds, axis=0)
        xc_min, yc_min, zc_min = np.amin(bounds, axis=0)
        return xc_max, yc_max, zc_max, xc_min, yc_min, zc_min

    # noinspection PyUnusedLocal
    def contact_check(self, node: Node, dt: float, tol=1e-15):
        """
        This is the detailed contact check that comes from section 3.2.1 in the Sandia paper.

        :param node: Node; The node object to be tested for contact.
        :param dt: float; The current time step in the analysis.
        :param tol: float; The tolerance of the area. If the calculated result is less than this, then it is considered
                    an area of zero.
        :return: tuple; Returns True or False indicated that the node will pass through the surface within the next time
                 step. Additionally, it returns the time until contact as well as the triangular areas that make up the
                 surface. If all areas are positive and the delta t is between 0 and dt, then the node will pass through
                 the surface.
        """

        # Find the time it takes for the slave node to reach the master surface
        # Finding delta t. Refer to the "Velocity Based Contact Check" file for the mathematical inspiration.
        x1, y1, z1 = self.nodes[0].pos
        x2, y2, z2 = self.nodes[1].pos
        x3, y3, z3 = self.nodes[2].pos
        x1_dot, y1_dot, z1_dot = self.nodes[0].vel
        x2_dot, y2_dot, z2_dot = self.nodes[1].vel
        x3_dot, y3_dot, z3_dot = self.nodes[2].vel

        xs, ys, zs = node.pos
        xs_dot, ys_dot, zs_dot = node.vel

        b0 = -x1*y2*z3 + x1*y2*zs + x1*y3*z2 - x1*y3*zs - x1*ys*z2 + x1*ys*z3 + x2*y1*z3 - x2*y1*zs - x2*y3*z1 + x2*y3*zs + x2*ys*z1 - x2*ys*z3 - x3*y1*z2 + x3*y1*zs + x3*y2*z1 - x3*y2*zs - x3*ys*z1 + x3*ys*z2 + xs*y1*z2 - xs*y1*z3 - xs*y2*z1 + xs*y2*z3 + xs*y3*z1 - xs*y3*z2
        b1 = -x1_dot*y2*z3 + x1_dot*y2*zs + x1_dot*y3*z2 - x1_dot*y3*zs - x1_dot*ys*z2 + x1_dot*ys*z3 + x2_dot*y1*z3 - x2_dot*y1*zs - x2_dot*y3*z1 + x2_dot*y3*zs + x2_dot*ys*z1 - x2_dot*ys*z3 - x3_dot*y1*z2 + x3_dot*y1*zs + x3_dot*y2*z1 - x3_dot*y2*zs - x3_dot*ys*z1 + x3_dot*ys*z2 + xs_dot*y1*z2 - xs_dot*y1*z3 - xs_dot*y2*z1 + xs_dot*y2*z3 + xs_dot*y3*z1 - xs_dot*y3*z2 + y1_dot*x2*z3 - y1_dot*x2*zs - y1_dot*x3*z2 + y1_dot*x3*zs + y1_dot*xs*z2 - y1_dot*xs*z3 - y2_dot*x1*z3 + y2_dot*x1*zs + y2_dot*x3*z1 - y2_dot*x3*zs - y2_dot*xs*z1 + y2_dot*xs*z3 + y3_dot*x1*z2 - y3_dot*x1*zs - y3_dot*x2*z1 + y3_dot*x2*zs + y3_dot*xs*z1 - y3_dot*xs*z2 - ys_dot*x1*z2 + ys_dot*x1*z3 + ys_dot*x2*z1 - ys_dot*x2*z3 - ys_dot*x3*z1 + ys_dot*x3*z2 - z1_dot*x2*y3 + z1_dot*x2*ys + z1_dot*x3*y2 - z1_dot*x3*ys - z1_dot*xs*y2 + z1_dot*xs*y3 + z2_dot*x1*y3 - z2_dot*x1*ys - z2_dot*x3*y1 + z2_dot*x3*ys + z2_dot*xs*y1 - z2_dot*xs*y3 - z3_dot*x1*y2 + z3_dot*x1*ys + z3_dot*x2*y1 - z3_dot*x2*ys - z3_dot*xs*y1 + z3_dot*xs*y2 + zs_dot*x1*y2 - zs_dot*x1*y3 - zs_dot*x2*y1 + zs_dot*x2*y3 + zs_dot*x3*y1 - zs_dot*x3*y2
        b2 = -x1_dot*y2_dot*z3 + x1_dot*y2_dot*zs + x1_dot*y3_dot*z2 - x1_dot*y3_dot*zs - x1_dot*ys_dot*z2 + x1_dot*ys_dot*z3 + x1_dot*z2_dot*y3 - x1_dot*z2_dot*ys - x1_dot*z3_dot*y2 + x1_dot*z3_dot*ys + x1_dot*zs_dot*y2 - x1_dot*zs_dot*y3 + x2_dot*y1_dot*z3 - x2_dot*y1_dot*zs - x2_dot*y3_dot*z1 + x2_dot*y3_dot*zs + x2_dot*ys_dot*z1 - x2_dot*ys_dot*z3 - x2_dot*z1_dot*y3 + x2_dot*z1_dot*ys + x2_dot*z3_dot*y1 - x2_dot*z3_dot*ys - x2_dot*zs_dot*y1 + x2_dot*zs_dot*y3 - x3_dot*y1_dot*z2 + x3_dot*y1_dot*zs + x3_dot*y2_dot*z1 - x3_dot*y2_dot*zs - x3_dot*ys_dot*z1 + x3_dot*ys_dot*z2 + x3_dot*z1_dot*y2 - x3_dot*z1_dot*ys - x3_dot*z2_dot*y1 + x3_dot*z2_dot*ys + x3_dot*zs_dot*y1 - x3_dot*zs_dot*y2 + xs_dot*y1_dot*z2 - xs_dot*y1_dot*z3 - xs_dot*y2_dot*z1 + xs_dot*y2_dot*z3 + xs_dot*y3_dot*z1 - xs_dot*y3_dot*z2 - xs_dot*z1_dot*y2 + xs_dot*z1_dot*y3 + xs_dot*z2_dot*y1 - xs_dot*z2_dot*y3 - xs_dot*z3_dot*y1 + xs_dot*z3_dot*y2 - y1_dot*z2_dot*x3 + y1_dot*z2_dot*xs + y1_dot*z3_dot*x2 - y1_dot*z3_dot*xs - y1_dot*zs_dot*x2 + y1_dot*zs_dot*x3 + y2_dot*z1_dot*x3 - y2_dot*z1_dot*xs - y2_dot*z3_dot*x1 + y2_dot*z3_dot*xs + y2_dot*zs_dot*x1 - y2_dot*zs_dot*x3 - y3_dot*z1_dot*x2 + y3_dot*z1_dot*xs + y3_dot*z2_dot*x1 - y3_dot*z2_dot*xs - y3_dot*zs_dot*x1 + y3_dot*zs_dot*x2 + ys_dot*z1_dot*x2 - ys_dot*z1_dot*x3 - ys_dot*z2_dot*x1 + ys_dot*z2_dot*x3 + ys_dot*z3_dot*x1 - ys_dot*z3_dot*x2
        b3 = -x1_dot*y2_dot*z3_dot + x1_dot*y2_dot*zs_dot + x1_dot*y3_dot*z2_dot - x1_dot*y3_dot*zs_dot - x1_dot*ys_dot*z2_dot + x1_dot*ys_dot*z3_dot + x2_dot*y1_dot*z3_dot - x2_dot*y1_dot*zs_dot - x2_dot*y3_dot*z1_dot + x2_dot*y3_dot*zs_dot + x2_dot*ys_dot*z1_dot - x2_dot*ys_dot*z3_dot - x3_dot*y1_dot*z2_dot + x3_dot*y1_dot*zs_dot + x3_dot*y2_dot*z1_dot - x3_dot*y2_dot*zs_dot - x3_dot*ys_dot*z1_dot + x3_dot*ys_dot*z2_dot + xs_dot*y1_dot*z2_dot - xs_dot*y1_dot*z3_dot - xs_dot*y2_dot*z1_dot + xs_dot*y2_dot*z3_dot + xs_dot*y3_dot*z1_dot - xs_dot*y3_dot*z2_dot

        # Solve the cubic equation. There could be three solutions, but there must be at least one real solution. If the
        # real solution is not between 0 and dt, then return False.
        del_tc = np.roots([b3, b2, b1, b0])
        del_tc = del_tc[np.imag(del_tc) == 0]
        del_tc = del_tc[np.logical_and(del_tc >= 0, del_tc <= dt)]

        if del_tc.size == 0:
            return False, None, []
        else:
            del_tc = np.real(del_tc[0])

        # Check if the contact point is within the bounds of the surface by computing the areas.
        contact_point = node.pos + node.vel*del_tc
        areas = []
        # The surface also needs to shifted to where the contact occurs.
        plane_points = self.points + self.vel_points*del_tc
        shifted = np.roll(plane_points, -1, axis=0)
        for p2, p1 in zip(plane_points, shifted):
            # Compute the cross product between a side of the surface and the vector going from the tip head of this
            # side vector to the contact point.
            vec1 = p1 - p2
            vec2 = contact_point - p1
            # noinspection PyUnreachableCode
            cross = np.cross(vec1, vec2)

            # If the cross product is in the direction of the surface direction, then it's a positive area.
            if np.dot(cross, self.dir) >= 0:
                s = 1
            else:
                s = -1

            area = s*1/2*np.linalg.norm(cross)  # area formula for cross products.
            area = 0. if -tol <= area <= tol else area

            areas.append(area)

        return all(np.array(areas) >= 0), del_tc, areas

    def __eq__(self, other):
        return sorted([node.label for node in self.nodes]) == sorted([node.label for node in other.nodes])

    def __getitem__(self, item):
        return self.nodes[item]

    def __repr__(self):
        return f'Surface({self.label}, {[node.label for node in self.nodes]})'


class Element:
    def __init__(self, label: int, element_type: str, connectivity: np.ndarray, surfaces: list):
        """
        :param label: int; The element id.
        :param element_type: str; The element type (i.e. hexahedron)
        :param connectivity: numpy.array; An array where each row consists of the nodes that make up the rectangular
                             faces of the element.
        :param surfaces: list; The list of surface objects.
        """
        self.label, self.type, self.connectivity = label, element_type, connectivity
        self.surfaces = surfaces

        points = np.array([node.pos for node in connectivity])
        self.centroid = np.sum(points, axis=0)/len(points)
        self.outify_surfaces()

    def outify_surfaces(self):
        """
        This makes sure that each surface of the element is pointed outward. This is necessary for defining a positive
        or negative area.
        """
        for surf in self.surfaces:
            internal_vec = self.centroid - surf[0].pos
            if np.dot(internal_vec, surf.dir) > 0:
                surf.reverse_dir()

    def __repr__(self):
        return f'Element({self.label}, {self.type!r}, {[node.label for node in self.connectivity]}, {[surf.label for surf in self.surfaces]})'


class GlobalMesh:
    def __init__(self, *MeshBodies, bs=0.1):
        """
        :param MeshBodies: MeshBody; The mesh body objects that make up the global contact check.
        :param bs: float; The bucket size of the mesh. This should be determined from the smallest master surface
                   dimension.

        Additional Instance Variables
        -----------------------------
        elements: numpy.array; An array of element objects.
        surfaces: numpy.array; An array of surface objects.
        nodes: numpy.array; An array of node objects.
        surface_count: numpy.array; An array consisting of the total count per surface. Where two elements meet, the
                       same surface object will be used. A surface with a count of one means that it is an external
                       surface. The index aligns with that of 'surfaces'.
        points: numpy.array; An array of nodal coordinates.
        velocities: numpy.array; An array of nodal velocities in the cartesian form.
        vx_max, vy_max, vz_max: float; Max absolute velocities from all nodes.
        x_max, y_max, ...: float; The maximum and minimum bounds for each axis.
        Sx, Sy, Sz: int; Number of buckets along the x, y, and z direction.
        n, nb: int; The number of nodes and buckets.
        nbox: numpy.array; An array consisting of the number of nodes that a bucket contains. For example, nbox[0] = 2
              would mean that bucket 0 has 2 nodes.
        lbox: numpy.array; An array consisting of the bucket id for each node. For example, lbox[5] = 12 would mean that
              node 5 is in bucket 12.
        nsort: numpy.array; An array consisting of sorted nodes based off the bucket id.
        npoint: numpy.array; An array consisting of indices into nsort where each index is the starting location. For
                example, nsort[npoint[5]] would return the starting node in bucket 5. This is a means for finding the
                nodes given a bucket id.
        """
        self.mesh_bodies = MeshBodies
        self.elements = np.concatenate([mesh.elements for mesh in self.mesh_bodies])
        self.surfaces = np.concatenate([mesh.surfaces for mesh in self.mesh_bodies])
        self.nodes = np.concatenate([mesh.nodes for mesh in self.mesh_bodies])
        self.surface_count = np.concatenate([mesh.surface_count for mesh in self.mesh_bodies])
        self.points = np.concatenate([mesh.points for mesh in self.mesh_bodies])
        self.velocities = np.array([node.vel for node in self.nodes])

        self.bs = bs

        vels = np.amax(np.abs(self.velocities), axis=0)
        for i, vel in enumerate(vels):
            if vel == 0:
                vels[i] = 1

        self.vx_max, self.vy_max, self.vz_max = vels

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
        """
        This is the start of the bucket search algorithm which constructs lbox, nbox, nsort, and npoint. The sorting
        algorithm is given in section 3.1.2 in the Sandia paper.
        """
        # Find the maximum and minimum bounds of the global bounding box.
        self.x_max, self.y_max, self.z_max = np.amax(self.points, axis=0)
        self.x_min, self.y_min, self.z_min = np.amin(self.points, axis=0)

        # Determine the number of buckets along each axis.
        self.Sx = int((self.x_max - self.x_min)/self.bs) + 1
        self.Sy = int((self.y_max - self.y_min)/self.bs) + 1
        self.Sz = int((self.z_max - self.z_min)/self.bs) + 1

        self.nb = self.Sx*self.Sy*self.Sz  # number of buckets

        # Initialize the datastructure.
        self.nbox = np.zeros(self.nb, dtype=int)
        self.lbox = np.zeros(self.n, dtype=int)
        self.npoint = np.zeros(self.nb, dtype=int)
        self.nsort = np.zeros(self.n, dtype=int)

        # Find the bucket id for each node by constructing lbox
        for i, p in enumerate(self.points):
            x, y, z = p
            Si_x = int((x - self.x_min)/self.bs) + 1
            Si_y = int((y - self.y_min)/self.bs) + 1
            Si_z = int((z - self.z_min)/self.bs) + 1
            self.lbox[i] = (Si_z - 1)*self.Sx*self.Sy + (Si_y - 1)*self.Sx + Si_x - 1
            self.nbox[self.lbox[i]] += 1  # increment nbox

        # Calculate the pointer for each bucket into a sorted list of nodes
        for j in range(1, self.nb):
            self.npoint[j] = self.npoint[j - 1] + self.nbox[j - 1]

        # Zero nbox
        self.nbox[:] = 0

        # Sort the slave nodes according to their bucket id into nsort.
        for i in range(self.n):
            self.nsort[self.nbox[self.lbox[i]] + self.npoint[self.lbox[i]]] = i
            self.nbox[self.lbox[i]] += 1

    def find_nodes(self, surface_id, dt):
        """
        Finds the nodes that could potentially contact a surface.

        :param surface_id: int; The surface id.
        :param dt: float; The time step.
        :return: tuple; A list of bucket ids and a list of node ids in each bucket.
        """
        surf = self.surfaces[surface_id]

        # Construct capture box.
        (xc_max, yc_max, zc_max,
         xc_min, yc_min, zc_min) = surf.capture_box(self.vx_max, self.vy_max, self.vz_max, dt)

        # Determine the buckets that intersect with the capture box.
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
        """
        :param bucket_id: int; The bucket id.
        :return: numpy.array; An array of node ids in the bucket.
        """
        return self.nsort[self.npoint[bucket_id]:self.npoint[bucket_id] + self.nbox[bucket_id]]

    def contact_check(self, surface_id: int, node_id: int, dt: float, tol=1e-15):
        """
        This is the detailed contact check that comes from section 3.2.1 in the Sandia paper.

        :param surface_id: int; The surface id.
        :param node_id: int; The node id.
        :param dt: float; The time step.
        :param tol: float; The tolerance of the area. If the calculated result is less than this, then it is considered
                    an area of zero.
        :return: tuple; Returns True or False indicated that the node will pass through the surface within the next time
                 step. Additionally, it returns the time until contact as well as the triangular areas that make up the
                 surface. If all areas are positive and the delta t is between 0 and dt, then the node will pass through
                 the surface.
        """
        surf = self.surfaces[surface_id]  # See contact_check from class surface.
        # noinspection PyUnresolvedReferences
        return surf.contact_check(self.nodes[node_id], dt, tol)
