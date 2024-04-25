import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import meshio

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)
stream_handler = logging.StreamHandler()
formatter = logging.Formatter('%(levelname)s:%(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class MeshBody:
    def __init__(self, points: np.ndarray, cells_dict: dict, velocity=np.float64([0, 0, 0]), mass=1.0):
        """
        :param points: numpy.array; Nodal coordinates. The order determines the node labels starting at zero.
        :param cells_dict: dict; Cell information as constructed from meshio.Mesh.cells_dict.
        :param velocity: numpy.array; Velocity to be added to all the nodes as constructed.
        :param mass: float; The mass of the nodes.

        Additional Instance Variables
        -----------------------------
        surfaces: list; A list of surface objects (corresponds to their label)
        surface_count: numpy.array; An array consisting of the total count per surface. Where two elements meet, the
                       same surface object will be used. A surface with a count of one means that it is an external
                       surface. The index aligns with that of 'surfaces'.
        surface_dict: dict; A dictionary of surface objects. The keys are tuples consisting of node ids in numerical
                      order. This is useful for getting the surface from the known nodes.
        get_element_by_surf: dict; A dictionary whose keys are surface objects and values are a list of element objects
                             corresponding to the element that contains the surface.
        nodes: numpy.array; An array of node objects
        """
        self.points, self.cells_dict, self.velocity, self.mass = points, cells_dict, velocity, mass
        self.surfaces, self.surface_count = [], np.zeros(self._surface_count(), dtype=np.int32)
        self.surface_dict = dict()
        self.get_element_by_surf = dict()
        self.elements = []

        if velocity.shape == (3,):
            velocity = np.full(points.shape, velocity)
        # noinspection PyTypeChecker
        self.nodes = np.array([Node(i, p, velocity[i], mass=mass) for i, p in enumerate(points)], dtype=object)

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

        elem = Element(label, elem_type, self.nodes[con], surfs)
        for surf in surfs:
            if self.get_element_by_surf.get(surf) is None:
                self.get_element_by_surf.update({surf: [elem]})
            else:
                self.get_element_by_surf[surf].append(elem)

        return elem

    def __repr__(self):
        return f'MeshBody(elements={len(self.elements)}, surfaces={len(self.surfaces)}, nodes={len(self.nodes)})'


class Node:
    def __init__(self, label: int, pos: np.ndarray, vel: np.ndarray, mass=1.0, corner_force=np.float64([0, 0, 0])):
        """
        :param label: int; The node id.
        :param pos: numpy.array; The coordinates of the node.
        :param vel: numpy.array; The vector velocity of the node.
        :param mass: float; The mass of the node
        :param corner_force: numpy.array; The corner force of the node.

        Optional Instance Variables
        ---------------------------
        ref: numpy.array; The reference coordinates for the node. This is used for the Newton-Raphson scheme.
        """
        self.label, self.pos, self.vel = label, pos, vel
        self.xi, self.eta, self.zeta = None, None, None
        self._ref = []

        self.mass = mass
        self.corner_force = corner_force
        self.contact_force = np.zeros((3,), dtype=np.float64)  # Force due to contact
        self.fc = 0  # placeholder for force increments

    def get_acc(self):
        """
        :return: The acceleration of the node
        """
        return (self.corner_force + self.contact_force)/self.mass

    def zero_contact(self):
        """
        Zero out the contact force.
        """
        self.contact_force = np.zeros((3,), dtype=np.float64)
        self.fc = 0

    @property
    def ref(self):
        return self._ref

    @ref.setter
    def ref(self, ref: list | np.ndarray):
        """
        Set the reference coordinates for the node.

        :param ref:
        :return:
        """
        if isinstance(ref, list):
            ref = np.array(ref)
        self._ref = ref
        self.xi, self.eta, self.zeta = ref

    @ref.getter
    def ref(self):
        return self._ref

    def __sub__(self, other):
        return self.pos - other.pos

    def __repr__(self):
        return f'Node({self.label}, {self.pos}, {self.vel})'


class Surface:
    def __init__(self, label: int, nodes: list[Node]):
        """
        :param label: int; The surface id.
        :param nodes: list; The list of node objects that define the surface.

        Additional Instance Variables
        -----------------------------
        points: numpy.array; An array of coordinate points corresponding to 'nodes'.
        vel_points: numpy.array; An array of velocity for each node in 'nodes'.
        dir: numpy.array; The direction of the surface. This should be outward for the external surfaces.
        ref_plane: tuple; A tuple consisting of the reference plane and the value of the reference plane. The reference
                   plane is represented by an integer (0 - xi, 1 - eta, 2 - zeta). The value must be either 1 or -1.
                   This does not get determined until _set_reference_plane is called.
        """
        self.label, self.nodes = label, nodes
        self.points = np.array([node.pos for node in self.nodes])
        self.vel_points = np.array([node.vel for node in self.nodes])

        shifted_points = np.roll(self.points, -1, axis=0)
        vecs = shifted_points - self.points

        # noinspection PyUnreachableCode
        self.dir = np.cross(vecs[0], vecs[1])
        self.ref_plane = None

        # Assume that the order of the nodes are always pointing outward and linear hex only. For higher order elements,
        # needs to change.
        self.xi_p, self.eta_p = np.array([-1, 1, 1, -1], dtype=np.float64), np.array([-1, -1, 1, 1], dtype=np.float64)

    def reverse_dir(self):
        """
        Reverses the direction of the surface.
        """
        temp = self.nodes[1:]
        self.nodes = [self.nodes[0]] + list(reversed(temp))
        self.points = np.array([node.pos for node in self.nodes])
        self.vel_points = np.array([node.vel for node in self.nodes])
        self.dir = -self.dir

        # if self.xi_p is not None and self.eta_p is not None:
        #     ref = np.array([node.ref for node in self.nodes])
        #     index = np.arange(3)
        #     ref = ref[:, index != self.ref_plane[0]]
        #     self.xi_p, self.eta_p = ref[:, 0], ref[:, 1]

    def construct_position_basis(self, del_t=0.0):
        """
        Construct the position basis matrix for the surface for getting the global position of the surface.

        ⎡p_{0x}  p_{1x}  p_{2x} ... p_{nx}⎤
        ⎢                                 ⎥
        ⎢p_{0y}  p_{1y}  p_{2y} ... p_{ny}⎥
        ⎢                                 ⎥
        ⎣p_{0z}  p_{1z}  p_{2z} ... p_{nz}⎦


        :param del_t: float; If the position of the surface at some time in the future is desired, then the time
                      increment should be provided.
        :return: np.array; The position basis matrix as detailed above.
        """

        p = np.copy(self.points)

        acc_points = np.array([node.get_acc() for node in self.nodes])

        if del_t:
            p = p + del_t*self.vel_points + 0.5*acc_points*del_t**2

        return p.transpose()

    def capture_box(self, vx_max, vy_max, vz_max, ax_max, ay_max, az_max, dt):
        """
        The capture box is used to determine which buckets penetrate the surface. The nodes in these buckets are
        considered for contact. The capture box is constructed based off the future position of the surface in the next
        time step. The nodes of the surface move with at a distance of [vx_max, vy_max, vz_max]*dt where the maximum
        values are determined from the velocity of all nodes. This will ensure that no nodes are missed. Refer to Figure
        11 in the Sandia Paper.

        :param vx_max: float; The max velocity in the x direction.
        :param vy_max: float; The max velocity in the y direction.
        :param vz_max: float; The max velocity in the z direction.
        :param ax_max: float; The max acceleration in the x direction.
        :param ay_max: float; The max acceleration in the y direction.
        :param az_max: float; The max acceleration in the z direction.
        :param dt: float; The time step of the analysis.
        :return: tuple; The bounding box parameters - xc_max, yc_max, zc_max, xc_min, yc_min, and zc_min.
        """
        vels = np.full(self.points.shape, np.float64([vx_max, vy_max, vz_max]))
        accs = np.full(self.points.shape, np.float64([ax_max, ay_max, az_max]))
        added = self.points + dt*vels + 0.5*accs*dt**2
        subtracted = self.points - dt*vels - 0.5*accs*dt**2
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

        # Find the centroid of the patch
        center = np.sum(self.points, axis=0)/len(self.points)
        vel_center = np.sum(self.vel_points, axis=0)/len(self.points)

        # Make triangle patches
        shift = np.roll(np.array(self.nodes, dtype=object), -1)  # Shifts all indices down one
        patches = [[n1, n2, Node(69, center, vel_center)] for n1, n2 in zip(self.nodes, shift)]

        # Loop through each patch, find the time if it exists, then determine whether the node is in the bounds of the
        # patch.
        for patch in patches:
            del_tc = find_time(patch, node, dt)

            if del_tc is not False:
                is_in, areas = check_in_bounds(patch, node, del_tc)
                if is_in:
                    return is_in, del_tc

        return False, None

    def get_contact_point(self, node: Node, guess: np.ndarray, tol=1e-10, max_iter=30):
        """
        Find the contact point in the reference space with the given slave node. The guess is an array consisting of
        [xi, eta, del_tc] if the surface is on the zeta=1 or -1 plane. It doesn't matter which reference plane the
        surface is on. This is automatically determined.

        :param node: Node; The slave node.
        :param guess: np.array; The initial guess for the Newton-Raphson scheme.
        :param tol: float; The tolerance for the Newton-Raphson scheme.
        :param max_iter: int; The maximum number of iterations for the Newton-Raphson scheme.
        :return: tuple; A tuple consisting of the converged reference coordinate up to 'max_iter' and the number of
                        iterations.
        """

        # if self.ref_plane is None:
        #     self._set_reference_plane()

        sol = guess
        acc_points = np.array([node.get_acc() for node in self.nodes])
        as_ = node.get_acc()

        for i in range(max_iter):
            # noinspection PyTypeChecker
            A = self.construct_position_basis(sol[2])
            xi, eta, del_t = sol
            phi_k_arr = phi_p_2D(xi, eta, self.xi_p, self.eta_p)
            rhs = A@phi_k_arr
            lhs = node.pos + node.vel*del_t + 0.5*as_*del_t**2
            F = rhs - lhs

            if np.linalg.norm(F) <= tol:
                break

            d_phi_d_xi = d_phi_p_2D_d_xi(eta, self.xi_p, self.eta_p)
            d_phi_d_eta = d_phi_p_2D_d_eta(xi, self.xi_p, self.eta_p)
            d_A_d_del_t = self.vel_points.transpose() + acc_points.transpose()*del_t

            J0 = A@d_phi_d_xi
            J1 = A@d_phi_d_eta
            J2 = d_A_d_del_t@phi_k_arr - node.vel - as_*del_t
            J = np.column_stack((J0, J1, J2))

            if 0 - tol <= np.linalg.det(J) <= 0 + tol:
                logger.info(f'Singularity calculated in contact check between patch {self.label} and '
                            f'node {node.label}. Backing out.')
                return sol, max_iter - 1

            sol = sol - np.linalg.inv(J)@F

        # noinspection PyUnboundLocalVariable
        return sol, i

    def find_fc(self, node: Node, guess: np.ndarray, dt: float, N: np.ndarray, tol=1e-12, max_iter=30):
        """
        Find the contact force increment for the current time step so that the node will not pass through the surface.

        :param node: Node; The slave node object.
        :param guess: np.array; The initial guess for the Newton-Raphson scheme (xi, eta, fc).
        :param dt: float; The current time step in the analysis.
        :param N: np.array; The unit normal vector, which is the direction of the force increment.
        :param tol: float; The tolerance for the Newton-Raphson scheme.
        :param max_iter: int; The maximum number of iterations for the Newton-Raphson scheme.
        :return: tuple; The solution (xi, eta, fc) and the number of iterations.
        """

        # if self.ref_plane is None:
        #     self._set_reference_plane()

        Fk = np.array([n.corner_force for n in self.nodes])
        mk = np.array([n.mass for n in self.nodes])
        Rk = np.array([n.contact_force for n in self.nodes])
        A_prime = self.points + self.vel_points*dt
        A_prime = A_prime.transpose() + (Fk + Rk).transpose()*dt**2/(2*mk)
        Fs, Rs, ms = node.corner_force, node.contact_force, node.mass
        vs, ps = node.vel, node.pos

        sol = guess

        for i in range(max_iter):
            xi, eta, fc = sol
            phi_k = phi_p_2D(xi, eta, self.xi_p, self.eta_p)
            A = A_prime - np.outer(N, fc/(2*mk)*phi_k*dt**2)

            # Compute F
            F = dt**2/(2*ms)*(Fs + N*fc + Rs) + dt*vs + ps - A@phi_k

            if np.linalg.norm(F) <= tol:
                break

            # Compute J
            d_phi_k_d_xi = d_phi_p_2D_d_xi(eta, self.xi_p, self.eta_p)
            d_phi_k_d_eta = d_phi_p_2D_d_eta(xi, self.xi_p, self.eta_p)
            d_A_d_xi = np.outer(-N, d_phi_k_d_xi*dt**2*fc/(2*mk))
            d_A_d_eta = np.outer(-N, d_phi_k_d_eta*dt**2*fc/(2*mk))
            d_A_d_fc = np.outer(-N, phi_k*dt**2/(2*mk))

            J0 = -A@d_phi_k_d_xi - d_A_d_xi@phi_k
            J1 = -A@d_phi_k_d_eta - d_A_d_eta@phi_k
            J2 = dt**2/(2*ms)*N - d_A_d_fc@phi_k
            J = np.column_stack((J0, J1, J2))

            if np.linalg.det(J) == 0:
                logger.warning(f'Singularity calculated in find force increment between patch {self.label} and '
                               f'node {node.label}. Returning zero force.')
                return np.float64([sol[0], sol[1], 0]), max_iter - 1

            sol = sol - np.linalg.inv(J)@F

        # noinspection PyUnboundLocalVariable
        return sol, i

    def find_glue_force(self, node: Node, guess: np.ndarray, dt: float, ref: np.ndarray, tol=1e-12, max_iter=30):
        """
        Find the contact force increment for the current time step so that the node will hit at the reference point
        "ref".

        :param node: Node; The slave node object.
        :param guess: np.array; The initial guess for the Newton-Raphson scheme (Gx, Gy, Gz). This is the glue force.
        :param dt: float; The current time step in the analysis.
        :param ref: np.array; The reference point (xi, eta) where the node will hit.
        :param tol: float; The tolerance for the Newton-Raphson scheme.
        :param max_iter: int; The maximum number of iterations for the Newton-Raphson scheme.
        :return: tuple; The solution (Gx, Gy, Gz) and the number of iterations.
        """

        # if self.ref_plane is None:
        #     self._set_reference_plane()

        Fk = np.array([n.corner_force for n in self.nodes])
        mk = np.array([n.mass for n in self.nodes])
        Rk = np.array([n.contact_force for n in self.nodes])
        A_prime = self.points + self.vel_points*dt
        A_prime = A_prime.transpose() + (Fk + Rk).transpose()*dt**2/(2*mk)
        Fs, Rs, ms = node.corner_force, node.contact_force, node.mass
        vs, ps = node.vel, node.pos

        sol = guess
        xi, eta = ref
        phi_k = phi_p_2D(xi, eta, self.xi_p, self.eta_p)

        for i in range(max_iter):
            A = A_prime - np.outer(sol, phi_k*dt**2/(2*mk))

            # Compute F
            F = dt**2/(2*ms)*(Fs + sol + Rs) + dt*vs + ps - A@phi_k

            if np.linalg.norm(F) <= tol:
                break

            # Compute J
            d_A_d_Gx = np.outer(-np.array([1, 0, 0]), phi_k*dt**2/(2*mk))
            d_A_d_Gy = np.outer(-np.array([0, 1, 0]), phi_k*dt**2/(2*mk))
            d_A_d_Gz = np.outer(-np.array([0, 0, 1]), phi_k*dt**2/(2*mk))

            J0 = -d_A_d_Gx@phi_k + np.array([dt**2/(2*ms), 0, 0])
            J1 = -d_A_d_Gy@phi_k + np.array([0, dt**2/(2*ms), 0])
            J2 = -d_A_d_Gz@phi_k + np.array([0, 0, dt**2/(2*ms)])
            J = np.column_stack((J0, J1, J2))

            sol = sol - np.linalg.inv(J)@F

        # noinspection PyUnboundLocalVariable
        return sol, i

    def normal_increment(self, nodes: list[Node], guesses: list[tuple], normals: list[np.ndarray], dt: float, tol=1e-12,
                         max_iter=30, ignore_off_edge=True):
        """
        Find and store the normal force increment for each node. This just does one iteration through all nodes.

        :param nodes: list; The list of node objects.
        :param guesses: list; The list of initial guesses (xi, eta, fc) corresponding to each node.
        :param normals: list; The list of normal vectors corresponding to each node. This is the direction that the
                        force increment will be applied.
        :param dt: float; The current time step in the analysis.
        :param tol: float; The tolerance for the Newton-Raphson scheme.
        :param max_iter: int; The maximum number of iterations for the Newton-Raphson scheme.
        :param ignore_off_edge: bool; If True, then a force will be returned regardless of whether the node lands off
                                the edge of the patch. If False, then a zero force will be returned and there will be
                                no force storage in the patch nodes.
        :return: list; A list of solutions (xi, eta, fc) for each node.
        """
        sols = []
        for node, guess, N in zip(nodes, guesses, normals):
            (xi, eta, fc), _ = self.find_fc(node, guess, dt, N, tol=tol, max_iter=max_iter)

            ref = np.float64([xi, eta])
            if not all(np.logical_and(ref >= -1 - tol, ref <= 1 + tol)) and not ignore_off_edge:
                fc = 0
                sols.append((xi, eta, fc))
                continue

            phi_k_arr = phi_p_2D(xi, eta, self.xi_p, self.eta_p)

            if node.fc + fc < 0:  # Force increment has become tensile
                node.contact_force += -N*node.fc

                for patch_node, phi_k in zip(self.nodes, phi_k_arr):
                    patch_node.contact_force += N*node.fc*phi_k

                node.fc = 0
                fc = 0
            else:
                node.fc += fc
                node.contact_force += N*fc

                for patch_node, phi_k in zip(self.nodes, phi_k_arr):
                    patch_node.contact_force += -N*fc*phi_k

            sols.append((xi, eta, fc))

        return sols

    def glue_increment(self, nodes: list[Node], guesses: list[tuple], refs: list[np.ndarray], dt: float, tol=1e-12,
                       max_iter=30):
        """
        Find and store the glue force increment for each node. This just does one iteration through all nodes.

        :param nodes: list; The list of node objects.
        :param guesses: list; The list of initial guesses (Gx, Gy, Gz) corresponding to each node.
        :param refs: list; The list of reference points (xi, eta) corresponding to each node.
        :param dt: float; The current time step in the analysis.
        :param tol: float; The tolerance for the Newton-Raphson scheme.
        :param max_iter: int; The maximum number of iterations for the Newton-Raphson scheme.
        :return:
        """
        sols = []
        for node, guess, ref in zip(nodes, guesses, refs):
            phi_k_arr = phi_p_2D(ref[0], ref[1], self.xi_p, self.eta_p)
            (Gx, Gy, Gz), _ = self.find_glue_force(node, guess, dt, ref, tol=tol, max_iter=max_iter)
            node.contact_force += np.array([Gx, Gy, Gz])

            for patch_node, phi_k in zip(self.nodes, phi_k_arr):
                patch_node.contact_force += -np.array([Gx, Gy, Gz])*phi_k

            sols.append((Gx, Gy, Gz))

        return sols

    def contact_check_through_reference(self, node: Node, dt: float, include_initial_penetration=False, tol=1e-12,
                                        max_iter=30):
        """
        This procedure finds the reference point and the delta_tc all at once using a Newton-Raphson scheme. If all the
        reference points are between -1 and 1, the delta_tc is between 0 and dt, and it took less than 25 iterations to
        solve then the node will pass through the patch.

        :param node: Node; The node object to be tested for contact.
        :param dt: float; The current time step in the analysis.
        :param include_initial_penetration: bool; If True, then if the node has already just penetrated the surface
                                            before time 0, then it will be considered. This is an alternative method
                                            to multiple stage detection.
        :param tol: float; The tolerance of the edge cases. For the end cases where either reference coordinate is
                    either 1 or -1, the tolerance will adjust for floating point error, ensuring that the edge case is
                    met.
        :param max_iter: int; The maximum number of iterations for the Newton-Raphson scheme.
        :return: tuple; Returns True or False indicating that the node will pass through the surface within the next
                 time step. Additionally, it returns the time until contact del_tc (if it's between 0 and dt), the
                 reference contact point (xi, eta), and the number of iterations for the solution.
        """

        # if self.ref_plane is None:
        #     self._set_reference_plane()

        # Compute centroid and slave at dt/2
        later_points = self.points + self.vel_points*dt/2 + 0.5*np.array([n.get_acc() for n in self.nodes])*(dt/2)**2
        centroid = np.mean(later_points, axis=0)
        slave_later = node.pos + node.vel*dt/2 + 0.5*node.get_acc()*(dt/2)**2

        # Construct the basis vectors
        A = self.construct_position_basis(dt/2)
        b1 = ref_to_physical((1, 0), A, self.xi_p, self.eta_p) - centroid
        b1 = b1/np.linalg.norm(b1)
        b2 = ref_to_physical((0, 1), A, self.xi_p, self.eta_p) - centroid
        b2 = b2/np.linalg.norm(b2)
        v = slave_later - centroid
        v = v/np.linalg.norm(v) if np.linalg.norm(v) != 0 else v
        A = np.array([b1, b2])
        xi_guess, eta_guess = A@v

        sol = self.get_contact_point(node, np.array([xi_guess, eta_guess, dt/2]), tol, max_iter=max_iter)
        ref = sol[0][:2]
        del_tc = sol[0][2]
        k = sol[1]

        a = -dt - tol if include_initial_penetration else 0 - tol
        if all(np.logical_and(ref >= -1 - tol, ref <= 1 + tol)) and a <= del_tc <= dt + tol and k <= max_iter - 2:
            del_tc = 0 if 0 - tol <= del_tc <= 0 + tol else del_tc
            del_tc = dt if dt - tol <= del_tc <= dt + tol else del_tc
            return True, del_tc, ref, k
        return False, del_tc, ref, k

    def contact_visual(self, axes: Axes3D, node: Node, dt: float, del_tc: float):
        """
        Generates a 3D plot of the contact check for visual confirmation.

        :param axes: Axes3D; A 3D axes object to generate the plot on.
        :param node: Node; The slave node object that is being tested.
        :param dt: float; The current time step in the analysis.
        :param del_tc: float; The delta time to contact.
        """

        # If there is any velocity, then plot the current state. If there is no velocity, then the future state is
        # the same as the current state.
        if any(self.vel_points.flatten()):
            self._decompose_and_plot(self.points, axes, decompose=False)

        # Plot the future state
        later_points = self.points + dt*self.vel_points
        self._decompose_and_plot(later_points, axes, patch_color='royalblue', point_color='navy')

        # Plot the slave node
        axes.scatter([node.pos[0]], [node.pos[1]], [node.pos[2]], color='lime')
        slave_later = node.pos + dt*node.vel
        axes.scatter([slave_later[0]], [slave_later[1]], [slave_later[2]], color='orangered', marker="^")
        axes.plot([node.pos[0], slave_later[0]], [node.pos[1], slave_later[1]],
                  [node.pos[2], slave_later[2]], color='black', ls='--')

        # Plot the intersection point and surface
        if del_tc and any(self.vel_points.flatten()):
            intersect_points = self.points + self.vel_points*del_tc
            self._decompose_and_plot(intersect_points, axes, patch_color='lightcoral',
                                     point_color='firebrick')
            for p1, p2 in zip(self.points, later_points):
                axes.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color='black', ls='--')

            contact = node.pos + node.vel*del_tc
            axes.scatter([contact[0]], [contact[1]], [contact[2]], color='firebrick', marker='x')
        elif del_tc:
            contact = node.pos + node.vel*del_tc
            axes.scatter([contact[0]], [contact[1]], [contact[2]], color='firebrick', marker='x')

    def contact_visual_through_reference(self, axes: Axes3D, nodes: Node | list[Node], dt: float,
                                         del_tc: float | None | list[float], only_contact=False, penetration=True,
                                         **kwargs):
        """
        Generates a 3D plot of the contact check for visual confirmation.

        :param axes: Axes3D; A 3D axes object to generate the plot on.
        :param nodes: Node or list of Nodes; The slave node objects that are being tested.
        :param dt: float; The current time step in the analysis.
        :param del_tc: float or list of floats; The delta time to contact corresponding to the Nodes list.
        :param only_contact: bool; Whether to only plot the contact point and surface.
        :param penetration: bool; Whether to plot the penetration components.
        """
        # If there is any velocity, then plot the current state. If there is no velocity, then the future state is
        # the same as the current state.
        if any(self.vel_points.flatten()) and not only_contact:
            self.project_surface(axes, 0, color='darkgrey', **kwargs)

        # Plot the future state
        if not only_contact:
            self.project_surface(axes, dt, color='navy', **kwargs)

        # Plot the node
        if isinstance(nodes, Node):
            nodes = [nodes]
            del_tc = [del_tc] if del_tc is not None else None
        else:
            assert isinstance(del_tc, list) or del_tc is None, 'List of del_tc not given despite list of nodes given.'
        for node in nodes:
            axes.scatter(node.pos[0], node.pos[1], node.pos[2], color='lime')
            slave_later = node.pos + dt*node.vel + 0.5*node.get_acc()*dt**2
            axes.scatter(slave_later[0], slave_later[1], slave_later[2], color='orangered', marker="^")
            # Discretize the line
            t_values = np.linspace(0, dt, 50)
            slave_path = []
            for t in t_values:
                slave_path.append(node.pos + node.vel*t + 0.5*node.get_acc()*t**2)
            slave_path = np.array(slave_path)
            axes.plot(slave_path[:, 0], slave_path[:, 1], slave_path[:, 2], color='black', ls='--')

        if del_tc is not None:
            for del_t, node in zip(del_tc, nodes):
                contact_point = node.pos + del_t*node.vel + 0.5*node.get_acc()*del_t**2
                axes.scatter(contact_point[0], contact_point[1], contact_point[2], color='gold', marker='x')

        if del_tc is not None and any(self.vel_points.flatten()):
            for del_t in del_tc:
                self.project_surface(axes, del_t, color='firebrick', **kwargs)

        if penetration and del_tc is not None:
            for del_t, node in zip(del_tc, nodes):
                slave_later = node.pos + dt*node.vel + 0.5*node.get_acc()*dt**2
                contact_point = node.pos + del_t*node.vel + 0.5*node.get_acc()*del_t**2
                ref, _ = self.get_contact_point(node, np.array([0.5, 0.5, del_t]))
                n = self.get_normal(ref[:2], del_t)
                print(f'Normal: {n} at {ref[:2]}')

                # noinspection PyUnboundLocalVariable
                p_vec = slave_later - contact_point  # Penetration vector
                p = np.dot(p_vec, n)*n  # Penetration projection
                normal_tip = contact_point + n*np.linalg.norm(p)  # Tip of the normal vector
                # penetration_depth = contact_point + p  # The endpoint of the penetration depth
                arrow_points = np.array([contact_point, normal_tip])
                # penetration_points = np.array([contact_point, penetration_depth, slave_later])
                arrow_prop_dict = dict(mutation_scale=20, arrowstyle='-|>', color='k', shrinkA=0, shrinkB=0)
                a = Arrow3D(arrow_points[:, 0], arrow_points[:, 1], arrow_points[:, 2], **arrow_prop_dict)
                axes.add_artist(a)
                # Don't need this for right now.
                # axes.plot(penetration_points[:, 0], penetration_points[:, 1], penetration_points[:, 2], 'k--')

    def project_surface(self, axes: Axes3D, del_t: float, N=9, alpha=0.25, color='navy', ls='--', show_grid=False,
                        triangulate=False):
        """
        Project the surface at time del_t later onto the given axes object.

        :param axes: Axes3D; The 3D axes object to project the surface onto.
        :param del_t: float; The time to project the surface.
        :param N: int; The size of the grid in each direction.
        :param alpha: float; The transparency of the surface.
        :param color: str; The color of the surface.
        :param ls: str; The line style of the surface.
        :param show_grid: bool; Whether to show the grid of the surface.
        :param triangulate: bool; Whether to use the triangulate method to generate a mesh.
        """
        # if self.ref_plane is None:
        #     self._set_reference_plane()

        A = self.construct_position_basis(del_t=del_t)
        xp, yp, zp = [], [], []
        for xi, eta in zip(self.xi_p, self.eta_p):
            x, y, z = ref_to_physical(np.array([xi, eta]), A, self.xi_p, self.eta_p)
            xp.append(x)
            yp.append(y)
            zp.append(z)

        x_values, y_values, z_values = [], [], []
        dim1 = np.linspace(-1, 1, N)
        dim2 = np.linspace(-1, 1, N)
        for c, xi in enumerate(dim1):
            for eta in dim2:
                x, y, z = ref_to_physical(np.array([xi, eta]), A, self.xi_p, self.eta_p)
                x_values.append(x)
                y_values.append(y)
                z_values.append(z)

        points = np.array([x_values, y_values, z_values])

        # Plotting the lines
        index = np.arange(0, N**2, N, dtype=int)
        for r, i in enumerate(index):
            # Plotting the column
            # noinspection PyTypeChecker
            axes.plot(x_values[i:i + N], y_values[i:i + N], z_values[i:i + N], color=color, ls=ls, alpha=alpha)

            # Plotting the row
            axes.plot(points[0, index + r], points[1, index + r], points[2, index + r], color=color, ls=ls,
                      alpha=alpha)

        if show_grid:
            axes.scatter(x_values, y_values, z_values, color=color, marker='.', alpha=1)

        if triangulate:
            triangle = tri.Triangulation(x_values, y_values)
            axes.plot_trisurf(triangle, z_values, color=color, alpha=alpha, linewidth=0)
        axes.scatter(xp, yp, zp, color=color, alpha=1)

    def get_normal(self, ref: np.ndarray, del_t: float):
        """
        Get the unit normal vector of the surface at the given reference point and time.

        :param ref: np.array; (xi, eta) coordinates.
        :param del_t: float; The instant in time to determine the normal.
        :return: np.array; The unit normal at the given reference point.
        """
        # The normal is defined as the cross product between dr/dxi and dr/deta where r is the position vector.
        xi, eta = ref

        d_phi_p_2D_d_xi_arr = d_phi_p_2D_d_xi(eta, self.xi_p, self.eta_p)
        d_phi_p_2D_d_eta_arr = d_phi_p_2D_d_eta(xi, self.xi_p, self.eta_p)
        A = self.construct_position_basis(del_t=del_t)
        dr_dxi = A@d_phi_p_2D_d_xi_arr
        dr_deta = A@d_phi_p_2D_d_eta_arr

        # noinspection PyUnreachableCode
        cross = np.cross(dr_dxi, dr_deta)
        return cross/np.linalg.norm(cross)

    def get_fc_guess(self, node: Node, N: np.ndarray, del_t: float, phi_k_arr: np.ndarray):
        """
        Compute the initial fc guess for normal forces.

        :param node: Node; The slave node object.
        :param N: np.array; The unit normal vector.
        :param del_t: float; The time increment.
        :param phi_k_arr: np.array; The array of phi_k values.
        :return: float; The initial guess for the normal force increment.
        """
        psn = np.dot(node.pos, N)
        vsn = np.dot(node.vel, N)
        Fsn = np.dot(node.corner_force, N)
        Rsn = np.dot(node.contact_force, N)
        ms = node.mass

        A_F = np.array([n.corner_force for n in self.nodes]).transpose()
        A_R = np.array([n.contact_force for n in self.nodes]).transpose()
        A_p = self.points.transpose()
        A_v = self.vel_points.transpose()
        pkn = np.dot(A_p@phi_k_arr, N)
        vkn = np.dot(A_v@phi_k_arr, N)
        Fkn = np.dot(A_F@phi_k_arr, N)
        Rkn = np.dot(A_R@phi_k_arr, N)
        m_avg = np.mean([n.mass for n in self.nodes])

        # @formatter:off
        return (Fkn*del_t**2*ms - Fsn*del_t**2*m_avg + Rkn*del_t**2*ms - Rsn*del_t**2*m_avg + 2*del_t*ms*m_avg*vkn - 2*del_t*ms*m_avg*vsn + 2*ms*m_avg*pkn - 2*ms*m_avg*psn)/(del_t**2*(ms + m_avg))
        # @formatter:on

    def zero_contact(self):
        """
        Zeroes out the contact forces for all the nodes in the surface.
        """
        for node in self.nodes: node.zero_contact()

    def physical_to_ref(self, point: np.ndarray, del_t: float, guess=np.float64([0, 0]), tol=1e-12, max_iter=30,
                        _omit=2):
        """
        Converts a physical point to a reference point on the surface. This is only valid if the point is on the
        surface.

        :param point: np.array; The physical point. Only include the x and y values as only two dimensions are needed.
        :param del_t: float; The time at which the surface is being projected.
        :param guess: np.array; The initial guess for the Newton-Raphson scheme.
        :param tol: float; The tolerance for the Newton-Raphson scheme.
        :param max_iter: int; The maximum number of iterations for the Newton-Raphson scheme.
        :param _omit: int; The dimension to omit. Only two equations are need, so only two dimensions are needed to
                      solve for the reference point. If the wrong dimensions are chosen, then the solution will result
                      in a singularity.
        :return: tuple; (xi, eta) coordinate and k iterations.
        """
        index = np.arange(3)
        i = index[index != _omit]

        sol = guess

        A = self.construct_position_basis(del_t)[i, :]
        p = point[i]

        for k in range(max_iter):

            xi, eta = sol
            phi_k_arr = phi_p_2D(xi, eta, self.xi_p, self.eta_p)
            F = A@phi_k_arr - p

            if np.linalg.norm(F) < tol:
                break

            d_phi_d_xi = d_phi_p_2D_d_xi(eta, self.xi_p, self.eta_p)
            d_phi_d_eta = d_phi_p_2D_d_eta(xi, self.xi_p, self.eta_p)
            J0 = A@d_phi_d_xi
            J1 = A@d_phi_d_eta
            J = np.column_stack((J0, J1))

            if np.linalg.det(J) == 0:
                logger.info(f'Singularity calculated in physical_to_ref for patch {self.label} and point {point}. '
                            f'This occurs when all points are co-planar. Trying again using difference dimensions.')
                return self.physical_to_ref(point, del_t, guess=guess, tol=tol, max_iter=max_iter, _omit=_omit - 1)

            sol = sol - np.linalg.inv(J)@F

        # noinspection PyUnboundLocalVariable
        return sol, k

    @staticmethod
    def _decompose_and_plot(points, axes, decompose=True, patch_color="darkgrey", point_color='navy'):
        # Decompose the surface into facets and plot
        centroid = np.sum(points, axis=0)/len(points) if decompose else []
        [axes.scatter([p[0]], [p[1]], [p[2]], color=point_color) for p in list(points) + [centroid] if len(p) != 0]

        if not decompose:
            # axes.plot(points[:, 0], points[:, 1], points[:, 2], color=patch_color)
            patch = Poly3DCollection([points], color=patch_color, alpha=0.25, lw=1.5)
            axes.add_collection(patch)
        else:
            shifted = np.roll(points, -1, axis=0)
            for p1, p2 in zip(points, shifted):
                patch = np.array([[p1, p2, centroid]])
                axes.plot(patch[0, :, 0], patch[0, :, 1], patch[0, :, 2], color=patch_color)
                patch = Poly3DCollection(patch, alpha=0.25, facecolor=patch_color)
                axes.add_collection(patch)

    def _set_reference_plane(self):
        logger.warning('This method is no longer required. The xi/eta points are defined at the initialization by the '
                       'assumption that the nodes are ordered correctly.')
        ref = np.array([node.ref for node in self.nodes])

        if ref.size == 0: raise RuntimeError('The reference coordinates have not been set.')

        for i in range(ref.shape[1]):
            unique_values = np.unique(ref[:, i])
            if len(unique_values) == 1:
                self.ref_plane = i, unique_values[0]

        if self.ref_plane is None: raise RuntimeError('Could not find the reference plane for the surface.')

        # Finding the xi_p and eta_p values
        index = np.arange(3)
        ref = ref[:, index != self.ref_plane[0]]
        self.xi_p, self.eta_p = ref[:, 0], ref[:, 1]

    def __eq__(self, other):
        return sorted([node.label for node in self.nodes]) == sorted([node.label for node in other.nodes])

    def __getitem__(self, item):
        return self.nodes[item]

    def __hash__(self):
        return hash(self.label)

    def __repr__(self):
        return f'Surface({self.label}, {[node.label for node in self.nodes]})'


class Element:
    def __init__(self, label: int, element_type: str, connectivity: np.ndarray, surfaces: list[Surface]):
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

    def set_node_refs(self):
        """
        Sets the node references coordinates xi, eta, and zeta for each node in the element.
        """
        assert self.type == 'hexahedron', 'Only hexahedron elements are supported.'
        for node in list(self.surfaces[0].nodes + self.surfaces[1].nodes):
            rel_pos = node.pos - self.centroid
            node.ref = np.sign(rel_pos)

    def __repr__(self):
        return f'Element({self.label}, {self.type!r}, {[node.label for node in self.connectivity]}, {[surf.label for surf in self.surfaces]})'


class GlobalMesh:
    def __init__(self, *MeshBodies, bs=0.1, master_patches=None, slave_nodes=None):
        """
        :param MeshBodies: MeshBody; The mesh body objects that make up the global contact check.
        :param bs: float; The bucket size of the mesh. This should be determined from the smallest master surface
                   dimension.
        :param master_patches: list; A list of integers corresponding to master patch IDs that are used for the contact
                               analysis. No nodes will be able to penetrate these patches. If this is None, then all the
                               external surfaces will be considered.
        :param slave_nodes: list; An integer list of slave node IDs that get paired to the master patches. Only the bank
                            of slave nodes will be considered.

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
        self.points = None
        self.velocities = None
        self.accelerations = None

        self.bs = bs

        self.vx_max, self.vy_max, self.vz_max = None, None, None
        self.ax_max, self.ay_max, self.az_max = None, None, None

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

        # Extending the get_element_by_surf dictionary
        self.get_element_by_surf = dict()
        for mesh in self.mesh_bodies:
            for surface, elements in mesh.get_element_by_surf.items():
                self.get_element_by_surf[surface] = elements

        self.get_patches_by_node = dict()  # {0: [0, 1, 2], 1: [3, 4, 5], ...}
        for patch in self.surfaces:
            for node in patch.nodes:
                if self.get_patches_by_node.get(node.label):
                    self.get_patches_by_node[node.label].append(patch.label)
                else:
                    self.get_patches_by_node[node.label] = [patch.label]

        self.x_max, self.y_max, self.z_max = None, None, None
        self.x_min, self.y_min, self.z_min = None, None, None

        self.Sx = None
        self.Sy = None
        self.Sz = None

        self.nb = None
        self.n = len(self.nodes)

        self.nbox, self.lbox, self.npoint, self.nsort = None, None, None, None

        self.master_patches = master_patches if master_patches else np.where(self.surface_count == 1)[0]
        self.slave_nodes = slave_nodes
        self.contact_pairs, self.get_pair_by_node, self.dynamic_pairs = [], {}, {}

        self.sort()

    def set_max_min(self):
        """
        Set the max and min velocity and acceleration.
        """

        self.points = np.array([node.pos for node in self.nodes])
        self.velocities = np.array([node.vel for node in self.nodes])
        self.accelerations = np.array([node.get_acc() for node in self.nodes])

        vels = np.amax(np.abs(self.velocities), axis=0)
        accs = np.amax(np.abs(self.accelerations), axis=0)
        for i, vel in enumerate(vels):
            if vel == 0:
                # When I get this into fierro, consider choosing a velocity that results in a distance that is half the
                # bucket size. I need access to the time step, which is not available here.
                vels[i] = 1

        self.vx_max, self.vy_max, self.vz_max = vels
        self.ax_max, self.ay_max, self.az_max = accs

    def sort(self):
        """
        This is the start of the bucket search algorithm which constructs lbox, nbox, nsort, and npoint. The sorting
        algorithm is given in section 3.1.2 in the Sandia paper.
        """

        self.set_max_min()

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
            Si_x = int((x - self.x_min)/self.bs)
            Si_y = int((y - self.y_min)/self.bs)
            Si_z = int((z - self.z_min)/self.bs)
            self.lbox[i] = Si_z*self.Sx*self.Sy + Si_y*self.Sx + Si_x
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
        surf: Surface = self.surfaces[surface_id]

        # Construct capture box.
        (xc_max, yc_max, zc_max,
         xc_min, yc_min, zc_min) = surf.capture_box(self.vx_max, self.vy_max, self.vz_max, self.ax_max, self.ay_max,
                                                    self.az_max, dt)

        # Determine the buckets that intersect with the capture box.
        ibox_min = max(0, min(self.Sx - 1, int((xc_min - self.x_min)/self.bs)))
        jbox_min = max(0, min(self.Sy - 1, int((yc_min - self.y_min)/self.bs)))
        kbox_min = max(0, min(self.Sz - 1, int((zc_min - self.z_min)/self.bs)))
        ibox_max = max(0, min(self.Sx - 1, int((xc_max - self.x_min)/self.bs)))
        jbox_max = max(0, min(self.Sy - 1, int((yc_max - self.y_min)/self.bs)))
        kbox_max = max(0, min(self.Sz - 1, int((zc_max - self.z_min)/self.bs)))

        buckets, nodes = [], []
        for i in range(ibox_min, ibox_max + 1):
            for j in range(jbox_min, jbox_max + 1):
                for k in range(kbox_min, kbox_max + 1):
                    buckets.append(k*self.Sx*self.Sy + j*self.Sx + i)

        for bucket_id in buckets:
            nodes.extend(self.bucket_search(bucket_id))

        # Remove nodes in master surface
        # TODO: This needs to only include nodes that are attached to an external surface
        nodes = np.setdiff1d(nodes, [node.label for node in surf.nodes])

        return buckets, nodes

    def bucket_search(self, bucket_id: int):
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

    def contact_check_through_reference(self, surface_id: int, node_id: int, dt: float,
                                        include_initial_penetration=False,
                                        tol=1e-12, max_iter=30):
        """
        This procedure finds the reference point and the delta_tc all at once using a Newton-Raphson scheme. If all the
        reference points are between -1 and 1 and the delta_tc is between 0 and dt, then the node will pass through the
        patch.

        :param surface_id: int; The surface id.
        :param node_id: int; The node id to be tested for contact.
        :param dt: float; The current time step in the analysis.
        :param include_initial_penetration: bool; If True, then if the node has already just penetrated the surface
                                            before time 0, then it will be considered. This is an alternative method
                                            to multiple stage detection.
        :param tol: float; The tolerance of the edge cases. For the end cases where either reference coordinate is
                    either 1 or -1, the tolerance will adjust for floating point error, ensuring that the edge case is
                    met.
        :param max_iter: int; The maximum number of iterations for the Newton-Raphson scheme.
        :return: tuple; Returns True or False indicating that the node will pass through the surface within the next
                 time step. Additionally, it returns the time until contact del_tc (if it's between 0 and dt), the
                 reference contact point (xi, eta), and the number of iterations for the solution.
        """
        surf = self.surfaces[surface_id]  # See contact_check from class surface.
        # noinspection PyUnresolvedReferences
        return surf.contact_check_through_reference(self.nodes[node_id], dt,
                                                    include_initial_penetration=include_initial_penetration,
                                                    tol=tol, max_iter=max_iter)

    def get_contact_pairs(self, dt: float, glue=False, include_initial_penetration=False, tol=1e-12, max_iter=30):
        """
        Get the contact pairs for the current time step.

        :param dt: float; The current time step.
        :param glue: bool; For glue contact pairs, some of the pairs will be omitted if a node becomes both master and
                     slave.
        :param include_initial_penetration: bool; If True, then if the node has already just penetrated the surface
                                            before time 0, then it will be considered. This is an alternative method
                                            to multiple stage detection.
        :param tol: float; The tolerance for the contact check.
        :param max_iter: int; The maximum number of iterations for the Newton-Raphson scheme.
        :return: list; A list of tuples where each tuple consists of (surface_id, node_id, (xi, eta, del_tc), iters).
        """
        contact_pairs, master_nodes, slave_nodes = [], [], []
        get_pair_by_node = {}
        for patch in self.master_patches:
            patch_obj = self.surfaces[patch]
            # The following three lines must be commented out to handle the elif is_hitting and node in master_nodes
            # condition
            # patch_nodes_used = [node.label in slave_nodes for node in patch_obj.nodes]
            # if all(patch_nodes_used):
            #     continue

            _, possible_nodes = self.find_nodes(patch, dt)
            for node in possible_nodes:
                if self.get_pair_by_node.get(node):
                    slave_nodes.append(node)
                    master_nodes.extend([node.label for node in patch_obj.nodes])
                    continue

                is_hitting, del_tc, (xi, eta), k = self.contact_check_through_reference(patch, node, dt,
                                                                                        include_initial_penetration=include_initial_penetration,
                                                                                        tol=tol,
                                                                                        max_iter=max_iter)
                if is_hitting and node not in master_nodes and node not in slave_nodes:
                    slave_nodes.append(node)
                    master_nodes.extend([node.label for node in patch_obj.nodes])
                    ref = np.float64([xi, eta])
                    # The normal direction is based off the patch position at time dt. In the future, it might be better
                    # to get this based off the minimum distance.
                    N = patch_obj.get_normal(ref, dt)
                    contact_pairs.append((patch, node, (xi, eta, del_tc), N, k))
                    get_pair_by_node[node] = (patch, (xi, eta, del_tc), N, k)
                elif is_hitting and node in slave_nodes:
                    patch_, (xi_, eta_, del_tc_), N_, k_ = get_pair_by_node[node]  # original

                    if del_tc + tol < del_tc_:  # This means that it will intersect with the current before the original
                        N = patch_obj.get_normal(np.float64([xi, eta]), dt)
                        contact_pairs.remove((patch_, node, (xi_, eta_, del_tc_), N_, k_))
                        contact_pairs.append((patch, node, (xi, eta, del_tc), N, k))
                        get_pair_by_node[node] = (patch, (xi, eta, del_tc), N, k)
                    elif del_tc_ - tol <= del_tc <= del_tc_ + tol:  # This means we are at an edge.
                        N = patch_obj.get_normal(np.float64([xi, eta]), dt)
                        pair = self.get_edge_pair((patch, node, (xi, eta, del_tc), N, k),
                                                  (patch_, node, (xi_, eta_, del_tc_), N_, k_), dt, tol=tol)
                        if pair[0] != patch_:
                            contact_pairs.remove((patch_, node, (xi_, eta_, del_tc_), N_, k_))
                            contact_pairs.append(pair)
                            get_pair_by_node[node] = (pair[0], *pair[2:])
                    elif del_tc > del_tc_:
                        pass
                    else:
                        raise RuntimeError('This should not happen.')
                elif is_hitting and node in master_nodes:
                    add_current_pair = False
                    hitting_after = []
                    # Remove contact that will occur after the current contact.
                    for patch_node in patch_obj.nodes:
                        pair = get_pair_by_node.get(patch_node.label)
                        if pair:
                            patch_, (xi_, eta_, del_tc_), N_, k_ = pair

                            if del_tc + tol < del_tc_:  # Master patch node is hitting after the current pair.
                                hitting_after.append(True)
                                if glue:
                                    contact_pairs.remove((patch_, patch_node.label, (xi_, eta_, del_tc_), N_, k_))
                                    slave_nodes.remove(patch_node.label)
                                    del get_pair_by_node[patch_node.label]
                            else:
                                hitting_after.append(False)

                    ref = np.float64([xi, eta])
                    if not hitting_after:
                        add_current_pair = True
                    elif all(hitting_after):
                        add_current_pair = True
                    elif any(hitting_after) and all(np.logical_and(ref > -1 + tol, ref < 1 - tol)):
                        add_current_pair = True

                    if add_current_pair:
                        N = patch_obj.get_normal(np.float64([xi, eta]), dt)
                        contact_pairs.append((patch, node, (xi, eta, del_tc), N, k))
                        get_pair_by_node[node] = (patch, (xi, eta, del_tc), N, k)
                        slave_nodes.append(node)

        self.contact_pairs.extend(contact_pairs)
        self.get_pair_by_node.update(get_pair_by_node)

        return contact_pairs

    def get_edge_pair(self, pair1: tuple, pair2: tuple, del_t: float, tol=1e-12):
        """
        If a contact point is found on an edge, then the better patch option needs to be determined. This function
        will return the better option based on the normal of the node.

        :param pair1: tuple; The first contact pair information.
        :param pair2: tuple; The second contact pair information.
        :param del_t: float; The time at which the normals are being compared.
        :param tol: float; The tolerance for equality check.
        :return: tuple; pair1 or pair2
        """
        patch1, node1, (xi1, eta1, del_tc1), N1, k1 = pair1
        patch2, node2, (_, _, _), N2, k2 = pair2

        assert node1 == node2, 'The nodes must be the same.'

        node = self.nodes[node1]
        N = self.get_node_normal(node, del_t)

        if np.dot(N, N2) - tol <= np.dot(N, N1) <= np.dot(N, N2) + tol:
            # Return the first patch with the normal being the average
            N_new = (N1 + N2)/2
            N_new = N_new/np.linalg.norm(N_new)
            return patch1, node1, (xi1, eta1, del_tc1), N_new, k1
        elif np.dot(N, N1) < np.dot(N, N2):
            return pair1
        elif np.dot(N, N1) > np.dot(N, N2):
            return pair2

    def get_edge_normal(self, ref: np.ndarray, surf: Surface, del_t: float, tol=1e-12,
                        max_iter=30):
        """
        Get the normal vector when the contact point lies on the edge of a patch.

        :param ref: np.array; The reference point.
        :param surf: Surface; The patch object.
        :param del_t: float; The desired time for which the normal is getting calculated.
        :param tol: float; The tolerance for the Newton-Raphson scheme.
        :param max_iter: int; The maximum number of iterations for the Newton-Raphson scheme.
        :return: np.array; The unit normal which is the average of the normals that are in the direction of the relative
                 velocity.
        """
        # For linear hex, we have this:
        # xi_p = [-1, 1, 1, -1]
        # eta_p = [-1, -1, 1, 1]
        xi, eta = ref
        nodes = np.array(surf.nodes, dtype=object)

        # We need to find the nodes that make up the edge or the node at the point.
        if -1 - tol <= xi <= -1 + tol:
            xi_index = surf.xi_p == -1
        elif 1 - tol <= xi <= 1 + tol:
            xi_index = surf.xi_p == 1
        else:
            xi_index = np.full(surf.xi_p.shape, False)

        if -1 - tol <= eta <= -1 + tol:
            eta_index = surf.eta_p == -1
        elif 1 - tol <= eta <= 1 + tol:
            eta_index = surf.eta_p == 1
        else:
            eta_index = np.full(surf.eta_p.shape, False)

        if any(xi_index) and any(eta_index):
            index = np.logical_and(xi_index, eta_index)
        elif any(xi_index):
            index = xi_index
        else:
            index = eta_index

        node_ids = [node.label for node in nodes[index]]

        # Get the shared patches between the node objects, but ensure that they are only external surfaces.
        assert len(node_ids) == 2 or len(node_ids) == 1, 'The edge normal must be along an edge.'
        if len(node_ids) == 2:
            patches = [self.get_patches_by_node[node_id] for node_id in node_ids]
            shared_patches = set(patches[0]).intersection(patches[1])
            patches = [patch for patch in shared_patches if self.surface_count[patch] == 1]
        else:
            patches = self.get_patches_by_node[node_ids[0]]
            patches = [patch for patch in patches if self.surface_count[patch] == 1]

        physical_point = ref_to_physical(ref, surf.construct_position_basis(del_t), surf.xi_p, surf.eta_p)
        avg = []
        for patch_id in patches:
            patch_obj: Surface = self.surfaces[patch_id]
            # Get the reference point relative to that surface
            rel_ref, _ = patch_obj.physical_to_ref(physical_point, del_t, tol=tol, max_iter=max_iter)
            N = patch_obj.get_normal(rel_ref, del_t)
            avg.append(N)

        if not avg:
            avg = surf.get_normal(ref, del_t)
            logger.warning(f'No normal direction found on edge with patch {surf.label}.')
        else:
            avg = np.mean(avg, axis=0)
            avg = avg/np.linalg.norm(avg)

        return avg

    def get_node_normal(self, node: Node, del_t: float):
        """
        Find the surface normal for a node. Only the external surfaces are considered.

        :param node: Node; The node object.
        :param del_t: float; The desired time for which the normal is getting calculated.
        :return: np.array; The unit normal vector which is the average of all the patch normals.
        """
        patches = self.get_patches_by_node[node.label]
        patches = [patch for patch in patches if self.surface_count[patch] == 1]

        normals = []
        for patch in patches:
            surf = self.surfaces[patch]
            ref = None

            for i, node_ in enumerate(surf.nodes):
                if node_.label == node.label:
                    ref = np.array([surf.xi_p[i], surf.eta_p[i]])
                    break

            normals.append(surf.get_normal(ref, del_t))

        N = np.mean(normals, axis=0)

        return N/np.linalg.norm(N)

    def get_dynamic_pair(self, ref: np.ndarray, surf: Surface, node: Node, dt: float, tol=1e-12, max_iter=30):
        signs = np.sign(ref)
        o_ref = np.copy(ref)  # original reference
        # Something here might need to be adjusted to where something in between -1 and 1 is preserved. This operation
        # brings the in between to zero, so the normals are going to be calculated at (0, 1), when a more accurate
        # representation would be to preserve it and have (0.554, 1).
        ref = np.floor(np.abs(ref) + tol)*signs
        # The sign function ensures that only adjacent patches are considered. It is assumed that a node will not move
        # more than the distance of an element.
        xi, eta = np.sign(ref)
        nodes = np.array(surf.nodes, dtype=object)

        # We need to find the nodes that make up the edge or the node at the point.
        if -1 - tol <= xi <= -1 + tol:
            xi_index = surf.xi_p == -1
        elif 1 - tol <= xi <= 1 + tol:
            xi_index = surf.xi_p == 1
        else:
            xi_index = np.full(surf.xi_p.shape, False)

        if -1 - tol <= eta <= -1 + tol:
            eta_index = surf.eta_p == -1
        elif 1 - tol <= eta <= 1 + tol:
            eta_index = surf.eta_p == 1
        else:
            eta_index = np.full(surf.eta_p.shape, False)

        corner_node = None
        if any(xi_index) and any(eta_index):
            index = np.logical_or(xi_index, eta_index)
            and_index = np.logical_and(xi_index, eta_index)
            # noinspection PyUnresolvedReferences
            corner_node = nodes[and_index][0].label
        elif any(xi_index):
            index = xi_index
        else:
            index = eta_index

        node_ids = [node.label for node in nodes[index]]
        physical_point = ref_to_physical(ref, surf.construct_position_basis(dt), surf.xi_p, surf.eta_p)

        # Get the shared patches between the node objects, but ensure that they are only external surfaces.
        assert len(node_ids) == 2 or len(node_ids) == 3, 'The node must pass a single edge.'
        if len(node_ids) == 2:
            # Crossing an edge
            patches = [self.get_patches_by_node[node_id] for node_id in node_ids]
            shared_patches = set(patches[0]).intersection(patches[1])
            patches = [patch for patch in shared_patches if self.surface_count[patch] == 1]
            patches.remove(surf.label)
            patch = patches[0]

            patch_obj: Surface = self.surfaces[patch]
            # Get the reference point relative to the new patch
            rel_ref, _ = patch_obj.physical_to_ref(physical_point, dt, tol=tol, max_iter=max_iter)

            if not is_concave(surf, patch_obj, ref, dt):
                # For the convex surface, return the adjacent surface no matter what.
                N = patch_obj.get_normal(rel_ref, dt)
            else:
                return surf.label, node.label, (o_ref[0], o_ref[1], None), surf.get_normal(o_ref, dt), None
        else:
            # Crossing a corner
            # We choose the patch that has the most opposing normal
            corner_node_patches = self.get_patches_by_node[corner_node]
            if len(corner_node_patches) == 3:  # Happens when it's the outer corner of the corner element
                return None
            other_patches = [patch for patch in corner_node_patches if self.surface_count[patch] == 1 and
                             patch != surf.label]
            patch1, patch2, patch3 = [self.surfaces[other_patches[i]] for i in range(3)]
            ref1, ref2, ref3 = (patch1.physical_to_ref(physical_point, dt, tol=tol, max_iter=max_iter)[0],
                                patch2.physical_to_ref(physical_point, dt, tol=tol, max_iter=max_iter)[0],
                                patch3.physical_to_ref(physical_point, dt, tol=tol, max_iter=max_iter)[0])
            N1, N2, N3 = (patch1.get_normal(ref1, dt), patch2.get_normal(ref2, dt), patch3.get_normal(ref3, dt))
            pair1, pair2, pair3 = (patch1.label, node.label, (ref1[0], ref1[1], None), N1, None), \
                (patch2.label, node.label, (ref2[0], ref2[1], None), N2, None), \
                (patch3.label, node.label, (ref3[0], ref3[1], None), N3, None)
            first = self.get_edge_pair(pair1, pair2, dt)
            last = self.get_edge_pair(first, pair3, dt)
            patch, _, new_ref, N, _ = last
            rel_ref = new_ref[:2]

            if is_concave(surf, self.surfaces[patch], ref, dt):
                return surf.label, node.label, (o_ref[0], o_ref[1], None), surf.get_normal(o_ref, dt), None

        return patch, node.label, (rel_ref[0], rel_ref[1], None), N, None

    def normal_increments(self, dt: float, tol=1e-12, max_iter=30):
        """
        Find the normal force across all patches and nodes in contact until there is no penetration.

        :param dt: float; The current time step.
        :param tol: float; The tolerance for all solving schemes. This affects a wider range of cases such as what is
                    considered a zero, the edge cases, and the convergence criteria.
        :param max_iter: int; The maximum number of iterations for all solving schemes.
        """
        if not self.contact_pairs:
            self.get_contact_pairs(dt, tol=tol, max_iter=max_iter)

        fc_list = []
        for iters1 in range(max_iter):

            if np.linalg.norm(fc_list) <= tol and fc_list:
                break

            fc_list.clear()

            for i, pair in enumerate(self.contact_pairs):
                surface_id, node_id, (xi, eta, del_tc), N, k = pair
                patch: Surface = self.surfaces[surface_id]
                node: Node = self.nodes[node_id]
                phi_k = phi_p_2D(xi, eta, patch.xi_p, patch.eta_p)

                # Construct the guess
                # This part needs to change for the fierro implementation. The guess should be based on the previous
                # contact solution.
                guess = (xi, eta, patch.get_fc_guess(node, N, dt, phi_k))
                [(xi, eta, fc)] = patch.normal_increment([node], [guess], [N], dt, tol=tol,
                                                         max_iter=max_iter, ignore_off_edge=True)

                ref = np.array([xi, eta])
                # if not all(np.logical_and(ref >= -1 - tol, ref <= 1 + tol)) and fc:
                if not all(np.logical_and(ref >= -1 - tol, ref <= 1 + tol)):
                    dynamic_pair = self.get_dynamic_pair(ref, patch, node, dt, tol=tol, max_iter=max_iter)
                    if dynamic_pair:
                        new_patch, _, (new_xi, new_eta, _), new_N, _ = dynamic_pair
                        self.dynamic_pairs[node_id] = (new_patch, (new_xi, new_eta, None), new_N, None)
                        self.contact_pairs[i] = dynamic_pair
                        self.get_pair_by_node[node_id] = (new_patch, (new_xi, new_eta, None), new_N, None)

                else:
                    # N = patch.get_normal(np.array([xi, eta]), dt)  # It's better to keep the normal direction the same.
                    self.contact_pairs[i] = (surface_id, node_id, (xi, eta, del_tc), N, k)
                    self.get_pair_by_node[node_id] = (surface_id, (xi, eta, del_tc), N, k)
                fc_list.append(fc)

        # Second stage detection might need to be added here.

        # noinspection PyUnboundLocalVariable
        return iters1

    def glue_increments(self, dt: float, tol=1e-12, max_iter=30):
        """
        Find the glue force across all patches and nodes until each contact pair rests on the glued contact point.

        :param dt: float; The current time step.
        :param tol: float; The tolerance for all solving schemes. This affects a wider range of cases such as what is
                    considered a zero, the edge cases, and the convergence criteria.
        :param max_iter: int; The maximum number of iterations for all solving schemes.
        :return:
        """
        if not self.contact_pairs:
            self.get_contact_pairs(dt, tol=tol, max_iter=max_iter)

        g_list = []
        for iters in range(max_iter):

            if np.linalg.norm(g_list) <= tol and g_list:
                return iters

            g_list.clear()

            for pair in self.contact_pairs:
                surface_id, node_id, (xi, eta, del_tc), N, _ = pair
                patch: Surface = self.surfaces[surface_id]
                node: Node = self.nodes[node_id]

                [(Gx, Gy, Gz)] = patch.glue_increment([node], [(1, 1, 1)], [np.array([xi, eta])],
                                                      dt, tol=tol, max_iter=max_iter)
                g_list.append(np.linalg.norm([Gx, Gy, Gz]))

        # noinspection PyUnboundLocalVariable
        return iters

    def remove_pairs(self, dt: float, tol=1e-12):
        """
        Remove appropriate pairs. Pairs are removed if the contact force is zero (that means a tensile force was
        required) or if the reference coordinate is out of bounds. The out-of-bounds case occurs when the node slides
        across a concave surface. These conditions should only be used for the normal force constraint. A different
        criteria should be implemented for the glue constraint.

        Additionally, the normals are updated for those pairs that are not to be removed.

        :param dt: float; The current time step. Used for getting the new normal.
        :param tol: float; The tolerance for the edge cases and zero condition.
        """

        for pair in self.contact_pairs[:]:  # looping over a copy
            patch, node, (xi, eta, del_tc), N, k = pair
            node_obj = self.nodes[node]
            ref = np.float64([xi, eta])

            if node_obj.fc == 0:
                self.contact_pairs.remove(pair)
                del self.get_pair_by_node[node]
            elif not all(np.logical_and(ref >= -1 - tol, ref <= 1 + tol)):
                self.contact_pairs.remove(pair)
                del self.get_pair_by_node[node]
            else:
                self.contact_pairs.remove(pair)

                patch_obj = self.surfaces[patch]
                N = patch_obj.get_normal(ref, dt)
                self.contact_pairs.append((patch, node, (xi, eta, None), N, None))
                self.get_pair_by_node[node] = (patch, (xi, eta, None), N, None)

    def update_nodes(self, dt: float):
        """
        Update the node positions and velocities. This should be done after the contact force has been determined.

        :param dt: float; The current time step.
        """
        for node in self.nodes:
            acc = node.get_acc()
            node.pos = node.pos + node.vel*dt + 0.5*acc*dt**2
            node.vel = node.vel + acc*dt

        for patch in self.surfaces:
            patch.points = np.array([node.pos for node in patch.nodes])
            patch.vel_points = np.array([node.vel for node in patch.nodes])

    def to_vtk(self, filename: str):
        """
        Write to a vtk file.

        :param filename: str; pathlike
        :return:
        """

        assert filename.endswith('.vtk'), 'The filename must end with .vtk.'

        connectivity = []
        for mesh in self.mesh_bodies:
            local_conn = mesh.cells_dict['hexahedron']
            for conn in local_conn:
                row = [mesh.nodes[c].label for c in conn]
                connectivity.append(row)

        cells_dict = {'hexahedron': connectivity}
        # noinspection PyTypeChecker
        mesh = meshio.Mesh(self.points, cells_dict)
        mesh.write(filename)

    def to_geo(self, filename: str, desc1=None, desc2=None):
        """
        Write to a geo file.

        :param filename: str; pathlike
        :param desc1: Description for the first line.
        :param desc2: Description for the second line.
        :return:
        """
        desc1 = desc1 if desc1 else "Description line 1"
        desc2 = desc2 if desc2 else "Description line 2"

        assert filename.endswith('.geo'), 'The filename must end with .geo.'

        out = f"""{desc1}
{desc2}
node id assign
element id assign
part
{1:>10}
Mesh
coordinates
{len(self.points):>10}
"""

        x_vals, y_vals, z_vals = self.points[:, 0], self.points[:, 1], self.points[:, 2]
        for x in x_vals:
            out += f'{x:.5e}\n'
        for y in y_vals:
            out += f'{y:.5e}\n'
        for z in z_vals:
            out += f'{z:.5e}\n'

        # noinspection PyUnresolvedReferences
        n = len(self.surfaces[0].nodes)*2  # Number of nodes in a surface
        n_elem = len(self.elements)  # Number of elements
        out += f'hexa{n}\n'
        out += f'{n_elem:>10}\n'

        connectivity = []
        for mesh in self.mesh_bodies:
            local_conn = mesh.cells_dict['hexahedron']
            for conn in local_conn:
                row = [mesh.nodes[c].label for c in conn]
                connectivity.append(row)

        for r in connectivity:
            for c in r:
                out += f'{c + 1:>10}'
            out += '\n'

        with open(filename, 'w') as f:
            f.write(out)


def find_time(patch_nodes: list[Node], slave_node: Node, dt: float) -> float:
    """
    Find the time it takes for a slave node to penetrate a triangular facet.

    :param patch_nodes: list; A list of three node objects that define the plane.
    :param slave_node: Node; The slave node to determine the time of penetration.
    :param dt: float; The current time step of the analysis.
    :return: float | bool; The time between 0 and dt of the intersection or False it doesn't exist or not between
             0 <= del_tc <= dt.
    """
    # Find the time it takes for the slave node to reach the triangular patch
    # Finding delta t, Refer to the "Velocity Based Contact Check" file for the mathematical derivation.
    x1, y1, z1 = patch_nodes[0].pos
    x2, y2, z2 = patch_nodes[1].pos
    x3, y3, z3 = patch_nodes[2].pos
    x1_dot, y1_dot, z1_dot = patch_nodes[0].vel
    x2_dot, y2_dot, z2_dot = patch_nodes[1].vel
    x3_dot, y3_dot, z3_dot = patch_nodes[2].vel

    xs, ys, zs = slave_node.pos
    xs_dot, ys_dot, zs_dot = slave_node.vel

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
        return False
    else:
        return np.real(del_tc[0])


# noinspection PyUnusedLocal
def check_in_bounds(patch_nodes: list[Node], slave_node: Node, dt: float, tol=1e-15) -> tuple[bool, list]:
    """
    Find the time it takes for a slave node to penetrate a triangular facet.

    :param patch_nodes: list; A list of three node objects that define the plane.
    :param slave_node: Node; The slave node to determine if it lies within the bounds.
    :param dt: float; The current time step of the analysis.
    :param tol: float; If the area is within this tolerance, then it is considered an area of zero.
    :return: tuple[bool, list]; Returns True or False and the areas of the sub-facets. It's True if all areas are
             greater than or equal to zero.
    """
    # noinspection PyUnusedLocal
    contact_point = slave_node.pos + dt*slave_node.vel  # Point of contact with plane
    current_points = np.array([node.pos for node in patch_nodes])  # Current nodal points of the patch
    vels = np.array([node.vel for node in patch_nodes])  # The velocity of patch points
    plane_points = current_points + vels*dt  # The plane points at the point of contact

    # Compute plane normal
    # noinspection PyUnreachableCode
    norm = np.cross(plane_points[1] - plane_points[0], plane_points[2] - plane_points[1])

    shifted = np.roll(plane_points, -1, axis=0)  # Shift entities (if a = [0, 1, 2], now a = [1, 2, 0])
    areas = []
    # Find the areas for each triangle formed by a side of the patch and the contact point
    for p1, p2 in zip(plane_points, shifted):
        vec1 = p2 - p1
        vec2 = contact_point - p2

        # noinspection PyUnreachableCode
        cross = np.cross(vec1, vec2)

        # If the above cross product is in the direction of the plane normal, then it is a positive area.
        s = np.sign(np.dot(cross, norm))  # returns the sign of the number (dot product will never be 0)

        area = s*1/2*np.linalg.norm(cross)  # area formula for cross products.
        area = 0. if -tol <= area <= tol else area
        areas.append(area)

    return all(np.array(areas) >= 0), areas


def phi_p_2D(xi, eta, xi_p, eta_p):
    """
    The shape function for a 2D linear quad element.
    """
    return 0.25*(1 + xi*xi_p)*(1 + eta*eta_p)


def d_phi_p_2D_d_xi(eta, xi_p, eta_p):
    """
    The derivative of the shape function with respect to xi for a 2D linear quad element.
    """
    return 0.25*xi_p*(1 + eta*eta_p)


def d_phi_p_2D_d_eta(xi, xi_p, eta_p):
    """
    The derivative of the shape function with respect to eta for a 2D linear quad element.
    """
    return 0.25*eta_p*(1 + xi*xi_p)


def phi_p_3D(xi, eta, zeta, xi_p, eta_p, zeta_p):
    """
    The shape function for a 3D linear hex element.
    """
    return 0.125*(1 + xi*xi_p)*(1 + eta*eta_p)*(1 + zeta*zeta_p)


def d_phi_p_3D_d_xi(eta, zeta, xi_p, eta_p, zeta_p):
    """
    The derivative of the shape function with respect to xi for a 3D linear hex element.
    """
    return 0.125*xi_p*(1 + eta*eta_p)*(1 + zeta*zeta_p)


def d_phi_p_3D_d_eta(xi, zeta, xi_p, eta_p, zeta_p):
    """
    The derivative of the shape function with respect to eta for a 3D linear hex element.
    """
    return 0.125*eta_p*(1 + xi*xi_p)*(1 + zeta*zeta_p)


def ref_to_physical(ref, A, xi_p, eta_p):
    """
    Map the reference coordinates defined by "ref" to the physical coordinates.

    :param ref: np.array; The (xi, eta) coordinates.
    :param A: np.array; The basis matrix as returned by Surface.construct_position_basis.
    :param xi_p: np.array; The xi coordinates of the nodes that define the surface.
    :param eta_p: np.array; The eta coordinates of the nodes that define the surface.
    :return: np.array; The physical coordinates (x, y, z).
    """
    xi, eta = ref
    phi_p_arr = phi_p_2D(xi, eta, xi_p, eta_p)
    return A@phi_p_arr


def is_concave(patch1: Surface, patch2: Surface, ref: np.ndarray, del_t: float, tol=1e-12):
    """
    Determine if the edge between two patches is concave. This only works if the point is on an edge, not a point.
    A perfectly flat surface is considered convex and will return false.

    :param patch1: Surface; The first patch object.
    :param patch2: Surface; The second patch object.
    :param ref: np.array; The reference point on the first patch. This is where concavity is determined.
    :param del_t: float; The time at which the concavity is determined.
    :param tol: float; The tolerance for ensuring that the point is on an edge.
    :return: bool; True if the edge is concave.
    """
    assert any(np.logical_and(-1 - tol <= ref, ref <= -1 + tol)) or \
           any(np.logical_and(1 - tol <= ref, ref <= 1 + tol)), 'The point must be on an edge.'

    # Compute the average normal.
    A1 = patch1.construct_position_basis(del_t)
    physical_point = ref_to_physical(ref, A1, patch1.xi_p, patch1.eta_p)
    other_ref = patch2.physical_to_ref(physical_point, del_t, tol=tol)[0]
    N1 = patch1.get_normal(ref, del_t)
    N2 = patch2.get_normal(other_ref, del_t)
    N = (N1 + N2)/2
    N = N/np.linalg.norm(N)

    # Compute partial derivatives. If the dot product between any of the partial derivatives and the normal is negative,
    # then the edge is concave.
    A2 = patch2.construct_position_basis(del_t)
    c1 = ref_to_physical(np.float64([0, 0]), A1, patch1.xi_p, patch1.eta_p)
    c2 = ref_to_physical(np.float64([0, 0]), A2, patch2.xi_p, patch2.eta_p)
    xi1, eta1 = ref
    xi2, eta2 = other_ref
    d_phi_d_xi1 = d_phi_p_2D_d_xi(eta1, patch1.xi_p, patch1.eta_p)
    d_phi_d_eta1 = d_phi_p_2D_d_eta(xi1, patch1.xi_p, patch1.eta_p)
    d_phi_d_xi2 = d_phi_p_2D_d_xi(eta2, patch2.xi_p, patch2.eta_p)
    d_phi_d_eta2 = d_phi_p_2D_d_eta(xi2, patch2.xi_p, patch2.eta_p)
    d_phi_d_xi1 = A1@d_phi_d_xi1
    d_phi_d_eta1 = A1@d_phi_d_eta1
    d_phi_d_xi2 = A2@d_phi_d_xi2
    d_phi_d_eta2 = A2@d_phi_d_eta2

    derivatives = [d_phi_d_xi1, d_phi_d_eta1, d_phi_d_xi2, d_phi_d_eta2]
    for c, der in zip((c1, c1, c2, c2), derivatives):
        rel = c - physical_point
        if np.dot(der, rel) < 0:
            der = -der

        if np.dot(der, N) + tol < 0:
            return True
    return False


# This class is for a better looking 3D arrow in plots.
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)
