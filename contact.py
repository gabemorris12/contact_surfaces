import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.tri as tri
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


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
        get_element_by_surf: dict; A dictionary whose keys are surface objects and values are a list of element objects
                             corresponding to the element that contains the surface.
        nodes: numpy.array; An array of node objects
        """
        self.points, self.cells_dict, self.velocity = points, cells_dict, velocity
        self.surfaces, self.surface_count = [], np.zeros(self._surface_count(), dtype=np.int32)
        self.surface_dict = dict()
        self.get_element_by_surf = dict()
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

    def get_acc(self):
        """
        :return: The acceleration of the node
        """
        return (self.corner_force + self.contact_force)/self.mass

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

        self.xi_p, self.eta_p = None, None

    def reverse_dir(self):
        """
        Reverses the direction of the surface.
        """
        temp = self.nodes[1:]
        self.nodes = [self.nodes[0]] + list(reversed(temp))
        self.points = np.array([node.pos for node in self.nodes])
        self.vel_points = np.array([node.vel for node in self.nodes])
        self.dir = -self.dir

        if self.xi_p is not None and self.eta_p is not None:
            ref = np.array([node.ref for node in self.nodes])
            index = np.arange(3)
            ref = ref[:, index != self.ref_plane[0]]
            self.xi_p, self.eta_p = ref[:, 0], ref[:, 1]

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

        if self.ref_plane is None:
            self._set_reference_plane()

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

            sol = sol - np.linalg.pinv(J)@F

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

        if self.ref_plane is None:
            self._set_reference_plane()

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

        if self.ref_plane is None:
            self._set_reference_plane()

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

    def contact_check_through_reference(self, node: Node, dt: float, tol=1e-12, max_iter=30):
        """
        This procedure finds the reference point and the delta_tc all at once using a Newton-Raphson scheme. If all the
        reference points are between -1 and 1, the delta_tc is between 0 and dt, and it took less than 25 iterations to
        solve then the node will pass through the patch.

        :param node: Node; The node object to be tested for contact.
        :param dt: float; The current time step in the analysis.
        :param tol: float; The tolerance of the edge cases. For the end cases where either reference coordinate is
                    either 1 or -1, the tolerance will adjust for floating point error, ensuring that the edge case is
                    met.
        :param max_iter: int; The maximum number of iterations for the Newton-Raphson scheme.
        :return: tuple; Returns True or False indicating that the node will pass through the surface within the next
                 time step. Additionally, it returns the time until contact del_tc (if it's between 0 and dt), the
                 reference contact point (xi, eta), and the number of iterations for the solution.
        """
        k = 0
        guesses = (
            (0.5, 0.5, dt/2),
            (-0.5, 0.5, dt/2),
            (-0.5, -0.5, dt/2),
            (0.5, -0.5, dt/2)
        )
        del_tc_vals, ref_vals = [], []
        for guess in guesses:
            sol = self.get_contact_point(node, np.array(guess), tol, max_iter=max_iter)
            ref = sol[0][:2]
            del_tc = sol[0][2]
            k += sol[1]

            if all(np.logical_and(ref >= -1 - tol, ref <= 1 + tol)) and 0 - tol <= del_tc <= dt + tol and \
                    sol[1] <= max_iter:
                del_tc = 0 if 0 - tol <= del_tc <= 0 + tol else del_tc
                del_tc = dt if dt - tol <= del_tc <= dt + tol else del_tc
                return True, del_tc, ref, k
            else:
                del_tc_vals.append(del_tc)
                ref_vals.append(ref)
        return False, del_tc_vals, ref_vals, k

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

    def contact_visual_through_reference(self, axes: Axes3D, node: Node, dt: float, del_tc: float | None,
                                         only_contact=False, penetration=True, **kwargs):
        """
        Generates a 3D plot of the contact check for visual confirmation.

        :param axes: Axes3D; A 3D axes object to generate the plot on.
        :param node: Node; The slave node object that is being tested.
        :param dt: float; The current time step in the analysis.
        :param del_tc: float; The delta time to contact.
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
            contact_point = node.pos + del_tc*node.vel + 0.5*node.get_acc()*del_tc**2
            axes.scatter(contact_point[0], contact_point[1], contact_point[2], color='gold', marker='x')

        if del_tc is not None and any(self.vel_points.flatten()):
            self.project_surface(axes, del_tc, color='firebrick', **kwargs)

        if penetration and del_tc is not None:
            ref, _ = self.get_contact_point(node, np.array([0.5, 0.5, del_tc]))
            n = self.get_normal(ref[:2], del_tc)
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

    def project_surface(self, axes: Axes3D, del_t: float, N=9, alpha=0.25, color='navy', show_grid=False,
                        triangulate=False):
        """
        Project the surface at time del_t later onto the given axes object.

        :param axes: Axes3D; The 3D axes object to project the surface onto.
        :param del_t: float; The time to project the surface.
        :param N: int; The size of the grid in each direction.
        :param alpha: float; The transparency of the surface.
        :param color: str; The color of the surface.
        :param show_grid: bool; Whether to show the grid of the surface.
        :param triangulate: bool; Whether to use the triangulate method to generate a mesh.
        """
        if self.ref_plane is None:
            self._set_reference_plane()

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
            axes.plot(x_values[i:i + N], y_values[i:i + N], z_values[i:i + N], color=color, ls='--', alpha=alpha)

            # Plotting the row
            axes.plot(points[0, index + r], points[1, index + r], points[2, index + r], color=color, ls='--',
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

    def find_min_distance(self, node: Node, dt: float):
        """
        Find the minimum distance between the node and the surface at the given time.

        :param node: Node; The slave node that is in proximity of the surface.
        :param dt: float; The time to project the surface.
        """

    def _get_distance(self, coord: np.ndarray):
        # Find the distance between a node and point on the surface
        pass

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
        # Determine which reference plane the surface is on. zeta=-1 plane? xi=1 plane? etc.
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
        self.accelerations = np.array([node.get_acc() for node in self.nodes])

        self.bs = bs

        vels = np.amax(np.abs(self.velocities), axis=0)
        accs = np.amax(np.abs(self.accelerations), axis=0)
        for i, vel in enumerate(vels):
            if vel == 0:
                # When I get this into fierro, consider choosing a velocity that results in a distance that is half the
                # bucket size. I need access to the time step, which is not available here.
                vels[i] = 1

        self.vx_max, self.vy_max, self.vz_max = vels
        self.ax_max, self.ay_max, self.az_max = accs

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
            self.get_element_by_surf.update(mesh.get_element_by_surf)

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
        ibox_min = min(self.Sx, int((xc_min - self.x_min)/self.bs))
        jbox_min = min(self.Sy, int((yc_min - self.y_min)/self.bs))
        kbox_min = min(self.Sz, int((zc_min - self.z_min)/self.bs))
        ibox_max = min(self.Sx, int((xc_max - self.x_min)/self.bs))
        jbox_max = min(self.Sy, int((yc_max - self.y_min)/self.bs))
        kbox_max = min(self.Sz, int((zc_max - self.z_min)/self.bs))

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

    def contact_check_through_reference(self, surface_id: int, node_id: int, dt: float, tol=1e-12):
        """
        This procedure finds the reference point and the delta_tc all at once using a Newton-Raphson scheme. If all the
        reference points are between -1 and 1 and the delta_tc is between 0 and dt, then the node will pass through the
        patch.

        :param surface_id: int; The surface id.
        :param node_id: int; The node id to be tested for contact.
        :param dt: float; The current time step in the analysis.
        :param tol: float; The tolerance of the edge cases. For the end cases where either reference coordinate is
                    either 1 or -1, the tolerance will adjust for floating point error, ensuring that the edge case is
                    met.
        :return: tuple; Returns True or False indicating that the node will pass through the surface within the next
                 time step. Additionally, it returns the time until contact del_tc (if it's between 0 and dt), the
                 reference contact point (xi, eta), and the number of iterations for the solution.
        """
        surf = self.surfaces[surface_id]  # See contact_check from class surface.
        # noinspection PyUnresolvedReferences
        return surf.contact_check_through_reference(self.nodes[node_id], dt, tol)


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

    :param ref: np.array; The (xi, eta, zeta) coordinates.
    :param A: np.array; The basis matrix as returned by Surface.construct_position_basis.
    :param xi_p: np.array; The xi coordinates of the nodes that define the surface.
    :param eta_p: np.array; The eta coordinates of the nodes that define the surface.
    :return: np.array; The physical coordinates (x, y, z).
    """
    xi, eta = ref
    phi_p_arr = phi_p_2D(xi, eta, xi_p, eta_p)
    return A@phi_p_arr


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
