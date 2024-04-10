import unittest
import logging
from itertools import groupby

import meshio
import numpy as np

from contact import MeshBody, GlobalMesh, Node, Surface

logger = logging.getLogger('contact')
logger.setLevel(logging.INFO)


class TestContact(unittest.TestCase):
    data1 = meshio.read('Meshes/Block.msh')
    data2 = meshio.read('Meshes/Block2.msh')
    mesh1 = MeshBody(data1.points, data1.cells_dict, velocity=np.float64([0, 500, 0]))
    mesh2 = MeshBody(data2.points, data2.cells_dict)
    global_mesh = GlobalMesh(mesh1, mesh2, bs=0.5)
    test_surf = global_mesh.surfaces[27]
    dt1 = 1.1e-6

    single_element = MeshBody(
        np.array([
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1]
        ], dtype=np.float64),
        {'hexahedron': np.array([
            [0, 1, 2, 3, 4, 5, 6, 7]
        ])}
    )

    half_points = np.array([
        [0, 0, 0],
        [0.5, 0, 0],
        [0.5, 0.5, 0],
        [0, 0.5, 0],
        [0, 0, 0.5],
        [0.5, 0, 0.5],
        [0.5, 0.5, 0.5],
        [0, 0.5, 0.5],
        [0, 1, 0],
        [0.5, 1, 0],
        [0.5, 1, 0.5],
        [0, 1, 0.5]
    ])

    half_points = np.add(half_points, np.array([0.1, 1.1, 0.1]))  # shift points to put it on top of the single element

    # A 0.5x1x0.5 mesh moving downward at 10 in/s
    half_by_1_by_half = MeshBody(
        half_points,
        {'hexahedron': np.array([
            [0, 1, 2, 3, 4, 5, 6, 7],
            [3, 2, 9, 8, 7, 6, 10, 11]
        ])}, velocity=np.array([0, -10, 0]))

    global_mesh2 = GlobalMesh(single_element, half_by_1_by_half, bs=0.25)

    # The following attributes are defined for the contact check with vary random velocities and points to verify the
    # math for finding delta_tc. A visual confirmation is provided in the contact_check_visual.py file.
    _points = np.array([
        [0.5, 0.5, 1],  # Updated Point 1
        [1, 0.5, 2],  # Updated Point 2
        [1, 1, 3],  # Updated Point 3
        [0.5, 1, 2]  # Updated Point 4
    ])

    # Define velocity for points
    _vels = np.array([
        [0.12, 0.08, -0.05],
        [0.7*3, 0.75*3, -0.25*3],
        [-0.06, -0.03, -0.34],
        [-0.065, -0.035, -0.42]
    ])

    _nodes = [Node(i, pos, vel) for i, (pos, vel) in enumerate(list(zip(_points, _vels)))]
    random_surf = Surface(0, _nodes)
    random_surf.reverse_dir()
    _sep_point = np.array([0.75, 0.75, 1])
    _v = np.array([2, -0.1, 10.5])
    sep_node = Node(len(_nodes), _sep_point, _v)

    # Define a simple surface on the x-y plane.
    simple_nodes = [
        Node(0, np.array([0, 0, 0]), np.array([0, 0, 0])),
        Node(0, np.array([1, 0, 0]), np.array([0, 0, 0])),
        Node(0, np.array([1, 1, 0]), np.array([0, 0, 0])),
        Node(0, np.array([0, 1, 0]), np.array([0, 0, 0]))
    ]
    simple_ref = np.array([
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1]
    ])
    simple_surf = Surface(0, simple_nodes)

    for i, node in enumerate(simple_surf.nodes):
        node.ref = simple_ref[i]

    def test_reverse_dir(self):
        dir1 = np.array([0, 0.25, 0])
        dir2 = -dir1
        nodes1 = [23, 17, 7, 18]
        nodes2 = [23, 18, 7, 17]

        np.testing.assert_array_equal(TestContact.test_surf.dir, dir1)
        self.assertEqual([node.label for node in TestContact.test_surf.nodes], nodes1)
        TestContact.test_surf.reverse_dir()
        np.testing.assert_array_equal(TestContact.test_surf.dir, dir2)
        self.assertEqual([node.label for node in TestContact.test_surf.nodes], nodes2)
        TestContact.test_surf.reverse_dir()
        np.testing.assert_array_equal(TestContact.test_surf.dir, dir1)
        self.assertEqual([node.label for node in TestContact.test_surf.nodes], nodes1)

    def test_capture_box(self):
        right = (0.5000011, 1.00055, 1.0000011, -1.1e-06, 0.99945, 0.4999988999999999)
        self.assertEqual(TestContact.test_surf.capture_box(1, 500, 1, 0, 0, 0, 1.1e-6), right)

    def test_outify_surfaces(self):
        element = TestContact.global_mesh.elements[1]
        element.outify_surfaces()
        np.testing.assert_array_almost_equal(element.surfaces[0].dir, np.array([0, 0, -0.25]), 12)
        np.testing.assert_array_almost_equal(element.surfaces[1].dir, np.array([0, 0, 0.25]), 12)
        np.testing.assert_array_almost_equal(element.surfaces[2].dir, np.array([0, -0.25, 0]), 12)
        np.testing.assert_array_almost_equal(element.surfaces[3].dir, np.array([0.25, 0, 0]), 12)
        np.testing.assert_array_almost_equal(element.surfaces[4].dir, np.array([0, 0.25, 0]), 12)
        np.testing.assert_array_almost_equal(element.surfaces[5].dir, np.array([-0.25, 0, 0]), 12)

    def test_sort(self):
        lbox = np.array(
            [0, 2, 8, 6, 30, 32, 38, 36, 0, 2, 7, 3, 0, 2, 30, 8, 32, 6, 37, 33, 4, 16, 20, 22, 18, 34, 19, 6, 8, 14,
             12, 36, 38, 44, 42, 6, 11, 13, 9, 6, 8, 36, 14, 41, 12, 43, 39, 10, 22, 26, 28, 24, 40, 25])
        np.testing.assert_array_equal(TestContact.global_mesh.lbox, lbox)
        nbox = np.array(
            [3, 0, 3, 1, 1, 0, 5, 1, 4, 1, 1, 1, 2, 1, 2, 0, 1, 0, 1, 1, 1, 0, 2, 0, 1, 1, 1, 0, 1, 0, 2, 0, 2, 1, 1, 0,
             3, 1, 2, 1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(TestContact.global_mesh.nbox, nbox)
        npoint = np.array(
            [0, 3, 3, 6, 7, 8, 8, 13, 14, 18, 19, 20, 21, 23, 24, 26, 26, 27, 27, 28, 29, 30, 30, 32, 32, 33, 34, 35,
             35, 36, 36, 38, 38, 40, 41, 42, 42, 45, 46, 48, 49, 50, 51, 52, 53])
        np.testing.assert_array_equal(TestContact.global_mesh.npoint, npoint)
        nsort = np.array(
            [0, 8, 12, 1, 9, 13, 11, 20, 3, 17, 27, 35, 39, 10, 2, 15, 28, 40, 38, 47, 36, 30, 44, 37, 29, 42, 21, 24,
             26, 22, 23, 48, 51, 53, 49, 50, 4, 14, 5, 16, 19, 25, 7, 31, 41, 18, 6, 32, 46, 52, 43, 34, 45, 33])
        np.testing.assert_array_equal(TestContact.global_mesh.nsort, nsort)

    def test_find_nodes(self):
        _, nodes = TestContact.global_mesh.find_nodes(TestContact.test_surf.label, TestContact.dt1)
        np.testing.assert_array_equal(nodes, np.array([3, 10, 11, 19, 20, 24, 25, 26, 27, 31, 35, 39, 41, 48]))

        _, nodes2 = TestContact.global_mesh2.find_nodes(4, 0.07)
        np.testing.assert_array_equal(nodes2, np.array([8, 9, 10, 11, 12, 13, 14, 15]))

    def test_set_node_refs(self):
        ele = TestContact.global_mesh.elements[0]
        ele.set_node_refs()
        ref_array = np.array([node.ref for node in ele.connectivity])
        np.testing.assert_array_equal(ref_array, np.array([
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1]
        ]))

    def test_get_contact_point(self):
        # Deprecating this behavior
        # ref_points = np.array([
        #     [1, 1, -1],
        #     [-1, 1, -1],
        #     [-1, 1, 1],
        #     [1, 1, 1]
        # ])
        # for i, node in enumerate(TestContact.random_surf.nodes):
        #     node.ref = ref_points[i]

        slave = Node(4, np.array([0.75, 0.75, 1]), np.array([2, -0.1, 10.5]))
        contact_point, _ = TestContact.random_surf.get_contact_point(slave, np.array([0.5, -0.5, 0.05]))
        np.testing.assert_array_almost_equal(contact_point, np.array([-0.41631963, 0.34774981, 0.08798188]), 8)

    def test_contact_check(self):
        nodes = [31, 39, 41, 48]
        sol = [TestContact.global_mesh.contact_check(27, node, TestContact.dt1) for node in nodes]
        true_list = [s[0] for s in sol]
        self.assertListEqual(true_list, [True]*4)

        # Testing random velocities
        sol_rand = TestContact.random_surf.contact_check(TestContact.sep_node, 0.1)
        self.assertTrue(sol_rand[0], True)
        self.assertAlmostEqual(sol_rand[1], 0.08544489159847724, 12)

    def test_contact_check_through_reference(self):
        simple_node = Node(0, np.array([0.5, 0.5, 0]), np.array([0, 0, -0.1]))
        sol = TestContact.simple_surf.contact_check_through_reference(simple_node, 0.1)
        self.assertEqual(sol[0], True)
        self.assertEqual(sol[1], 0)
        np.testing.assert_array_equal(sol[2], np.array([0, 0]))

        # Move to the edge cases
        simple_node = Node(0, np.array([1, 1, 0]), np.array([0, 0, -0.1]))
        sol = TestContact.simple_surf.contact_check_through_reference(simple_node, 0.1)
        self.assertEqual(sol[0], True)
        np.testing.assert_array_equal(sol[2], np.array([1, 1]))

        # Move to the middle
        simple_node = Node(0, np.array([0, 0.5, 0]), np.array([0, 0, -0.1]))
        sol = TestContact.simple_surf.contact_check_through_reference(simple_node, 0.1)
        self.assertEqual(sol[0], True)
        np.testing.assert_array_equal(sol[2], np.array([-1, 0]))

        # Move to off the surface
        simple_node = Node(0, np.array([1.1, 1, 0]), np.array([0, 0, -0.1]))
        sol = TestContact.simple_surf.contact_check_through_reference(simple_node, 0.1)
        self.assertEqual(sol[0], False)

        # Classic test as seen in surface_node_detection_through_reference.py
        nodes = [31, 39, 41, 48]
        elem = TestContact.global_mesh.get_element_by_surf[TestContact.global_mesh.surfaces[27]][0]
        elem.set_node_refs()
        sol = [TestContact.global_mesh.contact_check_through_reference(27, node, TestContact.dt1) for node in nodes]
        true_list = [s[0] for s in sol]
        self.assertListEqual(true_list, [True]*4)

        # Testing the case where the slave moves onto the surface, with a trajectory that is perpendicular to the
        # surface.
        nodes = [31, 39]
        sol = [TestContact.global_mesh.contact_check_through_reference(28, node, TestContact.dt1) for node in nodes]
        false_list = [s[0] for s in sol]
        self.assertListEqual(false_list, [False]*2)

    def test_get_normal(self):
        dt = 1

        nodes = [
            Node(0, np.array([1, 0, 0]), np.array([0, 1, 0])),
            Node(1, np.array([0, 0, 0]), np.array([0, 0.1, 0])),
            Node(2, np.array([0, 0, 1]), np.array([0, 0.2, 0])),
            Node(3, np.array([1, 0, 1]), np.array([0, 0, 0]))
        ]

        surf = Surface(0, nodes)

        slave = Node(4, np.array([0.25, 1, 0.2]), np.array([0.75, -1, 0]))
        sol = surf.contact_check_through_reference(slave, dt)

        self.assertEqual(sol[0], True)
        self.assertAlmostEqual(sol[1], 0.6221606424927081, 12)
        np.testing.assert_array_almost_equal(sol[2], np.array([-0.4332409637390621, -0.6]), 12)

        n = surf.get_normal(sol[2], sol[1])
        np.testing.assert_array_almost_equal(n, np.array([-0.36246426756409533, 0.8567492881880666,
                                                          0.36687915166777346]), 12)

    def test_get_contact_pairs(self):
        logger.setLevel(logging.WARNING)
        contact_pairs = TestContact.global_mesh.get_contact_pairs(TestContact.dt1)
        supposed_to_be = [
            (23, 27),
            (23, 35),
            (23, 39),
            (23, 48),
            (27, 31),
            (27, 41),
            (32, 28),
            (32, 40),
            (35, 32)
        ]
        self.assertListEqual([pair[:2] for pair in contact_pairs], supposed_to_be)

    def test_normal_increment(self):
        # Single patch, multiple nodes as seen in normal_increment_check.py
        dt = 1
        inc = 10

        patch_nodes = [
            Node(0, np.array([-1, -1, 0]), np.array([0, 0, 1])),
            Node(1, np.array([1, -1, 0]), np.array([0, 0, 1])),
            Node(2, np.array([1, 1, 0]), np.array([0, 0, 1])),
            Node(3, np.array([-1, 1, 0]), np.array([0, 0, 1]))
        ]

        slaves = [
            Node(4, np.array([0.5, 0, 1]), np.array([0, 0, -1.5])),
            Node(5, np.array([-0.5, 0, 1]), np.array([0, 0, -0.5])),
            Node(6, np.array([0, 0.5, 1]), np.array([0, 0, -1]))
            # Node(6, np.array([0, 0.5, 1]), np.array([0, 0, -0.25]))  # Tensile Behavior
        ]

        surf = Surface(0, patch_nodes)

        check_sol = [surf.contact_check_through_reference(slave, dt) for slave in slaves]
        guesses = [(xi, eta, 1) for _, _, (xi, eta), _ in check_sol]
        normals = [surf.get_normal(np.array([xi, eta]), del_tc) for _, del_tc, (xi, eta), _ in check_sol]

        for _ in range(inc):
            surf.normal_increment(slaves, guesses, normals, dt)

        slave_forces = np.array([[0., 0., 2.0414312617484804],
                                 [0., 0., 0.2636534839966621],
                                 [0., 0., 1.084745762715211]])

        np.testing.assert_array_almost_equal(np.array([slave.contact_force for slave in slaves]), slave_forces, 12)

        patch_forces = np.array([[0., 0., -0.4896421845567096],
                                 [0., 0., -0.9340866289946642],
                                 [0., 0., -1.2052730696734673],
                                 [0., 0., -0.7608286252355123]])

        np.testing.assert_array_almost_equal(np.array([node.contact_force for node in patch_nodes]), patch_forces, 12)

        surf.zero_contact()
        for slave in slaves: slave.zero_contact()

        # Testing the tensile behavior
        slaves[-1].vel = np.array([0, 0, -0.25])
        check_sol = [surf.contact_check_through_reference(slave, dt) for slave in slaves]
        guesses = [(xi, eta, 1) for _, _, (xi, eta), _ in check_sol]
        normals = [surf.get_normal(np.array([xi, eta]), del_tc) for _, del_tc, (xi, eta), _ in check_sol]

        for _ in range(inc):
            surf.normal_increment(slaves, guesses, normals, dt)

        slave_forces = np.array([[0., 0., 2.2222222222222223],
                                 [0., 0., 0.44444444444444453],
                                 [0., 0., 0.]])

        np.testing.assert_array_almost_equal(np.array([slave.contact_force for slave in slaves]), slave_forces, 12)

        patch_forces = np.array([[0., 0., -0.4444444444444447],
                                 [0., 0., -0.8888888888888891],
                                 [0., 0., -0.8888888888888891],
                                 [0., 0., -0.4444444444444447]])

        np.testing.assert_array_almost_equal(np.array([node.contact_force for node in patch_nodes]), patch_forces, 12)

        # Swap the last slave node with the first to show that the order of the nodes does not matter.
        slaves[0], slaves[-1] = slaves[-1], slaves[0]

        surf.zero_contact()
        for slave in slaves: slave.zero_contact()

        check_sol = [surf.contact_check_through_reference(slave, dt) for slave in slaves]
        guesses = [(xi, eta, 1) for _, _, (xi, eta), _ in check_sol]
        normals = [surf.get_normal(np.array([xi, eta]), del_tc) for _, del_tc, (xi, eta), _ in check_sol]

        for _ in range(inc):
            surf.normal_increment(slaves, guesses, normals, dt)

        slave_forces = np.array([[0., 0., 0.],
                                 [0., 0., 0.44444444444444453],
                                 [0., 0., 2.2222222222222223]])

        np.testing.assert_array_almost_equal(np.array([slave.contact_force for slave in slaves]), slave_forces, 12)

    def test_normal_increments(self):
        dt = 1

        mesh1_points = np.float64([
            [-0.5, -2, 1],
            [-0.5, 0, 2],
            [-0.5, 2, 1],
            [-0.5, -2, 0],
            [-0.5, 0, 1],
            [-0.5, 2, 0],
            [0.5, -2, 1],
            [0.5, 0, 2],
            [0.5, 2, 1],
            [0.5, -2, 0],
            [0.5, 0, 1],
            [0.5, 2, 0]
        ])

        mesh2_points = np.float64([
            [-0.25, -1.5, 2],
            [-0.25, -0.5, 2],
            [-0.25, 0.5, 2],
            [-0.25, 1.5, 2],
            [-0.25, 1.5, 3],
            [-0.25, 0.5, 3],
            [-0.25, -0.5, 3],
            [-0.25, -1.5, 3],
            [0.25, -1.5, 2],
            [0.25, -0.5, 2],
            [0.25, 0.5, 2],
            [0.25, 1.5, 2],
            [0.25, 1.5, 3],
            [0.25, 0.5, 3],
            [0.25, -0.5, 3],
            [0.25, -1.5, 3]
        ])

        mesh1_cells_dict = {
            'hexahedron': np.array([
                [0, 1, 4, 3, 6, 7, 10, 9],
                [1, 2, 5, 4, 7, 8, 11, 10]
            ])
        }

        mesh2_cells_dict = {
            'hexahedron': np.array([
                [0, 1, 6, 7, 8, 9, 14, 15],
                [1, 2, 5, 6, 9, 10, 13, 14],
                [2, 3, 4, 5, 10, 11, 12, 13]
            ])
        }

        mesh1 = MeshBody(mesh1_points, mesh1_cells_dict)
        mesh2 = MeshBody(mesh2_points, mesh2_cells_dict, velocity=np.float64([0, 0, -1]))
        glob_mesh = GlobalMesh(mesh1, mesh2, bs=0.9)

        iters = glob_mesh.normal_increments(dt)
        self.assertEqual(iters, 16)
        contact_pairs = [(2, 12), (2, 13), (2, 20), (2, 21), (8, 14), (8, 15), (8, 22), (8, 23)]
        self.assertListEqual([(pair[0], pair[1]) for pair in glob_mesh.contact_pairs], contact_pairs)

        all_patch_nodes = set()
        for patch_id, patch_stuff in groupby(glob_mesh.contact_pairs, lambda x: x[0]):
            surf = glob_mesh.surfaces[patch_id]
            all_patch_nodes.update([node.label for node in surf.nodes])

        slave_force = [glob_mesh.nodes[i[1]].contact_force for i in glob_mesh.contact_pairs]
        patch_force = [glob_mesh.nodes[i].contact_force for i in all_patch_nodes]

        np.testing.assert_array_almost_equal(slave_force,
                                             [np.array([0., -0.011499387462763189, 0.022998774925526377]),
                                              np.array([0., -0.33112171561802917, 0.6622434312360583]),
                                              np.array([0., -0.012319446199142407, 0.024638892398284814]),
                                              np.array([0., -0.3309365475869207, 0.6618730951738414]),
                                              np.array([0., 0.3312557928943872, 0.6625115857887744]),
                                              np.array([0., 0.024831153071986422, 0.049662306143972844]),
                                              np.array([0., 0.33089155280555577, 0.6617831056111115]),
                                              np.array([0., 0.024343387455427544, 0.04868677491085509])], 12)

        np.testing.assert_array_almost_equal(patch_force,
                                             [np.array([0., 0.14897311267489768, -0.29794622534979537]),
                                              np.array([0., -0.03396082880881164, -0.8431485099464727]),
                                              np.array([0., -0.12810640264900205, -0.2562128052980041]),
                                              np.array([0., 0.14632776843469555, -0.2926555368693911]),
                                              np.array([0., -0.029472979602432874, -0.8460239699050659]),
                                              np.array([0., -0.12920545940984796, -0.2584109188196959])], 12)

    def test_multi_body(self):
        dt = 1

        mesh1_points = np.float64([
            [-0.5, -2, 1],
            [-0.5, 0, 2],
            [-0.5, 2, 1],
            [-0.5, -2, 0],
            [-0.5, 0, 1],
            [-0.5, 2, 0],
            [0.5, -2, 1],
            [0.5, 0, 2],
            [0.5, 2, 1],
            [0.5, -2, 0],
            [0.5, 0, 1],
            [0.5, 2, 0]
        ])

        mesh2_points = np.float64([
            [-0.25, -1.5, 2],
            [-0.25, -0.5, 2],
            [-0.25, 0.5, 2],
            [-0.25, 1.5, 2],
            [-0.25, 1.5, 3],
            [-0.25, 0.5, 3],
            [-0.25, -0.5, 3],
            [-0.25, -1.5, 3],
            [0.25, -1.5, 2],
            [0.25, -0.5, 2],
            [0.25, 0.5, 2],
            [0.25, 1.5, 2],
            [0.25, 1.5, 3],
            [0.25, 0.5, 3],
            [0.25, -0.5, 3],
            [0.25, -1.5, 3]
        ])

        mesh3_points = np.float64([
            [-0.25, -0.5, -0.5],
            [-0.25, 0.5, -0.5],
            [-0.25, 0.5, 0.5],
            [-0.25, -0.5, 0.5],
            [0.25, -0.5, -0.5],
            [0.25, 0.5, -0.5],
            [0.25, 0.5, 0.5],
            [0.25, -0.5, 0.5]
        ])

        mesh4_points = np.float64([
            [0, 2, 0.5],
            [0, 2.5, 0.5],
            [0, 2.5, 0.75],
            [0, 2, 0.75],
            [0.5, 2, 0.5],
            [0.5, 2.5, 0.5],
            [0.5, 2.5, 0.75],
            [0.5, 2, 0.75]
        ])

        mesh1_cells_dict = {
            'hexahedron': np.array([
                [0, 1, 4, 3, 6, 7, 10, 9],
                [1, 2, 5, 4, 7, 8, 11, 10]
            ])
        }

        mesh2_cells_dict = {
            'hexahedron': np.array([
                [0, 1, 6, 7, 8, 9, 14, 15],
                [1, 2, 5, 6, 9, 10, 13, 14],
                [2, 3, 4, 5, 10, 11, 12, 13]
            ])
        }

        mesh3_cells_dict = {
            'hexahedron': np.array([
                [0, 1, 2, 3, 4, 5, 6, 7]
            ])
        }

        mesh4_cells_dict = {
            'hexahedron': np.array([
                [0, 1, 2, 3, 4, 5, 6, 7]
            ])
        }

        mesh1 = MeshBody(mesh1_points, mesh1_cells_dict, mass=5.0)
        mesh2 = MeshBody(mesh2_points, mesh2_cells_dict, velocity=np.float64([0, 0, -1]))
        mesh3 = MeshBody(mesh3_points, mesh3_cells_dict, velocity=np.float64([0, 0, 0.75]))
        mesh4 = MeshBody(mesh4_points, mesh4_cells_dict, velocity=np.float64([0, -0.5, -0.25]), mass=10)
        glob_mesh = GlobalMesh(mesh1, mesh2, mesh3, mesh4, bs=0.9)
        self.assertEqual(glob_mesh.normal_increments(dt), 23)

        all_patch_nodes = set()
        for patch_id, patch_stuff in groupby(glob_mesh.contact_pairs, lambda x: x[0]):
            surf = glob_mesh.surfaces[patch_id]
            all_patch_nodes.update([node.label for node in surf.nodes])

        slave_force = [glob_mesh.nodes[i[1]].contact_force for i in glob_mesh.contact_pairs]
        patch_force = [glob_mesh.nodes[i].contact_force for i in all_patch_nodes]

        np.testing.assert_array_almost_equal(slave_force,
                                             np.array([np.array([0., -0.1293446074581961, 0.2586892149163922]),
                                                       np.array([0., -0.5116510302602347, 1.0233020605204695]),
                                                       np.array([0., -0.12942908810494635, 0.2588581762098927]),
                                                       np.array([0., -0.511859484452958, 1.023718968905916]),
                                                       np.array([0., 0.3245433581681366, -0.6490867163362732]),
                                                       np.array([0., 0.32443310026354, -0.64886620052708]),
                                                       np.array([0., 0.4869618914857238, 0.9739237829714475]),
                                                       np.array([0., 0.07728684655619554, 0.15457369311239108]),
                                                       np.array([0., 0.47384294247874903, 0.9476858849574981]),
                                                       np.array([0., 0.04951141096336773, 0.09902282192673546]),
                                                       np.array([0., 3.699120291464036, 0.]),
                                                       np.array([0., 0., 0.]),
                                                       np.array([0., 0., 0.]),
                                                       np.array([0., 4.074872534974702, 0.]),
                                                       np.array([0., 1.6334994184996314, 0.]),
                                                       np.array([0., 0., 0.]),
                                                       np.array([0., 0., 0.]),
                                                       np.array([0., 2.1836341738395646, 0.]),
                                                       np.array([0., -0.3466690811880941, -0.6933381623761882]),
                                                       np.array([0., -0.35715227866517113, -0.7143045573303423])]), 12)

        np.testing.assert_array_almost_equal(patch_force,
                                             np.array([np.array([0., 0.3090992543985239, -0.6181985087970479]),
                                                       np.array([0., 0.009715199852176894, -1.308448068414747]),
                                                       np.array([0., -1.8352612819847614, -0.46354144942874065]),
                                                       np.array([0., -0.047963626551613056, 0.09592725310322611]),
                                                       np.array([0., 0.0020299953896788783, 1.1102686593408546]),
                                                       np.array([0., -2.21279813792167, 0.14141543605462073]),
                                                       np.array([0., 0.30887431566089046, -0.6177486313217809]),
                                                       np.array([0., 0.015942050379626378, -1.297479991989329]),
                                                       np.array([0., -3.405839239043681, -0.43435795356909646]),
                                                       np.array([0., -0.04813825632601285, 0.0962765126520257]),
                                                       np.array([0., 0.004932116185150905, 1.115153866025006]),
                                                       np.array([0., -4.442192788602356, 0.14655390939414958])]), 12)


if __name__ == '__main__':
    unittest.main()
