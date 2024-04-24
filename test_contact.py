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
            (27, 31),
            (27, 39),
            (32, 28),
            (32, 35),
            (35, 32),
            (35, 40),
            (35, 41),
            (35, 48)
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
        mesh4 = MeshBody(mesh4_points, mesh4_cells_dict, velocity=np.float64([0, -0.499, -0.25]), mass=10)
        glob_mesh = GlobalMesh(mesh1, mesh2, mesh3, mesh4, bs=0.9)
        self.assertEqual(glob_mesh.normal_increments(dt), 26)

        all_patch_nodes = set()
        for patch_id, patch_stuff in groupby(glob_mesh.contact_pairs, lambda x: x[0]):
            surf = glob_mesh.surfaces[patch_id]
            all_patch_nodes.update([node.label for node in surf.nodes])

        slave_force = [glob_mesh.nodes[i[1]].contact_force for i in glob_mesh.contact_pairs]
        patch_force = [glob_mesh.nodes[i].contact_force for i in all_patch_nodes]

        np.testing.assert_array_almost_equal(slave_force,
                                             np.array([np.array([0., -0.12934408593430582, 0.25868817186861165]),
                                                       np.array([0., -0.5116489473529776, 1.0232978947059552]),
                                                       np.array([0., -0.12942847458128567, 0.25885694916257135]),
                                                       np.array([0., -0.5118571076084528, 1.0237142152169056]),
                                                       np.array([0., 0.3245448212509412, -0.6490896425018824]),
                                                       np.array([0., 0.3244348292198854, -0.6488696584397708]),
                                                       np.array([0., 0.48701447516966156, 0.9740289503393231]),
                                                       np.array([0., 0.07740222058991228, 0.15480444117982456]),
                                                       np.array([0., 0.4739278547467477, 0.9478557094934954]),
                                                       np.array([0., 0.049694602639414664, 0.09938920527882933]),
                                                       np.array([0., 3.69174592303883, 0.]),
                                                       np.array([0., 4.066413072138012, 0.]),
                                                       np.array([0., 1.6303280153179127, 0.]),
                                                       np.array([0., 2.179038685507877, 0.]),
                                                       np.array([0., -0.3466209773675398, -0.6932419547350795]),
                                                       np.array([0., -0.3570764318384721, -0.7141528636769442])]), 12)

        np.testing.assert_array_almost_equal(patch_force,
                                             np.array([np.array([0., 0.30909798098198177, -0.6181959619639635]),
                                                       np.array([0., 0.009684735330807865, -1.3085032874620475]),
                                                       np.array([0., -1.8321081853209622, -0.4638693639322266]),
                                                       np.array([0., -0.04796397620635553, 0.09592795241271106]),
                                                       np.array([0., 0.002012561017061573, 1.1102385101814098]),
                                                       np.array([0., -2.2082370613021802, 0.14133786586277924]),
                                                       np.array([0., 0.30887295255466235, -0.6177459051093247]),
                                                       np.array([0., 0.01590346654095314, -1.2975510365559448]),
                                                       np.array([0., -3.3994825727208604, -0.4347699822220119]),
                                                       np.array([0., -0.04813860746927399, 0.09627721493854798]),
                                                       np.array([0., 0.0049086917459802365, 1.1151122625254624]),
                                                       np.array([0., -4.433118460087977, 0.14646031343276678])]), 12)

        # Testing the glue force condition
        for surf in glob_mesh.surfaces: surf.zero_contact()
        glob_mesh.contact_pairs = None

        self.assertEqual(glob_mesh.glue_increments(dt), 24)

        # noinspection PyTypeChecker
        slave_force = [glob_mesh.nodes[i[1]].contact_force for i in glob_mesh.contact_pairs]
        patch_force = [glob_mesh.nodes[i].contact_force for i in all_patch_nodes]

        np.testing.assert_array_almost_equal(slave_force,
                                             np.array([np.array([0., 0.007526039908617221, 0.3258927721061704]),
                                                       np.array([0., 0.027847124364670517, 1.1950498971162073]),
                                                       np.array([0., 0.008507454237624112, 0.326320729741142]),
                                                       np.array([0., 0.030942354171542008, 1.1963996096572733]),
                                                       np.array([0., 0.010047922019471245, -0.803385387623512]),
                                                       np.array([0., 0.011101859631906685, -0.8028830511814115]),
                                                       np.array([0., -0.10844021421617855, 1.1349938271977535]),
                                                       np.array([0., -0.40133597583392777, 0.14572456235080866]),
                                                       np.array([0., -0.1641291862699028, 1.1107099747312585]),
                                                       np.array([0., -0.5767071670867076, 0.06925182496309978]),
                                                       np.array([-3.3306690738754696e-16, 4.3239302602974874e+00,
                                                                 2.1115657456304531e+00]),
                                                       np.array([9.9920072216264089e-16, 3.8250150948040393e+00,
                                                                 1.5548145728311429e+00]),
                                                       np.array([-5.5511151231257827e-16, 2.3538365502336820e+00,
                                                                 1.2202716639785143e+00]),
                                                       np.array([-1.1102230246251565e-16, 1.6639067355104931e+00,
                                                                 5.9633163276355627e-01]),
                                                       np.array([0., -0.08642671869557028, -0.8483371934257533]),
                                                       np.array([0., -0.12507109781819936, -0.8667561963027838])]), 12)

        np.testing.assert_array_almost_equal(patch_force,
                                             np.array([np.array([0., -0.012983778072248806, -0.5433466524490542]),
                                                       np.array([0., 0.17965798909897926, -1.8563841768791507]),
                                                       np.array([-2.9143354396410359e-16, -2.1508885310831434e+00,
                                                                 -1.4881324007424859e+00]),
                                                       np.array([0., -0.0025778516056450262, 0.20081495087824675]),
                                                       np.array([0., 0.06433230529023554, 1.242151310743498]),
                                                       np.array([-4.1633363423443370e-17, -1.5350874985558212e+00,
                                                                 -5.0900777197525282e-01]),
                                                       np.array([0., -0.013738712171485326, -0.543675850629801]),
                                                       np.array([0., 0.22117936455702059, -1.838278276938024]),
                                                       np.array([6.938893903907228e-17, -4.503011539606349e+00,
                                                                 -2.513804199225705e+00]),
                                                       np.array([0., -0.0027095938071994563, 0.20075215882298408]),
                                                       np.array([0., 0.07842872085655825, 1.2488700606565966]),
                                                       np.array([2.6367796834847468e-16, -3.1231519101599563e+00,
                                                                 -1.2659241367957712e+00])]), 12)

    def test_edge_case1(self):
        logger.setLevel(logging.WARNING)
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

        mesh2_points = np.array([
            [0., 1., 1.85],
            [0., 2., 1.85],
            [0., 2., 2.85],
            [0., 1., 2.85],
            [0.5, 1., 1.85],
            [0.5, 2., 1.85],
            [0.5, 2., 2.85],
            [0.5, 1., 2.85]
        ])

        mesh1_cells_dict = {
            'hexahedron': np.array([
                [0, 1, 4, 3, 6, 7, 10, 9],
                [1, 2, 5, 4, 7, 8, 11, 10]
            ])
        }

        mesh2_cells_dict = {
            'hexahedron': np.array([
                [0, 1, 2, 3, 4, 5, 6, 7]
            ])
        }

        mesh1 = MeshBody(mesh1_points, mesh1_cells_dict)
        mesh2 = MeshBody(mesh2_points, mesh2_cells_dict, velocity=np.float64([0, -1.5, 0]))
        glob_mesh = GlobalMesh(mesh1, mesh2, bs=0.9, master_patches=None)
        glob_mesh.normal_increments(dt)

        patch_nodes, slave_nodes = set(), set()
        for pair in glob_mesh.contact_pairs:
            surf = glob_mesh.surfaces[pair[0]]
            node = glob_mesh.nodes[pair[1]]
            patch_nodes.update([node.label for node in surf.nodes])
            slave_nodes.add(node.label)

        patch_force = [glob_mesh.nodes[i].contact_force for i in patch_nodes]
        slave_force = [glob_mesh.nodes[i].contact_force for i in slave_nodes]

        np.testing.assert_array_almost_equal(patch_force, np.array(
            [np.array([0.00039405815017545825, 0.028060903387346552, 0.0561477838917793]),
             np.array([0.0007054580584197037, -0.09162995214053238, -0.17632283742972332]),
             np.array([-0.00010169135116741954, 0.013336908944505087, 0.025674488004020144]),
             np.array([-0.002696718756725031, -0.19293877250834124, -0.3859768710296208])]
        ), 12)
        np.testing.assert_array_almost_equal(slave_force, np.array(
            [np.array([0.002888944136789689, 0.08653652573906317, 0.1790479886077619]),
             np.array([-0.0011900502374924008, 0.1566343865779587, 0.30142944795578286])]
        ), 12)

    def test_edge_case2(self):
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

        mesh3_points = np.float64([
            [-0.5, 0, 2.5],
            [-0.5, 1, 2.5],
            [-0.5, 1, 3.5],
            [-0.5, 0, 3.5],
            [0.25, 0, 2.5],
            [0.25, 1, 2.5],
            [0.25, 1, 3.5],
            [0.25, 0, 3.5]
        ])

        mesh1_cells_dict = {
            'hexahedron': np.array([
                [0, 1, 4, 3, 6, 7, 10, 9],
                [1, 2, 5, 4, 7, 8, 11, 10]
            ])
        }

        mesh3_cells_dict = {
            'hexahedron': np.array([
                [0, 1, 2, 3, 4, 5, 6, 7]
            ])
        }

        mesh1 = MeshBody(mesh1_points, mesh1_cells_dict)
        mesh3 = MeshBody(mesh3_points, mesh3_cells_dict, velocity=np.float64([0, 0, -1]))
        glob_mesh = GlobalMesh(mesh1, mesh3, bs=0.9, master_patches=None)
        glob_mesh.normal_increments(dt)

        patch_nodes, slave_nodes = set(), set()
        for pair in glob_mesh.contact_pairs:
            surf = glob_mesh.surfaces[pair[0]]
            node = glob_mesh.nodes[pair[1]]
            patch_nodes.update([node.label for node in surf.nodes])
            slave_nodes.add(node.label)

        patch_force = [glob_mesh.nodes[i].contact_force for i in patch_nodes]
        slave_force = [glob_mesh.nodes[i].contact_force for i in slave_nodes]

        np.testing.assert_array_almost_equal(patch_force, np.array(
            [np.array([0., -0.01787344714052402, -0.03574689428104804]),
             np.array([0., -0.22726392385803681, -0.45452784771607363]),
             np.array([0., -0.025251547095337418, -0.050503094190674835]),
             np.array([0., -0.16086102426471618, -0.32172204852943237])]
        ), 12)
        np.testing.assert_array_almost_equal(slave_force, np.array(
            [np.array([0., 0.23831262854032026, 0.4766252570806405]),
             np.array([0., 0.19293731381829424, 0.3858746276365885])]
        ), 12)

    def test_edge_case3(self):
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

        mesh4_points = np.float64([
            [0, -1.1, 0.45],
            [0, -0.205572809, 0.8972135955],
            [0, 0.2416407865, 0.0027864045],
            [0, -0.6527864045, -0.444427191],
            [0.5, -1.1, 0.45],
            [0.5, -0.205572809, 0.8972135955],
            [0.5, 0.2416407865, 0.0027864045],
            [0.5, -0.6527864045, -0.444427191]
        ])

        mesh1_cells_dict = {
            'hexahedron': np.array([
                [0, 1, 4, 3, 6, 7, 10, 9],
                [1, 2, 5, 4, 7, 8, 11, 10]
            ])
        }

        mesh4_cells_dict = {
            'hexahedron': np.array([
                [0, 1, 2, 3, 4, 5, 6, 7]
            ])
        }

        mesh1 = MeshBody(mesh1_points, mesh1_cells_dict)
        mesh4 = MeshBody(mesh4_points, mesh4_cells_dict, velocity=np.float64([0, 0, 1]))
        glob_mesh = GlobalMesh(mesh1, mesh4, bs=0.9, master_patches=None)
        glob_mesh.normal_increments(dt)

        patch_nodes, slave_nodes = set(), set()
        for pair in glob_mesh.contact_pairs:
            surf = glob_mesh.surfaces[pair[0]]
            node = glob_mesh.nodes[pair[1]]
            patch_nodes.update([node.label for node in surf.nodes])
            slave_nodes.add(node.label)

        patch_force = [glob_mesh.nodes[i].contact_force for i in patch_nodes]
        slave_force = [glob_mesh.nodes[i].contact_force for i in slave_nodes]

        np.testing.assert_array_almost_equal(patch_force, np.array(
            [np.array([-4.0574952070499671e-06, -5.7791540019762284e-02, 1.1557427771560450e-01]),
             np.array([0.000956963642172054, -0.3045586351124726, 0.7393426738443416]),
             np.array([9.6497486606857278e-07, 2.5613804197574612e-03, 4.0167297195679184e-03]),
             np.array([0.00042392109081310574, -0.1616397928214352, 0.32419907223510785]),
             np.array([-0.03336359459851106, -0.45798492012543723, 1.471519904567675]),
             np.array([-3.5912114304813137e-05, 2.6901066916132136e-03, 4.1927994587043068e-03])]
        ), 12)
        np.testing.assert_array_almost_equal(slave_force, np.array(
            [np.array([-0.0010916734256534427, 0.3158743100648214, -0.6341162938360453]),
             np.array([0.03311338792582513, -0.05224490287742635, -0.29833225056742085]),
             np.array([0., 0.4321864879843813, -0.8643729759687626]),
             np.array([0., 0.28090750579596024, -0.8620239371687722])]
        ), 12)

    def test_edge_case4(self):
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

        mesh5_points = np.float64([
            [-0.5, -1, 2.5],
            [-0.5, 0, 2.5],
            [-0.5, 1, 2.5],
            [-0.5, 1, 3.5],
            [-0.5, 0, 3.5],
            [-0.5, -1, 3.5],
            [0.5, -1, 2.5],
            [0.5, 0, 2.5],
            [0.5, 1, 2.5],
            [0.5, 1, 3.5],
            [0.5, 0, 3.5],
            [0.5, -1, 3.5]
        ])

        mesh1_cells_dict = {
            'hexahedron': np.array([
                [0, 1, 4, 3, 6, 7, 10, 9],
                [1, 2, 5, 4, 7, 8, 11, 10]
            ])
        }

        mesh5_cells_dict = {
            'hexahedron': np.array([
                [0, 1, 4, 5, 6, 7, 10, 11],
                [1, 2, 3, 4, 7, 8, 9, 10]
            ])
        }

        mesh1 = MeshBody(mesh1_points, mesh1_cells_dict)
        mesh5 = MeshBody(mesh5_points, mesh5_cells_dict, velocity=np.float64([0, 0, -1]))
        glob_mesh = GlobalMesh(mesh1, mesh5, bs=0.9, master_patches=None)
        glob_mesh.normal_increments(dt)

        patch_nodes, slave_nodes = set(), set()
        for pair in glob_mesh.contact_pairs:
            surf = glob_mesh.surfaces[pair[0]]
            node = glob_mesh.nodes[pair[1]]
            patch_nodes.update([node.label for node in surf.nodes])
            slave_nodes.add(node.label)

        patch_force = [glob_mesh.nodes[i].contact_force for i in patch_nodes]
        slave_force = [glob_mesh.nodes[i].contact_force for i in slave_nodes]

        np.testing.assert_array_almost_equal(patch_force, np.array(
            [np.array([0., 0., 0.]),
             np.array([0., 0., -0.5]),
             np.array([0., 0., 0.]),
             np.array([0., 0., -0.5])]
        ), 12)
        np.testing.assert_array_almost_equal(slave_force, np.array(
            [np.array([0., 0., 0.5]),
             np.array([0., 0., 0.5])]
        ), 12)

    def test_dynamic_pair1(self):
        logger.setLevel(logging.WARNING)

        dt = 1

        concave_edge_data = np.float64([
            [1, -1, 0],
            [1, 0, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
            [1, -1, 1],
            [0, -1, 0],
            [0, 0, 0],
            [0, 1, 0],
            [0, 1, 0.75],
            [0, 0, 0.75],
            [0, -1, 1],
            [-1, -1, 0],
            [-1, 0, 0],
            [-1, 1, 0],
            [-1, 1, 1],
            [-1, 0, 1],
            [-1, -1, 1]
        ])

        mesh2_data = np.float64([
            [-0.25, 0.25, 1],
            [-0.25, 0.75, 1],
            [-0.25, 0.75, 1.5],
            [-0.25, 0.25, 1.5],
            [-0.75, 0.25, 1],
            [-0.75, 0.75, 1],
            [-0.75, 0.75, 1.5],
            [-0.75, 0.25, 1.5]
        ])

        concave_edge_cells_dict = {
            'hexahedron': np.array([
                [0, 1, 4, 5, 6, 7, 10, 11],
                [1, 2, 3, 4, 7, 8, 9, 10],
                [6, 7, 10, 11, 12, 13, 16, 17],
                [7, 8, 9, 10, 13, 14, 15, 16]
            ])
        }

        mesh2_cells_dict = {
            'hexahedron': np.array([
                [0, 1, 2, 3, 4, 5, 6, 7]
            ])
        }

        concave_edge = MeshBody(concave_edge_data, concave_edge_cells_dict)
        mesh2 = MeshBody(mesh2_data, mesh2_cells_dict, velocity=np.float64([0.4, -0.5, -0.5]))
        glob_mesh = GlobalMesh(concave_edge, mesh2, bs=0.499, master_patches=None)
        glob_mesh.normal_increments(dt)

        patch_nodes, slave_nodes = set(), set()
        for pair in glob_mesh.contact_pairs:
            surf = glob_mesh.surfaces[pair[0]]
            node = glob_mesh.nodes[pair[1]]
            patch_nodes.update([node.label for node in surf.nodes])
            slave_nodes.add(node.label)

        patch_force = [glob_mesh.nodes[i].contact_force for i in patch_nodes]
        slave_force = [glob_mesh.nodes[i].contact_force for i in slave_nodes]

        np.testing.assert_array_almost_equal(patch_force, np.array(
            [np.array([0.0005539015084259897, 0.0002314547943687331, -0.0017418355351483024]),
             np.array([0.0064693940635127505, -0.004171248148862138, -0.019753218327383013]),
             np.array([0.00137944514374141, -0.00137944514374141, -0.004169788577824197]),
             np.array([-0.00020772148813252587, 0.0009392083693601984, -0.016889582221308302]),
             np.array([-0.14486235161974242, -0.03099725641998173, -0.7289602421912597]),
             np.array([-0.0009988720628906148, -0.009777428678570562, -0.03247795442072837]),
             np.array([-0.0004100865581845867, 0., -0.001640346232738345]),
             np.array([-0.024440972449171164, -0.0044189187150610225, -0.08057939217249387]),
             np.array([-0.00254369341602789, -0.0013204864050972702, -0.00503960441697665])]
        ), 12)
        np.testing.assert_array_almost_equal(slave_force, np.array(
            [np.array([-0.004376416097753706, 0.03374499172183596, 0.21947870744051065]),
             np.array([0.02119140676689364, -0.00459632426310486, 0.16335412531988225]),
             np.array([0.09054095946869792, 0.021745452888854114, 0.27759910437294305]),
             np.array([0.057705006740631226, 0., 0.2308200269625249])]
        ), 12)


if __name__ == '__main__':
    unittest.main()
