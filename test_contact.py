import unittest

import meshio
import numpy as np

from contact import MeshBody, GlobalMesh


class TestContact(unittest.TestCase):
    mesh1, mesh2, global_mesh = None, None, None
    test_surf = None

    @classmethod
    def setUpClass(cls):
        data1 = meshio.read('Meshes/Block.msh')
        data2 = meshio.read('Meshes/Block2.msh')
        cls.mesh1 = MeshBody(data1.points, data1.cells_dict, velocity=np.float64([0, 500, 0]))
        cls.mesh2 = MeshBody(data2.points, data2.cells_dict)
        cls.global_mesh = GlobalMesh(cls.mesh1, cls.mesh2, bs=0.5)
        cls.test_surf = cls.global_mesh.surfaces[27]

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
        self.assertEqual(TestContact.test_surf.capture_box(1, 500, 1, 1.1e-6), right)

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


if __name__ == '__main__':
    unittest.main()
