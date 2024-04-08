from contact import MeshBody
import numpy as np

mesh1_points = np.float64([
    [-2, 1, -0.5],
    [0, 2, -0.5],
    [2, 1, -0.5],
    [-2, 0, -0.5],
    [0, 1, -0.5],
    [2, 0, -0.5],
    [-2, 1, 0.5],
    [0, 2, 0.5],
    [2, 1, 0.5],
    [-2, 0, 0.5],
    [0, 1, 0.5],
    [2, 0, 0.5]
])

mesh1_cells_dict = {
    'hexahedron': np.array([
        [0, 1, 4, 3, 6, 7, 10, 9],
        [1, 2, 5, 4, 7, 8, 11, 10]
    ])
}

mesh1 = MeshBody(mesh1_points, mesh1_cells_dict)
