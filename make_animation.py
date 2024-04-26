"""
Demonstrates the behavior for following time steps.
"""
import numpy as np
from contact import MeshBody, GlobalMesh
import matplotlib.pyplot as plt

dt = 0.1
frames = 35

mesh1_data = np.float64([
    [-0.5, -0.5, 0],
    [-0.5, -0.25, 0],
    [-0.5, 0.5, 1],
    [-0.5, -0.5, 1],
    [0.5, -0.5, 0],
    [0.5, -0.25, 0],
    [0.5, 0.5, 1],
    [0.5, -0.5, 1]
])

mesh2_data = np.float64([
    [-0.25, 0.2, 0.6],
    [-0.25, 0.7, 0.6],
    [-0.25, 0.7, 0.1],
    [-0.25, 0.2, 0.1],
    [0.25, 0.2, 0.6],
    [0.25, 0.7, 0.6],
    [0.25, 0.7, 0.1],
    [0.25, 0.2, 0.1]
])

mesh3_data = np.float64([
    [-0.5, -0.5, 1],
    [-0.5, 0.5, 1],
    [-0.5, 0.5, 2],
    [-0.5, -0.5, 2],
    [0.5, -0.5, 1],
    [0.5, 0.5, 1],
    [0.5, 0.5, 2],
    [0.5, -0.5, 2]
])

mesh4_data = np.float64([
    [-0.6, -0.6, 1],
    [-0.6, 0.6, 1],
    [-0.6, 0.6, 2],
    [-0.6, -0.6, 2],
    [0.6, -0.6, 1],
    [0.6, 0.6, 1],
    [0.6, 0.6, 2],
    [0.6, -0.6, 2]
])

mesh1_cells_dict = {
    'hexahedron': np.array([
        [0, 1, 2, 3, 4, 5, 6, 7]
    ])
}

mesh2_cells_dict = {
    'hexahedron': np.array([
        [0, 1, 2, 3, 4, 5, 6, 7]
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

mesh1 = MeshBody(mesh1_data, mesh1_cells_dict)
mesh1.color = 'black'
mesh2 = MeshBody(mesh2_data, mesh2_cells_dict, velocity=np.float64([0, -0.5, 0.5]))
mesh2.color = 'navy'
mesh3 = MeshBody(mesh3_data, mesh3_cells_dict)
mesh3.color = 'seagreen'
mesh4 = MeshBody(mesh4_data, mesh4_cells_dict)
mesh4.color = 'darkred'
# glob_mesh = GlobalMesh(mesh1, mesh2, mesh3, bs=0.49)
glob_mesh = GlobalMesh(mesh1, mesh2, mesh4, bs=0.49)

fig, ax = plt.subplots(subplot_kw=dict(projection='3d', proj_type='ortho'))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

for i in range(frames):
    t = i*dt
    ax.set_title(f'At $t={t:.3f}$')

    for mesh in glob_mesh.mesh_bodies:
        for surf in mesh.surfaces:
            surf.project_surface(ax, 0, N=2, color=mesh.color, ls='-', alpha=1)

    ax.set_aspect('equal')
    ax.view_init(elev=25, azim=25)
    fig.savefig(f'scratch/{i}.png')
    ax.cla()

    glob_mesh.sort()
    contact_pairs = glob_mesh.get_contact_pairs(dt)
    print(f'Total Iterations at t = {t:.5f}:', glob_mesh.normal_increments(dt, multi_stage=True))
    print()

    glob_mesh.remove_pairs(dt)
    glob_mesh.update_nodes(dt)
    for patch_obj in glob_mesh.surfaces:
        patch_obj.zero_contact()
