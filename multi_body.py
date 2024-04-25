from contact import MeshBody, GlobalMesh
import numpy as np
import matplotlib.pyplot as plt
from itertools import groupby

dt = 1

np.set_printoptions(precision=50)

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
mesh1.color = 'black'
mesh1.alpha = 1
mesh2 = MeshBody(mesh2_points, mesh2_cells_dict, velocity=np.float64([0, 0, -1]))
mesh2.color = 'navy'
mesh2.alpha = 0.25
mesh3 = MeshBody(mesh3_points, mesh3_cells_dict, velocity=np.float64([0, 0, 0.75]))
mesh3.color = 'seagreen'
mesh3.alpha = 0.25
# mesh4 = MeshBody(mesh4_points, mesh4_cells_dict, velocity=np.float64([-0.3, -0.499, -0.25]), mass=10)
mesh4 = MeshBody(mesh4_points, mesh4_cells_dict, velocity=np.float64([0, -0.499, -0.25]), mass=10)
mesh4.color = 'darkred'
mesh4.alpha = 0.25
glob_mesh = GlobalMesh(mesh1, mesh2, mesh3, mesh4, bs=0.9)

print('Normal Iteration Count:', glob_mesh.normal_increments(dt))

fig1, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, subplot_kw=dict(projection='3d', proj_type='ortho'))
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.view_init(elev=27, azim=-24)
ax1.set_title('At $t$')

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.view_init(elev=27, azim=-24)
ax2.set_title(r'At $t + \Delta t$ (Normal)')

for mesh in glob_mesh.mesh_bodies:
    for surf in mesh.surfaces:
        x, y, z = np.mean(surf.points, axis=0)
        ax1.text(x, y, z, str(surf.label), color=mesh.color)

        surf.project_surface(ax1, 0, N=2, ls='-', color=mesh.color, alpha=mesh.alpha)
        surf.project_surface(ax2, dt, N=2, ls='-', color=mesh.color, alpha=mesh.alpha)

for node in glob_mesh.nodes: ax1.text(*node.pos, str(node.label), color='black')

for pair in glob_mesh.contact_pairs:
    print(pair)

all_patch_nodes = set()
for patch_id, patch_stuff in groupby(glob_mesh.contact_pairs, lambda x_: x_[0]):
    surf = glob_mesh.surfaces[patch_id]
    all_patch_nodes.update([node.label for node in surf.nodes])

slave_force = [glob_mesh.nodes[i[1]].contact_force for i in glob_mesh.contact_pairs]
patch_force = [glob_mesh.nodes[i].contact_force for i in all_patch_nodes]
print('Total Slave Force:', np.sum(slave_force, axis=0))
print('Total Patch Force:', np.sum(patch_force, axis=0))

for surf in glob_mesh.surfaces: surf.zero_contact()
glob_mesh.contact_pairs = []

print('Glue Iteration Count:', glob_mesh.glue_increments(dt))

# noinspection PyTypeChecker
slave_force = [glob_mesh.nodes[i[1]].contact_force for i in glob_mesh.contact_pairs]
patch_force = [glob_mesh.nodes[i].contact_force for i in all_patch_nodes]
print('Total Slave Force:', np.sum(slave_force, axis=0))
print('Total Patch Force:', np.sum(patch_force, axis=0))

for mesh in glob_mesh.mesh_bodies:
    for surf in mesh.surfaces:
        surf.project_surface(ax3, dt, N=2, ls='-', color=mesh.color, alpha=mesh.alpha)

ax3.set_title(r'At $t + \Delta t$ (Glue)')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.view_init(elev=27, azim=-24)

ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')

plt.show()
