"""
Demonstrates the behavior for following time steps.
"""
import numpy as np
from contact import MeshBody, GlobalMesh
import matplotlib.pyplot as plt

dt = 0.5

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

mesh1 = MeshBody(mesh1_data, mesh1_cells_dict)
mesh1.color = 'black'
mesh2 = MeshBody(mesh2_data, mesh2_cells_dict, velocity=np.float64([0, -0.25, 0.25]))
mesh2.color = 'navy'
mesh3 = MeshBody(mesh3_data, mesh3_cells_dict)
mesh3.color = 'seagreen'
glob_mesh = GlobalMesh(mesh1, mesh2, mesh3, bs=0.49)

fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, subplot_kw=dict(projection='3d', proj_type='ortho'))
ax1.set_title('At $t$')
ax2.set_title(r'At $t + \Delta t$')
ax3.set_title(r'At $t + 2\cdot\Delta t$')
for ax in (ax1, ax2, ax3):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

for mesh in glob_mesh.mesh_bodies:
    for surf in mesh.surfaces:
        surf.project_surface(ax1, 0, N=2, color=mesh.color, ls='-', alpha=1)

        x, y, z = np.mean(surf.points, axis=0)
        ax1.text(x, y, z, f'{surf.label}', color=mesh.color)

for node in glob_mesh.nodes: ax1.text(*node.pos, f'{node.label}', color='lime')

contact_pairs = glob_mesh.get_contact_pairs(dt)
print('Contact Pairs at t = 0:')
for pair in contact_pairs: print(pair)
print()

print('Total Iterations:', glob_mesh.normal_increments(dt))
print()

glob_mesh.remove_pairs(dt)
glob_mesh.update_nodes(dt)
for patch_obj in glob_mesh.surfaces:
    patch_obj.zero_contact()

for mesh in glob_mesh.mesh_bodies:
    for surf in mesh.surfaces:
        patch_obj = glob_mesh.surfaces[surf.label]
        patch_obj.project_surface(ax2, 0, N=2, color=mesh.color, ls='-', alpha=1)

for node in glob_mesh.nodes: ax2.text(*node.pos, f'{node.label}', color='lime')

glob_mesh.sort()
glob_mesh.get_contact_pairs(dt, include_initial_penetration=True)
print('Contact Pairs at t = dt:')
for pair in glob_mesh.contact_pairs: print(pair)
print()

print('Total Iterations:', glob_mesh.normal_increments(dt))
print()

glob_mesh.remove_pairs(dt)
glob_mesh.update_nodes(dt)
# glob_mesh.sort()
print('Contact Pairs at t = 2*dt:')
for pair in glob_mesh.contact_pairs: print(pair)
print()

for mesh in glob_mesh.mesh_bodies:
    for surf in mesh.surfaces:
        patch_obj = glob_mesh.surfaces[surf.label]
        patch_obj.project_surface(ax3, 0, N=2, color=mesh.color, ls='-', alpha=1)

for ax in (ax1, ax2, ax3):
    ax.set_aspect('equal')
    ax.view_init(elev=25, azim=25)

plt.show()
