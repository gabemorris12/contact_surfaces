from contact import MeshBody, GlobalMesh
import matplotlib.pyplot as plt
import numpy as np

dt = 1

mesh1_data = np.float64([
    [1, -1, 0],
    [1, 0, 0],
    [1, 1, 0],
    [1, 1, 1],
    [1, 0, 1],
    [1, -1, 1],
    [0, -1, 0],
    [0, 0, 0],
    [0, 1, 0],
    [0, 1, 1],
    [0, 0, 1],
    [0, -1, 1],
    [-1, -1, 0],
    [-1, 0, 0],
    [-1, 1, 0],
    [-1, 1, 1],
    [-1, 0, 1],
    [-1, -1, 1]
])

mesh2_data = np.float64([
    [-0.25, -0.25, 1],
    [-0.25, 0.25, 1],
    [-0.25, 0.25, 1.5],
    [-0.25, -0.25, 1.5],
    [0.25, -0.25, 1],
    [0.25, 0.25, 1],
    [0.25, 0.25, 1.5],
    [0.25, -0.25, 1.5]
])

mesh1_cells_dict = {
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

mesh1 = MeshBody(mesh1_data, mesh1_cells_dict)
mesh1.color = 'black'
mesh2 = MeshBody(mesh2_data, mesh2_cells_dict, velocity=np.float64([0, 0, -0.25]))
mesh2.color = 'navy'
glob_mesh = GlobalMesh(mesh1, mesh2, bs=0.499)

contact_pairs = glob_mesh.get_contact_pairs(dt)
print('Contact Pairs Before:')
for pair in contact_pairs: print(pair, '---> Node Force:', glob_mesh.nodes[pair[1]].contact_force)
print()


print('Total Iterations', glob_mesh.normal_increments(dt))
print()

print('Contact Pairs After:')
for pair in glob_mesh.contact_pairs: print(pair, '---> Node Force:', glob_mesh.nodes[pair[1]].contact_force)
print()

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, subplot_kw=dict(projection='3d', proj_type='ortho'))
ax1.set_title('At $t$')
ax2.set_title(r'At $t + \Delta t$')
for ax in (ax1, ax2):
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

for mesh in glob_mesh.mesh_bodies:
    for surf in mesh.surfaces:
        surf.project_surface(ax1, 0, N=2, color=mesh.color, ls='-', alpha=1)
        surf.project_surface(ax2, dt, N=2, color=mesh.color, ls='-', alpha=1)

        x, y, z = np.mean(surf.points, axis=0)
        ax1.text(x, y, z, f'{surf.label}', color=mesh.color)

for node in glob_mesh.nodes: ax1.text(*node.pos, f'{node.label}', color='lime')

for ax in (ax1, ax2):
    ax.set_aspect('equal')
    ax.view_init(elev=25, azim=25)

plt.show()
