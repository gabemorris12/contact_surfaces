from contact import MeshBody, GlobalMesh
import numpy as np
import matplotlib.pyplot as plt

dt = 1

mesh1_data = np.array([
    [-0.5, -0.5, 0],
    [-0.5, 0.5, 0],
    [-0.5, 0.5, 1],
    [-0.5, -0.5, 1],
    [0.5, -0.5, 0],
    [0.5, 0.5, 0],
    [0.5, 0.5, 1],
    [0.5, -0.5, 1]
])

mesh2_data = np.array([
    [-0.25, 0, 1],
    [-0.25, 0.5, 1],
    [-0.25, 0.5, 1.5],
    [-0.25, 0, 1.5],
    [0.25, 0, 1],
    [0.25, 0.5, 1],
    [0.25, 0.5, 1.5],
    [0.25, 0, 1.5]
])

mesh3_data = np.array([
    [-0.5, 0, 1],
    [-0.5, 0.5, 1],
    [-0.5, 0.5, 1.5],
    [-0.5, 0, 1.5],
    [0, 0, 1],
    [0, 0.5, 1],
    [0, 0.5, 1.5],
    [0, 0, 1.5]
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
mesh2 = MeshBody(mesh2_data, mesh2_cells_dict, velocity=np.float64([0, -0.25, -0.499]))
mesh2.color = 'navy'
mesh3 = MeshBody(mesh3_data, mesh3_cells_dict, velocity=np.float64([-0.1, 0.25, -0.499]))
mesh3.color = 'seagreen'
# glob_mesh = GlobalMesh(mesh1, mesh2, bs=0.9, master_patches=None)
glob_mesh = GlobalMesh(mesh1, mesh3, bs=0.9, master_patches=None)

contact_pairs = glob_mesh.get_contact_pairs(dt)
print('Contact Pairs Before:')
for pair in contact_pairs: print(pair, '---> Node Force:', glob_mesh.nodes[pair[1]].contact_force)
print()


print('Total Iterations', glob_mesh.normal_increments(dt))
print()

print('Contact Pairs After:')
for pair in glob_mesh.contact_pairs: print(pair, '---> Node Force:', glob_mesh.nodes[pair[1]].contact_force)
print()


patch_nodes, slave_nodes = set(), set()
for pair in glob_mesh.contact_pairs:
    surf = glob_mesh.surfaces[pair[0]]
    node = glob_mesh.nodes[pair[1]]
    patch_nodes.update([node.label for node in surf.nodes])
    slave_nodes.add(node.label)

print('Total Slave Force:', np.sum([glob_mesh.nodes[i].contact_force for i in slave_nodes], axis=0))
print('Total Patch Force:', np.sum([glob_mesh.nodes[i].contact_force for i in patch_nodes], axis=0))

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

for node in glob_mesh.nodes: ax1.text(*node.pos, f'{node.label}', color='black')

for ax in (ax1, ax2):
    ax.set_aspect('equal')
    ax.view_init(elev=25, azim=25)

plt.show()
