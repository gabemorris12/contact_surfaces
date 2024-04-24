from contact import MeshBody, GlobalMesh
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=50)

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
convex_edge_data = np.float64([
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
    [0, 0, 1.2],
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

mesh3_data = np.float64([
    [-0.25, 0.25, 1.2],
    [-0.25, 0.75, 1.2],
    [-0.25, 0.75, 1.7],
    [-0.25, 0.25, 1.7],
    [-0.75, 0.25, 1.2],
    [-0.75, 0.75, 1.2],
    [-0.75, 0.75, 1.7],
    [-0.75, 0.25, 1.7]
])

concave_edge_cells_dict = {
    'hexahedron': np.array([
        [0, 1, 4, 5, 6, 7, 10, 11],
        [1, 2, 3, 4, 7, 8, 9, 10],
        [6, 7, 10, 11, 12, 13, 16, 17],
        [7, 8, 9, 10, 13, 14, 15, 16]
    ])
}
convex_edge_cells_dict = {
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

mesh3_cells_dict = {
    'hexahedron': np.array([
        [0, 1, 2, 3, 4, 5, 6, 7]
    ])
}

concave_edge = MeshBody(concave_edge_data, concave_edge_cells_dict)
concave_edge.color = 'black'
convex_edge = MeshBody(convex_edge_data, convex_edge_cells_dict)
convex_edge.color = 'black'
mesh2 = MeshBody(mesh2_data, mesh2_cells_dict, velocity=np.float64([0.4, -0.5, -0.5]))
mesh2.color = 'navy'
mesh3 = MeshBody(mesh3_data, mesh3_cells_dict, velocity=np.float64([0.4, -0.5, -0.5]))
mesh3.color = 'seagreen'
glob_mesh = GlobalMesh(concave_edge, mesh2, bs=0.499, master_patches=None)
# dt = 0.99  # 1 is too much
# glob_mesh = GlobalMesh(convex_edge, mesh3, bs=0.499, master_patches=None)

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

patch_force = [glob_mesh.nodes[i].contact_force for i in patch_nodes]
slave_force = [glob_mesh.nodes[i].contact_force for i in slave_nodes]
print('Total Slave Force:', np.sum(slave_force, axis=0))
print('Total Patch Force:', np.sum(patch_force, axis=0))

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
