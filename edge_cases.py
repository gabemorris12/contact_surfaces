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

mesh1 = MeshBody(mesh1_points, mesh1_cells_dict)
mesh1.color = 'black'
mesh2 = MeshBody(mesh2_points, mesh2_cells_dict, velocity=np.float64([0, -1.5, 0]))
mesh2.color = 'navy'
mesh3 = MeshBody(mesh3_points, mesh3_cells_dict, velocity=np.float64([0, 0, -1]))
mesh3.color = 'seagreen'
mesh4 = MeshBody(mesh4_points, mesh4_cells_dict, velocity=np.float64([0, 0, 1]))
mesh4.color = 'darkorange'
# glob_mesh = GlobalMesh(mesh1, mesh2, bs=0.9, master_patches=None)
# glob_mesh = GlobalMesh(mesh1, mesh3, bs=0.9, master_patches=None)
glob_mesh = GlobalMesh(mesh1, mesh4, bs=0.9, master_patches=None)

print('Contact Pairs:')
contact_pairs = glob_mesh.get_contact_pairs(dt)
# contact_pairs.extend([
#     (10, 13, (0.0, 1.0, 0), np.array([0., -0.4472135954999579, -0.8944271909999159]), 1),
#     (10, 17, (1.0, 1.0, 0), np.array([0., -0.4472135954999579, -0.8944271909999159]), 1)
# ])
for pair in contact_pairs:
    print(pair)

# Contact detection
fig1, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, subplot_kw=dict(projection='3d', proj_type='ortho'))
ax1.set_title('Contact Detection')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.view_init(elev=27, azim=-24)
ax2.set_title('At $t$')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.view_init(elev=27, azim=-24)
ax3.set_title(r'At $t + \Delta t$ (Normal)')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.view_init(elev=27, azim=-24)

for patch_id, patch_stuff in groupby(glob_mesh.contact_pairs, lambda x: x[0]):
    nodes, del_tc = [], []
    for things in patch_stuff:
        nodes.append(glob_mesh.nodes[things[1]])
        del_tc.append(things[2][-1])

    surf = glob_mesh.surfaces[patch_id]
    surf.contact_visual_through_reference(ax1, nodes, dt, del_tc)

ax1.set_aspect('equal')

print('\nTotal Normal Iterations:', glob_mesh.normal_increments(dt), '\n')

for node in glob_mesh.nodes: ax2.text(*node.pos, str(node.label), color='black')

for mesh in glob_mesh.mesh_bodies:
    for surf in mesh.surfaces:
        surf.project_surface(ax2, 0, N=2, ls='-', color=mesh.color, alpha=1)
        surf.project_surface(ax3, dt, N=2, ls='-', color=mesh.color, alpha=1)

        centroid = np.mean(surf.points, axis=0)
        ax2.text(*centroid, str(surf.label), color=mesh.color)

ax2.set_aspect('equal')
ax3.set_aspect('equal')

print('Contact Pairs After:')
for pair in glob_mesh.contact_pairs: print(pair)

patch_nodes, slave_nodes = set(), set()
for pair in glob_mesh.contact_pairs:
    surf = glob_mesh.surfaces[pair[0]]
    node = glob_mesh.nodes[pair[1]]
    patch_nodes.update([node.label for node in surf.nodes])
    slave_nodes.add(node.label)

print('\nTotal Slave Force:', np.sum([glob_mesh.nodes[i].contact_force for i in slave_nodes], axis=0))
print('Total Patch Force:', np.sum([glob_mesh.nodes[i].contact_force for i in patch_nodes], axis=0))

plt.show()
