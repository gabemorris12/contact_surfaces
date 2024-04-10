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
# This is the reason why we need to have the logic adjust the contact pairs. If the master patch 8 was selected, then
# it would incorrectly perform force calculation with this pair.
# mesh2 = MeshBody(mesh2_points, mesh2_cells_dict, velocity=np.float64([0, -1, 0]))
glob_mesh = GlobalMesh(mesh1, mesh2, bs=0.9)

print('Contact Pairs:')
for pair in glob_mesh.get_contact_pairs(dt):
    print(pair)

# Contact detection
fig1, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, subplot_kw=dict(projection='3d', proj_type='ortho'))
ax1.set_title('Contact Detection')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.view_init(elev=27, azim=-24)

for patch_id, patch_stuff in groupby(glob_mesh.contact_pairs, lambda x: x[0]):
    nodes, del_tc = [], []
    for things in patch_stuff:
        nodes.append(glob_mesh.nodes[things[1]])
        del_tc.append(things[2][-1])

    surf = glob_mesh.surfaces[patch_id]
    surf.contact_visual_through_reference(ax1, nodes, dt, del_tc)

ax1.set_aspect('equal')

ax2.set_title('Normal Force')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.view_init(elev=27, azim=-24)

print('\nTotal Iterations:', glob_mesh.normal_increments(dt), '\n')

all_patch_nodes = set()
for patch_id, patch_stuff in groupby(glob_mesh.contact_pairs, lambda x: x[0]):
    nodes = [glob_mesh.nodes[things[1]] for things in patch_stuff]

    for node in nodes:
        x_pos, y_pos, z_pos = node.pos
        ax2.text(x_pos, y_pos, z_pos, f'{node.label}', color='black')
        print(f'{node.label}: {node.contact_force}')

    surf = glob_mesh.surfaces[patch_id]
    all_patch_nodes.update([node.label for node in surf.nodes])
    surf.contact_visual_through_reference(ax2, nodes, dt, None)

slave_force = [glob_mesh.nodes[i[1]].contact_force for i in glob_mesh.contact_pairs]
patch_force = [glob_mesh.nodes[i].contact_force for i in all_patch_nodes]
print(f'\nTotal Force on Slaves: {np.sum(slave_force, axis=0)}')
print(f'Total Force on Patches: {np.sum(patch_force, axis=0)}')

ax2.set_aspect('equal')

plt.show()
