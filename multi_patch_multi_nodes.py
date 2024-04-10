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
fig1, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, subplot_kw=dict(projection='3d', proj_type='ortho'))
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

print('\nTotal Normal Iterations:', glob_mesh.normal_increments(dt), '\n')

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

for surf in glob_mesh.surfaces: surf.zero_contact()

ax3.set_title('Glue Force')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.view_init(elev=27, azim=-24)

print('\nTotal Glue Iterations:', glob_mesh.glue_increments(dt), '\n')

for patch_id, patch_stuff in groupby(glob_mesh.contact_pairs, lambda x: x[0]):
    nodes = [glob_mesh.nodes[things[1]] for things in patch_stuff]
    surf = glob_mesh.surfaces[patch_id]
    surf.contact_visual_through_reference(ax3, nodes, dt, None)

slave_force = [glob_mesh.nodes[i[1]].contact_force for i in glob_mesh.contact_pairs]
patch_force = [glob_mesh.nodes[i].contact_force for i in all_patch_nodes]
print(f'Total Force on Slaves: {np.sum(slave_force, axis=0)}')
print(f'Total Force on Patches: {np.sum(patch_force, axis=0)}')

ax3.set_aspect('equal')

plt.show()

"""
Should output this:

Contact Pairs:
(2, 12, (-0.5, -0.5000000000000001, 0.75), array([ 0.                , -0.4472135954999579,  0.8944271909999159]), 1)
(2, 13, (-0.5, 0.5, 0.25000000000000006), array([ 0.                , -0.4472135954999579,  0.8944271909999159]), 1)
(2, 20, (0.5, -0.5000000000000001, 0.75), array([ 0.                , -0.4472135954999579,  0.8944271909999159]), 1)
(2, 21, (0.5, 0.5, 0.25000000000000006), array([ 0.                , -0.4472135954999579,  0.8944271909999159]), 1)
(8, 14, (-0.5, -0.5, 0.25000000000000006), array([-0.                ,  0.4472135954999579,  0.8944271909999159]), 1)
(8, 15, (-0.5, 0.5000000000000001, 0.75), array([-0.                ,  0.4472135954999579,  0.8944271909999159]), 1)
(8, 22, (0.5, -0.5, 0.25000000000000006), array([-0.                ,  0.4472135954999579,  0.8944271909999159]), 1)
(8, 23, (0.5, 0.5000000000000001, 0.75), array([-0.                ,  0.4472135954999579,  0.8944271909999159]), 1)
Normal: [ 0.                 -0.4472135954999579  0.8944271909999159] at [-0.5 -0.5]
Normal: [ 0.                 -0.4472135954999579  0.8944271909999159] at [-0.5  0.5]
Normal: [ 0.                 -0.4472135954999579  0.8944271909999159] at [ 0.5 -0.5]
Normal: [ 0.                 -0.4472135954999579  0.8944271909999159] at [0.5 0.5]
Normal: [-0.                  0.4472135954999579  0.8944271909999159] at [-0.5 -0.5]
Normal: [-0.                  0.4472135954999579  0.8944271909999159] at [-0.5  0.5]
Normal: [-0.                  0.4472135954999579  0.8944271909999159] at [ 0.5 -0.5]
Normal: [-0.                  0.4472135954999579  0.8944271909999159] at [0.5 0.5]

Total Normal Iterations: 16 

12: [ 0.                   -0.011499387462763189  0.022998774925526377]
13: [ 0.                  -0.33112171561802917  0.6622434312360583 ]
20: [ 0.                   -0.012319446199142407  0.024638892398284814]
21: [ 0.                 -0.3309365475869207  0.6618730951738414]
14: [0.                 0.3312557928943872 0.6625115857887744]
15: [0.                   0.024831153071986422 0.049662306143972844]
22: [0.                  0.33089155280555577 0.6617831056111115 ]
23: [0.                   0.024343387455427544 0.04868677491085509 ]

Total Force on Slaves: [0.                  0.02544478936050145 2.794397966188425  ]
Total Force on Patches: [ 0.                   -0.025444789360501305 -2.7943979661884253  ]

Total Glue Iterations: 18 

Total Force on Slaves: [0.                 0.                 2.9629629629629486]
Total Force on Patches: [ 0.                  0.                 -2.9629629629629495]
"""
