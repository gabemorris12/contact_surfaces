from contact import Node, Surface
import numpy as np
import matplotlib.pyplot as plt

dt = 1
F = 2.6

nodes = [
    Node(0, np.array([1, 0, 0]), np.array([0, 0, 0])),
    Node(1, np.array([0, 0, 0]), np.array([0, 0, 0])),
    Node(2, np.array([0, 0, 1]), np.array([0, -0.5, 0])),
    Node(3, np.array([1, 0, 1]), np.array([0, 0.5, 0]))
]
nodes[2].corner_force = np.array([F, 0, 0])
nodes[3].corner_force = np.array([-F, 0, 0])

ref_points = [
    [-1, 1, -1],
    [1, 1, -1],
    [1, 1, 1],
    [-1, 1, 1]
]

for node, ref_point in zip(nodes, ref_points):
    node.ref = ref_point

surf = Surface(0, nodes)

slave = Node(4, np.array([0.5, 0.5, 0.8]), np.array([0, -0.75, 0]))
# slave = Node(4, np.array([0.5, 0.2, 0.8]), np.array([0, 0, 0]))
# This next case has some issues. It'll get the correct answer if you give it a good guess.
# slave = Node(4, np.array([0.5, 0.2, 0.8]), np.array([-0.1, -0.05, 0]))

sol = surf.contact_check_through_reference(slave, dt)
print(sol)
is_hitting, del_tc, *_ = sol

fig, ax = plt.subplots(subplot_kw=dict(projection='3d', proj_type='ortho'))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_aspect('equal')
ax.view_init(azim=110, elev=40)

del_tc = del_tc if is_hitting else None

surf.contact_visual_through_reference(ax, slave, dt, del_tc, only_contact=False, N=19)

plt.show()
