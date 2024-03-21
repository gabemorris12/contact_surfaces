"""
Visual confirmation using the reference point finding method.
"""
from contact import Node, Surface
import numpy as np
import matplotlib.pyplot as plt

dt = 1

nodes = [
    Node(0, np.array([1, 0, 0]), np.array([0, 1, 0])),
    Node(1, np.array([0, 0, 0]), np.array([0, 0.1, 0])),
    Node(2, np.array([0, 0, 1]), np.array([0, 0.2, 0])),
    Node(3, np.array([1, 0, 1]), np.array([0, 0, 0]))
]

ref_points = [
    [-1, 1, -1],
    [1, 1, -1],
    [1, 1, 1],
    [-1, 1, 1]
]

for node, ref_point in zip(nodes, ref_points):
    node.ref = ref_point

surf = Surface(0, nodes)

slave = Node(4, np.array([0.25, 1, 0.2]), np.array([0.75, -1, 0]))

fig, ax = plt.subplots(subplot_kw=dict(projection='3d', proj_type='ortho'))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_aspect('equal')

sol = surf.contact_check_through_reference(slave, dt)
print(sol)

if sol[0]:
    del_tc = sol[1]
else:
    del_tc = None

surf.contact_visual_through_reference(ax, slave, dt, del_tc)

plt.show()
