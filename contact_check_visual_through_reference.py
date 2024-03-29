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
ax.view_init(azim=90, elev=0)

sol = surf.contact_check_through_reference(slave, dt)
print(sol)

if sol[0]:
    del_tc = sol[1]
else:
    del_tc = None

surf.contact_visual_through_reference(ax, slave, dt, del_tc, only_contact=True)

# Let's plot the line as if it were subject to some force.
# n = surf.get_normal(sol[2], del_tc)
# f = .1
# t = np.linspace(0, dt, 50).reshape(-1, 1)
# p = slave.pos + slave.vel*t + 0.5*n*f*t**2  # Assume a mass of 1
# ax.plot(p[:, 0], p[:, 1], p[:, 2], color='r', ls='--')

plt.show()
