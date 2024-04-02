"""
Visual confirmation using the reference point finding method.
"""
from contact import Node, Surface, phi_p_2D
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

surf.contact_visual_through_reference(ax, slave, dt, del_tc, only_contact=False)

N = surf.get_normal(sol[2], del_tc)
force_sol = surf.find_fc(slave, np.array([sol[2][0], sol[2][1], 2.3]), dt, N)
print(force_sol)
xi, eta, fc = force_sol[0]

fig2, ax2 = plt.subplots(subplot_kw=dict(projection='3d', proj_type='ortho'))
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_aspect('equal')
ax2.view_init(azim=90, elev=0)

slave.R = N*fc
phi_k = phi_p_2D(xi, eta, surf.xi_p, surf.eta_p)
for phi, node in zip(phi_k, surf.nodes):
    node.R = -N*fc*phi

surf.contact_visual_through_reference(ax2, slave, dt, None, only_contact=False)

plt.show()
