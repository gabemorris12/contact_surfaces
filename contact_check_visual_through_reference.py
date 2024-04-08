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

surf = Surface(0, nodes)

slave = Node(4, np.array([0.25, 1, 0.2]), np.array([0.75, -1, 0]))
# slave.corner_force = np.array([-1, 0.1, 0.4])

fig, ax = plt.subplots(subplot_kw=dict(projection='3d', proj_type='ortho'))
ax.set_title('Contact Detection')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_aspect('equal')
ax.view_init(azim=45, elev=25)

sol = surf.contact_check_through_reference(slave, dt)
_, _, (xi, eta), _ = sol
print('Contact Detection Solution:')
print(sol, end='\n\n')

if sol[0]:
    del_tc = sol[1]
else:
    del_tc = None

surf.contact_visual_through_reference(ax, slave, dt, del_tc, only_contact=False)

N = surf.get_normal(sol[2], del_tc)
phi_k = phi_p_2D(xi, eta, surf.xi_p, surf.eta_p)
fc_guess = surf.get_fc_guess(slave, N, dt, phi_k)
force_sol = surf.find_fc(slave, np.array([xi, eta, fc_guess]), dt, N)
print('\nForce Solution:')
print(surf.normal_increment([slave], [(xi, eta, 1)], [N], dt))
print(force_sol, end='\n\n')
xi, eta, fc = force_sol[0]

fig2, ax2 = plt.subplots(subplot_kw=dict(projection='3d', proj_type='ortho'))
ax2.set_title('Normal Force')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_aspect('equal')
ax2.view_init(azim=45, elev=25)

# slave.contact_force = N*fc
# phi_k = phi_p_2D(xi, eta, surf.xi_p, surf.eta_p)
# for phi, node in zip(phi_k, surf.nodes):
#     node.contact_force = -N*fc*phi

surf.contact_visual_through_reference(ax2, slave, dt, None, only_contact=False)

# clear the previous force
slave.contact_force = np.zeros(3, dtype=np.float64)
for node in surf.nodes:
    node.contact_force = np.zeros(3, dtype=np.float64)

glued_force_sol = surf.find_glue_force(slave, np.array([1, 1, 1]), dt, sol[2])
print('Glue Force Solution:')
print(glued_force_sol, end='\n\n')
G = glued_force_sol[0]

fig3, ax3 = plt.subplots(subplot_kw=dict(projection='3d', proj_type='ortho'))
ax3.set_title('Glue Force')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.set_aspect('equal')
ax3.view_init(azim=45, elev=25)

slave.contact_force = G
phi_k = phi_p_2D(sol[2][0], sol[2][1], surf.xi_p, surf.eta_p)
for phi, node in zip(phi_k, surf.nodes):
    node.contact_force = -G*phi

surf.contact_visual_through_reference(ax3, slave, dt, None, only_contact=False)

plt.show()
