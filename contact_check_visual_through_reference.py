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

fig, (ax, ax2, ax3) = plt.subplots(nrows=1, ncols=3, subplot_kw=dict(projection='3d', proj_type='ortho'))
ax.set_title('Contact Detection')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
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
ax.set_aspect('equal')

N = surf.get_normal(sol[2], del_tc)
phi_k = phi_p_2D(xi, eta, surf.xi_p, surf.eta_p)
fc_guess = surf.get_fc_guess(slave, N, dt, phi_k)
force_sol = surf.find_fc(slave, np.array([xi, eta, fc_guess]), dt, N)
print('\nForce Solution:')
print(surf.normal_increment([slave], [(xi, eta, 1)], [N], dt))
print(force_sol, end='\n\n')
xi, eta, fc = force_sol[0]

ax2.set_title('Normal Force')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.view_init(azim=45, elev=25)

# slave.contact_force = N*fc
# phi_k = phi_p_2D(xi, eta, surf.xi_p, surf.eta_p)
# for phi, node in zip(phi_k, surf.nodes):
#     node.contact_force = -N*fc*phi

surf.contact_visual_through_reference(ax2, slave, dt, None, only_contact=False)
ax2.set_aspect('equal')

# clear the previous force
slave.zero_contact()
surf.zero_contact()

glued_force_sol = surf.find_glue_force(slave, np.array([1, 1, 1]), dt, sol[2])
print('Glue Force Solution:')
print(surf.glue_increment([slave], [(1, 1, 1)], [sol[2]], dt))
print(glued_force_sol, end='\n\n')
G = glued_force_sol[0]

ax3.set_title('Glue Force')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_zlabel('z')
ax3.view_init(azim=45, elev=25)

# slave.contact_force = G
# phi_k = phi_p_2D(sol[2][0], sol[2][1], surf.xi_p, surf.eta_p)
# for phi, node in zip(phi_k, surf.nodes):
#     node.contact_force = -G*phi

surf.contact_visual_through_reference(ax3, slave, dt, None, only_contact=False)
ax3.set_aspect('equal')

plt.show()

"""
The output should be this

Contact Detection Solution:
(True, 0.6221606424928471, array([-0.43324096, -0.6       ]), 4)

Normal: [-0.36246427  0.85674929  0.36687915] at [-0.43324096 -0.6       ]

Force Solution:
[(-0.5814653653050698, -0.17636753340567574, 0.8585511497812912)]
(array([-0.58146537, -0.17636753,  0.85855115]), 4)

Glue Force Solution:
[(-0.40372708391823187, 0.8652150936465433, 0.0)]
(array([-0.40372708,  0.86521509,  0.        ]), 1)
"""
