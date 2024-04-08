from contact import Node, Surface
import numpy as np
import matplotlib.pyplot as plt

dt = 1
inc = 5

patch_nodes = [
    Node(0, np.array([-1, -1, 0]), np.array([0, 0, 1])),
    Node(1, np.array([1, -1, 0]), np.array([0, 0, 1])),
    Node(2, np.array([1, 1, 0]), np.array([0, 0, 1])),
    Node(3, np.array([-1, 1, 0]), np.array([0, 0, 1]))
]

slaves = [
    Node(4, np.array([0.5, 0, 1]), np.array([0, 0, -1.5])),
    Node(5, np.array([-0.5, 0, 1]), np.array([0, 0, -0.5])),
    Node(6, np.array([0, 0.5, 1]), np.array([0, 0, -1]))
    # Node(6, np.array([0, 0.5, 1]), np.array([0, 0, -0.25]))  # Tensile Behavior
]

surf = Surface(0, patch_nodes)

check_sol = [surf.contact_check_through_reference(slave, dt) for slave in slaves]
print('Contact Detection Solution:')
for slave, sol in zip(slaves, check_sol):
    print(f'{slave.label}: ', sol)

fig1, ax1 = plt.subplots(subplot_kw=dict(projection='3d', proj_type='ortho'))
ax1.set_title('Contact Detection')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
ax1.set_aspect('equal')

surf.contact_visual_through_reference(ax1, slaves, dt, [sol[1] for sol in check_sol], only_contact=False)

guesses = [(xi, eta, 1) for _, _, (xi, eta), _ in check_sol]
normals = [surf.get_normal(np.array([xi, eta]), del_tc) for _, del_tc, (xi, eta), _ in check_sol]

for _ in range(inc):
    surf.normal_increment(slaves, guesses, normals, dt)

fig2, ax2 = plt.subplots(subplot_kw=dict(projection='3d', proj_type='ortho'))
ax2.set_title('Normal Force Increments')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_zlabel('z')
ax2.set_aspect('equal')

surf.contact_visual_through_reference(ax2, slaves, dt, None, only_contact=False)

plt.show()
