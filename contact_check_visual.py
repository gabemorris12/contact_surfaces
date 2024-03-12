import matplotlib.pyplot as plt
import numpy as np

from contact import Node, Surface

dt = 0.1

# Define points for surface
points = np.array([
    [0.5, 0.5, 1],  # Updated Point 1
    [1, 0.5, 2],  # Updated Point 2
    [1, 1, 3],  # Updated Point 3
    [0.5, 1, 2]  # Updated Point 4
])

# Define velocity for points
vels = np.array([
    [0.12, 0.08, -0.05],
    [0.7*3, 0.75*3, -0.25*3],
    [-0.06, -0.03, -0.34],
    [-0.065, -0.035, -0.42]
])

nodes = [Node(i, pos, vel) for i, (pos, vel) in enumerate(list(zip(points, vels)))]
surf = Surface(0, nodes)
surf.reverse_dir()


# Define a separate point
sep_point = np.array([0.75, 0.75, 1])

# Create the figure and axis
fig, ax = plt.subplots(subplot_kw=dict(projection='3d', proj_type='ortho'))
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.view_init(15, 15)

v = np.array([2, -0.1, 10.5])
later_sep_point = sep_point + v*dt
sep_node = Node(len(nodes), sep_point, v)

# Perform contact check
check, del_tc = surf.contact_check(sep_node, dt)
print('Time to Contact:', del_tc)
contact_point = sep_point + v*del_tc
print('Contact Point:', contact_point)
print()
at_contact = points + vels*del_tc
print('Surface Points at Contact:')
print(at_contact)

surf.contact_visual(ax, sep_node, dt, del_tc)

plt.show()
