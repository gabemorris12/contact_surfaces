import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
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
    [1.2, 0.8, -0.5],
    [0.7, 0.75, -0.25],
    [-0.6, -0.3, -3.4],
    [-0.65, -0.35, -4.2]
])

nodes = [Node(i, pos, vel) for i, (pos, vel) in enumerate(list(zip(points, vels)))]
surf = Surface(0, list(reversed(nodes)))

scale = 0.2  # For scaling velocity vectors. Only applied for visual purposes.

later_points = points + vels*dt  # Define the points in the next time step

# Define a separate point
sep_point = np.array([0.75, 0.75, 1])

# Create the figure and axis
fig, ax = plt.subplots(subplot_kw=dict(projection='3d', proj_type='ortho'))
ax: Axes3D = ax
ax.view_init(15, 15)

for p, v in zip(points, vels):
    args = list(p) + list(v*scale)
    ax.quiver(*args, arrow_length_ratio=0.15)

# c = 1/len(points)*np.sum(points, axis=0)
# v = (c - sep_point)*3 + np.array([.1, -2, 0])
v = np.array([0.1, -0.1, 8])
print(v)
ax.quiver(*(list(sep_point) + list(v*scale)), arrow_length_ratio=0.15)
later_sep_point = sep_point + v*dt
sep_node = Node(len(nodes), sep_point, v)

# Plot the surface
vertices = [list(points)]
surface = Poly3DCollection(vertices, alpha=.25, facecolor='cyan')
ax.add_collection3d(surface)

new_vertices = [list(later_points)]
new_surface = Poly3DCollection(new_vertices, alpha=0.25, facecolor='maroon')
ax.add_collection3d(new_surface)

# Plot the points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue')
ax.scatter(later_points[:, 0], later_points[:, 1], later_points[:, 2], color='orange')
ax.scatter([sep_point[0]], [sep_point[1]], [sep_point[2]], color='red')
ax.scatter([later_sep_point[0]], [later_sep_point[1]], [later_sep_point[2]], color='green')

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Perform contact check
print(surf.contact_check(sep_node, dt))

plt.show()
