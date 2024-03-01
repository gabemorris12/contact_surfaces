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
    [0.12, 0.08, -0.05],
    [0.7*3, 0.75*3, -0.25*3],
    [-0.06, -0.03, -0.34],
    [-0.065, -0.035, -0.42]
])

nodes = [Node(i, pos, vel) for i, (pos, vel) in enumerate(list(zip(points, vels)))]
surf = Surface(0, nodes)
surf.reverse_dir()

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
v = np.array([2, -0.1, 10.5])
print(v)
ax.quiver(*(list(sep_point) + list(v*scale)), arrow_length_ratio=0.15)
later_sep_point = sep_point + v*dt
sep_node = Node(len(nodes), sep_point, v)

# Plot the surface
vertices = [list(points)]
surface = Poly3DCollection(vertices, alpha=0.25, facecolor='cyan')
ax.add_collection3d(surface)

# Plot triangular patches with the centroid
shifted = np.roll(later_points, -1, axis=0)
centroid = np.sum(points, axis=0)/len(points)
vel_centroid = np.sum(vels, axis=0)/len(points)
later_centroid = centroid + vel_centroid*dt
for p1, p2 in zip(later_points, shifted):
    patch = [[p1, p2, later_centroid]]
    ax.plot(np.array(patch)[0][:, 0], np.array(patch)[0][:, 1], np.array(patch)[0][:, 2], color='maroon')
    patch = Poly3DCollection(patch, alpha=0.25, facecolor='maroon')
    ax.add_collection3d(patch)
# new_vertices = [list(later_points)]
# new_surface = Poly3DCollection(new_vertices, alpha=0.25, facecolor='maroon')
# ax.add_collection3d(new_surface)

# Plot the points
ax.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue')
ax.scatter(later_points[:, 0], later_points[:, 1], later_points[:, 2], color='orange')
ax.scatter([later_centroid[0]], [later_centroid[1]], [later_centroid[2]], color='orange')
ax.scatter([sep_point[0]], [sep_point[1]], [sep_point[2]], color='red')
ax.scatter([later_sep_point[0]], [later_sep_point[1]], [later_sep_point[2]], color='green')

# Labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Perform contact check
print(surf.contact_check(sep_node, dt))

plt.show()
