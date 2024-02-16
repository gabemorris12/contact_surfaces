import meshio
import numpy as np

mesh = meshio.read(r'..\Meshes\Two_Blocks.msh')  # mesh consisting of all the bodies

# Find global bounds
bs = 0.5  # bucket size - this is the smallest dimension of a patch (make dynamic later)

x_max, y_max, z_max = np.amax(mesh.points, axis=0)  # maximum bounds of the global bounding box
x_min, y_min, z_min = np.amin(mesh.points, axis=0)  # minimum bounds of the global bounding box

Sx = int((x_max - x_min)/bs) + 1  # number of slices in the x direction
Sy = int((y_max - y_min)/bs) + 1  # number of slices in the y direction
Sz = int((z_max - z_min)/bs) + 1  # number of slices in the z direction
nb = Sx*Sy*Sz  # total number of buckets
n, _ = mesh.points.shape  # total number of nodes

# zeroing lists
nbox = np.zeros(nb, dtype=int)  # number of nodes in each bucket (corresponds to lbox)
lbox = np.zeros(n, dtype=int)  # the bucket id for each node (corresponds to mesh.points right now)
npoint = np.zeros(nb, dtype=int)  # pointers/indices that tell us how to find the nodes in a bucket corresponding to nsort
nsort = np.zeros(n, dtype=int)  # list of nodes sorted according to the bucket id

# Constructing lbox and nbox
for i, p in enumerate(mesh.points):
    x, y, z = p
    Si_x = int((x - x_min)/bs) + 1
    Si_y = int((y - y_min)/bs) + 1
    Si_z = int((z - z_min)/bs) + 1
    lbox[i] = (Si_z - 1)*Sx*Sy + (Si_y - 1)*Sx + Si_x - 1  # Do not forget to subtract 1 here
    nbox[lbox[i]] += 1

for j in range(1, nb):
    npoint[j] = npoint[j - 1] + nbox[j - 1]

nbox[:] = 0

for i in range(n):
    nsort[nbox[lbox[i]] + npoint[lbox[i]]] = i
    nbox[lbox[i]] += 1

# Find the capture box for the master surface of interest (shown as surface 5 in the paper)
surface = np.array([7, 26, 43, 25])
vel = np.full(3, 500)  # in/s
# vel = np.array([0, 500, 0])  # in/s
dt = 1.1e-6

added = np.array([mesh.points[i] + vel*dt for i in surface])
subtracted = np.array([mesh.points[i] - vel*dt for i in surface])
capture_box = np.concatenate((added, subtracted))

xc_min, yc_min, zc_min = np.amin(capture_box, axis=0)
xc_max, yc_max, zc_max = np.amax(capture_box, axis=0)

# Find the buckets containing the master surface
ibox_min = min(Sx, int((xc_min - x_min)/bs) + 1)
jbox_min = min(Sy, int((yc_min - y_min)/bs) + 1)
kbox_min = min(Sz, int((zc_min - z_min)/bs) + 1)
ibox_max = min(Sx, int((xc_max - x_min)/bs) + 1)
jbox_max = min(Sy, int((yc_max - y_min)/bs) + 1)
kbox_max = min(Sz, int((zc_max - z_min)/bs) + 1)

buckets, nodes = [], []
for i in range(ibox_min, ibox_max + 1):
    for j in range(jbox_min, jbox_max + 1):
        for k in range(kbox_min, kbox_max + 1):
            buckets.append((k - 1)*Sx*Sy + (j - 1)*Sx + i - 1)

# print(buckets)
for b in buckets:
    if nbox[b]:
        # print(b)
        # print(nsort[npoint[b]: npoint[b] + nbox[b]])
        nodes.extend(nsort[npoint[b]: npoint[b] + nbox[b]])
        # print()
nodes = np.setdiff1d(nodes, surface)
print(np.array(nodes) + 1)  # nodes considered for contact

for n in [13, 36, 48, 33]:
    assert n in nodes, f'Where is {n}?'
