from contact import MeshBody, GlobalMesh
import meshio
import numpy as np

surf = 27
dt = 1.1e-6

mesh_data1 = meshio.read(r'Meshes\Block.msh')
mesh_data2 = meshio.read(r'Meshes\Block2.msh')
mesh1 = MeshBody(mesh_data1.points, mesh_data1.cells_dict, velocity=np.float64([0, 500, 0]))
mesh2 = MeshBody(mesh_data2.points, mesh_data2.cells_dict)
glob_mesh = GlobalMesh(mesh1, mesh2, bs=0.5)

_, nodes = glob_mesh.find_nodes(surf, dt)
for node in nodes:
    print(f'Testing contact between surface {surf} and node {node}:')
    print(glob_mesh.contact_check(surf, node, dt))
    print()
