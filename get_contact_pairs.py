from contact import MeshBody, GlobalMesh
import meshio
import numpy as np
import logging

logger = logging.getLogger('contact')
logger.setLevel(logging.INFO)

# master_patches = [27, 23, 32, 35]
master_patches = None
dt = 1.1e-6

mesh_data1 = meshio.read(r'Meshes/Block.msh')
mesh_data2 = meshio.read(r'Meshes/Block2.msh')
mesh1 = MeshBody(mesh_data1.points, mesh_data1.cells_dict, velocity=np.float64([0, 500, 0]))
mesh2 = MeshBody(mesh_data2.points, mesh_data2.cells_dict)
glob_mesh = GlobalMesh(mesh1, mesh2, bs=0.45, master_patches=master_patches)

contact_pairs = glob_mesh.get_contact_pairs(dt)
for pair in contact_pairs:
    print(pair)
