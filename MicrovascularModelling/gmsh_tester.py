import sys

try:
    import gmsh
except ImportError:
    print("This demo requires gmsh to be installed")
    sys.exit(0)


from dolfinx.io import XDMFFile, gmshio
from mpi4py import MPI

gmsh.initialize()

# Choose if Gmsh output is verbose
gmsh.option.setNumber("General.Terminal", 0)
model = gmsh.model()
model.add("Sphere")
model.setCurrent("Sphere")
sphere = model.occ.addSphere(0, 0, 0, 1, tag=1)

# Synchronize OpenCascade representation with gmsh model
model.occ.synchronize()

# Add physical marker for cells. It is important to call this function
# after OpenCascade synchronization
model.add_physical_group(3, [sphere])


# Generate the mesh
model.mesh.generate(3)

msh, cell_markers, facet_markers = gmshio.model_to_mesh(model, MPI.COMM_SELF, 0)
msh.name = "Sphere"
cell_markers.name = f"{msh.name}_cells"
facet_markers.name = f"{msh.name}_facets"

with XDMFFile(msh.comm, f"out_gmsh/mesh_rank_{MPI.COMM_WORLD.rank}.xdmf", "w") as file:
    file.write_mesh(msh)
    file.write_meshtags(cell_markers)
    msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
    file.write_meshtags(facet_markers)
