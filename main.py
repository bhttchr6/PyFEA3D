
import os
import numpy as np
import torch
import time
from FEModule import  FEA
from Solver import  LinearSolver
from Mesher import  GenerateMesh, meshIO_read, meshIO_write
from BoundaryConditions import  BoundaryCond, get_dofs
from utils import  get_device
from ForceExternal import ExternalForceEval


dev = get_device()



Lx, Ly, Lz = 10., 5., 5.
Nx, Ny, Nz = 40, 20, 20




cell_type, ele_type = GenerateMesh(Lx, Ly, Lz, Nx, Ny, Nz)

nnodes, ndim, nel, nnodes_per_elem, points, cells, mesh = meshIO_read(cell_type, ele_type)

re_order = [4, 5, 1, 0, 7, 6, 2, 3]; # to order the node number according to traditional FEA
edofMat = np.kron(cells[:,re_order] * ndim, np.ones((1, ndim))) + \
            np.kron(np.ones((nel, nnodes_per_elem)), [0, 1, 2])



## Boundary conditions
tol = 1e-05
X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
left = ((np.abs(X - 0.) < tol)) # fixed
right = ((np.abs(X - Lx) < tol)) # loaded
freedofs, fixeddofs, alldofs = BoundaryCond(left, ndim, nnodes)


## Specify loading
right_bottom = ((np.abs(X - Lx) < tol) & (np.abs(Y - 0) < tol)) # loaded
load_dofs_x, load_dofs_y, load_dofs_z, force_nodes = get_dofs(right_bottom, ndim, nnodes)
load_dof, force_mag = load_dofs_y, -0.01
Fext = torch.zeros((nnodes*ndim, 1), device=dev)
Fext = ExternalForceEval( Fext, cells, load_dof, force_nodes, nel, force_mag, nnodes, ndim, nnodes_per_elem)


U = torch.zeros([nnodes,ndim],device=dev) # initialize displacement

start = time.time()
# FEA solution
Fint, Ktan = FEA(nnodes, ndim, cells, points, edofMat, re_order, U, dev)
uhat = LinearSolver(Fext, Fint, Ktan, freedofs)
end = time.time()

print("FEA finished, Time taken (secs): {:.2F}".format(time.time() - start))


# new solution
Uhat = U.flatten()
Uhat[freedofs] = uhat

# Get deformed coordinates to generate displaced mesh
meshIO_write(Uhat, mesh)



    
    
    
   
    
    

    

        

