import numpy as np
import torch

def BoundaryCond(domain, ndim, nnodes):
    boundary = domain # define the boundary

    bc_num = np.count_nonzero(boundary == True)

    bc_dofs_x = np.zeros((bc_num,) ,dtype=int)
    bc_dofs_y = np.zeros((bc_num,) ,dtype=int)
    bc_dofs_z = np.zeros((bc_num,) ,dtype=int)
    ctr = 0
    for ib, vi in enumerate(boundary):

        if(boundary[ib] == True):

            bc_dofs_x[ctr] = ib *ndim
            bc_dofs_y[ctr] = ib *ndim + 1
            bc_dofs_z[ctr] = ib *ndim + 2
            ctr = ctr + 1


    all_dofs = np.arange(0 ,nnodes * ndim)
    bc_dofs_p = torch.tensor(np.sort(np.concatenate((bc_dofs_x, bc_dofs_y, bc_dofs_z))))     # prescribed dofs
    bc_dofs_f = torch.tensor(np.setdiff1d(all_dofs, bc_dofs_p))                              # free dofs

    return bc_dofs_f, bc_dofs_p, all_dofs

def get_dofs(domain, ndim, nnodes):



    dom_num = np.count_nonzero(domain == True)

    dofs_x = np.zeros((dom_num,), dtype=int)
    dofs_y = np.zeros((dom_num,), dtype=int)
    dofs_z = np.zeros((dom_num,), dtype=int)
    force_nodes = np.zeros((dom_num,), dtype=int)
    ctr = 0
    for ib, vi in enumerate(domain):

        if (domain[ib] == True):
            dofs_x[ctr] = ib * ndim
            dofs_y[ctr] = ib * ndim + 1
            dofs_z[ctr] = ib * ndim + 2
            force_nodes[ctr] = ib
            ctr = ctr + 1





    return dofs_x, dofs_y, dofs_z, force_nodes