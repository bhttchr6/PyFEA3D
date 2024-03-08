import numpy as np
import torch

def ExternalForceEval( Fext, LE, load_dof, force_nodes, nel, force_mag, nnodes, ndim, nnodes_per_elem):

    ## Evaluate the frequency of occurances for each node number to apply force based on that
    dof_freq = np.zeros((nnodes * ndim), dtype=int)
    node_freq = np.zeros((np.size(force_nodes)), dtype=int)
    node_ctrs = np.zeros((np.size(force_nodes), 2), dtype=int)

    for row_number in range(nel):
        idxs = LE[row_number, :]


        for col_number in range(nnodes_per_elem):
            for j in range(np.size(force_nodes)):
                if idxs[col_number] == force_nodes[j]:
                    node_freq[j] = node_freq[j] + 1

                    node_ctrs[j] = [idxs[col_number], node_freq[j]]

    for i in range(np.size(force_nodes)):
        dof_freq[force_nodes[i] * 3] = node_freq[i]
        dof_freq[force_nodes[i] * 3 + 1] = node_freq[i]
        dof_freq[force_nodes[i] * 3 + 2] = node_freq[i]


    for ni in range(np.size(load_dof)):
        Fext[load_dof[ni]] = dof_freq[load_dof[ni]] * force_mag


    return Fext