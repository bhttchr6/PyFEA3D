import torch
import numpy as np
from utils import  coo_submatrix_pull
import scipy.sparse.linalg as sla

def LinearSolver(Fext, Fint, Ktan, freedofs):


    Fint_f = torch.take(Fint, freedofs)
    Fext_f = torch.take(Fext, freedofs)
    Residual_f = torch.add(Fext_f, -Fint_f)

    Ktan_f = coo_submatrix_pull(Ktan, freedofs, freedofs)

    # Obtain displacement vector
    uhat = torch.tensor(sla.spsolve(Ktan_f.tocsr(), Residual_f)).float()

    return uhat
