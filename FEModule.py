import torch
import numpy as np
from utils import GaussPointsWeights, MatrixBuilder
from ShapeFunctions import SHAPEL3D
from GaussPointFunc import gausspointeval
import scipy.sparse
import scipy.sparse.linalg

def FEA(nnodes, ndim, cells, points, edofMat, re_order, U, dev):
    E, nu = 1., 0.3 # Material properties
    a_builder = MatrixBuilder()
    Fint = torch.zeros((nnodes * ndim, 1), device=dev)

    nnodes, ndim = points.shape
    nel, nnodes_per_elem = cells.shape

    U_vec = U.flatten()
    XYZ_vec = points.flatten()

    U_vec_global = U_vec[edofMat.flatten()]
    XYZ_vec_global = XYZ_vec[edofMat.flatten().astype(int)]

    global_el_disp = torch.reshape(U_vec_global,(nel, nnodes_per_elem, ndim)).to(dev)
    global_el_XYZ =  np.reshape(XYZ_vec_global,(nel, nnodes_per_elem, ndim))




    mu = E / (2 * (1 + nu))
    lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    gw, itpts = GaussPointsWeights()

    ## Initialize matrices
    fint = torch.zeros((nel, 24, 1), device=dev)
    K = torch.zeros((24, 24), device=dev)
    Kgeo = torch.zeros((24, 24), device=dev)
    Kelem = torch.zeros((1, nel, 24, 24), device=dev)

    #for gp in range(8):
    shi_0, eta_0, xi_0 = itpts[0, 0], itpts[0, 1], itpts[0, 2]
    shi_1, eta_1, xi_1 = itpts[1, 0], itpts[1, 1], itpts[1, 2]
    shi_2, eta_2, xi_2 = itpts[2, 0], itpts[2, 1], itpts[2, 2]
    shi_3, eta_3, xi_3 = itpts[3, 0], itpts[3, 1], itpts[3, 2]
    shi_4, eta_4, xi_4 = itpts[4, 0], itpts[4, 1], itpts[4, 2]
    shi_5, eta_5, xi_5 = itpts[5, 0], itpts[5, 1], itpts[5, 2]
    shi_6, eta_6, xi_6 = itpts[6, 0], itpts[6, 1], itpts[6, 2]
    shi_7, eta_7, xi_7 = itpts[7, 0], itpts[7, 1], itpts[7, 2]

    dN_0, BG_0, B_0, SHPD_0, DET_0 = SHAPEL3D(shi_0, eta_0, xi_0, global_el_XYZ, nel, ndim, dev)
    dN_1, BG_1, B_1, SHPD_1, DET_1 = SHAPEL3D(shi_1, eta_1, xi_1, global_el_XYZ, nel, ndim, dev)
    dN_2, BG_2, B_2, SHPD_2, DET_2 = SHAPEL3D(shi_2, eta_2, xi_2, global_el_XYZ, nel, ndim, dev)
    dN_3, BG_3, B_3, SHPD_3, DET_3 = SHAPEL3D(shi_3, eta_3, xi_3, global_el_XYZ, nel, ndim, dev)
    dN_4, BG_4, B_4, SHPD_4, DET_4 = SHAPEL3D(shi_4, eta_4, xi_4, global_el_XYZ, nel, ndim, dev)
    dN_5, BG_5, B_5, SHPD_5, DET_5 = SHAPEL3D(shi_5, eta_5, xi_5, global_el_XYZ, nel, ndim, dev)
    dN_6, BG_6, B_6, SHPD_6, DET_6 = SHAPEL3D(shi_6, eta_6, xi_6, global_el_XYZ, nel, ndim, dev)
    dN_7, BG_7, B_7, SHPD_7, DET_7 = SHAPEL3D(shi_7, eta_7, xi_7, global_el_XYZ, nel, ndim, dev)



    K_0, Kgeo_0, cauchy_stress_vec_0 = gausspointeval(dN_0, BG_0, B_0, SHPD_0, DET_0, global_el_disp, mu, lmbda, gw, ndim, nel, dev)
    K_1, Kgeo_1, cauchy_stress_vec_1 = gausspointeval(dN_1, BG_1, B_1, SHPD_1, DET_1, global_el_disp, mu, lmbda, gw, ndim, nel, dev)
    K_2, Kgeo_2, cauchy_stress_vec_2 = gausspointeval(dN_2, BG_2, B_2, SHPD_2, DET_2, global_el_disp, mu, lmbda, gw, ndim, nel, dev)
    K_3, Kgeo_3, cauchy_stress_vec_3 = gausspointeval(dN_3, BG_3, B_3, SHPD_3, DET_3, global_el_disp, mu, lmbda, gw, ndim, nel, dev)
    K_4, Kgeo_4, cauchy_stress_vec_4 = gausspointeval(dN_4, BG_4, B_4, SHPD_4, DET_4, global_el_disp, mu, lmbda, gw, ndim, nel, dev)
    K_5, Kgeo_5, cauchy_stress_vec_5 = gausspointeval(dN_5, BG_5, B_5, SHPD_5, DET_5, global_el_disp, mu, lmbda, gw, ndim, nel, dev)
    K_6, Kgeo_6, cauchy_stress_vec_6 = gausspointeval(dN_6, BG_6, B_6, SHPD_6, DET_6, global_el_disp, mu, lmbda, gw, ndim, nel, dev)
    K_7, Kgeo_7, cauchy_stress_vec_7 = gausspointeval(dN_7, BG_7, B_7, SHPD_7, DET_7, global_el_disp, mu, lmbda, gw, ndim, nel, dev)


    Kelem = Kelem + torch.add(K_0, Kgeo_0)
    Kelem = Kelem + torch.add(K_1, Kgeo_1)
    Kelem = Kelem + torch.add(K_2, Kgeo_2)
    Kelem = Kelem + torch.add(K_3, Kgeo_3)
    Kelem = Kelem + torch.add(K_4, Kgeo_4)
    Kelem = Kelem + torch.add(K_5, Kgeo_5)
    Kelem = Kelem + torch.add(K_6, Kgeo_6)
    Kelem = Kelem + torch.add(K_7, Kgeo_7)

    fint = fint + cauchy_stress_vec_0
    fint = fint + cauchy_stress_vec_1
    fint = fint + cauchy_stress_vec_2
    fint = fint + cauchy_stress_vec_3
    fint = fint + cauchy_stress_vec_4
    fint = fint + cauchy_stress_vec_5
    fint = fint + cauchy_stress_vec_6
    fint = fint + cauchy_stress_vec_7


    ## assemble elemental matrices and vectors
    iK = np.kron(edofMat, np.ones((24, 1))).flatten()
    jK = np.kron(edofMat, np.ones((1, 24))).flatten()
    sK = ((Kelem.flatten()[np.newaxis]).T).flatten()
    Ktan = scipy.sparse.coo_matrix(
        (sK, (iK, jK)), shape=(nnodes*ndim, nnodes*ndim)).tocsr().tocoo()


    #Ktan = a_builder.coo_matrix().tocsr().tocoo()

    fint_vec = ((fint.flatten()[np.newaxis]).T).flatten()
    iK_int = edofMat.flatten().astype(int)

    for id, vi in enumerate(iK_int):

        Fint[vi] += fint_vec[id]
        

    return Fint, Ktan