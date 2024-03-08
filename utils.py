import torch
import numpy as np
import scipy as S
import scipy.sparse as sparse

def get_device():
    if torch.cuda.is_available():
        print("CUDA is available, running on GPU")
        dev = torch.device('cuda')
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        print("CUDA not available, running on CPU")
        dev = torch.device('cpu')
    return dev
class MatrixBuilder:
    def __init__(self):
        self.rows = []
        self.cols = []
        self.vals = []

    def add(self, rows, cols, submat):
        for i, ri in enumerate(rows):
            for j, cj in enumerate(cols):
                self.rows.append(ri)
                self.cols.append(cj)
                self.vals.append(submat[i, j])

    def coo_matrix(self):
        return sparse.coo_matrix((self.vals, (self.rows, self.cols)))


def coo_submatrix_pull(matr, rows, cols):
    """
    Pulls out an arbitrary i.e. non-contiguous submatrix out of
    a sparse.coo_matrix.
    """
    if type(matr) != S.sparse.coo_matrix:
        raise TypeError('Matrix must be sparse COOrdinate format')

    gr = -1 * np.ones(matr.shape[0])
    gc = -1 * np.ones(matr.shape[1])

    lr = len(rows)
    lc = len(cols)

    ar = np.arange(0, lr)
    ac = np.arange(0, lc)
    gr[rows[ar]] = ar
    gc[cols[ac]] = ac
    mrow = matr.row
    mcol = matr.col
    newelem = (gr[mrow] > -1) & (gc[mcol] > -1)
    newrows = mrow[newelem]
    newcols = mcol[newelem]
    return S.sparse.coo_matrix((matr.data[newelem], np.array([gr[newrows],
                                                              gc[newcols]])), (lr, lc))



def GaussPointsWeights():
    GN_x = np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)])
    GN_y = GN_x;
    GN_z = GN_x;
    itpts = np.array([[-1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)], \
                      [1 / np.sqrt(3), -1 / np.sqrt(3), 1 / np.sqrt(3)], \
                      [1 / np.sqrt(3), -1 / np.sqrt(3), -1 / np.sqrt(3)], \
                      [-1 / np.sqrt(3), -1 / np.sqrt(3), -1 / np.sqrt(3)], \
                      [-1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)], \
                      [1 / np.sqrt(3), 1 / np.sqrt(3), 1 / np.sqrt(3)], \
                      [+1 / np.sqrt(3), 1 / np.sqrt(3), -1 / np.sqrt(3)], \
                      [-1 / np.sqrt(3), +1 / np.sqrt(3), -1 / np.sqrt(3)]])
    GaussWeight = np.array([1, 1])
    return GaussWeight, itpts