import torch
import numpy as np
from numpy.linalg import inv
from numpy.linalg import det


def SHAPEL3D(x, y, z, ELXYZ, nel, ndim, dev):
    dN = np.zeros((9, 24))
    BG = np.zeros((9, 24))

    dNx = (1 / 8) * np.array([-(1 - y) * (1 - z), (1 - y) * (1 - z), (1 + y) * (1 - z), -(1 + y) * (1 - z), \
                              -(1 - y) * (1 + z), (1 - y) * (1 + z), (1 + y) * (1 + z), -(1 + y) * (1 + z)])
    dNy = (1 / 8) * np.array([-(1 - x) * (1 - z), -(1 + x) * (1 - z), (1 + x) * (1 - z), (1 - x) * (1 - z), \
                              -(1 - x) * (1 + z), -(1 + x) * (1 + z), (1 + x) * (1 + z), (1 - x) * (1 + z)])
    dNz = (1 / 8) * np.array([-(1 - x) * (1 - y), -(1 + x) * (1 - y), -(1 + x) * (1 + y), -(1 - x) * (1 + y), \
                              (1 - x) * (1 - y), (1 + x) * (1 - y), (1 + x) * (1 + y), (1 - x) * (1 + y)])

    dNxyz = np.array([
        [-(1 - y) * (1 - z) * (1 / 8), (1 / 8) * (1 - y) * (1 - z), (1 / 8) * (1 + y) * (1 - z),
         -(1 + y) * (1 - z) * (1 / 8), -(1 - y) * (1 + z) * (1 / 8), (1 - y) * (1 + z) * (1 / 8),
         (1 + y) * (1 + z) * (1 / 8), -(1 + y) * (1 + z) * (1 / 8)],
        [-(1 - x) * (1 - z) * (1 / 8), -(1 + x) * (1 - z) * (1 / 8), (1 + x) * (1 - z) * (1 / 8),
         (1 - x) * (1 - z) * (1 / 8), -(1 - x) * (1 + z) * (1 / 8), -(1 + x) * (1 + z) * (1 / 8),
         (1 + x) * (1 + z) * (1 / 8), (1 - x) * (1 + z) * (1 / 8)],
        [-(1 - x) * (1 - y) * (1 / 8), -(1 + x) * (1 - y) * (1 / 8), -(1 + x) * (1 + y) * (1 / 8),
         -(1 - x) * (1 + y) * (1 / 8), (1 - x) * (1 - y) * (1 / 8), (1 + x) * (1 - y) * (1 / 8),
         (1 + x) * (1 + y) * (1 / 8), (1 - x) * (1 + y) * (1 / 8)]
    ])
    # Delta = np.array([
    #     [-dx, dx, dx, -dx, -dx, dx, dx, -dx],
    #     [-dy, -dy, dy, dy, -dy, -dy, dy, dy],
    #     [-dz,-dz,-dz, -dz, dz, dz, dz,dz]
    #     ])
    XI1 = x
    XI2 = y
    XI3 = z
    x1 = ELXYZ[:, 0, 0]
    y1 = ELXYZ[:, 0, 1]
    z1 = ELXYZ[:, 0, 2]
    # --------------
    x2 = ELXYZ[:, 1, 0]
    y2 = ELXYZ[:, 1, 1]
    z2 = ELXYZ[:, 1, 2]
    # --------------
    x3 = ELXYZ[:, 2, 0]
    y3 = ELXYZ[:, 2, 1]
    z3 = ELXYZ[:, 2, 2]
    # --------------
    x4 = ELXYZ[:, 3, 0]
    y4 = ELXYZ[:, 3, 1]
    z4 = ELXYZ[:, 3, 2]
    # --------------
    x5 = ELXYZ[:, 4, 0]
    y5 = ELXYZ[:, 4, 1]
    z5 = ELXYZ[:, 4, 2]
    # --------------
    x6 = ELXYZ[:, 5, 0]
    y6 = ELXYZ[:, 5, 1]
    z6 = ELXYZ[:, 5, 2]
    # --------------
    x7 = ELXYZ[:, 6, 0]
    y7 = ELXYZ[:, 6, 1]
    z7 = ELXYZ[:, 6, 2]
    # --------------
    x8 = ELXYZ[:, 7, 0]
    y8 = ELXYZ[:, 7, 1]
    z8 = ELXYZ[:, 7, 2]
    # --------------
    #print('ELXYZ',ELXYZ[:,0,0])

    size_XI2 = x1.size
    # print(size_XI2)
    one = 1.0
    J = np.array([
        [-0.125 * x1 * (one - XI2) * (one - XI3) + 0.125 * x2 * (one - XI2) * (one - XI3) + \
         0.125 * x3 * (one + XI2) * (one - XI3) - 0.125 * x4 * (one + XI2) * (one - XI3) - \
         0.125 * x5 * (one - XI2) * (one + XI3) + 0.125 * x6 * (one - XI2) * (one + XI3) + \
         0.125 * x7 * (one + XI2) * (one + XI3) - \
         0.125 * x8 * (one + XI2) * (one + XI3), -0.125 * (one - XI2) * (one - XI3) * y1 + \
         0.125 * (one - XI2) * (one - XI3) * y2 + 0.125 * (one + XI2) * (one - XI3) * y3 - \
         0.125 * (one + XI2) * (one - XI3) * y4 - 0.125 * (one - XI2) * (one + XI3) * y5 + \
         0.125 * (one - XI2) * (one + XI3) * y6 + 0.125 * (one + XI2) * (one + XI3) * y7 - \
         0.125 * (one + XI2) * (one + XI3) * y8, -0.125 * (one - XI2) * (one - XI3) * z1 + \
         0.125 * (one - XI2) * (one - XI3) * z2 + 0.125 * (one + XI2) * (one - XI3) * z3 - \
         0.125 * (one + XI2) * (one - XI3) * z4 - 0.125 * (one - XI2) * (one + XI3) * z5 + \
         0.125 * (one - XI2) * (one + XI3) * z6 + 0.125 * (one + XI2) * (one + XI3) * z7 - \
         0.125 * (one + XI2) * (one + XI3) * z8],
        [-0.125 * x1 * (one - XI1) * (one - XI3) + 0.125 * x4 * (one - XI1) * (one - XI3) - \
         0.125 * x2 * (one + XI1) * (one - XI3) + 0.125 * x3 * (one + XI1) * (one - XI3) - \
         0.125 * x5 * (one - XI1) * (one + XI3) + 0.125 * x8 * (one - XI1) * (one + XI3) - \
         0.125 * x6 * (one + XI1) * (one + XI3) + \
         0.125 * x7 * (one + XI1) * (one + XI3), -0.125 * (one - XI1) * (one - XI3) * y1 - \
         0.125 * (one + XI1) * (one - XI3) * y2 + 0.125 * (one + XI1) * (one - XI3) * y3 + \
         0.125 * (one - XI1) * (one - XI3) * y4 - 0.125 * (one - XI1) * (one + XI3) * y5 - \
         0.125 * (one + XI1) * (one + XI3) * y6 + 0.125 * (one + XI1) * (one + XI3) * y7 + \
         0.125 * (one - XI1) * (one + XI3) * y8, -0.125 * (one - XI1) * (one - XI3) * z1 - \
         0.125 * (one + XI1) * (one - XI3) * z2 + 0.125 * (one + XI1) * (one - XI3) * z3 + \
         0.125 * (one - XI1) * (one - XI3) * z4 - 0.125 * (one - XI1) * (one + XI3) * z5 - \
         0.125 * (one + XI1) * (one + XI3) * z6 + 0.125 * (one + XI1) * (one + XI3) * z7 + \
         0.125 * (one - XI1) * (one + XI3) * z8],
        [-0.125 * x1 * (one - XI1) * (one - XI2) + 0.125 * x5 * (one - XI1) * (one - XI2) - \
         0.125 * x2 * (one + XI1) * (one - XI2) + 0.125 * x6 * (one + XI1) * (one - XI2) - \
         0.125 * x4 * (one - XI1) * (one + XI2) + 0.125 * x8 * (one - XI1) * (one + XI2) - \
         0.125 * x3 * (one + XI1) * (one + XI2) + \
         0.125 * x7 * (one + XI1) * (one + XI2), -0.125 * (one - XI1) * (one - XI2) * y1 - \
         0.125 * (one + XI1) * (one - XI2) * y2 - 0.125 * (one + XI1) * (one + XI2) * y3 - \
         0.125 * (one - XI1) * (one + XI2) * y4 + 0.125 * (one - XI1) * (one - XI2) * y5 + \
         0.125 * (one + XI1) * (one - XI2) * y6 + 0.125 * (one + XI1) * (one + XI2) * y7 + \
         0.125 * (one - XI1) * (one + XI2) * y8, -0.125 * (one - XI1) * (one - XI2) * z1 - \
         0.125 * (one + XI1) * (one - XI2) * z2 - 0.125 * (one + XI1) * (one + XI2) * z3 - \
         0.125 * (one - XI1) * (one + XI2) * z4 + 0.125 * (one - XI1) * (one - XI2) * z5 + \
         0.125 * (one + XI1) * (one - XI2) * z6 + 0.125 * (one + XI1) * (one + XI2) * z7 + \
         0.125 * (one - XI1) * (one + XI2) * z8]
    ])

    # if ctr==0:
    # print(ctr)
    # print(J)
    J1 = -0.125 * x1 * (one - XI2) * (one - XI3) + 0.125 * x2 * (one - XI2) * (one - XI3) + \
         0.125 * x3 * (one + XI2) * (one - XI3) - 0.125 * x4 * (one + XI2) * (one - XI3) - \
         0.125 * x5 * (one - XI2) * (one + XI3) + 0.125 * x6 * (one - XI2) * (one + XI3) + \
         0.125 * x7 * (one + XI2) * (one + XI3) - \
         0.125 * x8 * (one + XI2) * (one + XI3), -0.125 * (one - XI2) * (one - XI3) * y1 + \
         0.125 * (one - XI2) * (one - XI3) * y2 + 0.125 * (one + XI2) * (one - XI3) * y3 - \
         0.125 * (one + XI2) * (one - XI3) * y4 - 0.125 * (one - XI2) * (one + XI3) * y5 + \
         0.125 * (one - XI2) * (one + XI3) * y6 + 0.125 * (one + XI2) * (one + XI3) * y7 - \
         0.125 * (one + XI2) * (one + XI3) * y8, -0.125 * (one - XI2) * (one - XI3) * z1 + \
         0.125 * (one - XI2) * (one - XI3) * z2 + 0.125 * (one + XI2) * (one - XI3) * z3 - \
         0.125 * (one + XI2) * (one - XI3) * z4 - 0.125 * (one - XI2) * (one + XI3) * z5 + \
         0.125 * (one - XI2) * (one + XI3) * z6 + 0.125 * (one + XI2) * (one + XI3) * z7 - \
         0.125 * (one + XI2) * (one + XI3) * z8

    J2 = -0.125 * x1 * (one - XI1) * (one - XI3) + 0.125 * x4 * (one - XI1) * (one - XI3) - \
         0.125 * x2 * (one + XI1) * (one - XI3) + 0.125 * x3 * (one + XI1) * (one - XI3) - \
         0.125 * x5 * (one - XI1) * (one + XI3) + 0.125 * x8 * (one - XI1) * (one + XI3) - \
         0.125 * x6 * (one + XI1) * (one + XI3) + \
         0.125 * x7 * (one + XI1) * (one + XI3), -0.125 * (one - XI1) * (one - XI3) * y1 - \
         0.125 * (one + XI1) * (one - XI3) * y2 + 0.125 * (one + XI1) * (one - XI3) * y3 + \
         0.125 * (one - XI1) * (one - XI3) * y4 - 0.125 * (one - XI1) * (one + XI3) * y5 - \
         0.125 * (one + XI1) * (one + XI3) * y6 + 0.125 * (one + XI1) * (one + XI3) * y7 + \
         0.125 * (one - XI1) * (one + XI3) * y8, -0.125 * (one - XI1) * (one - XI3) * z1 - \
         0.125 * (one + XI1) * (one - XI3) * z2 + 0.125 * (one + XI1) * (one - XI3) * z3 + \
         0.125 * (one - XI1) * (one - XI3) * z4 - 0.125 * (one - XI1) * (one + XI3) * z5 - \
         0.125 * (one + XI1) * (one + XI3) * z6 + 0.125 * (one + XI1) * (one + XI3) * z7 + \
         0.125 * (one - XI1) * (one + XI3) * z8

    J3 = -0.125 * x1 * (one - XI1) * (one - XI2) + 0.125 * x5 * (one - XI1) * (one - XI2) - \
         0.125 * x2 * (one + XI1) * (one - XI2) + 0.125 * x6 * (one + XI1) * (one - XI2) - \
         0.125 * x4 * (one - XI1) * (one + XI2) + 0.125 * x8 * (one - XI1) * (one + XI2) - \
         0.125 * x3 * (one + XI1) * (one + XI2) + \
         0.125 * x7 * (one + XI1) * (one + XI2), -0.125 * (one - XI1) * (one - XI2) * y1 - \
         0.125 * (one + XI1) * (one - XI2) * y2 - 0.125 * (one + XI1) * (one + XI2) * y3 - \
         0.125 * (one - XI1) * (one + XI2) * y4 + 0.125 * (one - XI1) * (one - XI2) * y5 + \
         0.125 * (one + XI1) * (one - XI2) * y6 + 0.125 * (one + XI1) * (one + XI2) * y7 + \
         0.125 * (one - XI1) * (one + XI2) * y8, -0.125 * (one - XI1) * (one - XI2) * z1 - \
         0.125 * (one + XI1) * (one - XI2) * z2 - 0.125 * (one + XI1) * (one + XI2) * z3 - \
         0.125 * (one - XI1) * (one + XI2) * z4 + 0.125 * (one - XI1) * (one - XI2) * z5 + \
         0.125 * (one + XI1) * (one - XI2) * z6 + 0.125 * (one + XI1) * (one + XI2) * z7 + \
         0.125 * (one - XI1) * (one + XI2) * z8
    J = np.c_[J1, J2, J3]
    J = np.ravel(J)
    J = np.reshape(J, (nel, ndim, ndim), order='F')

    # np.array([J1][J2],[J3])
    JINV = inv(J)
    # print(JINV)
    DET = det(J)
    # print(DET)
    # G = np.array([
    #     [JINV[0,0], JINV[0,1] , JINV[0,2], 0, 0, 0, 0, 0, 0],
    #     [JINV[1,0], JINV[1,1], JINV[1,2], 0, 0, 0, 0, 0, 0],
    #     [JINV[2,0], JINV[2,1], JINV[2,2], 0, 0, 0, 0, 0, 0],
    #     [0, 0, 0, JINV[0,0], JINV[0,1], JINV[0,2], 0, 0, 0],
    #     [0, 0, 0, JINV[1,0], JINV[1,1], JINV[1,2], 0, 0, 0],
    #     [0, 0, 0, JINV[2,0], JINV[2,1], JINV[2,2], 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, JINV[0,0], JINV[0,1], JINV[0,2]],
    #     [0, 0, 0, 0, 0, 0, JINV[1,0], JINV[1,1], JINV[1,2]],
    #     [0, 0, 0, 0, 0, 0, JINV[2,0], JINV[2,1], JINV[2,2]]
    #     ])
    GDSF = JINV @ dNxyz
    # print(GDSF)
    dN[0, 0:24:3] = dNx
    dN[1, 0:24:3] = dNy
    dN[2, 0:24:3] = dNz
    dN[3, 1:24:3] = dNx
    dN[4, 1:24:3] = dNy
    dN[5, 1:24:3] = dNz
    dN[6, 2:24:3] = dNx
    dN[7, 2:24:3] = dNy
    dN[8, 2:24:3] = dNz

    BG_1 = np.c_[
        GDSF[:, 0, 0], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 1], np.zeros(size_XI2), np.zeros(size_XI2), \
            GDSF[:, 0, 2], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 3], np.zeros(size_XI2), np.zeros(
            size_XI2), \
            GDSF[:, 0, 4], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 5], np.zeros(size_XI2), np.zeros(
            size_XI2), \
            GDSF[:, 0, 6], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 7], np.zeros(size_XI2), np.zeros(
            size_XI2)]
    BG_2 = np.c_[
        GDSF[:, 1, 0], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 1], np.zeros(size_XI2), np.zeros(size_XI2), \
            GDSF[:, 1, 2], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 3], np.zeros(size_XI2), np.zeros(
            size_XI2), \
            GDSF[:, 1, 4], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 5], np.zeros(size_XI2), np.zeros(
            size_XI2), \
            GDSF[:, 1, 6], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 7], np.zeros(size_XI2), np.zeros(
            size_XI2)]
    BG_3 = np.c_[
        GDSF[:, 2, 0], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 1], np.zeros(size_XI2), np.zeros(size_XI2), \
            GDSF[:, 2, 2], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 3], np.zeros(size_XI2), np.zeros(
            size_XI2), \
            GDSF[:, 2, 4], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 5], np.zeros(size_XI2), np.zeros(
            size_XI2), \
            GDSF[:, 2, 6], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 7], np.zeros(size_XI2), np.zeros(
            size_XI2)]
    BG_4 = np.c_[
        np.zeros(size_XI2), GDSF[:, 0, 0], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 1], np.zeros(size_XI2), \
            np.zeros(size_XI2), GDSF[:, 0, 2], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 3], np.zeros(
            size_XI2), \
            np.zeros(size_XI2), GDSF[:, 0, 4], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 5], np.zeros(
            size_XI2), \
            np.zeros(size_XI2), GDSF[:, 0, 6], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 7], np.zeros(
            size_XI2)]
    BG_5 = np.c_[
        np.zeros(size_XI2), GDSF[:, 1, 0], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 1], np.zeros(size_XI2), \
            np.zeros(size_XI2), GDSF[:, 1, 2], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 3], np.zeros(
            size_XI2), \
            np.zeros(size_XI2), GDSF[:, 1, 4], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 5], np.zeros(
            size_XI2), \
            np.zeros(size_XI2), GDSF[:, 1, 6], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 7], np.zeros(
            size_XI2)]
    BG_6 = np.c_[
        np.zeros(size_XI2), GDSF[:, 2, 0], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 1], np.zeros(size_XI2), \
            np.zeros(size_XI2), GDSF[:, 2, 2], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 3], np.zeros(
            size_XI2), \
            np.zeros(size_XI2), GDSF[:, 2, 4], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 5], np.zeros(
            size_XI2), \
            np.zeros(size_XI2), GDSF[:, 2, 6], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 7], np.zeros(
            size_XI2)]

    BG_7 = np.c_[
        np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 0], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 1],
        np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 2], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 3], \
            np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 4], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0,
                                                                                                           5], \
            np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 6], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0,
                                                                                                           7]]
    BG_8 = np.c_[
        np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 0], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 1],
        np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 2], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 3], \
            np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 4], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1,
                                                                                                           5], \
            np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 6], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1,
                                                                                                           7]]
    BG_9 = np.c_[
        np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 0], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 1],
        np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 2], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 3], \
            np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 4], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2,
                                                                                                           5], \
            np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 6], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2,
                                                                                                           7]]

    #### ====== BN matrix ========================
    # BN_1 = np.c_[F11*GDSF[:,0,0], F21*GDSF[:,0,0], F31*GDSF[:,0,0], F11*GDSF[:,0,1], F21*GDSF[:,0,1], F31*GDSF[:,0,1],\
    #            F11*GDSF[:,0,2], F21*GDSF[:,0,2], F31*GDSF[:,0,2], F11*GDSF[:,0,3], F21*GDSF[:,0,3], F31*GDSF[:,0,3],\
    #                F11*GDSF[:,0,4], F21*GDSF[:,0,4], F31*GDSF[:,0,4], F11*GDSF[:,0,5], F21*GDSF[:,0,5], F31*GDSF[:,0,5],\
    #                    F11*GDSF[:,0,6], F21*GDSF[:,0,6], F31*GDSF[:,0,6], F11*GDSF[:,0,7], F21*GDSF[:,0,7], F31*GDSF[:,0,7]]
    # BN_2 = np.c_[F12*GDSF[:,1,0], F22*GDSF[:,1,0], F32*GDSF[:,1,0], F12*GDSF[:,1,1], F22*GDSF[:,1,1], F32*GDSF[:,1,1],\
    #            F12*GDSF[:,1,2], F22*GDSF[:,1,2], F32*GDSF[:,1,2], F12*GDSF[:,1,3], F22*GDSF[:,1,3], F32*GDSF[:,1,3],\
    #                F12*GDSF[:,1,4], F22*GDSF[:,1,4], F32*GDSF[:,1,4], F12*GDSF[:,1,5], F22*GDSF[:,1,5], F32*GDSF[:,1,5],\
    #                    F12*GDSF[:,1,6], F22*GDSF[:,1,6], F32*GDSF[:,1,6], F12*GDSF[:,1,7], F22*GDSF[:,1,7], F32*GDSF[:,1,7]]

    # BN_3 = np.c_[F13*GDSF[:,2,0], F23*GDSF[:,2,0], F33*GDSF[:,2,0], F13*GDSF[:,2,1], F23*GDSF[:,2,1], F33*GDSF[:,2,1],\
    #             F13*GDSF[:,2,2], F23*GDSF[:,2,2], F33*GDSF[:,2,2], F13*GDSF[:,2,3], F23*GDSF[:,2,3], F33*GDSF[:,2,3],\
    #                 F13*GDSF[:,2,4], F23*GDSF[:,2,4], F33*GDSF[:,2,4], F13*GDSF[:,2,5], F23*GDSF[:,2,5], F33*GDSF[:,2,5],\
    #                     F13*GDSF[:,2,6], F23*GDSF[:,2,6], F33*GDSF[:,2,6], F13*GDSF[:,2,7], F23*GDSF[:,2,7], F33*GDSF[:,2,7]]

    # BN_4 = np.c_[F12*GDSF[:,0,0]+ F11*GDSF[:,1,0], F22*GDSF[:,0,0]+ F21*GDSF[:,1,0], F32*GDSF[:,0,0] + F31*GDSF[:,1,0], F12*GDSF[:,0,1]+ F11*GDSF[:,1,1], F22*GDSF[:,0,1]+ F21*GDSF[:,1,1], F32*GDSF[:,0,1] + F31*GDSF[:,1,1],\
    #            F12*GDSF[:,0,2]+ F11*GDSF[:,1,2], F22*GDSF[:,0,2]+ F21*GDSF[:,1,2], F32*GDSF[:,0,2] + F31*GDSF[:,1,2], F12*GDSF[:,0,3]+ F11*GDSF[:,1,3], F22*GDSF[:,0,3]+ F21*GDSF[:,1,3], F32*GDSF[:,0,3] + F31*GDSF[:,1,3],\
    #                F12*GDSF[:,0,4]+ F11*GDSF[:,1,4], F22*GDSF[:,0,4]+ F21*GDSF[:,1,4], F32*GDSF[:,0,4] + F31*GDSF[:,1,4], F12*GDSF[:,0,5]+ F11*GDSF[:,1,5], F22*GDSF[:,0,5]+ F21*GDSF[:,1,5], F32*GDSF[:,0,5] + F31*GDSF[:,1,5],\
    #                    F12*GDSF[:,0,6]+ F11*GDSF[:,1,6], F22*GDSF[:,0,6]+ F21*GDSF[:,1,6], F32*GDSF[:,0,6] + F31*GDSF[:,1,6], F12*GDSF[:,0,7]+ F11*GDSF[:,1,7], F22*GDSF[:,0,7]+ F21*GDSF[:,1,7], F32*GDSF[:,0,7] + F31*GDSF[:,1,7]]

    # BN_5 = np.c_[F13*GDSF[:,1,0]+ F12*GDSF[:,2,0], F23*GDSF[:,1,0]+ F22*GDSF[:,2,0], F33*GDSF[:,1,0] + F32*GDSF[:,2,0], F13*GDSF[:,1,1]+ F12*GDSF[:,2,1], F23*GDSF[:,1,1]+ F22*GDSF[:,2,1], F33*GDSF[:,1,1] + F32*GDSF[:,2,1],\
    #           F13*GDSF[:,1,2]+ F12*GDSF[:,2,2], F23*GDSF[:,1,2]+ F22*GDSF[:,2,2], F33*GDSF[:,1,2] + F32*GDSF[:,2,2], F13*GDSF[:,1,3]+ F12*GDSF[:,2,3], F23*GDSF[:,1,3]+ F22*GDSF[:,2,3], F33*GDSF[:,1,3] + F32*GDSF[:,2,3],\
    #               F13*GDSF[:,1,4]+ F12*GDSF[:,2,4], F23*GDSF[:,1,4]+ F22*GDSF[:,2,4], F33*GDSF[:,1,4] + F32*GDSF[:,2,4], F13*GDSF[:,1,5]+ F12*GDSF[:,2,5], F23*GDSF[:,1,5]+ F22*GDSF[:,2,5], F33*GDSF[:,1,5] + F32*GDSF[:,2,5],\
    #                   F13*GDSF[:,1,6]+ F12*GDSF[:,2,6], F23*GDSF[:,1,6]+ F22*GDSF[:,2,6], F33*GDSF[:,1,6] + F32*GDSF[:,2,6], F13*GDSF[:,1,7]+ F12*GDSF[:,2,7], F23*GDSF[:,1,7]+ F22*GDSF[:,2,7], F33*GDSF[:,1,7] + F32*GDSF[:,2,7]]

    # BN_6 = np.c_[F13*GDSF[:,0,0]+ F11*GDSF[:,2,0], F23*GDSF[:,1,0]+ F21*GDSF[:,2,0], F33*GDSF[:,0,0] + F31*GDSF[:,2,0], F13*GDSF[:,0,1]+ F11*GDSF[:,2,1], F23*GDSF[:,1,1]+ F21*GDSF[:,2,1], F33*GDSF[:,0,1] + F31*GDSF[:,2,1],\
    #           F13*GDSF[:,0,2]+ F11*GDSF[:,2,2], F23*GDSF[:,1,2]+ F21*GDSF[:,2,2], F33*GDSF[:,0,2] + F31*GDSF[:,2,2], F13*GDSF[:,0,3]+ F11*GDSF[:,2,3], F23*GDSF[:,1,3]+ F21*GDSF[:,2,3], F33*GDSF[:,0,3] + F31*GDSF[:,2,3],\
    #              F13*GDSF[:,0,4]+ F11*GDSF[:,2,4], F23*GDSF[:,1,4]+ F21*GDSF[:,2,4], F33*GDSF[:,0,4] + F31*GDSF[:,2,4], F13*GDSF[:,0,5]+ F11*GDSF[:,2,5], F23*GDSF[:,1,5]+ F21*GDSF[:,2,5], F33*GDSF[:,0,5] + F31*GDSF[:,2,5],\
    #                   F13*GDSF[:,0,6]+ F11*GDSF[:,2,6], F23*GDSF[:,1,6]+ F21*GDSF[:,2,6], F33*GDSF[:,0,6] + F31*GDSF[:,2,6], F13*GDSF[:,0,7]+ F11*GDSF[:,2,7], F23*GDSF[:,1,7]+ F21*GDSF[:,2,7], F33*GDSF[:,0,7] + F31*GDSF[:,2,7]]

    # BN = np.stack((BN_1,BN_2,BN_3,BN_4,BN_5,BN_6),1)
    # print('BG_1',BG_1)

    ####===================================================================
    B_1 = np.c_[
        GDSF[:, 0, 0], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 1], np.zeros(size_XI2), np.zeros(size_XI2), \
            GDSF[:, 0, 2], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 3], np.zeros(size_XI2), np.zeros(
            size_XI2), \
            GDSF[:, 0, 4], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 5], np.zeros(size_XI2), np.zeros(
            size_XI2), \
            GDSF[:, 0, 6], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 0, 7], np.zeros(size_XI2), np.zeros(
            size_XI2)]
    B_2 = np.c_[
        np.zeros(size_XI2), GDSF[:, 1, 0], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 1], np.zeros(size_XI2), \
            np.zeros(size_XI2), GDSF[:, 1, 2], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 3], np.zeros(
            size_XI2), \
            np.zeros(size_XI2), GDSF[:, 1, 4], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 5], np.zeros(
            size_XI2), \
            np.zeros(size_XI2), GDSF[:, 1, 6], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 1, 7], np.zeros(
            size_XI2)]

    B_3 = np.c_[
        np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 0], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 1], \
            np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 2], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2,
                                                                                                           3], \
            np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 4], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2,
                                                                                                           5], \
            np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2, 6], np.zeros(size_XI2), np.zeros(size_XI2), GDSF[:, 2,
                                                                                                           7]]

    B_4 = np.c_[GDSF[:, 1, 0], GDSF[:, 0, 0], np.zeros(size_XI2), GDSF[:, 1, 1], GDSF[:, 0, 1], np.zeros(size_XI2), \
        GDSF[:, 1, 2], GDSF[:, 0, 2], np.zeros(size_XI2), GDSF[:, 1, 3], GDSF[:, 0, 3], np.zeros(size_XI2), \
        GDSF[:, 1, 4], GDSF[:, 0, 4], np.zeros(size_XI2), GDSF[:, 1, 5], GDSF[:, 0, 5], np.zeros(size_XI2), \
        GDSF[:, 1, 6], GDSF[:, 0, 6], np.zeros(size_XI2), GDSF[:, 1, 7], GDSF[:, 0, 7], np.zeros(size_XI2)]

    B_5 = np.c_[np.zeros(size_XI2), GDSF[:, 2, 0], GDSF[:, 1, 0], np.zeros(size_XI2), GDSF[:, 2, 1], GDSF[:, 1, 1], \
        np.zeros(size_XI2), GDSF[:, 2, 2], GDSF[:, 1, 2], np.zeros(size_XI2), GDSF[:, 2, 3], GDSF[:, 1, 3], \
        np.zeros(size_XI2), GDSF[:, 2, 4], GDSF[:, 1, 4], np.zeros(size_XI2), GDSF[:, 2, 5], GDSF[:, 1, 5], \
        np.zeros(size_XI2), GDSF[:, 2, 6], GDSF[:, 1, 6], np.zeros(size_XI2), GDSF[:, 2, 7], GDSF[:, 1, 7]]

    B_6 = np.c_[GDSF[:, 2, 0], np.zeros(size_XI2), GDSF[:, 0, 0], GDSF[:, 2, 1], np.zeros(size_XI2), GDSF[:, 0, 1], \
        GDSF[:, 2, 2], np.zeros(size_XI2), GDSF[:, 0, 2], GDSF[:, 2, 3], np.zeros(size_XI2), GDSF[:, 0, 3], \
        GDSF[:, 2, 4], np.zeros(size_XI2), GDSF[:, 0, 4], GDSF[:, 2, 5], np.zeros(size_XI2), GDSF[:, 0, 5], \
        GDSF[:, 2, 6], np.zeros(size_XI2), GDSF[:, 0, 6], GDSF[:, 2, 7], np.zeros(size_XI2), GDSF[:, 0, 7]]

    BG = np.stack((BG_1, BG_2, BG_3, BG_4, BG_5, BG_6, BG_7, BG_8, BG_9), 1)
    B = np.stack((B_1, B_2, B_3, B_4, B_5, B_6), 1)
    # print('BG',BG)
    # BG = np.ravel(BG)
    # BG = np.reshape(BG,(nelx*nely*nelz,9,24), order = 'F')
    # print(BG)
    GDSF = torch.tensor(GDSF, device=dev).float()
    BG = torch.tensor(BG, device=dev).float()
    B = torch.tensor(B, device=dev).float()
    DET = torch.tensor(DET, device=dev).float()
    dN = torch.tensor(dN, device=dev).float()
    return dN, BG, B, GDSF, DET

