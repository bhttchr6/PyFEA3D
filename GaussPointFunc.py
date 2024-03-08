import torch
import numpy as np


def gausspointeval(dN, BG, B, SHPD, DET, ELU, mu, lmbda, gw, ndim, nel, dev):
    x = torch.eye(3)
    EYE = x
    # x = np.reshape(x,(1, 3, 3))
    x = torch.reshape(x, (1, 3, 3))
    x = x.to(dev)
    # y = np.repeat(x[np.newaxis,:,:],nelx*nely*nelz, axis=0)
    y = x.repeat(nel, 1, 1)
    # print(y)
    ELU_T = torch.permute(ELU, (0, 2, 1))
    SHPD_T = torch.permute(SHPD, (0, 2, 1))

    F1 = ELU_T @ SHPD_T
    F = F1 + y

    # if itpt==1:

    # u_pred_torch = ELU
    # xyz_tensor = ELXYZ
    # duxdxyz = grad(u_pred_torch[:, 0].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev),
    #        create_graph=True, retain_graph=True)[0]
    # duydxyz = grad(u_pred_torch[:, 1].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev),
    #        create_graph=True, retain_graph=True)[0]
    # duzdxyz = grad(u_pred_torch[:, 2].unsqueeze(1), xyz_tensor, torch.ones(xyz_tensor.size()[0], 1, device=dev),
    #        create_graph=True, retain_graph=True)[0]
    # F11 = duxdxyz[:, 0].unsqueeze(1) + 1
    # F12 = duxdxyz[:, 1].unsqueeze(1) + 0
    # F13 = duxdxyz[:, 2].unsqueeze(1) + 0
    # F21 = duydxyz[:, 0].unsqueeze(1) + 0
    # F22 = duydxyz[:, 1].unsqueeze(1) + 1
    # F23 = duydxyz[:, 2].unsqueeze(1) + 0
    # F31 = duzdxyz[:, 0].unsqueeze(1) + 0
    # F32 = duzdxyz[:, 1].unsqueeze(1) + 0
    # F33 = duzdxyz[:, 2].unsqueeze(1) + 1
    F11 = F[:, 0, 0]
    F12 = F[:, 0, 1]
    F13 = F[:, 0, 2]
    F21 = F[:, 1, 0]
    F22 = F[:, 1, 1]
    F23 = F[:, 1, 2]
    F31 = F[:, 2, 0]
    F32 = F[:, 2, 1]
    F33 = F[:, 2, 2]
    detF = F11 * (F22 * F33 - F23 * F32) - F12 * (F21 * F33 - F23 * F31) + F13 * (F21 * F32 - F22 * F31)
    J = detF
    # print(detF)
    F_T = torch.permute(F, (0, 2, 1))
    BB = F_T @ F
    # if itpt ==1:
    # print('FE_templates.py BB', BB.shape)
    # print('FE_templates.py y', y.shape)

    cnst_1 = mu / J
    # print('before unsqueeze', cnst_1)
    cnst_1 = cnst_1.unsqueeze(1)
    # print('after unsqueeze', cnst_1)
    CAUCHY_1 = torch.add(BB, -y)

    # if itpt ==1:
    # print('FE template.py CAUCHY_1', CAUCHY_1)
    cnst_2 = (lmbda / J) * torch.log(J)
    cnst_2 = cnst_2.unsqueeze(1)

    CAUCHY_1 = cnst_1[:, :, None] * CAUCHY_1

    CAUCHY_2 = cnst_2[:, :, None] * y
    # if itpt ==1:
    # print('FE template.py CNST_2', cnst_2)
    # print('FE template.py CAUCHY_2', torch.log(J))
    CAUCHY = torch.add(CAUCHY_1, CAUCHY_2)
    CAUCHY_00 = CAUCHY[:, 0, 0]
    CAUCHY_11 = CAUCHY[:, 1, 1]
    CAUCHY_22 = CAUCHY[:, 2, 2]
    CAUCHY_12 = CAUCHY[:, 1, 2]
    CAUCHY_02 = CAUCHY[:, 0, 2]
    CAUCHY_01 = CAUCHY[:, 0, 1]

    cauchy_cauchy = torch.stack((CAUCHY_00.unsqueeze(1), CAUCHY_11.unsqueeze(1), CAUCHY_22.unsqueeze(1),
                                 CAUCHY_01.unsqueeze(1), CAUCHY_12.unsqueeze(1), CAUCHY_02.unsqueeze(1)), 1)
    # cauchy_T = torch.permute(cauchy_cauchy,(0,2,1))
    # if itpt==1:
    # print('FE_templates.py F', F)
    # print('FE_templates.py CAUCH_cauchy', cauchy_cauchy)
    invF11 = (F22 * F33 - F23 * F32) / detF
    invF12 = -(F12 * F33 - F13 * F32) / detF
    invF13 = (F12 * F23 - F13 * F22) / detF
    invF21 = -(F21 * F33 - F23 * F31) / detF
    invF22 = (F11 * F33 - F13 * F31) / detF
    invF23 = -(F11 * F23 - F13 * F21) / detF
    invF31 = (F21 * F32 - F22 * F31) / detF
    invF32 = -(F11 * F32 - F12 * F31) / detF
    invF33 = (F11 * F22 - F12 * F21) / detF
    J = detF
    C11 = F11 ** 2 + F21 ** 2 + F31 ** 2
    C12 = F11 * F12 + F21 * F22 + F31 * F32
    C13 = F11 * F13 + F21 * F23 + F31 * F33
    C21 = F12 * F11 + F22 * F21 + F32 * F31
    C22 = F12 ** 2 + F22 ** 2 + F32 ** 2
    C23 = F12 * F13 + F22 * F23 + F32 * F33
    C31 = F13 * F11 + F23 * F21 + F33 * F31
    C32 = F13 * F12 + F23 * F22 + F33 * F32
    C33 = F13 ** 2 + F23 ** 2 + F33 ** 2

    E11 = 0.5 * (C11 - 1)
    E12 = 0.5 * C12
    E13 = 0.5 * C13
    E21 = 0.5 * C21
    E22 = 0.5 * (C22 - 1)
    E23 = 0.5 * C23
    E31 = 0.5 * C31
    E32 = 0.5 * C32
    E33 = 0.5 * (C33 - 1)

    P11 = mu * F11 + (lmbda * torch.log(detF) - mu) * invF11
    P12 = mu * F12 + (lmbda * torch.log(detF) - mu) * invF21
    P13 = mu * F13 + (lmbda * torch.log(detF) - mu) * invF31
    P21 = mu * F21 + (lmbda * torch.log(detF) - mu) * invF12
    P22 = mu * F22 + (lmbda * torch.log(detF) - mu) * invF22
    P23 = mu * F23 + (lmbda * torch.log(detF) - mu) * invF32
    P31 = mu * F31 + (lmbda * torch.log(detF) - mu) * invF13
    P32 = mu * F32 + (lmbda * torch.log(detF) - mu) * invF23
    P33 = mu * F33 + (lmbda * torch.log(detF) - mu) * invF33
    # print('P11',invF11)
    # print('P12',invF12)
    # print('P13',invF13)
    # print('P21',invF21)
    # print('P22',invF22)
    # print('P23',invF23)
    # print('P31',invF31)
    # print('P32',invF32)
    # print('P33',invF33)

    S11 = invF11 * P11 + invF12 * P21 + invF13 * P31
    S12 = invF11 * P12 + invF12 * P22 + invF13 * P32
    S13 = invF11 * P13 + invF12 * P23 + invF13 * P33
    S21 = invF21 * P11 + invF22 * P21 + invF23 * P31
    S22 = invF21 * P12 + invF22 * P22 + invF23 * P32
    S23 = invF21 * P13 + invF22 * P23 + invF23 * P33
    S31 = invF31 * P11 + invF32 * P21 + invF33 * P31
    S32 = invF31 * P12 + invF32 * P22 + invF33 * P32
    S33 = invF31 * P13 + invF32 * P23 + invF33 * P33
    # print(detF)

    C_11 = (F11 * (F11 * S11 + F12 * S12 + F13 * S13) + F12 * (F11 * S12 + F12 * S22 + F13 * S23) + F13 * (
                F11 * S13 + F12 * S23 + F13 * S33)) / J
    C_12 = (F11 * (F12 * S11 + F22 * S12 + F23 * S13) + F12 * (F12 * S12 + F22 * S22 + F23 * S23) + F13 * (
                F12 * S13 + F22 * S23 + F23 * S33)) / J
    C_13 = (F11 * (F13 * S11 + F23 * S12 + F33 * S13) + F12 * (F13 * S12 + F23 * S22 + F33 * S23) + F13 * (
                F13 * S13 + F23 * S23 + F33 * S33)) / J
    C_21 = (F12 * (F11 * S11 + F12 * S12 + F13 * S13) + F22 * (F11 * S12 + F12 * S22 + F13 * S23) + F23 * (
                F11 * S13 + F12 * S23 + F13 * S33)) / J
    C_22 = (F12 * (F12 * S11 + F22 * S12 + F23 * S13) + F22 * (F12 * S12 + F22 * S22 + F23 * S23) + F23 * (
                F12 * S13 + F22 * S23 + F23 * S33)) / J
    C_23 = (F12 * (F13 * S11 + F23 * S12 + F33 * S13) + F22 * (F13 * S12 + F23 * S22 + F33 * S23) + F23 * (
                F13 * S13 + F23 * S23 + F33 * S33)) / J
    C_31 = (F13 * (F11 * S11 + F12 * S12 + F13 * S13) + F23 * (F11 * S12 + F12 * S22 + F13 * S23) + F33 * (
                F11 * S13 + F12 * S23 + F13 * S33)) / J
    C_32 = (F13 * (F12 * S11 + F22 * S12 + F23 * S13) + F23 * (F12 * S12 + F22 * S22 + F23 * S23) + F33 * (
                F12 * S13 + F22 * S23 + F23 * S33)) / J
    C_33 = (F13 * (F13 * S11 + F23 * S12 + F33 * S13) + F23 * (F13 * S12 + F23 * S22 + F33 * S23) + F33 * (
                F13 * S13 + F23 * S23 + F33 * S33)) / J

    #### calculate the material stiffness for each element
    lam1 = lmbda / J;
    mu1 = (mu - lmbda * torch.log(J)) / J;
    # print(J)
    ####======= formulate the stiffness matrix for hyperleastic material=====
    lam1 = lam1.detach().cpu().numpy()
    mu1 = mu1.detach().cpu().numpy()
    CS11 = lam1 + 2 * mu1
    CS12 = lam1
    CS13 = lam1
    CS22 = lam1 + 2 * mu1
    CS23 = lam1
    CS33 = lam1 + 2 * mu1
    CS44 = mu1
    CS55 = mu1
    CS66 = mu1
    # print('FE_templates.py cs11', CS11.shape)
    # CS11_size = CS11.size(dim = 1)
    # theta = torch.zeros(theta_size)

    # print('FE_templates.py cs11',CS11)
    # print('FE_templates.py cs11 shape',CS11.size)
    ZEROS = np.zeros(CS11.size)
    r1 = np.c_[CS11, CS12, CS13, ZEROS, ZEROS, ZEROS]
    r2 = np.c_[CS12, CS22, CS23, ZEROS, ZEROS, ZEROS]
    r3 = np.c_[CS13, CS23, CS33, ZEROS, ZEROS, ZEROS]
    r4 = np.c_[ZEROS, ZEROS, ZEROS, CS44, ZEROS, ZEROS]
    r5 = np.c_[ZEROS, ZEROS, ZEROS, ZEROS, CS55, ZEROS]
    r6 = np.c_[ZEROS, ZEROS, ZEROS, ZEROS, ZEROS, CS66]

    Dhyp_base = torch.tensor(np.stack((r1, r2, r3, r4, r5, r6), 2), device=dev).float()
    KH = Dhyp_base
    # print('FE_templates.py KH', KH)
    ####=============================================

    # print(C11.size())
    ZERO = torch.zeros(C11.size())
    ZERO = ZERO.to(dev)
    # ZERO = torch.tensor(0.)
    # print(C11)
    # print(C_11)

    # SHEAD = torch.tensor([[C11,C12,C13,ZERO,ZERO,ZERO,ZERO,ZERO,ZERO],\
    #                       [C12,C22,C23,ZERO,ZERO,ZERO,ZERO,ZERO,ZERO],\
    #                        [C13,C23,C33,ZERO,ZERO,ZERO,ZERO,ZERO,ZERO],\
    #                            [ZERO,ZERO,ZERO,C11,C12,C13,ZERO,ZERO,ZERO],\
    #                                [ZERO,ZERO,ZERO,C12,C22,C23,ZERO,ZERO,ZERO],\
    #                                    [ZERO,ZERO,ZERO,C13,C23,C33,ZERO,ZERO,ZERO],\
    #                                        [ZERO,ZERO,ZERO,ZERO,ZERO,ZERO,C11,C12,C13],\
    #                                            [ZERO,ZERO,ZERO,ZERO,ZERO,ZERO,C12,C22,C23],\
    #                                                [ZERO,ZERO,ZERO,ZERO,ZERO,ZERO,C13,C23,C33]])

    SHEAD_1 = torch.stack((
                          C_11.unsqueeze(0), C_12.unsqueeze(0), C_13.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0),
                          ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0)), 2)
    SHEAD_2 = torch.stack((
                          C_12.unsqueeze(0), C_22.unsqueeze(0), C_23.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0),
                          ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0)), 2)
    SHEAD_3 = torch.stack((
                          C_13.unsqueeze(0), C_23.unsqueeze(0), C_33.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0),
                          ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0)), 2)
    SHEAD_4 = torch.stack((
                          ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0), C_11.unsqueeze(0), C_12.unsqueeze(0),
                          C_13.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0)), 2)
    SHEAD_5 = torch.stack((
                          ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0), C_12.unsqueeze(0), C_22.unsqueeze(0),
                          C_23.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0)), 2)
    SHEAD_6 = torch.stack((
                          ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0), C_13.unsqueeze(0), C_23.unsqueeze(0),
                          C_33.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0)), 2)
    SHEAD_7 = torch.stack((
                          ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0),
                          ZERO.unsqueeze(0), C_11.unsqueeze(0), C_12.unsqueeze(0), C_13.unsqueeze(0)), 2)
    SHEAD_8 = torch.stack((
                          ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0),
                          ZERO.unsqueeze(0), C_12.unsqueeze(0), C_22.unsqueeze(0), C_23.unsqueeze(0)), 2)
    SHEAD_9 = torch.stack((
                          ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0), ZERO.unsqueeze(0),
                          ZERO.unsqueeze(0), C_13.unsqueeze(0), C_23.unsqueeze(0), C_33.unsqueeze(0)), 2)
    SHEAD = torch.stack((SHEAD_1, SHEAD_2, SHEAD_3, SHEAD_4, SHEAD_5, SHEAD_6, SHEAD_7, SHEAD_8, SHEAD_9), 2)

    # SHEAD_1 = np.c_[C11,C12,C13,ZERO,ZERO,ZERO,ZERO,ZERO,ZERO]
    # SHEAD_2 = np.c_[C12,C22,C23,ZERO,ZERO,ZERO,ZERO,ZERO,ZERO]
    # SHEAD_3 = np.c_[C13,C23,C33,ZERO,ZERO,ZERO,ZERO,ZERO,ZERO]
    # SHEAD_4 = np.c_[ZERO,ZERO,ZERO,C11,C12,C13,ZERO,ZERO,ZERO]
    # SHEAD_5 = np.c_[ZERO,ZERO,ZERO,C12,C22,C23,ZERO,ZERO,ZERO]
    # SHEAD_6 = np.c_[ZERO,ZERO,ZERO,C13,C23,C33,ZERO,ZERO,ZERO]
    # SHEAD_7 = np.c_[ZERO,ZERO,ZERO,ZERO,ZERO,ZERO,C11,C12,C13]
    # SHEAD_8 = np.c_[ZERO,ZERO,ZERO,ZERO,ZERO,ZERO,C12,C22,C23]
    # SHEAD_9 = np.c_[ZERO,ZERO,ZERO,ZERO,ZERO,ZERO,C13,C23,C33]
    # SHEAD = torch.tensor(np.stack((SHEAD_1,SHEAD_2,SHEAD_3,SHEAD_4,SHEAD_5,SHEAD_6,SHEAD_7,SHEAD_8,SHEAD_9),1)).float()
    # print('shead', SHEAD)
    # if itpt ==1:
    # print('FE_templates.py SHEAD', SHEAD)

    # SHEAD = torch.reshape(SHEAD,(nelx*nely*nelz,9,9))
    # SHEAD_2 = torch.cat([C11,C12,C13,ZERO,ZERO,ZERO,ZERO,ZERO,ZERO],dim = 0)
    # print('BG', BG)
    # SHEAD = reshape_f(SHEAD,(nelx*nely*nelz,9,9))
    # print('bgshape',BG.shape)
    BG_T = torch.permute(BG, (0, 2, 1))
    # BG_T = torch.einsum('...ij->...ji', BG)
    EKF2 = BG_T @ SHEAD @ BG
    # BG_T = torch.trans(BG,0,2,1)
    DET = DET.unsqueeze(1)
    # print('DET',DET.shape)
    zero_np = torch.zeros(invF11.size())
    zero_np = zero_np.to(dev)

    G_1 = torch.stack((invF11.unsqueeze(0), invF12.unsqueeze(0), invF13.unsqueeze(0), zero_np.unsqueeze(0),
                       zero_np.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0),
                       zero_np.unsqueeze(0)), 2)
    G_2 = torch.stack((invF11.unsqueeze(0), invF12.unsqueeze(0), invF13.unsqueeze(0), zero_np.unsqueeze(0),
                       zero_np.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0),
                       zero_np.unsqueeze(0)), 2)
    G_3 = torch.stack((invF11.unsqueeze(0), invF12.unsqueeze(0), invF13.unsqueeze(0), zero_np.unsqueeze(0),
                       zero_np.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0),
                       zero_np.unsqueeze(0)), 2)
    G_4 = torch.stack((zero_np.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0), invF11.unsqueeze(0),
                       invF12.unsqueeze(0), invF13.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0),
                       zero_np.unsqueeze(0)), 2)
    G_5 = torch.stack((zero_np.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0), invF11.unsqueeze(0),
                       invF12.unsqueeze(0), invF13.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0),
                       zero_np.unsqueeze(0)), 2)
    G_6 = torch.stack((zero_np.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0), invF11.unsqueeze(0),
                       invF12.unsqueeze(0), invF13.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0),
                       zero_np.unsqueeze(0)), 2)
    G_7 = torch.stack((zero_np.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0),
                       zero_np.unsqueeze(0), zero_np.unsqueeze(0), invF11.unsqueeze(0), invF12.unsqueeze(0),
                       invF13.unsqueeze(0)), 2)
    G_8 = torch.stack((zero_np.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0),
                       zero_np.unsqueeze(0), zero_np.unsqueeze(0), invF11.unsqueeze(0), invF12.unsqueeze(0),
                       invF13.unsqueeze(0)), 2)
    G_9 = torch.stack((zero_np.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0), zero_np.unsqueeze(0),
                       zero_np.unsqueeze(0), zero_np.unsqueeze(0), invF11.unsqueeze(0), invF12.unsqueeze(0),
                       invF13.unsqueeze(0)), 2)

    G = torch.stack((G_1, G_2, G_3, G_4, G_5, G_6, G_7, G_8, G_9), 2)

    # B = L@G@dN

    B_T = torch.permute(B, (0, 2, 1))
    # if itpt ==1:
    # print('FE_templates.py B_T', B_T.shape)
    # print('FE_templates.py B', B.shape)
    # print('FE_templates.py KH', KH.shape)
    K = 1 * 1 * 1 * B_T @ KH @ B
    K = DET[:, :, None] * K

    # K1 = 1*1*1*torch.matmul(B_T, KH)
    # K = torch.matmul(K1, B)
    Kgeo = 1 * 1 * 1 * EKF2
    Kgeo = DET[:, :, None] * Kgeo

    fint = 1 * 1 * 1 * DET[:, :, None] * B_T @ cauchy_cauchy


    return K, Kgeo, fint