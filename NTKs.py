
# "Kernel-Based Smoothness Analysis of Residual Networks"

import math
import numpy as np

sigW2 = 1.0
sigV2 = 1.0

# NTK for MLP with L_hidden nonlinear hidden layers and ReLUs
# returns Gram matrix: an array K_NTK of size (N, N)
def kernel_value_mlp(X, L_hidden):
    S = np.matmul(X, X.T) / X.shape[1]
    H = np.zeros_like(S)
    for dep in range(L_hidden+1):
        H += S
        K_NTK = H
        L = np.diag(S)
        P = np.clip(np.sqrt(np.outer(L, L)), a_min = 1e-9, a_max = None)
        Sn = np.clip(S / P, a_min = -1, a_max = 1)
        S = (Sn * (math.pi - np.arccos(Sn)) + np.sqrt(1.0 - Sn * Sn)) * P / 2.0 / math.pi * sigW2
        H = H * (math.pi - np.arccos(Sn)) / 2.0 / math.pi * sigW2
    return K_NTK


# NTK for MLP with L_hidden nonlinear hidden layers and ReLUs
# returns Gram matrix: an array K_NTK of size (N, N)
# Equivalent to "kernel_value_mlp" but more interpretable (separates GP kernel and NTK)
# and structured more similarly to the ResNet case
def kernel_value_mlp_detailed(X, L_hidden):
    K_GP_all = np.zeros((L_hidden+1, X.shape[0], X.shape[0]))
    Sigma_all = np.zeros((L_hidden+1, X.shape[0], X.shape[0]))
    SigmaDot_all = np.zeros((L_hidden+1, X.shape[0], X.shape[0]))

    Sigma = np.matmul(X, X.T) / X.shape[1]
    K_GP = Sigma
    K_GP_all[0,:,:] = K_GP
    for ell in range(1,L_hidden+1):
        L = np.diag(K_GP)
        P = np.clip(np.sqrt(np.outer(L, L)), a_min = 1e-9, a_max = None)
        Sn = np.clip(K_GP / P, a_min = -1, a_max = 1)
        Sigma = (Sn * (math.pi - np.arccos(Sn)) + np.sqrt(1.0 - Sn * Sn)) * P / 2.0 / math.pi * sigW2
        SigmaDot = (math.pi - np.arccos(Sn)) / 2.0 / math.pi * sigW2

        K_GP = Sigma

        # save computations
        K_GP_all[ell,:,:] = K_GP
        Sigma_all[ell, :, :] = Sigma
        SigmaDot_all[ell, :, :] = SigmaDot

    Pi = np.ones((X.shape[0], X.shape[0]))
    K_NTK = np.zeros((X.shape[0], X.shape[0]))
    for ell in range(L_hidden,-1,-1):
        K_NTK += Pi * K_GP_all[ell, :, :]
        Pi = Pi * SigmaDot_all[ell, :, :]

    return K_NTK


# NTK for ResNet with L_hidden nonlinear hidden layers and ReLUs
# returns Gram matrix: an array K_NTK of size (N, N)
def kernel_value_resnet(X, L_hidden, alpha):
    K_GP_all = np.zeros((L_hidden+1, X.shape[0], X.shape[0]))
    Sigma_all = np.zeros((L_hidden+1, X.shape[0], X.shape[0]))
    SigmaDot_all = np.zeros((L_hidden+1, X.shape[0], X.shape[0]))
    # recall that python indices start from 0,
    # so K_GP_all[0,:,:] is K_1 in the paper and K_GP_all[L,:,:] is K_{L+1} in the paper

    Sigma = np.matmul(X, X.T) / X.shape[1]
    Sigma_all[0, :, :] = Sigma
    K_GP = Sigma
    K_GP_all[0,:,:] = K_GP
    for ell in range(1,L_hidden+1):
        L = np.diag(K_GP)
        P = np.clip(np.sqrt(np.outer(L, L)), a_min = 1e-9, a_max = None)
        Sn = np.clip(K_GP / P, a_min = -1, a_max = 1)
        Sigma = (Sn * (math.pi - np.arccos(Sn)) + np.sqrt(1.0 - Sn * Sn)) * P / 2.0 / math.pi * sigW2 * sigV2
        SigmaDot = (math.pi - np.arccos(Sn)) / 2.0 / math.pi * sigW2 * sigV2

        K_GP = K_GP + alpha**2 * Sigma

        # save computations
        K_GP_all[ell,:,:] = K_GP
        Sigma_all[ell, :, :] = Sigma
        SigmaDot_all[ell, :, :] = SigmaDot

    Pi = np.ones((X.shape[0], X.shape[0]))
    K_NTK = K_GP_all[L_hidden, :, :]
    for ell in range(L_hidden,0,-1):
        K_NTK += alpha**2 * Pi * ( Sigma_all[ell, :, :] + K_GP_all[ell-1, :, :] * SigmaDot_all[ell, :, :] )
        Pi = Pi * (1 + alpha**2 * SigmaDot_all[ell, :, :] )
    K_NTK += Pi * K_GP_all[0,:,:]

    return K_NTK


