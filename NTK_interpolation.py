
# "Kernel-Based Smoothness Analysis of Residual Networks"

import numpy as np
import matplotlib.pyplot as plt
from NTKs import *

L_nonlin_hidden = 5
N_training = 6
alpha_resnet = 0.1

N_testing = 1024*4


def runTargetFunc(x):
    y = 0.5 * np.cos(x) + np.sin(4*x)
    return y


# Closed-form interpolation by MLP NTK
def apply_MLP_NTK(X_test, X_train, y_train, L_hidden):
    diag_load = 1e-10
    #NTK_matrix = kernel_value_mlp(X_train, L_hidden) + diag_load*np.eye(X_train.shape[0])
    NTK_matrix = kernel_value_mlp_detailed(X_train, L_hidden) + diag_load*np.eye(X_train.shape[0])
    inv_NTK_matrix = np.linalg.inv(NTK_matrix)
    N_testing = X_test.shape[0]
    y_test = np.zeros(shape=(N_testing,1))
    for idx in range(N_testing):
        X_temp = np.concatenate((X_test[idx,None], X_train), axis=0)
        #NTK_matrix_temp = kernel_value_mlp(X_temp, L_hidden)
        NTK_matrix_temp = kernel_value_mlp_detailed(X_temp, L_hidden)
        NTK_vector_temp = NTK_matrix_temp[0,1:]
        y_inference = np.matmul(NTK_vector_temp, np.matmul(inv_NTK_matrix,y_train) )
        y_test[idx] = y_inference
        if np.mod(idx, 100) == 0:
            print('Inference for MLP NTK test idx: {}'.format(idx))
    return y_test.flatten()

# Closed-form interpolation by ResNet NTK
def apply_ResNet_NTK(X_test, X_train, y_train, L_hidden, alpha):
    diag_load = 1e-10
    NTK_matrix = kernel_value_resnet(X_train, L_hidden, alpha) + diag_load*np.eye(X_train.shape[0])
    inv_NTK_matrix = np.linalg.inv(NTK_matrix)
    N_testing = X_test.shape[0]
    y_test = np.zeros(shape=(N_testing,1))
    for idx in range(N_testing):
        X_temp = np.concatenate((X_test[idx,None], X_train), axis=0)
        NTK_matrix_temp = kernel_value_resnet(X_temp, L_hidden, alpha)
        NTK_vector_temp = NTK_matrix_temp[0,1:]
        y_inference = np.matmul(NTK_vector_temp, np.matmul(inv_NTK_matrix,y_train) )
        y_test[idx] = y_inference
        if np.mod(idx, 100) == 0:
            print('Inference for ResNet NTK test idx: {}'.format(idx))
    return y_test.flatten()


def main():

    T = 2*np.pi
    x_gt = T * np.linspace(-1/2,1/2,10000)
    y_func = lambda x: runTargetFunc(x)
    y_gt = y_func(x_gt)

    x_train = T * np.linspace(0.0, 1, N_training) - T/2
    y_train = y_func(x_train)
    x_train_orig = x_train
    y_train_orig = y_train
    x_train = x_train[:, None]
    x_train = np.concatenate((np.cos(x_train), np.sin(x_train)), axis=1)
    y_train = y_train[:, None]

    x_test = T/N_testing * np.arange(-N_testing/2,N_testing/2,1)
    x_test_orig = x_test
    x_test = x_test[:,None]
    x_test = np.concatenate((np.cos(x_test), np.sin(x_test)), axis=1)
    x_temp = np.concatenate((np.array([[1], [0]]).T, x_test), axis=0)

    MLP_NTK_regression = apply_MLP_NTK(x_test, x_train, y_train, L_nonlin_hidden)
    MLP_NTK_matrix = kernel_value_mlp(x_temp, L_nonlin_hidden)
    MLP_kernel = MLP_NTK_matrix[0, 1:]
    MLP_kernel = MLP_kernel / MLP_kernel.max()

    ResNet_NTK_regression = apply_ResNet_NTK(x_test, x_train, y_train, L_nonlin_hidden, alpha_resnet)
    ResNet_NTK_matrix = kernel_value_resnet(x_temp, L_nonlin_hidden, alpha_resnet)
    ResNet_kernel = ResNet_NTK_matrix[0, 1:]
    ResNet_kernel = ResNet_kernel / ResNet_kernel.max()

    plt.figure(1)
    plt.plot(x_gt, y_gt, 'silver')
    plt.plot(x_test_orig, MLP_NTK_regression)
    plt.plot(x_test_orig, ResNet_NTK_regression)
    plt.grid()
    plt.xlabel('x', fontsize=13)
    plt.ylabel('f(x)', fontsize=13)
    plt.legend(('Sampled signal','MLP', r'ResNet $\alpha$='+str(alpha_resnet)), fontsize=13)
    plt.plot(x_train_orig, y_train_orig, '*r')
    plt.savefig('NTK_regressions.png')

    plt.figure(2)
    plt.plot(x_test_orig, MLP_kernel)
    plt.plot(x_test_orig, ResNet_kernel)
    plt.grid()
    plt.xlabel('Angle(x,x\')', fontsize=13)
    plt.ylabel('Kernel', fontsize=13)
    plt.legend(('MLP', r'ResNet $\alpha$='+str(alpha_resnet)), fontsize=13)
    plt.savefig('NTK_shapes.png')


if __name__ == '__main__':
    main()