import time
from inpainting.discrete_gradient_operator import L,L_trans
import numpy as np

def FBF(X,maxiter,B_noise,tv_switch,K,lambda_1):
    # Define initial variables
    k = 1                               # iteration counting
    x = B_noise.copy()                  # starting point     # By using B_noise.copy() instead of B_noise directly, any changes made to the copy will not affect the original array. This can be useful when working with large arrays or when you want to keep the original data intact.
    v11 = np.zeros(X.shape)             # y1=x0
    v12 = np.zeros(X.shape)            
    # q11=v11                             # need for FBF_EP
    # q12=q11                             # need for FBF_EP
    # y0=x                                # need for FBF_EP
    a = np.zeros(x.shape)               # initialization for the weighted average z=a/b
    b = 0                               # initialization for the weighted average 


    x_opt = np.zeros(X.shape)
    isnrav = np.zeros(maxiter)
    isnrnonav = np.zeros(maxiter)

    print("********** FBF Penalty Splitting *************")
    tic = time.time()

    while k <= maxiter:
        # Compute gamma and beta
        gamma = (1 - 1e-1) / (k ** 0.75)        # \lambda_n
        beta = k ** 0.75                        # \beta_n 

        # Compute a and b
        a += gamma * x                          # a=\sum\lambda_n*x_n
        b += gamma                              # b=\sum \lamda_n
        z = a / b                               # z_n = weighted average of x_n

        # Compute isnrav and isnrnonav
        isnrav[k-1] = 10 * np.log10((np.linalg.norm(X.ravel() - B_noise.ravel()) ** 2) /
                                    (np.linalg.norm(X.ravel() - z.ravel()) ** 2))
        isnrnonav[k-1] = 10 * np.log10((np.linalg.norm(X.ravel() - B_noise.ravel()) ** 2) /
                                    (np.linalg.norm(X.ravel() - x.ravel()) ** 2))

        # Compute Prox_gamma f
        p1 = x - gamma * L_trans(v11, v12) - gamma * beta * K * (x - B_noise)
        p1 = np.clip(p1,0,1)  #np.maximum(0, np.minimum(p1, 1))

        # Compute Prox_gamma g*
        G1, G2 = L(x)
        p2_11 = v11 + gamma * G1
        p2_12 = v12 + gamma * G2

        if tv_switch == 'aniso':
            p2_11 = p2_11 / np.maximum(1, np.abs(p2_11) / lambda_1)
            p2_12 = p2_12 / np.maximum(1, np.abs(p2_12) / lambda_1)
        elif tv_switch == 'iso':
            Quotient = np.sqrt(p2_11 ** 2 + p2_12 ** 2)
            p2_11 = p2_11 / np.maximum(1, Quotient / lambda_1)
            p2_12 = p2_12 / np.maximum(1, Quotient / lambda_1)
        else:
            print(f'Unknown TV function "{tv_switch}".')

        G1, G2 = L(p1 - x)
        new_v11 = gamma * G1 + p2_11
        new_v12 = gamma * G2 + p2_12
        new_x = gamma * (L_trans(v11 - p2_11, v12 - p2_12)) + p1 + gamma * beta * K * (x - p1)

        x = new_x
        v11 = new_v11
        v12 = new_v12

        # Monitoring
        x_opt = x
        # run monitoring
        k += 1

    toc = time.time() - tic

    return x_opt, z, toc, isnrav, isnrnonav
