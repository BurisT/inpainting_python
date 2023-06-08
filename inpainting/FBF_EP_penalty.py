# Penalty Splitting with tv regularization for FWF-EP
# inf TV(x) s.t. Kx=b, x \in [0,1]^n, \Psi(x)=(1/2)*\|Kx-b\|^2

import time
from inpainting.discrete_gradient_operator import L,L_trans
import numpy as np

def FBF_EP(X,maxiter,B_noise,tv_switch,K,lambda_1): 

    # Define initial variables
    k = 1                               # iteration counting
    x = B_noise.copy()                  # starting point     # By using B_noise.copy() instead of B_noise directly, any changes made to the copy will not affect the original array. This can be useful when working with large arrays or when you want to keep the original data intact.
    v11 = np.zeros(X.shape)             # y1=x0
    v12 = np.zeros(X.shape)            
    q11=v11                             # need for FBF_EP
    q12=q11                             # need for FBF_EP
    y0=x                                # need for FBF_EP
    a = np.zeros(x.shape)               # initialization for the weighted average z=a/b
    b = 0     

    x_opt = np.zeros(X.shape)
    isnrav = np.zeros(maxiter)
    isnrnonav = np.zeros(maxiter)
    
    ######reused parameters
    R1 = L_trans(q11,q12);
    [G01,G02] = L(y0);

    print("********** Penalty Splitting for FWF-EP *************")
    tic2 = time.time()

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
        y1 = x - gamma * (R1) - gamma * beta * np.multiply(K,(y0 - B_noise)) #(K.*(y0 - B_noise))
        y1 = np.clip(y1,0,1)#np.maximum(0, np.minimum(y1, 1))


        # Compute Prox_gamma g*
        q2_11 = v11 + gamma * G01
        q2_12 = v12 + gamma * G02

        if tv_switch == 'aniso':
            q2_11 = q2_11 / np.maximum(1, np.abs(q2_11) / lambda_1)
            q2_12 = q2_12 / np.maximum(1, np.abs(q2_12) / lambda_1)
        elif tv_switch == 'iso':
            Quotient = np.sqrt(q2_11 ** 2 + q2_12 ** 2)
            q2_11 = q2_11 / np.maximum(1, Quotient / lambda_1)
            q2_12 = q2_12 / np.maximum(1, Quotient / lambda_1)
        else:
            print(f'Unknown TV function "{tv_switch}".')

        R2=L_trans(q2_11,q2_12)
        new_x = gamma*beta*( np.multiply(K,(y0 - y1))) + gamma*(R1-R2) + y1;

        G11, G12 = L(y1)
        G1 = G11-G01
        G2 = G12-G02
        new_v11 = gamma * G1 + q2_11
        new_v12 = gamma * G2 + q2_12
        
        #updating parameters
        x = new_x
        q11=q2_11
        q12=q2_12
        v11 = new_v11
        v12 = new_v12
        R1=R2                  #update L_trans(q11,q12)=L_trans(q2_11,q2_12)
        G01 = G11              #update L(y0)=L(y1)
        G02 = G12

        # Monitoring
        x_opt = x
        # run monitoring
        k += 1

    toc2 = time.time() - tic2

    FBF_EP_recovered = x_opt;
    avg_FBF_EP_recovered = z;

    f'Elapsed time is {toc2} seconds.'

    return x_opt, z, toc2, isnrav, isnrnonav
