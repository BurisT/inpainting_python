import time
from inpainting.discrete_gradient_operator import L,L_trans
import numpy as np

def FBF(X,maxiter,k,x,a,b,B_noise,v11,v12,tv_switch,K,lambda_1):
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
        #p1 = x - gamma * L_trans(v11, v12) - gamma * beta * K * (x - B_noise)
        #p1 = np.maximum(0, np.minimum(p1, 1))

        # Compute Prox_gamma f
        p1 = x - gamma * L_trans(v11, v12) - gamma * beta * K * (x - B_noise)
        p1 = np.maximum(0, np.minimum(p1, 1))

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

    #ISNR variable
    # isnraveraged = isnrav;
    # isnrnonaveraged = isnrnonav;

    return x_opt, z, tic, isnrav, isnrnonav

    # # Show image results
    # import matplotlib.pyplot as plt

    # fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    # axs[0].imshow(X, cmap='gray')
    # axs[0].set_title('original')
    # axs[1].imshow(B_noise, cmap='gray')
    # axs[1].set_title('noisy image')
    # axs[2].imshow(x_opt, cmap='gray')
    # axs[2].set_title('nonaveraged denoised image')
    # axs[3].imshow(z, cmap='gray')
    # axs[3].set_title('averaged denoised image')

    # # Plot results
    # plt.figure()
    # plt.plot(range(1, maxiter + 1), isnraveraged)
    # plt.plot(range(1, maxiter + 1), isnrnonaveraged, '--')
    # plt.xlabel('Iterations')
    # plt.ylabel('ISNR')
    # plt.legend(['isnr averaged', 'isnr nonaveraged'])
    # plt.show()