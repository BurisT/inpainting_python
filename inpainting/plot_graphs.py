## Plot results
import matplotlib.pyplot as plt
import numpy as np
def plot_graphs(maxiter,ISNR_av_FBF,ISNR_nonav_FBF,ISNR_av_FBF_EP,ISNR_nonav_FBF_EP):

    plt.figure()

    plt.plot(np.arange(1, maxiter + 1), ISNR_av_FBF, linewidth=2)
    plt.plot(np.arange(1, maxiter + 1), ISNR_nonav_FBF, '--', linewidth=2)
    plt.plot(np.arange(1, maxiter + 1), ISNR_av_FBF_EP, ':', linewidth=2, markersize=10, markerfacecolor=[0.5, 0.5, 0.5])
    plt.plot(np.arange(1, maxiter + 1), ISNR_nonav_FBF_EP, '-.', linewidth=2)

    plt.xlabel('Iterations')
    plt.ylabel('ISNR')

    plt.legend(['ISNR averaged', 'ISNR nonaveraged', 'ISNR averaged EP', 'ISNR nonaveraged EP'], loc='lower right')

    plt.show()

    # plt.figure()
    # plt.plot(range(1, maxiter + 1), ISNR_av_FBF)
    # plt.plot(range(1, maxiter + 1), ISNR_nonav_FBF, '--')
    # plt.xlabel('Iterations')
    # plt.ylabel('ISNR')
    # plt.legend(['ISNR averaged', 'ISNR nonaveraged'])