## Plot results
import matplotlib.pyplot as plt
def plot_graphs(maxiter,ISNR_av_FBF,ISNR_nonav_FBF):
    plt.figure()
    plt.plot(range(1, maxiter + 1), ISNR_av_FBF)
    plt.plot(range(1, maxiter + 1), ISNR_nonav_FBF, '--')
    plt.xlabel('Iterations')
    plt.ylabel('ISNR')
    plt.legend(['ISNR averaged', 'ISNR nonaveraged'])