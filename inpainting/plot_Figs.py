import matplotlib.pyplot as plt

def plot_Figs(X,B_noise,x_FBF,z_FBF,x_FBF_EP,z_FBF_EP):

    plt.figure()

    plt.subplot(2, 4, 1)
    plt.imshow(X, cmap='gray', vmin=0, vmax=1)
    plt.title('original')

    plt.subplot(2, 4, 2)
    plt.imshow(B_noise, cmap='gray', vmin=0, vmax=1)
    plt.title('noisy image')

    plt.subplot(2, 4, 3)
    plt.imshow(x_FBF, cmap='gray', vmin=0, vmax=1)
    plt.title('nonaveraged denoised image (FBF)')

    plt.subplot(2, 4, 4)
    plt.imshow(z_FBF, cmap='gray', vmin=0, vmax=1)
    plt.title('averaged denoised image (FBF)')

    plt.subplot(2, 4, 5)
    plt.imshow(X, cmap='gray', vmin=0, vmax=1)
    plt.title('original')

    plt.subplot(2, 4, 6)
    plt.imshow(B_noise, cmap='gray', vmin=0, vmax=1)
    plt.title('noisy image')

    plt.subplot(2, 4, 7)
    plt.imshow(x_FBF_EP, cmap='gray', vmin=0, vmax=1)
    plt.title('nonaveraged denoised image (FBF-EP)')

    plt.subplot(2, 4, 8)
    plt.imshow(z_FBF_EP, cmap='gray', vmin=0, vmax=1)
    plt.title('averaged denoised image (FBF-EP)')
