# Show image results
import matplotlib.pyplot as plt
def show_images(X,B_noise,x_FBF,z_FBF):
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))
    axs[0].imshow(X, cmap='gray')
    axs[0].set_title('original')
    axs[1].imshow(B_noise, cmap='gray')
    axs[1].set_title('noisy image')
    axs[2].imshow(x_FBF, cmap='gray')
    axs[2].set_title('nonaveraged denoised image')
    axs[3].imshow(z_FBF, cmap='gray')
    axs[3].set_title('averaged denoised image')
