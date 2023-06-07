import matplotlib.pyplot as plt

def check_after_noise(X,B_noise):
    '''
    use for check image after adding noise
    '''
    # Assuming X and B_noise are NumPy arrays
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Plot original image
    ax[0].imshow(X, vmin=0, vmax=1)
    ax[0].set_title('Original')

    # Plot noisy image
    ax[1].imshow(B_noise, vmin=0, vmax=1)
    ax[1].set_title('Noisy Image')

    plt.draw()
    print('continue computation')

    plt.show(block=False)
    # plt.show()

    return