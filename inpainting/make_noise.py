import numpy as np
from skimage.util import random_noise

def noiseimage(X):
    X_const = 0.5 * np.ones(X.shape)

    X_const_noise =  random_noise(X_const, mode='s&p', amount=0.80)

    # ## Display the noisy image
    # plt.imshow(X_const_noise)
    # plt.show()

    K = np.double(X_const_noise == 0.5)

    # Assuming K is a NumPy array
    m, n, o = K.shape
    for k in range(o):
        K[:, :, k] = K[:, :, 0]   
    '''The resulting K array will have the same dimensionsas 
        the original K array, but with all slices in the third dimension set 
        to be identical to the first slice.
    '''
   
    B_noise = K * X

    ''' In Python, the * operator performs element-wise multiplication between two 
        NumPy arrays of the same shape. This is equivalent to the .* operator 
        in MATLAB.

        The resulting B_noise array will have the same shape as the input arrays X 
        and K, and will contain the element-wise product of X and K.Assuming X and 
        K are NumPy arrays of the same shape
    '''
    return K, B_noise