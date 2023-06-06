import numpy as np

def L(X):
    """
    Discrete Linear Operator in TV-Norm L()
    Note that in Python, we use the NumPy library for array operations,
    which is equivalent to MATLAB's matrix operations.
    The concatenate function in NumPy is used to combine
    arrays along a specified axis.

    Also note that the indexing of arrays starts at 0 in Python, whereas
    it starts at 1 in MATLAB.
    """
    # Get dimensions of X
    m, n, r = X.shape
    
    # Compute X1 and X2
    X1 = np.concatenate((X[1:m,:,:] - X[0:m-1, :,:], np.zeros((1,n,r))), axis=0)
    X2 = np.concatenate((X[:,1:n,:] - X[:, 0:n-1, :], np.zeros((m,1,r))), axis=1)
    
    return X1, X2


import numpy as np

def L_trans(P1, P2):
    m, n, r = P1.shape

    #P = -(np.concatenate((P1[0:1, :, :], P1[1:m-1, :, :] - P1[0:m-2, :, :], -P1[m-1:m, :, :]), axis=0))
    #P -= np.concatenate((P2[:, 0:1, :], P2[:, 1:n-1, :] - P2[:, 0:n-2, :], -P2[:, n-1:n, :]), axis=1)

    P1_temp = np.vstack((
        P1[0:1, :, :],
         np.subtract(
            P1[ 1:m-1,:, :] , P1[0:m-2,: , :]
        ), -P1[m-2:m-1,: , :]
    ))

    P2_temp = np.concatenate((
        P2[:, 0:1, :],
        P2[:, 1:n-1, :] - P2[:, 0:n-2, :],
        -P2[:, n-2:n-1, :]
    ), axis=1)
    
    P = -P1_temp - P2_temp

    return P
