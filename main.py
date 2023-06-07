import numpy as np
import imageio.v2 as imageio
import cv2
import matplotlib.pyplot as plt
from inpainting.discrete_gradient_operator import L,L_trans


## Read image file
img = imageio.imread('Photos/Pisa_tower_2.JPG')

## Resize image 
img = cv2.resize(img, (256, 256))

# Store image array in a single variable
image_data = img

## Print shape of image
print("Shape of image:", image_data.shape)

X = image_data/255

## Add noise into the picture
from inpainting.make_noise import noiseimage
noise, B_noise = noiseimage(X)
K = noise

## Check image after adding noise
from inpainting.cheack_after_add_noise import check_after_noise
check_after_noise(X,B_noise)

## setting parameters
import math

K_norm=1
L_norm=math.sqrt(8)

tv_switch='iso'            # isotropic TV, change otherwise to 'aniso' for anisotopic TV
lambda_1=1                 # regularization parameter, not needed, therefore set to 1
RMSE_eps=1e-3              # one possible stopping criterion



# Define initial variables
k = 1                               # iteration counting
x = B_noise.copy()                  # starting point     # By using B_noise.copy() instead of B_noise directly, any changes made to the copy will not affect the original array. This can be useful when working with large arrays or when you want to keep the original data intact.
v11 = np.zeros(X.shape)             # y1=x0
v12 = np.zeros(X.shape)            
a = np.zeros(x.shape)               # initialization for the weighted average z=a/b
b = 0                               # initialization for the weighted average 
maxiter = 1000                      # count of iterations
#dev = np.zeros((maxiter, 2))        # for monitoring purposes


## FBF_Penalty_Scheme
from inpainting.FBF_penalty import FBF
x_FBF, z_FBF, time_FBF , ISNR_av_FBF, ISNR_nonav_FBF = FBF(X,maxiter,k,x,a,b,B_noise,v11,v12,tv_switch,K,lambda_1)


X.astype(float)
## Show image results
from inpainting.show_images import show_images
show_images(X,B_noise,x_FBF,z_FBF)

## Plot results
from inpainting.plot_graphs import plot_graphs
plot_graphs(maxiter,ISNR_av_FBF,ISNR_nonav_FBF)



print('Elapsed time of FBF is ',time_FBF)

# ## average time
# import timeit
# print(timeit.timeit(stmt='FBF(X,maxiter,k,x,a,b,B_noise,v11,v12,tv_switch,K,lambda_1)',setup='from __main__ import FBF,X,maxiter,k,x,a,b,B_noise,v11,v12,tv_switch,K,lambda_1', number=3))

plt.show()