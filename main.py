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

import time
T1 = time.time()
## FBF_Penalty_Scheme
from inpainting.FBF_penalty import FBF
x_FBF, z_FBF, time_FBF , ISNR_av_FBF, ISNR_nonav_FBF = FBF(X,maxiter,k,x,a,b,B_noise,v11,v12,tv_switch,K,lambda_1)
T1 = time.time()-T1


## Show image results
from inpainting.show_images import show_images
show_images(X,B_noise,x_FBF,z_FBF)

## Plot results
from inpainting.plot_graphs import plot_graphs
plot_graphs(maxiter,ISNR_av_FBF,ISNR_nonav_FBF)



print('Elapsed time of FBF is ',time_FBF)
print('Elapsed time of FBF2 is ',T1)
plt.show()
# x_opt = np.zeros(X.shape)
# isnrav = np.zeros(maxiter)
# isnrnonav = np.zeros(maxiter)

# print("********** Penalty Splitting *************")
# tic = time.time()

# while k <= maxiter:
#     # Compute gamma and beta
#     gamma = (1 - 1e-1) / (k ** 0.75)        # \lambda_n
#     beta = k ** 0.75                        # \beta_n 

#     # Compute a and b
#     a += gamma * x                          # a=\sum\lambda_n*x_n
#     b += gamma                              # b=\sum \lamda_n
#     z = a / b                               # z_n = weighted average of x_n

#     # Compute isnrav and isnrnonav
#     isnrav[k-1] = 10 * np.log10((np.linalg.norm(X.ravel() - B_noise.ravel()) ** 2) /
#                                 (np.linalg.norm(X.ravel() - z.ravel()) ** 2))
#     isnrnonav[k-1] = 10 * np.log10((np.linalg.norm(X.ravel() - B_noise.ravel()) ** 2) /
#                                    (np.linalg.norm(X.ravel() - x.ravel()) ** 2))
    
#     # Compute Prox_gamma f
#     #p1 = x - gamma * L_trans(v11, v12) - gamma * beta * K * (x - B_noise)
#     #p1 = np.maximum(0, np.minimum(p1, 1))

#     # Compute Prox_gamma f
#     p1 = x - gamma * L_trans(v11, v12) - gamma * beta * K * (x - B_noise)
#     p1 = np.maximum(0, np.minimum(p1, 1))

#     # Compute Prox_gamma g*
#     G1, G2 = L(x)
#     p2_11 = v11 + gamma * G1
#     p2_12 = v12 + gamma * G2

#     if tv_switch == 'aniso':
#         p2_11 = p2_11 / np.maximum(1, np.abs(p2_11) / lambda_1)
#         p2_12 = p2_12 / np.maximum(1, np.abs(p2_12) / lambda_1)
#     elif tv_switch == 'iso':
#         Quotient = np.sqrt(p2_11 ** 2 + p2_12 ** 2)
#         p2_11 = p2_11 / np.maximum(1, Quotient / lambda_1)
#         p2_12 = p2_12 / np.maximum(1, Quotient / lambda_1)
#     else:
#         print(f'Unknown TV function "{tv_switch}".')

#     G1, G2 = L(p1 - x)
#     new_v11 = gamma * G1 + p2_11
#     new_v12 = gamma * G2 + p2_12
#     new_x = gamma * (L_trans(v11 - p2_11, v12 - p2_12)) + p1 + gamma * beta * K * (x - p1)

#     x = new_x
#     v11 = new_v11
#     v12 = new_v12

#     # Monitoring
#     x_opt = x
#     # run monitoring
#     k += 1

# toc = time.time() - tic

# f'Elapsed time is {toc} seconds.'

# #ISNR variable
# isnraveraged = isnrav;
# isnrnonaveraged = isnrnonav;

# # Shoe image results
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