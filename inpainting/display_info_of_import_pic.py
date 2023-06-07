import cv2
import imageio
import matplotlib.pyplot as plt

def Displayinf(img):
    ## Display image using matplotlib
     plt.imshow(img)
     plt.show()

     ## Check size of image
     print("Size of image:", img.size)

     ## Check dimension of image
     print("Dimension of image:", img.shape)

     ## Get shape of image
     height, width = img.shape[:2]
     num_channels = img.shape[2] if len(img.shape) == 3 else 1

     ## Print shape of image
     print("Height of image:", height)
     print("Width of image:", width)
     print("Number of channels in image:", num_channels)