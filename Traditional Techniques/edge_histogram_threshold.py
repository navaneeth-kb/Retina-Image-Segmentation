import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import color,exposure, filters,morphology
from scipy import ndimage

data_dir = r'C:\Users\navan\Downloads\archive\DRIVE'
train_images_dir = os.path.join(data_dir, r'training\images')

for img_file in os.listdir(train_images_dir):
    img_path = os.path.join(train_images_dir, img_file)
    img = plt.imread(img_path)
    plt.imshow(img)
    #plt.show()

    gray = color.rgb2gray(img)

    # Apply histogram equalization
    gray_eq = exposure.equalize_hist(gray)

    # Apply Gaussian blurring
    gray_blurred = filters.gaussian(gray_eq, sigma= 1.5)

    kernel_laplace = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    out_l = ndimage.convolve(gray_blurred, kernel_laplace)

    # Apply thresholding
    threshold_value = filters.threshold_otsu(out_l)
    binary_out_l = np.zeros_like(out_l)

    # Iterate through each pixel in the 'out_l' image
    for i in range(out_l.shape[0]):
        for j in range(out_l.shape[1]):
            if out_l[i, j] > threshold_value:
                binary_out_l[i, j] = 1
            else:
                binary_out_l[i, j] = 0

    #binary_out_l = morphology.binary_closing(binary_out_l, morphology.disk(0))

    plt.imshow(binary_out_l, cmap='gray')
    plt.show()
