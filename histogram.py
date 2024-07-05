import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import color, exposure, filters
from scipy import ndimage

data_dir = r'C:\Users\navan\Downloads\archive\DRIVE'
train_images_dir = os.path.join(data_dir, r'training\images')

for img_file in os.listdir(train_images_dir):
    img_path = os.path.join(train_images_dir, img_file)
    img = plt.imread(img_path)
    plt.imshow(img)
    plt.title('Original Image')
    plt.show()

    gray = color.rgb2gray(img)

    # Apply histogram equalization
    gray_eq = exposure.equalize_hist(gray)

    plt.imshow(gray_eq, cmap='gray')
    plt.title('Histogram Equalized Image')
    plt.show()





