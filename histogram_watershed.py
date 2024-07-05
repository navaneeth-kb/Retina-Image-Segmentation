import matplotlib.pyplot as plt
import numpy as np
import os
from skimage import color, exposure, filters, segmentation, measure, morphology, util
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

    # Apply the watershed algorithm
    # Generate markers as local maxima of the distance to the background
    distance = ndimage.distance_transform_edt(gray_eq)
    local_maxi = morphology.local_maxima(distance)
    markers = measure.label(local_maxi)

    # Perform watershed segmentation
    labels = segmentation.watershed(-distance, markers, mask=gray_eq)

    # Display the results
    plt.imshow(color.label2rgb(labels, image=img, bg_label=1))
    plt.title('Watershed Segmentation')
    plt.show()
