import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Define paths to image directories
data_dir = r'C:\Users\navan\Downloads\archive\DRIVE'  # Replace with your dataset path
train_images_dir = os.path.join(data_dir, r'training\images')
train_masks_dir = os.path.join(data_dir, r'training\1st_manual')

# Function to load images and masks
def load_images(image_dir, mask_dir, image_size=(256, 256)):
    images = []
    masks = []
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)

        # Adjust mask file extension
        mask_file = img_file.replace('training', 'manual1').replace('.tif', '.gif')
        mask_path = os.path.join(mask_dir, mask_file)

        img = load_img(img_path, target_size=image_size)
        mask = load_img(mask_path, target_size=image_size, color_mode='grayscale')

        img = img_to_array(img) / 255.0
        mask = img_to_array(mask) / 255.0

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)

# Load data
x_train, y_train = load_images(train_images_dir, train_masks_dir)

# Define the Mask R-CNN model
def mask_rcnn_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    conv1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    output = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(conv2)  # Adjust output shape

    model = Model(inputs=inputs, outputs=output)
    return model

x = int(input("Enter Epoch value:"))

model = mask_rcnn_model()
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=x, batch_size=8, validation_split=0.1)

# Function to predict blood vessel mask from input image
def predict_blood_vessel_mask(model, image_path, image_size=(256, 256)):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict mask
    mask = model.predict(img_array)[0]

    return mask

# Accept multiple images for prediction until the user selects exit
while True:
    image_path = input("Enter the retinal image path (or type 'exit' to quit): ")
    if image_path.lower() == 'exit':
        break

    predicted_mask = predict_blood_vessel_mask(model, image_path)

    # Display the input image and predicted mask
    plt.figure(figsize=(12, 6))

    # Input image
    plt.subplot(1, 2, 1)
    input_img = load_img(image_path, target_size=(256, 256))
    plt.imshow(input_img)
    plt.title('Input Retinal Image')

    # Predicted mask
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Blood Vessel Mask')

    plt.tight_layout()
    plt.show()
