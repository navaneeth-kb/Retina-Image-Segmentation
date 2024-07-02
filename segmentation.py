import os  # For handling file paths
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting images
import tensorflow as tf  # For building and training the model
from tensorflow.keras.preprocessing.image import load_img, img_to_array  # For loading and converting images

# Define paths to image directories
data_dir = r'C:\Users\navan\Downloads\archive\DRIVE'  # Replace with your dataset path
train_images_dir = os.path.join(data_dir, r'training\images')
train_masks_dir = os.path.join(data_dir, r'training\1st_manual')

def load_images(image_dir,mask_dir,image_size=(256,256),image_format='tif'):
    images=[]
    masks=[]
    for img_file in os.listdir(image_dir):
        img_path=os.path.join(image_dir,img_file)

        # Adjust mask file extension
        if image_format == 'tif':
            mask_path = os.path.join(mask_dir, img_file.replace('training', 'manual1').replace('.tif', '.gif'))
        else:
            mask_path = os.path.join(mask_dir, img_file.replace('training', 'manual1'))

        img=load_img(img_path,target_size=image_size)
        mask=load_img(mask_path,target_size=image_size,color_mode='grayscale')

        img=img_to_array(img)/255.0
        mask=img_to_array(mask)/255.0

        images.append(img)
        masks.append(mask)

    return np.array(images),np.array(masks)

#Load data
x_train,y_train=load_images(train_images_dir,train_masks_dir)

#Display an eg
plt.figure(figsize=(10,5))

#Eg image
plt.subplot(1,2,1)
plt.imshow(x_train[0])
plt.title('Image')

#Eg mask
plt.subplot(1,2,2)
plt.imshow(y_train[0])
plt.title('Mask')

plt.show()

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate  # Import layers
from tensorflow.keras.models import Model  # Import Model class

def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)  # Define the input layer

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    # Decoder
    u3 = UpSampling2D(size=(2, 2))(c2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(u3)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)

    outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(c3)  # Define the output layer

    model = Model(inputs=inputs, outputs=outputs)  # Create the model

    return model

# Create model
model = unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  # Compile the model

# Display the model summary
model.summary()

x=int (input("Enter Epoch value:"))

# Train the model
history = model.fit(x_train, y_train, epochs=x, batch_size=8, validation_split=0.1)

# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Save the model
model.save('unet_model.h5')

# Function to predict the mask for a new image
def predict_mask(model, image_path, image_size=(256, 256)):
    img = load_img(image_path, target_size=image_size)
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    pred_mask = model.predict(img)[0]
    return pred_mask

y=input("Enter the retinal image path:")

# Predict the mask
predicted_mask = predict_mask(model,y)

# Display the input image and the predicted mask
plt.figure(figsize=(10, 5))

# Input image
plt.subplot(1, 2, 1)
input_img = load_img(y, target_size=(256, 256))
plt.imshow(input_img)
plt.title('Input Image')

# Predicted mask
plt.subplot(1, 2, 2)
plt.imshow(predicted_mask, cmap='gray')
plt.title('Predicted Blood Vessel Mask')

plt.show()

'''
U-Net model: 
Let's break it down step-by-step:

1. Defining the Input:

def unet_model(input_size=(256, 256, 3)): - This line defines a function named unet_model that takes an optional argument input_size. This argument is a tuple representing the size of the images the model will process. By default, it's set to (256, 256, 3), which means images with a height of 256 pixels, width of 256 pixels, and 3 color channels (RGB).
inputs = Input(input_size) - This line creates the starting point for the data flow in the model. It's like a placeholder for the images you'll feed into the network.

2. The Encoder (Compression):

This part of the code analyzes the input image and extracts important features. Imagine it like summarizing a book chapter by chapter.
c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs) - This line applies a convolutional layer. Think of it as a filter that scans the image, looking for specific patterns. It uses 64 filters of size 3x3 pixels each, and the relu activation function adds a non-linearity to the process. The padding='same' ensures the output image has the same dimensions as the input. We'll call the resulting data c1.
c1 = Conv2D(64, 3, activation='relu', padding='same')(c1) - This line applies another convolutional layer, but this time it uses the output of the previous layer (c1) as input. This helps refine the feature extraction.
p1 = MaxPooling2D(pool_size=(2, 2))(c1) - This line applies a pooling operation. Imagine taking a 2x2 grid of pixels and keeping only the most important value (like the maximum). This reduces the image size while retaining key features, making it more efficient to process later. We'll call the result p1.
We repeat this process (c2, p2) with more filters to capture even more complex features, but also reduce the image size further.

3. The Decoder (Expansion):

This part takes the compressed features and expands them back into a full image, like rebuilding the chapters into a complete story, but potentially with more detail.
u3 = UpSampling2D(size=(2, 2))(c2) - This line upsamples the data (c2) back to its original size. Think of it as duplicating each pixel to create a bigger image. We call the result u3.
c3 = Conv2D(64, 3, activation='relu', padding='same')(u3) - This line applies another convolutional layer, but this time on the upsampled data. It helps refine the details and integrate information from the earlier layers. We call the result c3.
We repeat this process (c3) to further refine the image.

4. The Output Layer:

outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(c3) - This line applies a final convolutional layer with only one filter. The sigmoid activation function squishes the output values between 0 and 1, making it suitable for tasks like image segmentation (where each pixel represents a probability of belonging to a specific category). We call the final image outputs.

5. Creating the Model:

model = Model(inputs=inputs, outputs=outputs) - This line combines all the layers we defined into a single model object. It specifies the input point (inputs) and the output point (outputs) of the data flow.

6. Returning the Model:

return model - This line returns the created model, making it usable for training and prediction tasks.'''
