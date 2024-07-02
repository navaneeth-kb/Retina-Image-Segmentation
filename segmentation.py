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

'''
#Eg image
plt.subplot(1,2,1)
plt.imshow(x_train[0])
plt.title('Image')

#Eg mask
plt.subplot(1,2,2)
plt.imshow(y_train[0])
plt.title('Mask')

plt.show()
'''

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

'''
# Plot training history
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
'''

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
