import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

data_dir = r'C:\Users\navan\Downloads\archive\DRIVE'  # Replace with your dataset path
train_images_dir = os.path.join(data_dir, r'training\images')
train_masks_dir = os.path.join(data_dir, r'training\1st_manual')

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

def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(64, 3, activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(128, 3, activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(128, 3, activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(256, 3, activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(256, 3, activation='relu', padding='same')(c3)
    c3 = BatchNormalization()(c3)

    # Decoder
    u4 = UpSampling2D(size=(2, 2))(c3)
    u4 = Conv2D(128, 3, activation='relu', padding='same')(u4)
    u4 = BatchNormalization()(u4)
    u4 = Conv2D(128, 3, activation='relu', padding='same')(u4)
    u4 = BatchNormalization()(u4)

    u5 = UpSampling2D(size=(2, 2))(u4)
    u5 = Conv2D(64, 3, activation='relu', padding='same')(u5)
    u5 = BatchNormalization()(u5)
    u5 = Conv2D(64, 3, activation='relu', padding='same')(u5)
    u5 = BatchNormalization()(u5)

    outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(u5)

    model = Model(inputs=inputs, outputs=outputs)

    return model

def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def f1_score(y_true, y_pred):
    def recall_m(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        possible_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_true * y_pred, 0, 1)))
        predicted_positives = tf.keras.backend.sum(tf.keras.backend.round(tf.keras.backend.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
        return precision

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+tf.keras.backend.epsilon()))

x = int(input("Enter Epoch value:"))

model = unet_model()
model.compile(optimizer=Adam(),
              loss=dice_loss,
              metrics=['accuracy',
                      tf.keras.metrics.MeanIoU(num_classes=2),
                      tf.keras.metrics.AUC(name='auc'),
                      f1_score])
history = model.fit(x_train, y_train, epochs=x, batch_size=8, validation_split=0.1)

def predict_blood_vessel_mask(model, image_path, image_size=(256, 256)):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict mask
    mask = model.predict(img_array)[0]

    return mask

while True:
    image_path = input("Enter the retinal image path (or type 'exit' to quit): ")
    if image_path.lower() == 'exit':
        break

    predicted_mask = predict_blood_vessel_mask(model, image_path)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    input_img = load_img(image_path, target_size=(256, 256))
    plt.imshow(input_img)
    plt.title('Input Retinal Image')

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_mask, cmap='gray')
    plt.title('Predicted Blood Vessel Mask')

    plt.tight_layout()
    plt.show()
