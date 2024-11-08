import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt

data_dir = r'C:\Users\navan\Downloads\archive\DRIVE'  # Replace with your dataset path
train_images_dir = os.path.join(data_dir, r'training\images')
train_masks_dir = os.path.join(data_dir, r'training\1st_manual')


def nn_unet_model(input_size=(256, 256, 3), num_filters=64, num_classes=1, dropout_rate=0.3):
    inputs = Input(input_size)

    def conv_block(x, num_filters):
        x = Conv2D(num_filters, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)  # Add dropout for regularization
        x = Conv2D(num_filters, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        return x

    def encoder_block(x, num_filters):
        x = conv_block(x, num_filters)
        p = MaxPooling2D(pool_size=(2, 2))(x)
        return x, p

    def decoder_block(x, skip, num_filters):
        x = UpSampling2D(size=(2, 2))(x)
        x = Conv2D(num_filters, 2, activation='relu', padding='same')(x)
        x = concatenate([x, skip])
        x = conv_block(x, num_filters)
        return x

    # Encoder: Deeper layers
    s1, p1 = encoder_block(inputs, num_filters)
    s2, p2 = encoder_block(p1, num_filters * 2)
    s3, p3 = encoder_block(p2, num_filters * 4)
    s4, p4 = encoder_block(p3, num_filters * 8)

    # Bottleneck
    b = conv_block(p4, num_filters * 16)

    # Decoder: Deeper layers
    d4 = decoder_block(b, s4, num_filters * 8)
    d3 = decoder_block(d4, s3, num_filters * 4)
    d2 = decoder_block(d3, s2, num_filters * 2)
    d1 = decoder_block(d2, s1, num_filters)

    # Output layer
    outputs = Conv2D(num_classes, 1, activation='sigmoid')(d1)

    model = Model(inputs, outputs)

    return model

def combined_dice_bce_loss(y_true, y_pred, smooth=1e-6):
    dice_loss = dice_coef_loss(y_true, y_pred)
    bce_loss = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    return dice_loss + bce_loss

def dice_coef_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)


def augment_images(images, masks):
    data_gen_args = dict(horizontal_flip=True, vertical_flip=True, rotation_range=90, zoom_range=0.2)
    image_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)
    mask_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(**data_gen_args)

    seed = np.random.randint(0, 10000)
    image_gen = image_data_gen.flow(images, seed=seed, batch_size=8)
    mask_gen = mask_data_gen.flow(masks, seed=seed, batch_size=8)

    return image_gen, mask_gen


def load_images(image_dir, mask_dir, image_size=(256, 256)):
    images = []
    masks = []
    for img_file in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_file)
        mask_file = img_file.replace('training', 'manual1').replace('.tif', '.gif')
        mask_path = os.path.join(mask_dir, mask_file)

        img = load_img(img_path, target_size=image_size)
        mask = load_img(mask_path, target_size=image_size, color_mode='grayscale')

        img = img_to_array(img) / 255.0
        mask = img_to_array(mask) / 255.0

        images.append(img)
        masks.append(mask)

    return np.array(images), np.array(masks)


x_train, y_train = load_images(train_images_dir, train_masks_dir)
x_train_aug, y_train_aug = augment_images(x_train, y_train)


model = nn_unet_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss=combined_dice_bce_loss,
              metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])



x = int(input("Enter Epoch value:"))  # High epoch value for training (e.g., 350)
history = model.fit(x_train_aug[0], y_train_aug[0], epochs=x, batch_size=8, validation_split=0.1)


def predict_blood_vessel_mask(model, image_path, image_size=(256, 256)):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

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
