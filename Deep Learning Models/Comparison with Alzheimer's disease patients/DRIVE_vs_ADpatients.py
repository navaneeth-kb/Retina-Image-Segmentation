import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

data_dir = r'C:\Users\navan\Downloads\archive\DRIVE'
train_images_dir = os.path.join(data_dir, r'training\images')
train_masks_dir = os.path.join(data_dir, r'training\1st_manual')

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


def unet_model(input_size=(256, 256, 3), filters=64):
    inputs = Input(input_size)

    # Encoder
    c1 = Conv2D(filters, 3, activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(filters, 3, activation='relu', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = Conv2D(filters * 2, 3, activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(filters * 2, 3, activation='relu', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    # Bottleneck
    c3 = Conv2D(filters * 4, 3, activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(filters * 4, 3, activation='relu', padding='same')(c3)
    c3 = BatchNormalization()(c3)

    # Decoder
    u4 = UpSampling2D(size=(2, 2))(c3)
    u4 = Conv2D(filters * 2, 3, activation='relu', padding='same')(u4)
    u4 = BatchNormalization()(u4)
    u4 = Conv2D(filters * 2, 3, activation='relu', padding='same')(u4)
    u4 = BatchNormalization()(u4)

    u5 = UpSampling2D(size=(2, 2))(u4)
    u5 = Conv2D(filters, 3, activation='relu', padding='same')(u5)
    u5 = BatchNormalization()(u5)
    u5 = Conv2D(filters, 3, activation='relu', padding='same')(u5)
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
    return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))


x = int(input("Enter Epoch value:"))

print("\nTeacher modal\n")

teacher_model1 = unet_model(filters=64)
teacher_model2 = unet_model(filters=64)

teacher_model1.compile(optimizer=Adam(), loss=dice_loss,
                       metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), tf.keras.metrics.AUC(name='auc'),
                                f1_score])
teacher_model2.compile(optimizer=Adam(), loss=dice_loss,
                       metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), tf.keras.metrics.AUC(name='auc'),
                                f1_score])


teacher_model1.fit(x_train, y_train, epochs=x, batch_size=8, validation_split=0.1)
teacher_model2.fit(x_train, y_train, epochs=x, batch_size=8, validation_split=0.1)

teacher_preds1 = teacher_model1.predict(x_train)
teacher_preds2 = teacher_model2.predict(x_train)


combined_teacher_preds = (teacher_preds1 + teacher_preds2) / 2

from tensorflow.keras.layers import Input, DepthwiseConv2D, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

from tensorflow.keras import layers, models

from tensorflow.keras import layers, models, Input

#for student modal
def new_unet(input_size=(256, 256, 1), num_classes=1, patch_size=None, depth=4, initial_filters=32):
    input_layer = Input(shape=(256, 256, 3))

    patch_height, patch_width = patch_size if patch_size else (input_size[0] // 8, input_size[1] // 8)
    filters = initial_filters

    # Encoder (Downsampling)
    encoder_blocks = []
    for i in range(depth):
        # First layer for encoder uses input_layer or previous encoder block
        c = layers.Conv2D(filters, 3, activation='relu', padding='same')(input_layer if i == 0 else encoder_blocks[-1])
        c = layers.BatchNormalization()(c)
        c = layers.Conv2D(filters, 3, activation='relu', padding='same')(c)
        c = layers.BatchNormalization()(c)
        encoder_blocks.append(c)

        # Max pooling or downsampling
        if i < depth - 1:
            encoder_blocks[-1] = layers.MaxPooling2D(pool_size=(2, 2))(encoder_blocks[-1])
            encoder_blocks[-1] = layers.Dropout(0.2)(encoder_blocks[-1])

        # Adapt the filter size at each layer
        filters *= 2

    # Bottleneck (Adaptive)
    bottleneck = layers.Conv2D(filters, 3, activation='relu', padding='same')(encoder_blocks[-1])
    bottleneck = layers.BatchNormalization()(bottleneck)
    bottleneck = layers.Conv2D(filters, 3, activation='relu', padding='same')(bottleneck)
    bottleneck = layers.BatchNormalization()(bottleneck)
    bottleneck = layers.Dropout(0.2)(bottleneck)

    # Decoder (Upsampling)
    filters //= 2
    decoder_blocks = []
    for i in range(depth - 1):
        # Adaptive upsampling: Uses either Conv2DTranspose or UpSampling2D
        u = layers.UpSampling2D(size=(2, 2))(bottleneck if i == 0 else decoder_blocks[-1])
        u = layers.Conv2D(filters, 3, activation='relu', padding='same')(u)
        u = layers.BatchNormalization()(u)
        u = layers.Conv2D(filters, 3, activation='relu', padding='same')(u)
        u = layers.BatchNormalization()(u)
        decoder_blocks.append(u)

        if i < depth - 2:
            decoder_blocks[-1] = layers.Dropout(0.2)(decoder_blocks[-1])

        filters //= 2

    # Output Layer (Adaptive based on num_classes)
    outputs = layers.Conv2D(num_classes, 1, activation='sigmoid', padding='same')(decoder_blocks[-1])

    # Create Model
    model = models.Model(inputs=input_layer, outputs=outputs)

    # Dynamic adjustment of model summary
    model.summary()

    return model


print("\nStudent modals\n")

student_models = []
for i in range(3):
    student_model = new_unet()
    student_model.compile(optimizer=Adam(), loss=dice_loss,
                          metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2),
                                   tf.keras.metrics.AUC(name='auc'),
                                   f1_score])
    student_models.append(student_model)

print("\nStudent modals\n")

# Knowledge distillation- Train students with both ground truth and teacher's predictions
for student_model in student_models:

    # Loss is combined: ground truth + combined teacher predictions
    student_model.fit(x_train, [y_train, combined_teacher_preds], epochs=250, batch_size=8, validation_split=0.1)


def evaluate_models(teacher_model1, teacher_model2, student_models, x_data, y_data):
    avg_student_auc = 0
    avg_student_mean_iou = 0
    avg_student_accuracy = 0
    avg_student_f1_score = 0

    print("\nEvaluating Teacher Model 1:")
    teacher_results1 = teacher_model1.evaluate(x_data, y_data, verbose=0)
    teacher1_accuracy = teacher_results1[1]
    teacher1_mean_iou = teacher_results1[2]
    teacher1_auc = teacher_results1[3]
    teacher1_f1 = teacher_results1[4]

    print(f"Teacher Model 1 Accuracy: {teacher1_accuracy:.4f}")
    print(f"Teacher Model 1 Mean IoU: {teacher1_mean_iou:.4f}")
    print(f"Teacher Model 1 AUC: {teacher1_auc:.4f}")
    print(f"Teacher Model 1 F1 Score: {teacher1_f1:.4f}")

    print("\nEvaluating Teacher Model 2:")
    teacher_results2 = teacher_model2.evaluate(x_data, y_data, verbose=0)
    teacher2_accuracy = teacher_results2[1]
    teacher2_mean_iou = teacher_results2[2]
    teacher2_auc = teacher_results2[3]
    teacher2_f1 = teacher_results2[4]

    print(f"Teacher Model 2 Accuracy: {teacher2_accuracy:.4f}")
    print(f"Teacher Model 2 Mean IoU: {teacher2_mean_iou:.4f}")
    print(f"Teacher Model 2 AUC: {teacher2_auc:.4f}")
    print(f"Teacher Model 2 F1 Score: {teacher2_f1:.4f}")


    avg_teacher_accuracy = (teacher1_accuracy + teacher2_accuracy) / 2
    avg_teacher_mean_iou = (teacher1_mean_iou + teacher2_mean_iou) / 2
    avg_teacher_auc = (teacher1_auc + teacher2_auc) / 2
    avg_teacher_f1 = (teacher1_f1 + teacher2_f1) / 2

    print("\nAverage Teacher Model Performance:")
    print(f"Average Teacher Accuracy: {avg_teacher_accuracy:.4f}")
    print(f"Average Teacher Mean IoU: {avg_teacher_mean_iou:.4f}")
    print(f"Average Teacher AUC: {avg_teacher_auc:.4f}")
    print(f"Average Teacher F1 Score: {avg_teacher_f1:.4f}")

    print("\nEvaluating Student Models:")
    for i, student_model in enumerate(student_models):
        results = student_model.evaluate(x_data, y_data, verbose=0)

        accuracy = results[1]
        mean_iou = results[2]
        auc = results[3]
        f1 = results[4]

        print(
            f"Student Model {i + 1} - Accuracy: {accuracy:.4f}, Mean IoU: {mean_iou:.4f}, AUC: {auc:.4f}, F1 Score: {f1:.4f}")

        avg_student_accuracy += accuracy
        avg_student_mean_iou += mean_iou
        avg_student_auc += auc
        avg_student_f1_score += f1

    num_students = len(student_models)
    avg_student_accuracy /= num_students
    avg_student_mean_iou /= num_students
    avg_student_auc /= num_students
    avg_student_f1_score /= num_students

    print("\nAverage Student Model Performance:")
    print(f"Average Student Accuracy: {avg_student_accuracy:.4f}")
    print(f"Average Student Mean IoU: {avg_student_mean_iou:.4f}")
    print(f"Average Student AUC: {avg_student_auc:.4f}")
    print(f"Average Student F1 Score: {avg_student_f1_score:.4f}")


    overall_avg_accuracy = (avg_teacher_accuracy + avg_student_accuracy) / (num_students + 1)
    overall_avg_mean_iou = (avg_teacher_mean_iou + avg_student_mean_iou) / (num_students + 1)
    overall_avg_auc = (avg_teacher_auc + avg_student_auc) / (num_students + 1)
    overall_avg_f1_score = (avg_teacher_f1 + avg_student_f1_score) / (num_students + 1)

    print("\nOverall Average Performance (Teachers + Students):")
    print(f"Overall Average Accuracy: {overall_avg_accuracy:.4f}")
    print(f"Overall Average Mean IoU: {overall_avg_mean_iou:.4f}")
    print(f"Overall Average AUC: {overall_avg_auc:.4f}")
    print(f"Overall Average F1 Score: {overall_avg_f1_score:.4f}")


evaluate_models(teacher_model1, teacher_model2, student_models, x_train, y_train)

def predict_with_student_models(student_models, image_path, image_size=(256, 256)):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = []
    for student_model in student_models:
        predictions.append(student_model.predict(img_array)[0])

    final_mask = np.mean(predictions, axis=0)

    return final_mask

mask=input(("Do you want the predict the mask for retinal images? (yes or no)"))
if (mask=='yes'):
    while True:
        image_path = input("Enter the retinal image path (or type 'exit' to quit): ")
        if image_path.lower() == 'exit':
            break

        predicted_mask = predict_with_student_models(student_models, image_path)

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


import cv2

def fractal_dimension(image):

    if len(image.shape) == 3:  # Convert to grayscale if it's RGB
        image = color.rgb2gray(image)

    # Convert to binary mask
    binary_image = img_as_ubyte(image) > 128

    # Perform box-counting
    def box_count(binary_image, box_size):
        count = 0
        h, w = binary_image.shape
        for i in range(0, h, box_size):
            for j in range(0, w, box_size):
                if np.any(binary_image[i:i + box_size, j:j + box_size]):
                    count += 1
        return count

    # Iterate through box sizes
    sizes = []
    counts = []
    min_size = 2
    max_size = min(binary_image.shape) // 2

    size = min_size
    while size <= max_size:
        sizes.append(size)
        counts.append(box_count(binary_image, size))
        size *= 2  # Increase the box size exponentially

    # Calculate fractal dimension using the slope of log-log plot
    log_sizes = np.log(1 / np.array(sizes))
    log_counts = np.log(counts)
    coeffs = np.polyfit(log_sizes, log_counts, 1)
    fractal_dim = coeffs[0]

    return fractal_dim

fractal_dimensions_test = []

# Function to calculate fractal dimension
def predict_masks_in_folder(student_models, folder_path):
    # List all files in the folder
    image_files = [f for f in os.listdir(folder_path) ]

    if not image_files:
        print("No valid image files found in the folder.")
        return

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing: {image_file}")

        # Predict mask
        predicted_mask = predict_with_student_models(student_models, image_path)

        # Calculate fractal dimension
        fractal_dim = fractal_dimension(predicted_mask)
        print(f"Fractal Dimension of the image: {fractal_dim:.4f}")
        fractal_dimensions_test.append(fractal_dim)


while True:
    folder_path = input("Enter the folder path containing retinal test images (or type 'exit' to quit): ")
    if folder_path.lower() == 'exit':
        break

    if not os.path.isdir(folder_path):
        print("Invalid folder path. Please try again.")
    else:
        predict_masks_in_folder(student_models, folder_path)

    print(fractal_dimensions_test)

fractal_dimensions_ADpatients = []

def predict_masks_AD(student_models, folder_path):
    # List all files in the folder
    image_files = [f for f in os.listdir(folder_path) ]

    if not image_files:
        print("No valid image files found in the folder.")
        return

    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        print(f"Processing: {image_file}")

        # Predict mask
        predicted_mask = predict_with_student_models(student_models, image_path)

        # Calculate fractal dimension
        fractal_dim = fractal_dimension(predicted_mask)
        print(f"Fractal Dimension: {fractal_dim}")
        fractal_dimensions_ADpatients.append(fractal_dim)

while True:
    folder_path = input("Enter the folder path containing AD patients retinal images (or type 'exit' to quit): ")
    if folder_path.lower() == 'exit':
        break

    if not os.path.isdir(folder_path):
        print("Invalid folder path. Please try again.")
    else:
        predict_masks_AD(student_models, folder_path)

    print(fractal_dimensions_ADpatients)

#box plot
data = [fractal_dimensions_test, fractal_dimensions_ADpatients]
plt.boxplot(data, labels=['Drive Test Dataset', 'AD Patients Dataset'])
plt.title('Box Plots of Two Datasets')
plt.ylabel('Values')
plt.show()
