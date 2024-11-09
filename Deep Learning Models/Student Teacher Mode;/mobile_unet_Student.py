import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load dataset
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


# Load training data
x_train, y_train = load_images(train_images_dir, train_masks_dir)


# Define UNet model
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


# Dice Loss and F1 Score remain the same
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

# Build teacher model (larger UNet model)
teacher_model = unet_model(filters=64)
teacher_model.compile(optimizer=Adam(), loss=dice_loss, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), tf.keras.metrics.AUC(name='auc'),
                               f1_score])
teacher_model.fit(x_train, y_train, epochs=x, batch_size=8, validation_split=0.1)

from tensorflow.keras.layers import Input, DepthwiseConv2D, Conv2D, BatchNormalization, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model

def mobile_unet(input_size=(256, 256, 3), filters=32):
    inputs = Input(input_size)

    # Encoder (Mobile UNet)
    c1 = DepthwiseConv2D(3, activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    c1 = Conv2D(filters, 1, activation='relu', padding='same')(c1)  # Pointwise convolution
    p1 = MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = DepthwiseConv2D(3, activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Conv2D(filters * 2, 1, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D(pool_size=(2, 2))(c2)

    # Bottleneck
    c3 = DepthwiseConv2D(3, activation='relu', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Conv2D(filters * 4, 1, activation='relu', padding='same')(c3)

    # Decoder (Mobile UNet)
    u4 = UpSampling2D(size=(2, 2))(c3)
    u4 = DepthwiseConv2D(3, activation='relu', padding='same')(u4)
    u4 = BatchNormalization()(u4)
    u4 = Conv2D(filters * 2, 1, activation='relu', padding='same')(u4)

    u5 = UpSampling2D(size=(2, 2))(u4)
    u5 = DepthwiseConv2D(3, activation='relu', padding='same')(u5)
    u5 = BatchNormalization()(u5)
    u5 = Conv2D(filters, 1, activation='relu', padding='same')(u5)

    outputs = Conv2D(1, 1, activation='sigmoid', padding='same')(u5)

    model = Model(inputs=inputs, outputs=outputs)
    return model

# Build three student models using Mobile UNet
student_models = []
for i in range(3):
    student_model = mobile_unet(filters=32)  # Use Mobile UNet here
    student_model.compile(optimizer=Adam(), loss=dice_loss, metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2), tf.keras.metrics.AUC(name='auc'),
                                   f1_score])
    student_models.append(student_model)


print("\nStudent modals\n")
# Knowledge distillation: Train students with both ground truth and teacher's predictions
for student_model in student_models:
    # Use both y_train (ground truth) and teacher_model's predictions (soft labels)
    teacher_predictions = teacher_model.predict(x_train)

    # Loss is combined: ground truth + teacher predictions
    student_model.fit(x_train, [y_train, teacher_predictions], epochs=2, batch_size=8, validation_split=0.1)


def evaluate_models(teacher_model, student_models, x_data, y_data):
    # Initialize accumulators for student models
    avg_student_auc = 0
    avg_student_mean_iou = 0
    avg_student_accuracy = 0
    avg_student_f1_score = 0

    # First, evaluate the teacher model
    print("\nEvaluating Teacher Model:")
    teacher_results = teacher_model.evaluate(x_data, y_data, verbose=0)
    teacher_accuracy = teacher_results[1]  # Accuracy is at index 1
    teacher_mean_iou = teacher_results[2]  # Mean IoU is at index 2
    teacher_auc = teacher_results[3]  # AUC is at index 3
    teacher_f1 = teacher_results[4]  # F1 score is at index 4

    print(f"Teacher Accuracy: {teacher_accuracy:.4f}")
    print(f"Teacher Mean IoU: {teacher_mean_iou:.4f}")
    print(f"Teacher AUC: {teacher_auc:.4f}")
    print(f"Teacher F1 Score: {teacher_f1:.4f}")

    # Now, evaluate each student model
    print("\nEvaluating Student Models:")
    for i, student_model in enumerate(student_models):
        results = student_model.evaluate(x_data, y_data, verbose=0)

        accuracy = results[1]
        mean_iou = results[2]
        auc = results[3]
        f1 = results[4]

        print(
            f"Student Model {i + 1} - Accuracy: {accuracy:.4f}, Mean IoU: {mean_iou:.4f}, AUC: {auc:.4f}, F1 Score: {f1:.4f}")

        # Accumulate results for averaging
        avg_student_accuracy += accuracy
        avg_student_mean_iou += mean_iou
        avg_student_auc += auc
        avg_student_f1_score += f1

    # Calculate the average for student models
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

    # Calculate the overall average performance (Teacher + Students)
    overall_avg_accuracy = (teacher_accuracy + avg_student_accuracy) / (num_students + 1)
    overall_avg_mean_iou = (teacher_mean_iou + avg_student_mean_iou) / (num_students + 1)
    overall_avg_auc = (teacher_auc + avg_student_auc) / (num_students + 1)
    overall_avg_f1_score = (teacher_f1 + avg_student_f1_score) / (num_students + 1)

    print("\nOverall Average Performance (Teacher + Students):")
    print(f"Overall Average Accuracy: {overall_avg_accuracy:.4f}")
    print(f"Overall Average Mean IoU: {overall_avg_mean_iou:.4f}")
    print(f"Overall Average AUC: {overall_avg_auc:.4f}")
    print(f"Overall Average F1 Score: {overall_avg_f1_score:.4f}")

# Evaluate teacher and student models, and calculate overall averages
evaluate_models(teacher_model, student_models, x_train, y_train)




# Predict using student models
def predict_with_student_models(student_models, image_path, image_size=(256, 256)):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = []
    for student_model in student_models:
        predictions.append(student_model.predict(img_array)[0])

    # Average the predictions of all student models
    final_mask = np.mean(predictions, axis=0)

    return final_mask


# Predict blood vessel masks using student models
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
