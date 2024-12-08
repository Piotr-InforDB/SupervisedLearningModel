import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from keras import models
from keras.src.legacy.preprocessing.image import image_utils
import matplotlib.pyplot as plt

model = models.load_model('models/anomaly_cells.keras')
threshold = 0.0004

def preprocess_image(image_path):
    img = image_utils.load_img(image_path, target_size=(512, 512))
    img_array = image_utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_anomaly(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    reconstruction_error = np.mean(np.square(img_array - prediction))
    is_anomaly = reconstruction_error > threshold
    return is_anomaly, reconstruction_error, prediction[0]

def process_images_in_directory(image_dir):
    images = []
    filenames = []
    statuses = []
    reconstruction_errors = []
    predicted_images = []

    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)
            is_anomaly, reconstruction_error, predicted_output = predict_anomaly(image_path)
            images.append(image_utils.load_img(image_path))
            filenames.append(filename)
            statuses.append("Anomaly" if is_anomaly else "Normal")
            reconstruction_errors.append(reconstruction_error)
            predicted_images.append(predicted_output)

    display_all_results(images, filenames, statuses, reconstruction_errors, predicted_images)

def display_all_results(images, filenames, statuses, reconstruction_errors, predicted_images):
    num_images = len(images)
    cols = 10
    rows = (num_images + cols - 1) // cols

    plt.figure(figsize=(20, 2.5 * rows))

    for i in range(num_images):
        plt.subplot(rows, cols, i + 1)
        text_color = 'red' if statuses[i] == "Anomaly" else 'green'
        plt.title(
            f"{filenames[i]}\n{statuses[i]}: {reconstruction_errors[i]:.4f}",
            color=text_color,
            fontsize=8
        )
        plt.imshow(predicted_images[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()

image_dir = 'test/classified_anomaly'
process_images_in_directory(image_dir)
