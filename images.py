from PIL import Image
import os

def split_images_in_folder(input_folder, output_folder, block_size=(100, 100)):
    # Loop through all the files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Add more formats if needed
            image_path = os.path.join(input_folder, filename)
            img = Image.open(image_path)
            img_width, img_height = img.size

            grid_x = img_width // block_size[0]
            grid_y = img_height // block_size[1]

            # Create the output folder if it doesn't exist
            base_name, ext = os.path.splitext(filename)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            # Loop through the grid and save each block as a separate image
            for i in range(grid_x):
                for j in range(grid_y):
                    # Define the bounding box for each block (left, upper, right, lower)
                    left = i * block_size[0]
                    upper = j * block_size[1]
                    right = left + block_size[0]
                    lower = upper + block_size[1]

                    # Crop the image and save it
                    cropped_img = img.crop((left, upper, right, lower))
                    cropped_img.save(os.path.join(output_folder, f"{base_name}_{i}_{j}.png"))

            print(f"Image '{filename}' successfully split into blocks and saved in '{output_folder}'")

import os
import random
import shutil
import argparse

def split_dataset(folder_path, split_ratio):
    # Create data directories for training and validation
    var = os.path.basename(os.path.normpath(folder_path))
    training_path = f"./data/{var}/training/"
    validation_path = f"./data/{var}/validation/"

    os.makedirs(training_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)

    # Loop through each class folder in the dataset
    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)

        if os.path.isdir(class_path):  # Check if it's a directory
            # Get all images in the class folder
            images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            random.shuffle(images)  # Shuffle images

            # Calculate the number of images for training and validation
            split_index = int(len(images) * split_ratio)

            # Randomly select training images
            training_images = random.sample(images, split_index)
            # Get validation images by excluding training images
            validation_images = [img for img in images if img not in training_images]

            # Create class folders in training and validation directories
            os.makedirs(os.path.join(training_path, class_folder), exist_ok=True)
            os.makedirs(os.path.join(validation_path, class_folder), exist_ok=True)

            # Copy images to training set
            for image in training_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(training_path, class_folder, image))

            # Copy images to validation set
            for image in validation_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(validation_path, class_folder, image))

            print(f"Processed class: {class_folder}")
            print(f"Training images: {len(training_images)}, Validation images: {len(validation_images)}")

split_dataset('datasets/ev_sliced', 0.8)
# split_images_in_folder('datasets/EV/cells', 'datasets/EV_sliced/cells')
