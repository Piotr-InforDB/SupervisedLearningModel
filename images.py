import random
import shutil

def split_images_in_folder(input_folder, output_folder, block_size=(100, 100)):
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            img = Image.open(image_path)
            img_width, img_height = img.size

            grid_x = img_width // block_size[0]
            grid_y = img_height // block_size[1]

            base_name, ext = os.path.splitext(filename)
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)

            for i in range(grid_x):
                for j in range(grid_y):
                    left = i * block_size[0]
                    upper = j * block_size[1]
                    right = left + block_size[0]
                    lower = upper + block_size[1]

                    cropped_img = img.crop((left, upper, right, lower))
                    cropped_img.save(os.path.join(output_folder, f"{base_name}_{i}_{j}.png"))

            print(f"Image '{filename}' successfully split into blocks and saved in '{output_folder}'")

from PIL import Image
import os

def split_image(image, output_folder, block_size=(100, 100)):
    output_folder = os.path.join(os.getcwd(), output_folder)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    img_width, img_height = image.size

    grid_x = img_width // block_size[0]
    grid_y = img_height // block_size[1]

    for i in range(grid_x):
        for j in range(grid_y):
            left = i * block_size[0]
            upper = j * block_size[1]
            right = left + block_size[0]
            lower = upper + block_size[1]

            cropped_img = image.crop((left, upper, right, lower))
            cropped_img.save(os.path.join(output_folder, f"block_{i}_{j}.png"))

    print(f"Image successfully split into blocks and saved in '{output_folder}'")



def split_dataset(folder_path, split_ratio, total_images):
    var = os.path.basename(os.path.normpath(folder_path))
    training_path = f"./data/{var}/training/"
    validation_path = f"./data/{var}/validation/"

    os.makedirs(training_path, exist_ok=True)
    os.makedirs(validation_path, exist_ok=True)

    for class_folder in os.listdir(folder_path):
        class_path = os.path.join(folder_path, class_folder)

        if os.path.isdir(class_path):
            images = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
            random.shuffle(images)

            selected_images = images[:total_images] if total_images < len(images) else images
            split_index = int(len(selected_images) * split_ratio)

            training_images = random.sample(selected_images, split_index)
            validation_images = [img for img in selected_images if img not in training_images]

            os.makedirs(os.path.join(training_path, class_folder), exist_ok=True)
            os.makedirs(os.path.join(validation_path, class_folder), exist_ok=True)

            for image in training_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(training_path, class_folder, image))

            for image in validation_images:
                shutil.copy(os.path.join(class_path, image), os.path.join(validation_path, class_folder, image))

            print(f"Processed class: {class_folder}")
            print(f"Training images: {len(training_images)}, Validation images: {len(validation_images)}")

split_dataset('datasets/EV_sliced', 0.8, 125)
# split_images_in_folder('datasets/EV/background', 'datasets/EV_sliced/background', (256, 256))
