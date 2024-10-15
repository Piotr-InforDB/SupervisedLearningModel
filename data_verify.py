import os
from PIL import Image

train_dir = "data/trin"
validation_dir = "data/validation"

def check_images(directory):
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            filepath = os.path.join(subdir, file)
            try:
                img = Image.open(filepath)
                img.verify()
                print('1')
            except (IOError, SyntaxError) as e:
                print(f'Invalid image: {filepath}, Error: {e}')

check_images(train_dir)
check_images(validation_dir)
