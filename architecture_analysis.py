import os
import matplotlib.pyplot as plt
from keras import layers, models
from keras.src.legacy.preprocessing.image import ImageDataGenerator

# Disable TensorFlow OneDNN optimizations for reproducibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Directories for training and validation datasets
train_dir = 'data/EV_sliced/training'
validation_dir = 'data/EV_sliced/validation'

# Global configuration
image_size = 128
image_count = 250
epochs = 50
batch_size = 25

# List of architectures to test
architectures = []

def architecture_0():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4, activation='softmax'))
    return model

architectures.append(('Single Convo(32)', architecture_0))

def architecture_1():
    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation="relu", input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4, activation='softmax'))
    return model

architectures.append(('Single Convo(128)', architecture_1))

def architecture_2():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    return model

architectures.append(('Single Convo(32) & Dense', architecture_2))


def architecture_3():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4, activation='softmax'))
    return model

architectures.append(('Triple Convo(32, 64, 128)', architecture_3))

def architecture_4():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    return model

architectures.append(('Triple Convo(32, 64, 128) & Dense', architecture_4))

def architecture_5():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4, activation='softmax'))
    return model

architectures.append(('Single Convo(32) & Dropout(0.5)', architecture_5))

def architecture_6():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    return model

architectures.append(('Single Convo(32), Dense & Dropout(0.5)', architecture_6))

def architecture_7():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    return model

architectures.append(('Single Convo(32), Dsense & Dropout(0.5)', architecture_7))

def architecture_8():
    model = models.Sequential()
    model.add(layers.Conv2D(128, (3, 3), activation="relu", input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(256, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(512, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))
    return model

architectures.append(('Deeper Triple Convo(128, 256, 512)', architecture_8))

def architecture_9():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    return model

architectures.append(('Single Convo(32), GlobalAveragePooling, Dense', architecture_9))

def architecture_10():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (5, 5), activation="relu", input_shape=(image_size, image_size, 3)))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    return model

architectures.append(('Larger kernels Single Convo(32, (5, 5)) & Dense', architecture_10))

def architecture_11():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(image_size, image_size, 3)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))


    return model

architectures.append(('Single Convo(32), BatchNormalization & Dense', architecture_11))

def architecture_12():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(image_size, image_size, 3)))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.AveragePooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(4, activation='softmax'))

    return model

architectures.append(('Triple Convo(32), AveragePooling & Dense', architecture_12))

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Training and Validation Generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(image_size, image_size),
    batch_size=batch_size,
    class_mode='categorical'
)

results = []

for name, architecture in architectures:
    print(f"Training architecture: {name}")

    model = architecture()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        verbose=1
    )

    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    avg_train_acc = sum(train_acc[-5:]) / 5
    avg_val_acc = sum(val_acc[-5:]) / 5

    overfit_gap = avg_train_acc - avg_val_acc
    overfit_status = "Overfitting" if overfit_gap > 0.1 else "No Overfitting"

    print(f"Average Train Accuracy: {avg_train_acc:.4f}")
    print(f"Average Validation Accuracy: {avg_val_acc:.4f}")
    print(f"Overfitting Gap: {overfit_gap:.4f} ({overfit_status})")

    # Save results
    results.append((name, avg_train_acc, avg_val_acc, overfit_gap, overfit_status))


# Print results summary
print("\nFinal Results:")
for name, avg_train_acc, avg_val_acc, overfit_gap, overfit_status in results:
    print(f"{name}: \nTrain Accuracy = {avg_train_acc:.4f}, \nVal Accuracy = {avg_val_acc:.4f}, \nOverfitting Gap = {overfit_gap:.4f}, \nStatus = {overfit_status} \n")
