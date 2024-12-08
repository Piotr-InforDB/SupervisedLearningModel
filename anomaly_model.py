import os
import numpy as np
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt

from keras import layers, models
from keras.src.legacy.preprocessing.image import ImageDataGenerator



size = 256
batch = 10
epochs = 65

# Datasets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # rotation_range=40,
    # width_shift_range=0.2,
    # height_shift_range=0.2,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True,
    # fill_mode='nearest'
)

datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    'data/battery_module/training',
    target_size=(size, size),
    batch_size=batch,
    class_mode='input'
)

validation_generator = datagen.flow_from_directory(
    'data/battery_module/validation',
    target_size=(size, size),
    batch_size=batch,
    class_mode='input'
)

# anomaly_generator = datagen.flow_from_directory(
#     'data/anomaly/anomaly',
#     target_size=(size, size),
#     batch_size=batch,
#     class_mode='input'
# )

# Model Definition
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(size, size, 3)))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))
model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2), padding='same'))

model.add(layers.Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.UpSampling2D((2, 2)))

model.add(layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
model.summary()

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    shuffle=True
)

model.save('models/module_anomaly_cells.keras')

def calculate_reconstruction_errors(model, generator):
    errors = []
    num_batches = len(generator)  # Calculate total batches in one epoch

    for i, batch in enumerate(generator):
        if i >= num_batches:  # Stop after one epoch
            break
        inputs, _ = batch
        reconstructions = model.predict(inputs)
        batch_errors = np.mean(np.square(inputs - reconstructions), axis=(1, 2, 3))
        errors.extend(batch_errors)

    return np.array(errors)

# 3. Calculate reconstruction errors on validation set (non-anomalous)
val_errors = calculate_reconstruction_errors(model, validation_generator)
threshold = np.percentile(val_errors, 95)  # Set threshold at 95th percentile of validation errors
print(f"Reconstruction error threshold: {threshold}")

# Plot the training and validation loss at each epoch
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()