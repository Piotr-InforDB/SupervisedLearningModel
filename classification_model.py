import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import matplotlib.pyplot as plt

from keras import layers, models
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.src.callbacks import EarlyStopping

train_dir = 'data/EV_sliced/training'
validation_dir = 'data/EV_sliced/validation'

# 0: Background
# 1: Case
# 2: Cells
# 3: Connectors

image_size = 128
image_count = 0
epochs = 2000
batch_size = 25

# Datasets
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
validation_datagen = ImageDataGenerator(
    rescale=1./255
)

# Generators
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


# Model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(image_size, image_size, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Dropout(0.2))

model.add(layers.Flatten())

model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(4, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
)

# EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=30,
    restore_best_weights=True,
)

# Train the model
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[early_stopping]
)

# Average accuracy
avg_val_accuracy = sum(history.history['val_accuracy'][-10:]) / 10
print(f"Average val_accuracy of the last 10 epochs: {avg_val_accuracy:.4f}")

# model.save(f'models/classifications_architecture/{image_size}_{image_count}_{epochs}.keras')
model.save(f'models/ev_slices2.keras')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()