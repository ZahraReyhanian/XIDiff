# -*- coding: utf-8 -*-
"""image_classifier.ipynb

"""# Import libraries"""

import pandas as pd
import numpy as np
import keras
from keras.utils import image_dataset_from_directory

"""# Preprocess dataset"""

# Reading/Loading the dataset files
print(keras.__version__)
dir = 'data/train'
dir_valid = 'data/valid'
# find the class names so in prediction time we can map the predictions to the painters properly


train_dataset = image_dataset_from_directory(
    dir,
    labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    seed=42,
    batch_size=32,
    shuffle=True
)

class_names = train_dataset.class_names
print('Class names:', class_names)

val_dataset = image_dataset_from_directory(
    dir_valid, labels="inferred",
    label_mode="categorical",
    color_mode="rgb",
    batch_size=32,
    shuffle=False
)

# Preprocessing step
from keras.applications.resnet50 import preprocess_input

# Preprocess the data
train_dataset = train_dataset.map(lambda x, y: (preprocess_input(x), y))
val_dataset = val_dataset.map(lambda x, y: (preprocess_input(x), y))

"""# Create Model"""
model = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(256,256,3),
    pooling='avg',
    classes=7,
)
model.trainable = False
x = keras.layers.Flatten()(model.output)
x = keras.layers.Dense(256,activation='relu')(x)
x = keras.layers.Dense(100,activation='relu')(x)
x = keras.layers.Dense(7, activation='softmax')(x)
model = keras.models.Model(model.input, x)

lr = 2e-4
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=lr),
    loss='categorical_crossentropy',
    metrics=[
      'accuracy'
    ],
)

epochs = 30
checkpoint_cb = 5

def scheduler(epoch, lr):
     if epoch % 2:
         return lr * 0.1
     else:
         return lr

learning_rate_cb = keras.callbacks.LearningRateScheduler(scheduler)

checkpoint_filepath = 'checkpoint2.model.keras'
model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)

history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=val_dataset,
    callbacks=[learning_rate_cb, model_checkpoint_callback, keras.callbacks.EarlyStopping(patience=checkpoint_cb),]
)

print('history', history['loss'])

model.save("my_model.keras")

# Evaluate
model.evaluate(val_dataset)

