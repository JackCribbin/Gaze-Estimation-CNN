# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 18:41:23 2023

@author: Jack Cribbin - 19328253

This file loads a collected dataset of labelled folders of images of eyes
looking at sections of a screen, and trains a CNN model to estimate which 
section an image of eyes are looking in that it hasn't seen before
"""

import matplotlib.pyplot as plt
import pathlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import os

# This is a copy-paste version of the tensorflow Rescaling class, but with 
# minor adjustments to account for excessive warning messages
class Rescaling(tf.keras.layers.Layer):
    """Multiply inputs by `scale` and adds `offset`.
    For instance:
    1. To rescale an input in the `[0, 255]` range
    to be in the `[0, 1]` range, you would pass `scale=1./255`.
    2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` 
    range,
    you would pass `scale=1./127.5, offset=-1`.
    The rescaling is applied both during training and inference.
    Input shape:
    Arbitrary.
    Output shape:
    Same as input.
    Arguments:
    scale: Float, the scale to apply to the inputs.
    offset: Float, the offset to apply to the inputs.
    name: A string, the name of the layer.
    """
    
    def __init__(self, scale, offset=0., name=None, **kwargs):
      self.scale = scale
      self.offset = offset
      super(Rescaling, self).__init__(name=name, **kwargs)
    
    def call(self, inputs):
      dtype = self._compute_dtype
      scale = tf.cast(self.scale, dtype)
      offset = tf.cast(self.offset, dtype)
      return tf.cast(inputs, dtype) * scale + offset
    
    def compute_output_shape(self, input_shape):
      return input_shape
    
    def get_config(self):
      config = {
          'scale': self.scale,
          'offset': self.offset,
      }
      base_config = super(Rescaling, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))
    

# Change to the correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Turn off unneccesary error messages
tf.get_logger().setLevel('ERROR')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



# Set the number of epochs to run for 
epochs = 5




# Get the path of the database of images
data_dir = pathlib.Path(os.path.dirname(os.path.abspath(__file__)) 
                        + '\TrainingImages')

# Count the number of images in it
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count,'images found')

# Set the batch size and the dimensions of the images
batch_size = 32
img_width = 200
img_height = 100

# Seperate out a training and validation dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

# Get the class names of the different image folders
class_names = train_ds.class_names
print('Class found:',class_names)

# Initialize a data augmenter for expanding the dataset
data_augmentation = keras.Sequential(
  [
    #layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)

# Print the shapes of the image and labels batch
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break

AUTOTUNE = tf.data.AUTOTUNE

# Shuffle the training and validation dataset
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Set a rescaling layer to make sure the dataset is normalized
normalization_layer = Rescaling(1./255)

# Normalize the training dataset
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))

# Set the number of classes and the number of layers in the input data
num_classes = len(class_names)
num_layers = 3

# Initialize the model structure
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, num_layers)),
  data_augmentation,
  layers.Conv2D(16, num_layers, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, num_layers, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, num_layers, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


# Compile the model and print a summary of its structure
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.summary()

# Train the model to the datasets and save the history of it
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

# Save the training and validation accuracy for later plotting
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

# Save the training and validation loss for later plotting
loss = history.history['loss']
val_loss = history.history['val_loss']

# Save the range of epochs for later plotting
epochs_range = range(epochs)

# Plot the accuracy over the epochs
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot the loss over the epochs
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Save the model
model.save('models/model_res=9H_newVersion')

print('\nDone')







