# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 13:52:03 2023

@author: jackp
"""


import matplotlib.pyplot as plt
import pathlib

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import os
import cv2

# Turn off unneccesary error messages
tf.get_logger().setLevel('ERROR')
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Change to the correct directory
directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(directory)


model = keras.models.load_model('models/model_res=4_4outof4')   

 

for i in range(len(model.layers)):
    layer = model.layers[i]
    if 'conv' not in layer.name:
        continue    
    print(i , layer.name , layer.output.shape)
    
model = keras.Model(inputs=model.inputs , outputs=model.layers[6].output) 

print('\n')
for i in range(len(model.layers)):
    layer = model.layers[i]
    if 'conv' not in layer.name:
        continue    
    print(i , layer.name , layer.output.shape)
    
image_path = os.path.join(directory, '5eyes.jpg')
image = cv2.imread(image_path)   

# Convert the image into a keras tensor array
img_array = tf.keras.utils.img_to_array(image)
img_array = tf.expand_dims(img_array, 0)
  

# Calculating features_map
features = model.predict(img_array)

fig = plt.figure(figsize=(20,15))
for i in range(1,features.shape[3]+1):
    #print(i)
    plt.subplot(8,8,i)
    plt.imshow(features[0,:,:,i-1] , cmap='gray')
    
plt.show()


print('\nDone')







