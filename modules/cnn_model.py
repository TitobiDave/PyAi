import keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from PIL import Image


class cnn_model:



    def prepare_image(self, img_path):
         img = image.load_img(img_path, target_size=(256, 256))
         img_array = image.img_to_array(img)
         img_array = np.expand_dims(img_array, axis=0)  # batch dimension
         img_array = img_array / 255.0  # scale same as training
         return img_array
    def create_model(self):
        model = keras.Sequential([
            keras.layers.InputLayer(input_shape=(256, 256, 3), name='input_layer_1'),
            keras.layers.Conv2D(16, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(16, (3,3), activation='relu'),
            keras.layers.MaxPooling2D(),
            keras.layers.Flatten(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(9, activation='sigmoid')
    ])
        return model