# Author: Marco Lugo
# Simple classification model to distinguish between the different shapes (rectangles vs circles)
# Data has 1800 128x128x3 images, with 20% of the data used for validation
# Achieves ~99.70% after 30 epochs

import numpy as np

import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Dropout, Flatten, Dense
from keras.optimizers impor

img_size = 128
nb_epochs = 15

#############################################################
# Data generators
#############################################################
train_datagen = ImageDataGenerator(
        rescale=1/255,
        zoom_range=0.25,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        'input/train/images',
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        'input/validation/images',
        target_size=(img_size, img_size),
        batch_size=32,
        class_mode='categorical')

#############################################################
# Model
#############################################################

def build_model(img_size=128):
    inputs = Input((img_size, img_size, 3))

    x = Conv2D(8, (3, 3), activation='relu', padding='same')(inputs)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2))(x)

    x = GlobalMaxPooling2D()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.1)(x)
    outputs = Dense(2, activation='softmax')(x)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model

model = build_model()
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit_generator(train_generator, validation_data=validation_generator, epochs=nb_epochs, verbose=1)
