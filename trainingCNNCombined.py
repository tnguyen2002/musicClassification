
import os
import IPython.display as ipd
import numpy as np
import pandas as pd
import tensorflow as tf
import PIL
import tensorflow.keras.preprocessing.image
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from tensorflow.keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten,
                                     Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout)

IMG_HEIGHT = 148
IMG_WIDTH = 385
batch_size = 32
train_dir = 'combinedImagesSplit/train'
test_dir = 'combinedImagesSplit/test'
val_dir = 'combinedImagesSplit/val'
MODEL_NAME = "combinedCNN2LDropout10E"
# tf.random.set_seed(1)
image_gen_train = ImageDataGenerator(rescale=1./255)
image_gen_val = ImageDataGenerator(rescale=1./255)
image_gen_test = ImageDataGenerator(rescale=1./255)

train_data_gen = image_gen_train.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    color_mode="grayscale",
    subset="training",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')

test_data_gen = image_gen_test.flow_from_directory(
    batch_size=batch_size,
    directory=test_dir,
    color_mode="grayscale",
    subset="training",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')

val_data_gen = image_gen_val.flow_from_directory(
    batch_size=batch_size,
    directory=val_dir,
    color_mode="grayscale",
    subset="training",
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='categorical')

print(train_data_gen.class_indices.keys())

x, y = next(train_data_gen)
print(x.shape)
print(y.shape)


def GenreModel(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), classes=16):
    X_input = Input(input_shape)

    X = Conv2D(8, kernel_size=(3, 3), strides=(1, 1))(X_input)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)

    X = Conv2D(16, kernel_size=(3, 3), strides=(1, 1))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((2, 2))(X)

    # X = Conv2D(32, kernel_size=(3, 3), strides=(1, 1))(X)
    # X = BatchNormalization(axis=3)(X)
    # X = Activation('relu')(X)
    # X = MaxPooling2D((2, 2))(X)

    # X = Conv2D(64, kernel_size=(3, 3), strides=(1, 1))(X)
    # X = BatchNormalization(axis=3)(X)
    # X = Activation('relu')(X)
    # X = MaxPooling2D((2, 2))(X)

    # X = Conv2D(128, kernel_size=(3, 3), strides=(1, 1))(X)
    # X = BatchNormalization(axis=3)(X)
    # X = Activation('relu')(X)
    # X = MaxPooling2D((2, 2))(X)

    X = Flatten()(X)
    X = Dropout(rate=0.3)(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    model = Model(inputs=X_input, outputs=X, name=MODEL_NAME)
    return model

# model = tf.keras.Sequential(name="Sequential_CNN")

# model.add(Conv2D(2, kernel_size=(3, 3), strides=(2, 2), padding="valid", activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH,1)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid"))

# model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding="valid",activation="relu", input_shape=(IMG_HEIGHT, IMG_WIDTH,1)))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding="valid"))

# model.add(Flatten())
# model.add(Dense(16, activation='softmax'))


model = GenreModel(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), classes=16)
model.summary()
opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data_gen, epochs=10, validation_data=test_data_gen)
model.save(MODEL_NAME)

model = tf.keras.models.load_model(MODEL_NAME)
model.evaluate(test_data_gen)
