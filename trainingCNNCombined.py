import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.preprocessing.image
import glob as glob
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img, array_to_img
from tensorflow.keras.layers import (Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten,
                                     Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout)
from sklearn.model_selection import train_test_split
IMG_HEIGHT = 148
IMG_WIDTH = 385
batch_size = 16

X = np.load("combinedNPY.npy")
y = np.load("onehot_encoded_y.npy")
print(X.shape)
print(y.shape)
MODEL_NAME = "combinedCNN2LDropout10E"

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1)
X_test, X_val, y_test, y_val = train_test_split(X_test, y, test_size=.5)


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
    # X = Dropout(rate=0.3)(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes))(X)

    model = Model(inputs=X_input, outputs=X, name=MODEL_NAME)
    return model


model = GenreModel(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), classes=16)
model.summary()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt, loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save(MODEL_NAME)

# model = tf.keras.models.load_model(MODEL_NAME)
# model.evaluate(test_data_gen)
