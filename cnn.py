# https://colab.research.google.com/drive/1FOxO55_MmyJDoorQ329xabaxKZ03Rv0z?usp=sharing

import os
from pathlib import Path

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, AveragePooling2D, Input,Concatenate, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, Convolution2D, BatchNormalization
import cv2

def get_class(name):
    data = pd.read_csv("integer.csv", delimiter=',')
    rslt_df = data.loc[data['filename'] == name]['Aesthetics'].values[0]
    print(rslt_df)
    return rslt_df


def load_data(data_dir):
    data_dir = Path(data_dir)
    directories = [d for d in data_dir.iterdir() if d.is_dir()]
    labels = []
    images = []
    for d in directories:
        screenshots = d.glob('*.png')

        index = 0
        for screenshot in screenshots:
            img = cv2.imread(str(screenshot.resolve()))
            try:
                print('1')
                imresize = cv2.resize(img, (900, 600))
                print('2')
                name_temp = get_class(screenshot.name)
                print('3')
                if name_temp.size != 0:
                    print('4')
                    images.append(imresize)
                    index += 1
                    labels.append(name_temp)

            except Exception as e:
                print(str(e))

    return images, labels

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

X_train, y_train = load_data("images2")
X_train = np.array(X_train).astype('float32')
X_train = X_train / 255.0

print(y_train)
y_train = np.array(y_train)
print(y_train)

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def createCNNModel():
    def Inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
        path1 = Conv2D(filters=f1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
        path2 = Conv2D(filters=f2_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
        path2 = Conv2D(filters=f2_conv3, kernel_size=(3, 3), padding='same', activation='relu')(path2)
        path3 = Conv2D(filters=f3_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
        path3 = Conv2D(filters=f3_conv5, kernel_size=(5, 5), padding='same', activation='relu')(path3)
        path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
        path4 = Conv2D(filters=f4, kernel_size=(1, 1), padding='same', activation='relu')(path4)

        output_layer = tensorflow.keras.layers.concatenate([path1, path2, path3, path4], axis=-1)

        return output_layer


    input_layer = Input(shape=(600, 900, 3))

    X = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='valid', activation='relu')(input_layer)
    X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)
    X = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(X)
    X = Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)
    X = Inception_block(X, f1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16, f3_conv5=32, f4=32)
    X = Inception_block(X, f1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32, f3_conv5=96, f4=64)
    X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)
    X = Inception_block(X, f1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16, f3_conv5=48, f4=64)
    X = Inception_block(X, f1=160, f2_conv1=112, f2_conv3=224, f3_conv1=24, f3_conv5=64, f4=64)
    X = Inception_block(X, f1=128, f2_conv1=128, f2_conv3=256, f3_conv1=24, f3_conv5=64, f4=64)
    X = Inception_block(X, f1=112, f2_conv1=144, f2_conv3=288, f3_conv1=32, f3_conv5=64, f4=64)
    X = GlobalAveragePooling2D(name='GAPL')(X)
    X = Dropout(0.4)(X)
    X = Dense(1)(X)

    model = Model(input_layer, X)

    epochs = 40
    model.compile(loss='mse', optimizer='adam', metrics=['mae', 'mse', coeff_determination])
    return model, epochs

model, epochs = createCNNModel()

history = model.fit(X_train, y_train, validation_split=0.20, epochs=epochs, callbacks=[keras.callbacks.TensorBoard('history11'), keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=6,
                              verbose=0, mode='auto')])
