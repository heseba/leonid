# https://colab.research.google.com/drive/1FOxO55_MmyJDoorQ329xabaxKZ03Rv0z?usp=sharing

import os
from pathlib import Path

from shared import coeff_determination, TimeHistory

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1" #non GPU version
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/home/sebastian/miniconda3/envs/tf/lib' #adapt if CUDA reports missing libraries


import tensorflow
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Input, Dense, Dropout, Conv2D, MaxPooling2D


class CNN:

    # target metrics: Complexity,Aesthetics,Orderliness
    def __init__(self, target_metric,
                 resize_x=900,
                 resize_y=600,
                 epochs=40,
                 tensorboard_directory='tensorboard',
                 domains=['news', 'health', 'gov', 'games', 'food', 'culture'],
                 all_metrics=None):
        self.target_metric = target_metric
        self.resize_x = resize_x
        self.resize_y = resize_y
        self.epochs = epochs
        self.tensorboard_directory = tensorboard_directory
        self.domains = domains
        self.all_metrics = all_metrics

    def value_of_metric(self, screenshot_name, all_metrics):
        metric_value = all_metrics.loc[all_metrics['filename'] == screenshot_name][self.target_metric].values[0]
        print(metric_value)
        return metric_value

    @staticmethod
    def filename_to_path(fn, available_screenshots):
        for (name, path) in available_screenshots:
            if name == fn:
                return str(path.resolve())

    def process_path(self, row):
        metric = row[self.target_metric]
        return tf.io.read_file(row['filepath']), metric

    def load_data(self, data_dir):
        if self.all_metrics is None:
            self.all_metrics = pd.read_csv("integer.csv", delimiter=',')

        screenshots_of_domain = self.all_metrics[self.all_metrics.domain.isin(self.domains)]
        available_screenshots = [(p.name, p) for p in Path(data_dir).glob('*/*.png')]

        samples = screenshots_of_domain[
            screenshots_of_domain.filename.isin([name for name, _ in available_screenshots])]
        samples = samples[['filename', self.target_metric]]
        samples['filepath'] = samples.filename.map(lambda fn: CNN.filename_to_path(fn, available_screenshots))

        samples = samples.sample(frac=1.0)  # shuffling

        image_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1 / 255, validation_split=0.20)

        image_data_train = image_generator.flow_from_dataframe(samples,
                                                               x_col='filepath',
                                                               y_col=self.target_metric,
                                                               subset='training',
                                                               class_mode='raw')

        image_data_val = image_generator.flow_from_dataframe(samples,
                                                             x_col='filepath',
                                                             y_col=self.target_metric,
                                                             subset='validation',
                                                             class_mode='raw')

        return image_data_train, image_data_val

    def create_model(self):
        def inception_block(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4):
            path1 = Conv2D(filters=f1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
            path2 = Conv2D(filters=f2_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
            path2 = Conv2D(filters=f2_conv3, kernel_size=(3, 3), padding='same', activation='relu')(path2)
            path3 = Conv2D(filters=f3_conv1, kernel_size=(1, 1), padding='same', activation='relu')(input_layer)
            path3 = Conv2D(filters=f3_conv5, kernel_size=(5, 5), padding='same', activation='relu')(path3)
            path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
            path4 = Conv2D(filters=f4, kernel_size=(1, 1), padding='same', activation='relu')(path4)

            output_layer = tensorflow.keras.layers.concatenate([path1, path2, path3, path4], axis=-1)

            return output_layer

        input_layer = Input(shape=(self.resize_y, self.resize_x, 3))

        X = Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='valid', activation='relu')(input_layer)
        X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)
        X = Conv2D(filters=64, kernel_size=(1, 1), strides=1, padding='same', activation='relu')(X)
        X = Conv2D(filters=192, kernel_size=(3, 3), padding='same', activation='relu')(X)
        X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)
        X = inception_block(X, f1=64, f2_conv1=96, f2_conv3=128, f3_conv1=16, f3_conv5=32, f4=32)
        X = inception_block(X, f1=128, f2_conv1=128, f2_conv3=192, f3_conv1=32, f3_conv5=96, f4=64)
        X = MaxPooling2D(pool_size=(3, 3), strides=2)(X)
        X = inception_block(X, f1=192, f2_conv1=96, f2_conv3=208, f3_conv1=16, f3_conv5=48, f4=64)
        X = inception_block(X, f1=160, f2_conv1=112, f2_conv3=224, f3_conv1=24, f3_conv5=64, f4=64)
        X = inception_block(X, f1=128, f2_conv1=128, f2_conv3=256, f3_conv1=24, f3_conv5=64, f4=64)
        X = inception_block(X, f1=112, f2_conv1=144, f2_conv3=288, f3_conv1=32, f3_conv5=64, f4=64)
        X = GlobalAveragePooling2D(name='GAPL')(X)
        X = Dropout(0.4)(X)
        X = Dense(1)(X)

        model = Model(input_layer, X)

        model.compile(loss='mse', optimizer='adam',
                      metrics=['mae', 'mse', coeff_determination, tf.keras.metrics.RootMeanSquaredError()])
        return model

    def train(self, images_dir):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=config)

        image_data_train, image_data_val = self.load_data(images_dir)

        model = self.create_model()

        time_callback = TimeHistory()

        print(f'\n\nTraining Model for {self.target_metric}\n\n')

        model.fit(image_data_train,
                  validation_data=image_data_val,
                  epochs=self.epochs,
                  verbose=1,
                  # verbose=0,
                  # steps_per_epoch=TotalTrainingSamples / TrainingBatchSize,
                  # validation_steps = TotalvalidationSamples / ValidationBatchSize,
                  callbacks=[keras.callbacks.TensorBoard(self.tensorboard_directory),
                             time_callback,
                             keras.callbacks.EarlyStopping(monitor='val_loss',
                                                           min_delta=0,
                                                           patience=6,
                                                           verbose=0, mode='auto')])
        print(f'\n\nTraining Done')
        return model, model.history, image_data_train.n, image_data_val.n, time_callback.times


if __name__ == '__main__':
    cnn = CNN('Aesthetics', domains=['food'])
    model, history, n_train, n_val, times = cnn.train('/windows/Users/Sebastian/Downloads/leonid/images-food-debug')
    print(history.history)
