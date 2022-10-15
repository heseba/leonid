# https://colab.research.google.com/drive/1xK4-uFEIbS0czVjTHEAgWG2MYuz_jwrr?usp=sharing
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow import keras
import pandas as pd
import tensorflow as tf

from shared import coeff_determination, RandomSearchWithTimer


class ANN:
    def __init__(self, target_metric,
                 domains=['news', 'health', 'gov', 'games', 'food', 'culture'],
                 ):
        self.target_metric = target_metric
        self.domains = domains

    def maximum_absolute_scaling(self, df):
        df_scaled = df.copy()
        for column in df_scaled.columns:
            df_scaled[column] = df_scaled[column] / df_scaled[column].abs().max()
        return df_scaled

    def load_data(self, path):
        data = pd.read_csv(path, delimiter=',')
        data_of_domain = data[data.domain.isin(self.domains)]
        filtered_data_of_domain = data_of_domain.loc[:, 'VA_PNG':'grid quality']
        filtered_data_of_domain = filtered_data_of_domain.join(data_of_domain.loc[:, self.target_metric])
        filtered_data_of_domain = filtered_data_of_domain.dropna()

        y = filtered_data_of_domain.loc[:, self.target_metric]
        print(y)
        x = filtered_data_of_domain.loc[:, 'VA_PNG':'grid quality']
        print(x)
        x = self.maximum_absolute_scaling(x)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        return x_train, y_train, x_test, y_test

    def build_model(self, hp):
        model = keras.Sequential()
        activation_choice = hp.Choice('activation', values=['relu', 'sigmoid', 'tanh', 'elu', 'selu'])
        model.add(Dense(units=hp.Int('units_input',
                                     min_value=512,
                                     max_value=1024,
                                     step=32),
                        activation=activation_choice))
        for i in range(2, 6):
            model.add(Dense(units=hp.Int('units_' + str(i),
                                         min_value=128,
                                         max_value=1024,
                                         step=32),
                            activation=activation_choice))
        model.add(Dense(1))
        model.compile(
            optimizer="adam",
            loss='mse',
            metrics=['mae', 'mse', coeff_determination, tf.keras.metrics.RootMeanSquaredError()])
        return model

    def train(self):
        x_train, y_train, x_test, y_test = self.load_data("combined.csv")

        print(x_train, y_train, x_test, y_test)

        tuner = RandomSearchWithTimer(
            self.build_model,
            objective='mse',
            max_trials=20,
            directory='ASN_AST',
            overwrite=True
        )

        tuner.search_space_summary()

        tuner.search(x_train, y_train,
                     epochs=100,
                     validation_split=0.2,
                     callbacks=[keras.callbacks.TensorBoard('history14_ann'),
                                keras.callbacks.EarlyStopping(monitor='val_loss',
                                                              min_delta=0,
                                                              patience=6,
                                                              verbose=0,
                                                              mode='auto')])

        tuner.results_summary()

        print(tuner.oracle.get_best_trials()[0].trial_id)

        models = tuner.get_best_models(num_models=1)

        for model in models:
            model.evaluate(x_test, y_test)
            model.summary()

        model = models[0]

        best_trial = tuner.oracle.trials[tuner.oracle.get_best_trials()[0].trial_id]

        return model, best_trial, len(x_train), len(x_test), tuner.times, tuner.epochs

        # for i in range(1,100):
        #     res = model.predict(np.expand_dims(x_test.iloc[[i]], axis=0))
        #     print(x_test.iloc[[i]])
        #     print(res)
        #     print(np.argmax(res))


if __name__ == '__main__':
    ann = ANN('Aesthetics', domains=['food'])
    model, best_trial, n_train, n_val, times = ann.train()
    print(best_trial)
