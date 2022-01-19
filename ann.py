# https://colab.research.google.com/drive/1xK4-uFEIbS0czVjTHEAgWG2MYuz_jwrr?usp=sharing

from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow import keras
import pandas as pd
from tensorflow.keras import backend as K
from kerastuner.tuners import RandomSearch, Hyperband, BayesianOptimization


def maximum_absolute_scaling(df):
    df_scaled = df.copy()
    for column in df_scaled.columns:
        df_scaled[column] = df_scaled[column] / df_scaled[column].abs().max()
    return df_scaled


def load_data(path):
    data = pd.read_csv(path, delimiter=',')
    y = data.loc[:, 'Orderliness']
    print(y)
    x = data.loc[:, 'VA_PNG':'grid quality']
    print(x)
    x = maximum_absolute_scaling(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data("train_without_index.csv")

print(x_train, y_train, x_test, y_test)

def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def build_model(hp):
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
        metrics=['mse', coeff_determination])
    return model


tuner = RandomSearch(
    build_model,
    objective='mse',
    max_trials=20,
    directory='ASN_AST',
    overwrite=True
)

tuner.search_space_summary()

tuner.search(x_train, y_train,
             epochs=100,
             validation_split=0.2,callbacks=[keras.callbacks.TensorBoard('history14'), keras.callbacks.EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=6,
                              verbose=0, mode='auto')])

tuner.results_summary()

print(tuner.oracle.get_best_trials()[0].trial_id)

models = tuner.get_best_models(num_models=1)

for model in models:
    model.evaluate(x_test, y_test)
    model.summary()


    # for i in range(1,100):
    #     res = model.predict(np.expand_dims(x_test.iloc[[i]], axis=0))
    #     print(x_test.iloc[[i]])
    #     print(res)
    #     print(np.argmax(res))
