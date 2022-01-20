import keras
import time

from tensorflow.keras import backend as K
from kerastuner.tuners import RandomSearch

def coeff_determination(y_true, y_pred):
    SS_res = K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - SS_res / (SS_tot + K.epsilon()))


class TimeHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, epoch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class RandomSearchWithTimer(RandomSearch):
    def __init__(
            self,
            hypermodel=None,
            objective=None,
            max_trials=10,
            seed=None,
            hyperparameters=None,
            tune_new_entries=True,
            allow_new_entries=True,
            **kwargs
    ):
        self.times = []
        super().__init__(
            hypermodel=hypermodel,
            objective=objective,
            max_trials=max_trials,
            seed=seed,
            hyperparameters=hyperparameters,
            tune_new_entries=tune_new_entries,
            allow_new_entries=allow_new_entries,
            **kwargs
        )

    def run_trial(self, trial, *args, **kwargs):
        trial_start = time.time()
        histories = super().run_trial(trial, *args, **kwargs)
        trial_end = time.time()
        trial_time = trial_end - trial_start
        self.times.append(trial_time)
        return histories