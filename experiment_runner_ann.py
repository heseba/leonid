import os

from ann import ANN
from experiment_runner import ExperimentRunner

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


from pathlib import Path

import pandas as pd
import csv

csv_header = ['news', 'health', 'gov', 'games', 'food', 'culture', 'target_metric', 'R2', 'mse', 'mae', 'rmse',
              'n_train', 'n_val', 'time', 'epochs']
csv_path = Path('experiments_ann.csv')

target_metrics = ['Complexity', 'Aesthetics', 'Orderliness']
domains = ['news', 'health', 'gov', 'games', 'food', 'culture']
screenshots_directory = 'images2'


def train(target_metric, domains, all_metrics, screenshots_directory=None):
    ann = ANN(target_metric, domains=domains)
    model, best_trial, n_train, n_val, times, epochs = ann.train()
    return domains, target_metric, model, best_trial, n_train, n_val, times, epochs


def record(experiment, target_metric, model, best_trial, n_train, n_val, times, epochs):
    csv_header = ['news', 'health', 'gov', 'games', 'food', 'culture', 'target_metric', 'R2', 'mse', 'mae', 'rmse',
                  'n_train', 'n_val', 'time', 'epochs', 'trials']

    csv_exists = csv_path.exists() and csv_path.is_file()

    with open(str(csv_path), 'a', newline='') as csvfile:
        experiments_csv = csv.writer(csvfile, delimiter=',')

        if not csv_exists:
            experiments_csv.writerow(csv_header)

        news, health, gov, games, food, culture = ExperimentRunner.encode_experiment(experiment)

        r2 = best_trial.metrics.metrics['val_coeff_determination']._observations[0].value[0]
        mse = best_trial.metrics.metrics['val_mse']._observations[0].value[0]
        mae = best_trial.metrics.metrics['val_mae']._observations[0].value[0]
        rmse = best_trial.metrics.metrics['val_root_mean_squared_error']._observations[0].value[0]
        time = sum(times)
        trials = len(times)

        experiments_csv.writerow(
            [news, health, gov, games, food, culture, target_metric, r2, mse, mae, rmse, n_train, n_val, time, epochs, trials])

    # model.save(Path('models', f'{target_metric}-{"-".join(str(experiment))}'))


if __name__ == '__main__':
    runner = ExperimentRunner(csv_path, train, record, target_metrics, domains)
    runner.run_experiments()
