import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from itertools import combinations
from pathlib import Path

import pandas as pd
import cv2
import csv

from cnn import CNN

resize_x = 900
resize_y = 600
csv_header = ['news', 'health', 'gov', 'games', 'food', 'culture', 'target_metric', 'R2', 'mse', 'mae', 'rmse',
              'n_train', 'n_val', 'time', 'epochs']
csv_path = Path('experiments.csv')

target_metrics = ['Complexity', 'Aesthetics', 'Orderliness']
domains = ['news', 'health', 'gov', 'games', 'food', 'culture']
screenshots_directory = 'images2'

# Debug Setup
#target_metrics = ['Aesthetics']
#domains = ['food']
#screenshots_directory = 'images-food-debug'

def resize_images(data_dir):
    for screenshot in Path(data_dir).glob('*/*.png'):
        screenshot = str(screenshot.resolve())
        # screenshot = screenshot.encode('utf-8', 'surrogateescape').decode('ISO-8859-1')
        print(f'Resizing {screenshot}')
        img = cv2.imread(screenshot)
        imresize = cv2.resize(img, (resize_x, resize_y))
        cv2.imwrite(screenshot, imresize)


def run_experiments(screenshots_directory):
    all_metrics = pd.read_csv("integer.csv", delimiter=',')

    for target_metric in target_metrics:
        for r in range(len(domains)):
            for experiment in combinations(domains, r=r + 1):
                if experiment_already_ran(experiment, target_metric):
                    continue
                cnn = CNN(target_metric, domains=experiment, all_metrics=all_metrics)
                model, history, n_train, n_val, times = cnn.train(screenshots_directory)
                record(experiment, target_metric, model, history, n_train, n_val, times)


def record(experiment, target_metric, model, history, n_train, n_val, times):
    csv_exists = csv_path.exists() and csv_path.is_file()

    with open(str(csv_path), 'a', newline='') as csvfile:
        experiments_csv = csv.writer(csvfile, delimiter=',')

        if not csv_exists:
            experiments_csv.writerow(csv_header)

        news, health, gov, games, food, culture = encode_experiment(experiment)
        r2 = history.history['val_coeff_determination'][-1]
        mse = history.history['val_mse'][-1]
        mae = history.history['val_mae'][-1]
        rmse = history.history['val_root_mean_squared_error'][-1]
        time = sum(times)
        epochs = len(times)

        experiments_csv.writerow(
            [news, health, gov, games, food, culture, target_metric, r2, mse, mae, rmse, n_train, n_val, time, epochs])

    # model.save(Path('models', f'{target_metric}-{"-".join(str(experiment))}'))


def encode_experiment(experiment):
    news = int('news' in experiment)
    health = int('health' in experiment)
    gov = int('gov' in experiment)
    games = int('games' in experiment)
    food = int('food' in experiment)
    culture = int('culture' in experiment)
    return news, health, gov, games, food, culture


def experiment_already_ran(experiment, target_metric):
    csv_exists = csv_path.exists() and csv_path.is_file()
    if not csv_exists:
        return False

    experiments = pd.read_csv(csv_path)

    news, health, gov, games, food, culture = encode_experiment(experiment)

    matching_experiments = experiments.loc[
        (experiments.news == news) &
        (experiments.health == health) &
        (experiments.gov == gov) &
        (experiments.games == games) &
        (experiments.food == food) &
        (experiments.culture == culture) &
        (experiments.target_metric == target_metric)]

    return len(matching_experiments) > 0


if __name__ == '__main__':
    # resize_images(screenshots_directory)
    run_experiments(screenshots_directory)
