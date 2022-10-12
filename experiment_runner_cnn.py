import os

from experiment_runner import ExperimentRunner

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['LD_LIBRARY_PATH'] = '$LD_LIBRARY_PATH:/home/sebastian/miniconda3/envs/tf/lib' #adapt if CUDA reports missing libraries

from itertools import combinations
from pathlib import Path

import pandas as pd
import cv2
import csv

from cnn import CNN

resize_x = 900
resize_y = 600

experiments_path = Path('experiments_cnn.csv')
#metrics_path = Path('integer.csv')
#metrics_path = Path('test_export.csv')
metrics_path = Path('combined.csv')

target_metrics = ['Complexity', 'Aesthetics', 'Orderliness']
domains = ['news', 'health', 'gov', 'games', 'food', 'culture', 'AM2022Banks', 'AM2022ECommerce', 'AVI_14', 'CHI_15', 'CHI_20', 'english', 'foreign', 'IJHCS']
#domains = ['AM2022Banks']
#screenshots_directory = '/windows/Users/Sebastian/Downloads/leonid/images2'
#screenshots_directory = '/windows/Users/Sebastian/Downloads/dataverse_files'
#resized_screenshots_directory = '/home/sebastian/Downloads/cnn_data_resized/screens'
#screenshots_directory = '/home/sebastian/Downloads/cnn_data_nstu_resized'
screenshots_directory = '/home/sebastian/Downloads/DATASET_COMBINED'

metrics_header = ['filename','domain','Complexity','Aesthetics','Orderliness','VA_PNG','VA_JPEG','elements','elem. types','visual complex','edge congestion','unique RGB','HSVcolours (avg Hue)','HSVcolours (avg Saturation)','HSVcolours (std Saturation)','HSVcolours (avg Value)','HSVcolours (std Value)','HSVspectrum (HSV)','HSVspectrum (Hue)','HSVspectrum (Saturation)','HSVspectrum (Value)','Hassler-Susstrunk (dist A)','Hassler-Susstrunk (std A)','Hassler-Susstrunk (dist B)','Hassler-Susstrunk (std B)','Hassler-Susstrunk (dist RGYB)','Hassler-Susstrunk (std RGYB)','Hassler-Susstrunk (Colorfulness)','Static clusters','DynamicCC (clusters)','DynamicCC (avg cluster colors)','QuadtreeDec (balance)','QuadtreeDec (symmetry)','QuadtreeDec (equilibrium)','QuadtreeDec (leaves)','whitespace','grid quality','count']

# Debug Setup
# target_metrics = ['Aesthetics']
# domains = ['food']
# screenshots_directory = 'images-food-debug'

def resize_images(data_dir, output_path=None):
    if output_path is None:
        output_path = data_dir

    for screenshot in Path(data_dir).glob('*/*.png'):
        screenshot = str(screenshot.resolve())
        # screenshot = screenshot.encode('utf-8', 'surrogateescape').decode('ISO-8859-1')
        print(f'Resizing {screenshot}')
        img = cv2.imread(screenshot)
        imresize = cv2.resize(img, (resize_x, resize_y))
        cv2.imwrite(str(Path(output_path).resolve() / Path(screenshot).name), imresize)


def train(target_metric, domains, all_metrics, screenshots_directory):
    cnn = CNN(target_metric, domains=domains, all_metrics=all_metrics)
    model, history, n_train, n_val, times = cnn.train(screenshots_directory)
    return domains, target_metric, model, history, n_train, n_val, times


def record(experiment, target_metric, model, history, n_train, n_val, times):
    csv_header = ['news', 'health', 'gov', 'games', 'food', 'culture', 'am2022banks', 'am2022ecommerce', 'avi_14', 'chi_15',
                  'chi_20', 'english', 'foreign', 'ijhcs', 'target_metric', 'R2', 'mse', 'mae',
                  'rmse', 'n_train', 'n_val', 'time', 'epochs']

    csv_exists = experiments_path.exists() and experiments_path.is_file()

    with open(str(experiments_path), 'a', newline='') as csvfile:
        experiments_csv = csv.writer(csvfile, delimiter=',')

        if not csv_exists:
            experiments_csv.writerow(csv_header)

        news, health, gov, games, food, culture, am2022banks, am2022ecommerce, \
        avi, chi15, chi20, english, foreign, ijhcs, = ExperimentRunner.encode_experiment(experiment)
        r2 = history.history['val_coeff_determination'][-1]
        mse = history.history['val_mse'][-1]
        mae = history.history['val_mae'][-1]
        rmse = history.history['val_root_mean_squared_error'][-1]
        time = sum(times)
        epochs = len(times)

        experiments_csv.writerow(
            [news, health, gov, games, food, culture, am2022banks, am2022ecommerce, avi, chi15, chi20, english, foreign,
              ijhcs, target_metric, r2, mse, mae, rmse, n_train, n_val, time, epochs])

    # model.save(Path('models', f'{target_metric}-{"-".join(str(experiment))}'))

def combine_metrics(metrics_csvs=[]):
    dataframes = []
    for metrics_csv in metrics_csvs:
        dataframes.append(pd.read_csv(metrics_csv, delimiter=','))
    combined = pd.concat(dataframes, sort=False)
    combined.to_csv('combined.csv')

if __name__ == '__main__':
    #resize_images(screenshots_directory, output_path=resized_screenshots_directory)
    #combine_metrics(['integer.csv', 'test_export.csv'])
    all_metrics = pd.read_csv(metrics_path, delimiter=',')
    runner = ExperimentRunner(experiments_path, train, record, target_metrics, domains, all_metrics=all_metrics)
    runner.run_experiments(screenshots_directory)
