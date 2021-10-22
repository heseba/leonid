from itertools import combinations
from pathlib import Path

import pandas as pd
import cv2


from cnn import CNN

resize_x = 900
resize_y = 600


def resize_images(data_dir):
    for screenshot in Path(data_dir).glob('*/*.png'):
        screenshot = str(screenshot.resolve())
        #screenshot = screenshot.encode('utf-8', 'surrogateescape').decode('ISO-8859-1')
        print(f'Resizing {screenshot}')
        img = cv2.imread(screenshot)
        imresize = cv2.resize(img, (resize_x, resize_y))
        cv2.imwrite(screenshot, imresize)


def run_experiments():
    all_metrics = pd.read_csv("integer.csv", delimiter=',')

    target_metrics = ['Complexity', 'Aesthetics', 'Orderliness']
    domains = ['news', 'health', 'gov', 'games', 'food', 'culture']

    for target_metric in target_metrics:
        for r in range(len(domains)):
            for experiment in combinations(domains, r=r+1):
                print(experiment)
                cnn = CNN(target_metric, domains=experiment, all_metrics=all_metrics)
                cnn.train()



if __name__ == '__main__':
    resize_images('images2')