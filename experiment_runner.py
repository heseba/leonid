from itertools import combinations

import pandas as pd

from cnn import CNN

all_metrics = pd.read_csv("integer.csv", delimiter=',')

target_metrics = ['Complexity', 'Aesthetics', 'Orderliness']
domains = ['news', 'health', 'gov', 'games', 'food', 'culture']

for target_metric in target_metrics:
    for r in range(len(domains)):
        for experiment in combinations(domains, r=r+1):
            print(experiment)
            cnn = CNN(target_metric, domains=experiment, all_metrics=all_metrics)
            cnn.train()