import csv
import pickle
from itertools import combinations
from pathlib import Path

import pandas as pd

class ExperimentRunner:
    def __init__(self, csv_path, train, record, target_metrics, domains, all_metrics=None):
        self.csv_path = csv_path
        self.train = train
        self.record = record
        self.all_metrics = all_metrics
        self.target_metrics = target_metrics
        self.domains = domains
        self.experiments_order = 'experiments_order.pkl'


    @staticmethod
    def encode_experiment(experiment):
        news = int('news' in experiment)
        health = int('health' in experiment)
        gov = int('gov' in experiment)
        games = int('games' in experiment)
        food = int('food' in experiment)
        culture = int('culture' in experiment)
        am2022banks = int('AM2022Banks' in experiment)
        am2022ecommerce = int('AM2022ECommerce' in experiment)
        avi = int('AVI_14' in experiment)
        chi15 = int('CHI_15' in experiment)
        chi20 = int('CHI_20' in experiment)
        english = int('english' in experiment)
        foreign = int('foreign' in experiment)
        ijhcs = int('IJHCS' in experiment)
        return news, health, gov, games, food, culture, am2022banks, am2022ecommerce, \
            avi, chi15, chi20, english, foreign, ijhcs


    def experiment_already_ran(self, experiment, target_metric):
        csv_exists = self.csv_path.exists() and self.csv_path.is_file()
        if not csv_exists:
            return False

        experiments = pd.read_csv(self.csv_path)

        news, health, gov, games, food, culture, am2022banks, am2022ecommerce, \
            avi, chi15, chi20, english, foreign, ijhcs, = self.encode_experiment(experiment)

        matching_experiments = experiments.loc[
            (experiments.news == news) &
            (experiments.health == health) &
            (experiments.gov == gov) &
            (experiments.games == games) &
            (experiments.food == food) &
            (experiments.culture == culture) &
            (experiments.am2022banks == am2022banks) &
            (experiments.am2022ecommerce == am2022ecommerce) &
            (experiments.avi == avi) &
            (experiments.chi_15 == chi15) &
            (experiments.chi_20 == chi20) &
            (experiments.english == english) &
            (experiments.foreign == foreign) &
            (experiments.ijhcs == ijhcs) &
            (experiments.target_metric == target_metric)]

        return len(matching_experiments) > 0

    def run_experiments(self, screenshots_directory=None):
        experiments_ordered = []
        if Path(self.experiments_order).exists() and Path(self.experiments_order).is_file():
            print('Found existing experiments order, loading...')
            with open(self.experiments_order, 'rb') as f:
                experiments_ordered = pickle.load(f)
        else:
            print('Ordering experiments by dataset sizes')
            for r in range(len(self.domains)):
                experiments_ordered += list(combinations(self.domains, r + 1))
            experiments_ordered.sort(key=lambda experiment: sum(
                self.all_metrics['domain'].value_counts()[domain] for domain in experiment), reverse=True)

        for experiment in experiments_ordered:
            for target_metric in self.target_metrics:
                if self.experiment_already_ran(experiment, target_metric):
                    continue
                self.record(*self.train(target_metric, domains=experiment, all_metrics=self.all_metrics,
                                        screenshots_directory=screenshots_directory))
