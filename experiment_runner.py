import csv
from itertools import combinations
import pandas as pd

class ExperimentRunner:
    def __init__(self, csv_path, train, record, target_metrics, domains, all_metrics=None):
        self.csv_path = csv_path
        self.train = train
        self.record = record
        self.all_metrics = all_metrics
        self.target_metrics = target_metrics
        self.domains = domains


    @staticmethod
    def encode_experiment(experiment):
        news = int('news' in experiment)
        health = int('health' in experiment)
        gov = int('gov' in experiment)
        games = int('games' in experiment)
        food = int('food' in experiment)
        culture = int('culture' in experiment)
        return news, health, gov, games, food, culture


    def experiment_already_ran(self, experiment, target_metric):
        csv_exists = self.csv_path.exists() and self.csv_path.is_file()
        if not csv_exists:
            return False

        experiments = pd.read_csv(self.csv_path)

        news, health, gov, games, food, culture = self.encode_experiment(experiment)

        matching_experiments = experiments.loc[
            (experiments.news == news) &
            (experiments.health == health) &
            (experiments.gov == gov) &
            (experiments.games == games) &
            (experiments.food == food) &
            (experiments.culture == culture) &
            (experiments.target_metric == target_metric)]

        return len(matching_experiments) > 0

    def run_experiments(self, screenshots_directory=None):
        for target_metric in self.target_metrics:
            for r in range(len(self.domains)):
                for experiment in combinations(self.domains, r=r + 1):
                    if self.experiment_already_ran(experiment, target_metric):
                        continue
                    self.record(*self.train(target_metric, domains=experiment, all_metrics=self.all_metrics, screenshots_directory=screenshots_directory))

