import numpy as np

class MetricsAggregator():
    """ Class to aggregate metrics of crossvalidation runs."""
    def __init__(self, metric_names: list):
        self.results = {}
        self.metric_names = metric_names
        for m in metric_names:
            self.results[m] = []

    def aggregate_results(self, results_dict: dict):
        for key, value in results_dict.items():
            self.results[key].append(value)

    def calculate_mean_metrics(self):
        mean_results = {}
        for m in self.metric_names:
            mean_results[m] = np.mean(self.results[m])
        return mean_results
