import mlflow
import argparse
import pandas as pd
import numpy as np
import yaml
from model.model_handler import ModelHandler
from utils.loading import load_signal_and_annotations, load_diagnostic_aggregation
from model.data_generator import get_cross_validation_split
from model.metrics import MetricsAggregator


def run_crossvalidation_experiment(X: pd.DataFrame, Y: pd.DataFrame, config: dict):
    """
    Run 10 crossvalidation experiments for the pre-defined 10 crossvalidation experiments.
    The config dict should contain the parameters
    conv_layers (int), conv_kernels (int), learning_rate (float), epochs (int)
    :param X: Dataframe of input signals
    :param Y: Dataframe with labels
    :param config: dict with model configuration: conv_layers (int), conv_kernels (int), learning_rate (float), epochs (int)
    :return: dict of mean crossvalidations results
    """
    metric_names = ['test_NORM_acc', 'test_NORM_recall', 'test_NORM_precision', 'test_MI_acc', 'test_STTC_acc',
                    'test_CD_acc', 'test_HYP_acc']
    metric_aggregator = MetricsAggregator(metric_names)

    for i in np.arange(1, 11, 1): # crossvalidation split indices are from 1 to 10
        config['run_name'] = config['model_name'] + '_crossval_' + str(i)
        model_handler = ModelHandler(X.shape[2], X.shape[1], 5, config)
        train_generator, test_generator = get_cross_validation_split(X, Y, i)
        model_handler.train(train_generator)
        results = model_handler.validate(test_generator, 'test')
        print(results)
        metric_aggregator.aggregate_results(results)
        model_handler.end_run()

    mlflow.set_experiment(experiment_name='PTB_XL')
    mlflow.start_run(run_name=config['model_name'] + '_crossvalidation')

    mean_results = metric_aggregator.calculate_mean_metrics()
    mlflow.log_metrics(mean_results)
    return mean_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cancer Classification")
    parser.add_argument("--config", "-c", type=str, default="./experiments/configs/config_large_model.yaml",
                        help="Config path (yaml file expected) to default config.")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.full_load(file)

    path = './data/physionet.org/files/ptb-xl/1.0.2/'
    sampling_rate = 100

    X, Y = load_signal_and_annotations(sampling_rate, path)
    Y = load_diagnostic_aggregation(path, Y)

    run_crossvalidation_experiment(X, Y, config)
