import mlflow
import argparse
import yaml
from model.model_handler import ModelHandler
from utils.loading import load_signal_and_annotations, load_diagnostic_aggregation
from model.data_generator import get_cross_validation_split
from model.metrics import MetricsAggregator


def run_crossvalidation_experiment(X, Y, config):
    metric_names = ['test_NORM_acc', 'test_NORM_recall', 'test_NORM_precision', 'test_MI_acc', 'test_STTC_acc',
                    'test_CD_acc', 'test_HYP_acc']
    metric_aggregator = MetricsAggregator(metric_names)

    for i in range(10):
        config['run_name'] = config['model_name'] + '_crossval_' + str(i)
        model_handler = ModelHandler(X.shape[2], X.shape[1], 5, config)
        train_generator, test_generator = get_cross_validation_split(X, Y, i)
        model_handler.train(train_generator)
        results = model_handler.validate(train_generator, 'test')
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
    parser.add_argument("--config", "-c", type=str, default="./experiments/configs/config_small_model.yaml",
                        help="Config path (yaml file expected) to default config.")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.full_load(file)

    path = './data/physionet.org/files/ptb-xl/1.0.2/'
    sampling_rate = 100
    channels = ["I", "II", "III", "AVL", "AVR", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    X, Y = load_signal_and_annotations(sampling_rate, path)
    Y = load_diagnostic_aggregation(path, Y)

    run_crossvalidation_experiment(X, Y, config)

    # config = {'conv_layers': 3, 'conv_kernels': 5, 'learning_rate': 0.0001, 'epochs': 100}
    # model_handler = ModelHandler(12, 1000, 5, config)
    # train_generator, test_generator = get_cross_validation_split(X, Y, 0)
    # model_handler.train(train_generator)
    # results = model_handler.validate(train_generator, 'test')
    # print(results)






