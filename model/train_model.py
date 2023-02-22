from model.model_handler import ModelHandler
from utils.loading import load_signal_and_annotations, load_diagnostic_aggregation
from model.data_generator import get_cross_validation_split

if __name__ == "__main__":
    path = './data/physionet.org/files/ptb-xl/1.0.2/'
    sampling_rate = 100
    channels = ["I", "II", "III", "AVL", "AVR", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    X, Y = load_signal_and_annotations(sampling_rate, path)
    Y = load_diagnostic_aggregation(path, Y)

    model_handler = ModelHandler(12, 1000, 5)
    train_generator, test_generator = get_cross_validation_split(X, Y, 0)
    model_handler.train(train_generator)
    results = model_handler.validate(train_generator, 'test')






