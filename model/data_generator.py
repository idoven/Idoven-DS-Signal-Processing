import torch
import pandas as pd
import numpy as np

class ECGDataset(torch.utils.data.Dataset):
    """ Dataset of ECG signals to train a deep learning model."""
    def __init__(self, X, Y_labels):
        'Initialization'
        self.X = X
        self.Y_labels = Y_labels

    def __len__(self):
        'Denotes the total number of samples'
        return self.X.shape[0]

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        x = self.X[index]
        y = np.array(self.Y_labels[index])

        return x, y


def get_cross_validation_split(X:pd.DataFrame, Y:pd.DataFrame, test_fold: int = 0):
    """ Get one split of the cross validation experiment. Returns generators that can be used for model training."""
    batch_size = 8
    X_switched_channels = np.swapaxes(X, 1, 2)
    X_train = X_switched_channels[np.where(Y.strat_fold != test_fold)]
    y_train = np.array(Y[(Y.strat_fold != test_fold)].oneshot_labels)

    train_set = ECGDataset(X_train, y_train)
    train_generator = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # Test
    X_test = X_switched_channels[np.where(Y.strat_fold == test_fold)]
    y_test = np.array(Y[Y.strat_fold == test_fold].oneshot_labels)

    test_set = ECGDataset(X_test, y_test)
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_generator, test_generator
